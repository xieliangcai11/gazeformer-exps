"""DINOv2 注视回归统一训练入口（RGBDGaze 与 GazeCapture 共用同一条训练循环）。

数据集由 --dataset 选择：
    rgbdgaze    : 域内训练/消融（索引 data/RGBDGaze/index）
    gazecapture : 大规模预训练（索引 data/GazeCapture/index，约 149 万帧）
两者只有数据集构造与索引路径不同，训练循环、损失、评估口径完全一致
（评估统一走 predict_frames/summarize，物理量取自每个样本自己的 screen_cm）。

--init-ckpt 用于从既有权重起训（如 GazeCapture 预训练 -> RGBDGaze 微调）；
加载时打印缺失/多余的键，避免"其实没加载上"的静默错误。

消融维度（可组合）：
    --loss mse|mse_angle   损失函数
        mse        : 纯 MSE（与旧基线同口径）
        mse_angle  : MSE + 几何角度损失（训练时就按"物理cm + 30cm眼距"算，
                     与 evaluate_split 验收口径一致；注意不是早期那个有缺陷的代理角度）
    --geom / --no-geom
        是否注入屏幕几何 token（跨设备泛化的关键信息）
    --unfreeze N
        解冻 DINOv2 最后 N 个 block 参与微调（N=0 表示全部冻结）

输出：out/<dataset>/ablation/<tag>/（按数据集分层，不互相污染）。
指标：与 train.py 完全一致的多口径（L2 / cm / 角度 mean·median·p95·前95%·≤1.91%）。

用法示例：
    python -m scripts.train --dataset rgbdgaze --loss mse --unfreeze 0 --epochs 8
    python -m scripts.train --dataset rgbdgaze --loss mse --no-geom --unfreeze 0 --epochs 8
    python -m scripts.train --dataset gazecapture --resolution 224 --geom --augment \
        --unfreeze 12 --epochs 3 --batch-size 32 --lr 2.3e-4 --tag gc_pretrain
"""

import argparse
import os
import sys
from math import ceil
from pathlib import Path

# ---- 路径引导：把项目根与 src/ 加入 sys.path ----
# 已改为「向上搜索项目根」而不是数 __file__ 的目录层级：后者在文件被移动到
# 不同深度后会静默失效（曾因 scripts/rgbdgaze/ablation.py -> scripts/train.py
# 的移动导致 `No module named 'gazelab'`）。
def _bootstrap() -> Path:
    """从本文件向上找到含 src/gazelab/__init__.py 的目录，并加入 sys.path。"""
    here = Path(__file__).resolve()
    for d in (here.parent, *here.parents):
        if (d / "src" / "gazelab" / "__init__.py").is_file():
            for p in (str(d), str(d / "src")):
                if p not in sys.path:
                    sys.path.insert(0, p)
            return d
    raise RuntimeError(
        f"找不到项目根：从 {here} 向上未发现 src/gazelab/__init__.py")


_PROJECT_ROOT = _bootstrap()

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from datetime import datetime

from configs.paths import DEVICE as DEFAULT_DEVICE, experiment_dirs
from configs.rgbdgaze_config import (RGBDGaze_INDEX_DIR, VIEWING_DISTANCE_CM,
                                     AUG_SHIFT, AUG_SCALE, AUG_ROTATE)
from gazelab.datasets import (RGBDGazeDataset, GazeCaptureDataset,
                              rgb_preprocess)
from gazelab.datasets.table_index import TableIndex
from configs.gazecapture_config import GAZECAPTURE_INDEX_DIR
from gazelab.models.rgbdgaze import RGBDGazeDINOv2, load_state_dict_any_res

PI = 3.141592653589793


class Subset(torch.utils.data.Dataset):
    """按 idces 对基础数据集取样（模块级，可被 DataLoader 多进程 pickle）。"""

    def __init__(self, base, idces):
        self.base = base
        self.idces = idces

    def __len__(self):
        return len(self.idces)

    def __getitem__(self, i):
        return self.base[self.idces[i]]

    @property
    def cm_px(self):
        return self.base.cm_px[self.idces]

    @property
    def screen_size(self):
        return self.base.screen_size[self.idces]


#: 各数据集的差异点：数据集类、索引里的路径列名、默认索引目录
DATASET_SPECS = {
    "rgbdgaze": dict(cls=RGBDGazeDataset, path_col="rgb_path",
                     default_index=RGBDGaze_INDEX_DIR, supports_eye=True,
                     supports_orient=True,
                     kwargs=dict(color_transform=rgb_preprocess)),
    "gazecapture": dict(cls=GazeCaptureDataset, path_col="frame_path",
                        default_index=GAZECAPTURE_INDEX_DIR, kwargs={}),
}


def build_dataloader(dataset, split, index_dir, batch_size, shuffle, num_workers,
                     augment=False, resolution=None, distributed=False,
                     geom_randomize=None, eye_crops=False, orient="raw",
                     eye_scale=0.0):
    """构造 train/val/test 的 DataLoader。

    dataset 决定用哪个数据集类与索引。两个数据集的 edict 字段集兼容
    （rgb / screen_cm / imu / geom；GazeCapture 另带 face_grid 但训练不使用），
    因此训练循环与评估口径无需分叉。

    geom_randomize：几何域随机化触发概率，None=交给数据集用各自 config 默认。
    调用方只需在训练集上传非 None 值；val/test 传什么都没关系，因为两个数据集
    在 augment=False 时都会把它强制为 0（评估口径固定为真实几何）。
    """
    spec = DATASET_SPECS[dataset]
    index_csv = Path(index_dir) / "index.csv"
    list_txt = Path(index_dir) / f"{split}.txt"
    paths = [ln.strip() for ln in
             list_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # 列式索引 + lookup：避免为每行构造临时 dict（125 万行 -> 瞬时 2.4GB 峰值）
    idx_table = TableIndex(index_csv)
    path_to_idx = idx_table.lookup(spec["path_col"])
    missing = [p for p in paths if p not in path_to_idx]
    if missing:
        raise KeyError(f"{len(missing)} 条清单路径不在 {index_csv} 中，"
                       f"例：{missing[:2]}（索引与清单是否来自同一次预处理？）")
    idx = [path_to_idx[p] for p in paths]

    # 仅声明支持的数据集才接受这两个参数（GazeCapture 目前没有眼框表 / 朝向参数）
    kwargs = dict(spec["kwargs"])
    if eye_crops:
        if not spec.get("supports_eye"):
            raise ValueError(f"数据集 {dataset} 不支持眼部支流（eye_crops=True）")
        kwargs["eye_crops"] = True
        if eye_scale > 0:
            kwargs["eye_scale"] = eye_scale
    if orient != "raw":
        # 评估历史 ckpt 用（其输入约定是躺倒朝向）；训练一律默认 raw
        if not spec.get("supports_orient"):
            raise ValueError(f"数据集 {dataset} 不支持 orient={orient!r}")
        kwargs["orient"] = orient
    ds = spec["cls"](index_csv, augment=augment, image_size=resolution,
                     geom_randomize=geom_randomize, **kwargs)
    sub = Subset(ds, idx)
    # persistent_workers：worker 进程跨 epoch 复用，避免每个 epoch 重新
    # fork（125 万行索引的 fork 成本很高，且是 epoch 边界卡顿的来源）。
    common = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=False, prefetch_factor=2 if num_workers > 0 else None,
                  persistent_workers=num_workers > 0)
    if distributed:
        # 每个 rank 只吃 1/world 的数据；shuffle 交给 sampler（DataLoader 的
        # shuffle 必须为 False）。每个 epoch 前要调 sampler.set_epoch(epoch)。
        sampler = DistributedSampler(sub, shuffle=shuffle)
        return DataLoader(sub, sampler=sampler, shuffle=False, **common)
    return DataLoader(sub, shuffle=shuffle, **common)


def init_distributed():
    """若由 torchrun 启动则初始化 NCCL 进程组。

    返回 (rank, world_size, local_rank, is_dist)。未设置 RANK/WORLD_SIZE 时
    返回 (0, 1, 0, False)，此时行为与单卡完全一致。
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0, False
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", 0))
    if not torch.cuda.is_available():
        raise RuntimeError("使用 torchrun 多卡训练需要 CUDA")
    torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    return rank, world, local, True


def geometric_angular_loss(pred, label, screen_cm, D):
    """与验收口径一致的几何角度损失（可微、数值稳健）。

    pred/label: [B,2] 归一化坐标；screen_cm: [B,2] 每样本屏宽/屏高(cm)。
    归一化坐标 -> 物理cm -> 相对屏中心 -> 构造视线方向(眼距 D) -> 夹角。
    用 atan2(|u×v|, u·v) 求 3D 单位向量夹角，梯度有界。
    """
    sw = screen_cm[:, 0]
    sh = screen_cm[:, 1]
    px = pred[:, 0] * sw - sw / 2
    py = pred[:, 1] * sh - sh / 2
    tx = label[:, 0] * sw - sw / 2
    ty = label[:, 1] * sh - sh / 2
    pv = torch.stack([px, py, torch.full_like(px, D)], dim=-1)
    tv = torch.stack([tx, ty, torch.full_like(tx, D)], dim=-1)
    pn = pv / pv.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    tn = tv / tv.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    dot = (pn * tn).sum(-1).clamp(-1.0, 1.0)                 # cos(θ)
    cross = torch.cross(pn, tn, dim=-1).norm(dim=-1)          # |sin(θ)|
    ang = torch.atan2(cross, dot) * 180.0 / PI
    return ang


#: 眼图对齐对照（--eye-shuffle）：置 True 时把眼部裁剪沿 batch 维 roll 一拍。
#: 目的：输入分布、参数量、算力全部不变，仅破坏"眼图 ↔ 当前帧"的对应关系 ⇒
#: 若模型仍能超过无眼基线，说明收益来自正则化/额外算力；若退化到无眼水平，
#: 说明收益确实来自虹膜细节。
EYE_SHUFFLE = False


def _shuffle_eyes(eye_l, eye_r):
    """对照实验用：破坏眼图与当前帧的对应（batch 内循环错位）。"""
    if not EYE_SHUFFLE or eye_l.shape[0] < 2:
        return eye_l, eye_r
    return eye_l.roll(1, dims=0), eye_r.roll(1, dims=0)


def predict_frames(model, dl, device):
    """在给定 DataLoader 上跑一遍，返回逐帧物理量（供任意分组聚合）。

    返回 dict，每个字段是 numpy 数组，长度 = 样本数：
        pred, label : [N, 2] 归一化屏幕坐标（预测 / 真值）
        cm_x, cm_y  : [N] 预测误差换算到屏幕物理厘米（带符号）
        ang_deg     : [N] 几何视线角度误差（度）

    注意：物理距离 cm_dist = hypot(cm_x, cm_y) 由 summarize 现算，**不在本返回值里**
    （此前 docstring 误列了它；2026-10-14 修正）。
    评估口径集中在这里，summarize 与 evaluate_split 共用，避免多处各写一套。
    """
    from configs.rgbdgaze_config import VIEWING_DISTANCE_CM as D

    model.eval()
    use_imu = getattr(model, "use_imu", False)
    use_geom = getattr(model, "use_geom", False)
    use_eye = getattr(model, "use_eye", False)

    preds, labels, cmxs, cmys, angs = [], [], [], [], []
    with torch.no_grad():
        for inp, label in dl:
            face = inp.rgb.to(device).float()
            label = label.to(device)
            imu = inp.imu.to(device).float() if use_imu else None
            geom = inp.geom.to(device).float() if use_geom else None
            if use_eye:
                # 眼部支流：数据集需以 eye_crops=True 构造，否则此处会 AttributeError
                eye_l, eye_r = _shuffle_eyes(inp.eye_l.to(device).float(),
                                             inp.eye_r.to(device).float())
                pred = model(face, imu=imu, geom=geom, eye_l=eye_l, eye_r=eye_r)
            else:
                pred = model(face, imu=imu, geom=geom)

            # 物理换算所需量来自数据项本身，避免依赖 DataLoader 的样本顺序
            cm_b = inp.screen_cm.to(device).float()
            scr_w_cm = cm_b[:, 0]
            scr_h_cm = cm_b[:, 1]
            cm_x = (pred[:, 0] - label[:, 0]) * scr_w_cm
            cm_y = (pred[:, 1] - label[:, 1]) * scr_h_cm

            # 几何视线角度：把屏幕点按「到屏幕中心的偏移 + 基准视距 D」构成 3D 向量，
            # 预测与真值各自归一化后取夹角。
            px = pred[:, 0] * scr_w_cm - scr_w_cm / 2
            py = pred[:, 1] * scr_h_cm - scr_h_cm / 2
            tx = label[:, 0] * scr_w_cm - scr_w_cm / 2
            ty = label[:, 1] * scr_h_cm - scr_h_cm / 2
            p_n = torch.stack([px, py, torch.full_like(px, D)], -1)
            t_n = torch.stack([tx, ty, torch.full_like(tx, D)], -1)
            p_n = p_n / p_n.norm(dim=-1, keepdim=True)
            t_n = t_n / t_n.norm(dim=-1, keepdim=True)
            cosim = (p_n * t_n).sum(-1).clamp(-1.0, 1.0)

            preds.append(pred.cpu().numpy())
            labels.append(label.cpu().numpy())
            cmxs.append(cm_x.cpu().numpy())
            cmys.append(cm_y.cpu().numpy())
            angs.append((torch.acos(cosim) * 180.0 / PI).cpu().numpy())

    if not preds:
        raise ValueError("DataLoader 为空，无法评估")
    return dict(
        pred=np.concatenate(preds), label=np.concatenate(labels),
        cm_x=np.concatenate(cmxs), cm_y=np.concatenate(cmys),
        ang_deg=np.concatenate(angs),
    )


def summarize(fr, sel=None):
    """由 predict_frames 的逐帧结果算多口径指标；sel = 布尔/整数索引子集。"""
    if sel is not None:
        fr = {k: v[sel] for k, v in fr.items()}
    pred, label = fr["pred"], fr["label"]
    cm_x, cm_y = fr["cm_x"], fr["cm_y"]
    ang_all = fr["ang_deg"]
    cnt = len(ang_all)
    if cnt == 0:
        return dict(l2_norm=0.0, l2_x=0.0, l2_y=0.0, em_dist=0.0, em_x=0.0,
                    em_y=0.0, ang_mean=0.0, ang_median=0.0, ang_top95_mean=0.0,
                    ang_p95=0.0, ang_ratio_le_191=0.0, count=0)
    diff = pred - label
    cm_dist = np.hypot(cm_x, cm_y)
    n95 = max(1, int(cnt * 0.95))
    top95_mean = float(np.sort(ang_all)[:n95].mean())
    return dict(
        l2_norm=float(np.linalg.norm(diff, axis=1).mean()),
        l2_x=float(np.abs(diff[:, 0]).mean()),
        l2_y=float(np.abs(diff[:, 1]).mean()),
        em_dist=float(cm_dist.mean()),
        em_x=float(np.abs(cm_x).mean()), em_y=float(np.abs(cm_y).mean()),
        ang_mean=float(ang_all.mean()),
        ang_median=float(np.median(ang_all)),
        ang_top95_mean=top95_mean,
        ang_p95=float(np.percentile(ang_all, 95)),
        ang_ratio_le_191=float((ang_all <= 1.91).mean() * 100),
        count=cnt,
    )


def evaluate_split(model, dl, device):
    """与 train.py 完全一致的多口径评估（predict_frames + summarize）。"""
    return summarize(predict_frames(model, dl, device))


class ModelEMA:
    """模型权重的指数移动平均（Exponential Moving Average）。

    训练时每步更新：ema = decay * ema + (1 - decay) * model_params。
    验证/保存用 EMA 权重（更平滑、泛化通常更好），这是视觉训练的标准技巧。
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self._backup = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        """把 EMA 权重临时写入模型（保存原权重以便恢复）。"""
        self._backup = {k: v.detach().clone()
                        for k, v in model.state_dict().items()
                        if k in self.shadow}
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if self._backup is not None:
            model.load_state_dict({**model.state_dict(), **self._backup}, strict=False)
            self._backup = None

    def state_dict(self):
        return {k: v for k, v in self.shadow.items()}


def preflight_gpu_check(rank, local_rank, is_dist):
    """启动前 GPU 自检：打印每张卡的型号，并用一次 matmul(+all_reduce) 验证可用。

    为什么需要：机器上某张卡故障（驱动 Xid 错误、掉卡）时，症状是"长跑到一半
    rank 某个进程 exitcode 1、其他 rank 被 SIGTERM"，且往往拿不到 traceback，
    排查成本极高。把这段放到训练开始前，坏卡会在几秒内暴露，并明确打印出
    "哪个 rank 落在哪张卡上"，而不是烧掉几小时。

    注意 CUDA_VISIBLE_DEVICES 的重映射：CUDA_VISIBLE_DEVICES=1,4 时
    LOCAL_RANK 0 对应的是**物理 1 号卡**，所以"rank0 挂了"很可能是 1 号卡坏了。
    """
    if not torch.cuda.is_available():
        return
    # 注意 set_device / get_device_name 也在 try 内：掉卡时它们同样会抛
    # "invalid device ordinal"，若放在 try 外，我们自己的排查提示就被绕过了。
    name = "?"
    try:
        torch.cuda.set_device(local_rank)
        name = torch.cuda.get_device_name(local_rank)
        # 每个 rank 都直接 print（不走 write_log）：需要每张卡一行，不只 rank0
        print(f"[preflight] rank {rank} local_rank {local_rank} -> {name}", flush=True)
        a = torch.randn(512, 512, device=f"cuda:{local_rank}")
        b = a @ a
        if is_dist:
            dist.all_reduce(b)      # 顺带验证 NCCL 通
            dist.barrier()
        torch.cuda.synchronize()
    except Exception as e:                       # noqa: BLE001  启动自检要给出明确信息
        raise RuntimeError(
            f"[preflight] rank {rank} 在 local_rank {local_rank}（{name}）上自检失败："
            f"{type(e).__name__}: {e}\n"
            f"          该卡很可能已故障，请用 nvidia-smi / dmesg -T | grep -i xid 确认，"
            f"并通过 CUDA_VISIBLE_DEVICES 排除它再重试。") from e
    if not bool(torch.isfinite(b).all()):
        raise RuntimeError(f"[preflight] rank {rank} 在 {name} 上算出了非有限值，卡可能已损坏")
    print(f"[preflight] rank {rank} 自检通过", flush=True)


def parse():
    p = argparse.ArgumentParser(
        description="DINOv2 注视回归训练/消融（RGBDGaze 与 GazeCapture 共用）")
    p.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="rgbdgaze",
                   help="rgbdgaze=域内训练/消融；gazecapture=大规模预训练")
    p.add_argument("--index-dir", default=None,
                   help="索引目录；默认按 --dataset 取各自 config 里的路径")
    p.add_argument("--init-ckpt", default=None,
                   help="从该权重起训（如 GazeCapture 预训练 -> RGBDGaze 微调）")
    p.add_argument("--batch-size", type=int, default=64,
                   help="**每张卡**的批大小（torchrun 多卡时有效批 = 该值 x 卡数）")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="学习率；多卡放大有效批后建议按 sqrt(有效批/24)x1e-4 起步")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--loss", choices=["mse", "mse_angle", "l1"], default="mse",
                   help="mse=纯MSE；mse_angle=MSE+几何角度；"
                        "l1=平均绝对误差（贴近验收口径的平均距离，预期减轻预测收缩）")
    p.add_argument("--geom", dest="use_geom", action="store_true", default=True,
                   help="注入屏幕几何 token（宽高比 + 物理尺寸，默认开）")
    p.add_argument("--no-geom", dest="use_geom", action="store_false",
                   help="关闭屏幕几何 token（消融用）")
    p.add_argument("--geom-randomize", type=float, default=None,
                   help="几何域随机化的触发概率（P2，把宽高比外推变内插）；"
                        "None=用各自 config 默认（rgbdgaze=0.35 / gazecapture=0.5）。"
                        "仅在 --augment 下生效，显式传 0 可关闭（消融用）")
    p.add_argument("--use-eye", dest="use_eye", action="store_true",
                   help="加入眼部放大支流：人脸 + 左右眼各一张裁剪（共用 DINOv2 骨干）。"
                        "眼框表由 tools/data/rgbdgaze/build_eye_boxes.py 生成")
    p.add_argument("--eye-shuffle", dest="eye_shuffle", action="store_true",
                   help="【对照实验】把眼部裁剪沿 batch 维 roll 一拍：输入分布与算力"
                        "完全不变，只破坏 眼图↔当前帧 的对应。用于分离眼支流收益"
                        "来自图像内容 还是来自正则化/额外算力（须与 --use-eye 同用）")
    p.add_argument("--eye-scale", type=float, default=0.0,
                   help="眼部裁剪边长 = eye_scale × 眼框高度（正方形，居中）。"
                        "0=历史行为（边长取 max(宽,高)，虹膜仅占约 28%%）。"
                        "取 1.5 左右可让虹膜占比升到约 2/3 ⇒ 等效虹膜像素提升数倍")
    p.add_argument("--unfreeze", type=int, default=0,
                   help="解冻 DINOv2 最后 N 个 block（0=全冻结）")
    p.add_argument("--diff-lr", action="store_true",
                   help="差分学习率：解冻的骨干用小lr(0.1x)，其余用 --lr")
    p.add_argument("--use-imu", action="store_true",
                   help="注入 IMU 姿态 token（需 index 含 imu 列）")
    p.add_argument("--ema", action="store_true",
                   help="使用 EMA 权重做验证与保存（decay=0.999）")
    p.add_argument("--augment", action="store_true",
                   help="训练集数据增强（水平翻转+注视镜像、几何抖动、光度扰动）")
    p.add_argument("--aug-shift", type=float, default=AUG_SHIFT,
                   help="裁剪中心随机偏移幅度（相对裁剪边长，配合 --augment）")
    p.add_argument("--aug-scale", type=float, default=AUG_SCALE,
                   help="裁剪边长随机缩放幅度（配合 --augment）")
    p.add_argument("--aug-rotate", type=float, default=AUG_ROTATE,
                   help="随机旋转角度（度，配合 --augment）")
    p.add_argument("--resolution", type=int, default=224,
                   help="模型输入分辨率（224 或 448）")
    p.add_argument("--tag", default=None,
                   help="输出目录标识（默认由超参自动生成）")
    return p.parse_args()


def main():
    args = parse()
    global EYE_SHUFFLE
    EYE_SHUFFLE = bool(getattr(args, "eye_shuffle", False))
    if EYE_SHUFFLE and not getattr(args, "use_eye", False):
        raise SystemExit("--eye-shuffle 需要与 --use-eye 同用（没有眼图可打乱）")
    rank, world, local_rank, is_dist = init_distributed()
    is_main = (rank == 0)
    if is_dist:
        device = f"cuda:{local_rank}"
    else:
        device = args.device if args.device != "auto" else DEFAULT_DEVICE

    # 可复现性：所有 rank 同一随机种子（数据由 DistributedSampler 切分，
    # 各 rank 的增强随机性仍互相独立）
    preflight_gpu_check(rank, local_rank, is_dist)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.index_dir is None:
        args.index_dir = str(DATASET_SPECS[args.dataset]["default_index"])

    if args.tag is None:
        args.tag = (f"loss{args.loss}_unfreeze{args.unfreeze}"
                    + ("_difflr" if args.diff_lr else "")
                    + ("_imu" if args.use_imu else "")
                    + ("_ema" if args.ema else "")
                    + ("_aug" if args.augment else "")
                    + (f"_res{args.resolution}" if args.resolution != 224 else ""))
    run_dirs = experiment_dirs(args.dataset, "ablation")
    save_dir = run_dirs["checkpoint"] / args.tag
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dirs["log"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 日志名带上卡数，避免单卡/多卡产物互相覆盖；只有 rank0 写文件，
    # 否则 8 个 rank 会同时往同一文件追加导致内容交错。
    log_file = log_dir / (f"{log_time}_ablation_{args.tag}"
                          + (f"_w{world}" if is_dist else "") + ".txt")

    def write_log(msg):
        if is_main:
            print(msg, flush=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    model = RGBDGazeDINOv2(
        unfreeze_last=args.unfreeze,
        use_imu=args.use_imu,
        use_geom=args.use_geom,
        use_eye=args.use_eye,
        img_size=args.resolution,
    ).to(device)
    if args.init_ckpt:
        state = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        # 用 any_res 版加载：init-ckpt 是"跨运行复用"的权重（如 224 预训练 ->
        # 448 微调），pos_embed 形状常不一致，普通 load_state_dict 会直接崩。
        missing, unexpected, resampled = load_state_dict_any_res(model, state)
        # 必须打印：缺 proj_geom 说明几何 token 是随机初始化的；
        # 出现 unexpected 说明架构不匹配（如残留的 depth 分支）。
        write_log(f"[init] 从 {args.init_ckpt} 载入权重")
        if resampled is not None:
            cur = tuple(model.state_dict()["rgb.backbone.pos_embed"].shape)
            write_log(f"[init] pos_embed 分辨率不一致 {resampled} -> {cur}，已插值")
        write_log(f"[init] 缺失键 {list(missing) or '无'} | 多余键 {list(unexpected) or '无'}")
    # DDP：先加载 init-ckpt 再包装，保证权重加载路径不依赖 DDP 的 key 前缀
    raw_model = model
    if is_dist:
        model = DDP(model, device_ids=[local_rank])
    criterion = nn.L1Loss() if args.loss == "l1" else nn.MSELoss()
    if args.diff_lr:
        # 差分学习率：被解冻的骨干层用 0.1x，其余（投影/融合/头）用全量 lr
        backbone_params, other_params = [], []
        # 用 raw_model：DDP 包装后参数名会带 "module." 前缀，前缀判断会失效
        for n, p in raw_model.named_parameters():
            if not p.requires_grad:
                continue
            (backbone_params if n.startswith("rgb.backbone") else other_params).append(p)
        optimizer = optim.Adam([
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": other_params, "lr": args.lr},
        ])
    else:
        optimizer = optim.Adam(raw_model.parameters(), lr=args.lr)
    # EMA 只在 rank0 维护：验证/保存都只在 rank0 做，其余 rank 白占显存。
    ema = ModelEMA(raw_model) if (args.ema and is_main) else None

    # geom_randomize 只在训练集上传：val/test 的 augment=False 会让数据集
    # 把它强制为 0，评估口径始终是真实几何。
    train_dl = build_dataloader(args.dataset, "train", Path(args.index_dir),
                                args.batch_size, True, args.num_workers,
                                augment=args.augment, resolution=args.resolution,
                                distributed=is_dist,
                                geom_randomize=args.geom_randomize,
                                eye_crops=args.use_eye,
                                eye_scale=args.eye_scale)
    val_dl = build_dataloader(args.dataset, "val", Path(args.index_dir),
                              args.batch_size, False, args.num_workers,
                              augment=False, resolution=args.resolution,
                              eye_crops=args.use_eye,
                              eye_scale=args.eye_scale)
    test_dl = build_dataloader(args.dataset, "test", Path(args.index_dir),
                               args.batch_size, False, args.num_workers,
                               augment=False, resolution=args.resolution,
                               eye_crops=args.use_eye,
                               eye_scale=args.eye_scale)

    tr_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    eff_bs = args.batch_size * world
    header = (f"{'='*70}\n 消融实验 | {args.tag}\n"
              f" 数据集={args.dataset}  索引={args.index_dir}\n"
              f" loss={args.loss} geom={args.use_geom} "
              f"geom_rand={train_dl.dataset.base.geom_randomize} "
              f"imu={args.use_imu} "
              f"ema={args.ema} augment={args.augment} unfreeze={args.unfreeze} "
              f"eye={args.use_eye} "
              f"diff_lr={args.diff_lr} "
              f"aug(shift/scale/rot)={args.aug_shift}/{args.aug_scale}/{args.aug_rotate}\n"
              f" 训练集 {len(train_dl.dataset)}"
              # DDP 下 len(train_dl.dataset) 是**全集**（DistributedSampler 在
              # 迭代时才切分），不写清每卡份数会让人误判每轮步数
              + (f"（每卡约 {ceil(len(train_dl.dataset) / world)}）" if is_dist else "")
              + f" / 验证集 {len(val_dl.dataset)} / 测试集 {len(test_dl.dataset)}\n"
              f" 分辨率={args.resolution} batch/卡={args.batch_size} "
              f"卡数={world} 有效批={eff_bs} lr={args.lr} epochs={args.epochs}\n"
              # 参考 g6 的 batch=24 / lr=1e-4，按 sqrt(有效批/24) 缩放；只提示不改值
              f" lr 参考: 按 sqrt 缩放的有效批基准为 "
              f"{1e-4 * (eff_bs / 24) ** 0.5:.2e}（g6: batch24/lr1e-4）\n"
              f" 可训练参数: {tr_params:,}   seed={args.seed}\n"
              f" 命令: {' '.join(sys.argv)}\n"
              # 复现多卡跑法：把 sys.argv 的脚本路径换成 "-m <模块>"，
              # 前面加 torchrun。CUDA_VISIBLE_DEVICES 原样带上（若设置了）。
              + (f" 复现: CUDA_VISIBLE_DEVICES="
                 f"{os.environ.get('CUDA_VISIBLE_DEVICES', '(全部可见)')} "
                 f"torchrun --standalone --nproc_per_node={world} -m "
                 f"scripts.train {' '.join(sys.argv[1:])}\n"
                 if is_dist else "")
              + f" 权重输出: {save_dir}\n 日志: {log_file}\n{'='*70}")
    write_log(header)

    # 部分数据集（或小样本子集）可能没有 val 划分：此时不做验证、每轮覆盖保存
    # best.pt（最终停在末轮权重），并在日志里明确警告，避免静默行为。
    has_val = len(val_dl.dataset) > 0
    if not has_val:
        write_log("[warn] 验证集为空，将不做验证、每轮覆盖 best.pt（末轮权重生效）")
    best_val = float("inf")
    for epoch in range(args.epochs):
        if is_dist:
            # DistributedSampler 需要每 epoch 变一次种子，否则各 rank 的
            # 分片顺序固定不变，等价于每轮喂同一批数据顺序。
            train_dl.sampler.set_epoch(epoch)
        model.train()
        run_loss = 0.0
        n_step = 0
        t0 = time.time()
        for i, (inp, label) in enumerate(train_dl):
            face = inp.rgb.to(device).float()
            label = label.to(device)
            imu = inp.imu.to(device).float() if args.use_imu else None
            geom = inp.geom.to(device).float() if args.use_geom else None
            optimizer.zero_grad()
            if args.use_eye:
                eye_l, eye_r = _shuffle_eyes(inp.eye_l.to(device).float(),
                                             inp.eye_r.to(device).float())
                pred = model(face, imu=imu, geom=geom, eye_l=eye_l, eye_r=eye_r)
            else:
                pred = model(face, imu=imu, geom=geom)
            loss_coord = criterion(pred, label)
            loss = loss_coord
            if args.loss == "mse_angle":
                scm = inp.screen_cm.to(device).float()
                loss_ang = geometric_angular_loss(
                    pred, label, scm, VIEWING_DISTANCE_CM).mean()
                loss = loss_coord + loss_ang
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(raw_model)
            run_loss += loss.item()
            n_step += 1
            if i % 100 == 0 or i == len(train_dl) - 1:
                write_log(f"[epoch {epoch}] step {i}/{len(train_dl)}  "
                          f"loss {loss.item():.4f}  {time.time()-t0:.1f}s")
                t0 = time.time()

        # 验证只在 rank0 做：其他 rank 在 barrier 等待（DDP 只在 backward
        # 时同步，rank0 单独跑前向不会与它们死锁）。
        saved_ema = ""
        vres = evaluate_split(raw_model, val_dl, device) if (has_val and is_main) else None
        if has_val and is_main and ema is not None:
            # 用 EMA 权重评估（训练权重继续训练，EMA 更平滑）
            ema.apply_to(raw_model)
            vres_ema = evaluate_split(raw_model, val_dl, device)
            ema.restore(raw_model)
            if vres_ema["ang_mean"] < vres["ang_mean"]:
                vres = vres_ema
                saved_ema = " [EMA]"
            else:
                saved_ema = " [raw更优]"
        mean_train = run_loss / max(n_step, 1)
        mean_ang = vres["ang_mean"] if vres else float("inf")
        saved = ""
        if is_main and (not has_val or mean_ang < best_val):
            best_val = mean_ang
            ckpt = save_dir / "best.pt"
            if ema is not None:
                ema.apply_to(raw_model)
                torch.save(raw_model.state_dict(), ckpt)
                ema.restore(raw_model)
            else:
                torch.save(raw_model.state_dict(), ckpt)
            saved = f"  [保存 → {ckpt} ({mean_ang:.2f}°)]"
        if is_dist:
            dist.barrier()      # 等 rank0 评估+保存完再进下一轮
        if vres is None:
            write_log(f"[epoch {epoch}]  train_loss {mean_train:.4f}  "
                      f"(无验证集){saved}")
        else:
            write_log(f"[epoch {epoch}]  train_loss {mean_train:.4f}  "
                      f"val_L2 {vres['l2_norm']:.4f}  val_em {vres['em_dist']:.2f}cm  "
                      f"val_ang 平均 {vres['ang_mean']:.2f}° | 中位 {vres['ang_median']:.2f}° | "
                      f"95%分位 {vres['ang_p95']:.2f}° | 前95%平均 {vres['ang_top95_mean']:.2f}° | "
                      f"≤1.91° {vres['ang_ratio_le_191']:.1f}%"
                      + (saved_ema if ema is not None else "") + f"{saved}")

    # ---- 训练结束：用 best 权重在 test 集做独立终评（test 不参与选模） ----
    # 同样只在 rank0 做；其他 rank 直接在 barrier 等结束。
    best_ckpt = save_dir / "best.pt"
    if is_main and best_ckpt.exists() and len(test_dl.dataset) > 0:
        raw_model.load_state_dict(torch.load(best_ckpt, map_location=device,
                                             weights_only=False))
        fres = evaluate_split(raw_model, test_dl, device)
        write_log("=" * 70)
        sel = "无验证集，取末轮" if not has_val else f"val {best_val:.2f}°"
        write_log(f"[final] 用 best 权重（{sel}）在 test 集独立终评:")
        write_log(f"[final] test_L2 {fres['l2_norm']:.4f}  "
                  f"test_em {fres['em_dist']:.2f}cm "
                  f"(x:{fres['em_x']:.2f} y:{fres['em_y']:.2f})")
        write_log(f"[final] test_ang 平均 {fres['ang_mean']:.2f}° | "
                  f"中位 {fres['ang_median']:.2f}° | "
                  f"95%分位 {fres['ang_p95']:.2f}° | "
                  f"前95%平均 {fres['ang_top95_mean']:.2f}° | "
                  f"≤1.91° {fres['ang_ratio_le_191']:.1f}% "
                  f"(n={fres['count']})")
        if args.dataset == "rgbdgaze":
            # 论文基线只对 RGBDGaze 口径成立（GazeCapture 含 iPad，屏幕物理
            # 尺寸达 20cm，cm 误差不可与 1.89 直接比较）
            write_log(f"[final] 对比基线: 论文 RGBD 1.89cm | "
                      f"{'已超越基线' if fres['em_dist'] < 1.89 else '未达基线'}")
        write_log("=" * 70)
    elif is_main:
        write_log("[final] 未找到 best.pt 或测试集为空，跳过 test 终评")
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


def run_with_rank_traceback():
    """包住 main()，让**每个 rank** 的异常带 rank 信息打到 stdout。

    为什么需要：torchrun 在 stdout 非 TTY（例如 `2>&1 | tee log`）时，
    失败 rank 的堆栈常常抓不到（表现为 `error_file: <N/A>`、Root Cause 只有
    exitcode 1），只剩"某个 rank 崩了、其余被 SIGTERM"，无法定位。
    这里显式打印，保证堆栈一定出现在日志流里。
    """
    import traceback
    rank = os.environ.get("RANK", "0")
    try:
        main()
    except BaseException:
        # 用 BaseException：KeyboardInterrupt / CUDA 侧异常也要留下记录
        print(f"\n{'=' * 70}\n[rank {rank}] 训练异常终止，完整堆栈如下：\n"
              f"{'=' * 70}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise


if __name__ == "__main__":
    run_with_rank_traceback()