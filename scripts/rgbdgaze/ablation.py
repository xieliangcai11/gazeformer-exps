"""RGBDGaze 新模型消融实验脚本（独立，不改动 train.py / train_baseline.py）。

消融维度（可组合）：
    --loss mse|mse_angle   损失函数
        mse        : 纯 MSE（与旧基线同口径）
        mse_angle  : MSE + 几何角度损失（训练时就按"物理cm + 30cm眼距"算，
                     与 evaluate_split 验收口径一致；注意不是早期那个有缺陷的代理角度）
    --use-depth / --no-depth
        是否使用逆深度分支（RGB-only 消融）
    --unfreeze N
        解冻 DINOv2 最后 N 个 block 参与微调（N=0 表示全部冻结）

输出：独立目录 out/rgbdgaze/ablation/<tag>/（不污染 train.py 的 best_dinov2.pt）。
指标：与 train.py 完全一致的多口径（L2 / cm / 角度 mean·median·p95·前95%·≤1.91%）。

用法示例：
    python -m scripts.rgbdgaze.ablation --loss mse --use-depth --unfreeze 0 --epochs 8
    python -m scripts.rgbdgaze.ablation --loss mse --no-depth --unfreeze 0 --epochs 8
    python -m scripts.rgbdgaze.ablation --loss mse_angle --unfreeze 2 --epochs 8
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from torch.utils.data import DataLoader
from datetime import datetime

from configs.gaze360_config import DEVICE as DEFAULT_DEVICE, experiment_dirs
from configs.rgbdgaze_config import RGBDGaze_INDEX_DIR, VIEWING_DISTANCE_CM
from gazelab.datasets.rgbdgaze import (RGBDGazeDataset, rgb_preprocess,
                                       inverse_depth_preprocess)
from gazelab.models.rgbdgaze import RGBDGazeDINOv2

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


def build_dataloader(split, index_dir, batch_size, shuffle, num_workers,
                     augment=False, crop_jitter=0.0, resolution=None):
    index_csv = Path(index_dir) / "index.csv"
    list_txt = Path(index_dir) / f"{split}.txt"
    paths = [l.strip() for l in list_txt.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    import csv
    with open(index_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rgb_to_idx = {r["rgb_path"]: i for i, r in enumerate(rows)}
    idx = [rgb_to_idx[p] for p in paths]

    ds = RGBDGazeDataset(index_csv,
                         color_transform=rgb_preprocess,
                         depth_transform=inverse_depth_preprocess,
                         augment=augment, crop_jitter=crop_jitter,
                         image_size=resolution)
    sub = Subset(ds, idx)
    return DataLoader(sub, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)


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


def evaluate_split(model, dl, device):
    """与 train.py 完全一致的多口径评估。"""
    from configs.rgbdgaze_config import VIEWING_DISTANCE_CM as D

    model.eval()
    dset = dl.dataset
    cm_arr = np.asarray(dset.cm_px, dtype=np.float32)
    scr = np.asarray(dset.screen_size, dtype=np.float32)

    L2n = 0.0; xA = 0.0; yA = 0.0; cnt = 0
    cma = [0.0, 0.0, 0.0]
    ang_acc = []
    start = 0
    with torch.no_grad():
        for inp, label in dl:
            face = inp.rgb.to(device).float()
            depth = inp.depth.to(device).float()
            label = label.to(device)
            imu = inp.imu.to(device).float() if getattr(model, "use_imu", False) else None
            pred = model(face, depth, imu=imu)
            diff = pred - label
            B = label.size(0)
            cm_b = torch.from_numpy(cm_arr[start:start+B]).to(device)
            scr_b = torch.from_numpy(scr[start:start+B]).to(device)
            cm_x = diff[:, 0] * scr_b[:, 0] * cm_b[:, 0]
            cm_y = diff[:, 1] * scr_b[:, 1] * cm_b[:, 1]
            cm_dist = torch.sqrt(cm_x**2 + cm_y**2)
            L2n += diff.norm(dim=-1).sum().item()
            xA += diff[:, 0].abs().sum().item()
            yA += diff[:, 1].abs().sum().item()
            cma[0] += cm_dist.sum().item()
            cma[1] += cm_x.abs().sum().item()
            cma[2] += cm_y.abs().sum().item()

            scr_w_cm = scr_b[:, 0] * cm_b[:, 0]
            scr_h_cm = scr_b[:, 1] * cm_b[:, 1]
            px = pred[:, 0] * scr_w_cm - scr_w_cm / 2
            py = pred[:, 1] * scr_h_cm - scr_h_cm / 2
            tx = label[:, 0] * scr_w_cm - scr_w_cm / 2
            ty = label[:, 1] * scr_h_cm - scr_h_cm / 2
            p_vec = torch.stack([px, py, torch.full_like(px, D)], dim=-1)
            t_vec = torch.stack([tx, ty, torch.full_like(tx, D)], dim=-1)
            p_n = p_vec / p_vec.norm(dim=-1, keepdim=True)
            t_n = t_vec / t_vec.norm(dim=-1, keepdim=True)
            cosim = (p_n * t_n).sum(-1).clamp(-1.0, 1.0)
            ang_acc.append((torch.acos(cosim) * 180.0 / PI).cpu().numpy())
            cnt += B
            start += B

    ang_all = np.concatenate(ang_acc) if ang_acc else np.array([0.0])
    n = len(ang_all)
    sorted_a = np.sort(ang_all)
    top95_mean = float(sorted_a[:int(n * 0.95)].mean()) if n > 0 else 0.0
    return dict(
        l2_norm=L2n/max(cnt, 1), l2_x=xA/max(cnt, 1), l2_y=yA/max(cnt, 1),
        em_dist=cma[0]/max(cnt, 1),
        em_x=cma[1]/max(cnt, 1), em_y=cma[2]/max(cnt, 1),
        ang_mean=float(ang_all.mean()),
        ang_median=float(np.median(ang_all)),
        ang_top95_mean=top95_mean,
        ang_p95=float(np.percentile(ang_all, 95)),
        ang_ratio_le_191=float((ang_all <= 1.91).mean() * 100),
        count=cnt,
    )


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


def parse():
    p = argparse.ArgumentParser(description="RGBDGaze DINOv2 消融实验")
    p.add_argument("--index-dir", default=str(RGBDGaze_INDEX_DIR))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--loss", choices=["mse", "mse_angle"], default="mse",
                   help="mse=纯MSE；mse_angle=MSE+几何角度")
    p.add_argument("--use-depth", dest="use_depth", action="store_true",
                   default=True, help="使用逆深度分支（默认开）")
    p.add_argument("--no-depth", dest="use_depth", action="store_false",
                   help="只用 RGB，关闭逆深度分支")
    p.add_argument("--depth-mode", choices=["token", "attn"], default="token",
                   help="深度用法：token=旧版grid token；attn=特征图空间注意力相乘(论文同款)")
    p.add_argument("--unfreeze", type=int, default=0,
                   help="解冻 DINOv2 最后 N 个 block（0=全冻结）")
    p.add_argument("--diff-lr", action="store_true",
                   help="差分学习率：解冻的骨干用小lr(0.1x)，其余用 --lr")
    p.add_argument("--use-imu", action="store_true",
                   help="注入 IMU 姿态 token（需 index 含 imu 列）")
    p.add_argument("--ema", action="store_true",
                   help="使用 EMA 权重做验证与保存（decay=0.999）")
    p.add_argument("--augment", action="store_true",
                   help="训练集数据增强（水平翻转+注视镜像、裁剪抖动、光度扰动）")
    p.add_argument("--crop-jitter", type=float, default=0.05,
                   help="裁剪框抖动比例（默认0.05，配合 --augment）")
    p.add_argument("--resolution", type=int, default=224,
                   help="模型输入分辨率（224 或 448）")
    p.add_argument("--tag", default=None,
                   help="输出目录标识（默认由超参自动生成）")
    return p.parse_args()


def main():
    args = parse()
    device = args.device if args.device != "auto" else DEFAULT_DEVICE

    if args.tag is None:
        args.tag = (f"loss{args.loss}_depth{args.use_depth}"
                    f"{args.depth_mode}_unfreeze{args.unfreeze}"
                    + ("_difflr" if args.diff_lr else "")
                    + ("_imu" if args.use_imu else "")
                    + ("_ema" if args.ema else "")
                    + ("_aug" if args.augment else "")
                    + (f"_res{args.resolution}" if args.resolution != 224 else ""))
    run_dirs = experiment_dirs("rgbdgaze", "ablation")
    save_dir = run_dirs["checkpoint"] / args.tag
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dirs["log"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{log_time}_ablation_{args.tag}.txt"

    def write_log(msg):
        print(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    model = RGBDGazeDINOv2(
        use_depth=args.use_depth,
        unfreeze_last=args.unfreeze,
        depth_mode=args.depth_mode,
        use_imu=args.use_imu,
        img_size=args.resolution,
    ).to(device)
    criterion = nn.MSELoss()
    if args.diff_lr:
        # 差分学习率：被解冻的骨干层用 0.1x，其余（投影/融合/头）用全量 lr
        backbone_params, other_params = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (backbone_params if n.startswith("rgb.backbone") else other_params).append(p)
        optimizer = optim.Adam([
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": other_params, "lr": args.lr},
        ])
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ema = ModelEMA(model) if args.ema else None

    train_dl = build_dataloader("train", Path(args.index_dir), args.batch_size,
                                True, args.num_workers,
                                augment=args.augment, crop_jitter=args.crop_jitter,
                                resolution=args.resolution)
    val_dl = build_dataloader("val", Path(args.index_dir), args.batch_size,
                              False, args.num_workers,
                              augment=False, crop_jitter=0.0,
                              resolution=args.resolution)
    test_dl = build_dataloader("test", Path(args.index_dir), args.batch_size,
                               False, args.num_workers,
                               augment=False, crop_jitter=0.0,
                               resolution=args.resolution)

    tr_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    header = (f"{'='*70}\n 消融实验 | {args.tag}\n"
              f" loss={args.loss} depth={args.use_depth} "
              f"unfreeze={args.unfreeze}\n"
              f" 训练集 {len(train_dl.dataset)} / 验证集 {len(val_dl.dataset)} / 测试集 {len(test_dl.dataset)}\n"
              f" batch={args.batch_size} lr={args.lr} epochs={args.epochs}\n"
              f" 可训练参数: {tr_params:,}\n"
              f" 输出: {save_dir}\n{'='*70}")
    write_log(header)

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        run_loss = 0.0
        n_step = 0
        t0 = time.time()
        for i, (inp, label) in enumerate(train_dl):
            face = inp.rgb.to(device).float()
            depth = inp.depth.to(device).float()
            label = label.to(device)
            imu = inp.imu.to(device).float() if args.use_imu else None
            optimizer.zero_grad()
            pred = model(face, depth, imu=imu)
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
                ema.update(model)
            run_loss += loss.item()
            n_step += 1
            if i % 100 == 0 or i == len(train_dl) - 1:
                write_log(f"[epoch {epoch}] step {i}/{len(train_dl)}  "
                          f"loss {loss.item():.4f}  {time.time()-t0:.1f}s")
                t0 = time.time()

        vres = evaluate_split(model, val_dl, device)
        if ema is not None:
            # 用 EMA 权重评估（训练权重继续训练，EMA 更平滑）
            ema.apply_to(model)
            vres_ema = evaluate_split(model, val_dl, device)
            ema.restore(model)
            if vres_ema["ang_mean"] < vres["ang_mean"]:
                vres = vres_ema
                saved_ema = " [EMA]"
            else:
                saved_ema = " [raw更优]"
        mean_train = run_loss / max(n_step, 1)
        mean_ang = vres["ang_mean"]
        saved = ""
        if mean_ang < best_val:
            best_val = mean_ang
            ckpt = save_dir / "best.pt"
            if ema is not None:
                ema.apply_to(model)
                torch.save(model.state_dict(), ckpt)
                ema.restore(model)
            else:
                torch.save(model.state_dict(), ckpt)
            saved = f"  [保存 → {ckpt} ({mean_ang:.2f}°)]"
        write_log(f"[epoch {epoch}]  train_loss {mean_train:.4f}  "
                  f"val_L2 {vres['l2_norm']:.4f}  val_em {vres['em_dist']:.2f}cm  "
                  f"val_ang 平均 {vres['ang_mean']:.2f}° | 中位 {vres['ang_median']:.2f}° | "
                  f"95%分位 {vres['ang_p95']:.2f}° | 前95%平均 {vres['ang_top95_mean']:.2f}° | "
                  f"≤1.91° {vres['ang_ratio_le_191']:.1f}%"
                  + (saved_ema if ema is not None else "") + f"{saved}")

    # ---- 训练结束：用 best 权重在 test 集做独立终评（test 不参与选模） ----
    best_ckpt = save_dir / "best.pt"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device,
                                         weights_only=False))
        fres = evaluate_split(model, test_dl, device)
        write_log("=" * 70)
        write_log(f"[final] 用 best 权重（val {best_val:.2f}°）在 test 集独立终评:")
        write_log(f"[final] test_L2 {fres['l2_norm']:.4f}  "
                  f"test_em {fres['em_dist']:.2f}cm "
                  f"(x:{fres['em_x']:.2f} y:{fres['em_y']:.2f})")
        write_log(f"[final] test_ang 平均 {fres['ang_mean']:.2f}° | "
                  f"中位 {fres['ang_median']:.2f}° | "
                  f"95%分位 {fres['ang_p95']:.2f}° | "
                  f"前95%平均 {fres['ang_top95_mean']:.2f}° | "
                  f"≤1.91° {fres['ang_ratio_le_191']:.1f}% "
                  f"(n={fres['count']})")
        write_log(f"[final] 对比基线: 论文 RGBD 1.89cm | "
                  f"{'已超越基线' if fres['em_dist'] < 1.89 else '未达基线'}")
        write_log("=" * 70)
    else:
        write_log("[final] 未找到 best.pt，跳过 test 终评")


if __name__ == "__main__":
    main()