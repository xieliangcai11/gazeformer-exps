"""无标签测试时自适应（TTA）的入口：先建先验，再逐会话适配 + 评估。

与 MetaPrompt(AAAI'24)/Alfa(AAAI'26) **同类设置**（冻结骨干、测试时极少参数、无标签目标域数据），
损失函数为我们自己的设计（见 `gazelab/models/rgbdgaze/tta.py` 的说明，避免误引）。

用法：
    # 1) 从训练划分的标签建"输出分布先验"（离线一次，CPU）
    python -m scripts.rgbdgaze.tta_personalize --build-prior \
        --out out/rgbdgaze/analysis/gaze_prior.json

    # 2) 逐会话无标签适配 + 评估（**需要 GPU**；评估用真值仅用于打分）
    CUDA_VISIBLE_DEVICES=0 python -m scripts.rgbdgaze.tta_personalize \
        --ckpt out/rgbdgaze/ablation/checkpoints/eye224_mgpu/best.pt --split test \
        --resolution 224 --use-eye --no-geom \
        --steps 300 --lr 1e-3 --rank 8 --adapt-frac 1.0

    # 3) 只验证"不报错"的小冒烟（CPU、2 帧 3 步，不做实验）
    CUDA_VISIBLE_DEVICES= python -m scripts.rgbdgaze.tta_personalize --smoke \
        --ckpt out/rgbdgaze/ablation/checkpoints/eye224_mgpu/best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _bootstrap() -> Path:
    here = Path(__file__).resolve()
    for d in (here.parent, *here.parents):
        if (d / "src" / "gazelab" / "__init__.py").is_file():
            for p in (str(d), str(d / "src")):
                if p not in sys.path:
                    sys.path.insert(0, p)
            return d
    raise RuntimeError(f"找不到项目根：从 {here} 向上未发现 src/gazelab/__init__.py")


_PROJECT_ROOT = _bootstrap()

from configs.rgbdgaze_config import RGBDGaze_INDEX_DIR  # noqa: E402
from gazelab.models.dinov2_gaze import RGBDGazeDINOv2, load_state_dict_any_res  # noqa: E402
from gazelab.models.dinov2_gaze.tta import (TTAConfig, adapt, build_prior,  # noqa: E402
                                         detach_adapter)
from scripts.train import (DATASET_SPECS, Subset, build_dataloader,  # noqa: E402
                           predict_frames)


def build_prior_cmd(args):
    index_csv = Path(args.index_csv)
    split_txt = Path(args.index_dir or RGBDGaze_INDEX_DIR) / "train.txt"
    paths = [ln.strip() for ln in split_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    want = set(paths)
    lab = []
    with open(index_csv, encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        fields = rd.fieldnames or []
        pcol = "rgb_path" if "rgb_path" in fields else "frame_path"   # RGBDGaze / GazeCapture
        for r in rd:
            if r[pcol] in want:
                lab.append([float(r["gaze_x"]) / float(r["screen_w"]),
                            float(r["gaze_y"]) / float(r["screen_h"])])
    prior = build_prior(np.array(lab))
    prior["source"] = f"train split ({len(lab)} frames) of {index_csv.name}"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prior, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[prior] {prior['source']}")
    print(f"[prior] mean={np.round(prior['mean'], 4).tolist()} "
          f"std={np.round(prior['std'], 4).tolist()} -> {out}")


def _sessions(index_dir, split):
    paths = [ln.strip() for ln in (Path(index_dir) / f"{split}.txt").read_text(
        encoding="utf-8").splitlines() if ln.strip()]
    sess = defaultdict(list)
    for k, p in enumerate(paths):
        sess[str(p).rsplit("/", 1)[0]].append(k)
    return paths, {s: sorted(v) for s, v in sess.items()}


def _apply_closed(fr, args, global_prior=None, log=print):
    """CMM：闭式**边际矩匹配**输出校正  y' = s⊙(y−c)+c+b（逐轴），c=(0.5,0.5) 屏心。

    s = σ_target/σ_pred、b 使校正后均值对齐 μ_target ⇒ 4 个参数、**零训练、零额外前向**
    （完全在既有预测上做逐帧算术 ✓）。

    目标边际 (μ_t, σ_t) 由 --prior-mode 决定：
      global  = 训练集标签先验（纯无标签）
      session = 本会话目标点集的边际（只用**边缘矩**，不做逐帧配对；真实部署里 App 自己
                决定展示哪些点 ⇒ 该边缘可无标签获得，但**不是**纯无标签设定 ⇒ 论文须分开报）
      blend   = 两者按 --prior-blend 插值
    拟合窗口：整会话（离线）／前 --prefix-frac（因果）／逐块 prequential（--closed-prequential）。
    返回新 fr；因果模式下附 tail_from ⇒ 调用方只统计该帧之后的 EM（避免自我美化 ✗）。
    """
    pred, lab = fr["pred"], fr["label"]
    n = len(pred)
    C = 0.5
    S = []                                  # 会话内 "cm / 归一化单位"（中位，剔除小分母）
    for ax, cm in ((0, "cm_x"), (1, "cm_y")):
        v = [float(fr[cm][i] / (pred[i, ax] - lab[i, ax])) for i in range(n)
             if abs(pred[i, ax] - lab[i, ax]) > 0.01]
        S.append(float(np.median(v)) if v else 0.0)

    def _fit(idx):
        p, l = pred[idx], lab[idx]
        if args.prior_mode == "global" and global_prior is not None:
            mu_t = np.asarray(global_prior["mean"]); sd_t = np.asarray(global_prior["std"])
        elif args.prior_mode == "blend" and global_prior is not None:
            a = float(args.prior_blend)
            mu_t = (1 - a) * np.asarray(global_prior["mean"]) + a * l.mean(0)
            sd_t = (1 - a) * np.asarray(global_prior["std"]) + a * l.std(0)
        else:
            mu_t, sd_t = l.mean(0), l.std(0)
        s = sd_t / (p.std(0) + 1e-6)
        return s, mu_t - (s * (p.mean(0) - C) + C)

    if args.closed_half_life and args.closed_half_life > 0:
        # ---- 因果在线 CMM：指数遗忘 EMA 边际（只用更早帧，O(n) 增量）----
        # 为什么需要它（本仓库实测）：会话内目标分布与 shrinkage 都会漂移 ⇒
        # 单一变换（含用真值拟合的 oracle 仿射）有天花板；短记忆 EMA 能跟踪漂移，
        # 实测 20/20 会话改善、无退化，并**超过**离线上界。
        hl = float(args.closed_half_life)
        dec = 0.5 ** (1.0 / hl)
        warm = max(20, min(100, n // 5))
        W = 0.0
        sp = np.zeros(2); spp = np.zeros(2); sl = np.zeros(2); sll = np.zeros(2)
        sbs = []
        _lag = int(args.closed_target_lag)
        _q = []                                   # 目标样本队列：按 _lag 延迟注入（归因诊断 ✓）
        for i in range(n):
            if i >= warm and W > 1e-9:
                mp = sp / W; ml = sl / W
                vp = np.maximum(spp / W - mp ** 2, 0.0)
                vl = np.maximum(sll / W - ml ** 2, 0.0)
                s = np.sqrt(vl) / np.maximum(np.sqrt(vp), 1e-6)
                if args.prior_mode == "global" and global_prior is not None:
                    # 纯无标签臂：目标矩固定为训练集先验（不用任何目标点信息）
                    mu_t = np.asarray(global_prior["mean"]); sd_t = np.asarray(global_prior["std"])
                    s = sd_t / np.maximum(np.sqrt(vp), 1e-6)
                else:
                    mu_t = ml
                if args.damp > 0:                     # 证据阻尼（实测：通常会**欠校正** ⇒ 默认关 ✓）
                    lam = W / (W + args.damp)
                    s = 1 + lam * (s - 1)
                sbs.append((s, mu_t - (s * (mp - C) + C)))
            else:
                sbs.append((np.ones(2), np.zeros(2)))
            W = W * dec + 1.0
            sp = sp * dec + pred[i]; spp = spp * dec + pred[i] ** 2
            # 目标统计：延迟 _lag 帧注入。
            # 只有显式 --oracle-targets 才允许注入**测试真值**；否则全程只用预测。
            if args.oracle_targets:
                _q.append(lab[i])
                if len(_q) > _lag:
                    _lo = _q.pop(0)
                    sl = sl * dec + _lo; sll = sll * dec + _lo ** 2
        tail_from = warm
    elif args.closed_prequential:
        w = max(20, n // 10)
        sbs = [_fit(np.arange(0, max((i // w) * w, 1))) for i in range(n)]
        tail_from = w
    elif args.prefix_frac < 1.0:
        tail_from = int(n * args.prefix_frac)
        sbs = [_fit(np.arange(0, max(tail_from, 1)))] * n
    else:
        tail_from = 0
        sbs = [_fit(np.arange(n))] * n
    sarr = np.stack([x[0] for x in sbs]); barr = np.stack([x[1] for x in sbs])
    dx = sarr[:, 0] * (pred[:, 0] - C) + C + barr[:, 0] - pred[:, 0]
    dy = sarr[:, 1] * (pred[:, 1] - C) + C + barr[:, 1] - pred[:, 1]
    out = dict(fr)
    out["pred"] = np.stack([pred[:, 0] + dx, pred[:, 1] + dy], 1)
    out["cm_x"], out["cm_y"] = fr["cm_x"] + dx * S[0], fr["cm_y"] + dy * S[1]
    out["tail_from"] = tail_from
    log(f"    [cmm] mode={args.prior_mode} "
        + (f"因果：前 {tail_from}/{n} 帧拟合，EM 只统计之后 {n - tail_from} 帧"
           if tail_from else f"离线：全会话拟合")
        + f"；s=[{sarr[-1][0]:.2f},{sarr[-1][1]:.2f}]"
        + (f"；ang_deg 仍为基线值（未随校正重算，仅影响角度报表 ✗）" if True else ""))
    return out


def main():
    ap = argparse.ArgumentParser(description="无标签测试时自适应（TTA）")
    ap.add_argument("--ckpt")
    ap.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="rgbdgaze")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--index-csv", default=str(Path(RGBDGaze_INDEX_DIR) / "index.csv"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--orient", choices=["raw", "rot90"], default="raw")
    ap.add_argument("--geom", dest="use_geom", action="store_true", default=None,
                    help="骨干含屏幕几何 token；默认读 ckpt 旁 train_args.json（缺失时 True）")
    ap.add_argument("--no-geom", dest="use_geom", action="store_false",
                    help="骨干不含几何 token（须与 ckpt 一致）")
    ap.add_argument("--use-eye", dest="use_eye", action="store_true", default=None,
                    help="骨干含眼支流；默认读 ckpt 旁 train_args.json（缺失时 True）")
    ap.add_argument("--no-eye", dest="use_eye", action="store_false",
                    help="无眼支流模型（如 noeye224_ctl）必须加此项："
                         "同时关掉模型与数据集的眼支流（否则数据集白算眼裁剪）")
    ap.add_argument("--eye-scale", type=float, default=0.0)
    ap.add_argument("--prior",
                    default=str(Path(_PROJECT_ROOT) / "out/rgbdgaze/analysis/gaze_prior.json"))
    # U2：矩匹配的**目标分布**从哪来。
    #   global  = 训练集标签全局先验（历史做法；对"会话点阵图案≠训练均值"的会话会过/欠校正 ✗）
    #   session = **本会话展示过的目标点集合的边缘分布**（只取边缘 mean/std，不做逐帧配对 ✓）
    #   blend   = 两者按 --prior-blend 插值
    # 诚实标注：session 模式用了"目标点集"这一侧信息。在真实部署里 App 自己决定展示哪些点 ⇒
    # 它**不需要**用户真值即可获得该边缘分布 ✓；但它也**不是**纯无标签设定 ⇒ 论文必须分开报。
    ap.add_argument("--prior-mode", choices=["global", "session", "blend"], default="global",
                    help="目标边际来源：global=训练集先验（纯无标签）/ session=本会话目标点边缘 / "
                         "blend=插值。后两者需要 --oracle-targets")
    ap.add_argument("--oracle-targets", action="store_true",
                    help="允许使用**测试集真值**构造目标分布（session/blend 先验、EMA 目标矩）。"
                         "开启后结果只能作为 oracle 上界，不能声明为无标签。默认关闭 ⇒ "
                         "只允许 --prior-mode global（训练集先验）")
    ap.add_argument("--prior-blend", type=float, default=0.5,
                    help="blend 模式下 session 边缘的权重")
    # CMM（闭式边际矩匹配）：y' = s⊙(y−c)+c+b，4 参数、零训练、零额外前向。
    #   --corr grad   = 历史做法：低秩适配器 + 梯度下降（默认，保持向后可比）
    #   --corr closed = 闭式矩匹配（本仓库新方法）：s=σ_target/σ_pred（逐轴）、b 使均值对齐
    # 目标边际来自 --prior-mode（global=训练集先验 / session=本会话目标点 / blend=插值）。
    # 因果口径：--prefix-frac 只用前 k 帧拟合，只在**之后的帧**上报告 EM（不自我美化）；
    #           --closed-prequential 用"每 10% 块、以此前全部帧重拟合"的在线版本。
    ap.add_argument("--corr", choices=["grad", "closed"], default="grad")
    ap.add_argument("--prefix-frac", type=float, default=0.3,
                    help="闭式因果模式：用会话前多少比例的帧拟合（之后才计入 EM）")
    ap.add_argument("--closed-prequential", action="store_true",
                    help="闭式在线模式：逐块用'此前全部帧'重拟合（prequential）")
    ap.add_argument("--closed-half-life", type=float, default=0.0,
                    help="闭式在线模式：指数遗忘半衰期（帧）。>0 时启用 EMA 因果版。"
                         "【实测边界·必须声明目标信息等级】收益对**目标信息时效**极敏感："
                         "hl=2（≈精确目标序列）L=0 时 −47.8%%，但目标滞后 2 帧就掉到 −4.0%% ✗；"
                         "hl=100 对 5–10 帧滞后稳健（−18.6%% / −11.1%% ✓），滞后 100 帧则崩（+119%% ✗）。"
                         "⇒ 小 hl 实际在**追踪目标的局部轨迹**；真实系统（有延迟）请用 hl≥100 并在"
                         "**声明延迟**的前提下用 LOSO 选 hl；配 --prior-mode global（纯无标签）会失稳 ✗")
    ap.add_argument("--damp", type=float, default=0.0,
                    help="证据阻尼强度（λ=W/(W+damp) 把 s 拉向 1）。实测通常会欠校正 ⇒ 默认 0")
    ap.add_argument("--closed-target-lag", type=int, default=0,
                    help="归因诊断：目标统计延迟多少帧注入（0=用到 t-1）。用于测量收益对"
                         "'目标信息时效'的敏感度；真实系统请按实际延迟设置并在该延迟下选 hl ✓")
    ap.add_argument("--probe-true-em", action="store_true",
                    help="【仅诊断 ✗ 真值不参与方法 ✓】每个 eval_every 步额外记录一次基于真值的"
                         "归一化误差，用于检验'无监督选择准则是否与真误差一致'（这是无标签 TTA "
                         "最可能的创新点所在：准则错位 ⇒ 永远选到烂步）")
    ap.add_argument("--policy-file", default=None,
                    help="学得的选步策略 json（由 scripts/analysis/tta_policy_check.py "
                         "--dump-policy 导出 ✓）。给定时在运行时按策略选步（只用无标签特征 ✓）"
                         "并恢复该步状态 ⇒ 替代内置的无监督准则 ✓")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0,
                    help="适配随机种子（影响 adapter 初始化与采样 ✓）。论文需多 seed 复核时使用 ✓")
    ap.add_argument("--w-moment", type=float, default=1.0, help="输出矩匹配权重（我们的）")
    ap.add_argument("--w-photo", type=float, default=1.0, help="光度一致性权重")
    ap.add_argument("--w-geo", type=float, default=0.0, help="几何一致性权重（注视不变的小仿射）")
    ap.add_argument("--w-sym", type=float, default=0.0, help="左右翻转对称权重（2D: x→1-x）")
    ap.add_argument("--gate", action="store_true", help="无标签安全门控：留出损失变差则回退")
    ap.add_argument("--gate-margin", type=float, default=0.0)
    ap.add_argument("--gate-alpha", type=float, default=0.75)
    ap.add_argument("--gate-min-gain", type=float, default=0.0,
                    help="留出损失至少改善该比例才接受（0=只要不变差）")
    ap.add_argument("--init-adapter", default="", help="adapter 初始状态（源域预训练/元学习）")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--adapt-frac", type=float, default=1.0,
                    help="每会话用前多少比例的**无标签**帧适配（1.0=全用；<1 为因果版本）")
    ap.add_argument("--holdout-frac", type=float, default=0.3,
                    help="无标签安全层：把每会话后这么多比例的帧留出，只用于按无监督损失选步")
    ap.add_argument("--eval-every", type=int, default=10,
                    help="安全层的验证间隔（步）")
    ap.add_argument("--no-early-stop", dest="early_stop", action="store_false", default=True,
                    help="关闭安全层（用于对照：不做早停/不回退）")
    ap.add_argument("--sessions", default=None,
                    help="只跑会话路径包含该子串的会话（多个用逗号分隔；用于定向复现某几个会话）")
    ap.add_argument("--max-sessions", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0, help="每会话适配帧数上限（调试用）")
    # 取帧 DataLoader 的 worker 数。此参数**此前被漏定义** ⇒ args.num_workers 抛
    # AttributeError，导致 5 臂损失消融 / 3 臂 lr 扫描全部在开跑前崩溃（服务器 10-16 实测）。
    # Windows 上 >0 会因 spawn 无法 pickle build_dataloader 内的局部 _worker_init ⇒ 本地用 0。
    ap.add_argument("--num-workers", type=int, default=4,
                    help="取帧 DataLoader 的 worker 数；Windows 本地请传 0（spawn 不能 pickle 局部 worker_init）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-preds", default=None,
                    help="逐帧预测 CSV（每帧 base/TTA 两套预测 + 误差 + 帧号），"
                         "供机制分析（偏差分解）与时序一致性分析（P2 选择准则诊断）")
    ap.add_argument("--build-prior", action="store_true", help="只建先验后退出")
    ap.add_argument("--smoke", action="store_true",
                    help="极小冒烟：1 个会话、2 帧、3 步（CPU 可跑，不做实验）")
    args = ap.parse_args()

    if args.build_prior:
        return build_prior_cmd(args)
    if not args.ckpt:
        raise SystemExit("需要 --ckpt（或 --build-prior）")

    # 开关默认值以 ckpt 侧车为准：否则"带几何/带眼的 ckpt"会被当无几何/无眼评估，
    # 对应权重只出现在 unexpected 里被丢弃，指标静默失真。
    _side = Path(args.ckpt).parent / "train_args.json"
    _cfg = {}
    if _side.is_file():
        try:
            _cfg = json.loads(_side.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            _cfg = {}
    # 没有 sidecar 时，从 checkpoint 的键**推断**结构；否则会静默丢掉/随机初始化
    # 几何/眼/IMU 分支（2026-10-27 冒烟实测：RGD ckpt 缺 proj_geom 两键）。
    import torch
    _state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    _sd = _state.get("model", _state)
    _inf_geom = any(k.startswith("proj_geom") for k in _sd)
    _inf_eye = any(k.startswith("proj_eye") for k in _sd)
    _inf_imu = any(k.startswith("proj_imu") for k in _sd)
    _inf_orient = False
    if _inf_geom:
        _w = _sd.get("proj_geom.weight")
        _inf_orient = bool(_w is not None and _w.shape[1] >= 4)
    src = _side.name if _side.is_file() else "checkpoint keys"
    if args.use_geom is None:
        args.use_geom = bool(_cfg["use_geom"]) if "use_geom" in _cfg else _inf_geom
        print(f"[cfg] use_geom={args.use_geom}（来源: {src}）")
    if args.use_eye is None:
        args.use_eye = bool(_cfg["use_eye"]) if "use_eye" in _cfg else _inf_eye
        print(f"[cfg] use_eye={args.use_eye}（来源: {src}）")
    args.use_imu = bool(_cfg["use_imu"]) if "use_imu" in _cfg else _inf_imu
    print(f"[cfg] use_imu={args.use_imu}（来源: {src}）")
    args.geom_orient = (bool(_cfg["geom_orient"]) if "geom_orient" in _cfg else _inf_orient) and bool(args.use_geom)
    print(f"[cfg] geom_orient={args.geom_orient}（来源: {src}）")

    # 硬性门禁：目标矩/目标点分布若来自测试真值，必须显式声明，否则结果只能当无标签报
    if not args.oracle_targets and args.prior_mode != "global":
        raise SystemExit(
            f"--prior-mode {args.prior_mode!r} 需要测试集真值构造目标分布 ⇒ "
            f"要么加 --oracle-targets（结果只能作为 oracle 上界），要么改用 "
            f"--prior-mode global（训练集先验，纯无标签）")
    if args.oracle_targets:
        print("[oracle] 已启用测试真值构造目标分布 ⇒ 结果**不可**声明为零用户真值")

    spec = DATASET_SPECS[args.dataset]
    index_dir = Path(args.index_dir) if args.index_dir else Path(spec["default_index"])
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
        print("[warn] CUDA 不可用 ⇒ 回退 CPU（结果仅供流程验证）")
    if args.smoke:
        print(f"[smoke] device={device}")
    print(f"[run] dataset={args.dataset} split={args.split} index={index_dir} "
          f"ckpt={args.ckpt}")
    print(f"[run] res={args.resolution} eye={args.use_eye} geom={args.use_geom} "
          f"geom_orient={args.geom_orient} eye_scale={args.eye_scale} "
          f"prior={args.prior}(mode={args.prior_mode})")
    print(f"[run] steps={args.steps} lr={args.lr} rank={args.rank} bs={args.batch_size} "
          f"w_moment={args.w_moment} w_photo={args.w_photo} w_geo={args.w_geo} "
          f"w_sym={args.w_sym} gate={args.gate}(margin={args.gate_margin},alpha={args.gate_alpha}) "
          f"init={args.init_adapter or '-'} early_stop={args.early_stop}")
    paths, sess_all = _sessions(index_dir, args.split)
    if args.smoke:
        sess_items = list(sess_all.items())[:1]
        args.steps, args.batch_size = 3, 2
        print("[smoke] 1 会话 / 2 帧 / 3 步；仅验证流程不报错（不做实验）")

    # ---- 数据（评估口径与 scripts/train.py 完全一致）----
    dl = build_dataloader(args.dataset, args.split, index_dir, 64, False, 0,
                          augment=False, resolution=args.resolution,
                          eye_crops=args.use_eye, orient=args.orient,
                          eye_scale=args.eye_scale,
                          geom_orient=args.geom_orient)
    ds = dl.dataset
    prior = json.loads(Path(args.prior).read_text(encoding="utf-8")) \
        if Path(args.prior).is_file() else {"mean": [0.5, 0.5], "std": [0.3, 0.3]}
    if not Path(args.prior).is_file():
        print(f"[warn] 先验缺失 {args.prior} ⇒ 用占位先验（请先 --build-prior）")

    # ---- 模型（冻结）----
    model = RGBDGazeDINOv2(unfreeze_last=0, use_geom=args.use_geom,
                           use_geom_orient=args.geom_orient,
                           use_imu=args.use_imu,
                           use_eye=args.use_eye, img_size=args.resolution).to(device)
    state = _state
    miss, unexp, resampled = load_state_dict_any_res(model, state)
    print(f"[load] {args.ckpt}  缺失键={len(miss)} 多余键={len(unexp)}")
    if miss:
        print(f"[load] missing[:12]={list(miss)[:12]}")
    if unexp:
        print(f"[load] unexpected[:12]={list(unexp)[:12]}")
    model.eval()

    def fwd(m, x):
        """与 predict_frames 相同的调用方式（含眼支流时给眼图）。"""
        if args.use_eye:
            return m(x)      # 眼支流需要 eye_l/eye_r ⇒ 见下方 batch 版
        return m(x)

    results = []
    dump_rows = []
    items = sess_items if args.smoke else list(sess_all.items())
    if args.sessions:
        _keys = [k.strip() for k in args.sessions.split(",") if k.strip()]
        items = [(s, v) for (s, v) in items if any(k in s for k in _keys)]
        print(f"[filter] 只跑 {len(items)} 个会话（匹配 {_keys}）")
    if args.max_sessions:
        items = items[:args.max_sessions]
    done_sessions = set()
    if args.dump_preds and Path(args.dump_preds).is_file():
        # 断点续跑：已落盘的会话直接跳过（GPU 偶发掉线时不用从头再来 ✓）
        with open(args.dump_preds, encoding="utf-8") as fh:
            rd = csv.reader(fh)
            next(rd, None)
            for row in rd:
                if row:
                    done_sessions.add(row[0])
                    dump_rows.append(row)
        before = len(items)
        items = [(s, v) for (s, v) in items if s not in done_sessions]
        print(f"[resume] 已有 {len(done_sessions)} 个会话，跳过；剩余 {len(items)}/{before}")

    def _flush_dump():
        if not args.dump_preds or not dump_rows:
            return
        dp = Path(args.dump_preds)
        dp.parent.mkdir(parents=True, exist_ok=True)
        tmp = dp.with_suffix(dp.suffix + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["session", "path", "frame_no",
                        "base_px", "base_py", "tta_px", "tta_py",
                        "label_x", "label_y",
                        "em_base_cm", "em_tta_cm", "ang_base_deg", "ang_tta_deg"])
            w.writerows(dump_rows)
            print(f"[dump] {dp} rows={len(dump_rows)}")
        tmp.replace(dp)

    for si, (s, idx) in enumerate(items):
        sub = Subset(ds, idx)
        n_adapt = max(1, int(len(idx) * args.adapt_frac))
        n_adapt = min(n_adapt, args.max_frames) if args.max_frames else n_adapt
        if args.smoke:
            n_adapt = 2
        # 取**无标签**输入（逐帧 kwargs ⇒ 采样时整帧配对，杜绝脸/眼错配）
        # ⚠️ 必须用多进程 DataLoader 取帧：此前是主进程单线程 cv2.imread，
        # 在训练 dataloader 抢 CPU 的机器上会被饿死（实测 30 分钟/会话 ✗）。
        # 只对"确实需要的 n_adapt 帧"建 loader 并**完整迭代**：
        # 此前用 break 提前跳出 ⇒ DataLoader 的 worker 被遗弃、逐会话泄漏 6 个进程 ⇒ 越跑越慢 ✗
        need = list(range(min(n_adapt, len(sub))))
        dl_f = torch.utils.data.DataLoader(Subset(sub, need), batch_size=32,
                                           shuffle=False,
                                           num_workers=int(args.num_workers))
        frames = []
        lab_chunks = []
        for inp, _lab in dl_f:
            lab_chunks.append(_lab.float())
            for b in range(inp.rgb.shape[0]):
                f = {"face": inp.rgb[b]}
                if args.use_eye:
                    f["eye_l"] = inp.eye_l[b]
                    f["eye_r"] = inp.eye_r[b]
                if args.use_geom:
                    f["geom"] = inp.geom[b]
                if args.use_imu:
                    f["imu"] = inp.imu[b]
                frames.append(f)
        # U2：矩匹配的目标分布（global=训练集全局先验 / session=本会话目标点边缘 / blend=插值）
        if args.prior_mode in ("session", "blend") and lab_chunks:
            _L = torch.cat(lab_chunks, 0)
            _pm = {"mean": _L.mean(0).tolist(), "std": (_L.std(0) + 1e-6).tolist()}
            if args.prior_mode == "session":
                prior_s = _pm
            else:
                _a = float(args.prior_blend)
                prior_s = {
                    "mean": [(1 - _a) * g + _a * t for g, t in zip(prior["mean"], _pm["mean"])],
                    "std": [(1 - _a) * g + _a * t for g, t in zip(prior["std"], _pm["std"])],
                }
            print(f"    [prior] mode={args.prior_mode} "
                  f"global μ={[round(x, 4) for x in prior['mean']]} σ={[round(x, 4) for x in prior['std']]} | "
                  f"session μ={[round(x, 4) for x in _pm['mean']]} σ={[round(x, 4) for x in _pm['std']]}")
        else:
            prior_s = prior
        print(f"[{si + 1}/{len(items)}] {s[-40:]}  适配帧 {len(frames)}（无标签）")

        # 基线（未适配）
        dl_s = torch.utils.data.DataLoader(sub, batch_size=64, shuffle=False, num_workers=0)
        fr0 = predict_frames(model, dl_s, device)
        if args.corr == "closed":
            # CMM：闭式矩匹配（零训练、零额外前向 ✓）
            fr1 = _apply_closed(fr0, args, prior, log=print)
        else:
            # 历史做法：无标签梯度适配（低秩适配器）
            _pol = None
            _st = {"adapter": None, "best": float("inf"), "step": -1, "state": None}
            _probe = None
            if args.probe_true_em and lab_chunks:
                _L = torch.cat(lab_chunks, 0)
                _k = min(64, len(frames))
                _kw = {k: torch.stack([frames[i][k] for i in range(_k)]).to(device).float()
                       for k in frames[0]}

                _init = {}      # 存"恒等"时的预测，用于先验无关的一致性准则 ✓
                _kw_ad = {k: torch.stack([frames[i][k] for i in range(min(64,
                                                              len(frames)))]).to(device).float()
                          for k in frames[0]}       # 适配池小批（用于"泛化一致性"特征 ✓）
                # ---- 策略选步（源域学得 ✓ 运行时不看真值 ✓）：在探针里算特征 + 记最优状态，
                #      adapt 结束后恢复该状态 ⇒ 与"最佳步保存"同机制、但不改 tta.py ✓
                _pol = (json.loads(Path(args.policy_file).read_text(encoding="utf-8"))
                        if args.policy_file else None)
                _st = {"adapter": None, "best": float("inf"), "step": -1, "state": None}

                def _score(f):
                    # Δ 基准用**本会话第 0 步**的特征 ✓（= 相对"本会话不动"的变化量 ✓）——
                    # 实测：若改用策略文件里写死的源域基准，目标域上会永远选第 0 步 ✗✗
                    f0 = _st.get("f0") or {}
                    row = [np.log1p(f["step"])]
                    for k in _pol["delta_feats"]:
                        v = f.get(k, float("nan")); b = f0.get(k, _pol["base"].get(k))
                        # nan 处理必须与离线 design() 逐字一致 ✓：此前只查了 v 是不是 nan ✗，
                        # 漏了"基准 b 是 nan"的情形 ⇒ 第 0 步 val=nan ⇒ 之后每步 Δval 都是 nan ⇒
                        # 分数全 nan（nan<best 恒 False）⇒ 策略永远选第 0 步 ✗✗（根因已定位 ✓）
                        row.append(0.0 if not (v == v) else ((v - b) if (b is not None and b == b) else v))
                    for k in _pol["raw_feats"]:
                        v = f.get(k, float("nan")); row.append(0.0 if v != v else v)
                    row.append(1.0 if f["step"] == 0 else 0.0)
                    z = [(x - m) / s for x, m, s in zip(row, _pol["mu"], _pol["sd"])]
                    return float(sum(a * b for a, b in zip(z, _pol["w"])))

                def _probe(step, v, best_step, adapter=None, _kw=_kw, _L=_L[:_k]):
                    with torch.no_grad():
                        _p = model(**_kw)
                        _pa = model(**_kw_ad)
                    _e = float((_p - _L.to(_p.device)).abs().mean())
                    # 候选无监督准则（都不需要真值 ✓）——用于与真误差做秩相关筛选 / 学策略
                    _sm = float((_p[1:] - _p[:-1]).abs().mean()) if len(_p) > 1 else float('nan')
                    _bx = float(adapter.bias.abs().mean()) if adapter is not None else float('nan')
                    _sg = _p.std(0).tolist()
                    _sga = float(_pa.std(0).mean())            # 适配池分散度 ✓（与探针批比较 ⇒ 一致性）
                    if "v0" not in _init: _init["v0"] = v if v == v else float('nan')
                    # 【先验无关的泛化一致性 ✓】把探针批分两半，看适配器对两半的**预测位移**是否一致：
                    # 越小 ⇒ 适配器造成的改变在"未见过的帧"上也同样出现 ⇒ 可信 ✓（不用任何先验/真值 ✓）
                    if step == 0 or "p" not in _init:
                        _init["p"] = _p.detach().clone()
                    _d = _p - _init["p"]
                    _h = len(_d) // 2
                    _cons = float((_d[:_h].mean(0) - _d[_h:].mean(0)).abs().mean()) if _h else float('nan')
                    _dmag = float(_d.abs().mean())             # 校正总位移 ✓（特征）
                    # 【输入质量 ✓ 无标签】亮度/对比度、模糊度代理（相邻像素差分 ⇒ 越小越糊 ✗）
                    _img = _kw["face"]
                    _bright = float(_img.mean())
                    _sharp = float((_img[:, :, 1:, :] - _img[:, :, :-1, :]).abs().mean())
                    # 【不确定度代理 ✓ 无标签】光度抖动下预测的分散度（1 次额外前向 ✓）
                    _g = 1.0 + 0.05 * torch.randn(len(_p), 1, 1, 1, device=_p.device)
                    with torch.no_grad():
                        _pj = model(**{k: (v * _g if v.dim() == 4 else v) for k, v in _kw.items()})
                    _jvar = float((_pj - _p).std(0).mean())
                    _feat = {"step": step, "val": v if v == v else float("nan"),
                             "sm": _sm, "cons": _cons, "bias": _bx,
                             "sig": float(np.hypot(_sg[0], _sg[1])), "sigad": _sga, "dmag": _dmag,
                             "bright": _bright, "sharp": _sharp, "jvar": _jvar}
                    _sc = float("nan")
                    if _pol is not None:                        # 策略选步（只看特征 ✓）
                        if adapter is not None:
                            _st["ref"] = adapter            # 始终保留真实适配器引用 ✓
                        if step == 0:
                            _st["f0"] = dict(_feat)         # 本会话第 0 步特征 = Δ 基准 ✓
                        _sc = _score(_feat)
                        if _sc < _st["best"]:
                            _st.update(best=_sc, step=step, adapter=adapter,
                                       state=({k: t.detach().clone()
                                               for k, t in adapter.state_dict().items()}
                                              if adapter is not None else None))
                    print(f"    [probe] step {step:>4}  val {v:.5f}  true(L1,归一化) {_e:.5f}  "
                          f"sm {_sm:.5f}  cons {_cons:.5f}  bias {_bx:.5f}  "
                          f"sig [{_sg[0]:.4f},{_sg[1]:.4f}]  sigad {_sga:.4f}  dmag {_dmag:.5f}  "
                          f"bright {_bright:.4f}  sharp {_sharp:.5f}  jvar {_jvar:.5f}  "
                          f"pol {_sc:.5f}  best@{best_step}")

            if _probe is not None:      # 先测"恒等"（未挂适配器 ✓）⇒ 曲线必须有的零点 ✓
                _probe(0, float("nan"), 0)
            adapt(model, frames, prior_s, TTAConfig(steps=args.steps, lr=args.lr,
                                                 rank=args.rank,
                                                 seed=args.seed,
                                                 batch_size=args.batch_size,
                                                 w_moment=args.w_moment,
                                                 w_cons=args.w_photo,
                                                 w_geo=args.w_geo,
                                                 w_sym=args.w_sym,
                                                 gate=args.gate,
                                                 gate_margin=args.gate_margin,
                                                 gate_alpha=args.gate_alpha,
                                                 gate_min_gain=args.gate_min_gain,
                                                 init_adapter=args.init_adapter,
                                                 holdout_frac=args.holdout_frac,
                                                 eval_every=args.eval_every,
                                                 early_stop=args.early_stop),
                  probe=_probe)
            if _pol is not None and _st["step"] >= 0:      # 恢复策略选中的步（运行时零真值 ✓）
                _ref = _st.get("ref")
                if _st["state"] is not None and _st["adapter"] is not None:
                    _st["adapter"].load_state_dict(_st["state"])
                elif _ref is not None:                     # 选中"恒等" ⇒ 归零适配器 ✓
                    with torch.no_grad():
                        _ref.bias.zero_()
                        if getattr(_ref, "up", None) is not None:
                            _ref.up.weight.zero_()
                print(f"    [policy] 采用策略选中步 {_st['step']}（预测误差 {_st['best']:.5f} ✓）")
            fr1 = predict_frames(model, dl_s, device)
            detach_adapter(model)
            # 因果口径（--adapt-frac < 1）：只统计**适配窗口之后**的帧，否则会含"已看过答案"的帧
            # ⇒ 与闭式因果臂不可比、且会自我美化 ✗（2026-10-16 发现并修）。
            fr1["tail_from"] = int(n_adapt) if args.adapt_frac < 1.0 else 0
        if args.dump_preds:
            # 逐帧落盘：base 与 TTA 的预测/误差/角度 + 帧号（= 时序顺序，文件名即帧号）
            for j, _p in enumerate([paths[i] for i in idx]):
                _stem = Path(_p).stem
                dump_rows.append([
                    s, _p, int(_stem) if _stem.isdigit() else -1,
                    float(fr0["pred"][j][0]), float(fr0["pred"][j][1]),
                    float(fr1["pred"][j][0]), float(fr1["pred"][j][1]),
                    float(fr0["label"][j][0]), float(fr0["label"][j][1]),
                    float(np.hypot(fr0["cm_x"][j], fr0["cm_y"][j])),
                    float(np.hypot(fr1["cm_x"][j], fr1["cm_y"][j])),
                    float(fr0["ang_deg"][j]), float(fr1["ang_deg"][j]),
                ])
        em0 = np.hypot(fr0["cm_x"], fr0["cm_y"])
        em1 = np.hypot(fr1["cm_x"], fr1["cm_y"])
        t0 = int(fr1.get("tail_from", 0))     # 因果模式：只统计"适配窗口之后"的帧（不自我美化 ✓）
        e0 = float(np.mean(em0[t0:])); e1 = float(np.mean(em1[t0:]))
        print(f"    EM：不标定 {e0:.3f} → 校正后 {e1:.3f}"
              f"（{100 * (e1 / e0 - 1):+.1f}%）"
              + (f"  [后段口径：{t0}/{len(em0)} 帧起]" if t0 else ""))
        results.append((s, e0, e1, len(frames)))
        _flush_dump()          # 每会话落盘 ⇒ 崩溃后可从断点续跑 ✓
    if len(results) > 1:
        a = np.array([[r[1], r[2]] for r in results])
        print(f"\n  总体：不标定 {a[:, 0].mean():.3f} → TTA 后 {a[:, 1].mean():.3f}"
              f"（{100 * (a[:, 1].mean() / a[:, 0].mean() - 1):+.1f}%）  "
              f"会话数 {len(results)}")
    if args.dump_preds and dump_rows:
        _flush_dump()
        print(f"  逐帧 CSV: {args.dump_preds}（{len(dump_rows)} 帧）")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["session", "em_base", "em_tta", "adapt_frames"])
            w.writerows(results)
        print(f"  明细 CSV: {out}")


if __name__ == "__main__":
    main()
