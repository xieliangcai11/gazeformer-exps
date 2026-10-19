"""逐用户误差分解（假设无关的测量）。

目的：在**不预设任何校准方法**的前提下，量出"每人一个常数修正"这个前提到底有多少空间。
它限定了所有校准方向的上限，因此是决定"要不要继续投入校准"的先行实验。

两个测量：

M1 误差分解 —— 对每个测试用户，把均方误差拆成两块：
        E||e||^2 = ||b||^2 + Var(e)
    b  = 该用户的平均误差向量（"每人一个常数"能吃掉的部分）
    Var= 误差围绕该均值的波动（任何校准都改不动）
    "偏置占比" = ||b||^2 / E||e||^2  —— 这个数就是常数校准的理论上限

M2 偏置稳定性 —— 同一个用户内部，偏置是否稳定？
    (a) 跨姿势：按 sitting/standing/walking/lying 分别算 b，看它们的一致程度
    (b) 跨时段：按帧序前后对半切，看 b 的一致程度
    若用户内部都不稳定，"每人一个常数"这个前提本身不成立，M1 的上限也拿不到。

口径：误差一律换算到屏幕物理厘米（与训练期验收一致，复用 ablation.predict_frames）。

用法：
    python -m scripts.analysis.bias_decomposition \
        --ckpt out/rgbdgaze/ablation/checkpoints/g6_nodepth_448_aug/best.pt \
        --no-geom --resolution 448
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

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

from configs.paths import DEVICE as DEFAULT_DEVICE, experiment_dirs  # noqa: E402
from scripts.train import DATASET_SPECS, Subset, predict_frames  # noqa: E402
from gazelab.models.dinov2_gaze import load_state_dict_any_res  # noqa: E402


def load_ckpt(model, ckpt, log):
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    # ---- 守卫（2026-10-15 审计 B6）----
    # ckpt 带眼支流权重、而模型没建眼支流时，这里会**静默**给出与主评估分叉的数字
    # （曾导致分析结论与主表不一致）。宁可失败，也不要一个"看起来对"的错数字。
    _st_eye = any(".proj_eye." in k or k.startswith("proj_eye.") for k in state)
    _mdl_eye = any(k.startswith("proj_eye.") for k in model.state_dict())
    if _st_eye and not _mdl_eye:
        raise SystemExit(
            f"{ckpt} 带眼支流权重（proj_eye.*），但本脚本按无眼支流建模 ⇒ "
            f"结果会与主评估口径分叉。请改用 dump_predictions.py（支持 --use-eye/"
            f"--eye-scale/--eye-shuffle）导出逐帧预测 CSV，再做后续分析。")
    # any_res：--ckpt 与 --resolution 是独立指定的，权重可能来自另一种分辨率
    # （如 224 预训练 -> 448 分析），此时 pos_embed 形状不一致需先插值。
    missing, unexpected, resampled = load_state_dict_any_res(model, state)
    log(f"[load] {ckpt}")
    if resampled is not None:
        cur = tuple(model.state_dict()["rgb.backbone.pos_embed"].shape)
        log(f"[load] pos_embed {resampled} -> {cur}（已插值）")
    log(f"[load] 缺失键 {list(missing) or '无'} | 多余键 {list(unexpected) or '无'}")


def main():
    ap = argparse.ArgumentParser(description="逐用户误差分解（M1/M2）")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="rgbdgaze")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--split", default="test", help="用哪个划分做分析")
    ap.add_argument("--resolution", type=int, default=448)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--no-geom", dest="use_geom", action="store_false", default=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = args.device if args.device != "auto" else DEFAULT_DEVICE
    if args.index_dir is None:
        args.index_dir = str(DATASET_SPECS[args.dataset]["default_index"])

    log_dir = experiment_dirs(args.dataset, "analysis")["log"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{datetime.now():%Y%m%d_%H%M%S}_bias_decomposition.txt"
    lines = []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("=" * 78)
    log(f"逐用户误差分解 | ckpt={args.ckpt}")
    log(f"数据集={args.dataset} 划分={args.split} 分辨率={args.resolution} "
        f"use_geom={args.use_geom}")
    log("=" * 78)

    # ---- 模型与数据 ----
    from gazelab.models.dinov2_gaze import RGBDGazeDINOv2
    model = RGBDGazeDINOv2(img_size=args.resolution, use_geom=args.use_geom).to(device)
    load_ckpt(model, args.ckpt, log)
    model.eval()

    spec = DATASET_SPECS[args.dataset]
    index_csv = Path(args.index_dir) / "index.csv"
    list_txt = Path(args.index_dir) / f"{args.split}.txt"
    paths = [ln.strip() for ln in list_txt.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    import csv
    with open(index_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    p2i = {r[spec["path_col"]]: i for i, r in enumerate(rows)}
    idx = [p2i[p] for p in paths]

    ds = spec["cls"](index_csv, image_size=args.resolution, **spec["kwargs"])
    dl = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers)
    fr = predict_frames(model, dl, device)      # 与 idx 同序

    e = np.stack([fr["cm_x"], fr["cm_y"]], axis=1)          # [N,2] 物理厘米
    subj = np.array([rows[i]["subject"] for i in idx])
    act = np.array([rows[i].get("activity", "") for i in idx])
    order = np.array([int(rows[i].get("frame_index", 0)) for i in idx])

    log(f"样本 {len(idx)} 帧 / {len(set(subj))} 名被试")
    log("")

    # ---------------- M1 ----------------
    log("=" * 78)
    log("M1 误差分解： E||e||^2 = ||bias||^2 + Var(e)")
    log("  bias = 该用户平均误差向量（常数校准的理论可消除部分）")
    log("  resid= 误差围绕均值的波动（任何校准都改不掉）")
    log("  偏置占比 = ||bias||^2 / E||e||^2  ← 常数校准的理论上限")
    log("=" * 78)
    log(f"{'被试':>6} {'帧数':>6} {'EM(cm)':>8} {'RMS(cm)':>8} {'|bias|(cm)':>11} "
        f"{'残差RMS':>9} {'偏置占比':>9}")
    log("  EM  = 平均欧氏距离（本项目标准口径，与 cross_eval/summarize 一致）")
    log("  RMS = sqrt(E||e||^2)（分解恒等式用平方量，开方后即 RMS，恒 >= EM）")
    tot_sq = tot_b = tot_r = 0.0
    em_sum = 0.0
    per_sub = {}
    for s in sorted(set(subj)):
        m = subj == s
        es = e[m]
        b = es.mean(axis=0)
        sq = float((es ** 2).sum(axis=1).mean())
        bn = float((b ** 2).sum())
        resid = float(((es - b) ** 2).sum(axis=1).mean())
        em = float(np.hypot(es[:, 0], es[:, 1]).mean())
        per_sub[s] = dict(n=int(m.sum()), em=em, rms=float(np.sqrt(sq)), b=b,
                          bias=float(np.sqrt(bn)), resid=float(np.sqrt(resid)),
                          share=bn / max(sq, 1e-12))
        tot_sq += sq * m.sum()
        tot_b += bn * m.sum()
        tot_r += resid * m.sum()
        em_sum += em * m.sum()
        log(f"{s:>6} {int(m.sum()):>6} {em:>8.3f} {np.sqrt(sq):>8.3f} "
            f"{np.sqrt(bn):>11.3f} {np.sqrt(resid):>9.3f} "
            f"{bn / max(sq, 1e-12) * 100:>8.1f}%")
    n_all = int(sum(v["n"] for v in per_sub.values()))
    log(f"{'总体':>6} {n_all:>6} {em_sum / n_all:>8.3f} "
        f"{np.sqrt(tot_sq / n_all):>8.3f} {np.sqrt(tot_b / n_all):>11.3f} "
        f"{np.sqrt(tot_r / n_all):>9.3f} {tot_b / max(tot_sq, 1e-12) * 100:>8.1f}%")
    log("")
    log(f"  ⇒ 常数校准的理论天花板（能量口径）: "
        f"{tot_b / max(tot_sq, 1e-12) * 100:.1f}% 的误差能量")
    log(f"     RMS 从 {np.sqrt(tot_sq / n_all):.3f} -> {np.sqrt(tot_r / n_all):.3f} cm"
        f"（缺口 {(1 - np.sqrt(tot_r / n_all) / np.sqrt(tot_sq / n_all)) * 100:.1f}%）")
    log("     注意：这是「若 bias 被完美估计」的上限，实际取不到；"
        "且只在每人单独修正时才可用")

    # ---------------- M2(a) 跨姿势 ----------------
    log("")
    log("=" * 78)
    log("M2a 偏置稳定性（跨姿势）：同一用户在不同姿势下的 bias 是否一致")
    log("  一致性 = 1 - Var_姿势(b) / E||e||^2 ；越高说明「一个常数」越站得住")
    log("=" * 78)
    acts = sorted({a for a in act if a})
    if acts:
        log(f"{'被试':>6} {'姿势数':>7} " + " ".join(f"{a[:8]:>9}" for a in acts)
            + f" {'跨姿势一致性':>13}")
        cons = []
        for s in sorted(set(subj)):
            bs = []
            cells = []
            for a in acts:
                m = (subj == s) & (act == a)
                if m.sum() < 20:
                    cells.append("     -   ")
                    continue
                b = e[m].mean(axis=0)
                bs.append(b)
                cells.append(f"({b[0]:+.2f},{b[1]:+.2f})")
            if len(bs) < 2:
                continue
            bs = np.array(bs)
            var_b = float(bs.var(axis=0).sum())              # 姿势间偏置方差
            es = e[subj == s]
            sq = float((es ** 2).sum(axis=1).mean())
            # 零假设基线：bias 跨姿势恒定，各姿势均值只是抽样噪声 sigma^2/n_a
            ns = [int(((subj == s) & (act == a)).sum()) for a in acts
                  if ((subj == s) & (act == a)).sum() >= 20]
            sig2 = float(((es - es.mean(axis=0)) ** 2).sum(axis=1).mean())
            K = len(ns)
            null_var = np.mean([sig2 / n for n in ns]) * (K - 1) / K
            c = 1.0 - var_b / max(sq, 1e-12)
            c_null = 1.0 - null_var / max(sq, 1e-12)
            cons.append((c, c_null))
            log(f"{s:>6} {len(bs):>7} " + " ".join(f"{c_:>9}" for c_ in cells)
                + f" {c:>12.3f}  (零假设 {c_null:.3f})")
        if cons:
            c_m = np.mean([a for a, _ in cons]); cn_m = np.mean([b_ for _, b_ in cons])
            log(f"  跨姿势一致性 均值 {c_m:.3f}；零假设基线均值 {cn_m:.3f}")
            log(f"  两者之差 {cn_m - c_m:.3f} = 纯由「姿势相关偏置」造成的误差能量占比"
                f"（≈0 说明偏置跨姿势恒定）")

    # ---------------- M2(b) 跨时段 ----------------
    log("")
    log("=" * 78)
    log("M2b 偏置稳定性（跨时段）：同一用户前后半段（按帧序）的 bias 是否一致")
    log("=" * 78)
    log(f"{'被试':>6} {'前半bias':>16} {'后半bias':>16} {'差值(cm)':>9} {'|bias|(cm)':>10}")
    diffs, mags, nulls = [], [], []
    for s in sorted(set(subj)):
        m = subj == s
        if m.sum() < 40:
            continue
        es = e[m]
        o = np.argsort(order[m])
        half = len(o) // 2
        n1, n2 = half, len(o) - half
        b1 = es[o[:half]].mean(axis=0)
        b2 = es[o[half:]].mean(axis=0)
        d = float(np.linalg.norm(b1 - b2))
        mag = float(np.linalg.norm((b1 + b2) / 2))
        # 零假设：bias 恒定，b1-b2 ~ N(0, sig2(1/n1+1/n2) I)；2D 下 E||X||=s*sqrt(pi/2)
        sig2 = float(((es - es.mean(axis=0)) ** 2).sum(axis=1).mean())
        nulls.append(float(np.sqrt(sig2 * (1.0 / n1 + 1.0 / n2)) * np.sqrt(np.pi / 2)))
        diffs.append(d)
        mags.append(mag)
        log(f"{s:>6} ({b1[0]:+.2f},{b1[1]:+.2f})   ({b2[0]:+.2f},{b2[1]:+.2f}) "
            f"{d:>9.3f} {mag:>10.3f}")
    if diffs:
        log(f"  前后半段偏置差 均值 {np.mean(diffs):.3f} cm；"
            f"平均偏置幅度 {np.mean(mags):.3f} cm "
            f"-> 比值 {np.mean(diffs) / max(np.mean(mags), 1e-9):.2f}")
        log(f"  零假设基线（bias 恒定，差异纯属抽样噪声）: "
            f"{np.mean(nulls):.3f} cm -> 比值 {np.mean(nulls) / max(np.mean(mags), 1e-9):.2f}")
        log("  观察值 ≈ 零假设 => 偏置在用户内部稳定；观察值 >> 零假设 => "
            "「每人一个常数」不成立")

    log("")
    log(f"日志: {log_file}")


if __name__ == "__main__":
    main()
