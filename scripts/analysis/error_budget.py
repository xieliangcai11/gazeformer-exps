"""误差预算分解：把测试集误差按"可干预的层级"拆开，定位缺陷在哪里。

动机：不做"先定故事再找证据"。误差必须被测量出结构，才知道哪一层值得投入。
上一轮已测出（scripts/analysis/bias_decomposition.py）：
        E||e||^2 = ||每人常数偏置||^2 + Var(e)  ->  20% / 80%
但"20% 是常数偏置"只回答了"常数校准的天花板"，**没有回答那 80% 是什么**。
本脚本继续往下拆。

四个测量
--------
A. 嵌套方差分解（序贯，可精确配平）
       R0 = E||e||^2                            （不做任何修正）
       R1 = E||e - 全局均值||^2                  （去掉全局系统偏置）
       R2 = E||e - 被试均值||^2                  （再去掉"每人一个常数"）
       R3 = E||e - (被试,姿势)均值||^2            （再去掉姿势相关部分）
       增量 R0-R1 / R1-R2 / R2-R3 / R3 分别 = 全局 / 人 / 姿势 / 单元内
       四者之和严格等于 R0。回答："误差能量住在哪一层"。

B. 时域结构（描述性，非 oracle）
       误差能量在时间尺度上的分布：对每个被试按帧序排好误差序列，
       看"相对于窗口 W 的局部均值，误差能量的低频占比"。
       低频占比高 => 误差是慢漂移；低 => 是逐帧白噪声。
       注意：这里量的是**误差自身的频谱结构**，不是"可实现的方法收益"。

C. 屏幕空间结构（描述性）
       按注视点所在的屏幕网格分桶，看误差是否在某些屏幕区域系统性更大。
       若有，说明缺陷与"屏幕位置"有关（可能是标签空间/几何建模问题）。

D. 分量不对称
       x / y 方向误差，同时给 cm 与"归一化（除以屏幕物理尺寸）"两种口径。

用法：
    python -m scripts.analysis.error_budget \
        --ckpt out/rgbdgaze/ablation/checkpoints/g6_nodepth_448_aug/best.pt \
        --no-geom --resolution 448
"""

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---- 路径引导：向上搜索项目根 ----
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

from configs.paths import DEVICE as DEFAULT_DEVICE, experiment_dirs  # noqa: E402
from scripts.train import DATASET_SPECS, Subset, predict_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="误差预算分解（A 嵌套方差 / B 时域 / C 空间 / D 分量）")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="rgbdgaze")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fit-split", default="val",
                    help="拟合 F 段线性修正所用的划分（不碰测试集）")
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
    log_file = log_dir / f"{datetime.now():%Y%m%d_%H%M%S}_error_budget.txt"
    lines = []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("=" * 78)
    log(f"误差预算分解 | ckpt={args.ckpt}")
    log(f"数据集={args.dataset} 划分={args.split} 分辨率={args.resolution} use_geom={args.use_geom}")
    log("=" * 78)

    # ---------- 前向一遍，拿逐帧预测 ----------
    from gazelab.models.dinov2_gaze import RGBDGazeDINOv2, load_state_dict_any_res
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # ---- 守卫（2026-10-15 审计 B6）----
    # 本脚本只构造"无眼支流"模型。若 ckpt 带 proj_eye.*，继续跑会得到一个**静默
    # 与主评估分叉**的数字（曾让分析与主表不一致）。这里直接失败并给出可行路径。
    if any(".proj_eye." in k or k.startswith("proj_eye.") for k in state):
        raise SystemExit(
            f"{args.ckpt} 带眼支流权重（proj_eye.*），而本脚本按无眼支流建模 ⇒ "
            f"结果会与主评估口径分叉。请改用 dump_predictions.py（支持 --use-eye/"
            f"--eye-scale/--eye-shuffle）导出逐帧预测 CSV，再做后续分析。")
    model = RGBDGazeDINOv2(img_size=args.resolution, use_geom=args.use_geom).to(device)
    # any_res：--ckpt 与 --resolution 独立指定，权重可能来自另一种分辨率
    miss, unexp, resampled = load_state_dict_any_res(model, state)
    if resampled is not None:
        cur = tuple(model.state_dict()["rgb.backbone.pos_embed"].shape)
        log(f"[load] pos_embed {resampled} -> {cur}（已插值）")
    log(f"[load] 缺失键 {list(miss) or '无'} | 多余键 {list(unexp) or '无'}")
    model.eval()

    spec = DATASET_SPECS[args.dataset]
    index_csv = Path(args.index_dir) / "index.csv"
    paths = [ln.strip() for ln in
             (Path(args.index_dir) / f"{args.split}.txt").read_text(
                 encoding="utf-8").splitlines() if ln.strip()]
    idx_table = spec["cls"](index_csv, image_size=args.resolution,
                            **spec["kwargs"]).rows
    p2i = idx_table.lookup(spec["path_col"])
    idx = [p2i[p] for p in paths]
    ds = spec["cls"](index_csv, image_size=args.resolution, **spec["kwargs"])
    dl = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers)
    fr = predict_frames(model, dl, device)

    e = np.stack([fr["cm_x"], fr["cm_y"]], axis=1)                 # [N,2] 厘米
    label = fr["label"]                                            # [N,2] 归一化屏幕坐标
    rows = [idx_table[i] for i in idx]
    subj = np.array([r["subject"] for r in rows])
    act = np.array([r.get("activity", "") or r.get("orientation", "") for r in rows])
    # 帧序：优先用 frame_index 列；RGBDGaze 的索引没有该列，从文件名解析。
    # 之前直接 .get("frame_index", 0) 会让所有帧的"时间"都等于 0，时序分析失效。
    def _frame_no(r):
        if "frame_index" in r:
            return int(float(r["frame_index"]))
        stem = Path(r[spec["path_col"]]).stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        return int(digits) if digits else 0

    fidx = np.array([_frame_no(r) for r in rows])
    scr = np.stack([[float(r["screen_w"]) * float(r["cm_px_x"]),
                     float(r["screen_h"]) * float(r["cm_px_y"])] for r in rows])  # [N,2] cm
    log(f"样本 {len(idx)} 帧 / {len(set(subj))} 名被试 / 姿势 {sorted(set(act))}")
    log("")

    # ================= A. 嵌套方差分解 =================
    log("=" * 78)
    log("A. 嵌套方差分解（序贯；四段增量之和严格等于总量）")
    log("   R0 不做任何修正 -> R1 去全局均值 -> R2 去被试均值 -> R3 去(被试,姿势)均值")
    log("   四段增量：全局偏置 / 人的差异 / 姿势差异 / 单元内（逐帧）")
    log("=" * 78)
    R0 = float((e ** 2).sum(axis=1).mean())

    def residual_after(keys):
        """用 keys 分组的组内均值作预测，返回残差平方均值。"""
        resid = np.zeros(len(e))
        groups = defaultdict(list)
        for i, k in enumerate(keys):
            groups[k].append(i)
        for k, ix in groups.items():
            ix = np.asarray(ix)
            resid[ix] = ((e[ix] - e[ix].mean(axis=0)) ** 2).sum(axis=1)
        return float(resid.mean())

    R1 = residual_after(np.zeros(len(e), dtype=int))
    R2 = residual_after(subj)
    R3 = residual_after([f"{s}|{a}" for s, a in zip(subj, act)])
    segs = [("全局系统偏置", R0 - R1), ("人的差异（每人一个常数）", R1 - R2),
            ("姿势差异（同一人内部）", R2 - R3), ("单元内（逐帧波动）", R3)]
    log(f"  总量 R0 = {R0:.4f} cm^2   (即 RMS {np.sqrt(R0):.3f} cm)")
    for name, v in segs:
        log(f"    {name:24s} {v:9.4f} cm^2   占 {v / R0 * 100:5.1f}%")
    log(f"    {'配平检查(应=100%)':24s} {sum(v for _, v in segs) / R0 * 100:5.1f}%")
    log("")
    log("  判读：单元内那一项 = 既不能靠全局修正、也不能靠'每人/每姿势一个常数'")
    log("        消除的部分。它若占绝大多数，则缺陷在'逐帧预测本身'，")
    log("        而不在任何形式的偏移校准。")

    # ================= D. 分量不对称（放在 A 之后，便于对照） =================
    log("")
    log("=" * 78)
    log("D. x / y 分量不对称")
    log("=" * 78)
    ex_cm = np.abs(e[:, 0]).mean()
    ey_cm = np.abs(e[:, 1]).mean()
    ex_n = np.abs(e[:, 0] / scr[:, 0]).mean()
    ey_n = np.abs(e[:, 1] / scr[:, 1]).mean()
    log(f"  厘米口径   : x {ex_cm:.3f} cm | y {ey_cm:.3f} cm | y/x = {ey_cm / ex_cm:.2f}")
    log(f"  归一化口径 : x {ex_n:.4f}   | y {ey_n:.4f}   | y/x = {ey_n / ex_n:.2f}")
    log(f"  屏幕物理尺寸(screen_cm) 中位: {np.median(scr[:, 0]):.2f} x {np.median(scr[:, 1]):.2f} cm")
    log("  判读：两种口径给出相反的结论时，说明「哪一维更差」取决于用哪种误差度量——")
    log("        写论文必须明确报告口径。")

    # ================= B. 时域结构 =================
    log("")
    log("=" * 78)
    log("B. 时域结构（描述性）：误差能量在时间尺度上的分布")
    log("   对每个 (被试,姿势) 段，按 frame_index 排序，计算")
    log("   lowfreq(W) = 1 - E||e - MA_W(e)||^2 / E||e||^2")
    log("   W 是以帧为单位的局部均值窗口（RGBDGaze 约 30fps，W=30 ≈ 1 秒）")
    log("   ⚠ 这是对误差序列自身频谱结构的**描述**，不是可实现的方法收益")
    log("=" * 78)
    tasks = defaultdict(list)
    for i, (s, a) in enumerate(zip(subj, act)):
        tasks[(s, a)].append(i)
    Ws = [5, 15, 30, 60, 120]
    fracs = {W: [] for W in Ws}
    n_used = 0
    for (s, a), ix in sorted(tasks.items()):
        ix = np.asarray(ix)[np.argsort(fidx[ix])]
        if len(ix) < 200:
            continue
        n_used += len(ix)
        es = e[ix]
        for W in Ws:
            if len(es) <= 2 * W:
                continue
            k = np.ones(W) / W
            # mode="same" 得到等长输出；两端各裁 W//2 避开边界效应。
            # （之前用 mode="valid" 但按"裁 2(W-1)"对齐，长度不匹配会直接报错）
            ma = np.stack([np.convolve(es[:, 0], k, mode="same"),
                           np.convolve(es[:, 1], k, mode="same")], axis=1)
            h = W // 2
            a, b = h, len(es) - h
            tot = float((es[a:b] ** 2).sum(axis=1).mean())
            r = es[a:b] - ma[a:b]
            fracs[W].append(1.0 - float((r ** 2).sum(axis=1).mean()) / tot)
    log(f"  参与统计: {n_used} 帧（每个 (被试,姿势) 段 >=200 帧）")
    base_em_b = float(np.hypot(e[:, 0], e[:, 1]).mean())
    log(f"  {'窗口 W':>8} {'≈时长':>8} {'低频能量占比':>14} "
        f"{'oracle 后平均误差':>18}")
    log(f"  （oracle = 减去该窗口的误差局部均值，需要知道真值，**不可实现**；"
        f"当前平均误差 {base_em_b:.3f} cm）")
    for W in Ws:
        if not fracs[W]:
            continue
        # oracle：减去 MA_W(误差) 后的平均误差（逐段计算再汇总）
        ems = []
        for (s_, a_), ixx in sorted(tasks.items()):
            ixx = np.asarray(ixx)[np.argsort(fidx[ixx])]
            if len(ixx) <= 2 * W:
                continue
            es = e[ixx]
            k = np.ones(W) / W
            ma = np.stack([np.convolve(es[:, 0], k, mode="same"),
                           np.convolve(es[:, 1], k, mode="same")], axis=1)
            h = W // 2
            r = es[h:len(es) - h] - ma[h:len(es) - h]
            ems.append(np.hypot(r[:, 0], r[:, 1]).mean())
        log(f"  {W:>8} {W / 30:>7.1f}s {np.mean(fracs[W]) * 100:>13.1f}% "
            f"{np.mean(ems):>17.3f} cm")
    log("  ⚠ oracle 上界：任何「随时间自适应」的方法**最多**能到这一列的数字，")
    log("    且实际方法需要测试时的真值反馈（校准点/用户交互）才可能逼近它。")
    log("  判读：若 W=30 时占比就已很小（<20%），说明误差主体是逐帧白噪声，")
    log("        任何「随时间自适应」的方法都无从下手；占比高则说明存在慢漂移。")

    # ================= E. 与注视点运动的关系（描述性） =================
    log("")
    log("=" * 78)
    log("E. 误差与注视点运动的关系（描述性）")
    log("   点的位置是连续移动的；若误差与'移动速度'相关，说明缺陷与运动/追随有关")
    log("=" * 78)
    mag_e, mag_v, cos_ev = [], [], []
    for (s_, a_), ix in sorted(tasks.items()):
        ix = np.asarray(ix)[np.argsort(fidx[ix])]
        if len(ix) < 200:
            continue
        l = label[ix]
        # 归一化标签差 -> 物理厘米（按该段的屏幕尺寸）
        scm = scr[ix]
        v = np.diff(l, axis=0) * scm[:-1]
        ev = e[ix][:-1]
        me = np.linalg.norm(ev, axis=1)
        mv = np.linalg.norm(v, axis=1)
        ok = mv > 1e-6
        if ok.sum() < 50:
            continue
        mag_e.append(me[ok]); mag_v.append(mv[ok])
        c = (ev[ok] * v[ok]).sum(axis=1) / (me[ok] * mv[ok] + 1e-9)
        cos_ev.append(c)
    if mag_e:
        me = np.concatenate(mag_e); mv = np.concatenate(mag_v); cs = np.concatenate(cos_ev)
        log(f"  帧间位移 中位 {np.median(mv):.3f} cm，均值 {mv.mean():.3f} cm")
        log(f"  corr(|误差|, |帧间位移|) = {np.corrcoef(me, mv)[0,1]:+.3f}")
        log(f"  误差方向与位移方向的平均余弦 = {cs.mean():+.3f}"
            f"（+1=同向/超前，-1=反向/滞后，0=无关）")
        # 按位移大小分桶看误差
        qs = np.quantile(mv, [0.25, 0.5, 0.75])
        edges = [-np.inf, *qs, np.inf]
        log("  按帧间位移分桶的平均误差：")
        for i in range(4):
            m = (mv > edges[i]) & (mv <= edges[i + 1])
            if m.sum() >= 50:
                log(f"    位移 ∈ [{edges[i] if np.isfinite(edges[i]) else 0:.3f},"
                    f"{edges[i+1] if np.isfinite(edges[i+1]) else mv.max():.3f}] cm"
                    f"  n={int(m.sum()):5d}  平均误差 {me[m].mean():.3f} cm")
    log("  ⚠ 描述性统计：只说明误差与运动的关联，不构成因果，也不代表可实现收益")

    # ================= C. 屏幕空间结构 =================
    log("")
    log("=" * 78)
    log("C. 屏幕空间结构（描述性）：误差是否在某些屏幕区域系统性更大")
    log("=" * 78)
    nb = 4
    bins = np.clip((label * nb).astype(int), 0, nb - 1)
    log(f"  {nb}x{nb} 网格，格内 EM(cm) 与帧数：")
    grid = np.full((nb, nb), np.nan)
    cnt = np.zeros((nb, nb), int)
    for bx in range(nb):
        row = []
        for by in range(nb):
            m = (bins[:, 0] == bx) & (bins[:, 1] == by)
            cnt[bx, by] = int(m.sum())
            if cnt[bx, by] >= 20:
                grid[bx, by] = float(np.hypot(e[m, 0], e[m, 1]).mean())
            row.append(f"{grid[bx, by]:5.2f}" if not np.isnan(grid[bx, by]) else "  -  ")
        log(f"    x∈[{bx/nb:.2f},{(bx+1)/nb:.2f})  " + " ".join(row))
    valid = ~np.isnan(grid)
    if valid.sum() >= 4:
        log(f"  格间极差 {np.nanmax(grid) - np.nanmin(grid):.3f} cm，"
            f"最差格 {np.nanmax(grid):.3f} vs 最佳格 {np.nanmin(grid):.3f}")
        log(f"  格间标准差 {np.nanstd(grid):.3f} cm；全体均值 {np.hypot(e[:,0], e[:,1]).mean():.3f} cm")
    # 误差与屏幕位置的线性相关
    for ax, nm in ((0, "x"), (1, "y")):
        c = np.corrcoef(label[:, ax], e[:, ax])[0, 1]
        log(f"  corr(label_{nm}, error_{nm}) = {c:+.3f}")

    # ================= G. 关键混淆检验：时域结构 vs 空间结构 =================
    log("")
    log("=" * 78)
    log("G. 混淆检验：B 段测到的「低频能量」是「时间漂移」，还是「空间结构+目标平滑运动」的假象？")
    log("   做法：先用屏幕位置(4x4 网格的全局均值)解释误差，再看残余的低频占比。")
    log("   若剩余低频占比骤降到接近白噪声水平，则 B 段的结论是假象。")
    log("=" * 78)
    # 全局空间模型：4x4 网格均值（描述性分解，不是方法）
    spat = np.zeros_like(e)
    for bx in range(nb):
        for by in range(nb):
            m = (bins[:, 0] == bx) & (bins[:, 1] == by)
            if m.sum():
                spat[m] = e[m].mean(axis=0)
    R0b = float((e ** 2).sum(axis=1).mean())
    Rsp = float(((e - spat) ** 2).sum(axis=1).mean())
    log(f"  空间(4x4 网格均值)可解释的误差能量: {(R0b - Rsp) / R0b * 100:.1f}%"
        f"  (RMS {np.sqrt(R0b):.3f} -> {np.sqrt(Rsp):.3f} cm)")
    e_res = e - spat                      # 去掉空间结构后的残差误差
    log("")
    log(f"  {'窗口 W':>8} {'原低频占比':>12} {'去空间后':>12} {'白噪声基线':>12}")
    for W in Ws:
        row = [W]
        for arr in (e, e_res):
            fs = []
            for (s_, a_), ixx in sorted(tasks.items()):
                ixx = np.asarray(ixx)[np.argsort(fidx[ixx])]
                if len(ixx) <= 2 * W:
                    continue
                es = arr[ixx]
                k = np.ones(W) / W
                ma = np.stack([np.convolve(es[:, 0], k, mode="same"),
                               np.convolve(es[:, 1], k, mode="same")], axis=1)
                h = W // 2
                a_, b_ = h, len(es) - h
                tot = float((es[a_:b_] ** 2).sum(axis=1).mean())
                fs.append(1.0 - float(((es[a_:b_] - ma[a_:b_]) ** 2).sum(axis=1).mean())
                          / max(tot, 1e-12))
            row.append(np.mean(fs) * 100 if fs else np.nan)
        row.append(100.0 / W)             # 白噪声下"局部均值解释掉 1/W"的基线
        log(f"  {row[0]:>8} {row[1]:>11.1f}% {row[2]:>11.1f}% {row[3]:>11.1f}%")
    log("  判读：若「去空间后」仍远高于「白噪声基线」，则时域慢漂移是真实的；")
    log("        若降到接近基线，则 B 段结论作废。")

    # ================= F. 预测范围压缩 与 gain/offset 修正 =================
    log("")
    log("=" * 78)
    log("F. 预测范围压缩（regression-to-mean）与线性修正的**合法**收益")
    log("   MSE 回归的固有性质：预测方差小于目标方差。")
    log("   若成立，则一个全局(每轴 y = a + b*x)修正即可吃掉这部分误差，")
    log("   且修正参数可**在不看测试集的前提下**从 val 划分拟合（非 oracle）。")
    log("=" * 78)
    sx_p, sx_l = float(fr["pred"][:, 0].std()), float(fr["label"][:, 0].std())
    sy_p, sy_l = float(fr["pred"][:, 1].std()), float(fr["label"][:, 1].std())
    log(f"  测试集 预测/目标 标准差： x {sx_p:.4f}/{sx_l:.4f} = {sx_p / sx_l:.3f}"
        f" | y {sy_p:.4f}/{sy_l:.4f} = {sy_p / sy_l:.3f}")
    log("  （比值 <1 即为压缩；=1 表示无系统性范围偏差）")

    def _err(pred, lab, scm):
        d = (pred - lab) * scm
        return float(np.hypot(d[:, 0], d[:, 1]).mean())

    base_em = _err(fr["pred"], fr["label"], scr)
    # oracle：在测试集自己上拟合
    A_or = np.polyfit(fr["pred"][:, 0], fr["label"][:, 0], 1)
    B_or = np.polyfit(fr["pred"][:, 1], fr["label"][:, 1], 1)
    pred_or = fr["pred"].copy()
    pred_or[:, 0] = A_or[0] * fr["pred"][:, 0] + A_or[1]
    pred_or[:, 1] = B_or[0] * fr["pred"][:, 1] + B_or[1]
    log(f"  [oracle] 测试集自拟合: x↦{A_or[0]:.3f}x{A_or[1]:+.3f}, "
        f"y↦{B_or[0]:.3f}y{B_or[1]:+.3f} -> EM {_err(pred_or, fr['label'], scr):.3f} cm"
        f"（原 {base_em:.3f}，{(1 - _err(pred_or, fr['label'], scr) / base_em) * 100:+.1f}%）")

    if args.fit_split and args.fit_split != args.split:
        fpaths = [ln.strip() for ln in
                  (Path(args.index_dir) / f"{args.fit_split}.txt").read_text(
                      encoding="utf-8").splitlines() if ln.strip()]
        fidx_l = [p2i[q] for q in fpaths]
        fds = spec["cls"](index_csv, image_size=args.resolution, **spec["kwargs"])
        fdl = DataLoader(Subset(fds, fidx_l), batch_size=args.batch_size,
                         shuffle=False, num_workers=args.num_workers)
        ffr = predict_frames(model, fdl, device)
        A_f = np.polyfit(ffr["pred"][:, 0], ffr["label"][:, 0], 1)
        B_f = np.polyfit(ffr["pred"][:, 1], ffr["label"][:, 1], 1)
        pred_f = fr["pred"].copy()
        pred_f[:, 0] = A_f[0] * fr["pred"][:, 0] + A_f[1]
        pred_f[:, 1] = B_f[0] * fr["pred"][:, 1] + B_f[1]
        em_f = _err(pred_f, fr["label"], scr)
        log(f"  [合法] 在 {args.fit_split} 上拟合: x↦{A_f[0]:.3f}x{A_f[1]:+.3f}, "
            f"y↦{B_f[0]:.3f}y{B_f[1]:+.3f}")
        log(f"         应用到 {args.split} -> EM {em_f:.3f} cm"
            f"（原 {base_em:.3f}，{(1 - em_f / base_em) * 100:+.1f}%）")
        log("  ⚠ 这只是一个「修正预测范围」的后处理；若收益可观，说明误差里有一块")
        log("    与「模型输出尺度」有关的系统项，与校准数据无关，属于可免费拿掉的量。")

    log("")
    log(f"日志: {log_file}")


if __name__ == "__main__":
    main()
