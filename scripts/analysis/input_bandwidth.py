"""输入信息预算审计：当前输入表示还能承载多细的注视精度？

为什么需要它（与已有分析的分工）
------------------------------
项目已有的分析回答了"误差住在哪一层"：
  - bias_decomposition：E||e||² = 每人常数偏置(20%) + 单元内(80%)
  - error_budget：全局 / 人 / 姿势 / 单元内 四段分解，以及时域、屏幕空间、分量结构
但它们**都建立在同一个输入表示之上**。有一个更前置的问题一直没被测量：

    这个输入表示本身还剩多少注视信息？

若信号在输入侧就已经被降采样到像素网格以下，那么改架构、改损失、加预训练
都动不了天花板 —— 这属于**固有缺陷**，且可以在不训练任何模型的前提下被测量。

本脚本只量"输入端的物理信息预算"：
  1) 用 MediaPipe FaceLandmarker 在**原图**上定位眼区与虹膜（按各数据集自己的
     朝向约定，见 *_prepare_image）；
  2) 按管线真实的裁剪/缩放比例（bbox 边长 × CROP_MARGIN → resize 到 S），把
     原生像素尺度折算到模型输入端；
  3) 用解剖常数把像素换算成"每 1° 注视对应的虹膜位移像素"：
         mm_per_px  = 虹膜水平直径(mm) / 虹膜像素直径
         px_per_deg = 眼球半径 × (π/180) / mm_per_px
      代入 虹膜 12mm、眼球半径 11.8mm ⇒ px_per_deg = 0.01745 × 虹膜像素直径
  4) 报告 px_per_deg 的分布、DINOv2 patch token 的覆盖数、以及"要 1°/px 需要
     多大的输入分辨率"。

判读口径（严谨性声明，引用结论时必须一并给出）
  - px_per_deg 只刻画**虹膜中心位移**这一条线索的灵敏度，是可用信息的**下界**；
    眼睑形变、角膜反光、以及时间累积（RGBDGaze 约 30fps）都可能提供额外信息，
    本脚本不度量它们；
  - 因此结论的正确形式是"该输入表示在 S 分辨率下**至多**提供 X°/px"，
    **不是**"模型一定能达到该精度"；
  - 解剖常数取成人水平虹膜直径 12mm、眼球半径 11.8mm，个体差异约 ±10%，
    故所有绝对值只应按量级解读，横向比较（不同 S、不同数据集）才有意义。

用法：
    python -m scripts.analysis.input_bandwidth --dataset rgbdgaze --num-samples 400
    python -m scripts.analysis.input_bandwidth --dataset gazecapture --sizes 224,448,518
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


# ---- 路径引导：向上搜索项目根（与其它分析脚本一致）--------------
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

from configs.paths import experiment_dirs  # noqa: E402
from configs.rgbdgaze_config import CROP_MARGIN  # noqa: E402
from gazelab.datasets.table_index import TableIndex  # noqa: E402
from scripts.train import DATASET_SPECS  # noqa: E402

# ---- MediaPipe 468+10 点模型里的眼区索引（与已废弃的 extract_eye_crops 保持一致）----
LEFT_EYE_CORNERS = (33, 133)      # 外眼角 / 内眼角
RIGHT_EYE_CORNERS = (362, 263)
LEFT_EYE_APERTURE = (159, 145)    # 上眼睑 / 下眼睑
RIGHT_EYE_APERTURE = (386, 374)
LEFT_IRIS_CENTER, LEFT_IRIS_RING = 468, (469, 470, 471, 472)
RIGHT_IRIS_CENTER, RIGHT_IRIS_RING = 473, (474, 475, 476, 477)
N_LM_WITH_IRIS = 478

# ---- 解剖常数 ----
IRIS_DIAMETER_MM = 12.0     # 成人水平虹膜直径
EYEBALL_RADIUS_MM = 11.8    # 眼球半径；1° 转动 ⇒ 虹膜位移 ≈ R·(π/180) ≈ 0.206 mm
MM_PER_DEG = EYEBALL_RADIUS_MM * np.pi / 180.0
DINO_PATCH = 14             # DINOv2 ViT-S/14 的 patch 边长（像素）
#: px_per_deg = PX_PER_DEG_PER_IRIS_PX × 虹膜像素直径（由上面三个常数导出）
PX_PER_DEG_PER_IRIS_PX = MM_PER_DEG / IRIS_DIAMETER_MM
#: 要求 1° 至少移动 1 像素 ⇒ 虹膜直径至少要占这么多像素
IRIS_PX_FOR_ONE_DEG = 1.0 / PX_PER_DEG_PER_IRIS_PX


def q(a, p):
    """百分位（a 可为含 NaN 的数组）。"""
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.percentile(a, p))


def _prepare_image(dataset: str, img: np.ndarray) -> np.ndarray:
    """按数据集约定把原图转到与索引 bbox 同一坐标系。

    RGBDGaze：原始 RGB 为横向存储，作者 bbox 基于逆时针旋转 90° 的竖图
    （见 gazelab.datasets.rgbdgaze._make_face），故先 rot90。
    GazeCapture：无需旋转（坐标天然对齐，见 docs 的分析结论）。
    """
    if dataset == "rgbdgaze":
        return np.rot90(img, 1)
    return img


def load_landmarker():
    import mediapipe as mp
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import RunningMode
    from mediapipe.tasks.python.vision.face_landmarker import (
        FaceLandmarker, FaceLandmarkerOptions,
    )

    path = _PROJECT_ROOT / "assets" / "face_landmarker.task"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 MediaPipe 模型: {path}")
    opts = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(path)),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
    )
    return mp, FaceLandmarker.create_from_options(opts)


def detect_landmarks(landmarker, mp, img_bgr):
    """返回 (pts[K,2] 像素坐标, 图像 (h,w), 地标数 K)；无人脸返回 (None, hw, 0)。"""
    import cv2
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks:
        return None, (h, w), 0
    lms = res.face_landmarks[0]
    pts = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float64)
    return pts, (h, w), len(lms)


def eye_metrics(pts: np.ndarray, n_lm: int) -> dict:
    """眼区与虹膜的原生像素尺度。"""
    def dist(i, j):
        return float(np.linalg.norm(pts[i] - pts[j]))

    out = {
        "eye_w": 0.5 * (dist(*LEFT_EYE_CORNERS) + dist(*RIGHT_EYE_CORNERS)),
        "eye_h": 0.5 * (dist(*LEFT_EYE_APERTURE) + dist(*RIGHT_EYE_APERTURE)),
    }
    if n_lm >= N_LM_WITH_IRIS:
        def iris_diam(c, ring):
            return 2.0 * float(np.mean([dist(c, k) for k in ring]))
        out["iris_d"] = 0.5 * (iris_diam(LEFT_IRIS_CENTER, LEFT_IRIS_RING) +
                               iris_diam(RIGHT_IRIS_CENTER, RIGHT_IRIS_RING))
    else:
        out["iris_d"] = float("nan")
    return out


def sample_rows(index_csv: Path, wants_path: Path | None, num_samples: int,
                per_subject: int, rng: np.random.RandomState):
    """分层抽样：随机候选 + 每被试封顶，避免单个被试主导分布。"""
    table = TableIndex(index_csv)
    n = len(table)
    cand = rng.choice(n, size=min(n, max(num_samples * 6, 512)), replace=False)
    want = None
    if wants_path is not None and Path(wants_path).is_file():
        want = set(ln.strip() for ln in
                   Path(wants_path).read_text(encoding="utf-8").splitlines() if ln.strip())
    out, per_subj = [], {}
    for i in cand:
        row = table[int(i)]
        if want is not None and row.get("rgb_path", row.get("frame_path")) not in want:
            continue
        subj = row.get("subject", "?")
        if per_subj.get(subj, 0) >= per_subject:
            continue
        per_subj[subj] = per_subj.get(subj, 0) + 1
        out.append(row)
        if len(out) >= num_samples:
            break
    return out


def main():
    ap = argparse.ArgumentParser(
        description="输入信息预算审计（输入表示还能承载多细的注视精度）")
    ap.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="rgbdgaze")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--split", default=None,
                    help="限定某个划分的清单（默认整个索引，按被试分层抽样）")
    ap.add_argument("--num-samples", type=int, default=400)
    ap.add_argument("--per-subject", type=int, default=6,
                    help="每个被试最多取多少帧（防单被试主导分布）")
    ap.add_argument("--sizes", default="224,448,518",
                    help="要评估的输入分辨率（逗号分隔）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    spec = DATASET_SPECS[args.dataset]
    index_dir = Path(args.index_dir) if args.index_dir else Path(spec["default_index"])
    index_csv = index_dir / "index.csv"
    if not index_csv.is_file():
        raise FileNotFoundError(f"索引不存在: {index_csv}")

    log_dir = experiment_dirs(args.dataset, "analysis")["log"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{datetime.now():%Y%m%d_%H%M%S}_input_bandwidth_{args.dataset}.txt"

    def log(msg):
        print(msg, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("=" * 96)
    log(f"输入信息预算审计 | 数据集={args.dataset} 索引={index_csv}")
    log(f"裁剪边距倍数 CROP_MARGIN={CROP_MARGIN} | 评估分辨率 {sizes} | "
        f"DINOv2 patch={DINO_PATCH}")
    log(f"解剖常数: 虹膜直径 {IRIS_DIAMETER_MM}mm, 眼球半径 {EYEBALL_RADIUS_MM}mm "
        f"⇒ 1° 注视 ≈ {MM_PER_DEG:.3f} mm 虹膜位移")
    log("=" * 96)

    rng = np.random.RandomState(args.seed)
    wants = (index_dir / f"{args.split}.txt") if args.split else None
    rows = sample_rows(index_csv, wants, args.num_samples, args.per_subject, rng)
    log(f"抽样 {len(rows)} 帧（每被试最多 {args.per_subject} 帧）\n")

    import cv2
    mp, landmarker = load_landmarker()

    rec = []          # 每帧的原始记录
    n_no_face = n_no_iris = 0
    for k, row in enumerate(rows):
        path_col = spec["path_col"]
        p = Path(row[path_col])
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = _prepare_image(args.dataset, img)
        pts, (h, w), n_lm = detect_landmarks(landmarker, mp, img)
        if pts is None:
            n_no_face += 1
            continue
        m = eye_metrics(pts, n_lm)
        if not np.isfinite(m["iris_d"]):
            n_no_iris += 1
        bw, bh = float(row["bbox_w"]), float(row["bbox_h"])
        crop_px = max(bw, bh) * CROP_MARGIN          # 送入 resize 的正方形边长（原生 px）
        if crop_px <= 1:
            continue
        mm_per_px = IRIS_DIAMETER_MM / m["iris_d"] if np.isfinite(m["iris_d"]) else np.nan
        px_per_deg_native = MM_PER_DEG / mm_per_px if np.isfinite(mm_per_px) else np.nan
        rec.append(dict(
            subject=row.get("subject", "?"), frame=f"{w}x{h}", crop_px=crop_px,
            eye_w=m["eye_w"], eye_h=m["eye_h"], iris_d=m["iris_d"],
            face_frac=max(bw, bh) / min(w, h),      # bbox 边长 / 帧短边
            px_per_deg_native=px_per_deg_native,
            **{f"pxd_{S}": px_per_deg_native * S / crop_px for S in sizes},
            **{f"iris_{S}": m["iris_d"] * S / crop_px for S in sizes},
        ))
        if (k + 1) % 100 == 0:
            log(f"  ...{k + 1}/{len(rows)}")

    if not rec:
        log("没有可用样本，终止。")
        return
    log(f"\n有效 {len(rec)} 帧（无人脸 {n_no_face}，无虹膜地标 {n_no_iris}）"
        f"，被试 {len(set(r['subject'] for r in rec))} 人\n")

    # ---------- 1. 原生尺度 ----------
    log("=" * 96)
    log("1. 原生尺度（未做任何缩放）")
    log("=" * 96)
    log(f"  {'量':<26}{'中位':>10}{'p10':>10}{'p90':>10}   单位")
    for key, name, unit in [("crop_px", "裁剪边长(送入 resize)", "px"),
                            ("eye_w", "眼睑宽度(内外眼角)", "px"),
                            ("eye_h", "眼睑开度(上下眼睑)", "px"),
                            ("iris_d", "虹膜直径", "px"),
                            ("face_frac", "bbox 边长 / 帧短边", "-")]:
        v = [r[key] for r in rec]
        log(f"  {name:<26}{q(v, 50):>10.2f}{q(v, 10):>10.2f}{q(v, 90):>10.2f}   {unit}")
    log(f"\n  这些数字请与「要多少像素才够」对照看：解剖上虹膜直径约 "
        f"{IRIS_DIAMETER_MM}mm 且几乎所有成人相同，")
    log(f"  因此「虹膜占多少像素」直接决定了下游一切方法的分辨率上限。")

    # ---------- 2. 输入端的度数分辨率 ----------
    log("")
    log("=" * 96)
    log("2. 模型输入端：1° 注视 = 多少像素的虹膜位移")
    log("=" * 96)
    log(f"  {'输入分辨率':<14}{'px/deg 中位':>14}{'p10':>10}{'p90':>10}"
        f"{'虹膜像素中位':>14}{'虹膜占 patch':>14}{'<1px/deg 占比':>14}")
    for S in sizes:
        v = [r[f"pxd_{S}"] for r in rec]
        ir = [r[f"iris_{S}"] for r in rec]
        frac = float(np.mean([x < 1.0 for x in v if np.isfinite(x)]))
        log(f"  {S:<14}{q(v, 50):>14.3f}{q(v, 10):>10.3f}{q(v, 90):>10.3f}"
            f"{q(ir, 50):>14.2f}{q(ir, 50) / DINO_PATCH:>14.2f}{frac * 100:>13.1f}%")
    vn = [r["px_per_deg_native"] for r in rec]
    log(f"  {'原生(未缩放)':<14}{q(vn, 50):>14.3f}{q(vn, 10):>10.3f}{q(vn, 90):>10.3f}"
        f"{q([r['iris_d'] for r in rec], 50):>14.2f}"
        f"{q([r['iris_d'] for r in rec], 50) / DINO_PATCH:>14.2f}"
        f"{str(round(float(np.mean([x < 1.0 for x in vn if np.isfinite(x)])) * 100, 1)) + '%':>14}")
    log("  读法：px/deg < 1 表示「1° 的注视变化只能移动不到 1 个像素」，")
    log("        即该分辨率下虹膜位移信号已经落到网格以下（抗锯齿能泄露一点亚像素信息，")
    log("        但可提取量随采样密度单调下降 —— 这解释了为何 224→448 有效、448→518 无效）。")

    # ---------- 3. 典型误差对应的像素 ----------
    log("")
    log("=" * 96)
    log("3. 把已测得的误差换算成像素：模型离网格有多近")
    log("=" * 96)
    log(f"  {'角度误差':<12}" + "".join(f"{str(S) + ' 下':>14}" for S in sizes))
    for deg in (0.5, 1.0, 1.91, 3.14, 5.0):
        line = f"  {str(deg) + '°':<12}"
        for S in sizes:
            line += f"{q([r[f'pxd_{S}'] for r in rec], 50) * deg:>14.2f}"
        log(line)
    log("  注：3.14° / 1.91° 分别是本项目 G6 实测的平均角度误差与论文口径阈值。")

    # ---------- 4. 达到 1°/px 需要多大输入 ----------
    log("")
    log("=" * 96)
    log("4. 若要求 1° 至少移动 1 个像素，输入分辨率需要多大")
    log("=" * 96)
    need = [r["crop_px"] / r["px_per_deg_native"] for r in rec
            if np.isfinite(r["px_per_deg_native"]) and r["px_per_deg_native"] > 1e-9]
    log(f"  需要的输入边长（正方形输入）: 中位 {q(need, 50):.0f} px "
        f"| p10 {q(need, 10):.0f} | p90 {q(need, 90):.0f}")
    log(f"  等价条件：虹膜在输入端至少占 {IRIS_PX_FOR_ONE_DEG:.0f} px 直径"
        "（由 px/deg = 0.01745 x 虹膜像素 反解）")
    log(f"  当前 448 输入下虹膜中位仅占 {q([r['iris_448'] for r in rec], 50):.2f} px"
        f"；原生也只有 {q([r['iris_d'] for r in rec], 50):.2f} px。")

    # ---------- 5. 被试间差异 ----------
    log("")
    log("=" * 96)
    log("5. 被试间差异（同一输入分辨率下，不同人的可用分辨率差多少）")
    log("=" * 96)
    S0 = sizes[0]
    by_subj = {}
    for r in rec:
        by_subj.setdefault(r["subject"], []).append(r[f"pxd_{S0}"])
    rows5 = sorted(((s, q(v, 50), len(v)) for s, v in by_subj.items()),
                   key=lambda x: x[1])
    log(f"  {'被试':<10}{'px/deg @' + str(S0):>16}{'帧数':>8}")
    for s, m, c in rows5[:8] + (rows5[-8:] if len(rows5) > 16 else []):
        log(f"  {str(s):<10}{m:>16.3f}{c:>8}")
    if len(rows5) > 1:
        log(f"\n  被试间极差 {rows5[-1][1] / rows5[0][1]:.2f}x —— 同一模型在不同人身上"
            "的可用信息量本身就不一样，")

    log("")
    log(f"日志: {log_file}")


if __name__ == "__main__":
    main()
