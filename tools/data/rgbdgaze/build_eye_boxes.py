"""从 eye_crops.jsonl 生成紧凑眼框表（**统一到数据集裁剪所用的旋转坐标系**），并验证坐标系。

背景（已用可视化定案：out/rgbdgaze/eye_frame_check.png，勿再改方向）
------------------------------------------------------------------
  - eye_crops.jsonl 的框来自 MediaPipe，坐标系 = **原始存储图**（1440×1080，W=1080）；
  - 数据集裁人脸前会 np.rot90(img, 1)，作者 bbox 也在**旋转后**坐标系；
  - 映射（np.rot90(a,1): b[i,j]=a[j, W-1-i]）⇒ 原图点 (x,y) → 旋转后 (x'=y, y'=W-1-x)；
    对框 (x,y,w,h) ⇒ (x'=y, y'=W-1-(x+w), w'=h, h'=w)。
  - 实测：映射后 face_mp 与索引 bbox 的 IoU 中位 0.53~0.75（不映射仅 0.00~0.03），
    且映射后的眼框裁剪就是清晰的眼睛 ⇒ 映射正确。

配对与缺失行
-----------
  jsonl 只有 142,661 有效行（8 行 MediaPipe 检测失败），idx 因此**不连续**。
  早期脚本按"位置"配对，失败行之后全部错位，导致过一份荒谬的标定表 —— 故本工具
  **一律按 idx 精确配对**；缺失行用**同被试的中位数框**兜底，并记录 filled 掩码备查。

产出：data/RGBDGaze/index/eye_boxes.npz
      idx    [N]   int64   0..N-1（对应 index.csv 行号）
      boxes  [N,8] float32 左眼(x,y,w,h) + 右眼(x,y,w,h)，**旋转后坐标系**（w,h 已互换）
      filled [N]   bool    该行是否为兜底值

用法：
    python -m tools.data.rgbdgaze.build_eye_boxes --check-n 60
"""

from __future__ import annotations

import argparse
import csv
import json
import random
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

JSONL = _PROJECT_ROOT / "out" / "rgbdgaze" / "eye_crops" / "eye_crops.jsonl"
OUT_NPZ = RGBDGaze_INDEX_DIR / "eye_boxes.npz"


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description="生成旋转坐标系下的眼框表并验证")
    ap.add_argument("--jsonl", default=str(JSONL))
    ap.add_argument("--out", default=str(OUT_NPZ))
    ap.add_argument("--check-n", type=int, default=60)
    ap.add_argument("--max-center-dist", type=float, default=0.10,
                    help="索引bbox反解中心与 MediaPipe 人脸框中心的距离上限（按脸边长归一）")
    ap.add_argument("--size-check-n", type=int, default=4,
                    help="抽查多少张原图确认尺寸恒定（映射依赖常数 W）")
    args = ap.parse_args()

    # ---------- 0) 读 jsonl（按 idx 建表）+ 读索引 ----------
    by_idx = {}
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("ok") and d.get("left_eye") and d.get("right_eye") and d.get("face_mp"):
                by_idx[int(d["idx"])] = (d["left_eye"], d["right_eye"], d["face_mp"])
    if not by_idx:
        raise SystemExit("jsonl 无有效行")
    with open(RGBDGaze_INDEX_DIR / "index.csv", encoding="utf-8") as f:
        index_rows = list(csv.DictReader(f))
    N = len(index_rows)
    missing = [i for i in range(N) if i not in by_idx]
    print(f"索引 {N} 行 | jsonl 有效 {len(by_idx)} | 缺失 {len(missing)} 行 {missing[:12]}")

    # ---------- 1) 原图尺寸是否为常数（映射依赖常数 W）----------
    import cv2
    rng = random.Random(0)
    cand = rng.sample(range(N), min(args.size_check_n, N))
    sizes = set()
    for i in cand:
        p = Path(index_rows[i]["rgb_path"])
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        img = cv2.imread(str(p))
        if img is not None:
            sizes.add(img.shape[:2])
    print(f"抽查 {len(cand)} 张原图尺寸: {sorted(sizes)}")
    if len(sizes) != 1:
        raise SystemExit("原图尺寸不唯一 —— 无法用常数 W 做映射，需逐帧读取后重做。")
    H0, W0 = next(iter(sizes))
    print(f"原图尺寸常数 H0={H0} W0={W0}（索引 bbox 反映射回原图时需要 W0）")

    # ---------- 2) 缺失行兜底：同被试中位数（记录 filled）----------
    per_subj = defaultdict(list)
    for i, (le, re_, _fm) in by_idx.items():
        per_subj[index_rows[i]["subject"]].append((tuple(le), tuple(re_)))
    all_pairs = np.array([v for vs in per_subj.values() for v in vs], dtype=np.float64)
    global_med = np.median(all_pairs, axis=0)

    boxes = np.zeros((N, 8), dtype=np.float32)
    filled = np.zeros(N, dtype=bool)
    for i in range(N):
        if i in by_idx:
            le, re_, _fm = by_idx[i]
        else:
            pool = per_subj.get(index_rows[i]["subject"])
            le, re_ = (np.median(np.array(pool, dtype=np.float64), axis=0)
                       if pool else global_med)
            filled[i] = True
        boxes[i] = [*le, *re_]          # 原样保存（原图坐标，与裁剪所用帧一致）

    # ---------- 3) 抽样验证坐标系 ----------
    cand2 = rng.sample(range(N), min(args.check_n, N))
    dists = []
    for i in cand2:
        if i not in by_idx:
            continue
        ix = index_rows[i]
        fx, fy, fw, fh = (float(ix["bbox_x"]), float(ix["bbox_y"]),
                          float(ix["bbox_w"]), float(ix["bbox_h"]))
        # 索引 bbox（作者坐标系）→ 原图：x = W0-1-(y+h), y = x, w = h, h = w
        bx, by = W0 - 1 - (fy + fh), fx
        fm = by_idx[i][2]                       # MediaPipe 人脸框（原图坐标）
        d = float(np.hypot((bx + fh / 2) - (fm[0] + fm[2] / 2),
                           (by + fw / 2) - (fm[1] + fm[3] / 2))) / max(fw, fh)
        dists.append(d)
    med = float(np.median(dists)) if dists else float("nan")
    print(f"坐标一致性: 索引bbox反解中心 vs MediaPipe人脸框中心（按脸边长归一）"
          f"中位 = {med:.4f}（{len(dists)} 帧）")
    if not (med <= args.max_center_dist):
        raise SystemExit(f"坐标一致性未通过（{med:.4f} > {args.max_center_dist}），"
                         f"不要使用该表。")

    np.savez_compressed(args.out, idx=np.arange(N, dtype=np.int64),
                        boxes=boxes, filled=filled)
    print(f"已写出 {args.out}（{N} 帧，{Path(args.out).stat().st_size / 1e6:.1f} MB，"
          f"兜底 {int(filled.sum())} 帧）")
    print("眼框边长中位（旋转后）: 左 %.1f  右 %.1f px"
          % (np.median(np.maximum(boxes[:, 2], boxes[:, 3])),
             np.median(np.maximum(boxes[:, 6], boxes[:, 7]))))


if __name__ == "__main__":
    main()
