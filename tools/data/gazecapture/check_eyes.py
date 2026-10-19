"""眼框**退化/单眼**专项检查（量化 + 把最差的帧拼出来看 ✓）。

对每帧算：左/右眼框面积 ÷ 人脸框面积、框像素尺寸 ✓
定义退化 ✗：任一框 < 8px 宽 或 面积占比 < 0.3%（远侧眼轮廓点退化/侧脸/遮挡 ✗）

用法：
    nohup python -u -m tools.data.gazecapture.check_eyes --n 300 --seed 5 \
        --worst 24 --out out/gazecapture/analysis/eye_degenerate.jpg \
        > out/gazecapture/analysis/eye_degenerate.log 2>&1 &
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np


def _bootstrap() -> Path:
    here = Path(__file__).resolve()
    for d in (here.parent, *here.parents):
        if (d / "src" / "gazelab" / "__init__.py").is_file():
            for p in (str(d), str(d / "src")):
                if p not in sys.path:
                    sys.path.insert(0, p)
            return d
    raise RuntimeError("找不到项目根")


_ROOT = _bootstrap()
from configs.gazecapture_config import GAZECAPTURE_INDEX_DIR  # noqa: E402
from tools.data.common.eye_detect import detect_mp, make_landmarker, montage, draw  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--worst", type=int, default=24)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--cell", type=int, default=180)
    ap.add_argument("--out", default="out/gazecapture/analysis/eye_degenerate.jpg")
    args = ap.parse_args()

    lm = make_landmarker(_ROOT)
    paths = [ln.strip() for ln in (GAZECAPTURE_INDEX_DIR / f"{args.split}.txt")
             .read_text(encoding="utf-8").splitlines() if ln.strip()]
    rng = random.Random(args.seed)
    sample = rng.sample(paths, min(args.n, len(paths)))
    print(f"[eyes] 抽样 {len(sample)} 帧（seed={args.seed} ✓）", flush=True)

    rows = []  # (min_ratio, path, img, face, le, re)
    noface = 0
    for p in sample:
        img = cv2.imread(str(_ROOT / p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        face, le, re, _ = detect_mp(img, lm)
        if face is None:
            noface += 1
            continue
        fw, fh = max(face[2], 1), max(face[3], 1)
        fa = fw * fh
        rl = (le[2] * le[3]) / fa
        rr = (re[2] * re[3]) / fa
        rows.append((min(rl, rr), p, img, face, le, re))

    def bad(r):
        _, p, img, face, le, re = r
        return min(le[2], le[3]) < 8 or min(re[2], re[3]) < 8

    n_bad = sum(1 for r in rows if bad(r))
    n_small = sum(1 for r in rows if r[0] < 0.003)
    print(f"[eyes] n={len(rows)} 无脸={noface} | 任一眼框<8px={n_bad}"
          f"（{100*n_bad/max(len(rows),1):.1f}%）| 最小框面积占比<0.3%={n_small}"
          f"（{100*n_small/max(len(rows),1):.1f}%）", flush=True)
    print(f"[eyes] 面积占比分位: p01={np.percentile([r[0] for r in rows],1):.4f}"
          f" p05={np.percentile([r[0] for r in rows],5):.4f}"
          f" p50={np.percentile([r[0] for r in rows],50):.4f}", flush=True)

    rows.sort(key=lambda r: (r[4][2] * r[4][3] + r[5][2] * r[5][3]) / max(r[3][2] * r[3][3], 1))
    worst = rows[: args.worst]
    tiles = []
    for i, (_, p, img, face, le, re) in enumerate(worst, 1):
        print(f"  [{i}/{len(worst)}] L={le[2]}x{le[3]} R={re[2]}x{re[3]} "
              f"face={face[2]}x{face[3]}  {Path(p).name}", flush=True)
        tiles.append(draw(img, face, le, re))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), montage(tiles, args.cols, args.cell))
    print(f"[eyes] 最差 {len(tiles)} 帧 ⇒ {out}", flush=True)


if __name__ == "__main__":
    main()
