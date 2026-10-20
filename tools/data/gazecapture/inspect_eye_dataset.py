"""检查眼支流数据：把 Dataset 实际产出的 `eye_l / eye_r` 张量还原成图片拼图。

用途：验证 eye_boxes.npz 与帧是否**逐行对齐**（错位时看到的是别人的眼睛/额头），
以及退化框修复（filled=True）后的框是否仍然套住眼睛。

用法：
    python -m tools.data.gazecapture.inspect_eye_dataset --n 12
    python -m tools.data.gazecapture.inspect_eye_dataset --n 12 --filled-only
"""
from __future__ import annotations

import argparse
import random
import sys
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
    raise RuntimeError("找不到项目根")


_ROOT = _bootstrap()
from configs.gazecapture_config import GAZECAPTURE_INDEX_DIR  # noqa: E402
from gazelab.datasets.gazecapture import GazeCaptureDataset  # noqa: E402

#: ImageNet 反归一化（数据集默认用 rgb_preprocess）
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _to_bgr(t) -> np.ndarray:
    x = t.detach().cpu().numpy().transpose(1, 2, 0)
    x = (x * _STD + _MEAN) * 255.0
    return np.clip(x[:, :, ::-1], 0, 255).astype(np.uint8)   # RGB -> BGR


def main():
    ap = argparse.ArgumentParser(description="眼支流数据可视化核验")
    ap.add_argument("--index-dir", default=str(GAZECAPTURE_INDEX_DIR))
    ap.add_argument("--n", type=int, default=12, help="样本数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--filled-only", action="store_true",
                    help="只抽 filled=True（眼框不可用，训练时会被剔除）的行")
    ap.add_argument("--recovered", action="store_true",
                    help="只抽“原本退化、经放大重检恢复”的行（验证恢复框是否真的在眼睛上）")
    ap.add_argument("--out", default="out/gazecapture/analysis/eye_dataset_montage.jpg")
    args = ap.parse_args()

    import cv2
    cv2.setNumThreads(1)
    ds = GazeCaptureDataset(index_csv=Path(args.index_dir) / "index.csv",
                            eye_crops=True, image_size=224, augment=False)
    filled = np.asarray(ds.eye_filled, bool)
    if args.recovered:
        from tools.data.gazecapture.refine_eye_boxes import degenerate_mask, _load_index
        idx_dir = Path(args.index_dir)
        raw = np.load(idx_dir / "eye_boxes.raw.npz")
        paths, faces = _load_index(idx_dir / "index.csv")
        was_deg = degenerate_mask(raw["boxes"].astype(np.float32),
                                  faces[:, 2] * faces[:, 3], 8.0, 0.003)
        pool = np.nonzero(was_deg & ~filled)[0]
    elif args.filled_only:
        pool = np.nonzero(filled)[0]
    else:
        pool = np.arange(len(filled))
    rng = random.Random(args.seed)
    pick = rng.sample(list(pool), min(args.n, len(pool)))
    print(f"[inspect] 池 {len(pool)} | 抽样 {len(pick)} | filled_only={args.filled_only}")

    tiles = []
    for i in pick:
        inp, _ = ds[int(i)]
        a, b = _to_bgr(inp.eye_l), _to_bgr(inp.eye_r)
        row = np.hstack([a, np.full((a.shape[0], 4, 3), 255, np.uint8), b])
        box_l = ds.eye_boxes[i, 0:4].astype(int).tolist()
        box_r = ds.eye_boxes[i, 4:8].astype(int).tolist()
        print(f"  idx={i:>7d} filled={bool(filled[i])} L={box_l} R={box_r}")
        tiles.append(row)

    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    th, tw = tiles[0].shape[:2]
    canvas = np.full((rows * th, cols * tw, 3), 255, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        canvas[r * th:r * th + t.shape[0], c * tw:c * tw + t.shape[1]] = t
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print(f"[inspect] 拼图 -> {out}")


if __name__ == "__main__":
    main()
