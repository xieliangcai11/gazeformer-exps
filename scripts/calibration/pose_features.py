"""从索引+眼框提取头姿/眼部几何代理特征（两个数据集通用）。

不依赖 IMU，只用：
- 人脸 bbox；
- 左右眼 bbox（MediaPipe 已生成）；
- 相对人脸框的归一化位置/角度。

输出特征（每帧 8 维）：
  yaw_proxy, pitch_proxy, roll_proxy, iod_ratio,
  openness, eye_edge_l, eye_edge_r, face_ratio

所有量都相对人脸框归一化，因此跨设备/分辨率可比。

用法：
  python -m scripts.calibration.pose_features \
    --dataset rgbdgaze --index-dir data/RGBDGaze/index \
    --eye-boxes data/RGBDGaze/index/eye_boxes.npz \
    --out out/rgbdgaze/analysis/pose_features.npz
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = ["src", "."]


def load_eye_boxes(path):
    z = np.load(path, allow_pickle=True)
    boxes = z["boxes"].astype(np.float32)
    idx = z["idx"].astype(np.int64)
    filled = z["filled"].astype(bool) if "filled" in z else np.zeros(len(boxes), bool)
    return boxes, idx, filled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--eye-boxes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="all", help="all/train/val/test（按 split.txt 过滤）")
    ap.add_argument("--index-bbox-rot90", action="store_true",
                    help="索引 bbox 在 rot90 坐标系（RGD），先反变换到原图")
    ap.add_argument("--image-width", type=float, default=1080.0,
                    help="原图宽度（rot90 反变换用）")
    a = ap.parse_args()

    index_csv = Path(a.index_dir) / "index.csv"
    rows = list(csv.DictReader(open(index_csv, encoding="utf-8")))
    print(f"[pose] index rows={len(rows)} dataset={a.dataset}")

    boxes, idx, filled = load_eye_boxes(a.eye_boxes)
    print(f"[pose] eye boxes={len(boxes)} idx=[{idx.min()},{idx.max()}] "
          f"filled={int(filled.sum())} ({100*filled.mean():.2f}%)")
    # 对齐：eye_boxes 的 idx 是 index.csv 的行号（0 基）
    max_idx = int(idx.max()) if len(idx) else -1
    if max_idx >= len(rows):
        raise SystemExit(f"eye_boxes idx 超出 index 行数：max={max_idx} rows={len(rows)}")
    dup = len(idx) - len(np.unique(idx))
    print(f"[pose] duplicate idx={dup}")
    box_at = np.full((len(rows), 8), np.nan, np.float32)
    fill_at = np.ones(len(rows), bool)
    box_at[idx] = boxes
    fill_at[idx] = filled

    paths = []
    if a.split != "all":
        keep = {ln.strip() for ln in (Path(a.index_dir) / f"{a.split}.txt").read_text(
            encoding="utf-8").splitlines() if ln.strip()}
        for i, r in enumerate(rows):
            if r.get("rgb_path") or r.get("frame_path"):
                p = r.get("rgb_path") or r.get("frame_path")
                if p in keep:
                    paths.append(i)
        print(f"[pose] split={a.split} frames={len(paths)}")
    else:
        paths = list(range(len(rows)))

    feats = np.zeros((len(paths), 8), np.float32)
    valid = np.zeros(len(paths), bool)
    inside = []
    for k, i in enumerate(paths):
        r = rows[i]
        bx, by = float(r["bbox_x"]), float(r["bbox_y"])
        bw, bh = float(r["bbox_w"]), float(r["bbox_h"])
        if a.index_bbox_rot90:
            # raw→rot90: (x',y')=(y,W0-1-x) ⇒ 反解
            bx, by, bw, bh = (a.image_width - 1 - (by + bh), bx, bh, bw)
        if not np.isfinite([bx, by, bw, bh]).all() or bw <= 1 or bh <= 1:
            continue
        b = box_at[i]
        if not np.isfinite(b).all() or fill_at[i]:
            continue
        lx, ly, lw, lh, rx, ry, rw, rh = b
        if lw <= 1 or lh <= 1 or rw <= 1 or rh <= 1:
            continue
        lcx, lcy = lx + lw / 2, ly + lh / 2
        rcx, rcy = rx + rw / 2, ry + rh / 2
        fcx, fcy = bx + bw / 2, by + bh / 2
        mx, my = (lcx + rcx) / 2, (lcy + rcy) / 2
        yaw = (mx - fcx) / bw
        pitch = (my - fcy) / bh
        roll = np.arctan2(rcy - lcy, rcx - lcx) / np.pi
        iod = (rcx - lcx) / bw
        openness = 0.5 * (lh / lw + rh / rw)
        edge_l = (fcx - lcx) / bw
        edge_r = (rcx - fcx) / bw
        face_ratio = bw / bh
        feats[k] = [yaw, pitch, roll, iod, openness, edge_l, edge_r, face_ratio]
        valid[k] = True
        inside.append(((lx >= bx - 0.2*bw) and (lx+lw <= bx + 1.2*bw) and
                       (ly >= by - 0.2*bh) and (ly+lh <= by + 1.2*bh) and
                       (rx >= bx - 0.2*bw) and (rx+rw <= bx + 1.2*bw) and
                       (ry >= by - 0.2*bh) and (ry+rh <= by + 1.2*bh)))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out,
                        row_idx=np.array(paths, np.int64),
                        feats=feats, valid=valid)
    print(f"[pose] saved {out} frames={len(paths)} valid={int(valid.sum())} "
          f"({100*valid.mean():.2f}%)")
    if inside:
        frac = float(np.mean(inside))
        print(f"[pose] 眼框落在人脸框(±20%)内比例={frac:.3f}")
        if frac < 0.8:
            print("[pose][warn] 眼框与人脸框坐标可能不一致，请先核对坐标系")
    if valid.any():
        v = feats[valid]
        names = ["yaw", "pitch", "roll", "iod", "open", "edge_l", "edge_r", "face_ratio"]
        for j, n in enumerate(names):
            print(f"  {n:10s} mean={v[:,j].mean():+.4f} std={v[:,j].std():.4f} "
                  f"p1={np.percentile(v[:,j],1):+.4f} p99={np.percentile(v[:,j],99):+.4f}")


if __name__ == "__main__":
    main()
