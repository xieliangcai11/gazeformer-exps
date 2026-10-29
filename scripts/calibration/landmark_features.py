"""Extract shared eye/iris/face-geometry features with MediaPipe FaceLandmarker.

Both RGBDGaze and GazeCapture have RGB images and face boxes, so these features
are available on both datasets without retraining any base model.

Features per frame (all face/eye relative, so cross-resolution comparable):
  left/right iris relative position, eye aspect ratios, inter-iris distance,
  face box geometry, brightness/contrast/blur, iris pixel diameter (diagnostic).
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

sys.path[:0] = ["src", "."]

# MediaPipe FaceMesh indices
LEFT_OUTER, LEFT_INNER, LEFT_UP, LEFT_LOW = 33, 133, 159, 145
RIGHT_OUTER, RIGHT_INNER, RIGHT_UP, RIGHT_LOW = 263, 362, 386, 374
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

NAMES = [
    "iris_lx", "iris_ly", "iris_rx", "iris_ry",
    "eye_l_w", "eye_l_h", "eye_r_w", "eye_r_h",
    "ear_l", "ear_r", "iod_face", "face_cx", "face_cy",
    "face_w", "face_h", "face_ratio", "bright", "contrast", "blur",
    "iris_px_l", "iris_px_r",
]


def log(msg):
    print(msg, flush=True)


def make_landmarker():
    base = mpp.BaseOptions(model_asset_path="assets/face_landmarker.task")
    opts = vision.FaceLandmarkerOptions(
        base_options=base, num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def eye_feats(lm, outer, inner, up, low, iris_ids, W, H):
    def P(i):
        return np.array([lm[i].x * W, lm[i].y * H], float)
    o, n, u, low_pt = P(outer), P(inner), P(up), P(low)
    iris = np.stack([P(i) for i in iris_ids])
    ic = iris.mean(0)
    w = float(np.linalg.norm(o - n))
    h = float(np.linalg.norm(u - low_pt))
    if w < 1 or h < 1:
        raise ValueError("degenerate eye")
    x0 = min(o[0], n[0])
    rx = (ic[0] - x0) / w
    ry = (ic[1] - (u[1] + low_pt[1]) / 2.0) / h
    iris_px = float(np.linalg.norm(iris - ic, axis=1).max() * 2)
    return ic, rx, ry, w, h, w / h, iris_px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()

    idx_dir = Path(a.index_dir)
    idx_csv = idx_dir / "index.csv"
    rows = list(csv.DictReader(open(idx_csv, encoding="utf-8")))
    pcol = "frame_path" if rows and "frame_path" in rows[0] else "rgb_path"
    keep = None
    if (idx_dir / f"{a.split}.txt").is_file():
        keep = {ln.strip() for ln in (idx_dir / f"{a.split}.txt").read_text(
            encoding="utf-8").splitlines() if ln.strip()}
    sel = [i for i, r in enumerate(rows)
           if (keep is None or r[pcol] in keep)]
    if a.limit:
        sel = sel[:a.limit]
    if a.num_shards > 1:
        sel = sel[a.shard::a.num_shards]
    log(f"[lm] split={a.split} shard={a.shard}/{a.num_shards} "
        f"frames={len(sel)} out={a.out}")

    lm = make_landmarker()
    feats = np.zeros((len(sel), len(NAMES)), np.float32)
    valid = np.zeros(len(sel), bool)
    t0 = time.time()
    n_ok = n_bad = 0
    for k, i in enumerate(sel):
        r = rows[i]
        p = r[pcol]
        im = cv2.imread(p)
        if im is None:
            n_bad += 1
            continue
        H, W = im.shape[:2]
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks:
            n_bad += 1
            continue
        L = res.face_landmarks[0]
        try:
            lc, lrx, lry, lw, lh, lear, lpx = eye_feats(
                L, LEFT_OUTER, LEFT_INNER, LEFT_UP, LEFT_LOW, LEFT_IRIS, W, H)
            rc, rrx, rry, rw, rh, rear, rpx = eye_feats(
                L, RIGHT_OUTER, RIGHT_INNER, RIGHT_UP, RIGHT_LOW, RIGHT_IRIS, W, H)
        except Exception:
            n_bad += 1
            continue
        bx, by = float(r["bbox_x"]), float(r["bbox_y"])
        bw, bh = float(r["bbox_w"]), float(r["bbox_h"])
        x0, y0 = int(max(0, bx)), int(max(0, by))
        x1, y1 = int(min(W, bx + bw)), int(min(H, by + bh))
        if x1 <= x0 + 2 or y1 <= y0 + 2:
            n_bad += 1
            continue
        crop = cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        bright = float(crop.mean() / 255.0)
        contrast = float(crop.std() / 255.0)
        blur = float(cv2.Laplacian(crop, cv2.CV_64F).var())
        iod = float(np.linalg.norm(lc - rc) / max(bw, 1))
        feats[k] = [
            lrx, lry, rrx, rry, lw / max(bw, 1), lh / max(bw, 1),
            rw / max(bw, 1), rh / max(bw, 1), lear, rear, iod,
            (bx + bw / 2) / W, (by + bh / 2) / H,
            bw / W, bh / H, bw / max(bh, 1), bright, contrast, blur,
            lpx, rpx,
        ]
        valid[k] = True
        n_ok += 1
        if k % a.log_every == 0:
            log(f"[lm] {k}/{len(sel)} ok={n_ok} bad={n_bad} "
                f"elapsed={time.time()-t0:.1f}s")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, row_idx=np.array(sel, np.int64),
                        feats=feats, valid=valid, names=np.array(NAMES, object))
    log(f"[lm] saved {out} frames={len(sel)} valid={int(valid.sum())} "
        f"({100*valid.mean():.2f}%) bad={n_bad} elapsed={time.time()-t0:.1f}s")
    v = feats[valid]
    if len(v):
        for j, n in enumerate(NAMES):
            log(f"[lm] {n:12s} mean={v[:,j].mean():+.4f} std={v[:,j].std():.4f} "
                f"p1={np.percentile(v[:,j],1):+.4f} p99={np.percentile(v[:,j],99):+.4f}")
    try:
        lm.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
