"""Test-time input canonicalisation (experiment B).

Hypothesis
----------
The 2D gaze model has an orientation-dependent bias because the face crop is
fed in the device orientation.  If we rotate the crop to a canonical upright
orientation using the IMU gravity vector, run the model, and rotate the
predicted screen point back, the bias should shrink and a single calibration
should transfer across orientations.

Modes
-----
  base  : no rotation (matched baseline)
  canon : rotate crop by ``-sign * phi`` and rotate the prediction back

``phi = atan2(imu_x, -imu_y)`` is the in-plane device rotation used by the
earlier canonicalisation experiments.  ``--sign`` selects the rotation
direction so both conventions can be compared empirically.

Everything is logged: arguments, per-batch progress, per-orientation error,
and the exact rotation convention used.
"""
import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF

sys.path[:0] = ["src", "."]
from gazelab.datasets import RGBDGazeDataset, GazeCaptureDataset  # noqa: E402
from gazelab.datasets.table_index import TableIndex  # noqa: E402
from gazelab.models.dinov2_gaze import (  # noqa: E402
    RGBDGazeDINOv2, load_state_dict_any_res,
)


def log(msg):
    print(msg, flush=True)


def load_model(ckpt, device, use_geom=False, geom_orient=False):
    model = RGBDGazeDINOv2(
        gaze_dim=2, d_model=384, num_heads=8, num_layers=6,
        inter_dim=1536, freeze_dino=True, unfreeze_last=0,
        use_imu=False, use_geom=use_geom, use_geom_orient=geom_orient,
        use_eye=True, img_size=224,
    ).to(device)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    miss, unexp, resampled = load_state_dict_any_res(model, state)
    log(f"[canon] load missing={len(miss)} unexpected={len(unexp)} resampled={resampled}")
    if miss or unexp:
        raise RuntimeError("checkpoint mismatch")
    model.eval()
    return model


def rot_to_canon(p, phi):
    """Rotate normalised screen points about (0.5, 0.5) by -phi (radians)."""
    x = p[:, 0] - 0.5
    y = p[:, 1] - 0.5
    c, s = np.cos(phi), np.sin(phi)
    return np.stack([c * x + s * y + 0.5, -s * x + c * y + 0.5], 1)


def rotate_batch(x, angles_deg):
    """Rotate each sample of a [B, C, H, W] tensor by its own angle (degrees)."""
    if not np.any(angles_deg):
        return x
    return torch.stack([
        TF.rotate(x[i], float(angles_deg[i])) for i in range(x.shape[0])
    ]).to(x.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--feat-npz", required=True)
    ap.add_argument("--index-dir", default="data/GazeCapture/index")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--variants", nargs="+", default=["id", "flip"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dataset", choices=["rgbdgaze", "gazecapture"],
                    default="gazecapture")
    ap.add_argument("--use-geom", action="store_true")
    ap.add_argument("--geom-orient", action="store_true")
    ap.add_argument("--mode", choices=["base", "canon"], default="base")
    ap.add_argument("--sign", type=int, choices=[1, -1], default=1,
                    help="rotate crop by -sign*phi")
    ap.add_argument("--back-sign", type=int, choices=[1, -1], default=-1,
                    help="rotate prediction back by back_sign*phi")
    ap.add_argument("--geom-mode", choices=["keep", "canon"], default="keep",
                    help="keep original geom (incl. orientation one-hot) or "
                         "replace it with the canonical orientation")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N frames")
    ap.add_argument("--log-every", type=int, default=25)
    a = ap.parse_args()

    torch.set_num_threads(4)
    device = "cuda" if (a.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    log(f"[canon] device={device} args={vars(a)}")
    idx_dir = Path(a.index_dir)
    if a.dataset == "gazecapture":
        ds = GazeCaptureDataset(idx_dir / "index.csv", augment=False,
                                image_size=224, eye_crops=True,
                                geom_orient=a.geom_orient)
    else:
        ds = RGBDGazeDataset(idx_dir / "index.csv", augment=False,
                             image_size=224, eye_crops=True)
    table = TableIndex(idx_dir / "index.csv")
    with open(idx_dir / "index.csv", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
    pcol = "frame_path" if "frame_path" in header else "rgb_path"
    lookup = table.lookup(pcol)

    # IMU angle + orientation + screen size per frame path
    phi_of = {}
    ori_of = {}
    with open(idx_dir / "index.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            p = r[pcol]
            try:
                gx, gy = float(r["imu_x"]), float(r["imu_y"])
                phi_of[p] = float(np.arctan2(gx, -gy))
            except Exception:
                phi_of[p] = 0.0
            ori_of[p] = str(r.get("orientation", "0"))

    z = np.load(a.feat_npz, allow_pickle=True)
    paths = [str(p) for p in z["path"].tolist()]
    label = z["label"].astype(np.float32)
    fidx = z["frame_index"].astype(np.int64)
    if a.limit:
        paths, label, fidx = paths[:a.limit], label[:a.limit], fidx[:a.limit]
    idxs = [lookup[p] for p in paths if p in lookup]
    keep = [i for i, p in enumerate(paths) if p in lookup]
    log(f"[canon] frames={len(idxs)} variants={a.variants} mode={a.mode} "
        f"sign={a.sign} geom_mode={a.geom_mode}")
    model = load_model(a.ckpt, device, a.use_geom, a.geom_orient)

    out = np.zeros((len(idxs), 2), np.float32)
    phis = np.zeros(len(idxs), np.float64)
    t0 = time.time()
    for st in range(0, len(idxs), a.batch):
        sub = idxs[st:st + a.batch]
        ps = [paths[i] for i in keep[st:st + a.batch]]
        items = [ds[i] for i in sub]
        face = torch.stack([x[0].rgb for x in items]).to(device).float()
        el = torch.stack([x[0].eye_l for x in items]).to(device).float()
        er = torch.stack([x[0].eye_r for x in items]).to(device).float()
        gm = (torch.stack([x[0].geom for x in items]).to(device).float()
              if a.use_geom else None)
        phi = np.array([phi_of.get(p, 0.0) for p in ps])
        phis[st:st + len(sub)] = phi
        ang_deg = np.degrees(-a.sign * phi)
        acc = []
        with torch.no_grad():
            for v in a.variants:
                if v == "id":
                    f, elv, erv = face, el, er
                elif v == "flip":
                    f = torch.flip(face, [3])
                    elv, erv = torch.flip(er, [3]), torch.flip(el, [3])
                else:
                    raise ValueError(v)
                if a.mode == "canon":
                    f = rotate_batch(f, ang_deg)
                    elv = rotate_batch(elv, ang_deg)
                    erv = rotate_batch(erv, ang_deg)
                gmv = gm
                if a.mode == "canon" and a.geom_mode == "canon" and gm is not None:
                    gmv = gm.clone()
                    if gmv.shape[1] >= 4:  # [aspect, log w, log h, onehot...]
                        gmv[:, 3:] = 0.0
                        gmv[:, 3] = 1.0  # canonical = orientation 1
                q = model(f, eye_l=elv, eye_r=erv, geom=gmv).cpu().numpy()
                if v == "flip":
                    q = np.stack([1.0 - q[:, 0], q[:, 1]], 1)
                if a.mode == "canon":
                    # canonical prediction -> original screen frame
                    q = rot_to_canon(q.astype(np.float64), a.back_sign * phi)
                acc.append(q)
        out[st:st + len(sub)] = np.mean(acc, 0)
        if st % (a.batch * a.log_every) == 0:
            log(f"[canon] {st}/{len(idxs)} elapsed={time.time()-t0:.1f}s")
    label = label[keep]
    err = np.hypot(out[:, 0] - label[:, 0], out[:, 1] - label[:, 1])
    log(f"[canon] RESULT mode={a.mode} sign={a.sign} geom={a.geom_mode} "
        f"back_sign={a.back_sign} frames={len(out)} err={err.mean():.4f} "
        f"elapsed={time.time()-t0:.1f}s")
    by = defaultdict(lambda: [0.0, 0])
    for i, p in enumerate([paths[i] for i in keep]):
        o = ori_of.get(p, "0")
        by[o][0] += err[i]
        by[o][1] += 1
    for o, (s, n) in sorted(by.items()):
        log(f"[canon] ori={o} n={n} err={s/n:.4f}")
    np.savez_compressed(a.out, pred=out, label=label,
                        path=np.array([paths[i] for i in keep], object),
                        frame_index=fidx[keep], phi=phis,
                        orientation=np.array([ori_of.get(p, "0")
                                              for p in [paths[i] for i in keep]],
                                             object))
    log(f"[canon] saved {a.out}")


if __name__ == "__main__":
    main()
