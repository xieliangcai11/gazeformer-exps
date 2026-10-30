"""Phase-0 diagnostic: how equivariant is the 2D gaze model to screen context?

For a frame with real context c we build counterfactual contexts c' (other
device orientations / screen geometries), run the SAME image through the
model, transform the prediction and the label geometrically, and report:

  eq_err    || f(x,c') - T_{c->c'} f(x,c) ||
  bias_eq   || (f(x,c') - T y) - T (f(x,c) - y) ||

If the model were perfectly equivariant both are zero.  ``bias_eq`` isolates
the context-dependent residual bias beta(c), which is exactly what the
residual-equivariance fine-tuning would minimise.

IMU handling
------------
Two angle sources are supported:
  nominal : 0, 90, 180, 270 degrees from the orientation label
  imu     : phi = atan2(imu_x, -imu_y), target phi shifted by the nominal
            orientation delta

The transform uses physical centimetres, so non-square screens are handled
correctly (a 90-degree rotation is not a 90-degree rotation of normalised
coordinates).

All results are logged per context pair and per tilt bin, and raw per-frame
arrays are saved for further analysis.
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = ["src", "."]
from gazelab.datasets import GazeCaptureDataset  # noqa: E402
from gazelab.datasets.table_index import TableIndex  # noqa: E402
from gazelab.models.dinov2_gaze import (  # noqa: E402
    RGBDGazeDINOv2, load_state_dict_any_res,
)


def log(msg):
    print(msg, flush=True)


NOMINAL = {"1": 0.0, "2": np.pi, "3": -np.pi / 2, "4": np.pi / 2}
SWAP = {("1", "3"), ("3", "1"), ("1", "4"), ("4", "1"),
        ("2", "3"), ("3", "2"), ("2", "4"), ("4", "2")}


def load_model(ckpt, device, use_geom=True, geom_orient=True):
    model = RGBDGazeDINOv2(
        gaze_dim=2, d_model=384, num_heads=8, num_layers=6,
        inter_dim=1536, freeze_dino=True, unfreeze_last=0,
        use_imu=False, use_geom=use_geom, use_geom_orient=geom_orient,
        use_eye=True, img_size=224,
    ).to(device)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        log(f"[eq] ckpt epoch={state.get('epoch')} best_val={state.get('best_val')}")
        state = state["model"]
    miss, unexp, resampled = load_state_dict_any_res(model, state)
    log(f"[eq] load missing={len(miss)} unexpected={len(unexp)} resampled={resampled}")
    if miss or unexp:
        raise RuntimeError("checkpoint mismatch")
    model.eval()
    return model


def rot(v, ang):
    """Rotate 2D vectors by -ang (canonicalisation convention)."""
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([c * v[..., 0] + s * v[..., 1],
                     -s * v[..., 0] + c * v[..., 1]], -1)


def geom_target(base, c, t):
    """Counterfactual geometry token for orientation t (base layout 7 dims)."""
    g = base.copy()
    if (c, t) in SWAP:
        g[0] = 1.0 / max(base[0], 1e-6)
        g[1], g[2] = base[2], base[1]
    g[3:7] = 0.0
    g[3 + int(t) - 1] = 1.0
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index-dir", default="data/GazeCapture/index")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--angle-source", choices=["imu", "nominal"], default="imu")
    ap.add_argument("--t-sign", type=int, choices=[1, -1], default=1,
                    help="sign of the geometric rotation in T")
    ap.add_argument("--targets", nargs="+", default=["1", "2", "3", "4"])
    ap.add_argument("--use-geom", action="store_true")
    ap.add_argument("--geom-orient", action="store_true")
    ap.add_argument("--log-every", type=int, default=25)
    a = ap.parse_args()

    log(f"[eq] args={vars(a)}")
    device = "cuda" if (a.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    log(f"[eq] device={device}")
    idx_dir = Path(a.index_dir)
    ds = GazeCaptureDataset(idx_dir / "index.csv", augment=False,
                            image_size=224, eye_crops=True,
                            geom_orient=a.geom_orient)
    table = TableIndex(idx_dir / "index.csv")
    with open(idx_dir / "index.csv", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
    pcol = "frame_path" if "frame_path" in header else "rgb_path"
    lookup = table.lookup(pcol)

    meta = {}
    paths = []
    with open(idx_dir / "index.csv", encoding="utf-8") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            if a.split and r.get("split", a.split) != a.split:
                continue
            if i % a.stride:
                continue
            p = r[pcol]
            if p not in lookup:
                continue
            try:
                gx, gy, gz = (float(r["imu_x"]), float(r["imu_y"]),
                              float(r["imu_z"]))
            except Exception:
                gx, gy, gz = 0.0, -1.0, 0.0
            meta[p] = dict(
                ori=str(r.get("orientation", "0")),
                sw=float(r["screen_w"]), sh=float(r["screen_h"]),
                cmx=float(r["cm_px_x"]), cmy=float(r["cm_px_y"]),
                gx=gx, gy=gy, gz=gz,
                phi=float(np.arctan2(gx, -gy)),
                tilt=float(np.degrees(np.arccos(np.clip(-gz, -1.0, 1.0)))),
                gx_eye=float(r.get("gaze_x", 0.0)),
                gy_eye=float(r.get("gaze_y", 0.0)),
            )
            paths.append(p)
            if a.limit and len(paths) >= a.limit:
                break
    log(f"[eq] split={a.split} frames={len(paths)} stride={a.stride} "
        f"angle_source={a.angle_source} t_sign={a.t_sign}")
    if not paths:
        raise SystemExit("no frames")
    idxs = [lookup[p] for p in paths]
    model = load_model(a.ckpt, device, a.use_geom, a.geom_orient)

    tgt = [str(t) for t in a.targets]
    acc = defaultdict(lambda: defaultdict(float))
    cnt = defaultdict(int)
    tilt_bins = defaultdict(lambda: defaultdict(float))
    tilt_cnt = defaultdict(int)
    raw = defaultdict(list)
    t0 = time.time()
    for st in range(0, len(idxs), a.batch):
        sub = idxs[st:st + a.batch]
        ps = paths[st:st + a.batch]
        items = [ds[i] for i in sub]
        face = torch.stack([x[0].rgb for x in items]).to(device).float()
        el = torch.stack([x[0].eye_l for x in items]).to(device).float()
        er = torch.stack([x[0].eye_r for x in items]).to(device).float()
        geom = torch.stack([x[0].geom for x in items]).to(device).float()
        with torch.no_grad():
            p0 = model(face, eye_l=el, eye_r=er, geom=geom).cpu().numpy()
        # per-target batch forward
        for t in tgt:
            g2 = torch.stack([
                torch.from_numpy(geom_target(geom[j].cpu().numpy(),
                                             meta[p]["ori"], t))
                for j, p in enumerate(ps)
            ]).float().to(device)
            with torch.no_grad():
                pt = model(face, eye_l=el, eye_r=er, geom=g2).cpu().numpy()
            for j, p in enumerate(ps):
                m = meta[p]
                c = m["ori"]
                if t == c:
                    continue
                if a.angle_source == "imu":
                    phic = m["phi"]
                else:
                    phic = NOMINAL.get(c, 0.0)
                wcm, hcm = m["sw"] * m["cmx"], m["sh"] * m["cmy"]
                y = np.array([m["gx_eye"] / m["sw"], m["gy_eye"] / m["sh"]])
                if (c, t) in SWAP:
                    wcm2, hcm2 = hcm, wcm
                else:
                    wcm2, hcm2 = wcm, hcm
                if a.angle_source == "imu":
                    dphi = NOMINAL.get(t, 0.0) - phic
                else:
                    dphi = NOMINAL.get(t, 0.0) - NOMINAL.get(c, 0.0)
                dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
                dphi = a.t_sign * dphi

                def T(q):
                    q = np.asarray(q, np.float64)
                    v = np.stack([(q[..., 0] - 0.5) * wcm,
                                  (q[..., 1] - 0.5) * hcm], -1)
                    v = rot(v, dphi)
                    return np.stack([v[..., 0] / wcm2 + 0.5,
                                     v[..., 1] / hcm2 + 0.5], -1)

                Tp = T(p0[j])
                Ty = T(y)
                r = p0[j] - y
                rt = pt[j] - Ty
                Tr = Tp - Ty
                eq = float(np.linalg.norm(pt[j] - Tp))
                beq = float(np.linalg.norm(rt - Tr))
                b0 = float(np.linalg.norm(r))
                bt = float(np.linalg.norm(rt))
                key = f"{c}->{t}"
                acc[key]["eq"] += eq
                acc[key]["bias_eq"] += beq
                acc[key]["base"] += b0
                acc[key]["target"] += bt
                cnt[key] += 1
                tb = min(60, int(m["tilt"] // 15) * 15)
                tilt_bins[tb]["eq"] += eq
                tilt_bins[tb]["bias_eq"] += beq
                tilt_cnt[tb] += 1
                raw["pair"].append(key)
                raw["eq"].append(eq)
                raw["bias_eq"].append(beq)
                raw["base"].append(b0)
                raw["target"].append(bt)
                raw["tilt"].append(m["tilt"])
        if st % (a.batch * a.log_every) == 0:
            log(f"[eq] {st}/{len(idxs)} elapsed={time.time()-t0:.1f}s")

    log("[eq] ---- per context pair ----")
    summary = {}
    for k in sorted(acc):
        n = cnt[k]
        d = {m: acc[k][m] / n for m in acc[k]}
        summary[k] = d
        log(f"[eq] {k} n={n} base={d['base']:.4f} target={d['target']:.4f} "
            f"eq={d['eq']:.4f} bias_eq={d['bias_eq']:.4f}")
    log("[eq] ---- per tilt bin ----")
    tilt_summary = {}
    for b in sorted(tilt_bins):
        n = tilt_cnt[b]
        d = {m: tilt_bins[b][m] / n for m in tilt_bins[b]}
        tilt_summary[str(b)] = d
        log(f"[eq] tilt>={b} n={n} eq={d['eq']:.4f} bias_eq={d['bias_eq']:.4f}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"),
                        **{k: np.array(v) for k, v in raw.items()})
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(dict(args=vars(a), pairs=summary, tilt=tilt_summary), fh,
                  indent=2)
    log(f"[eq] saved {out}.npz / {out}.json")


if __name__ == "__main__":
    main()
