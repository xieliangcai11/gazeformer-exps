"""RGBDGaze context-bias diagnostic.

The RGD device orientation is fixed (portrait), so cross-orientation
equivariance pairs do not exist.  The relevant context is *activity* and
*device tilt* (sitting / standing / walking / lying, gravity z).  This script
quantifies how much of the residual

    r = pred - label        (normalised screen coordinates)

is systematic (group mean) versus random (within group), in centimetres.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path[:0] = ["src", "."]


def log(msg):
    print(msg, flush=True)


def stats(rs, wcm, hcm):
    """rs: [N,2] normalised residuals; return cm statistics."""
    rcm = np.stack([rs[:, 0] * wcm, rs[:, 1] * hcm], 1)
    mu = rcm.mean(0)
    tot = float(np.linalg.norm(rcm, axis=1).mean())
    sysn = float(np.linalg.norm(mu))
    return dict(n=int(len(rs)), mean_x=float(mu[0]), mean_y=float(mu[1]),
                sys=sysn, tot=tot, ratio=sysn / max(tot, 1e-9),
                std_x=float(rcm[:, 0].std()), std_y=float(rcm[:, 1].std()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    log(f"[rgd] args={vars(a)}")

    z = np.load(a.test_npz, allow_pickle=True)
    pred = z["pred"].astype(np.float64)
    label = z["label"].astype(np.float64)
    paths = [str(p) for p in z["path"].tolist()]
    need = set(paths)
    meta = {}
    with open(a.index_csv, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            p = r["rgb_path"]
            if p not in need:
                continue
            sw, sh = float(r["screen_w"]), float(r["screen_h"])
            cmx, cmy = float(r["cm_px_x"]), float(r["cm_px_y"])
            gx, gy, gz = (float(r["imu_x"]), float(r["imu_y"]),
                          float(r["imu_z"]))
            meta[p] = dict(
                subject=r["subject"], activity=r["activity"],
                device=r["device"], sw=sw, sh=sh, cmx=cmx, cmy=cmy,
                wcm=sw * cmx, hcm=sh * cmy,
                phi=float(np.arctan2(gx, -gy)),
                tilt=float(np.degrees(np.arccos(np.clip(-gz, -1.0, 1.0)))),
                gx=gx, gy=gy, gz=gz,
            )
    keep = [i for i, p in enumerate(paths) if p in meta]
    log(f"[rgd] frames={len(paths)} joined={len(keep)}")
    if not keep:
        raise SystemExit("no joined frames")
    res = pred[keep] - label[keep]
    wcm = np.array([meta[paths[i]]["wcm"] for i in keep])
    hcm = np.array([meta[paths[i]]["hcm"] for i in keep])
    tilt = np.array([meta[paths[i]]["tilt"] for i in keep])
    rcm = np.stack([res[:, 0] * wcm, res[:, 1] * hcm], 1)

    def grouped(keyfn, name):
        g = defaultdict(list)
        for j, i in enumerate(keep):
            g[keyfn(meta[paths[i]])].append(j)
        rows = {}
        log(f"[rgd] ---- by {name} ----")
        for k in sorted(g):
            j = np.array(g[k])
            st = stats(res[j], wcm[j], hcm[j])
            rows[str(k)] = st
            log(f"[rgd] {name}={k} n={st['n']} mean=({st['mean_x']:+.3f},"
                f"{st['mean_y']:+.3f})cm sys={st['sys']:.3f} "
                f"tot={st['tot']:.3f} ratio={st['ratio']:.3f} "
                f"std=({st['std_x']:.3f},{st['std_y']:.3f})")
        return rows

    by_act = grouped(lambda m: m["activity"], "activity")
    by_tilt = grouped(lambda m: int(min(90, m["tilt"] // 15) * 15), "tilt")
    by_dev = grouped(lambda m: m["device"], "device")
    by_sub = grouped(lambda m: m["subject"], "subject")

    # variance explained by context (eta^2) on cm residuals
    def eta2(keyfn):
        g = defaultdict(list)
        for j, i in enumerate(keep):
            g[keyfn(meta[paths[i]])].append(j)
        mu = rcm.mean(0)
        between = 0.0
        within = 0.0
        for k, js in g.items():
            js = np.array(js)
            m = rcm[js].mean(0)
            between += len(js) * np.sum((m - mu) ** 2)
            within += np.sum((rcm[js] - m) ** 2)
        tot = between + within
        return dict(between=float(between), within=float(within),
                    eta2=float(between / max(tot, 1e-12)))

    e_act = eta2(lambda m: m["activity"])
    e_tilt = eta2(lambda m: int(min(90, m["tilt"] // 15) * 15))
    e_dev = eta2(lambda m: m["device"])
    e_sub = eta2(lambda m: m["subject"])
    log(f"[rgd] eta2 activity={e_act['eta2']:.4f} tilt={e_tilt['eta2']:.4f} "
        f"device={e_dev['eta2']:.4f} subject={e_sub['eta2']:.4f}")

    # correlation of residual components with tilt / gravity
    corr = {}
    for nm, v in [("tilt", tilt),
                  ("gx", np.array([meta[paths[i]]["gx"] for i in keep])),
                  ("gy", np.array([meta[paths[i]]["gy"] for i in keep])),
                  ("gz", np.array([meta[paths[i]]["gz"] for i in keep]))]:
        for ax, rv in [("x", rcm[:, 0]), ("y", rcm[:, 1])]:
            c = float(np.corrcoef(v, rv)[0, 1]) if np.std(v) > 1e-9 else 0.0
            corr[f"{nm}_{ax}"] = c
            log(f"[rgd] corr {nm} vs r_{ax} = {c:+.4f}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(dict(args=vars(a), activity=by_act, tilt=by_tilt,
                       device=by_dev, subject=by_sub,
                       eta2=dict(activity=e_act, tilt=e_tilt,
                                 device=e_dev, subject=e_sub),
                       corr=corr), fh, indent=2)
    np.savez_compressed(out.with_suffix(".npz"),
                        path=np.array([paths[i] for i in keep], object),
                        res=res, rcm=rcm, tilt=tilt)
    log(f"[rgd] saved {out}.json / {out}.npz")


if __name__ == "__main__":
    main()
