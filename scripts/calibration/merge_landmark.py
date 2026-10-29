"""Merge sharded landmark feature npz files."""
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    feats = []
    valid = []
    rows = []
    names = None
    for p in a.inputs:
        z = np.load(p, allow_pickle=True)
        feats.append(z["feats"])
        valid.append(z["valid"])
        rows.append(z["row_idx"])
        names = z["names"]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, feats=np.concatenate(feats),
                        valid=np.concatenate(valid),
                        row_idx=np.concatenate(rows), names=names)
    print(f"[merge] saved {out} frames={sum(len(x) for x in valid)}", flush=True)


if __name__ == "__main__":
    main()
