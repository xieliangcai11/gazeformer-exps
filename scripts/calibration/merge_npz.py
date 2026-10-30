"""合并多卡分片推理结果（canon_infer --num-shards/--shard）。"""
import argparse
from pathlib import Path

import numpy as np


def log(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arrays = {}
    seen = set()
    total = 0
    for f in a.inputs:
        z = np.load(f, allow_pickle=True)
        log(f"[merge] {f} keys={list(z.keys())} n={len(z['path'])}")
        paths = [str(p) for p in z["path"].tolist()]
        keep = [i for i, p in enumerate(paths) if p not in seen]
        for k in z.files:
            v = z[k]
            if v.shape and len(v) == len(paths):
                arrays.setdefault(k, []).append(v[keep])
            elif k not in arrays:
                arrays[k] = [v]
        seen.update(paths[i] for i in keep)
        total += len(keep)
        log(f"[merge] kept={len(keep)} dup={len(paths)-len(keep)} total={total}")
    out = {}
    for k, vs in arrays.items():
        if all(isinstance(v, np.ndarray) and v.ndim >= 1 for v in vs):
            out[k] = np.concatenate(vs, 0)
        else:
            out[k] = vs[0]
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **out)
    log(f"[merge] saved {p} frames={len(out.get('path', []))}")


if __name__ == "__main__":
    main()
