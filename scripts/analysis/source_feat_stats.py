"""计算源域特征统计（mean/cov），供 CORAL 使用。

只用无标签特征（不涉及标签）；用于目标域适配时的特征分布对齐。

用法：
  python -m scripts.analysis.source_feat_stats \
    --feats out/.../feats_gc_train300.npz --out out/.../coral_gc.npz
"""
import argparse
import sys

import numpy as np

sys.path[:0] = ["src", "."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, default=384)
    a = ap.parse_args()
    z = np.load(a.feats, allow_pickle=True)
    h = z["feats"].astype(np.float32)[:, :a.dim]
    print(f"[coral] frames={len(h)} dim={h.shape[1]} "
          f"norm_mean={np.linalg.norm(h, axis=1).mean():.3f}")
    mu = h.mean(0)
    hc = h - mu
    cov = (hc.T @ hc) / max(len(h) - 1, 1)
    np.savez(a.out, mean=mu.astype(np.float32), cov=cov.astype(np.float32),
             count=len(h))
    print(f"[coral] saved {a.out} mean_norm={np.linalg.norm(mu):.3f} "
          f"trace_cov={np.trace(cov):.3f}")


if __name__ == "__main__":
    main()
