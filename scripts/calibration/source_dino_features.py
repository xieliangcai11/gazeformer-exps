"""为当前 cm 模型的训练预测生成对齐的 DINO PCA 特征。"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path[:0] = ["src", "."]


def log(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-npz", required=True)
    ap.add_argument("--pred-npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--components", type=int, default=16)
    a = ap.parse_args()
    z = np.load(a.feat_npz, allow_pickle=True)
    X = np.asarray(z["feats"], np.float32)
    fpath = [str(p) for p in z["path"].tolist()]
    mp = {p: i for i, p in enumerate(fpath)}
    pz = np.load(a.pred_npz, allow_pickle=True)
    ppath = [str(p) for p in pz["path"].tolist()]
    log(f"[src] feat frames={len(fpath)} pred frames={len(ppath)}")
    pca = PCA(n_components=a.components, svd_solver="randomized",
              random_state=0)
    pca.fit(X)
    keep = [i for i, p in enumerate(ppath) if p in mp]
    rows = [mp[ppath[i]] for i in keep]
    Z = pca.transform(X[rows]).astype(np.float32)
    names = [f"dino_pca_{i}" for i in range(a.components)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, path=np.array([ppath[i] for i in keep], object),
                        feats=Z, valid=np.ones(len(keep), bool),
                        names=np.array(names, object))
    log(f"[src] saved {out} n={len(keep)} explained="
        f"{float(pca.explained_variance_ratio_.sum()):.4f}")


if __name__ == "__main__":
    main()
