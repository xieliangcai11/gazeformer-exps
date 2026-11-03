"""把 DINOv2 1536 维特征做 PCA，生成校准可用的低维 extra 特征。"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path[:0] = ["src", "."]
from gazelab.datasets.table_index import TableIndex  # noqa: E402


def log(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--components", type=int, default=16)
    a = ap.parse_args()
    log(f"[dino] args={vars(a)}")
    tr = np.load(a.train_npz, allow_pickle=True)
    te = np.load(a.test_npz, allow_pickle=True)
    Xtr = np.asarray(tr["feats"], np.float32)
    Xte = np.asarray(te["feats"], np.float32)
    log(f"[dino] train={Xtr.shape} test={Xte.shape}")
    pca = PCA(n_components=a.components, svd_solver="randomized",
              random_state=0)
    pca.fit(Xtr)
    Z = pca.transform(Xte).astype(np.float32)
    log(f"[dino] explained={float(pca.explained_variance_ratio_.sum()):.4f}")

    rows = np.full(len(Xte), -1, np.int64)
    ti = TableIndex(a.index_csv)
    pcol = "frame_path" if "frame_path" in ti.fieldnames else "rgb_path"
    lut = ti.lookup(pcol)
    paths = [str(p) for p in te["path"].tolist()]
    for i, p in enumerate(paths):
        if p in lut:
            rows[i] = lut[p]
    valid = rows >= 0
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    names = [f"dino_pca_{i}" for i in range(a.components)]
    np.savez_compressed(out, row_idx=rows[valid], feats=Z[valid],
                        valid=np.ones(int(valid.sum()), bool),
                        names=np.array(names, object))
    log(f"[dino] saved {out} valid={int(valid.sum())}/{len(valid)}")


if __name__ == "__main__":
    main()
