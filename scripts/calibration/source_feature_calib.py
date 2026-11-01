"""源域特征修正：在训练用户上学习 DINO 特征 -> 残差 的映射。

源域：当前 cm 模型在训练集上的预测 + DINO PCA 特征
目标：残差 r = label - pred，按会话中心化后，用特征预测 r
测试：每会话/朝向用 K 个标签估计偏置，再加固定的源域特征修正。
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


def fit_ridge(Z, y, lam):
    return np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ y)


def load_feats(path):
    z = np.load(path, allow_pickle=True)
    feats = z["feats"]
    if "path" in z:
        key = {str(p): i for i, p in enumerate(z["path"].tolist())
               if bool(z["valid"][i])}
    else:
        key = {int(r): i for i, r in enumerate(z["row_idx"].tolist())
               if bool(z["valid"][i])}
    return key, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-pred", required=True)
    ap.add_argument("--source-feat", required=True)
    ap.add_argument("--test-pred", required=True)
    ap.add_argument("--test-feat", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-list", type=int, nargs="+", default=[10, 20, 50])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    log(f"[src] args={vars(a)}")

    # metadata
    need = set()
    for f in (a.source_pred, a.test_pred):
        z = np.load(f, allow_pickle=True)
        need.update(str(p) for p in z["path"].tolist())
    log("[src] meta scan start")
    meta = {}
    path_to_row = {}
    with open(a.index_csv, encoding="utf-8") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            p = r["frame_path"]
            if p not in need:
                continue
            path_to_row[p] = i
            try:
                scm = np.array([float(r["screen_w"]) * float(r["cm_px_x"]),
                                float(r["screen_h"]) * float(r["cm_px_y"])])
                phi = float(np.arctan2(float(r["imu_x"]),
                                       -float(r["imu_y"])))
            except Exception:
                continue
            meta[p] = dict(scm=scm, phi=phi, ori=str(r.get("orientation", "")),
                           session=p.rsplit("/", 1)[0],
                           subject=r["subject"])

    log(f"[src] meta done={len(meta)}")
    sf_key, sf_arr = load_feats(a.source_feat)
    log(f"[src] source feat loaded={len(sf_key)}")
    tf_key, tf_arr = load_feats(a.test_feat)
    log(f"[src] test feat loaded={len(tf_key)}")

    zp = np.load(a.source_pred, allow_pickle=True)
    sp = zp["pred"].astype(np.float64); sl = zp["label"].astype(np.float64)
    spath = [str(p) for p in zp["path"].tolist()]
    sidx = [i for i, p in enumerate(spath)
            if p in meta and p in sf_key]
    log(f"[src] source joined={len(sidx)}")
    Xs = np.asarray(sf_arr[[sf_key[spath[i]] for i in sidx]], np.float64)
    Ps = sp[sidx]; Ls = sl[sidx]
    sess = [meta[spath[i]]["session"] for i in sidx]
    mu_x = Xs.mean(0); sd_x = Xs.std(0) + 1e-6
    Xs = np.clip((Xs - mu_x) / sd_x, -5, 5)
    Rs = Ls - Ps
    # remove per-session bias so W captures feature-dependent residual
    g = defaultdict(list)
    for j, s in enumerate(sess):
        g[s].append(j)
    for s, idx in g.items():
        idx = np.array(idx)
        Rs[idx] -= Rs[idx].mean(0)
    Zs = np.column_stack([Xs, np.ones(len(Xs))])
    W = [fit_ridge(Zs, Rs[:, ax], a.lam) for ax in range(2)]
    log(f"[src] source W norm={np.linalg.norm(W[0]):.4f},"
        f"{np.linalg.norm(W[1]):.4f}")

    tp = np.load(a.test_pred, allow_pickle=True)
    tp_pred = tp["pred"].astype(np.float64); tp_lab = tp["label"].astype(np.float64)
    tpath = [str(p) for p in tp["path"].tolist()]
    def tkey(p):
        if p in tf_key:
            return tf_key[p]
        r = path_to_row.get(p, -1)
        return tf_key.get(r, None)
    tidx = [i for i, p in enumerate(tpath) if p in meta and tkey(p) is not None]
    log(f"[src] test joined={len(tidx)}")
    P = tp_pred[tidx]; L = tp_lab[tidx]
    Xt = np.asarray(tf_arr[[tkey(tpath[i]) for i in tidx]], np.float64)
    Xt = np.clip((Xt - mu_x) / sd_x, -5, 5)
    Zt = np.column_stack([Xt, np.ones(len(Xt))])
    corr = np.stack([Zt @ W[0], Zt @ W[1]], 1)
    corr = np.clip(corr, -0.15, 0.15)
    groups = defaultdict(list)
    for j, i in enumerate(tidx):
        m = meta[tpath[i]]
        groups[(m["session"], m["ori"])].append(j)
    log(f"[src] test sessions={len(groups)}")

    def metrics(q, idx):
        cm = np.linalg.norm((q[idx] - L[idx]) * np.array(
            [meta[tpath[tidx[j]]]["scm"] for j in idx]), axis=1)
        nm = np.linalg.norm(q[idx] - L[idx], axis=1)
        return cm, nm

    out = {}
    for k in a.k_list:
        cs = []; ns = []; csb = []; nsb = []
        for seed in range(a.seeds):
            rng = np.random.RandomState(2000 + seed)
            cc, nn, cb, nb = [], [], [], []
            for key, idx in groups.items():
                idx = np.array(idx)
                if len(idx) < k + 10:
                    continue
                sup = rng.choice(idx, k, replace=False)
                rest = np.setdiff1d(idx, sup)
                b = np.median(L[sup] - P[sup], axis=0)
                q = P + b + corr
                qb = P + b
                c, n = metrics(q, rest); cc.append(c); nn.append(n)
                c, n = metrics(qb, rest); cb.append(c); nb.append(n)
            if cc:
                cs.append(np.concatenate(cc)); ns.append(np.concatenate(nn))
                csb.append(np.concatenate(cb)); nsb.append(np.concatenate(nb))
        if cs:
            c = np.concatenate(cs); n = np.concatenate(ns)
            cb = np.concatenate(csb); nb = np.concatenate(nsb)
            out[f"K{k}"] = dict(cm=float(c.mean()), norm=float(n.mean()),
                                bias_cm=float(cb.mean()),
                                bias_norm=float(nb.mean()),
                                n=int(len(c)))
            log(f"[src] RESULT K={k} srcfeat cm={c.mean():.4f} "
                f"norm={n.mean():.4f} | bias cm={cb.mean():.4f} "
                f"norm={nb.mean():.4f} n={len(c)}")
    o = Path(a.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    with open(o.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(dict(args=vars(a), results=out), fh, indent=2)
    log(f"[src] saved {o}.json")


if __name__ == "__main__":
    main()
