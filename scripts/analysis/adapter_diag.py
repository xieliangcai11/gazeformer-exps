"""诊断 adapter 初始化到底学到了什么（特征级，完全可复现）。

adapter 作用在 head 之前：h' = h + up(down(h)) + bias
head 是线性的：pred = W h + b
所以 pred' = pred + (up(down(h)) + bias) @ W^T
不需要跑主干，直接用 dump 的 fused 特征即可。

输出：
- 零 adapter 与加载 adapter 的 EM；
- 每会话的平移/斜率/残差分解；
- 会话间修正方向的一致性；
- 与误差改善的相关性。

用法：
  python -m scripts.analysis.adapter_diag \
    --ckpt out/.../best.pt \
    --feats out/.../feats_gc_test_m2_full.npz \
    --index-csv data/GazeCapture/index/index.csv \
    --adapter out/.../pretrain_adapter_rgd.pt
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = ["src", "."]


def load_index(index_csv, needed):
    need = set(needed.tolist())
    meta = {}
    with open(index_csv, encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        pcol = "frame_path" if "frame_path" in rd.fieldnames else "rgb_path"
        for r in rd:
            p = r[pcol]
            if p not in need:
                continue
            sx = float(r["screen_w"]) * float(r["cm_px_x"])
            sy = float(r["screen_h"]) * float(r["cm_px_y"])
            meta[p] = (sx, sy, r.get("orientation", r.get("activity", "")))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--adapter", default="", help="空=零 adapter 自检")
    ap.add_argument("--out-csv", default="")
    a = ap.parse_args()

    z = np.load(a.feats, allow_pickle=True)
    h = z["feats"].astype(np.float32)[:, :384]
    pred0 = z["pred"].astype(np.float32)
    label = z["label"].astype(np.float32)
    path = z["path"]
    meta = load_index(a.index_csv, path)
    print(f"[diag] frames={len(h)} mapped_meta={len(meta)}")

    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = st.get("model", st)
    W = sd["head.weight"].float()

    if a.adapter:
        ast = torch.load(a.adapter, map_location="cpu")
        b = ast["bias"].float()
        down = ast["down.weight"].float()
        up = ast["up.weight"].float()
        print(f"[diag] adapter rank={down.shape[0]} norm="
              f"{sum(float(v.norm()) for v in ast.values()):.6f}")
        H = torch.from_numpy(h)
        delta = (torch.nn.functional.linear(H, down))
        delta = torch.nn.functional.linear(delta, up) + b
        pred1 = pred0 + (delta @ W.T).numpy()
    else:
        pred1 = pred0.copy()
        print("[diag] 零 adapter 自检")

    def em(p):
        out = np.full(len(p), np.nan, np.float32)
        for i in range(len(p)):
            if path[i] not in meta:
                continue
            sx, sy, _ = meta[path[i]]
            out[i] = np.hypot((p[i, 0] - label[i, 0]) * sx,
                              (p[i, 1] - label[i, 1]) * sy)
        return out

    e0, e1 = em(pred0), em(pred1)
    ok = np.isfinite(e0) & np.isfinite(e1)
    print(f"[diag] EM base={e0[ok].mean():.4f} -> adapter={e1[ok].mean():.4f} "
          f"({100*(e1[ok].mean()/e0[ok].mean()-1):+.2f}%) n={ok.sum()}")

    # 会话级分解
    sess = defaultdict(list)
    for i in range(len(h)):
        if not ok[i]:
            continue
        s = path[i].rsplit("/", 1)[0]
        sess[s].append(i)
    rows = []
    cos = []
    for s, idx in sess.items():
        idx = np.array(idx)
        d = pred1[idx] - pred0[idx]
        md = d.mean(0)
        # 斜率：delta ~ a * pred + c（逐轴）
        slopes = []
        for ax in range(2):
            X = np.stack([pred0[idx, ax], np.ones(len(idx))], 1)
            coef = np.linalg.lstsq(X, d[:, ax], rcond=None)[0]
            slopes.append(coef[0])
        rows.append((s, len(idx), float(e0[idx].mean()), float(e1[idx].mean()),
                     float(md[0]), float(md[1]), float(np.linalg.norm(md)),
                     float(slopes[0]), float(slopes[1])))
        cos.append(md)
    R = np.array([[r[4], r[5]] for r in rows])
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
    if len(Rn) > 1:
        cc = Rn @ Rn.T
        print(f"[diag] 会话平均平移之间的余弦相似度："
              f"mean={cc[np.triu_indices(len(cc),1)].mean():+.3f} "
              f"median={np.median(cc[np.triu_indices(len(cc),1)]):+.3f}")
    print(f"[diag] 每会话平均平移 norm：mean={np.linalg.norm(R,axis=1).mean():.5f} "
          f"std={np.linalg.norm(R,axis=1).std():.5f}")
    print(f"[diag] 每会话斜率：sx mean={np.mean([r[7] for r in rows]):+.4f} "
          f"sy mean={np.mean([r[8] for r in rows]):+.4f}")
    imp = np.array([r[3] / r[2] - 1 for r in rows])
    print(f"[diag] 会话改善：mean={100*imp.mean():+.2f}% "
          f"改善 {int((imp<0).sum())}/{len(imp)} 最差 {100*imp.max():+.1f}%")
    corr = np.corrcoef([r[6] for r in rows], imp)[0, 1]
    print(f"[diag] 平移幅度 vs 改善相关：{corr:+.3f}")

    if a.out_csv:
        Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["session", "n", "em_base", "em_adapted", "dx", "dy",
                        "d_norm", "slope_x", "slope_y"])
            w.writerows(rows)
        print(f"[diag] saved {a.out_csv}")


if __name__ == "__main__":
    main()
