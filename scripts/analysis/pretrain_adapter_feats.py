"""在特征级预训练 adapter（源域标签）。

pred' = pred + (up(down(h)) + b) @ W^T
只训练 adapter（bias/down/up），主干与头部冻结。

用法：
  python -m scripts.analysis.pretrain_adapter_feats \
    --ckpt out/.../best.pt --feats out/.../feats_gc_train300.npz \
    --index-csv data/GazeCapture/index/index.csv --out .../adapter_gc.pt
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = ["src", "."]


def load_scale(index_csv, needed):
    need = set(needed.tolist())
    out = {}
    with open(index_csv, encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        pcol = "frame_path" if "frame_path" in rd.fieldnames else "rgb_path"
        for r in rd:
            p = r[pcol]
            if p in need:
                out[p] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                          float(r["screen_h"]) * float(r["cm_px_y"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--val-feats", default="")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-reg", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    z = np.load(a.feats, allow_pickle=True)
    h = torch.from_numpy(z["feats"].astype(np.float32)[:, :384])
    lab = torch.from_numpy(z["label"].astype(np.float32))
    pred0 = z["pred"].astype(np.float32)
    path = z["path"]
    scale = load_scale(a.index_csv, path)
    print(f"[pre] frames={len(h)} mapped_scale={len(scale)} rank={a.rank}")

    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = st.get("model", st)
    W = sd["head.weight"].float().to(a.device)
    print(f"[pre] head W={tuple(W.shape)}")

    bias = torch.zeros(384, device=a.device, requires_grad=True)
    down = torch.nn.Parameter(torch.randn(a.rank, 384, device=a.device) * 0.02)
    up = torch.nn.Parameter(torch.zeros(384, a.rank, device=a.device))
    opt = torch.optim.Adam([bias, down, up], lr=a.lr, weight_decay=a.w_reg)
    hd = h.to(a.device)
    ld = lab.to(a.device)
    P0 = torch.from_numpy(pred0).to(a.device)

    def preds_tensor(idx):
        hh = hd[idx]
        d = torch.nn.functional.linear(hh, down)
        d = torch.nn.functional.linear(d, up) + bias
        return P0[idx] + (d @ W.T)

    def em(p, idx):
        p = p.detach().cpu().numpy()
        out = []
        for j, i in enumerate(idx.tolist()):
            pth = path[i]
            if pth not in scale:
                continue
            sx, sy = scale[pth]
            out.append(np.hypot((p[j, 0] - lab[i, 0].item()) * sx,
                                (p[j, 1] - lab[i, 1].item()) * sy))
        return float(np.mean(out)) if out else float("nan")

    n = len(h)
    rng = np.random.RandomState(0)
    for ep in range(a.epochs):
        perm = torch.from_numpy(rng.permutation(n))
        tot = 0.0; nb = 0
        for b in range(0, n, a.batch_size):
            idx = perm[b:b + a.batch_size]
            pred = preds_tensor(idx)
            loss = torch.nn.functional.mse_loss(pred, ld[idx])
            opt.zero_grad(); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_([bias, down, up], 1.0)
            opt.step()
            tot += float(loss); nb += 1
        if (ep + 1) % 2 == 0 or ep == 0:
            all_idx = torch.arange(n, device=a.device)
            base_em = em(P0, all_idx)
            new_em = em(preds_tensor(all_idx), all_idx)
            print(f"[pre] ep{ep+1}/{a.epochs} loss={tot/max(nb,1):.6f} "
                  f"base_EM={base_em:.4f} new_EM={new_em:.4f} "
                  f"({100*(new_em/base_em-1):+.2f}%) "
                  f"|down|={float(down.norm()):.4f} |up|={float(up.norm()):.4f} "
                  f"|b|={float(bias.norm()):.4f} grad={float(gn):.4f}")
    state = {"bias": bias.detach().cpu(), "down.weight": down.detach().cpu(),
             "up.weight": up.detach().cpu()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, a.out)
    print(f"[pre] saved {a.out} norm={sum(float(v.norm()) for v in state.values()):.6f}")


if __name__ == "__main__":
    main()
