"""Low-rank adapter on the *base model's own features and head*.

The previous feature TTA used a newly trained 1536-d head; that head was
weaker than the base model and confounded the comparison.  Here we use:

    fused 384-d feature (model.norm output)  +  base head (Linear 384->2)

which reproduces the exported base prediction exactly (checked in code).
Only a low-rank adapter on the 384-d feature is fine-tuned per target session
on K labels; the base head stays frozen.
"""
import argparse
import copy
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path[:0] = ["src", "."]
from scripts.calibration.seq_bayes_calib import load_index, em  # noqa: E402
from scripts.calibration.meta_residual_calib import (  # noqa: E402
    fit_theta_huber, apply_theta_np, support_indices,
)


def log(msg):
    print(msg, flush=True)


class LowRank(nn.Module):
    def __init__(self, dim=384, rank=4):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(self.down(x))


def load_sessions_feat(npz_path, index_csv, context_key, dim=384):
    z = np.load(npz_path, allow_pickle=True)
    F = z["feats"][:, :dim].astype(np.float32)
    pred = z["pred"].astype(np.float64)
    label = z["label"].astype(np.float64)
    path = [str(p) for p in z["path"].tolist()]
    fidx = z["frame_index"].astype(np.int64)
    meta = load_index(index_csv, np.array(path, object))
    groups = defaultdict(list)
    for i, p in enumerate(path):
        if p not in meta:
            continue
        ctx = meta[p][4] if context_key != "none" else ""
        groups[f"{p.rsplit('/', 1)[0]}||{ctx}"].append(i)
    out = {}
    for s, idx in groups.items():
        idx = np.array(sorted(idx, key=lambda i: int(fidx[i])), np.int64)
        if len(idx) < 40:
            continue
        out[s] = dict(
            F=F[idx], pred=pred[idx], label=label[idx],
            sx=np.array([meta[path[i]][2] for i in idx]),
            sy=np.array([meta[path[i]][3] for i in idx]),
            ctx=np.array([meta[path[i]][4] for i in idx]),
        )
    log(f"[base-head] loaded {npz_path} sessions={len(out)} "
        f"frames={sum(len(d['F']) for d in out.values())} dim={F.shape[1]}")
    return out


def affine_prior(sessions):
    xs, ys = [], []
    for d in sessions.values():
        for ax in (0, 1):
            P = d["pred"][:, ax]
            U = d["label"][:, ax]
            X = np.stack([P, np.ones(len(P))], 1)
            th = np.linalg.solve(X.T @ X + 1e-6 * np.eye(2), X.T @ U)
            (xs if ax == 0 else ys).append(th)
    return np.concatenate([np.mean(xs, 0), np.mean(ys, 0)])


def predict(head, adapter, F, device):
    with torch.no_grad():
        x = torch.tensor(F, dtype=torch.float32, device=device)
        return head(adapter(x)).cpu().numpy()


def eval_huber_single(d, pred_all, sel, mu, lam, delta):
    q = np.setdiff1d(np.arange(len(d["pred"])), sel)
    if len(q) < 20:
        return None
    th, _ = fit_theta_huber(pred_all[sel], d["label"][sel], lam, mu, delta)
    q1 = apply_theta_np(pred_all[q], th)
    return float(em(q1, d["label"][q], d["sx"][q], d["sy"][q]).mean())


def adapt_session(d, head, src_adapter, K, device, args, sel=None):
    if sel is None:
        sel = support_indices(d, K, np.random.RandomState(90000))
    adapter = copy.deepcopy(src_adapter).to(device)
    src = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.adapt_lr, weight_decay=0.0)
    F = torch.tensor(d["F"], dtype=torch.float32, device=device)
    Y = torch.tensor(d["label"][sel], dtype=torch.float32, device=device)
    sx = torch.tensor(d["sx"][sel], dtype=torch.float32, device=device)
    sy = torch.tensor(d["sy"][sel], dtype=torch.float32, device=device)
    head.eval()
    for _ in range(args.adapt_steps):
        adapter.train()
        out = adapter(F)
        q = head(out[sel])
        loss = torch.sqrt(((q[:, 0] - Y[:, 0]) * sx) ** 2 +
                          ((q[:, 1] - Y[:, 1]) * sy) ** 2).mean()
        anchor = sum(((v - src[k]) ** 2).mean()
                     for k, v in adapter.state_dict().items())
        preserve = ((out - F) ** 2).mean()
        total = loss + args.anchor * anchor + args.preserve * preserve
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
    adapter.eval()
    pred = predict(head, adapter, d["F"], device)
    return pred, sel, float(adapter.up.weight.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--val-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--context-key", choices=["none", "orientation", "activity"], default="none")
    ap.add_argument("--K", type=int, default=9)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1.345)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--adapt-steps", type=int, default=100)
    ap.add_argument("--adapt-lr", type=float, default=1e-3)
    ap.add_argument("--anchor", type=float, default=1e-2)
    ap.add_argument("--preserve", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=-1)
    a = ap.parse_args()

    torch.set_num_threads(4)
    device = "cuda" if (a.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    log(f"[base-head] start={time.strftime('%F %T')} device={device} args={vars(a)}")
    D = load_sessions_feat(a.train_npz, a.index_csv, a.context_key)
    T = load_sessions_feat(a.test_npz, a.index_csv, a.context_key)
    mu_aff = affine_prior(D)
    log(f"[base-head] affine_prior={np.round(mu_aff,4).tolist()}")

    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    W = st["head.weight"].float()
    b = st["head.bias"].float()
    head = nn.Linear(W.shape[1], W.shape[0]).to(device)
    with torch.no_grad():
        head.weight.copy_(W.to(device))
        head.bias.copy_(b.to(device))
    head.eval()
    adapter = LowRank(W.shape[1], a.rank).to(device)
    adapter.eval()
    F0 = torch.tensor(D[list(D)[0]]["F"], dtype=torch.float32, device=device)
    with torch.no_grad():
        q0 = head(adapter(F0)).cpu().numpy()
    base0 = D[list(D)[0]]["pred"]
    log(f"[base-head] check head+features vs exported pred: "
        f"mean_abs_diff={np.abs(q0-base0).mean():.8f} "
        f"max_abs_diff={np.abs(q0-base0).max():.8f}")

    rows = []
    session_stats = []
    acc_base = acc_ada = 0.0
    nsum = 0
    t0 = time.time()
    for si, (s, d) in enumerate(T.items()):
        n = len(d["pred"])
        base_vals, ada_vals = [], []
        base_direct = ada_direct = 0.0
        dn = 0.0
        for seed in range(a.seeds):
            rng = np.random.RandomState(90000 + seed)
            sel = support_indices(d, a.K, rng)
            b = eval_huber_single(d, d["pred"], sel, mu_aff, a.lam, a.delta)
            if b is None:
                continue
            ada_pred, _, dn = adapt_session(d, head, adapter, a.K, device, a,
                                             sel=sel)
            aa = eval_huber_single(d, ada_pred, sel, mu_aff, a.lam, a.delta)
            if aa is None:
                continue
            base_vals.append(b)
            ada_vals.append(aa)
            base_direct += float(em(d["pred"], d["label"], d["sx"], d["sy"]).mean())
            ada_direct += float(em(ada_pred, d["label"], d["sx"], d["sy"]).mean())
        if not base_vals:
            continue
        bmean = float(np.mean(base_vals))
        amean = float(np.mean(ada_vals))
        acc_base += bmean * n
        acc_ada += amean * n
        nsum += n
        session_stats.append((s, n, bmean, amean))
        rows.append([s, n, bmean, amean, 100 * (amean / bmean - 1),
                     base_direct / len(base_vals), ada_direct / len(ada_vals), dn])
        if si % 5 == 0 or si == len(T) - 1:
            log(f"[base-head] session={s} n={n} direct(base/ada)="
                f"{base_direct/len(base_vals):.4f}/{ada_direct/len(ada_vals):.4f} "
                f"cal={bmean:.4f}/{amean:.4f} ({100*(amean/bmean-1):+.2f}%) "
                f"adapter_up={dn:.3f} elapsed={time.time()-t0:.1f}s")
    B, A = acc_base / nsum, acc_ada / nsum
    ratios = np.array([x[3] / max(x[2], 1e-12) - 1 for x in session_stats])
    deltas = np.array([x[3] - x[2] for x in session_stats])
    rng = np.random.RandomState(12345)
    boot_r, boot_d = [], []
    for _ in range(a.bootstrap):
        idx = rng.choice(len(session_stats), len(session_stats), replace=True)
        w = np.array([session_stats[j][1] for j in idx], float)
        boot_r.append(float(np.sum(ratios[idx] * w) / w.sum()))
        boot_d.append(float(np.sum(deltas[idx] * w) / w.sum()))
    boot_r = np.array(boot_r)
    boot_d = np.array(boot_d)
    lo, hi = np.percentile(boot_r, [2.5, 97.5])
    p_improve = float(np.mean(boot_d >= 0.0))
    log(f"[base-head] RESULT K={a.K} seeds={a.seeds} base_cal={B:.4f} "
        f"adapter_cal={A:.4f} adapter_vs_base={100*(A/B-1):+.2f}% "
        f"session_ratio_mean={100*ratios.mean():+.2f}% "
        f"boot95=[{100*lo:+.2f},{100*hi:+.2f}]% "
        f"p(delta>=0)={p_improve:.4f} "
        f"improved_sessions={int((ratios<0).sum())}/{len(ratios)}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "n", "base_cal", "ada_cal", "pct",
                    "base_direct", "ada_direct", "adapter_up"])
        w.writerows(rows)
    log(f"[base-head] saved {out}")


if __name__ == "__main__":
    main()
