"""Supervised low-rank adapter + K-shot calibration.

Feature-level adaptation with *target supervision* (K labels), not pure
unlabeled TTA:

  source:  F -> F + U V^T F -> head -> 2D
  target:  freeze head, fine-tune only the low-rank adapter on K labels,
           with an anchor to the source adapter and an optional
           feature-preservation term on unlabeled target frames.

The target adapter never sees target labels outside the K support set.
Logs include source/target EM, adapter parameter norms, per-session results,
and the usual memory/time fields.
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


class Head(nn.Module):
    def __init__(self, dim, hidden=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


class LowRankAdapter(nn.Module):
    """F' = F + U (V^T F); zero-initialized to identity."""

    def __init__(self, dim=1536, rank=4):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(self.down(x))

    def delta(self):
        return self.up.weight @ self.down.weight


def load_feat_sessions(npz_path, index_csv, context_key):
    z = np.load(npz_path, allow_pickle=True)
    F = z["feats"].astype(np.float32)
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
    log(f"[adapter] loaded {npz_path} sessions={len(out)} frames={sum(len(d['F']) for d in out.values())}")
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


def em_torch(q, y, sx, sy):
    return torch.sqrt(((q[:, 0] - y[:, 0]) * sx) ** 2 +
                      ((q[:, 1] - y[:, 1]) * sy) ** 2).mean()


def train_source(train, val, device, args):
    Fall = np.concatenate([d["F"] for d in train.values()])
    mu = Fall.mean(0)
    sd = Fall.std(0) + 1e-6
    Y = np.concatenate([d["label"] for d in train.values()])
    F = torch.tensor((Fall - mu) / sd, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    Fv = torch.tensor((np.concatenate([d["F"] for d in val.values()]) - mu) / sd,
                      dtype=torch.float32).to(device)
    Yv = torch.tensor(np.concatenate([d["label"] for d in val.values()]),
                      dtype=torch.float32).to(device)
    model = nn.ModuleDict({
        "adapter": LowRankAdapter(F.shape[1], args.rank),
        "head": Head(F.shape[1], args.hidden, args.dropout),
    }).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(F))
        tot = n = 0.0
        for st in range(0, len(perm), args.batch):
            idx = perm[st:st + args.batch]
            x = F[idx].to(device)
            y = Yt[idx].to(device)
            q = model["head"](model["adapter"](x))
            loss = ((q - y) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
            n += len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            qv = model["head"](model["adapter"](Fv))
            ev = torch.sqrt(((qv - Yv) ** 2).sum(1)).mean()
        if float(ev) < best:
            best = float(ev)
            best_state = copy.deepcopy(model.state_dict())
        log(f"[adapter] src_epoch={ep:03d}/{args.epochs} loss={tot/n:.6f} "
            f"val_EM={float(ev):.4f} lr={sched.get_last_lr()[0]:.2e}")
    model.load_state_dict(best_state)
    model.eval()
    log(f"[adapter] source best_val_EM={best:.4f} "
        f"delta_norm={float(model['adapter'].delta().norm()):.4f}")
    return model, mu, sd, best


def predict(model, F, mu, sd, device):
    x = torch.tensor((F - mu) / sd, dtype=torch.float32, device=device)
    with torch.no_grad():
        return model["head"](model["adapter"](x)).cpu().numpy()


def eval_huber(d, pred_all, K, mu_aff, lam, delta, seeds):
    acc = nn = 0.0
    for seed in range(seeds):
        rng = np.random.RandomState(90000 + seed)
        sel = support_indices(d, K, rng)
        q = np.setdiff1d(np.arange(len(d["pred"])), sel)
        if len(q) < 20:
            continue
        th, _ = fit_theta_huber(pred_all[sel], d["label"][sel], lam, mu_aff, delta)
        q1 = apply_theta_np(pred_all[q], th)
        acc += em(q1, d["label"][q], d["sx"][q], d["sy"][q]).mean() * len(q)
        nn += len(q)
    return acc / max(nn, 1)


def eval_direct(d, pred_all):
    return float(em(pred_all, d["label"], d["sx"], d["sy"]).mean())


def adapt_target(session, model, mu, sd, K, device, args):
    """Fine-tune a copy of the adapter on K support labels."""
    d = session
    rng = np.random.RandomState(90000)
    sel = support_indices(d, K, rng)
    Fn = torch.tensor((d["F"] - mu) / sd, dtype=torch.float32, device=device)
    Ysup = torch.tensor(d["label"][sel], dtype=torch.float32, device=device)
    sx = torch.tensor(d["sx"][sel], dtype=torch.float32, device=device)
    sy = torch.tensor(d["sy"][sel], dtype=torch.float32, device=device)
    adapter = copy.deepcopy(model["adapter"]).to(device)
    src_state = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.adapt_lr, weight_decay=0.0)
    head = model["head"]
    for it in range(args.adapt_steps):
        adapter.train()
        out = adapter(Fn)
        pred = head(out[sel])
        loss = em_torch(pred, Ysup, sx, sy)
        # Anchor to source adapter.
        anchor = 0.0
        for k, v in adapter.state_dict().items():
            anchor = anchor + ((v - src_state[k]) ** 2).mean()
        # Preserve unlabeled feature geometry (all target frames, no labels).
        preserve = ((out - Fn) ** 2).mean()
        total = loss + args.anchor * anchor + args.preserve * preserve
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
    adapter.eval()
    with torch.no_grad():
        out_all = adapter(Fn)
        pred_all = head(out_all).cpu().numpy()
    return pred_all, sel, float(adapter.delta().norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--val-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--context-key", choices=["none", "orientation", "activity"], default="none")
    ap.add_argument("--K", type=int, default=9)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1.345)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--adapt-steps", type=int, default=50)
    ap.add_argument("--adapt-lr", type=float, default=1e-3)
    ap.add_argument("--anchor", type=float, default=1e-2)
    ap.add_argument("--preserve", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=-1)
    a = ap.parse_args()

    torch.set_num_threads(4)
    device = "cuda" if (a.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    log(f"[adapter] start={time.strftime('%F %T')} device={device} args={vars(a)}")
    D = load_feat_sessions(a.train_npz, a.index_csv, a.context_key)
    V = load_feat_sessions(a.val_npz, a.index_csv, a.context_key)
    T = load_feat_sessions(a.test_npz, a.index_csv, a.context_key)
    mu_aff = affine_prior(D)
    log(f"[adapter] affine_prior={np.round(mu_aff,4).tolist()}")

    model, mu, sd, val_em = train_source(D, V, device, a)
    log(f"[adapter] target sessions={len(T)} features={list(T.values())[0]['F'].shape}")

    rows = []
    acc_base = acc_src = acc_ada = 0.0
    nn = 0
    t0 = time.time()
    for si, (s, d) in enumerate(T.items()):
        base_direct = eval_direct(d, d["pred"])
        src_pred = predict(model, d["F"], mu, sd, device)
        src_direct = eval_direct(d, src_pred)
        ada_pred, sel, dn = adapt_target(d, model, mu, sd, a.K, device, a)
        ada_direct = eval_direct(d, ada_pred)
        base_cal = eval_huber(d, d["pred"], a.K, mu_aff, a.lam, a.delta, a.seeds)
        src_cal = eval_huber(d, src_pred, a.K, mu_aff, a.lam, a.delta, a.seeds)
        ada_cal = eval_huber(d, ada_pred, a.K, mu_aff, a.lam, a.delta, a.seeds)
        n = len(d["pred"])
        acc_base += base_cal * n
        acc_src += src_cal * n
        acc_ada += ada_cal * n
        nn += n
        rows.append([s, base_direct, src_direct, ada_direct,
                     base_cal, src_cal, ada_cal, dn, len(sel)])
        if si % 5 == 0 or si == len(T) - 1:
            log(f"[adapter] session={s} n={n} direct(base/src/ada)="
                f"{base_direct:.4f}/{src_direct:.4f}/{ada_direct:.4f} "
                f"cal={base_cal:.4f}/{src_cal:.4f}/{ada_cal:.4f} "
                f"adapter_delta={dn:.3f} elapsed={time.time()-t0:.1f}s")
    B, S, A = acc_base / nn, acc_src / nn, acc_ada / nn
    log(f"[adapter] RESULT K={a.K} base_cal={B:.4f} source_cal={S:.4f} "
        f"target_adapter_cal={A:.4f} src_vs_base={100*(S/B-1):+.2f}% "
        f"ada_vs_base={100*(A/B-1):+.2f}% ada_vs_src={100*(A/S-1):+.2f}%")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "base_direct", "src_direct", "ada_direct",
                    "base_cal", "src_cal", "ada_cal", "ada_delta", "K_used"])
        w.writerows(rows)
    log(f"[adapter] saved {out}")


if __name__ == "__main__":
    main()
