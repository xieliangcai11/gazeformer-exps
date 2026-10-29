"""Huber + meta-learned nonlinear residual head.

Pipeline:
  q0 = Huber_theta(p)                      (K support labels)
  q  = q0 + g_phi(p, pose, screen, context)

The residual head g is meta-trained on source sessions (Reptile-style inner
adaptation on K support labels, outer update on query labels) and fine-tuned
per target session on its own K support labels.  g is zero-initialized, so the
initial model is exactly the Huber baseline; a fallback to Huber is always
possible.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path[:0] = ["src", "."]
from scripts.calibration.seq_bayes_calib import load_sessions, em  # noqa: E402
from scripts.calibration.meta_residual_calib import (  # noqa: E402
    fit_source_mu, fit_theta_huber, apply_theta_np, support_indices,
)


def log(msg):
    print(msg, flush=True)


class Head(nn.Module):
    def __init__(self, dim, hidden=64, layers=2):
        super().__init__()
        mods = []
        d = dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        mods += [nn.Linear(d, 2)]
        self.net = nn.Sequential(*mods)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def build_features(d, ctx_map):
    pose = d["pose"].astype(np.float32)
    p = d["pred"].astype(np.float32)
    w = d["w"].astype(np.float32)[:, None]
    h = d["h"].astype(np.float32)[:, None]
    ctx = np.array([ctx_map.get(str(c), 0) for c in d["ctx"]], np.int64)
    ctx_oh = np.zeros((len(ctx), len(ctx_map)), np.float32)
    ctx_oh[np.arange(len(ctx)), ctx] = 1.0
    r = np.hypot(p[:, 0] - 0.5, p[:, 1] - 0.5)[:, None]
    return np.concatenate([p, pose, w / 10.0, h / 15.0, r, ctx_oh], 1)


def standardize(F, mu, sd):
    return (F - mu) / sd


def residual_loss(head, F, R):
    return ((head(F) - R) ** 2).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--val-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--pose-npz", required=True)
    ap.add_argument("--context-key", choices=["none", "orientation", "activity"], default="none")
    ap.add_argument("--K", type=int, default=52)
    ap.add_argument("--query", type=int, default=32)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=1.345)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--inner-steps", type=int, default=3)
    ap.add_argument("--inner-lr", type=float, default=0.05)
    ap.add_argument("--meta-lr", type=float, default=0.1)
    ap.add_argument("--target-steps", type=int, default=5)
    ap.add_argument("--target-lr", type=float, default=0.05)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=-1)
    a = ap.parse_args()

    torch.set_num_threads(4)
    device = "cuda" if (a.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    log(f"[head] start={time.strftime('%F %T')} device={device} args={vars(a)}")
    D = load_sessions(a.train_npz, a.index_csv, a.pose_npz, a.context_key)
    S = load_sessions(a.val_npz if a.split == "val" else a.test_npz,
                      a.index_csv, a.pose_npz, a.context_key)
    mu_aff, _ = fit_source_mu(D)
    ctx_names = sorted({str(d["ctx"][0]) for d in D.values()})
    ctx_map = {c: i for i, c in enumerate(ctx_names)}
    log(f"[head] source={len(D)} eval={len(S)} K={a.K} "
        f"ctx={ctx_names} prior={np.round(mu_aff,4).tolist()}")

    Fall = np.concatenate([build_features(d, ctx_map) for d in D.values()])
    fmu, fsd = Fall.mean(0), Fall.std(0) + 1e-6
    dim = Fall.shape[1]
    head = Head(dim, a.hidden, a.layers).to(device)
    log(f"[head] feat_dim={dim} params={sum(p.numel() for p in head.parameters())}")

    # ---- Meta-training (FOMAML) ----
    keys = list(D)
    for ep in range(1, a.epochs + 1):
        rng = np.random.RandomState(1000 + ep)
        rng.shuffle(keys)
        tot_s = tot_q = 0.0
        nb = 0
        meta_opt = torch.optim.Adam(head.parameters(), lr=a.meta_lr)
        t0 = time.time()
        for s in keys:
            d = D[s]
            n = len(d["pred"])
            if n < a.K + a.query:
                continue
            idx = rng.choice(n, a.K + a.query, replace=False)
            sup, qry = idx[:a.K], idx[a.K:]
            th, _ = fit_theta_huber(d["pred"][sup], d["label"][sup],
                                    a.lam, mu_aff, a.delta)
            F = torch.tensor(standardize(build_features(d, ctx_map), fmu, fsd),
                             dtype=torch.float32, device=device)
            R = torch.tensor(d["label"] - apply_theta_np(d["pred"], th),
                             dtype=torch.float32, device=device)
            names = [n for n, _ in head.named_parameters()]
            phi = [p.detach().clone().requires_grad_(True) for p in head.parameters()]
            # inner: first-order support adaptation
            for _ in range(a.inner_steps):
                out = torch.func.functional_call(head, dict(zip(names, phi)), (F[sup],))
                loss = ((out - R[sup]) ** 2).mean()
                g = torch.autograd.grad(loss, phi, create_graph=False)
                phi = [(p - a.inner_lr * gg).detach().requires_grad_(True)
                       for p, gg in zip(phi, g)]
            # outer: first-order FOMAML gradient at adapted parameters
            out_q = torch.func.functional_call(head, dict(zip(names, phi)), (F[qry],))
            loss_q = ((out_q - R[qry]) ** 2).mean()
            if not torch.isfinite(loss_q):
                log(f"[head] WARNING non-finite loss task={s}; skip")
                continue
            gq = torch.autograd.grad(loss_q, phi)
            meta_opt.zero_grad(set_to_none=True)
            for p, gg in zip(head.parameters(), gq):
                p.grad = gg.detach().clone()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            meta_opt.step()
            tot_s += float(loss.detach())
            tot_q += float(loss_q.detach())
            nb += 1
            if nb % 10 == 0 or nb == len(keys):
                log(f"[head] meta_ep={ep} task={s} sup_loss={float(loss):.6f} "
                    f"qry_loss={float(loss_q):.6f} elapsed={time.time()-t0:.1f}s")
        log(f"[head] meta_epoch={ep}/{a.epochs} tasks={nb} "
            f"sup_loss={tot_s/max(nb,1):.6f} qry_loss={tot_q/max(nb,1):.6f}")

    # ---- Target adaptation + evaluation ----
    def run(d, sel):
        q = np.setdiff1d(np.arange(len(d["pred"])), sel)
        if len(q) < 20:
            return None
        th, _ = fit_theta_huber(d["pred"][sel], d["label"][sel],
                                a.lam, mu_aff, a.delta)
        F = torch.tensor(standardize(build_features(d, ctx_map), fmu, fsd),
                         dtype=torch.float32, device=device)
        R = torch.tensor(d["label"] - apply_theta_np(d["pred"], th),
                         dtype=torch.float32, device=device)
        phi = [p.detach().clone().requires_grad_(True) for p in head.parameters()]
        for _ in range(a.target_steps):
            out = torch.func.functional_call(head, dict(zip(
                [n for n, _ in head.named_parameters()], phi)), (F[sel],))
            loss = ((out - R[sel]) ** 2).mean()
            g = torch.autograd.grad(loss, phi)
            phi = [p - a.target_lr * gg for p, gg in zip(phi, g)]
        with torch.no_grad():
            out = torch.func.functional_call(head, dict(zip(
                [n for n, _ in head.named_parameters()], phi)), (F[q],))
        q0 = apply_theta_np(d["pred"][q], th)
        q1 = q0 + out.cpu().numpy()
        e0 = float(em(q0, d["label"][q], d["sx"][q], d["sy"][q]).mean())
        e1 = float(em(q1, d["label"][q], d["sx"][q], d["sy"][q]).mean())
        return e0, e1, len(q), float(loss.detach())

    rows = []
    acc0 = acc1 = nn = 0.0
    ratios = []
    t0 = time.time()
    for si, (s, d) in enumerate(S.items()):
        r0 = r1 = rn = 0.0
        for seed in range(a.seeds):
            sel = support_indices(d, a.K, np.random.RandomState(90000 + seed))
            r = run(d, sel)
            if r is None:
                continue
            e0, e1, n, loss = r
            r0 += e0 * n
            r1 += e1 * n
            rn += n
        if rn:
            b, e = r0 / rn, r1 / rn
            acc0 += b * rn
            acc1 += e * rn
            nn += rn
            ratios.append(e / max(b, 1e-12) - 1)
            rows.append([s, rn, b, e, 100 * (e / b - 1)])
            if si % 5 == 0 or si == len(S) - 1:
                log(f"[head] session={s} n={rn:.0f} base={b:.4f} "
                    f"head={e:.4f} ({100*(e/b-1):+.2f}%) "
                    f"elapsed={time.time()-t0:.1f}s")
    B, E = acc0 / nn, acc1 / nn
    rr = np.array(ratios)
    log(f"[head] RESULT K={a.K} base={B:.4f} head={E:.4f} "
        f"({100*(E/B-1):+.2f}%) sessions={len(rr)} "
        f"improved={int((rr<0).sum())} worst={100*rr.max():+.1f}%")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "n", "base", "head", "pct"])
        w.writerows(rows)
    log(f"[head] saved {out}")


if __name__ == "__main__":
    main()
