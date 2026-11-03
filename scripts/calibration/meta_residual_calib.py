"""Route 1: residual meta-calibration.

Huber affine calibration is a strong, low-variance baseline for K calibration
points.  A plain sequence model that predicts absolute calibration parameters
can overfit the source subjects and be worse than that baseline.  This script
therefore predicts only a *small residual* on top of the Huber affine fit:

    theta = theta_huber + max_res * tanh(net(support_pairs, context))

A zero-initialized head makes the initial model exactly the Huber baseline, and
the residual is bounded.  The model is trained on source-session episodes and
selected on source validation sessions; test sessions are never used for model
selection.

All logs include shapes, data ranges, model size, per-epoch loss/residual norm,
per-session validation/test results, aggregate frame-weighted EM, and memory.
"""
import argparse
import copy
import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

try:
    import resource
except ImportError:  # Windows
    resource = None

import numpy as np
import torch
import torch.nn as nn

sys.path[:0] = ["src", "."]
from scripts.calibration.seq_bayes_calib import load_sessions, em  # noqa: E402
from scripts.calibration.cc_eb import select_coverage  # noqa: E402
from scripts.calibration.robust_calib import fit_huber, fit_ridge  # noqa: E402


def maxrss_gb():
    if resource is None:
        return 0.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def log(msg):
    print(msg, flush=True)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_theta_huber(pred, label, lam, mu, delta):
    """Per-axis Huber affine fit; returns theta=[sx,bx,sy,by]."""
    tx, wx = fit_huber(pred[:, 0], label[:, 0], lam, np.asarray(mu[0:2], np.float64), delta)
    ty, wy = fit_huber(pred[:, 1], label[:, 1], lam, np.asarray(mu[2:4], np.float64), delta)
    return np.array([tx[0], tx[1], ty[0], ty[1]], np.float64), (wx, wy)


def fit_source_mu(sessions):
    """Source affine parameters, used as the ridge prior for target sessions."""
    thetas = []
    for d in sessions.values():
        tx = fit_ridge(d["pred"][:, 0], d["label"][:, 0], 1e-6, 0.0)
        ty = fit_ridge(d["pred"][:, 1], d["label"][:, 1], 1e-6, 0.0)
        thetas.append(np.concatenate([tx, ty]))
    th = np.asarray(thetas, np.float64)
    return th.mean(0), th


def apply_theta_np(pred, theta):
    q = pred.astype(np.float64, copy=True)
    q[:, 0] = theta[0] * q[:, 0] + theta[1]
    q[:, 1] = theta[2] * q[:, 1] + theta[3]
    return q


def apply_theta_torch(pred, theta):
    """pred [B,M,2], theta [B,4] -> [B,M,2]."""
    q = pred.clone()
    q[..., 0] = theta[:, None, 0] * pred[..., 0] + theta[:, None, 1]
    q[..., 1] = theta[:, None, 2] * pred[..., 1] + theta[:, None, 3]
    return q


class SetResidual(nn.Module):
    """Small permutation-invariant encoder with a zero-initialized residual head."""

    def __init__(self, n_ctx, hidden=64, layers=2, heads=4, dropout=0.1,
                 max_res=0.2, use_frame_ctx=True):
        super().__init__()
        self.max_res = float(max_res)
        self.use_frame_ctx = bool(use_frame_ctx)
        self.pair = nn.Sequential(
            nn.Linear(6, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        enc = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=2 * hidden,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.theta_proj = nn.Sequential(nn.Linear(4, hidden), nn.GELU())
        self.ctx_emb = nn.Embedding(n_ctx, hidden)
        nn.init.normal_(self.ctx_emb.weight, std=0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 4),
        )
        # Per-frame residual: a local query MLP plus a support-conditioned shift.
        self.frame_head = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        self.frame_ctx = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 2),
        )
        # Near-zero residual at initialization, but non-zero weights so the
        # preceding layers receive gradients from the first step.
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)
        nn.init.normal_(self.frame_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.frame_head[-1].bias)
        nn.init.normal_(self.frame_ctx[-1].weight, std=1e-3)
        nn.init.zeros_(self.frame_ctx[-1].bias)

    def encode(self, pairs, theta0, ctx):
        x = self.pair(pairs)
        x = self.encoder(x)
        z = x.mean(dim=1)
        return z + self.theta_proj(theta0) + self.ctx_emb(ctx)

    def forward(self, pairs, theta0, ctx):
        # pairs [B,K,6], theta0 [B,4], ctx [B]; residual on 4 affine parameters.
        z = self.encode(pairs, theta0, ctx)
        raw = self.head(z)
        delta = self.max_res * torch.tanh(raw)
        return theta0 + delta, delta

    def forward_frame(self, pairs, theta0, ctx, query):
        """Residual on calibration output for each query point.

        pairs [B,K,6], theta0 [B,4], ctx [B], query [B,M,2]
        returns corrected [B,M,2] and residual [B,M,2].
        """
        z = self.encode(pairs, theta0, ctx)
        raw = self.frame_head(query)
        if self.use_frame_ctx:
            z = z[:, None, :].expand(-1, query.size(1), -1)
            raw = raw + self.frame_ctx(z)
        res = self.max_res * torch.tanh(raw)
        base = apply_theta_torch(query, theta0)
        return base + res, res


def select_grid3(pred, k):
    """One point nearest each 3x3 cell center; deterministic 3x3 calibration."""
    n = len(pred)
    cells = defaultdict(list)
    for j in range(n):
        cx = min(2, max(0, int(pred[j, 0] * 3)))
        cy = min(2, max(0, int(pred[j, 1] * 3)))
        cells[(cx, cy)].append(j)
    chosen = []
    used = set()
    for cy in range(3):
        for cx in range(3):
            center = np.array([(cx + 0.5) / 3.0, (cy + 0.5) / 3.0])
            vs = cells.get((cx, cy), [])
            if vs:
                j = min(vs, key=lambda j: float(np.linalg.norm(pred[j] - center)))
            else:
                rest = [j for j in range(n) if j not in used]
                if not rest:
                    break
                j = min(rest, key=lambda j: float(np.linalg.norm(pred[j] - center)))
            if j not in used:
                chosen.append(j)
                used.add(j)
    if len(chosen) < min(k, n):
        rest = [j for j in range(n) if j not in used]
        chosen += list(rest[:min(k, n) - len(chosen)])
    return np.asarray(sorted(chosen)[:min(k, n)], np.int64)


def support_indices(d, k, rng, mode="coverage"):
    """Exactly k support indices (coverage, deterministic 3x3 grid, or random)."""
    n = len(d["pred"])
    kk = min(int(k), n - 1)
    if mode == "grid" and kk == 9:
        sel = select_grid3(d["pred"], kk)
    elif mode == "random":
        sel = np.sort(rng.choice(n, kk, replace=False))
    else:
        sel = np.asarray(select_coverage(d["pred"], kk, rng), np.int64)
    if len(sel) < kk:
        rest = np.setdiff1d(np.arange(n), sel)
        need = kk - len(sel)
        if len(rest) < need:
            raise RuntimeError(f"cannot select K={kk} support points from n={n}")
        sel = np.sort(np.concatenate([sel, rng.choice(rest, need, replace=False)]))
    return sel[:kk]


def make_episode(d, k, m, rng, train, select_mode="coverage"):
    n = len(d["pred"])
    sel = support_indices(d, k, rng, select_mode)
    rest = np.setdiff1d(np.arange(n), sel)
    if len(rest) == 0:
        return None
    if train:
        q = rng.choice(rest, min(int(m), len(rest)), replace=False)
    else:
        q = rest
    return sel, np.asarray(q, np.int64)


def batch_from_episodes(batch, k, m, rng, device, ctx_map, train, select_mode="coverage"):
    """Build a padded batch. Returns None if no valid episode."""
    pairs, theta0s, ctxs, pqs, lqs, sxs, sys_, masks = [], [], [], [], [], [], [], []
    for d in batch:
        ep = make_episode(d, k, m, rng, train, select_mode)
        if ep is None:
            continue
        sel, q = ep
        theta0, _ = fit_theta_huber(d["pred"][sel], d["label"][sel], batch_lam, batch_mu, batch_delta)
        ps = d["pred"][sel]
        ls = d["label"][sel]
        pair = np.concatenate([ps, ls, ls - ps], axis=1).astype(np.float32)
        c = ctx_map.get(str(d["ctx"][0]), 0)
        mm = np.zeros(int(m), np.float32)
        mm[:len(q)] = 1.0
        qpad = np.zeros((int(m), 2), np.float32)
        lpad = np.zeros((int(m), 2), np.float32)
        spad = np.ones(int(m), np.float32)
        tpad = np.ones(int(m), np.float32)
        qpad[:len(q)] = d["pred"][q]
        lpad[:len(q)] = d["label"][q]
        spad[:len(q)] = d["sx"][q]
        tpad[:len(q)] = d["sy"][q]
        pairs.append(pair)
        theta0s.append(theta0)
        ctxs.append(c)
        pqs.append(qpad)
        lqs.append(lpad)
        sxs.append(spad)
        sys_.append(tpad)
        masks.append(mm)
    if not pairs:
        return None
    return {
        "pairs": torch.from_numpy(np.stack(pairs)).to(device),
        "theta0": torch.from_numpy(np.stack(theta0s).astype(np.float32)).to(device),
        "ctx": torch.tensor(ctxs, dtype=torch.long, device=device),
        "pq": torch.from_numpy(np.stack(pqs)).to(device),
        "lq": torch.from_numpy(np.stack(lqs)).to(device),
        "sx": torch.from_numpy(np.stack(sxs)).to(device),
        "sy": torch.from_numpy(np.stack(sys_)).to(device),
        "mask": torch.from_numpy(np.stack(masks)).to(device),
    }


def loss_batch(model, b, theta_reg, mode):
    if mode == "theta":
        theta, delta = model(b["pairs"], b["theta0"], b["ctx"])
        q = apply_theta_torch(b["pq"], theta)
    else:
        q, delta = model.forward_frame(b["pairs"], b["theta0"], b["ctx"], b["pq"])
    e2 = ((q[..., 0] - b["lq"][..., 0]) * b["sx"]) ** 2 + \
         ((q[..., 1] - b["lq"][..., 1]) * b["sy"]) ** 2
    loss = (e2 * b["mask"]).sum() / b["mask"].sum().clamp_min(1.0)
    reg = (delta ** 2).mean()
    return loss + theta_reg * reg, delta, loss.detach()


@torch.no_grad()
def evaluate_sessions(model, sessions, k, mu, lam, delta, seq_seed, device, ctx_map,
                      split_name, mode, select_mode="coverage", log_every=50):
    model.eval()
    rows = []
    acc0 = acc1 = nn = 0.0
    rng = np.random.RandomState(seq_seed)
    t0 = time.time()
    for i, (s, d) in enumerate(sessions.items()):
        sel = support_indices(d, k, rng, select_mode)
        q = np.setdiff1d(np.arange(len(d["pred"])), sel)
        if len(q) < 20:
            log(f"[{split_name}] skip session={s} n={len(d['pred'])} q={len(q)}")
            continue
        theta0, _ = fit_theta_huber(d["pred"][sel], d["label"][sel], lam, mu, delta)
        ps = d["pred"][sel]
        ls = d["label"][sel]
        pair = np.concatenate([ps, ls, ls - ps], axis=1).astype(np.float32)[None]
        c = ctx_map.get(str(d["ctx"][0]), 0)
        pt = torch.from_numpy(pair).to(device)
        th0t = torch.from_numpy(theta0.astype(np.float32))[None].to(device)
        ctt = torch.tensor([c], dtype=torch.long, device=device)
        with torch.no_grad():
            if mode == "theta":
                theta, dth = model(pt, th0t, ctt)
                q1 = apply_theta_np(d["pred"][q], theta[0].cpu().numpy())
            else:
                qq, dth = model.forward_frame(pt, th0t, ctt,
                                              torch.from_numpy(d["pred"][q][None]).to(device))
                q1 = qq[0].cpu().numpy()
                theta = theta0
        q0 = apply_theta_np(d["pred"][q], theta0)
        e0 = em(q0, d["label"][q], d["sx"][q], d["sy"][q])
        e1 = em(q1, d["label"][q], d["sx"][q], d["sy"][q])
        b, e = float(e0.mean()), float(e1.mean())
        acc0 += b * len(q)
        acc1 += e * len(q)
        nn += len(q)
        ratio = e / max(b, 1e-12) - 1.0
        if mode == "theta":
            dsum = dth[0].cpu().numpy()
        else:
            # Per-frame residual has shape [1,M,2]; store mean |dx|,|dy|.
            dmean = dth[0].abs().mean(0).cpu().numpy()
            dsum = np.array([dmean[0], 0.0, dmean[1], 0.0], np.float64)
        rows.append((s, len(q), b, e, ratio, theta0, theta, dsum))
        if i % log_every == 0 or i == len(sessions) - 1:
            dt = time.time() - t0
            log(f"[{split_name}] session={s} n={len(q)} base={b:.4f} calib={e:.4f} "
                f"({100*ratio:+.2f}%) elapsed={dt:.1f}s maxrss={maxrss_gb():.2f}GB")
    if not nn:
        return None, rows
    B, E = acc0 / nn, acc1 / nn
    rr = np.array([r[4] for r in rows])
    log(f"[{split_name}] frame-weighted base={B:.4f} calib={E:.4f} ({100*(E/B-1):+.2f}%) "
        f"sessions={len(rows)} improved={int((rr<0).sum())} degrade2={int((rr>0.02).sum())} "
        f"worst={100*rr.max():+.1f}% maxrss={maxrss_gb():.2f}GB")
    return (B, E, rows), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--val-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--pose-npz", required=True)
    ap.add_argument("--extra-train-npz", default="")
    ap.add_argument("--extra-index-csv", default="")
    ap.add_argument("--extra-pose-npz", default="")
    ap.add_argument("--extra-context-key", choices=["none", "orientation", "activity"], default="none")
    ap.add_argument("--context-key", choices=["none", "orientation", "activity"], default="none")
    ap.add_argument("--K", type=int, default=9)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--theta-reg", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-res", type=float, default=0.2)
    ap.add_argument("--residual-mode", choices=["theta", "frame"], default="theta",
                    help="theta: residual on 4 affine params; frame: per-frame residual")
    ap.add_argument("--no-frame-ctx", action="store_true",
                    help="frame mode: drop support-conditioned shift (global residual only)")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1.345)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--select-mode", choices=["coverage", "grid", "random"], default="coverage")
    ap.add_argument("--episodes-per-session", type=int, default=1)
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-train", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    global batch_lam, batch_mu, batch_delta
    torch.set_num_threads(max(1, a.threads))
    device = "cuda" if (a.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    log(f"[meta] start={time.strftime('%F %T')} device={device} gpu={a.gpu} "
        f"torch={torch.__version__} numpy={np.__version__} maxrss={maxrss_gb():.2f}GB")
    log(f"[meta] args={vars(a)}")
    set_seed(a.train_seed)

    D = load_sessions(a.train_npz, a.index_csv, a.pose_npz, a.context_key)
    V = load_sessions(a.val_npz, a.index_csv, a.pose_npz, a.context_key)
    T = load_sessions(a.test_npz, a.index_csv, a.pose_npz, a.context_key)
    # Target-domain prior is fitted on the primary source only; extra source
    # sessions are used exclusively to train the residual meta-model.
    mu, TH = fit_source_mu(D)
    batch_lam, batch_mu, batch_delta = a.lam, mu, a.delta
    log(f"[meta] groups primary_train/val/test={len(D)}/{len(V)}/{len(T)} "
        f"batches={a.batch} K={a.K} M={a.M} lam={a.lam} delta={a.delta}")
    if a.extra_train_npz:
        eidx = a.extra_index_csv or a.index_csv
        epose = a.extra_pose_npz or a.pose_npz
        E = load_sessions(a.extra_train_npz, eidx, epose, a.extra_context_key)
        before = len(D)
        D.update({f"EXTRA::{k}": v for k, v in E.items()})
        log(f"[meta] extra source sessions={len(E)} total_train={len(D)} "
            f"(primary={before}) extra_keys={a.extra_context_key}")
    log(f"[meta] source prior (primary only) theta_mean={np.round(mu,4).tolist()} "
        f"std={np.round(TH.std(0),4).tolist()}")
    log(f"[meta] source theta mean={np.round(mu,4).tolist()} std={np.round(TH.std(0),4).tolist()}")
    for name, S in (("train", D), ("val", V), ("test", T)):
        ns = [len(d["pred"]) for d in S.values()]
        cs = Counter(str(d["ctx"][0]) for d in S.values())
        log(f"[meta] {name} sessions={len(S)} frames={sum(ns)} n[min/med/max]="
            f"{min(ns)}/{int(np.median(ns))}/{max(ns)} contexts={dict(cs)}")

    ctx_names = sorted({str(d["ctx"][0]) for d in list(D.values()) + list(V.values())})
    ctx_map = {c: i for i, c in enumerate(ctx_names)}
    log(f"[meta] context vocab={ctx_map}")
    unseen = sorted({str(d["ctx"][0]) for d in T.values()} - set(ctx_names))
    if unseen:
        log(f"[meta] WARNING unseen test contexts map to 0: {unseen}")

    model = SetResidual(len(ctx_names), a.hidden, a.layers, a.heads, a.dropout,
                        a.max_res, use_frame_ctx=not a.no_frame_ctx).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    log(f"[meta] model params={nparam} hidden={a.hidden} layers={a.layers} "
        f"heads={a.heads} dropout={a.dropout} max_res={a.max_res} theta_reg={a.theta_reg}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    best_state = copy.deepcopy(model.state_dict())
    init_val, _ = evaluate_sessions(model, V, a.K, mu, a.lam, a.delta,
                                    seq_seed=12345, device=device,
                                    ctx_map=ctx_map, split_name="init",
                                    mode=a.residual_mode, select_mode=a.select_mode)
    best_val = init_val[1] if init_val is not None else float("inf")
    best_epoch = 0
    log(f"[meta] initial val_calib={best_val:.4f} (fallback to this state)")
    hist = []
    keys = list(D.keys())
    train_rng = np.random.RandomState(a.train_seed)

    for epoch in range(1, a.epochs + 1):
        model.train()
        perm = np.concatenate([
            train_rng.permutation(len(keys))
            for _ in range(max(1, a.episodes_per_session))
        ])
        ep_loss = 0.0
        ep_delta = 0.0
        nb = 0
        t0 = time.time()
        for st in range(0, len(perm), a.batch):
            batch_keys = [keys[j] for j in perm[st:st + a.batch]]
            b = batch_from_episodes([D[j] for j in batch_keys], a.K, a.M,
                                    train_rng, device, ctx_map, train=True,
                                    select_mode=a.select_mode)
            if b is None:
                continue
            if nb == 0:
                log(f"[meta] first batch shapes pairs={tuple(b['pairs'].shape)} "
                    f"theta0={tuple(b['theta0'].shape)} pq={tuple(b['pq'].shape)} "
                    f"mask_sum={float(b['mask'].sum()):.0f}")
            loss, delta, raw = loss_batch(model, b, a.theta_reg, a.residual_mode)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss epoch={epoch} batch={st}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(raw)
            ep_delta += float(delta.detach().pow(2).mean().sqrt())
            nb += 1
        sched.step()
        if nb == 0:
            raise RuntimeError("no training episodes; check K/M/session lengths")
        log(f"[meta] epoch={epoch:03d}/{a.epochs} eps={a.episodes_per_session} "
            f"loss={ep_loss/nb:.6f} "
            f"delta_rms={ep_delta/nb:.5f} lr={sched.get_last_lr()[0]:.2e} "
            f"time={time.time()-t0:.1f}s maxrss={maxrss_gb():.2f}GB")
        hist.append((epoch, ep_loss / nb, ep_delta / nb))

        if epoch % a.eval_every == 0 or epoch == a.epochs:
            val, _ = evaluate_sessions(model, V, a.K, mu, a.lam, a.delta,
                                       seq_seed=12345, device=device,
                                       ctx_map=ctx_map, split_name="val",
                                       mode=a.residual_mode, select_mode=a.select_mode)
            if val is not None and val[1] < best_val:
                best_val = val[1]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                log(f"[meta] new best epoch={epoch} val_calib={best_val:.4f}")

    model.load_state_dict(best_state)
    log(f"[meta] restore best epoch={best_epoch} val_calib={best_val:.4f}")

    # Validation with the selected model, then test over several support seeds.
    val_sel, _ = evaluate_sessions(model, V, a.K, mu, a.lam, a.delta, seq_seed=12345,
                                   device=device, ctx_map=ctx_map, split_name="val-best",
                                   mode=a.residual_mode, select_mode=a.select_mode)
    if a.eval_train:
        evaluate_sessions(model, D, a.K, mu, a.lam, a.delta, seq_seed=777,
                          device=device, ctx_map=ctx_map, split_name="train-best",
                          mode=a.residual_mode, select_mode=a.select_mode)
    all_rows = []
    for seed in range(a.seeds):
        log(f"[meta] test support seed={seed}")
        res, rows = evaluate_sessions(model, T, a.K, mu, a.lam, a.delta,
                                      seq_seed=90000 + seed, device=device,
                                      ctx_map=ctx_map, split_name=f"test/s{seed}",
                                      mode=a.residual_mode, select_mode=a.select_mode)
        if res is None:
            log(f"[meta] WARNING empty test seed={seed}")
            continue
        B, E, _ = res
        for s, n, b, e, r, th0, th, dth in rows:
            all_rows.append([seed, s, n, b, e, r, *th0.tolist(), *th.tolist(), *dth.tolist()])

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "session", "n", "base", "calib", "ratio",
                    "th0_sx", "th0_bx", "th0_sy", "th0_by",
                    "th_sx", "th_bx", "th_sy", "th_by",
                    "d_sx", "d_bx", "d_sy", "d_by"])
        w.writerows(all_rows)
    log(f"[meta] saved {out} rows={len(all_rows)} best_epoch={best_epoch} "
        f"val_calib={best_val:.4f} maxrss={maxrss_gb():.2f}GB")


if __name__ == "__main__":
    # Module-level values set in main(); declared here for batch_from_episodes.
    batch_lam = 1.0
    batch_mu = np.zeros(4)
    batch_delta = 1.345
    main()
