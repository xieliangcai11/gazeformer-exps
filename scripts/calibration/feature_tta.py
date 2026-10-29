"""Feature-level test-time adaptation (TTA) pilot for 2D gaze calibration.

Pipeline:
  1. Load frozen DINOv2 features and 2D labels.
  2. Train a small 2D head on source features.
  3. At target, adapt feature statistics per session using unlabeled frames
     (diagonal or full covariance CORAL), then predict.
  4. Apply the same K-shot Huber calibration to base predictions, head
     predictions, and TTA predictions.

No target labels are used in TTA.  Target labels are used only for K support
frames and for evaluation.  All stages log shapes, feature distances, losses,
per-session base/calibrated errors, and memory.
"""
import argparse
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
    def __init__(self, dim=1536, hidden=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def load_sessions_feat(npz_path, index_csv, context_key):
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
    log(f"[ftta] loaded {npz_path} sessions={len(out)} frames={sum(len(d['F']) for d in out.values())}")
    return out


def session_norm(F):
    return (F - F.mean(0)) / (F.std(0) + 1e-6)


def affine_prior(train):
    # replicate fit_source_mu semantics [sx,bx,sy,by]
    xs, ys = [], []
    for d in train.values():
        for ax in (0, 1):
            P = d["pred"][:, ax]
            U = d["label"][:, ax]
            X = np.stack([P, np.ones(len(P))], 1)
            th = np.linalg.solve(X.T @ X + 1e-6 * np.eye(2), X.T @ U)
            (xs if ax == 0 else ys).append(th)
    return np.concatenate([np.mean(xs, 0), np.mean(ys, 0)])


def train_head(train, val, device, epochs, batch, lr, hidden, dropout, feature_norm):
    def prep(d):
        return session_norm(d["F"]) if feature_norm == "session" else d["F"]
    Fall = np.concatenate([prep(d) for d in train.values()])
    mu = Fall.mean(0)
    sd = Fall.std(0) + 1e-6
    F = Fall
    Y = np.concatenate([d["label"] for d in train.values()])
    Fv = np.concatenate([prep(d) for d in val.values()]) if val else F[-2000:]
    Yv = np.concatenate([d["label"] for d in val.values()]) if val else Y[-2000:]
    log(f"[ftta] head train={F.shape} val={Fv.shape} feat_mean={mu[:3].round(3).tolist()} "
        f"feat_std[min/med/max]={sd.min():.3f}/{np.median(sd):.3f}/{sd.max():.3f}")
    model = Head(F.shape[1], hidden, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    Ft = torch.tensor((F - mu) / sd, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    Fvt = torch.tensor((Fv - mu) / sd, dtype=torch.float32).to(device)
    Yvt = torch.tensor(Yv, dtype=torch.float32).to(device)
    best = float("inf")
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Ft))
        tot = n = 0.0
        t0 = time.time()
        for st in range(0, len(perm), batch):
            idx = perm[st:st + batch]
            x = Ft[idx].to(device)
            y = Yt[idx].to(device)
            q = model(x)
            loss = ((q - y) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
            n += len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            qv = model(Fvt)
            ev = torch.sqrt(((qv - Yvt) ** 2).sum(1)).mean()
        if float(ev) < best:
            best = float(ev)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log(f"[ftta] head_epoch={ep:03d}/{epochs} train_loss={tot/n:.6f} "
            f"val_EM={float(ev):.4f} lr={sched.get_last_lr()[0]:.2e} "
            f"time={time.time()-t0:.1f}s")
    model.load_state_dict(best_state)
    model.eval()
    log(f"[ftta] head best_val_EM={best:.4f}")
    return model, mu, sd


@torch.no_grad()
def predict_head(model, F, mu, sd, device, batch=1024):
    out = []
    for st in range(0, len(F), batch):
        x = torch.tensor((F[st:st + batch] - mu) / sd, dtype=torch.float32).to(device)
        out.append(model(x).cpu().numpy())
    return np.concatenate(out)


def coral_diag(F, mu_t, sd_t, mu_s, sd_s):
    return (F - mu_t) / sd_t * sd_s + mu_s


def coral_full(F, mu_t, cov_t, mu_s, cov_s):
    # Shrink covariance for numerical stability.
    d = F.shape[1]
    lam = 1e-2
    cov_t = (1 - lam) * cov_t + lam * (np.trace(cov_t) / d) * np.eye(d)
    cov_s = (1 - lam) * cov_s + lam * (np.trace(cov_s) / d) * np.eye(d)
    # symmetric whitening / coloring
    et, Vt = np.linalg.eigh(cov_t)
    es, Vs = np.linalg.eigh(cov_s)
    Wt = Vt @ np.diag(1.0 / np.sqrt(np.maximum(et, 1e-8))) @ Vt.T
    Ws = Vs @ np.diag(np.sqrt(np.maximum(es, 1e-8))) @ Vs.T
    return (F - mu_t) @ Wt @ Ws + mu_s


def build_gaze_projection(F, Y, lam=1.0):
    """Ridge gaze subspace P = W^T (W W^T)^-1 W, shape [d,d]."""
    X = F.astype(np.float64)
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d)
    W = np.linalg.solve(A, X.T @ Y.astype(np.float64)).T  # [2,d]
    G = W @ W.T + 1e-8 * np.eye(2)
    return W.T @ np.linalg.solve(G, W)  # [d,d]


def coral_projected(F, Fsrc, P):
    """Align only P-perpendicular residuals; preserve P-parallel gaze signal."""
    Fpar = F @ P
    Fperp = F - Fpar
    Sperp = Fsrc - Fsrc @ P
    tmu, tsd = Fperp.mean(0), Fperp.std(0) + 1e-6
    smu, ssd = Sperp.mean(0), Sperp.std(0) + 1e-6
    return Fpar + (Fperp - tmu) / tsd * ssd + smu


def eval_pred(d, pred_all, K, mu, lam, delta, seeds):
    """Huber K-shot calibration and evaluation on held-out frames."""
    acc0 = nn = 0.0
    ratios = []
    for seed in range(seeds):
        rng = np.random.RandomState(90000 + seed)
        sel = support_indices(d, K, rng)
        q = np.setdiff1d(np.arange(len(d["pred"])), sel)
        if len(q) < 20:
            continue
        th0, _ = fit_theta_huber(pred_all[sel], d["label"][sel], lam, mu, delta)
        q0 = apply_theta_np(pred_all[q], th0)
        e0 = em(q0, d["label"][q], d["sx"][q], d["sy"][q]).mean()
        acc0 += e0 * len(q)
        nn += len(q)
        ratios.append(e0)
    return acc0 / max(nn, 1), nn


def eval_direct(d, pred_all):
    q = np.arange(len(d["pred"]))
    return float(em(pred_all[q], d["label"][q], d["sx"][q], d["sy"][q]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--val-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--context-key", choices=["none", "orientation", "activity"], default="none")
    ap.add_argument("--K", type=int, default=9)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1.345)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--coral", choices=["none", "mean", "diag", "diag_reg", "projected", "full"],
                    default="diag")
    ap.add_argument("--feature-norm", choices=["global", "session"], default="global")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=-1)
    a = ap.parse_args()

    torch.set_num_threads(4)
    device = "cuda" if (a.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    log(f"[ftta] start={time.strftime('%F %T')} device={device} args={vars(a)}")
    D = load_sessions_feat(a.train_npz, a.index_csv, a.context_key)
    V = load_sessions_feat(a.val_npz, a.index_csv, a.context_key)
    T = load_sessions_feat(a.test_npz, a.index_csv, a.context_key)
    mu_aff = affine_prior(D)
    log(f"[ftta] source affine prior={np.round(mu_aff,4).tolist()} "
        f"target sessions={len(T)}")

    model, fmu, fsd = train_head(D, V, device, a.epochs, a.batch, a.lr,
                                 a.hidden, a.dropout, a.feature_norm)
    # Source feature statistics for CORAL.
    Fall = np.concatenate([d["F"] for d in D.values()])
    smu = Fall.mean(0)
    ssd = Fall.std(0) + 1e-6
    scov = np.cov(Fall[::7], rowvar=False) if a.coral == "full" else None
    Pgaze = None
    if a.coral == "projected":
        Yall = np.concatenate([d["label"] for d in D.values()])
        Pgaze = build_gaze_projection(Fall, Yall)
        log(f"[ftta] gaze projection rank=2 trace={np.trace(Pgaze):.4f}")

    acc_base = acc_head = acc_tta = 0.0
    nn = 0
    diag = []
    t0 = time.time()
    for si, (s, d) in enumerate(T.items()):
        F = d["F"]
        if a.feature_norm == "session":
            F = session_norm(F)
        tmu = F.mean(0)
        tsd = F.std(0) + 1e-6
        if a.coral == "diag":
            Ftta = coral_diag(F, tmu, tsd, smu, ssd)
        elif a.coral == "diag_reg":
            floor = 0.25 * np.median(tsd)
            Ftta = coral_diag(F, tmu, np.maximum(tsd, floor), smu, ssd)
        elif a.coral == "mean":
            Ftta = F - tmu + smu
        elif a.coral == "projected":
            Ftta = coral_projected(F, Fall, Pgaze)
        elif a.coral == "full":
            tcov = np.cov(F, rowvar=False)
            Ftta = coral_full(F, tmu, tcov, smu, scov)
        else:
            Ftta = F
        h_raw = predict_head(model, F, fmu, fsd, device)
        h_tta = predict_head(model, Ftta, fmu, fsd, device)
        b_direct = eval_direct(d, d["pred"])
        h_direct = eval_direct(d, h_raw)
        t_direct = eval_direct(d, h_tta)
        b_cal = eval_pred(d, d["pred"], a.K, mu_aff, a.lam, a.delta, a.seeds)[0]
        h_cal = eval_pred(d, h_raw, a.K, mu_aff, a.lam, a.delta, a.seeds)[0]
        t_cal = eval_pred(d, h_tta, a.K, mu_aff, a.lam, a.delta, a.seeds)[0]
        n = len(d["pred"])
        acc_base += b_cal * n
        acc_head += h_cal * n
        acc_tta += t_cal * n
        nn += n
        diag.append((s, b_direct, h_direct, t_direct, b_cal, h_cal, t_cal))
        if si % 5 == 0 or si == len(T) - 1:
            log(f"[ftta] session={s} n={n} direct(base/head/tta)="
                f"{b_direct:.4f}/{h_direct:.4f}/{t_direct:.4f} "
                f"cal={b_cal:.4f}/{h_cal:.4f}/{t_cal:.4f} "
                f"feat_mean_shift={np.linalg.norm((tmu-smu)/ssd):.3f} "
                f"elapsed={time.time()-t0:.1f}s")
    B, H, TT = acc_base / nn, acc_head / nn, acc_tta / nn
    log(f"[ftta] RESULT coral={a.coral} K={a.K} base={B:.4f} "
        f"head={H:.4f} tta={TT:.4f} "
        f"head_vs_base={100*(H/B-1):+.2f}% tta_vs_base={100*(TT/B-1):+.2f}% "
        f"tta_vs_head={100*(TT/H-1):+.2f}%")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "base_direct", "head_direct", "tta_direct",
                    "base_cal", "head_cal", "tta_cal"])
        w.writerows(diag)
    log(f"[ftta] saved {out}")


if __name__ == "__main__":
    main()
