"""Alfa-style SVD right-factor adaptation for the 2D gaze model.

Faithful parts of Alfa (AAAI 2026):
  - truncated SVD of selected linear weights: W = U_d S_d V_d^T
  - freeze U_d and the residual W - U_d S_d V_d^T
  - adapt only the right factor via multi-head attention over rank slices
  - target adaptation uses only unlabeled images and a left-right symmetry loss

We apply it to the last fusion block's feed-forward layers (fc1/fc2), where
the dense 384/1536 matrices make SVD meaningful.  The 2D head is kept frozen.

Target protocol:
  - first 5 frames of each target session are the unlabeled adaptation set
  - no target labels are used for adaptation
  - K support labels are used only for the final Huber calibration
"""
import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path[:0] = ["src", "."]
from configs.rgbdgaze_config import RGBDGaze_INDEX_DIR  # noqa: E402
from gazelab.datasets import RGBDGazeDataset  # noqa: E402
from gazelab.datasets.table_index import TableIndex  # noqa: E402
from gazelab.models.dinov2_gaze import (  # noqa: E402
    RGBDGazeDINOv2, load_state_dict_any_res,
)
from scripts.calibration.seq_bayes_calib import load_index, em  # noqa: E402
from scripts.calibration.meta_residual_calib import (  # noqa: E402
    fit_theta_huber, apply_theta_np, support_indices,
)


def log(msg):
    print(msg, flush=True)


class AlfaLinear(nn.Module):
    """W x + bias, with W = U_d V_base + residual; only V side is adapted."""

    def __init__(self, linear, svd_rank=64, lora_rank=8, heads=4):
        super().__init__()
        W = linear.weight.detach().float()
        b = linear.bias.detach().float()
        m, n = W.shape
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        d = min(int(svd_rank), len(S))
        self.register_buffer("U", U[:, :d].contiguous())
        self.register_buffer("Vbase", (S[:d, None] * Vh[:d]).contiguous())
        # exact residual so the initial module is bit-close to the original.
        self.register_buffer("Wres", (W - self.U @ self.Vbase).contiguous())
        self.register_buffer("bias", b.contiguous())
        self.d = d
        self.r = int(lora_rank)
        self.h = int(heads)
        # Per-head query-side low-rank modules, zero back-projection.
        self.Aq = nn.Parameter(torch.randn(self.h, self.r, d) * 0.02)
        self.Bq = nn.Parameter(torch.zeros(self.h, d, self.r))
        # Aggregation modules, zero back-projection.
        self.Ap = nn.Parameter(torch.randn(self.r * self.h, self.h * d) * 0.02)
        self.Bp = nn.Parameter(torch.zeros(d, self.r * self.h))
        self.to(linear.weight.device)

    def right_factor(self):
        Vb = self.Vbase
        K = Vb                                       # [d,n]
        V = Vb.t()                                   # [n,d]
        zs = []
        for h in range(self.h):
            # Q_h = B_h^Q A_h^Q Vb; K=Vb; V=Vb^T
            Q = self.Bq[h] @ (self.Aq[h] @ Vb)       # [d,n]
            att = torch.softmax(
                Q @ K.t() / math.sqrt(max(Vb.shape[1], 1)), dim=-1
            )
            Z = V @ att.t()                          # [n,d]
            zs.append(Z.t())                         # [d,n]
        Zcat = torch.cat(zs, dim=0)                  # [H*d,n]
        return self.Bp @ (self.Ap @ Zcat) + Vb

    def forward(self, x):
        Va = self.right_factor()
        # x [..., n] -> [..., m]
        out = torch.matmul(x, Va.t()) @ self.U.t()
        out = out + torch.matmul(x, self.Wres.t())
        return out + self.bias


def replace_last_block(model):
    """Replace fc1/fc2 in the last dense BlockMoba with AlfaLinear."""
    blk = model.layers[-1]
    if not hasattr(blk, "moe") or not hasattr(blk.moe, "fc1"):
        raise RuntimeError("last block has no fc1")
    blk.moe.fc1 = AlfaLinear(blk.moe.fc1)
    blk.moe.fc2 = AlfaLinear(blk.moe.fc2)
    return [blk.moe.fc1, blk.moe.fc2]


def alfa_params(model):
    return [p for m in model.modules() if isinstance(m, AlfaLinear)
            for p in (m.Aq, m.Bq, m.Ap, m.Bp)]


def load_model(ckpt, device):
    model = RGBDGazeDINOv2(
        gaze_dim=2, d_model=384, num_heads=8, num_layers=6,
        inter_dim=1536, freeze_dino=True, unfreeze_last=0,
        use_imu=False, use_geom=False, use_geom_orient=False,
        use_eye=True, img_size=224,
    ).to(device)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    miss, unexp, resampled = load_state_dict_any_res(model, state)
    log(f"[alfa] load {ckpt} missing={len(miss)} unexpected={len(unexp)} "
        f"resampled={resampled}")
    if miss or unexp:
        raise RuntimeError(f"model mismatch missing={list(miss)[:5]} "
                           f"unexpected={list(unexp)[:5]}")
    model.eval()
    return model


def batch_items(items, device):
    face = torch.stack([x[0].rgb for x in items]).to(device).float()
    eye_l = torch.stack([x[0].eye_l for x in items]).to(device).float()
    eye_r = torch.stack([x[0].eye_r for x in items]).to(device).float()
    label = torch.stack([x[1] for x in items]).to(device).float()
    return face, eye_l, eye_r, label


def symmetry_loss(model, face, eye_l, eye_r):
    q = model(face, eye_l=eye_l, eye_r=eye_r)
    face_f = torch.flip(face, dims=[3])
    eye_lf = torch.flip(eye_r, dims=[3])
    eye_rf = torch.flip(eye_l, dims=[3])
    qf = model(face_f, eye_l=eye_lf, eye_r=eye_rf)
    # horizontal screen mirror: x -> 1-x, y unchanged
    return ((qf[:, 0] - (1.0 - q[:, 0])) ** 2 +
            (qf[:, 1] - q[:, 1]) ** 2).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--index-dir", default=str(RGBDGaze_INDEX_DIR))
    ap.add_argument("--index-csv", default="")
    ap.add_argument("--K", type=int, default=9)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1.345)
    ap.add_argument("--svd-rank", type=int, default=64)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-sessions", type=int, default=0, help="debug limit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=-1)
    a = ap.parse_args()

    torch.set_num_threads(4)
    device = "cuda" if (a.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    log(f"[alfa] start={time.strftime('%F %T')} device={device} args={vars(a)}")
    idx_csv = a.index_csv or str(Path(a.index_dir) / "index.csv")
    tr = np.load(a.train_npz, allow_pickle=True)
    trP = tr["pred"].astype(np.float64)
    trL = tr["label"].astype(np.float64)
    thetas = []
    for ax in range(2):
        X = np.stack([trP[:, ax], np.ones(len(trP))], 1)
        th = np.linalg.solve(X.T @ X + 1e-6 * np.eye(2), X.T @ trL[:, ax])
        thetas.append(th)
    mu_aff = np.concatenate(thetas)
    log(f"[alfa] source affine prior={np.round(mu_aff,4).tolist()}")

    z = np.load(a.test_npz, allow_pickle=True)
    paths = [str(p) for p in z["path"].tolist()]
    fidx = z["frame_index"].astype(np.int64)
    label = z["label"].astype(np.float64)
    pred = z["pred"].astype(np.float64)
    meta = load_index(idx_csv, np.array(paths, object))
    groups = defaultdict(list)
    for i, p in enumerate(paths):
        if p not in meta:
            continue
        groups[p.rsplit("/", 1)[0]].append(i)
    groups = {s: np.array(sorted(v, key=lambda i: int(fidx[i])), np.int64)
              for s, v in groups.items() if len(v) >= 40}
    if a.max_sessions:
        keep = list(groups)[:a.max_sessions]
        groups = {k: groups[k] for k in keep}
    log(f"[alfa] sessions={len(groups)} frames={sum(len(v) for v in groups.values())}")

    model = load_model(a.ckpt, device)
    # Baseline equivalence check before replacing layers.
    ds = RGBDGazeDataset(Path(idx_csv), augment=False, image_size=224,
                         eye_crops=True)
    table = TableIndex(Path(idx_csv))
    lookup = table.lookup("rgb_path")

    # Check on one batch of the first session.
    s0 = next(iter(groups))
    idx0 = [lookup[paths[i]] for i in groups[s0][:min(a.batch, len(groups[s0]))]]
    b0 = [ds[i] for i in idx0]
    face, eye_l, eye_r, _ = batch_items(b0, device)
    with torch.no_grad():
        q_before = model(face, eye_l=eye_l, eye_r=eye_r).cpu().numpy()
        base_np = pred[groups[s0][:len(idx0)]]
    log(f"[alfa] base check max|diff|={np.abs(q_before-base_np).max():.8f}")
    # Replace and check that the residual keeps outputs identical.
    alfas = replace_last_block(model)
    nparam = sum(p.numel() for p in alfa_params(model))
    log(f"[alfa] adapted layers={len(alfas)} alfa_params={nparam} "
        f"svd_rank={a.svd_rank} lora_rank={a.lora_rank} heads={a.heads}")
    with torch.no_grad():
        q_after = model(face, eye_l=eye_l, eye_r=eye_r).cpu().numpy()
    log(f"[alfa] replaced head check max|diff|={np.abs(q_after-q_before).max():.8f}")

    rows = []
    acc_base = acc_alfa = 0.0
    nsum = 0
    t0 = time.time()
    for si, (s, ii) in enumerate(groups.items()):
        # adaptation set: first 5 unlabeled frames of this session
        adapt_idx = [lookup[paths[i]] for i in ii[:5]]
        adapt_items = [ds[i] for i in adapt_idx]
        # reset Alfa parameters to the pre-adaptation state? We intentionally
        # continue from the source/previous state; per-session clone would be
        # more faithful. Clone adapters for each session.
        state = {n: p.detach().clone() for n, p in model.named_parameters()
                 if n.endswith(("Aq", "Bq", "Ap", "Bp"))}
        opt = torch.optim.AdamW(alfa_params(model), lr=a.lr, weight_decay=0.0)
        for step in range(a.steps):
            items = [adapt_items[k % len(adapt_items)] for k in range(a.batch)]
            face, eye_l, eye_r, _ = batch_items(items, device)
            loss = symmetry_loss(model, face, eye_l, eye_r)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == 0 or step == a.steps - 1:
                log(f"[alfa] session={s} step={step+1}/{a.steps} "
                    f"sym_loss={float(loss):.6f}")
        # Evaluate all frames of the session.
        preds = []
        for st in range(0, len(ii), a.batch):
            batch = [ds[lookup[paths[i]]] for i in ii[st:st + a.batch]]
            face, eye_l, eye_r, _ = batch_items(batch, device)
            with torch.no_grad():
                preds.append(model(face, eye_l=eye_l, eye_r=eye_r).cpu().numpy())
        q = np.concatenate(preds)
        # restore per-session adapter after evaluation; next session starts fresh.
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in state:
                    p.copy_(state[n])
        base_direct = float(em(pred[ii], label[ii],
                               np.array([meta[paths[i]][2] for i in ii]),
                               np.array([meta[paths[i]][3] for i in ii])).mean())
        alfa_direct = float(em(q, label[ii],
                               np.array([meta[paths[i]][2] for i in ii]),
                               np.array([meta[paths[i]][3] for i in ii])).mean())
        # K calibration on the adapted predictions; support chosen on base pred.
        base_vals, alfa_vals = [], []
        for seed in range(a.seeds):
            rng = np.random.RandomState(90000 + seed)
            # local arrays for support selection
            dloc = dict(pred=pred[ii], label=label[ii],
                        sx=np.array([meta[paths[i]][2] for i in ii]),
                        sy=np.array([meta[paths[i]][3] for i in ii]),
                        ctx=np.zeros(len(ii), object))
            sel = support_indices(dloc, a.K, rng)
            qidx = np.setdiff1d(np.arange(len(ii)), sel)
            thb, _ = fit_theta_huber(pred[ii][sel], label[ii][sel],
                                     a.lam, mu_aff, a.delta)
            tha, _ = fit_theta_huber(q[sel], label[ii][sel],
                                     a.lam, mu_aff, a.delta)
            base_vals.append(float(em(apply_theta_np(pred[ii][qidx], thb),
                                      label[ii][qidx], dloc["sx"][qidx],
                                      dloc["sy"][qidx]).mean()))
            alfa_vals.append(float(em(apply_theta_np(q[qidx], tha),
                                      label[ii][qidx], dloc["sx"][qidx],
                                      dloc["sy"][qidx]).mean()))
        bmean, amean = float(np.mean(base_vals)), float(np.mean(alfa_vals))
        acc_base += bmean * len(ii)
        acc_alfa += amean * len(ii)
        nsum += len(ii)
        rows.append([s, len(ii), base_direct, alfa_direct, bmean, amean,
                     100 * (amean / bmean - 1)])
        if si % 2 == 0 or si == len(groups) - 1:
            log(f"[alfa] session={s} n={len(ii)} direct(base/alfa)="
                f"{base_direct:.4f}/{alfa_direct:.4f} "
                f"cal={bmean:.4f}/{amean:.4f} "
                f"({100*(amean/bmean-1):+.2f}%) elapsed={time.time()-t0:.1f}s")
    B, A = acc_base / nsum, acc_alfa / nsum
    log(f"[alfa] RESULT K={a.K} base_cal={B:.4f} alfa_cal={A:.4f} "
        f"alfa_vs_base={100*(A/B-1):+.2f}%")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "n", "base_direct", "alfa_direct",
                    "base_cal", "alfa_cal", "pct"])
        w.writerows(rows)
    log(f"[alfa] saved {out}")


if __name__ == "__main__":
    main()
