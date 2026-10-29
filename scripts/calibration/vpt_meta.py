"""VPT prompt + meta-auxiliary adaptation for 2D screen calibration.

Design (inspired by Meta Prompt / ELF-UA, not Alfa):
  - frozen RGBDGazeDINOv2 backbone
  - P learnable prompt tokens injected into the fusion sequence (VPT-style)
  - auxiliary task: 4x4 jigsaw permutation prediction
  - source meta-training:
        inner: update prompt with jigsaw loss on unlabeled support
        outer: update prompt with 2D gaze loss on labeled query
  - target: adapt the prompt with jigsaw loss on 5 unlabeled frames,
            then fit K-shot Huber calibration

No target labels are used for prompt adaptation.  All stages log shapes,
losses, per-session results, and timing.
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
import torch.nn.functional as F

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


class PromptInjector:
    def __init__(self, model, prompt):
        self.model = model
        self.prompt = prompt
        self.handle = model.layers[0].register_forward_pre_hook(self._hook)

    def _hook(self, module, inp):
        if self.prompt is None:
            return None
        x = inp[0]
        p = self.prompt.expand(x.size(0), -1, -1)
        return (torch.cat([x[:, :1], p, x[:, 1:]], dim=1),) + tuple(inp[1:])

    def close(self):
        self.handle.remove()


def load_model(ckpt, device):
    model = RGBDGazeDINOv2(
        gaze_dim=2, d_model=384, num_heads=8, num_layers=6,
        inter_dim=1536, freeze_dino=True, unfreeze_last=0,
        use_imu=False, use_geom=False, use_geom_orient=False,
        use_eye=True, img_size=224,
    ).to(device)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    miss, unexp, resampled = load_state_dict_any_res(model, state)
    log(f"[vpt] load missing={len(miss)} unexpected={len(unexp)} resampled={resampled}")
    if miss or unexp:
        raise RuntimeError(f"checkpoint mismatch {list(miss)[:5]} {list(unexp)[:5]}")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


def make_perms(n=16, seed=0):
    rng = np.random.RandomState(seed)
    perms = [np.arange(16)]
    seen = {tuple(perms[0])}
    while len(perms) < n:
        p = rng.permutation(16)
        if tuple(p) not in seen:
            seen.add(tuple(p))
            perms.append(p)
    return [torch.tensor(p, dtype=torch.long) for p in perms]


def jigsaw_images(face, perm):
    # face [B,3,224,224] -> patches 4x4 of 56x56
    b, c, h, w = face.shape
    ps = h // 4
    x = face.view(b, c, 4, ps, 4, ps).permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(b, 16, c, ps, ps)
    x = x[:, perm]
    x = x.reshape(b, 4, 4, c, ps, ps).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(b, c, h, w)


def batch_items(items, device):
    face = torch.stack([x[0].rgb for x in items]).to(device).float()
    eye_l = torch.stack([x[0].eye_l for x in items]).to(device).float()
    eye_r = torch.stack([x[0].eye_r for x in items]).to(device).float()
    label = torch.stack([x[1] for x in items]).to(device).float()
    return face, eye_l, eye_r, label


def feature_of(model, face, eye_l, eye_r):
    """CLS feature consumed by the gaze head, via a pre-hook."""
    bag = {}
    h = model.head.register_forward_pre_hook(
        lambda m, inp: bag.__setitem__("x", inp[0])
    )
    try:
        q = model(face, eye_l=eye_l, eye_r=eye_r)
    finally:
        h.remove()
    return q, bag["x"]


def source_prior(train_npz):
    z = np.load(train_npz, allow_pickle=True)
    P = z["pred"].astype(np.float64)
    L = z["label"].astype(np.float64)
    th = []
    for ax in range(2):
        X = np.stack([P[:, ax], np.ones(len(P))], 1)
        th.append(np.linalg.solve(X.T @ X + 1e-6 * np.eye(2), X.T @ L[:, ax]))
    return np.concatenate(th)


def session_paths(npz, index_csv):
    z = np.load(npz, allow_pickle=True)
    paths = [str(p) for p in z["path"].tolist()]
    meta = load_index(index_csv, np.array(paths, object))
    groups = defaultdict(list)
    for i, p in enumerate(paths):
        if p in meta:
            groups[p.rsplit("/", 1)[0]].append(i)
    return paths, meta, {s: np.array(v, np.int64) for s, v in groups.items()
                         if len(v) >= 40}


def eval_huber_single(d, pred, sel, mu, lam, delta):
    q = np.setdiff1d(np.arange(len(pred)), sel)
    th, _ = fit_theta_huber(pred[sel], d["label"][sel], lam, mu, delta)
    return float(em(apply_theta_np(pred[q], th), d["label"][q],
                    d["sx"][q], d["sy"][q]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--index-dir", default=str(RGBDGaze_INDEX_DIR))
    ap.add_argument("--K", type=int, default=9)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1.345)
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--aux-task", choices=["jigsaw", "rotation"], default="jigsaw")
    ap.add_argument("--meta-loss", choices=["aux", "eye"], default="aux")
    ap.add_argument("--aux-classes", type=int, default=16)
    ap.add_argument("--aux-hidden", type=int, default=256)
    ap.add_argument("--pretrain-steps", type=int, default=300)
    ap.add_argument("--pretrain-batch", type=int, default=16)
    ap.add_argument("--pretrain-lr", type=float, default=1e-3)
    ap.add_argument("--meta-epochs", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=0,
                    help="0 = all source sessions per epoch")
    ap.add_argument("--inner-steps", type=int, default=2)
    ap.add_argument("--inner-lr", type=float, default=0.05)
    ap.add_argument("--meta-lr", type=float, default=1e-3)
    ap.add_argument("--prompt-lr", type=float, default=0.0,
                    help="0 = use --meta-lr; source prompt learning rate")
    ap.add_argument("--aux-lr", type=float, default=0.0,
                    help="0 = use --meta-lr; aux-head learning rate")
    ap.add_argument("--adapt-loss", choices=["aux", "gaze", "residual", "eye"], default="aux")
    ap.add_argument("--calib-mode", choices=["refit", "fixed"], default="refit")
    ap.add_argument("--target-steps", type=int, default=20)
    ap.add_argument("--target-lr", type=float, default=0.02)
    ap.add_argument("--target-anchor", type=float, default=0.0)
    ap.add_argument("--support", type=int, default=5)
    ap.add_argument("--target-support", type=int, default=0,
                    help="0 = use --support")
    ap.add_argument("--query", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-sessions", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=-1)
    a = ap.parse_args()

    torch.set_num_threads(4)
    device = "cuda" if (a.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    log(f"[vpt] start={time.strftime('%F %T')} device={device} args={vars(a)}")
    idx_dir = Path(a.index_dir)
    idx_csv = idx_dir / "index.csv"
    ds = RGBDGazeDataset(idx_csv, augment=False, image_size=224, eye_crops=True)
    table = TableIndex(idx_csv)
    lookup = table.lookup("rgb_path")

    tr_paths, tr_meta, tr_groups = session_paths(a.train_npz, str(idx_csv))
    te_paths, te_meta, te_groups = session_paths(a.test_npz, str(idx_csv))
    if a.max_sessions:
        te_groups = {k: te_groups[k] for k in list(te_groups)[:a.max_sessions]}
    log(f"[vpt] source sessions={len(tr_groups)} target sessions={len(te_groups)}")

    model = load_model(a.ckpt, device)
    prompt = nn.Parameter(torch.randn(1, a.prompt_len, 384, device=device) * 0.02)
    injector = PromptInjector(model, prompt)
    aux_head = nn.Sequential(
        nn.Linear(384, a.aux_hidden), nn.GELU(),
        nn.Linear(a.aux_hidden, a.aux_classes),
    ).to(device)
    perms = make_perms(a.aux_classes)

    # Helpers for an episode.
    def load_item(path):
        return ds[lookup[path]]

    def aux_batch(indices):
        items = [ds[i] for i in indices]
        face, eye_l, eye_r, _ = batch_items(items, device)
        if a.aux_task == "rotation":
            perm_ids = torch.randint(0, 4, (len(items),), device=device)
            jf = torch.stack([
                torch.rot90(face[k], int(perm_ids[k]), dims=(1, 2))
                for k in range(len(items))
            ])
        else:
            perm_ids = torch.randint(0, len(perms), (len(items),), device=device)
            jf = torch.stack([jigsaw_images(face[k:k + 1], perms[int(pid)])[0]
                              for k, pid in enumerate(perm_ids)])
        q, feat = feature_of(model, jf, eye_l, eye_r)
        return F.cross_entropy(aux_head(feat), perm_ids), q

    train_keys = list(tr_groups)
    plr = a.prompt_lr or a.meta_lr
    alr = a.aux_lr or a.meta_lr
    opt = torch.optim.AdamW([prompt], lr=plr, weight_decay=1e-4)
    aux_opt = torch.optim.AdamW(aux_head.parameters(), lr=alr,
                                weight_decay=1e-4)
    log(f"[vpt] diff_lr prompt={plr:g} aux={alr:g} target={a.target_lr:g}")

    # ---- Auxiliary-head pretraining on source images, prompt disabled ----
    if a.pretrain_steps > 0:
        injector.prompt = None
        pretrain_opt = torch.optim.AdamW(aux_head.parameters(),
                                         lr=a.pretrain_lr, weight_decay=1e-4)
        src_indices = [lookup[p] for p in tr_paths if p in lookup]
        rng = np.random.RandomState(7)
        t0 = time.time()
        acc_hist = []
        for step in range(1, a.pretrain_steps + 1):
            take = rng.choice(len(src_indices), min(a.pretrain_batch, len(src_indices)),
                              replace=False)
            items = [ds[src_indices[i]] for i in take]
            face, eye_l, eye_r, _ = batch_items(items, device)
            if a.aux_task == "rotation":
                perm_ids = torch.randint(0, 4, (len(items),), device=device)
                jf = torch.stack([
                    torch.rot90(face[k], int(perm_ids[k]), dims=(1, 2))
                    for k in range(len(items))
                ])
            else:
                perm_ids = torch.randint(0, len(perms), (len(items),), device=device)
                jf = torch.stack([jigsaw_images(face[k:k + 1], perms[int(pid)])[0]
                                  for k, pid in enumerate(perm_ids)])
            _, feat = feature_of(model, jf, eye_l, eye_r)
            logits = aux_head(feat)
            loss = F.cross_entropy(logits, perm_ids)
            pretrain_opt.zero_grad(set_to_none=True)
            loss.backward()
            pretrain_opt.step()
            acc = float((logits.argmax(1) == perm_ids).float().mean())
            acc_hist.append(acc)
            if step == 1 or step % 25 == 0 or step == a.pretrain_steps:
                log(f"[vpt] aux_pretrain={step}/{a.pretrain_steps} "
                    f"loss={float(loss):.4f} acc={acc:.3f} "
                    f"elapsed={time.time()-t0:.1f}s")
        injector.prompt = prompt
        log(f"[vpt] aux_pretrain_done mean_last50={np.mean(acc_hist[-50:]):.3f}")

    for ep in range(1, a.meta_epochs + 1):
        rng = np.random.RandomState(1000 + ep)
        keys = train_keys if not a.episodes else list(
            rng.choice(train_keys, min(a.episodes, len(train_keys)), replace=False))
        tot_g = tot_a = 0.0
        nb = 0
        t0 = time.time()
        for si, s in enumerate(keys):
            ii = tr_groups[s]
            if len(ii) < a.support + a.query:
                continue
            sel = rng.choice(len(ii), a.support + a.query, replace=False)
            sup_i = [lookup[tr_paths[ii[k]]] for k in sel[:a.support]]
            qry_i = [lookup[tr_paths[ii[k]]] for k in sel[a.support:]]
            # FOMAML: inner prompt update on unlabeled jigsaw support.
            p0 = prompt.detach().clone().requires_grad_(True)
            injector.prompt = p0
            for _ in range(a.inner_steps):
                if a.meta_loss == "eye":
                    sup_items = [ds[i] for i in sup_i]
                    mface, meye_l, meye_r, _ = batch_items(sup_items, device)
                    ql = model(mface, eye_l=meye_l, eye_r=meye_l)
                    qr = model(mface, eye_l=meye_r, eye_r=meye_r)
                    aux_loss = ((ql - qr) ** 2).mean()
                    g = torch.autograd.grad(aux_loss, p0, retain_graph=True)[0]
                else:
                    aux_loss, _ = aux_batch(sup_i)
                    g = torch.autograd.grad(aux_loss, p0, retain_graph=True)[0]
                    aux_opt.zero_grad(set_to_none=True)
                    aux_loss.backward(retain_graph=True)
                    aux_opt.step()
                p0 = (p0 - a.inner_lr * g).detach().requires_grad_(True)
                injector.prompt = p0
            # Outer: supervised 2D gaze on query; FOMAML meta-gradient.
            qitems = [ds[i] for i in qry_i]
            face, eye_l, eye_r, label = batch_items(qitems, device)
            q, _ = feature_of(model, face, eye_l, eye_r)
            loss_g = ((q - label) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss_g.backward()
            with torch.no_grad():
                prompt.grad = p0.grad.detach().clone()
            opt.step()
            injector.prompt = prompt
            tot_g += float(loss_g.detach())
            tot_a += float(aux_loss.detach())
            nb += 1
            if si % 10 == 0 or si == len(keys) - 1:
                log(f"[vpt] meta_ep={ep} session={s} "
                    f"gaze={float(loss_g):.6f} aux={float(aux_loss):.6f} "
                    f"elapsed={time.time()-t0:.1f}s")
        log(f"[vpt] meta_epoch={ep}/{a.meta_epochs} episodes={nb} "
            f"gaze={tot_g/max(nb,1):.6f} aux={tot_a/max(nb,1):.6f}")

    # Target adaptation + evaluation.
    mu_aff = source_prior(a.train_npz)
    log(f"[vpt] source affine prior={np.round(mu_aff,4).tolist()}")

    z2 = np.load(a.test_npz, allow_pickle=True)
    base_all = z2["pred"].astype(np.float64)
    label_all = z2["label"].astype(np.float64)
    rows = []
    acc_b = acc_a = acc_fixed = 0.0
    nsum = 0
    t0 = time.time()
    src_state = copy.deepcopy(prompt.detach())
    for si, (s, ii) in enumerate(te_groups.items()):
        tsup = a.target_support or a.support
        if a.adapt_loss == "residual":
            base_s = base_all[ii]
            lab_s = label_all[ii]
            sx_s = np.array([te_meta[te_paths[i]][2] for i in ii])
            sy_s = np.array([te_meta[te_paths[i]][3] for i in ii])
            dloc_s = dict(pred=base_s, label=lab_s, sx=sx_s, sy=sy_s)
            sup_idx = support_indices(dloc_s, a.K, np.random.RandomState(90000))
            tsup = len(sup_idx)
        else:
            sup_idx = np.arange(tsup)
        sup_i = [lookup[te_paths[ii[k]]] for k in sup_idx]
        prompt.data.copy_(src_state)
        p = prompt
        injector.prompt = p
        adapt_loss = None
        grad_norm = 0.0
        sup_items = [ds[i] for i in sup_i]
        sface, seye_l, seye_r, _ = batch_items(sup_items, device)
        sy = torch.tensor(label_all[ii[:tsup]], dtype=torch.float32,
                          device=device)
        theta_fixed = None
        if a.adapt_loss == "residual":
            theta_fixed = fit_theta_huber(
                base_all[ii[sup_idx]], label_all[ii[sup_idx]],
                a.lam, mu_aff, a.delta)[0]
        with torch.no_grad():
            q_before = model(sface, eye_l=seye_l, eye_r=seye_r)
        if a.adapt_loss in ("gaze", "residual"):
            th_t = (torch.tensor(theta_fixed, dtype=torch.float32, device=device)
                    if theta_fixed is not None else None)
            for step in range(a.target_steps):
                with torch.enable_grad():
                    sq = model(sface, eye_l=seye_l, eye_r=seye_r)
                    if th_t is not None:
                        sq = torch.stack([
                            th_t[0] * sq[:, 0] + th_t[1],
                            th_t[2] * sq[:, 1] + th_t[3],
                        ], dim=1)
                    adapt_loss = ((sq - sy) ** 2).mean()
                    total_loss = adapt_loss + a.target_anchor * ((p - src_state) ** 2).mean()
                    g = torch.autograd.grad(total_loss, p, create_graph=False)[0]
                grad_norm = float(g.norm())
                with torch.no_grad():
                    p -= a.target_lr * g
                injector.prompt = p
        elif a.adapt_loss == "eye":
            # Gaze-specific self-supervision: left-eye-only and right-eye-only
            # predictions should agree.  No target gaze labels are used.
            for step in range(a.target_steps):
                with torch.enable_grad():
                    q_l = model(sface, eye_l=seye_l, eye_r=seye_l)
                    q_r = model(sface, eye_l=seye_r, eye_r=seye_r)
                    adapt_loss = ((q_l - q_r) ** 2).mean()
                    total_loss = adapt_loss + a.target_anchor * ((p - src_state) ** 2).mean()
                    g = torch.autograd.grad(total_loss, p, create_graph=False)[0]
                grad_norm = float(g.norm())
                with torch.no_grad():
                    p -= a.target_lr * g
                injector.prompt = p
        else:
            for step in range(a.target_steps):
                adapt_loss, _ = aux_batch(sup_i)
                g = torch.autograd.grad(adapt_loss, p, create_graph=False)[0]
                grad_norm = float(g.norm())
                with torch.no_grad():
                    p -= a.target_lr * g
                injector.prompt = p
        injector.prompt = p
        with torch.no_grad():
            q_after = model(sface, eye_l=seye_l, eye_r=seye_r)
            if a.adapt_loss == "eye":
                eye_l_b = model(sface, eye_l=seye_l, eye_r=seye_l)
                eye_r_b = model(sface, eye_l=seye_r, eye_r=seye_r)
                adapt_loss = ((eye_l_b - eye_r_b) ** 2).mean()
        pred_delta = float((q_after - q_before).abs().mean())
        prompt_delta = float((p - src_state).norm())
        if a.adapt_loss in ("gaze", "residual"):
            if theta_fixed is not None:
                th_t2 = torch.tensor(theta_fixed, dtype=torch.float32, device=device)
                qb2 = torch.stack([th_t2[0]*q_before[:,0]+th_t2[1],
                                   th_t2[2]*q_before[:,1]+th_t2[3]],1)
                qa2 = torch.stack([th_t2[0]*q_after[:,0]+th_t2[1],
                                   th_t2[2]*q_after[:,1]+th_t2[3]],1)
            else:
                qb2, qa2 = q_before, q_after
            loss_before = float(((qb2 - sy) ** 2).mean())
            loss_after = float(((qa2 - sy) ** 2).mean())
        else:
            loss_before = loss_after = float("nan")
        log(f"[vpt] target adapt session={s} tsup={tsup} loss={a.adapt_loss} "
            f"final={float(adapt_loss):.6f} grad_norm={grad_norm:.3e} "
            f"prompt_delta={prompt_delta:.4f} pred_delta={pred_delta:.6f} "
            f"sup_loss_before={loss_before:.6f} after={loss_after:.6f}")
        preds = []
        for st in range(0, len(ii), a.batch):
            items = [ds[lookup[te_paths[i]]] for i in ii[st:st + a.batch]]
            face, eye_l, eye_r, _ = batch_items(items, device)
            with torch.no_grad():
                preds.append(model(face, eye_l=eye_l, eye_r=eye_r).cpu().numpy())
        q = np.concatenate(preds)
        injector.prompt = prompt
        prompt.data.copy_(src_state)

        base = base_all[ii]
        lab = label_all[ii]
        sx = np.array([te_meta[te_paths[i]][2] for i in ii])
        sy = np.array([te_meta[te_paths[i]][3] for i in ii])
        dloc = dict(pred=base, label=lab, sx=sx, sy=sy)
        bvals, avals, fvals = [], [], []
        eval_i = np.setdiff1d(np.arange(len(ii)), sup_idx)
        for seed in range(a.seeds):
            if a.calib_mode == "fixed" and theta_fixed is not None:
                thb = theta_fixed
                tha = theta_fixed
                qidx = eval_i
            else:
                rng = np.random.RandomState(90000 + seed)
                sel = support_indices(dloc, a.K, rng)
                qidx = np.setdiff1d(np.arange(len(ii)), sel)
                thb, _ = fit_theta_huber(base[sel], lab[sel], a.lam, mu_aff, a.delta)
                tha, _ = fit_theta_huber(q[sel], lab[sel], a.lam, mu_aff, a.delta)
            bvals.append(float(em(apply_theta_np(base[qidx], thb), lab[qidx],
                                  sx[qidx], sy[qidx]).mean()))
            avals.append(float(em(apply_theta_np(q[qidx], tha), lab[qidx],
                                  sx[qidx], sy[qidx]).mean()))
            fvals.append(float(em(apply_theta_np(q[qidx], thb), lab[qidx],
                                  sx[qidx], sy[qidx]).mean()))
        bmean, amean = float(np.mean(bvals)), float(np.mean(avals))
        fmean = float(np.mean(fvals))
        acc_b += bmean * len(ii)
        acc_a += amean * len(ii)
        acc_fixed += fmean * len(ii)
        nsum += len(ii)
        bd = float(em(base[eval_i], lab[eval_i], sx[eval_i], sy[eval_i]).mean())
        ad = float(em(q[eval_i], lab[eval_i], sx[eval_i], sy[eval_i]).mean())
        rows.append([s, len(ii), bd, ad, bmean, amean, fmean,
                     100 * (amean / bmean - 1)])
        if si % 2 == 0 or si == len(te_groups) - 1:
            log(f"[vpt] target session={s} n={len(ii)} direct(base/vpt)="
                f"{bd:.4f}/{ad:.4f} cal(base/vpt/fixed)="
                f"{bmean:.4f}/{amean:.4f}/{fmean:.4f} "
                f"({100*(amean/bmean-1):+.2f}%) elapsed={time.time()-t0:.1f}s")
    B, A, FIX = acc_b / nsum, acc_a / nsum, acc_fixed / nsum
    log(f"[vpt] RESULT K={a.K} base_cal={B:.4f} vpt_cal={A:.4f} "
        f"fixed_theta_q={FIX:.4f} vpt_vs_base={100*(A/B-1):+.2f}% "
        f"fixed_vs_base={100*(FIX/B-1):+.2f}%")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "n", "base_direct", "vpt_direct",
                    "base_cal", "vpt_cal", "fixed_cal", "pct"])
        w.writerows(rows)
    log(f"[vpt] saved {out}")

if __name__ == "__main__":
    main()
