"""残差等变探针训练（小步数、全日志、可随时停）。

目标：验证「残差等变损失」能否在不显著伤害精度的前提下降低反事实等变误差。
每一步记录：
  loss_gaze / loss_eq / total / grad_norm / lr / elapsed
每 --val-every 步记录：
  val_cm、val_eq、每个上下文对的 val_eq
满足以下条件才建议继续：
  val_eq 下降 >= --min-eq-drop，且 val_cm 上升 <= --max-cm-rise
"""
import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path[:0] = ["src", "."]
from gazelab.datasets import GazeCaptureDataset  # noqa: E402
from gazelab.models.dinov2_gaze import (  # noqa: E402
    RGBDGazeDINOv2, load_state_dict_any_res,
)

NOMINAL = {"1": 0.0, "2": math.pi, "3": -math.pi / 2, "4": math.pi / 2}


def log(msg):
    print(msg, flush=True)


def rot_t(v, ang):
    c, s = torch.cos(ang), torch.sin(ang)
    x, y = v[:, 0], v[:, 1]
    return torch.stack([c * x + s * y, -s * x + c * y], 1)


def target_geom(geom, t):
    """geom: [B,7] = [aspect, logw, logh, onehot4]; t: [B] int64."""
    c = geom[:, 3:7].argmax(1) + 1
    swap = ((c <= 2) != (t <= 2))
    g = geom.clone()
    g[:, 0] = torch.where(swap, 1.0 / geom[:, 0].clamp_min(1e-6), geom[:, 0])
    g[:, 1] = torch.where(swap, geom[:, 2], geom[:, 1])
    g[:, 2] = torch.where(swap, geom[:, 1], geom[:, 2])
    g[:, 3:7] = 0.0
    g[torch.arange(len(t)), 2 + t] = 1.0
    return g, c, swap


def to_target(v, screen_cm, c, t, swap, point=True):
    """v: points or residual vectors in source normalised frame -> target frame."""
    sc = screen_cm.clone()
    tc = screen_cm.clone()
    if swap.any():
        tc[swap] = torch.stack([screen_cm[swap, 1], screen_cm[swap, 0]], 1)
    dphi = torch.tensor([NOMINAL[str(int(x))] for x in t],
                        device=v.device) - torch.tensor(
        [NOMINAL[str(int(x))] for x in c], device=v.device)
    # points: remove centre, rotate in cm, restore target centre
    if point:
        pts = torch.stack([v[:, 0] * sc[:, 0] - 0.5 * sc[:, 0],
                           v[:, 1] * sc[:, 1] - 0.5 * sc[:, 1]], 1)
    else:
        pts = torch.stack([v[:, 0] * sc[:, 0],
                           v[:, 1] * sc[:, 1]], 1)
    pts = rot_t(pts, dphi)
    out = torch.stack([pts[:, 0] / tc[:, 0] + (0.5 if point else 0.0),
                       pts[:, 1] / tc[:, 1] + (0.5 if point else 0.0)], 1)
    return out, tc


def load_model(ckpt, device):
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    use_geom = "proj_geom.weight" in state
    geom_orient = use_geom and state["proj_geom.weight"].shape[1] == 7
    use_eye = "proj_eye.weight" in state
    use_imu = "proj_imu.weight" in state
    log(f"[probe] ckpt keys use_geom={use_geom} geom_orient={geom_orient} "
        f"use_eye={use_eye} use_imu={use_imu}")
    model = RGBDGazeDINOv2(
        gaze_dim=2, d_model=384, num_heads=8, num_layers=6,
        inter_dim=1536, freeze_dino=True, unfreeze_last=12,
        use_imu=use_imu, use_geom=use_geom, use_geom_orient=geom_orient,
        use_eye=use_eye, img_size=224,
    ).to(device)
    miss, unexp, resampled = load_state_dict_any_res(model, state)
    log(f"[probe] load missing={len(miss)} unexpected={len(unexp)} "
        f"resampled={resampled}")
    return model, use_geom, use_eye, use_imu, geom_orient


def split_indices(index_csv, split, stride=1, limit=0):
    out = []
    with open(index_csv, encoding="utf-8") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            if split and r.get("split", split) != split:
                continue
            if i % stride:
                continue
            out.append(i)
            if limit and len(out) >= limit:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index-dir", default="data/GazeCapture/index")
    ap.add_argument("--out", default="out/gazecapture/equiv_probe")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--backbone-lr-ratio", type=float, default=0.1)
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lambda-eq", type=float, default=0.5)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--train-limit", type=int, default=200000)
    ap.add_argument("--train-stride", type=int, default=5)
    ap.add_argument("--val-stride", type=int, default=200)
    ap.add_argument("--val-limit", type=int, default=1024)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--max-cm-rise", type=float, default=0.3)
    ap.add_argument("--min-eq-drop", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    log(f"[probe] args={vars(a)}")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if (a.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    log(f"[probe] device={device}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "args.json", "w", encoding="utf-8") as fh:
        json.dump(vars(a), fh, indent=2)

    idx_dir = Path(a.index_dir)
    ds_tr = GazeCaptureDataset(idx_dir / "index.csv", augment=False,
                               image_size=224, eye_crops=True,
                               geom_orient=True)
    ds_va = GazeCaptureDataset(idx_dir / "index.csv", augment=False,
                               image_size=224, eye_crops=True,
                               geom_orient=True)
    tr_idx = split_indices(idx_dir / "index.csv", "train",
                           a.train_stride, a.train_limit)
    va_idx = split_indices(idx_dir / "index.csv", "test",
                           a.val_stride, a.val_limit)
    log(f"[probe] train frames={len(tr_idx)} val frames={len(va_idx)}")
    dl_tr = DataLoader(Subset(ds_tr, tr_idx), batch_size=a.batch,
                       shuffle=True, num_workers=a.num_workers,
                       pin_memory=True, drop_last=True)
    dl_va = DataLoader(Subset(ds_va, va_idx), batch_size=a.batch,
                       shuffle=False, num_workers=a.num_workers,
                       pin_memory=True)
    model, use_geom, use_eye, use_imu, geom_orient = load_model(a.ckpt, device)

    if a.freeze_backbone:
        for p in model.rgb.parameters():
            p.requires_grad = False
        log("[probe] backbone frozen")
    backbone = [p for p in model.rgb.parameters() if p.requires_grad]
    head = [p for n, p in model.named_parameters()
            if not n.startswith("rgb.")]
    opt = torch.optim.AdamW([
        {"params": backbone, "lr": a.lr * a.backbone_lr_ratio},
        {"params": head, "lr": a.lr},
    ], weight_decay=a.weight_decay)
    crit = nn.MSELoss()

    def forward(inp):
        face = inp["rgb"].to(device).float()
        geom = inp["geom"].to(device).float() if use_geom else None
        imu = inp["imu"].to(device).float() if use_imu else None
        if use_eye:
            return model(face, imu=imu, geom=geom,
                         eye_l=inp["eye_l"].to(device).float(),
                         eye_r=inp["eye_r"].to(device).float())
        return model(face, imu=imu, geom=geom)

    def eval_val():
        model.eval()
        cm_sum = eq_sum = n = 0
        pair = defaultdict(lambda: [0.0, 0])
        with torch.no_grad():
            for inp, lab in dl_va:
                lab = lab.to(device).float()
                scm = inp["screen_cm"].to(device).float()
                p = forward(inp)
                cm = ((p - lab) * scm).norm(dim=1)
                geom = inp["geom"].to(device).float()
                c = geom[:, 3:7].argmax(1) + 1
                t = torch.where(c < 4, c + 1, torch.ones_like(c))
                g2, _, swap = target_geom(geom, t)
                inp2 = inp
                inp2["geom"] = g2
                p2 = forward(inp2)
                y2, _ = to_target(lab, scm, c, t, swap)
                eq = (p2 - y2).norm(dim=1)
                cm_sum += float(cm.sum())
                eq_sum += float(eq.sum())
                n += len(cm)
                for cc, tt, e in zip(c.tolist(), t.tolist(), eq.tolist()):
                    pair[f"{cc}->{tt}"][0] += e
                    pair[f"{cc}->{tt}"][1] += 1
        model.train()
        d = {k: v[0] / v[1] for k, v in pair.items()}
        return cm_sum / max(n, 1), eq_sum / max(n, 1), d

    base_cm, base_eq, base_pair = eval_val()
    log(f"[probe] BASE val_cm={base_cm:.4f} val_eq={base_eq:.4f} "
        f"pairs={ {k: round(v,4) for k,v in base_pair.items()} }")
    model.train()
    t0 = time.time()
    step = 0
    losses = []
    for inp, lab in dl_tr:
        if step >= a.steps:
            break
        lab = lab.to(device).float()
        scm = inp["screen_cm"].to(device).float()
        geom = inp["geom"].to(device).float()
        p = forward(inp)
        loss_g = crit(p, lab)
        c = geom[:, 3:7].argmax(1) + 1
        t = torch.where(c < 4, c + 1, torch.ones_like(c))
        g2, _, swap = target_geom(geom, t)
        inp2 = inp
        inp2["geom"] = g2
        p2 = forward(inp2)
        y2, _ = to_target(lab, scm, c, t, swap)
        r = p.detach() - lab
        tr, _ = to_target(r, scm, c, t, swap, point=False)
        loss_e = crit(p2 - y2, tr)
        loss = loss_g + a.lambda_eq * loss_e
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0))
        opt.step()
        losses.append(float(loss.item()))
        if step % a.log_every == 0:
            log(f"[probe] step={step} loss={loss.item():.5f} "
                f"gaze={loss_g.item():.5f} eq={loss_e.item():.5f} "
                f"gn={gn:.3f} lr={opt.param_groups[1]['lr']:.2e} "
                f"elapsed={time.time()-t0:.1f}s")
        step += 1
        if step % a.val_every == 0:
            cm, eq, pair = eval_val()
            log(f"[probe] VAL step={step} val_cm={cm:.4f} val_eq={eq:.4f} "
                f"d_cm={cm-base_cm:+.4f} d_eq={eq-base_eq:+.4f} "
                f"pairs={ {k: round(v,4) for k,v in pair.items()} }")
            torch.save({"model": model.state_dict(), "step": step,
                        "val_cm": cm, "val_eq": eq, "args": vars(a)},
                       out / "last.pt")
            if cm > base_cm + a.max_cm_rise:
                log(f"[probe] STOP val_cm rose above {a.max_cm_rise}cm")
                break
    cm, eq, pair = eval_val()
    torch.save({"model": model.state_dict(), "step": step,
                "val_cm": cm, "val_eq": eq, "args": vars(a)}, out / "last.pt")
    log(f"[probe] FINAL step={step} val_cm={cm:.4f} val_eq={eq:.4f} "
        f"d_cm={cm-base_cm:+.4f} d_eq={eq-base_eq:+.4f} "
        f"eq_drop={100*(1-eq/max(base_eq,1e-9)):.2f}%")
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(dict(base_cm=base_cm, base_eq=base_eq,
                       final_cm=cm, final_eq=eq, step=step,
                       pairs=pair), fh, indent=2)
    log(f"[probe] saved {out}/last.pt summary.json")


if __name__ == "__main__":
    main()
