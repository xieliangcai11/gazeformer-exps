"""在「标准物理厘米」空间训练 2D 注视模型。

模型输出 (X, Y) cm：
  原点 = 屏幕中心，坐标轴 = 竖屏标准方向。
训练时直接 MSE；验证时再转换回当前屏幕归一化坐标，
以便与旧标准（归一化 L2 / cm 误差）比较。
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path[:0] = ["src", "."]
from gazelab.datasets import GazeCaptureDataset  # noqa: E402
from gazelab.models.dinov2_gaze import (  # noqa: E402
    RGBDGazeDINOv2, load_state_dict_any_res,
)

_RANK = 0


def log(msg):
    if _RANK == 0:
        print(msg, flush=True)


def load_model(ckpt, device):
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model = RGBDGazeDINOv2(
        gaze_dim=2, d_model=384, num_heads=8, num_layers=6,
        inter_dim=1536, freeze_dino=True, unfreeze_last=12,
        use_imu=False, use_geom=True, use_geom_orient=False,
        use_eye=True, img_size=224,
    ).to(device)
    tgt = model.state_dict()
    drop = [k for k, v in state.items()
            if k in tgt and tuple(v.shape) != tuple(tgt[k].shape)]
    for k in drop:
        state.pop(k)
    log(f"[cm] dropped shape-mismatch keys={drop}")
    miss, unexp, resampled = load_state_dict_any_res(model, state)
    log(f"[cm] load missing={len(miss)} unexpected={len(unexp)} "
        f"resampled={resampled}")
    return model


def split_indices(index_csv, split, stride=1, limit=0):
    out = []
    with open(index_csv, encoding="utf-8") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            if r.get("split") != split:
                continue
            if i % stride:
                continue
            out.append(i)
            if limit and len(out) >= limit:
                break
    return out


def to_norm(pred_cm, scm, phi):
    c, s = torch.cos(phi), torch.sin(phi)
    x = c * pred_cm[:, 0] - s * pred_cm[:, 1]
    y = s * pred_cm[:, 0] + c * pred_cm[:, 1]
    return torch.stack([x / scm[:, 0] + 0.5, y / scm[:, 1] + 0.5], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index-dir", default="data/GazeCapture/index")
    ap.add_argument("--out", default="out/gazecapture/cm_train")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--backbone-lr-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--aug-shift", type=float, default=0.05)
    ap.add_argument("--aug-scale", type=float, default=0.05)
    ap.add_argument("--aug-rotate", type=float, default=0.0)
    ap.add_argument("--val-stride", type=int, default=300)
    ap.add_argument("--val-limit", type=int, default=1024)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--max-cm-rise", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--resume", default="")
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--distributed", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = bool(a.distributed or world > 1)
    rank = 0
    if distributed:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank(); world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local)
        device = f"cuda:{local}"
    else:
        device = "cuda" if (a.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    global _RANK
    _RANK = rank
    log(f"[cm] device={device} rank={rank}/{world} args={vars(a)}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        with open(out / "args.json", "w", encoding="utf-8") as fh:
            json.dump(vars(a), fh, indent=2)
        mode = "a" if a.resume else "w"
        with open(out / "train_log.csv", mode, encoding="utf-8") as fh:
            if mode == "w":
                fh.write("step,loss,gn,lr,elapsed\n")
        with open(out / "val_log.csv", mode, encoding="utf-8") as fh:
            if mode == "w":
                fh.write("step,val_cm,val_norm,d_cm,ori1,ori2,ori3,ori4\n")

    idx_dir = Path(a.index_dir)
    tr = GazeCaptureDataset(idx_dir / "index.csv", augment=a.augment,
                            aug_shift=a.aug_shift, aug_scale=a.aug_scale,
                            aug_rotate=a.aug_rotate,
                            image_size=224, eye_crops=True,
                            geom_orient=False, target_space="cm")
    va = GazeCaptureDataset(idx_dir / "index.csv", augment=False,
                            image_size=224, eye_crops=True,
                            geom_orient=False, target_space="cm")
    tr_idx = split_indices(idx_dir / "index.csv", "train", 1, 0)
    va_idx = split_indices(idx_dir / "index.csv", "test",
                           a.val_stride, a.val_limit)
    log(f"[cm] train frames={len(tr_idx)} val frames={len(va_idx)}")
    sampler = (DistributedSampler(Subset(tr, tr_idx), num_replicas=world,
                                  rank=rank, shuffle=True, seed=a.seed)
               if distributed else None)
    dl = DataLoader(Subset(tr, tr_idx), batch_size=a.batch, sampler=sampler,
                    shuffle=(sampler is None), num_workers=a.num_workers,
                    pin_memory=True, drop_last=True)
    dl_va = DataLoader(Subset(va, va_idx), batch_size=a.batch, shuffle=False,
                       num_workers=a.num_workers, pin_memory=True)

    model = load_model(a.ckpt, device)
    if distributed:
        model = DDP(model, device_ids=[int(device.split(":")[1])],
                    find_unused_parameters=True)
    core = model.module if distributed else model
    if a.freeze_backbone:
        for p in core.rgb.parameters():
            p.requires_grad = False
        log("[cm] backbone frozen")
    opt = torch.optim.AdamW([
        {"params": [p for p in core.rgb.parameters() if p.requires_grad],
         "lr": a.lr * a.backbone_lr_ratio},
        {"params": [p for n, p in core.named_parameters()
                    if not n.startswith("rgb.")], "lr": a.lr},
    ], weight_decay=a.weight_decay)
    crit = nn.MSELoss()

    def fwd(inp):
        return model(inp["rgb"].to(device).float(),
                     geom=inp["geom"].to(device).float(),
                     eye_l=inp["eye_l"].to(device).float(),
                     eye_r=inp["eye_r"].to(device).float())

    def eval_val():
        model.eval()
        cm_s = norm_s = n = 0
        ob = {i: [0.0, 0] for i in (1, 2, 3, 4)}
        with torch.no_grad():
            for inp, lab in dl_va:
                lab = lab.to(device).float()
                scm = inp["screen_cm"].to(device).float()
                phi = inp["phi"].to(device).float()
                ln = inp["label_norm"].to(device).float()
                ori = inp["orientation"].to(device).long()
                p = fwd(inp)
                pn = to_norm(p, scm, phi)
                cm = ((pn - ln) * scm).norm(dim=1)
                nm = (pn - ln).norm(dim=1)
                cm_s += float(cm.sum()); norm_s += float(nm.sum()); n += len(cm)
                for o in ob:
                    m = ori == o
                    ob[o][0] += float(cm[m].sum()); ob[o][1] += int(m.sum())
        model.train()
        return (cm_s / max(n, 1), norm_s / max(n, 1),
                {o: (v[0] / v[1] if v[1] else float("nan"))
                 for o, v in ob.items()})

    start = 0
    base_cm = -1.0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        core.load_state_dict(ck["model"])
        if "optimizer" in ck:
            try:
                opt.load_state_dict(ck["optimizer"])
            except ValueError as e:
                log(f"[cm] skip optimizer state: {e}")
        start = int(ck.get("step", 0))
        base_cm = float(ck.get("base_cm", -1.0))
        log(f"[cm] resumed {a.resume} step={start} base_cm={base_cm}")
    if base_cm < 0:
        base_cm, base_norm, base_ori = eval_val()
    else:
        _, base_norm, base_ori = eval_val()
    log(f"[cm] BASE val_cm={base_cm:.4f} val_norm={base_norm:.4f} "
        f"ori={ {k: round(v,3) for k,v in base_ori.items()} }")

    step = start; seen = 0; t0 = time.time()
    base_lrs = [g["lr"] for g in opt.param_groups]
    best_cm = float("inf"); best_step = -1
    for inp, lab in dl:
        if seen < start:
            seen += 1; continue
        seen += 1
        if step >= a.steps:
            break
        if a.warmup > 0:
            scale = min(1.0, (step - start + 1) / a.warmup)
            for g, lr0 in zip(opt.param_groups, base_lrs):
                g["lr"] = lr0 * scale
        lab = lab.to(device).float()
        p = fwd(inp)
        loss = crit(p, lab)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(
            [p for p in core.parameters() if p.requires_grad], 1.0))
        opt.step()
        if step % a.log_every == 0:
            log(f"[cm] step={step} loss={loss.item():.5f} gn={gn:.3f} "
                f"lr={opt.param_groups[-1]['lr']:.2e} "
                f"elapsed={time.time()-t0:.1f}s")
            if rank == 0:
                with open(out / "train_log.csv", "a", encoding="utf-8") as fh:
                    fh.write(f"{step},{loss.item():.6f},{gn:.3f},"
                             f"{opt.param_groups[-1]['lr']:.3e},"
                             f"{time.time()-t0:.2f}\n")
        step += 1
        if step % a.val_every == 0:
            cm, nm, ob = eval_val()
            log(f"[cm] VAL step={step} val_cm={cm:.4f} val_norm={nm:.4f} "
                f"d_cm={cm-base_cm:+.4f} "
                f"ori={ {k: round(v,3) for k,v in ob.items()} }")
            if rank == 0:
                with open(out / "val_log.csv", "a", encoding="utf-8") as fh:
                    fh.write(f"{step},{cm:.6f},{nm:.6f},{cm-base_cm:+.6f},"
                             + ",".join(f"{ob[o]:.4f}" for o in (1,2,3,4))
                             + "\n")
                torch.save({"model": core.state_dict(),
                            "optimizer": opt.state_dict(), "step": step,
                            "val_cm": cm, "val_norm": nm,
                            "base_cm": base_cm, "args": vars(a)},
                           out / "last.pt")
                if cm < best_cm:
                    best_cm, best_step = cm, step
                    torch.save({"model": core.state_dict(),
                                "optimizer": opt.state_dict(), "step": step,
                                "val_cm": cm, "val_norm": nm,
                                "base_cm": base_cm, "args": vars(a)},
                               out / "best.pt")
                    log(f"[cm] BEST step={step} val_cm={cm:.4f}")
                if step % a.save_every == 0:
                    torch.save({"model": core.state_dict(),
                                "optimizer": opt.state_dict(), "step": step,
                                "val_cm": cm, "val_norm": nm,
                                "base_cm": base_cm, "args": vars(a)},
                               out / f"ckpt_step{step}.pt")
            stop = torch.tensor(
                [1 if cm > base_cm + a.max_cm_rise else 0],
                device=device, dtype=torch.int32)
            if distributed:
                dist.broadcast(stop, 0)
            if int(stop.item()):
                log(f"[cm] STOP val_cm rose above {a.max_cm_rise}cm")
                break
    if rank == 0:
        torch.save({"model": core.state_dict(),
                    "optimizer": opt.state_dict(), "step": step,
                    "base_cm": base_cm, "args": vars(a)}, out / "last.pt")
    if (out / "best.pt").exists():
        ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        core.load_state_dict(ck["model"])
        log(f"[cm] restored best.pt step={ck.get('step')}")
    cm, nm, ob = eval_val()
    log(f"[cm] FINAL step={step} best_step={best_step} "
        f"val_cm={cm:.4f} val_norm={nm:.4f} "
        f"ori={ {k: round(v,3) for k,v in ob.items()} }")


if __name__ == "__main__":
    main()
