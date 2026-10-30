"""厘米空间训练（epoch 级续训，沿用之前 train.py 的方式）。

- 每个 epoch 结束验证并保存 last.pt / best.pt；
- 中断后从 ckpt["epoch"] + 1 开始；
- DistributedSampler.set_epoch 保证每个 epoch 的采样顺序可复现；
- 支持 DDP、梯度累积、warmup、归一化 + 厘米混合损失。
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, DistributedSampler, RandomSampler
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
    miss, unexp, resampled = load_state_dict_any_res(model, state)
    log(f"[cm] dropped={drop} missing={len(miss)} unexpected={len(unexp)}")
    return model


def data_stats(index_csv):
    from collections import Counter
    n = 0
    splits = Counter(); oris = Counter(); devs = Counter()
    bad_imu = 0; nan_cm = 0
    with open(index_csv, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            n += 1
            splits[r.get("split", "")] += 1
            oris[str(r.get("orientation", ""))] += 1
            devs[r.get("device", "")] += 1
            try:
                gx, gy = float(r["imu_x"]), float(r["imu_y"])
                cmx, cmy = float(r["cm_px_x"]), float(r["cm_px_y"])
            except Exception:
                continue
            if (gx * gx + gy * gy) < 0.05 ** 2:
                bad_imu += 1
            if cmx != cmx or cmy != cmy:
                nan_cm += 1
    return dict(n=n, splits=dict(splits), orientations=dict(oris),
                devices=dict(devs.most_common(20)), bad_imu=bad_imu,
                nan_cm=nan_cm)


def split_indices(index_csv, split):
    out = []
    with open(index_csv, encoding="utf-8") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            if r.get("split") == split:
                out.append(i)
    return out


def to_norm(pred_cm, scm, phi):
    c, s = torch.cos(phi), torch.sin(phi)
    x = c * pred_cm[:, 0] - s * pred_cm[:, 1]
    y = s * pred_cm[:, 0] + c * pred_cm[:, 1]
    return torch.stack([x / scm[:, 0] + 0.5, y / scm[:, 1] + 0.5], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--resume", default="")
    ap.add_argument("--index-dir", default="data/GazeCapture/index")
    ap.add_argument("--out", default="out/gazecapture/cm_epoch")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--limit-batches", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.4e-4)
    ap.add_argument("--backbone-lr-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--loss", choices=["cm", "norm_plus_cm"], default="norm_plus_cm")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--aug-shift", type=float, default=0.05)
    ap.add_argument("--aug-scale", type=float, default=0.05)
    ap.add_argument("--aug-rotate", type=float, default=0.0)
    ap.add_argument("--geom-randomize", type=float, default=0.0)
    ap.add_argument("--val-stride", type=int, default=300)
    ap.add_argument("--val-limit", type=int, default=512)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=0,
                    help=">0: mid-epoch validation every N optimizer steps")
    ap.add_argument("--max-norm-rise", type=float, default=0.01)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=3)
    ap.add_argument("--distributed", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.aug_rotate != 0:
        raise ValueError("cm 训练不允许 aug_rotate != 0（标签未旋转）")

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
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                      text=True).strip()
    except Exception:
        git = "unknown"
    log(f"[cm] device={device} rank={rank}/{world} git={git} args={vars(a)}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        with open(out / "args.json", "w", encoding="utf-8") as fh:
            json.dump(vars(a), fh, indent=2)
        mode = "a" if a.resume else "w"
        with open(out / "train_log.csv", mode, encoding="utf-8") as fh:
            if mode == "w":
                fh.write("epoch,batch,step,loss,lnorm,lcm,gn,lr,elapsed\n")
        with open(out / "val_log.csv", mode, encoding="utf-8") as fh:
            if mode == "w":
                fh.write("epoch,step,val_cm,val_norm,ori1,ori2,ori3,ori4\n")

    idx_dir = Path(a.index_dir)
    tr = GazeCaptureDataset(idx_dir / "index.csv", augment=a.augment,
                            aug_shift=a.aug_shift, aug_scale=a.aug_scale,
                            aug_rotate=a.aug_rotate,
                            geom_randomize=a.geom_randomize,
                            image_size=224, eye_crops=True,
                            geom_orient=False, target_space="cm")
    tr.set_aug_seed(a.seed)
    va = GazeCaptureDataset(idx_dir / "index.csv", augment=False,
                            image_size=224, eye_crops=True,
                            geom_orient=False, target_space="cm")
    tr_idx = split_indices(idx_dir / "index.csv", "train")
    va_idx = split_indices(idx_dir / "index.csv", "test")
    va_idx = va_idx[::a.val_stride][:a.val_limit]
    if rank == 0:
        stats = data_stats(idx_dir / "index.csv")
        log(f"[cm] train={len(tr_idx)} val={len(va_idx)} "
            f"augment={a.augment} geom_randomize={tr.geom_randomize} "
            f"aug_seed={tr._seed} imu_bad={stats['bad_imu']} "
            f"nan_cm={stats['nan_cm']} splits={stats['splits']} "
            f"oris={stats['orientations']} top_devices={stats['devices']}")
    tr_sub = Subset(tr, tr_idx)
    if distributed:
        sampler = DistributedSampler(tr_sub, num_replicas=world, rank=rank,
                                     shuffle=True, seed=a.seed)
    else:
        sampler = RandomSampler(tr_sub)
    dl = DataLoader(tr_sub, batch_size=a.batch, sampler=sampler,
                    num_workers=a.num_workers, pin_memory=True,
                    drop_last=True)
    dl_va = DataLoader(Subset(va, va_idx), batch_size=a.batch, shuffle=False,
                       num_workers=a.num_workers, pin_memory=True)

    if a.scratch:
        core = RGBDGazeDINOv2(
            gaze_dim=2, d_model=384, num_heads=8, num_layers=6,
            inter_dim=1536, freeze_dino=True, unfreeze_last=12,
            use_imu=False, use_geom=True, use_geom_orient=False,
            use_eye=True, img_size=224,
        ).to(device)
        log("[cm] scratch init")
    else:
        core = load_model(a.ckpt, device)
    model = DDP(core, device_ids=[int(device.split(":")[1])],
                find_unused_parameters=True) if distributed else core
    opt = torch.optim.AdamW([
        {"params": [p for p in core.rgb.parameters() if p.requires_grad],
         "lr": a.lr * a.backbone_lr_ratio},
        {"params": [p for n, p in core.named_parameters()
                    if not n.startswith("rgb.")], "lr": a.lr},
    ], weight_decay=a.weight_decay)
    nparam = sum(p.numel() for p in core.parameters())
    ntrain = sum(p.numel() for p in core.parameters() if p.requires_grad)
    log(f"[cm] params total={nparam/1e6:.2f}M trainable={ntrain/1e6:.2f}M")
    base_lrs = [g["lr"] for g in opt.param_groups]
    huber = nn.SmoothL1Loss(beta=1.0, reduction="none")

    def fwd(inp):
        return model(inp["rgb"].to(device).float(),
                     geom=inp["geom"].to(device).float(),
                     eye_l=inp["eye_l"].to(device).float(),
                     eye_r=inp["eye_r"].to(device).float())

    def eval_val():
        model.eval(); cm_s = nm_s = n = 0
        ob = {i: [0.0, 0, 0.0, 0.0] for i in (1, 2, 3, 4)}
        with torch.no_grad():
            for inp, _ in dl_va:
                scm = inp["screen_cm"].to(device).float()
                phi = inp["phi"].to(device).float()
                ln = inp["label_norm"].to(device).float()
                ori = inp["orientation"].to(device).long()
                pn = to_norm(fwd(inp), scm, phi)
                rcm = (pn - ln) * scm
                cm = rcm.norm(dim=1)
                nm = (pn - ln).norm(dim=1)
                cm_s += float(cm.sum()); nm_s += float(nm.sum()); n += len(cm)
                for o in ob:
                    m = ori == o
                    ob[o][0] += float(cm[m].sum()); ob[o][1] += int(m.sum())
                    ob[o][2] += float(rcm[m, 0].sum())
                    ob[o][3] += float(rcm[m, 1].sum())
        model.train()
        out = {}
        for o, v in ob.items():
            if v[1]:
                out[o] = (v[0] / v[1], v[1], v[2] / v[1], v[3] / v[1])
            else:
                out[o] = (float("nan"), 0, float("nan"), float("nan"))
        return cm_s / max(n, 1), nm_s / max(n, 1), out

    start_epoch = 0
    global_step = 0
    best_norm = float("inf")
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        core.load_state_dict(ck["model"])
        if "optimizer" in ck:
            try:
                opt.load_state_dict(ck["optimizer"])
            except ValueError as e:
                log(f"[cm] skip optimizer: {e}")
        start_epoch = int(ck.get("epoch", -1)) + 1
        global_step = int(ck.get("step", 0))
        best_norm = float(ck.get("best_norm", float("inf")))
        log(f"[cm] resume {a.resume} start_epoch={start_epoch} "
            f"step={global_step} best_norm={best_norm:.4f}")

    t0 = time.time()
    bad_epochs = 0
    for epoch in range(start_epoch, a.epochs):
        if distributed:
            sampler.set_epoch(epoch)
        tr.set_epoch(epoch)
        model.train()
        opt.zero_grad(set_to_none=True)
        ep_loss = 0.0; ep_n = 0; ep_t0 = time.time()
        for bi, (inp, lab) in enumerate(dl):
            if a.limit_batches and bi >= a.limit_batches:
                break
            scale = min(1.0, (global_step + 1) / max(1, a.warmup))
            for g, lr0 in zip(opt.param_groups, base_lrs):
                g["lr"] = lr0 * scale
            lab_cm = lab.to(device).float()
            lab_norm = inp["label_norm"].to(device).float()
            scm = inp["screen_cm"].to(device).float()
            phi = inp["phi"].to(device).float()
            valid = inp["imu_valid"].to(device).float()
            p = fwd(inp)
            if a.loss == "cm":
                lcm = huber(p, lab_cm).mean(dim=1)
                loss = ((lcm * valid).sum() / valid.sum().clamp_min(1.0))
                lnorm = torch.zeros((), device=device)
            else:
                pn = to_norm(p, scm, phi)
                lnorm = ((pn - lab_norm) ** 2).mean(dim=1)
                lcm = huber(p, lab_cm).mean(dim=1)
                w = valid
                loss = ((lnorm * w).sum() + 0.1 * (lcm * w).sum()) / w.sum().clamp_min(1.0)
            if not torch.isfinite(loss):
                log(f"[cm] NONFINITE loss epoch={epoch} batch={bi} "
                    f"loss={loss} lnorm={lnorm} lcm={lcm}")
                raise RuntimeError("non-finite loss")
            ep_loss += float(loss.item()); ep_n += 1
            (loss / a.accum).backward()
            if (bi + 1) % a.accum == 0:
                gn = float(torch.nn.utils.clip_grad_norm_(
                    [pp for pp in core.parameters() if pp.requires_grad], 1.0))
                opt.step(); opt.zero_grad(set_to_none=True)
            else:
                gn = -1.0
            global_step += 1
            if (bi + 1) % a.accum == 0 and (bi // a.accum) % max(1, a.log_every) == 0:
                elapsed = time.time() - t0
                sps = (bi + 1) * a.batch * world / max(elapsed, 1e-9)
                total_b = min(len(dl), a.limit_batches) if a.limit_batches else len(dl)
                eta = (total_b - bi - 1) * a.batch * world / max(sps, 1e-9) / 60
                log(f"[cm] epoch={epoch} batch={bi}/{len(dl)} "
                    f"step={global_step} loss={loss.item():.5f} "
                    f"lnorm={lnorm.mean().item():.5f} "
                    f"lcm={lcm.mean().item():.5f} gn={gn:.3f} "
                    f"lr={opt.param_groups[-1]['lr']:.2e} "
                    f"sps={sps:.1f} eta={eta:.1f}min "
                    f"elapsed={elapsed:.1f}s")
                if rank == 0:
                    with open(out / "train_log.csv", "a", encoding="utf-8") as fh:
                        fh.write(f"{epoch},{bi},{global_step},{loss.item():.6f},"
                                 f"{lnorm.mean().item():.6f},"
                                 f"{lcm.mean().item():.6f},"
                                 f"{gn:.4f},{opt.param_groups[-1]['lr']:.3e},"
                                 f"{time.time()-t0:.2f}\n")
            if a.val_every and global_step % a.val_every == 0:
                cm, nm, ob = eval_val()
                log(f"[cm] MID epoch={epoch} step={global_step} "
                    f"val_cm={cm:.4f} val_norm={nm:.4f} "
                    f"ori={ {k: round(v,3) for k,v in ob.items()} }")
        cm, nm, ob = eval_val()
        train_loss = ep_loss / max(ep_n, 1)
        log(f"[cm] EPOCH {epoch} train_loss={train_loss:.5f} "
            f"val_cm={cm:.4f} val_norm={nm:.4f} "
            f"epoch_time={(time.time()-ep_t0)/60:.1f}min")
        for o, v in ob.items():
            log(f"[cm] EPOCH {epoch} ori={o} n={v[1]} cm={v[0]:.4f} "
                f"bias_cm=({v[2]:+.3f},{v[3]:+.3f})")
        opt.zero_grad(set_to_none=True)
        improved = nm < best_norm
        if improved:
            best_norm = nm
            bad_epochs = 0
            log(f"[cm] BEST epoch={epoch} val_norm={nm:.4f}")
        else:
            bad_epochs += 1
            log(f"[cm] no improvement ({bad_epochs})")
        if rank == 0:
            state = {"model": core.state_dict(), "optimizer": opt.state_dict(),
                     "epoch": epoch, "step": global_step,
                     "best_norm": best_norm, "val_cm": cm, "val_norm": nm,
                     "args": vars(a)}
            torch.save(state, out / f"epoch_{epoch}.pt")
            torch.save(state, out / "last.pt")
            if improved:
                torch.save(state, out / "best.pt")
            with open(out / "val_log.csv", "a", encoding="utf-8") as fh:
                fh.write(f"{epoch},{global_step},{cm:.6f},{nm:.6f},"
                         + ",".join(f"{ob[o][0]:.4f}" for o in (1, 2, 3, 4))
                         + "\n")
        stop = torch.tensor([1 if bad_epochs >= 2 else 0],
                            device=device, dtype=torch.int32)
        if distributed:
            dist.broadcast(stop, 0)
        if int(stop.item()):
            log("[cm] stop: two epochs without improvement")
            break


if __name__ == "__main__":
    main()
