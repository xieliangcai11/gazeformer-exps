"""RGBDGaze 微调 v2：验证集、分阶段解冻、早停、best checkpoint、LR 衰减。

流程：
  1. 读取 fold 的 train/val/test；若 val 为空，从 train 按被试划出 val；
  2. 阶段 A：冻结 DINOv2，只训头（--head-epochs）；
  3. 阶段 B：解冻 12 block，backbone lr = ratio × head lr；
  4. warmup + cosine 衰减；
  5. 每 epoch 验证，保存 best，patience 早停；
  6. 训练结束加载 best，在 test 上评估一次。
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path[:0] = ["src", "."]
from gazelab.datasets import RGBDGazeDataset  # noqa: E402
from gazelab.datasets.table_index import TableIndex  # noqa: E402
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
        use_imu=False, use_geom=True, use_geom_orient=True,
        use_eye=True, img_size=224,
    ).to(device)
    miss, unexp, resampled = load_state_dict_any_res(model, state)
    log(f"[rgd] load missing={len(miss)} unexpected={len(unexp)} "
        f"resampled={resampled}")
    return model


def read_split(path):
    if not path.exists():
        return []
    return [ln.strip() for ln in open(path, encoding="utf-8") if ln.strip()]


def to_norm(pred_cm, scm, phi):
    c, s = torch.cos(phi), torch.sin(phi)
    x = c * pred_cm[:, 0] - s * pred_cm[:, 1]
    y = s * pred_cm[:, 0] + c * pred_cm[:, 1]
    return torch.stack([x / scm[:, 0] + 0.5, y / scm[:, 1] + 0.5], 1)


def make_opt(core, head_lr, ratio, wd, freeze_backbone):
    if freeze_backbone:
        for p in core.rgb.parameters():
            p.requires_grad = False
    else:
        for p in core.rgb.parameters():
            p.requires_grad = True
    return torch.optim.AdamW([
        {"params": [p for p in core.rgb.parameters() if p.requires_grad],
         "lr": head_lr * ratio},
        {"params": [p for n, p in core.named_parameters()
                    if not n.startswith("rgb.")], "lr": head_lr},
    ], weight_decay=wd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--scratch", action="store_true",
                    help="不加载 GC checkpoint，从随机初始化训练当前 cm+geom 模型")
    ap.add_argument("--index-csv", default="data/RGBDGaze/index/index_imu.csv")
    ap.add_argument("--split-dir", default="data/RGBDGaze/index/cv/fold3")
    ap.add_argument("--out", default="out/rgbdgaze/cm_ft_v2")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--head-epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--head-lr", type=float, default=2e-4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--backbone-lr-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--select-metric", default="cm", choices=["cm", "norm"])
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--limit-batches", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=3)
    ap.add_argument("--distributed", action="store_true")
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
        torch.cuda.set_device(local); device = f"cuda:{local}"
    else:
        device = "cuda" if (a.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    global _RANK
    _RANK = rank
    log(f"[rgd] device={device} rank={rank}/{world} args={vars(a)}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        with open(out / "args.json", "w", encoding="utf-8") as fh:
            json.dump(vars(a), fh, indent=2)
        with open(out / "train_log.csv", "w", encoding="utf-8") as fh:
            fh.write("epoch,batch,step,loss,lnorm,lcm,gn,lr,elapsed\n")
        with open(out / "val_log.csv", "w", encoding="utf-8") as fh:
            fh.write("epoch,step,split,norm,cm\n")

    # ---- data ----
    rows = TableIndex(a.index_csv)
    lut = rows.lookup("rgb_path")
    subj = {}
    with open(a.index_csv, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            subj[r["rgb_path"]] = r["subject"]
    train_paths = read_split(Path(a.split_dir) / "train.txt")
    val_paths = read_split(Path(a.split_dir) / "val.txt")
    test_paths = read_split(Path(a.split_dir) / "test.txt")
    if not val_paths:
        by_sub = defaultdict(list)
        for p in train_paths:
            by_sub[subj.get(p, "")].append(p)
        subs = sorted(s for s in by_sub if s)
        rng = np.random.RandomState(a.seed)
        rng.shuffle(subs)
        n_val = max(1, min(len(subs) - 1, int(round(len(subs) * a.val_frac))))
        val_subs = set(subs[:n_val])
        val_paths = [p for p in train_paths if subj.get(p, "") in val_subs]
        train_paths = [p for p in train_paths if subj.get(p, "") not in val_subs]
        log(f"[rgd] carved val subjects={len(val_subs)} frames={len(val_paths)}")
    tr_idx = [lut[p] for p in train_paths if p in lut]
    va_idx = [lut[p] for p in val_paths if p in lut]
    te_idx = [lut[p] for p in test_paths if p in lut]
    log(f"[rgd] splits train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")
    tr = RGBDGazeDataset(a.index_csv, augment=a.augment, eye_crops=True,
                         geom_orient=True, target_space="cm",
                         geom_randomize=0.0)
    va = RGBDGazeDataset(a.index_csv, augment=False, eye_crops=True,
                         geom_orient=True, target_space="cm",
                         geom_randomize=0.0)
    tr.set_aug_seed(a.seed)
    tr_sub = Subset(tr, tr_idx)
    sampler = (DistributedSampler(tr_sub, num_replicas=world, rank=rank,
                                  shuffle=True, seed=a.seed)
               if distributed else None)
    dl = DataLoader(tr_sub, batch_size=a.batch, sampler=sampler,
                    shuffle=(sampler is None), num_workers=a.num_workers,
                    pin_memory=True, drop_last=True)
    dl_va = DataLoader(Subset(va, va_idx), batch_size=a.batch, shuffle=False,
                       num_workers=a.num_workers, pin_memory=True)
    dl_te = DataLoader(Subset(va, te_idx), batch_size=a.batch, shuffle=False,
                       num_workers=a.num_workers, pin_memory=True)

    if a.scratch:
        core = RGBDGazeDINOv2(
            gaze_dim=2, d_model=384, num_heads=8, num_layers=6,
            inter_dim=1536, freeze_dino=True, unfreeze_last=12,
            use_imu=False, use_geom=True, use_geom_orient=True,
            use_eye=True, img_size=224,
        ).to(device)
        log("[rgd] scratch init (cm+geom, no GC checkpoint)")
        start_epoch = 0; step0 = 0; best = float("inf")
    else:
        if not a.ckpt:
            raise ValueError("需要 --ckpt 或 --scratch")
        core = load_model(a.ckpt, device)
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        core.load_state_dict(ck["model"])
        start_epoch = int(ck.get("epoch", -1)) + 1
        step0 = int(ck.get("step", 0))
        best = float(ck.get("best", float("inf")))
        log(f"[rgd] resume {a.resume} epoch={start_epoch} step={step0} "
            f"best={best}")
    else:
        start_epoch = 0; step0 = 0; best = float("inf")
    model = DDP(core, device_ids=[int(device.split(":")[1])],
                find_unused_parameters=True) if distributed else core

    freeze = start_epoch < a.head_epochs
    head_lr = a.head_lr if freeze else a.lr
    opt = make_opt(core, head_lr, a.backbone_lr_ratio, a.weight_decay, freeze)
    log(f"[rgd] stage={'head' if freeze else 'full'} stage_lr={head_lr}")
    huber = nn.SmoothL1Loss(beta=1.0, reduction="none")

    def fwd(inp):
        return model(inp["rgb"].to(device).float(),
                     geom=inp["geom"].to(device).float(),
                     eye_l=inp["eye_l"].to(device).float(),
                     eye_r=inp["eye_r"].to(device).float())

    def evaluate(loader):
        model.eval(); ns = cs = n = 0
        with torch.no_grad():
            for inp, _ in loader:
                scm = inp["screen_cm"].to(device).float()
                phi = inp["phi"].to(device).float()
                ln = inp["label_norm"].to(device).float()
                pn = to_norm(fwd(inp), scm, phi)
                cs += float(((pn - ln) * scm).norm(dim=1).sum())
                ns += float((pn - ln).norm(dim=1).sum())
                n += len(pn)
        model.train()
        return ns / max(n, 1), cs / max(n, 1)

    step = step0
    t0 = time.time()
    bad = 0
    total_steps = max(1, a.epochs * len(dl))
    for epoch in range(start_epoch, a.epochs):
        if epoch >= a.head_epochs and freeze:
            freeze = False
            opt = make_opt(core, a.lr, a.backbone_lr_ratio,
                           a.weight_decay, False)
            log(f"[rgd] stage switch -> full at epoch={epoch}")
        if distributed:
            sampler.set_epoch(epoch)
        tr.set_epoch(epoch)
        model.train(); opt.zero_grad(set_to_none=True)
        ep_loss = 0.0; ep_n = 0
        for bi, (inp, lab) in enumerate(dl):
            if a.limit_batches and bi >= a.limit_batches:
                break
            if step < a.warmup:
                scale = (step + 1) / max(1, a.warmup)
            else:
                prog = (step - a.warmup) / max(1, total_steps - a.warmup)
                prog = min(1.0, max(0.0, prog))
                scale = a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (
                    1 + math.cos(math.pi * prog))
            base = (a.head_lr if freeze else a.lr)
            for g in opt.param_groups:
                g["lr"] = base * scale * (
                    a.backbone_lr_ratio if g is opt.param_groups[0] else 1.0)
            lab_cm = lab.to(device).float()
            lab_norm = inp["label_norm"].to(device).float()
            scm = inp["screen_cm"].to(device).float()
            phi = inp["phi"].to(device).float()
            valid = inp["imu_valid"].to(device).float()
            p = fwd(inp)
            pn = to_norm(p, scm, phi)
            lnorm = (((pn - lab_norm) ** 2).mean(dim=1) * valid).sum() / \
                valid.sum().clamp_min(1.0)
            lcm = (huber(p, lab_cm).mean(dim=1) * valid).sum() / \
                valid.sum().clamp_min(1.0)
            loss = lnorm + 0.01 * lcm
            (loss / a.accum).backward()
            if (bi + 1) % a.accum == 0:
                gn = float(torch.nn.utils.clip_grad_norm_(
                    [pp for pp in core.parameters() if pp.requires_grad], 1.0))
                opt.step(); opt.zero_grad(set_to_none=True)
            else:
                gn = 0.0
            ep_loss += float(loss.item()); ep_n += 1
            step += 1
            if bi % a.log_every == 0:
                log(f"[rgd] epoch={epoch} batch={bi}/{len(dl)} step={step} "
                    f"loss={loss.item():.5f} lnorm={lnorm.item():.5f} "
                    f"lcm={lcm.item():.5f} gn={gn:.3f} "
                    f"lr={opt.param_groups[-1]['lr']:.2e} "
                    f"elapsed={time.time()-t0:.1f}s")
            if a.val_every and step % a.val_every == 0:
                nm, cm = evaluate(dl_va)
                log(f"[rgd] VAL step={step} norm={nm:.4f} cm={cm:.4f}")
        nm, cm = evaluate(dl_va)
        log(f"[rgd] EPOCH {epoch} train_loss={ep_loss/max(ep_n,1):.5f} "
            f"val_norm={nm:.4f} val_cm={cm:.4f}")
        score = cm if a.select_metric == "cm" else nm
        improved = score < best - 1e-6
        if improved:
            best = score; bad = 0
        else:
            bad += 1
        if rank == 0:
            state = {"model": core.state_dict(),
                     "optimizer": opt.state_dict(), "epoch": epoch,
                     "step": step, "best": best, "val_norm": nm,
                     "val_cm": cm, "args": vars(a)}
            torch.save(state, out / "last.pt")
            with open(out / "val_log.csv", "a", encoding="utf-8") as fh:
                fh.write(f"{epoch},{step},val,{nm:.6f},{cm:.6f}\n")
            if improved:
                torch.save(state, out / "best.pt")
                log(f"[rgd] BEST epoch={epoch} score={best:.4f}")
        stop = torch.tensor([1 if bad >= a.patience else 0],
                            device=device, dtype=torch.int32)
        if distributed:
            dist.broadcast(stop, 0)
        if int(stop.item()):
            log(f"[rgd] early stop bad={bad}")
            break

    if (out / "best.pt").exists():
        ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        core.load_state_dict(ck["model"])
        log(f"[rgd] restored best epoch={ck.get('epoch')}")
    nm, cm = evaluate(dl_va)
    nt, ct = evaluate(dl_te)
    log(f"[rgd] FINAL val_norm={nm:.4f} val_cm={cm:.4f} "
        f"test_norm={nt:.4f} test_cm={ct:.4f}")
    if rank == 0:
        with open(out / "val_log.csv", "a", encoding="utf-8") as fh:
            fh.write(f"final,{step},test,{nt:.6f},{ct:.6f}\n")
        torch.save({"model": core.state_dict(), "epoch": a.epochs - 1,
                    "step": step, "val_norm": nm, "val_cm": cm,
                    "test_norm": nt, "test_cm": ct, "args": vars(a)},
                   out / "final.pt")


if __name__ == "__main__":
    main()
