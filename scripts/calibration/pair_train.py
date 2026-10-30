"""真实跨朝向配对训练（R1）。

每个样本是一对真实帧：
  source: (rgb, eyes, geom, screen_cm, label)
  target: (rgb, eyes, geom, screen_cm, label)

损失：
  L = MSE(f(src), y_src) + MSE(f(tgt), y_tgt)
      + lambda * MSE(f(tgt), T_{src->tgt}(f(src)))

T 使用两帧真实的屏幕几何/朝向，不使用合成 token。
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path[:0] = ["src", "."]
from gazelab.datasets import GazeCaptureDataset  # noqa: E402
from gazelab.models.dinov2_gaze import (  # noqa: E402
    RGBDGazeDINOv2, load_state_dict_any_res,
)

NOMINAL = {"1": 0.0, "2": math.pi, "3": -math.pi / 2, "4": math.pi / 2}
_RANK = 0


def log(msg):
    if _RANK == 0:
        print(msg, flush=True)


class PairDataset(Dataset):
    def __init__(self, index_csv, pairs_npz, image_size=224):
        self.ds = GazeCaptureDataset(index_csv, augment=False,
                                     image_size=image_size, eye_crops=True,
                                     geom_orient=True)
        z = np.load(pairs_npz, allow_pickle=True)
        self.src = z["src_row"].astype(np.int64)
        self.tgt = z["tgt_row"].astype(np.int64)
        self.dist = z["dist"].astype(np.float32)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        s = self.ds[int(self.src[i])]
        t = self.ds[int(self.tgt[i])]
        def pack(x):
            inp, lab = x
            return (inp["rgb"], inp["eye_l"], inp["eye_r"], inp["geom"],
                    inp["screen_cm"], lab)
        return (*pack(s), *pack(t), torch.tensor(self.dist[i]))


def rot_t(v, ang):
    c, s = torch.cos(ang), torch.sin(ang)
    return torch.stack([c * v[:, 0] + s * v[:, 1],
                        -s * v[:, 0] + c * v[:, 1]], 1)


def T_pair(p, scm_s, scm_t, c, t):
    dphi = torch.tensor([NOMINAL[str(int(x))] for x in t],
                        device=p.device) - torch.tensor(
        [NOMINAL[str(int(x))] for x in c], device=p.device)
    pts = torch.stack([(p[:, 0] - 0.5) * scm_s[:, 0],
                       (p[:, 1] - 0.5) * scm_s[:, 1]], 1)
    pts = rot_t(pts, dphi)
    return torch.stack([pts[:, 0] / scm_t[:, 0] + 0.5,
                        pts[:, 1] / scm_t[:, 1] + 0.5], 1)


def load_model(ckpt, device):
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    use_geom = "proj_geom.weight" in state
    geom_orient = use_geom and state["proj_geom.weight"].shape[1] == 7
    use_eye = "proj_eye.weight" in state
    use_imu = "proj_imu.weight" in state
    log(f"[pair] use_geom={use_geom} geom_orient={geom_orient} "
        f"use_eye={use_eye} use_imu={use_imu}")
    model = RGBDGazeDINOv2(
        gaze_dim=2, d_model=384, num_heads=8, num_layers=6,
        inter_dim=1536, freeze_dino=True, unfreeze_last=12,
        use_imu=use_imu, use_geom=use_geom, use_geom_orient=geom_orient,
        use_eye=use_eye, img_size=224,
    ).to(device)
    miss, unexp, resampled = load_state_dict_any_res(model, state)
    log(f"[pair] load missing={len(miss)} unexpected={len(unexp)} "
        f"resampled={resampled}")
    return model, use_geom, use_eye, use_imu, geom_orient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--val-pairs", default="")
    ap.add_argument("--index-dir", default="data/GazeCapture/index")
    ap.add_argument("--out", default="out/gazecapture/pair_train")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--backbone-lr-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lambda-eq", type=float, default=0.2)
    ap.add_argument("--lambda-ramp", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--val-limit", type=int, default=512)
    ap.add_argument("--val-pair-limit", type=int, default=256)
    ap.add_argument("--val-stride", type=int, default=300)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-cm-rise", type=float, default=0.1)
    ap.add_argument("--resume", default="")
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--distributed", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = bool(a.distributed or world > 1)
    rank = 0
    if distributed:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local)
        device = f"cuda:{local}"
    else:
        device = "cuda" if (a.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    global _RANK
    _RANK = rank
    log(f"[pair] device={device} rank={rank}/{world} args={vars(a)}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        with open(out / "args.json", "w", encoding="utf-8") as fh:
            json.dump(vars(a), fh, indent=2)
        mode = "a" if a.resume else "w"
        with open(out / "train_log.csv", mode, encoding="utf-8") as fh:
            if mode == "w":
                fh.write("step,loss,gaze,eq,lam,gn,lr,elapsed\n")
        with open(out / "val_log.csv", mode, encoding="utf-8") as fh:
            if mode == "w":
                fh.write("step,val_cm,val_pair_eq,d_cm\n")

    idx_dir = Path(a.index_dir)
    tr = PairDataset(idx_dir / "index.csv", a.pairs)
    sampler = (DistributedSampler(tr, num_replicas=world, rank=rank,
                                  shuffle=True, seed=a.seed)
               if distributed else None)
    dl = DataLoader(tr, batch_size=a.batch, sampler=sampler,
                    shuffle=(sampler is None), num_workers=a.num_workers,
                    pin_memory=True, drop_last=True)
    # standard val (single frames) for accuracy
    ds_va = GazeCaptureDataset(idx_dir / "index.csv", augment=False,
                               image_size=224, eye_crops=True,
                               geom_orient=True)
    import csv
    va = []
    with open(idx_dir / "index.csv", encoding="utf-8") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            if r.get("split") != "test" or i % a.val_stride:
                continue
            va.append(i)
            if len(va) >= a.val_limit:
                break
    dl_va = DataLoader(torch.utils.data.Subset(ds_va, va),
                       batch_size=a.batch, shuffle=False,
                       num_workers=a.num_workers, pin_memory=True)
    vp = None
    if a.val_pairs:
        vp_full = PairDataset(idx_dir / "index.csv", a.val_pairs)
        nvp = min(len(vp_full), a.val_pair_limit)
        vp = torch.utils.data.Subset(vp_full, list(range(nvp)))
        dl_vp = DataLoader(vp, batch_size=a.batch, shuffle=False,
                           num_workers=a.num_workers, pin_memory=True)
        log(f"[pair] train pairs={len(tr)} val pairs={len(vp)} val frames={len(va)}")
    else:
        log(f"[pair] train pairs={len(tr)} val frames={len(va)}")

    model, use_geom, use_eye, use_imu, _ = load_model(a.ckpt, device)
    if distributed:
        model = DDP(model, device_ids=[int(device.split(":")[1])],
                    find_unused_parameters=True)
    core = model.module if distributed else model
    if a.freeze_backbone:
        for p in core.rgb.parameters():
            p.requires_grad = False
        log("[pair] backbone frozen")
    opt = torch.optim.AdamW([
        {"params": [p for p in core.rgb.parameters() if p.requires_grad],
         "lr": a.lr * a.backbone_lr_ratio},
        {"params": [p for n, p in core.named_parameters()
                    if not n.startswith("rgb.")], "lr": a.lr},
    ], weight_decay=a.weight_decay)
    crit = nn.MSELoss()

    def fwd(rgb, el, er, geom, imu, scm):
        if use_eye:
            return model(rgb, imu=imu, geom=geom, eye_l=el, eye_r=er)
        return model(rgb, imu=imu, geom=geom)

    def run_pair(batch):
        (srgb, sel, ser, sgeom, sscm, slab,
         trgb, tel, ter, tgeom, tscm, tlab, distv) = batch
        srgb = srgb.to(device).float(); trgb = trgb.to(device).float()
        sel = sel.to(device).float(); ser = ser.to(device).float()
        tel = tel.to(device).float(); ter = ter.to(device).float()
        sgeom = sgeom.to(device).float(); tgeom = tgeom.to(device).float()
        sscm = sscm.to(device).float(); tscm = tscm.to(device).float()
        slab = slab.to(device).float(); tlab = tlab.to(device).float()
        imu_s = torch.zeros(len(srgb), 3, device=device)
        imu_t = torch.zeros(len(trgb), 3, device=device)
        ps = fwd(srgb, sel, ser, sgeom, imu_s, sscm)
        pt = fwd(trgb, tel, ter, tgeom, imu_t, tscm)
        c = sgeom[:, 3:7].argmax(1) + 1
        t = tgeom[:, 3:7].argmax(1) + 1
        tp = T_pair(ps.detach(), sscm, tscm, c, t)
        return ps, pt, slab, tlab, tp

    def eval_std():
        model.eval()
        s = n = 0
        with torch.no_grad():
            for inp, lab in dl_va:
                lab = lab.to(device).float()
                scm = inp["screen_cm"].to(device).float()
                p = fwd(inp["rgb"].to(device).float(),
                        inp["eye_l"].to(device).float(),
                        inp["eye_r"].to(device).float(),
                        inp["geom"].to(device).float(),
                        torch.zeros(len(lab), 3, device=device), scm)
                s += float((((p - lab) * scm).norm(dim=1)).sum())
                n += len(lab)
        model.train()
        return s / max(n, 1)

    def eval_pairs():
        if vp is None:
            return 0.0
        model.eval()
        s = n = 0
        with torch.no_grad():
            for batch in dl_vp:
                ps, pt, _, _, tp = run_pair(batch)
                s += float((pt - tp).norm(dim=1).sum())
                n += len(pt)
        model.train()
        return s / max(n, 1)

    start = 0
    base_cm = -1.0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        core.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        start = int(ck.get("step", 0))
        base_cm = float(ck.get("base_cm", -1.0))
        log(f"[pair] resumed {a.resume} step={start} base_cm={base_cm}")
    if base_cm < 0:
        base_cm = eval_std()
    log(f"[pair] BASE val_cm={base_cm:.4f}")

    step = start
    seen = 0
    t0 = time.time()
    best_cm = float("inf")
    best_step = -1
    for batch in dl:
        if seen < start:
            seen += 1
            continue
        seen += 1
        if step >= a.steps:
            break
        ps, pt, slab, tlab, tp = run_pair(batch)
        lg = 0.5 * (crit(ps, slab) + crit(pt, tlab))
        le = crit(pt, tp)
        lam = a.lambda_eq * min(1.0, step / max(1, a.lambda_ramp))
        loss = lg + lam * le
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(
            [p for p in core.parameters() if p.requires_grad], 1.0))
        opt.step()
        if step % a.log_every == 0:
            log(f"[pair] step={step} loss={loss.item():.5f} "
                f"gaze={lg.item():.5f} eq={le.item():.5f} lam={lam:.3f} "
                f"gn={gn:.3f} elapsed={time.time()-t0:.1f}s")
            if rank == 0:
                with open(out / "train_log.csv", "a", encoding="utf-8") as fh:
                    fh.write(f"{step},{loss.item():.6f},{lg.item():.6f},"
                             f"{le.item():.6f},{lam:.4f},{gn:.3f},"
                             f"{opt.param_groups[-1]['lr']:.3e},"
                             f"{time.time()-t0:.2f}\n")
        step += 1
        if step % a.val_every == 0:
            cm = eval_std()
            pe = eval_pairs()
            log(f"[pair] VAL step={step} val_cm={cm:.4f} pair_eq={pe:.4f} "
                f"d_cm={cm-base_cm:+.4f}")
            if rank == 0:
                torch.save({"model": core.state_dict(),
                            "optimizer": opt.state_dict(), "step": step,
                            "val_cm": cm, "pair_eq": pe,
                            "args": vars(a)}, out / "last.pt")
                with open(out / "val_log.csv", "a", encoding="utf-8") as fh:
                    fh.write(f"{step},{cm:.6f},{pe:.6f},{cm-base_cm:+.6f}\n")
                if cm < best_cm:
                    best_cm, best_step = cm, step
                    torch.save({"model": core.state_dict(),
                                "optimizer": opt.state_dict(), "step": step,
                                "val_cm": cm, "pair_eq": pe,
                                "args": vars(a)}, out / "best.pt")
                    log(f"[pair] BEST step={step} val_cm={cm:.4f}")
                if step % a.save_every == 0:
                    torch.save({"model": core.state_dict(),
                                "optimizer": opt.state_dict(), "step": step,
                                "val_cm": cm, "pair_eq": pe,
                                "args": vars(a)},
                               out / f"ckpt_step{step}.pt")
                    log(f"[pair] SAVED ckpt_step{step}.pt")
    if rank == 0:
        torch.save({"model": core.state_dict(),
                    "optimizer": opt.state_dict(), "step": step,
                    "args": vars(a)}, out / "last.pt")
    if (out / "best.pt").exists():
        ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        core.load_state_dict(ck["model"])
        log(f"[pair] restored best.pt step={ck.get('step')}")
    cm = eval_std(); pe = eval_pairs()
    log(f"[pair] FINAL step={step} best_step={best_step} "
        f"val_cm={cm:.4f} pair_eq={pe:.4f}")


if __name__ == "__main__":
    main()
