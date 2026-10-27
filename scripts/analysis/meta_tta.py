"""元学习 TTA 初始化（FOMAML 一阶近似）。

目标：在**源域被试**上学习 adapter 的初始状态 θ0，使得：
  从 θ0 出发，用**无标签**损失在目标会话上做少量内循环适配后，
  在**有标签**的留出帧上误差最小。

部署时（目标用户）：
  - 从 θ0 开始，用无标签损失适配；
  - 不使用任何当前用户的标签。

本脚本只用源域标签（训练阶段允许）。

用法（RGD）：
  python -m scripts.analysis.meta_tta \
    --dataset rgbdgaze --ckpt out/rgbdgaze/ablation/checkpoints/eye224_mgpu/best.pt \
    --split train --index-dir data/RGBDGaze/index \
    --inner-steps 20 --meta-iters 200 --out out/rgbdgaze/analysis/meta_init_rgd.pt
"""
import argparse
import json
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path[:0] = ["src", "."]

from gazelab.models.dinov2_gaze import RGBDGazeDINOv2, load_state_dict_any_res  # noqa: E402
from gazelab.models.dinov2_gaze.tta import (  # noqa: E402
    TTAConfig, attach_adapter, detach_adapter, _moment_loss, _flip_kw, _sym_loss,
)
from scripts.train import DATASET_SPECS, build_dataloader  # noqa: E402


def load_model(args, cfg_side):
    m = RGBDGazeDINOv2(unfreeze_last=0, use_geom=cfg_side.get("use_geom", True),
                       use_geom_orient=cfg_side.get("geom_orient", False),
                       use_eye=cfg_side.get("use_eye", True),
                       use_imu=cfg_side.get("use_imu", False),
                       img_size=args.resolution)
    st = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    miss, unexp, resampled = load_state_dict_any_res(m, st.get("model", st))
    print(f"[meta] load missing={len(miss)} unexpected={len(unexp)} resampled={resampled}")
    if miss:
        print(f"[meta] missing[:12]={list(miss)[:12]}")
    if unexp:
        print(f"[meta] unexpected[:12]={list(unexp)[:12]}")
    return m.to(args.device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="rgbdgaze")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--inner-steps", type=int, default=20)
    ap.add_argument("--inner-lr", type=float, default=1e-3)
    ap.add_argument("--meta-iters", type=int, default=200)
    ap.add_argument("--meta-lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--adapt-frames", type=int, default=400)
    ap.add_argument("--eval-frames", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--w-moment", type=float, default=1.0)
    ap.add_argument("--w-sym", type=float, default=1.0)
    ap.add_argument("--w-reg", type=float, default=1e-3)
    ap.add_argument("--max-sessions", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    spec = DATASET_SPECS[args.dataset]
    index_dir = args.index_dir
    side = {}
    for name in ("train_args.json",):
        p = __import__("pathlib").Path(args.ckpt).parent / name
        if p.is_file():
            side = json.loads(p.read_text(encoding="utf-8"))
    st_raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = st_raw.get("model", st_raw)
    use_geom = any(k.startswith("proj_geom") for k in sd)
    use_eye = any(k.startswith("proj_eye") for k in sd)
    use_imu = any(k.startswith("proj_imu") for k in sd)
    geom_orient = False
    if use_geom:
        w = sd.get("proj_geom.weight")
        geom_orient = bool(w is not None and w.shape[1] >= 4)
    side = {"use_eye": use_eye, "use_geom": use_geom, "geom_orient": geom_orient,
            "use_imu": use_imu}
    print(f"[meta] 模型结构: geom={use_geom} eye={use_eye} imu={use_imu} geom_orient={geom_orient}")

    dl = build_dataloader(args.dataset, args.split, index_dir, 64, False, 0,
                          orient="raw", eye_crops=use_eye, resolution=args.resolution,
                          geom_orient=geom_orient)
    ds = dl.dataset
    base = ds.base
    paths = [base.rows[int(i)][spec["path_col"]] for i in ds.idces]
    sess = defaultdict(list)
    for pos, p in enumerate(paths):
        sess[str(__import__("pathlib").Path(p).parent)].append(pos)
    for k in sess:
        sess[k].sort(key=lambda pos: int(base.rows[int(ds.idces[pos])].get("frame_index", pos)))
    keys = sorted(sess)
    if args.max_sessions and args.max_sessions < len(keys):
        rng = np.random.RandomState(args.seed)
        keys = list(rng.choice(keys, args.max_sessions, replace=False))
    print(f"[meta] 源域会话 {len(keys)}  split={args.split}")
    print(f"[meta] 超参 inner_steps={args.inner_steps} inner_lr={args.inner_lr} "
          f"meta_iters={args.meta_iters} meta_lr={args.meta_lr} rank={args.rank} "
          f"adapt_frames={args.adapt_frames} eval_frames={args.eval_frames} bs={args.batch_size}")

    model = load_model(args, side)
    for p in model.parameters():
        p.requires_grad_(False)

    cfg = TTAConfig(steps=args.inner_steps, lr=args.inner_lr, rank=args.rank,
                    batch_size=args.batch_size, w_moment=args.w_moment,
                    w_cons=0.0, w_geo=0.0, w_sym=args.w_sym, w_reg=args.w_reg,
                    holdout_frac=0.0, early_stop=False, gate=False)
    adapter = attach_adapter(model, cfg)
    # 用 adapter 的默认初始化（up=0, down~N(0,0.02)）作为元参数起点。
    # 注意：不能把 down 也清零，否则 up 的梯度恒为 0，元学习完全学不动。
    meta = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
    named_params = list(adapter.named_parameters())
    params = [p for _, p in named_params]

    gen = torch.Generator().manual_seed(args.seed)
    dev = torch.device(args.device)
    print(f"[meta] meta init norm={sum(float(v.norm()) for v in meta.values()):.6f}")

    def batch_from(idxs):
        items = [ds[i] for i in idxs]
        kw = {}
        allowed = {"rgb": "face"}
        if use_eye:
            allowed.update({"eye_l": "eye_l", "eye_r": "eye_r"})
        if use_geom:
            allowed["geom"] = "geom"
        if use_imu:
            allowed["imu"] = "imu"
        for k in items[0][0].keys():
            if k not in allowed:
                continue
            vals = [it[0][k] for it in items]
            kw[allowed[k]] = torch.stack(vals).to(dev).float()
        lab = torch.stack([it[1] for it in items]).to(dev).float()
        return kw, lab

    prior_labels = []
    for s in keys[: min(5, len(keys))]:
        rr = sess[s][:: max(1, len(sess[s]) // 100)]
        for i in rr:
            prior_labels.append(ds[i][1].numpy())
    prior_arr = np.stack(prior_labels)
    prior = {"mean": prior_arr.mean(0).tolist(),
             "std": (prior_arr.std(0) + 1e-6).tolist()}
    print(f"[meta] prior mean={np.round(prior['mean'],4).tolist()} "
          f"std={np.round(prior['std'],4).tolist()} n={len(prior_arr)}")

    t0 = time.time()
    for it in range(args.meta_iters):
        s = keys[torch.randint(len(keys), (1,), generator=gen).item()]
        idx = sess[s]
        n = len(idx)
        cut = max(args.inner_steps, int(0.7 * n))
        adapt_idx = idx[:cut]
        eval_idx = idx[cut:]
        if len(eval_idx) < 8:
            print(f"[meta] skip {s}: eval={len(eval_idx)} < 8")
            continue
        if len(adapt_idx) > args.adapt_frames:
            sel = torch.randperm(len(adapt_idx), generator=gen)[:args.adapt_frames].tolist()
            adapt_idx = [adapt_idx[i] for i in sel]
        if len(eval_idx) > args.eval_frames:
            sel = torch.randperm(len(eval_idx), generator=gen)[:args.eval_frames].tolist()
            eval_idx = [eval_idx[i] for i in sel]

        # 内循环：无标签适配
        with torch.no_grad():
            for k, v in meta.items():
                adapter.state_dict()[k].copy_(v)
            adapter.zero_grad(set_to_none=True)
        opt = torch.optim.Adam(params, lr=args.inner_lr)
        first_loss = float("nan")
        last_loss = float("nan")
        for step in range(args.inner_steps):
            bs = min(args.batch_size, len(adapt_idx))
            sel = torch.randint(len(adapt_idx), (bs,), generator=gen).tolist()
            kw, _ = batch_from([adapt_idx[i] for i in sel])
            pred = model(**kw)
            loss = cfg.w_moment * _moment_loss(pred, prior)
            if cfg.w_sym > 0:
                kwf = _flip_kw(kw)
                predf = model(**kwf)
                loss = loss + cfg.w_sym * _sym_loss(pred, predf)
            loss = loss + cfg.w_reg * adapter.reg_loss()
            if step == 0:
                first_loss = float(loss)
            last_loss = float(loss)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        # 外循环：有标签评估（FOMAML：用适配后参数的梯度）
        adapter.zero_grad(set_to_none=True)
        kw, lab = batch_from(eval_idx)
        pred = model(**kw)
        loss_out = ((pred - lab) ** 2).mean()
        loss_out.backward()
        with torch.no_grad():
            for (name, p) in named_params:
                g = p.grad.detach() if p.grad is not None else torch.zeros_like(p)
                meta[name] = meta[name] - args.meta_lr * g
        gn = sum(float(p.grad.norm()) for _, p in named_params if p.grad is not None)
        mn = sum(float(v.norm()) for v in meta.values())
        if it == 0 or (it + 1) % 5 == 0:
            print(f"[meta] iter {it+1}/{args.meta_iters} session={s.split('/')[-2]}/{s.split('/')[-1]} "
                  f"adapt={len(adapt_idx)} eval={len(eval_idx)} "
                  f"inner_first={first_loss:.6f} inner_last={last_loss:.6f} "
                  f"outer={float(loss_out):.6f} grad={gn:.4f} meta_norm={mn:.6f} "
                  f"t={time.time()-t0:.1f}s")

    detach_adapter(model)
    torch.save(meta, args.out)
    print(f"[meta] saved {args.out} keys={ {k: tuple(v.shape) for k, v in meta.items()} }")


if __name__ == "__main__":
    main()
