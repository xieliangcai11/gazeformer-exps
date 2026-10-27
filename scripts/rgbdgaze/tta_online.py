"""在线 TTA：严格因果、逐块适配。

流程（每个会话）：
  1. 先用原始模型算一遍 base 预测（仅用于对照）；
  2. 按时间顺序读取帧，攒成 block（默认 100 帧）；
  3. 每个 block：
     - 用**过去**的窗口（默认 300 帧）适配 adapter；
     - 窗口内前 70% 适配、后 30% 做无标签门控；
     - 门控变差则回退；
     - 用当前 adapter 预测当前 block（block 不参与本次适配）；
  4. 评估：base vs online，帧加权/会话等权/最差/退化。

用法：
  python -m scripts.rgbdgaze.tta_online \
    --ckpt out/.../best.pt --split test \
    --block-size 100 --window 300 --steps 20 --lr 1e-4 \
    --w-sym 1 --w-moment 1 --gate --out out/.../online.csv
"""
import argparse
import csv
import gc
import json
import resource
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = ["src", "."]

from gazelab.models.dinov2_gaze import RGBDGazeDINOv2, load_state_dict_any_res  # noqa: E402
from gazelab.models.dinov2_gaze.tta import (  # noqa: E402
    TTAConfig, attach_adapter, _moment_loss, _flip_kw, _sym_loss,
    _photometric_4d,
)
from scripts.train import Subset, build_dataloader  # noqa: E402


def sessions(index_dir, split):
    paths = [ln.strip() for ln in (Path(index_dir) / f"{split}.txt").read_text(
        encoding="utf-8").splitlines() if ln.strip()]
    sess = defaultdict(list)
    for k, p in enumerate(paths):
        sess[str(p).rsplit("/", 1)[0]].append(k)
    return paths, {s: sorted(v) for s, v in sess.items()}


def infer_flags(sd):
    use_geom = any(k.startswith("proj_geom") for k in sd)
    use_eye = any(k.startswith("proj_eye") for k in sd)
    use_imu = any(k.startswith("proj_imu") for k in sd)
    orient = False
    if use_geom and "proj_geom.weight" in sd:
        orient = sd["proj_geom.weight"].shape[1] >= 4
    return use_geom, use_eye, use_imu, orient


def frame_list(inp, use_eye, use_geom, use_imu):
    out = []
    for b in range(inp.rgb.shape[0]):
        f = {"face": inp.rgb[b]}
        if use_eye:
            f["eye_l"] = inp.eye_l[b]
            f["eye_r"] = inp.eye_r[b]
        if use_geom:
            f["geom"] = inp.geom[b]
        if use_imu:
            f["imu"] = inp.imu[b]
        out.append(f)
    return out


def stack_frames(frames, idx, dev):
    keys = list(frames[0].keys())
    return {k: torch.stack([frames[int(i)][k] for i in idx]).to(dev).float() for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rgbdgaze")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--block-size", type=int, default=100)
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--w-moment", type=float, default=1.0)
    ap.add_argument("--w-photo", type=float, default=0.0)
    ap.add_argument("--w-sym", type=float, default=1.0)
    ap.add_argument("--w-reg", type=float, default=1e-3)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--gate-margin", type=float, default=0.0)
    ap.add_argument("--gate-alpha", type=float, default=0.75)
    ap.add_argument("--min-adapt", type=int, default=80)
    ap.add_argument("--prior", default="")
    ap.add_argument("--sessions", default="")
    ap.add_argument("--max-sessions", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = torch.device(a.device)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = st.get("model", st)
    use_geom, use_eye, use_imu, orient = infer_flags(sd)
    print(f"[online] model geom={use_geom} eye={use_eye} imu={use_imu} orient={orient} "
          f"res={a.resolution}")
    model = RGBDGazeDINOv2(unfreeze_last=0, use_geom=use_geom, use_geom_orient=orient,
                           use_imu=use_imu, use_eye=use_eye,
                           img_size=a.resolution).to(dev)
    miss, unexp, _ = load_state_dict_any_res(model, sd)
    print(f"[online] load missing={len(miss)} unexpected={len(unexp)}")
    if miss:
        print(f"[online] missing[:12]={list(miss)[:12]}")
    model.eval()

    dl = build_dataloader(a.dataset, a.split, a.index_dir, a.batch_size, False,
                          a.num_workers, augment=False, resolution=a.resolution,
                          eye_crops=use_eye, geom_orient=orient)
    paths, sess = sessions(a.index_dir, a.split)
    items = list(sess.items())
    if a.sessions:
        keys = [k.strip() for k in a.sessions.split(",") if k.strip()]
        items = [(s, v) for s, v in items if any(k in s for k in keys)]
    if a.max_sessions:
        items = items[:a.max_sessions]
    print(f"[online] pid={__import__('os').getpid()} "
          f"maxrss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024:.2f}GB")
    print(f"[online] sessions={len(items)} block={a.block_size} window={a.window} "
          f"steps={a.steps} lr={a.lr} rank={a.rank} bs={a.batch_size} "
          f"w_moment={a.w_moment} w_sym={a.w_sym} w_photo={a.w_photo} w_reg={a.w_reg} "
          f"gate={a.gate}(margin={a.gate_margin},alpha={a.gate_alpha}) min_adapt={a.min_adapt}")

    prior = {"mean": [0.5, 0.5], "std": [0.3, 0.3]}
    if a.prior and Path(a.prior).is_file():
        prior = json.loads(Path(a.prior).read_text(encoding="utf-8"))
    print(f"[online] prior mean={np.round(prior['mean'],4).tolist()} "
          f"std={np.round(prior['std'],4).tolist()}")

    cfg = TTAConfig(steps=a.steps, lr=a.lr, rank=a.rank, batch_size=a.batch_size,
                    w_moment=a.w_moment, w_cons=a.w_photo, w_geo=0.0,
                    w_sym=a.w_sym, w_reg=a.w_reg, holdout_frac=0.0,
                    early_stop=False, gate=False, seed=0)
    adapter = attach_adapter(model, cfg)
    params = list(adapter.parameters())
    for p in model.parameters():
        p.requires_grad_(False)

    rows = []
    all_e0, all_e1 = [], []
    for si, (s, idx) in enumerate(items):
        idx = sorted(idx, key=lambda i: int(dl.dataset.base.rows[dl.dataset.idces[i]].get(
            "frame_index", i)))
        sub = Subset(dl.dataset, idx)
        dl_s = torch.utils.data.DataLoader(sub, batch_size=a.batch_size, shuffle=False,
                                           num_workers=a.num_workers)
        # base predictions（adapter 零）
        with torch.no_grad():
            for p in params:
                p.zero_()
        base_pred, base_lab = [], []
        for inp, lab in dl_s:
            fl = frame_list(inp, use_eye, use_geom, use_imu)
            kb = {k: torch.stack([f[k] for f in fl]).to(dev) for k in fl[0]}
            with torch.no_grad():
                base_pred.append(model(**kb).cpu().numpy())
            base_lab.append(lab.numpy())
        base_pred = np.concatenate(base_pred); base_lab = np.concatenate(base_lab)
        # cm scale per frame
        sc = []
        for i in idx:
            r = dl.dataset.base.rows[dl.dataset.idces[i]]
            sc.append((float(r["screen_w"]) * float(r["cm_px_x"]),
                       float(r["screen_h"]) * float(r["cm_px_y"])))
        sc = np.array(sc)

        # 重新按 block 读取（避免保存所有 frames）
        online_pred = np.zeros_like(base_pred)
        queue = []
        block = []
        pos = 0
        gate_acc = gate_rej = 0
        opt = torch.optim.Adam(params, lr=a.lr)
        for inp, lab in dl_s:
            fl = frame_list(inp, use_eye, use_geom, use_imu)
            for f in fl:
                block.append(f)
                if len(block) < a.block_size:
                    continue
                # 预测当前 block（adapter 当前状态）
                with torch.no_grad():
                    kb = {k: torch.stack([f[k] for f in block]).to(dev) for k in block[0]}
                    online_pred[pos:pos + len(block)] = model(**kb).cpu().numpy()
                # 适配：用队列（过去帧）
                if len(queue) >= a.min_adapt:
                    n_adapt = int(0.7 * len(queue))
                    pool_a, pool_v = queue[:n_adapt], queue[n_adapt:]
                    state_before = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
                    def vloss(pool):
                        if not pool:
                            return float("nan")
                        m = min(a.batch_size, len(pool))
                        kw = stack_frames(pool, torch.arange(m), dev)
                        with torch.no_grad():
                            p1 = model(**kw)
                            ls = a.w_moment * _moment_loss(p1, prior)
                            if a.w_photo > 0:
                                ge = torch.Generator(device=dev).manual_seed(12345)
                                g = 1.0 + 0.05 * (torch.rand(m, 1, 1, 1, device=dev, generator=ge) * 2 - 1)
                                b = 0.05 * (torch.rand(m, 1, 1, 1, device=dev, generator=ge) * 2 - 1)
                                kw2 = {k: (_photometric_4d(v, g, b) if v.dim() == 4 else v)
                                       for k, v in kw.items()}
                                ls = ls + a.w_photo * ((p1 - model(**kw2)) ** 2).mean()
                            if a.w_sym > 0:
                                ls = ls + a.w_sym * _sym_loss(p1, model(**_flip_kw(kw)))
                        return float(ls)
                    v0 = vloss(pool_v)
                    for step in range(a.steps):
                        sel = torch.randint(0, len(pool_a), (min(a.batch_size, len(pool_a)),)).tolist()
                        kw = stack_frames(pool_a, sel, dev)
                        pred = model(**kw)
                        loss = a.w_moment * _moment_loss(pred, prior)
                        if a.w_photo > 0:
                            g = 1.0 + 0.05 * (torch.rand(len(sel), 1, 1, 1, device=dev) * 2 - 1)
                            b = 0.05 * (torch.rand(len(sel), 1, 1, 1, device=dev) * 2 - 1)
                            kw2 = {k: (_photometric_4d(v, g, b) if v.dim() == 4 else v)
                                   for k, v in kw.items()}
                            loss = loss + a.w_photo * ((pred - model(**kw2)) ** 2).mean()
                        if a.w_sym > 0:
                            loss = loss + a.w_sym * _sym_loss(pred, model(**_flip_kw(kw)))
                        loss = loss + a.w_reg * adapter.reg_loss()
                        opt.zero_grad(); loss.backward(); opt.step()
                    v1 = vloss(pool_v)
                    if a.gate:
                        accept = v1 <= v0 * (1.0 + a.gate_margin)
                        scale = a.gate_alpha if accept else 0.0
                        with torch.no_grad():
                            for name, p in adapter.named_parameters():
                                p.copy_(state_before[name] + scale * (p - state_before[name]))
                        if accept:
                            gate_acc += 1
                        else:
                            gate_rej += 1
                        if (not accept) or scale != 1.0:
                            # 参数被回退/收缩后，旧 Adam 动量不再匹配 ⇒ 重置
                            opt = torch.optim.Adam(params, lr=a.lr)
                        an = sum(float(p.detach().norm()) for p in params)
                        print(f"    [block] {s} n={len(queue)} v0={v0:.5f} v1={v1:.5f} "
                              f"accept={accept} scale={scale:.2f} adapter_norm={an:.5f}")
                    print(f"    [adapt] {s} block={pos//a.block_size} queue={len(queue)} "
                          f"steps={a.steps} loss={float(loss):.5f}")
                pos += len(block)
                queue.extend(block)
                if len(queue) > a.window:
                    queue = queue[-a.window:]
                block = []
        # 剩余不足一个 block
        if block:
            with torch.no_grad():
                kb = {k: torch.stack([f[k] for f in block]).to(dev) for k in block[0]}
                online_pred[pos:pos + len(block)] = model(**kb).cpu().numpy()
            pos += len(block)

        def em(p, l):
            return np.hypot((p[:, 0] - l[:, 0]) * sc[:, 0], (p[:, 1] - l[:, 1]) * sc[:, 1])
        e0 = em(base_pred, base_lab); e1 = em(online_pred, base_lab)
        all_e0.append(e0); all_e1.append(e1)
        rows.append((s, len(e0), float(e0.mean()), float(e1.mean())))
        print(f"[{si+1}/{len(items)}] {s} n={len(e0)} base={e0.mean():.4f} "
              f"online={e1.mean():.4f} ({100*(e1.mean()/e0.mean()-1):+.2f}%) "
              f"gate_acc={gate_acc} gate_rej={gate_rej} "
              f"maxrss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024:.2f}GB")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    R = np.array([[r[2], r[3]] for r in rows])
    print(f"\n[online] 会话等权 base={R[:,0].mean():.4f} online={R[:,1].mean():.4f} "
          f"({100*(R[:,1].mean()/R[:,0].mean()-1):+.2f}%)")
    E0 = np.concatenate(all_e0); E1 = np.concatenate(all_e1)
    print(f"[online] 帧加权 base={E0.mean():.4f} online={E1.mean():.4f} "
          f"({100*(E1.mean()/E0.mean()-1):+.2f}%) n={len(E0)}")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "n", "em_base", "em_online"])
        w.writerows(rows)
    print(f"[online] saved {out} maxrss="
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024:.2f}GB")


if __name__ == "__main__":
    main()
