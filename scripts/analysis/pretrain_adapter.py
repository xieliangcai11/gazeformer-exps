"""源域监督预训练 adapter（给 TTA 一个更好的起点）。

训练阶段允许用源域标签；部署时仍然零标签。
输出：adapter state_dict（可传给 tta_personalize --init-adapter）。
"""
import argparse
import sys

import torch

sys.path[:0] = ["src", "."]

from gazelab.models.dinov2_gaze import RGBDGazeDINOv2, load_state_dict_any_res  # noqa: E402
from gazelab.models.dinov2_gaze.tta import TTAConfig, attach_adapter, detach_adapter  # noqa: E402
from scripts.train import build_dataloader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rgbdgaze")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--w-reg", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    st = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = st.get("model", st)
    use_geom = any(k.startswith("proj_geom") for k in sd)
    use_eye = any(k.startswith("proj_eye") for k in sd)
    use_imu = any(k.startswith("proj_imu") for k in sd)
    geom_orient = False
    if use_geom:
        w = sd.get("proj_geom.weight")
        geom_orient = bool(w is not None and w.shape[1] >= 4)
    side = {"use_eye": use_eye, "use_geom": use_geom, "geom_orient": geom_orient}
    print(f"[pretrain] 模型结构: geom={use_geom} eye={use_eye} imu={use_imu} geom_orient={geom_orient} res={args.resolution}")
    print(f"[pretrain] split={args.split} epochs={args.epochs} lr={args.lr} rank={args.rank} "
          f"w_reg={args.w_reg} bs={args.batch_size} max_batches={args.max_batches}")
    m = RGBDGazeDINOv2(unfreeze_last=0, use_geom=use_geom,
                       use_geom_orient=geom_orient, use_imu=use_imu,
                       use_eye=use_eye, img_size=args.resolution)
    miss, unexp, resampled = load_state_dict_any_res(m, st.get("model", st))
    print(f"[pretrain] load missing={len(miss)} unexpected={len(unexp)} resampled={resampled}")
    if miss:
        print(f"[pretrain] missing[:12]={list(miss)[:12]}")
    if unexp:
        print(f"[pretrain] unexpected[:12]={list(unexp)[:12]}")
    m.to(args.device).eval()
    for q in m.parameters():
        q.requires_grad_(False)

    cfg = TTAConfig(steps=1, lr=args.lr, rank=args.rank, w_moment=0.0,
                    w_cons=0.0, w_geo=0.0, w_sym=0.0, w_reg=args.w_reg,
                    early_stop=False, gate=False)
    adapter = attach_adapter(m, cfg)
    opt = torch.optim.Adam(adapter.parameters(), lr=args.lr)

    dl = build_dataloader(args.dataset, args.split, args.index_dir, args.batch_size,
                          True, args.num_workers, augment=True,
                          resolution=args.resolution,
                          eye_crops=side.get("use_eye", True),
                          geom_orient=side.get("geom_orient", False))
    steps = 0
    for ep in range(int(args.epochs)):
        for inp, lab in dl:
            kw = {"face": inp.rgb.to(args.device).float()}
            if use_eye:
                kw["eye_l"] = inp.eye_l.to(args.device).float()
                kw["eye_r"] = inp.eye_r.to(args.device).float()
            if use_geom:
                kw["geom"] = inp.geom.to(args.device).float()
            if use_imu:
                kw["imu"] = inp.imu.to(args.device).float()
            lab = lab.to(args.device).float()
            pred = m(**kw)
            mse = ((pred - lab) ** 2).mean()
            reg = args.w_reg * adapter.reg_loss()
            loss = mse + reg
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()
            steps += 1
            if steps == 1 or steps % 50 == 0:
                print(f"[pretrain] ep{ep} step{steps} mse={float(mse):.6f} reg={float(reg):.6f} "
                      f"loss={float(loss):.6f} grad={float(gn):.4f} "
                      f"|bias|={float(adapter.bias.abs().mean()):.6f}")
            if args.max_batches and steps >= args.max_batches:
                break
        if args.max_batches and steps >= args.max_batches:
            break
    detach_adapter(m)
    torch.save(adapter.state_dict(), args.out)
    print(f"[pretrain] saved {args.out} steps={steps} "
          f"norm={sum(float(v.norm()) for v in adapter.state_dict().values()):.6f}")


if __name__ == "__main__":
    main()
