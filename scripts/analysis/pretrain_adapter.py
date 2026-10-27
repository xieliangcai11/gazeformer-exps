"""源域监督预训练 adapter（给 TTA 一个更好的起点）。

训练阶段允许用源域标签；部署时仍然零标签。
输出：adapter state_dict（可传给 tta_personalize --init-adapter）。
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path[:0] = ["src", "."]

from gazelab.models.dinov2_gaze import RGBDGazeDINOv2, load_state_dict_any_res  # noqa: E402
from gazelab.models.dinov2_gaze.tta import TTAConfig, attach_adapter, detach_adapter  # noqa: E402


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

    side = {}
    p = Path(args.ckpt).parent / "train_args.json"
    if p.is_file():
        side = json.loads(p.read_text(encoding="utf-8"))
    m = RGBDGazeDINOv2(unfreeze_last=0, use_geom=side.get("use_geom", True),
                       use_geom_orient=side.get("geom_orient", False),
                       use_eye=side.get("use_eye", True), img_size=args.resolution)
    st = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    load_state_dict_any_res(m, st.get("model", st))
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
            if side.get("use_eye", True):
                kw["eye_l"] = inp.eye_l.to(args.device).float()
                kw["eye_r"] = inp.eye_r.to(args.device).float()
            if side.get("use_geom", True):
                kw["geom"] = inp.geom.to(args.device).float()
            lab = lab.to(args.device).float()
            pred = m(**kw)
            loss = ((pred - lab) ** 2).mean() + args.w_reg * adapter.reg_loss()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()
            steps += 1
            if steps % 50 == 0:
                print(f"[pretrain] ep{ep} step{steps} loss={float(loss):.5f}")
            if args.max_batches and steps >= args.max_batches:
                break
        if args.max_batches and steps >= args.max_batches:
            break
    detach_adapter(m)
    torch.save(adapter.state_dict(), args.out)
    print(f"[pretrain] saved {args.out}  steps={steps}")


if __name__ == "__main__":
    main()
