"""Test-time augmentation (TTA) inference for the 2D gaze model.

Variants:
  identity
  horizontal flip (screen x -> 1-x)
  eye swap
  flip + eye swap
Average predictions, then optionally recalibrate with Huber + landmark ridge.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF

sys.path[:0] = ["src", "."]
from configs.rgbdgaze_config import RGBDGaze_INDEX_DIR  # noqa: E402
from gazelab.datasets import RGBDGazeDataset, GazeCaptureDataset  # noqa: E402
from gazelab.datasets.table_index import TableIndex  # noqa: E402
from gazelab.models.dinov2_gaze import (  # noqa: E402
    RGBDGazeDINOv2, load_state_dict_any_res,
)


def log(msg):
    print(msg, flush=True)


def load_model(ckpt, device, use_geom=False, geom_orient=False):
    model = RGBDGazeDINOv2(
        gaze_dim=2, d_model=384, num_heads=8, num_layers=6,
        inter_dim=1536, freeze_dino=True, unfreeze_last=0,
        use_imu=False, use_geom=use_geom, use_geom_orient=geom_orient,
        use_eye=True, img_size=224,
    ).to(device)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    miss, unexp, resampled = load_state_dict_any_res(model, state)
    log(f"[tta] load missing={len(miss)} unexpected={len(unexp)} resampled={resampled}")
    if miss or unexp:
        raise RuntimeError("checkpoint mismatch")
    model.eval()
    return model


def augment(x, mode):
    """Small geometric / photometric augmentations on a batch of images."""
    if mode == "scale90":
        c = int(x.shape[-1] * 0.90)
        return TF.resize(TF.center_crop(x, c), [x.shape[-2], x.shape[-1]])
    if mode == "scale110":
        c = int(x.shape[-1] * 1.10)
        return TF.resize(TF.center_crop(x, c), [x.shape[-2], x.shape[-1]])
    if mode == "rot5":
        return TF.rotate(x, 5.0)
    if mode == "rotm5":
        return TF.rotate(x, -5.0)
    if mode == "color":
        return TF.adjust_contrast(TF.adjust_brightness(x, 1.05), 1.05)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--feat-npz", required=True, help="test feature npz with paths/labels")
    ap.add_argument("--index-dir", default=str(RGBDGaze_INDEX_DIR))
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--variants", nargs="+",
                    default=["id", "flip", "swap", "flipswap"])
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--dataset", choices=["rgbdgaze", "gazecapture"], default="rgbdgaze")
    ap.add_argument("--use-geom", action="store_true")
    ap.add_argument("--geom-orient", action="store_true")
    a = ap.parse_args()

    torch.set_num_threads(4)
    device = "cuda" if (a.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    log(f"[tta] device={device} args={vars(a)}")
    idx_dir = Path(a.index_dir)
    if a.dataset == "gazecapture":
        ds = GazeCaptureDataset(idx_dir / "index.csv", augment=False,
                                image_size=224, eye_crops=True,
                                geom_orient=a.geom_orient)
    else:
        ds = RGBDGazeDataset(idx_dir / "index.csv", augment=False,
                             image_size=224, eye_crops=True)
    table = TableIndex(idx_dir / "index.csv")
    with open(idx_dir / "index.csv", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
    pcol = "frame_path" if "frame_path" in header else "rgb_path"
    lookup = table.lookup(pcol)
    log(f"[tta] path column={pcol}")

    z = np.load(a.feat_npz, allow_pickle=True)
    paths = [str(p) for p in z["path"].tolist()]
    label = z["label"].astype(np.float32)
    fidx = z["frame_index"].astype(np.int64)
    idxs = [lookup[p] for p in paths if p in lookup]
    keep = [i for i, p in enumerate(paths) if p in lookup]
    log(f"[tta] frames={len(idxs)} variants={a.variants}")
    model = load_model(a.ckpt, device, a.use_geom, a.geom_orient)

    out = np.zeros((len(idxs), 2), np.float32)
    t0 = time.time()
    for st in range(0, len(idxs), a.batch):
        sub = idxs[st:st + a.batch]
        items = [ds[i] for i in sub]
        face = torch.stack([x[0].rgb for x in items]).to(device).float()
        el = torch.stack([x[0].eye_l for x in items]).to(device).float()
        er = torch.stack([x[0].eye_r for x in items]).to(device).float()
        gm = (torch.stack([x[0].geom for x in items]).to(device).float()
              if a.use_geom else None)
        acc = []
        with torch.no_grad():
            for v in a.variants:
                if v == "id":
                    q = model(face, eye_l=el, eye_r=er, geom=gm)
                elif v == "flip":
                    q = model(torch.flip(face, [3]),
                              eye_l=torch.flip(er, [3]),
                              eye_r=torch.flip(el, [3]), geom=gm)
                    q = torch.stack([1.0 - q[:, 0], q[:, 1]], 1)
                elif v == "swap":
                    q = model(face, eye_l=er, eye_r=el, geom=gm)
                elif v == "flipswap":
                    q = model(torch.flip(face, [3]),
                              eye_l=torch.flip(el, [3]),
                              eye_r=torch.flip(er, [3]), geom=gm)
                    q = torch.stack([1.0 - q[:, 0], q[:, 1]], 1)
                else:
                    q = model(augment(face, v), eye_l=augment(el, v),
                              eye_r=augment(er, v), geom=gm)
                acc.append(q.cpu().numpy())
        out[st:st + len(sub)] = np.mean(acc, 0)
        if st % (a.batch * 25) == 0:
            log(f"[tta] {st}/{len(idxs)} elapsed={time.time()-t0:.1f}s")
    np.savez_compressed(a.out,
                        pred=out, label=label[keep],
                        path=np.array([paths[i] for i in keep], object),
                        frame_index=fidx[keep])
    e = np.hypot((out[:, 0] - label[keep, 0]),
                 (out[:, 1] - label[keep, 1])).mean()
    log(f"[tta] saved {a.out} frames={len(out)} normalized_err={e:.4f} "
        f"elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
