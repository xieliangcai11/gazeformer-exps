"""厘米空间模型推理，并转换回当前屏幕归一化坐标（兼容旧评测）。"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path[:0] = ["src", "."]
from gazelab.datasets import GazeCaptureDataset  # noqa: E402
from gazelab.models.dinov2_gaze import (  # noqa: E402
    RGBDGazeDINOv2, load_state_dict_any_res,
)


def log(msg):
    print(msg, flush=True)


def load_model(ckpt, device):
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    use_geom_orient = ("proj_geom.weight" in state
                       and state["proj_geom.weight"].shape[1] == 7)
    log(f"[cmi] geom_orient={use_geom_orient}")
    model = RGBDGazeDINOv2(
        gaze_dim=2, d_model=384, num_heads=8, num_layers=6,
        inter_dim=1536, freeze_dino=True, unfreeze_last=0,
        use_imu=False, use_geom=True, use_geom_orient=use_geom_orient,
        use_eye=True, img_size=224,
    ).to(device)
    tgt = model.state_dict()
    drop = [k for k, v in state.items()
            if k in tgt and tuple(v.shape) != tuple(tgt[k].shape)]
    for k in drop:
        state.pop(k)
    miss, unexp, resampled = load_state_dict_any_res(model, state)
    log(f"[cmi] dropped={drop} missing={len(miss)} unexpected={len(unexp)}")
    model.eval()
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    a = ap.parse_args()
    device = "cuda" if (a.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    log(f"[cmi] args={vars(a)} device={device}")
    idx_dir = Path(a.index_dir)
    ds = GazeCaptureDataset(idx_dir / "index.csv", augment=False,
                            image_size=224, eye_crops=True,
                            geom_orient=True, target_space="cm")
    idx = split_indices(idx_dir / "index.csv", a.split, a.stride, a.limit)
    if a.num_shards > 1:
        idx = idx[a.shard::a.num_shards]
    log(f"[cmi] shard={a.shard}/{a.num_shards}")
    log(f"[cmi] frames={len(idx)}")
    dl = DataLoader(Subset(ds, idx), batch_size=a.batch, shuffle=False,
                    num_workers=4, pin_memory=True)
    model = load_model(a.ckpt, device)
    preds, labels, paths, fidx = [], [], [], []
    cm_s = nm_s = n = 0
    by = defaultdict(lambda: [0.0, 0])
    with open(idx_dir / "index.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for inp, lab in dl:
      with torch.no_grad():
        p = model(inp["rgb"].to(device).float(),
                  geom=inp["geom"].to(device).float(),
                  eye_l=inp["eye_l"].to(device).float(),
                  eye_r=inp["eye_r"].to(device).float())
        pn = to_norm(p, inp["screen_cm"].to(device).float(),
                     inp["phi"].to(device).float()).cpu()
        ln = inp["label_norm"].float()
        scm = inp["screen_cm"].float()
        cm = ((pn - ln) * scm).norm(dim=1)
        nm = (pn - ln).norm(dim=1)
        ori = inp["orientation"].long()
        cm_s += float(cm.sum()); nm_s += float(nm.sum()); n += len(cm)
        for o in (1, 2, 3, 4):
            m = ori == o
            by[o][0] += float(cm[m].sum()); by[o][1] += int(m.sum())
        preds.append(pn.numpy())
        labels.append(ln.numpy())
    # Rebuild paths deterministically from idx order
    paths = [rows[i]["frame_path"] for i in idx]
    fidx = np.array([int(float(rows[i].get("frame_index", 0)))
                     for i in idx], np.int64)
    pred = np.concatenate(preds, 0)
    label = np.concatenate(labels, 0)
    log(f"[cmi] RESULT frames={len(pred)} norm={nm_s/n:.4f} cm={cm_s/n:.4f} "
        f"ori={ {o: round(v[0]/max(v[1],1),3) for o,v in by.items()} }")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, pred=pred, label=label,
                        path=np.array(paths, object), frame_index=fidx)
    log(f"[cmi] saved {out}")


if __name__ == "__main__":
    main()
