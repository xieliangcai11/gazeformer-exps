"""RGBDGaze 统一评估：归一化模型 / cm 模型，同一 fold test。

输出：
  overall norm / cm
  per activity
  per subject
日志与 npz 同时保存，便于后续诊断。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path[:0] = ["src", "."]
from gazelab.datasets import RGBDGazeDataset  # noqa: E402
from gazelab.datasets.table_index import TableIndex  # noqa: E402
from gazelab.models.dinov2_gaze import (  # noqa: E402
    RGBDGazeDINOv2, load_state_dict_any_res,
)


def log(msg):
    print(msg, flush=True)


def load_model(ckpt, device):
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    has_geom = "proj_geom.weight" in state
    geom_dim = int(state["proj_geom.weight"].shape[1]) if has_geom else 0
    geom_orient = has_geom and geom_dim == 7
    log(f"[eval] ckpt_geom={has_geom} geom_dim={geom_dim} "
        f"geom_orient={geom_orient} keys={len(state)}")
    model = RGBDGazeDINOv2(
        gaze_dim=2, d_model=384, num_heads=8, num_layers=6,
        inter_dim=1536, freeze_dino=True, unfreeze_last=0,
        use_imu=False, use_geom=has_geom,
        use_geom_orient=geom_orient, use_eye=True, img_size=224,
    ).to(device)
    tgt = model.state_dict()
    drop = [k for k, v in state.items()
            if k in tgt and tuple(v.shape) != tuple(tgt[k].shape)]
    for k in drop:
        state.pop(k)
    miss, unexp, resampled = load_state_dict_any_res(model, state)
    log(f"[eval] dropped={drop} missing={len(miss)} unexpected={len(unexp)} "
        f"resampled={resampled}")
    model.eval()
    return model, has_geom, geom_orient


def read_split(p):
    return [ln.strip() for ln in open(p, encoding="utf-8") if ln.strip()]


def to_norm(pred_cm, scm, phi):
    c, s = torch.cos(phi), torch.sin(phi)
    x = c * pred_cm[:, 0] - s * pred_cm[:, 1]
    y = s * pred_cm[:, 0] + c * pred_cm[:, 1]
    return torch.stack([x / scm[:, 0] + 0.5, y / scm[:, 1] + 0.5], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index-csv", default="data/RGBDGaze/index/index_imu.csv")
    ap.add_argument("--split-dir", default="data/RGBDGaze/index/cv/fold3")
    ap.add_argument("--split", default="test", choices=["test", "train", "val"])
    ap.add_argument("--target-space", default="norm", choices=["norm", "cm"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    a = ap.parse_args()
    device = "cuda" if (a.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    log(f"[eval] args={vars(a)} device={device}")
    rows = TableIndex(a.index_csv)
    lut = rows.lookup("rgb_path")
    paths = read_split(Path(a.split_dir) / f"{a.split}.txt")
    idx = [lut[p] for p in paths if p in lut]
    if a.stride > 1:
        idx = idx[::a.stride]
    if a.limit:
        idx = idx[:a.limit]
    if a.num_shards > 1:
        idx = idx[a.shard::a.num_shards]
    log(f"[eval] split={a.split} paths={len(paths)} matched={len(idx)} "
        f"shard={a.shard}/{a.num_shards}")
    ds = RGBDGazeDataset(
        a.index_csv, augment=False, eye_crops=True,
        geom_orient=True, target_space=a.target_space, geom_randomize=0.0)
    dl = DataLoader(Subset(ds, idx), batch_size=a.batch, shuffle=False,
                    num_workers=a.num_workers, pin_memory=True)
    model, has_geom, geom_orient = load_model(a.ckpt, device)

    act_codes, act_uniq = rows.code_column("activity")
    subj_codes, subj_uniq = rows.code_column("subject")
    metrics = defaultdict(lambda: [0.0, 0.0, 0])
    preds, labels = [], []
    with torch.inference_mode():
        for bi, (inp, _) in enumerate(dl):
            scm = inp["screen_cm"].to(device).float()
            phi = inp["phi"].to(device).float()
            ln = inp["label_norm"].to(device).float()
            kw = {}
            if has_geom:
                kw["geom"] = inp["geom"].to(device).float()
            p = model(inp["rgb"].to(device).float(),
                      eye_l=inp["eye_l"].to(device).float(),
                      eye_r=inp["eye_r"].to(device).float(), **kw)
            pn = to_norm(p, scm, phi) if a.target_space == "cm" else p
            cm = ((pn - ln) * scm).norm(dim=1)
            nm = (pn - ln).norm(dim=1)
            start = bi * a.batch
            for j in range(len(cm)):
                k = idx[start + j]
                for grp, code, uniq in (
                        ("activity", act_codes[k], act_uniq),
                        ("subject", subj_codes[k], subj_uniq)):
                    name = uniq[int(code)]
                    m = metrics[f"{grp}:{name}"]
                    m[0] += float(cm[j]); m[1] += float(nm[j]); m[2] += 1
                m = metrics["all"]
                m[0] += float(cm[j]); m[1] += float(nm[j]); m[2] += 1
            preds.append(pn.cpu().numpy()); labels.append(ln.cpu().numpy())
            if bi % 100 == 0:
                log(f"[eval] batch={bi}/{len(dl)} n={metrics['all'][2]}")
    pred = np.concatenate(preds, 0); label = np.concatenate(labels, 0)
    result = {"args": vars(a), "overall": {}, "groups": {}}
    for key, m in metrics.items():
        rec = {"n": int(m[2]), "cm": m[0] / m[2], "norm": m[1] / m[2]}
        if key == "all":
            result["overall"] = rec
        else:
            result["groups"][key] = rec
        log(f"[eval] {key:24s} n={m[2]:6d} cm={m[0]/m[2]:.4f} "
            f"norm={m[1]/m[2]:.4f}")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if a.num_shards > 1:
        out = out.with_name(out.name + f"_shard{a.shard}")
    np.savez_compressed(out.with_suffix(".npz"), pred=pred, label=label,
                        idx=np.array(idx))
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    log(f"[eval] saved {out.with_suffix('.npz')} / .json")


if __name__ == "__main__":
    main()
