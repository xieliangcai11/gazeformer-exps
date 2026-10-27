"""上下文校准器：用头姿 + 屏幕尺寸修正模型输出。

推断时输入：base prediction (2) + pose (8) + screen (2)
训练时用源域标签；推断时不用任何目标真值。

用法（同域）：
  python -m scripts.analysis.train_context_calib \
    --train-npz ... --val-npz ... --test-npz ... \
    --index-csv ... --pose-npz ... --variant both

跨域：
  用 GC 的 train/val，RGD 的 test（index/pose 也要换）
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path[:0] = ["src", "."]


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return dict(pred=z["pred"].astype(np.float32), label=z["label"].astype(np.float32),
                path=z["path"])


def load_meta(index_csv, needed):
    need = set(needed.tolist())
    out = {}
    with open(index_csv, encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        pcol = "frame_path" if "frame_path" in rd.fieldnames else "rgb_path"
        for r in rd:
            p = r[pcol]
            if p not in need:
                continue
            out[p] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                      float(r["screen_h"]) * float(r["cm_px_y"]),
                      r.get("device", ""))
    return out


def load_pose(pose_npz, index_csv, needed):
    pz = np.load(pose_npz, allow_pickle=True)
    row_idx = pz["row_idx"].astype(np.int64)
    feats = pz["feats"].astype(np.float32)
    valid = pz["valid"].astype(bool)
    pose_row = np.full((int(row_idx.max()) + 1, 8), np.nan, np.float32)
    valid_row = np.zeros(int(row_idx.max()) + 1, bool)
    pose_row[row_idx] = feats
    valid_row[row_idx] = valid
    need = set(needed.tolist())
    out = {}
    with open(index_csv, encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        pcol = "frame_path" if "frame_path" in rd.fieldnames else "rgb_path"
        for i, r in enumerate(rd):
            p = r[pcol]
            if p not in need:
                continue
            if i < len(pose_row) and valid_row[i]:
                out[p] = pose_row[i]
    return out


def build(d, meta, pose, use_pose, use_screen, shuffle_pose=False, seed=0):
    ok = [i for i, p in enumerate(d["path"]) if p in meta and p in pose]
    ok = np.array(ok, np.int64)
    pred = d["pred"][ok]
    lab = d["label"][ok]
    pose_arr = np.stack([pose[d["path"][i]] for i in ok])
    screen = np.stack([[meta[d["path"][i]][0], meta[d["path"][i]][1]] for i in ok])
    if shuffle_pose:
        rng = np.random.RandomState(seed)
        pose_arr = pose_arr[rng.permutation(len(pose_arr))]
    sx = np.array([meta[d["path"][i]][0] for i in ok])
    sy = np.array([meta[d["path"][i]][1] for i in ok])
    parts = [pred]
    if use_pose:
        parts.append(pose_arr)
    if use_screen:
        parts.append(screen)
    X = np.concatenate(parts, 1).astype(np.float32)
    return dict(X=X, pred=pred, lab=lab, sx=sx, sy=sy, n=len(ok))


class Calib(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def em(pred, lab, sx, sy):
    return np.hypot((pred[:, 0] - lab[:, 0]) * sx, (pred[:, 1] - lab[:, 1]) * sy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--val-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--pose-npz", required=True)
    ap.add_argument("--test-index-csv", default="",
                    help="跨数据集时 test 的索引（默认同 train）")
    ap.add_argument("--test-pose-npz", default="",
                    help="跨数据集时 test 的 pose（默认同 train）")
    ap.add_argument("--variant", choices=["pred", "pose", "screen", "both"], default="both")
    ap.add_argument("--shuffle-pose", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    D, V, T = load_npz(a.train_npz), load_npz(a.val_npz), load_npz(a.test_npz)
    need_tr = np.concatenate([D["path"], V["path"]])
    meta = load_meta(a.index_csv, need_tr)
    pose = load_pose(a.pose_npz, a.index_csv, need_tr)
    test_index = a.test_index_csv or a.index_csv
    test_pose = a.test_pose_npz or a.pose_npz
    meta_te = load_meta(test_index, T["path"])
    pose_te = load_pose(test_pose, test_index, T["path"])
    print(f"[calib] frames train/val/test={len(D['pred'])}/{len(V['pred'])}/{len(T['pred'])}")
    print(f"[calib] train meta/pose={len(meta)}/{len(pose)}  "
          f"test meta/pose={len(meta_te)}/{len(pose_te)}  test_index={test_index}")
    use_pose = a.variant in ("pose", "both")
    use_screen = a.variant in ("screen", "both")
    dtr = build(D, meta, pose, use_pose, use_screen, a.shuffle_pose, 0)
    dva = build(V, meta, pose, use_pose, use_screen, a.shuffle_pose, 1)
    dte = build(T, meta_te, pose_te, use_pose, use_screen, a.shuffle_pose, 2)
    print(f"[calib] usable train/val/test={dtr['n']}/{dva['n']}/{dte['n']}")

    mu = dtr["X"].mean(0); sd = dtr["X"].std(0) + 1e-6
    def norm(x): return (x - mu) / sd
    Xtr = torch.from_numpy(norm(dtr["X"])).to(a.device)
    Ytr = torch.from_numpy(dtr["pred"]).to(a.device)
    Ltr = torch.from_numpy(dtr["lab"]).to(a.device)
    model = Calib(dtr["X"].shape[1]).to(a.device)
    print(f"[calib] variant={a.variant} d_in={dtr['X'].shape[1]} "
          f"params={sum(p.numel() for p in model.parameters())}")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)

    def eval_split(d, name):
        X = torch.from_numpy(norm(d["X"])).to(a.device)
        with torch.no_grad():
            r = model(X).cpu().numpy()
        pred = d["pred"] + r
        e0 = em(d["pred"], d["lab"], d["sx"], d["sy"])
        e1 = em(pred, d["lab"], d["sx"], d["sy"])
        print(f"[{name}] base={e0.mean():.4f} calib={e1.mean():.4f} "
              f"({100*(e1.mean()/e0.mean()-1):+.2f}%) "
              f"改善 {int((e1<e0).sum())}/{len(e0)}")
        return e0.mean(), e1.mean()

    for ep in range(a.epochs):
        perm = torch.randperm(len(Xtr), device=a.device)
        tot = 0.0; nb = 0
        for b in range(0, len(Xtr), a.batch_size):
            idx = perm[b:b+a.batch_size]
            r = model(Xtr[idx])
            pred = Ytr[idx] + r
            loss = torch.nn.functional.mse_loss(pred, Ltr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"[train] ep{ep+1}/{a.epochs} loss={tot/max(nb,1):.6f}")
            eval_split(dva, "val")
    eval_split(dte, "test")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "mu": mu, "sd": sd,
                "args": vars(a)}, a.out)
    print(f"[save] {a.out}")


if __name__ == "__main__":
    main()
