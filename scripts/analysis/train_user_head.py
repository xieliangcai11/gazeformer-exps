"""头姿条件 + 用户嵌入的小模型（特征级）。

输入（全部来自冻结主干，不需要改主干）：
- 目标帧特征 h (D)；
- 目标帧头姿代理 pose (P=8)；
- 同一用户前 K 帧的**无标签**特征/头姿 → 用户向量 z (Z)。

变体：
  none : h -> gaze
  pose : [h, pose] -> gaze
  user : [h, z] -> gaze
  both : [h, pose, z] -> gaze

训练用源域标签；部署时目标用户只用无标签支持帧。
评估：每个用户前 K 帧作支持，其余帧评估。

用法：
  python -m scripts.analysis.train_user_head \
    --dataset gazecapture \
    --train-npz out/.../feats_gc_train300.npz \
    --val-npz out/.../feats_gc_val_m2_full.npz \
    --test-npz out/.../feats_gc_test_m2_full.npz \
    --pose-npz out/gazecapture/analysis/pose_features.npz \
    --index-csv data/GazeCapture/index/index.csv \
    --variant both --support-k 8 --epochs 20
"""
import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path[:0] = ["src", "."]


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return dict(feats=z["feats"].astype(np.float32),
                label=z["label"].astype(np.float32),
                pred=z["pred"].astype(np.float32) if "pred" in z else None,
                path=z["path"], subject=z["subject"].astype(str),
                frame_index=z["frame_index"].astype(np.int64))


def build_pose_map(index_csv, needed_paths, pose_npz):
    """path -> (pose[8], scale_x, scale_y)。"""
    need = set(needed_paths.tolist())
    pz = np.load(pose_npz, allow_pickle=True)
    row_idx = pz["row_idx"].astype(np.int64)
    pfeat = pz["feats"].astype(np.float32)
    valid = pz["valid"].astype(bool)
    pose_by_row = np.full((int(row_idx.max()) + 1, 8), np.nan, np.float32)
    valid_by_row = np.zeros(int(row_idx.max()) + 1, bool)
    pose_by_row[row_idx] = pfeat
    valid_by_row[row_idx] = valid

    out = {}
    with open(index_csv, encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        pcol = "frame_path" if "frame_path" in rd.fieldnames else "rgb_path"
        for i, r in enumerate(rd):
            p = r[pcol]
            if p not in need:
                continue
            if i >= len(pose_by_row) or not valid_by_row[i]:
                continue
            sx = float(r["screen_w"]) * float(r["cm_px_x"])
            sy = float(r["screen_h"]) * float(r["cm_px_y"])
            ctx = r.get("orientation")
            if ctx is None or ctx == "":
                ctx = r.get("activity", "")
            out[p] = (pose_by_row[i], sx, sy, str(ctx))
    return out


class UserHead(nn.Module):
    def __init__(self, d_feat, d_pose, z_dim=32, hidden=512, use_pose=True,
                 use_user=True, d_enc=1536, cond="concat", n_users=0):
        super().__init__()
        self.use_pose = use_pose
        self.use_user = use_user
        self.cond = cond
        self.d_feat, self.d_pose = d_feat, d_pose
        if use_user:
            self.enc_norm = nn.LayerNorm(2 * d_enc + 2 * d_pose)
            self.enc = nn.Sequential(
                nn.Linear(2 * d_enc + 2 * d_pose, 256), nn.ReLU(),
                nn.Linear(256, z_dim), nn.ReLU())
            if n_users:
                self.cls = nn.Linear(z_dim, n_users)
            if cond == "film":
                self.film = nn.Linear(z_dim, 2 * d_feat)
            elif cond == "bias":
                self.bias = nn.Linear(z_dim, 2)
        in_dim = d_feat + (d_pose if use_pose else 0)
        if use_user and cond == "concat":
            in_dim += z_dim
        self.pred_norm = nn.LayerNorm(in_dim)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 256), nn.ReLU(),
            nn.Linear(256, 2))

    def encode(self, sfeat, spose):
        # sfeat: [B,K,D], spose: [B,K,P]
        m = torch.cat([sfeat.mean(1), sfeat.std(1),
                       spose.mean(1), spose.std(1)], dim=1)
        return self.enc(self.enc_norm(m))

    def forward(self, h, pose, z=None):
        if self.use_user and self.cond == "film":
            gb = self.film(z)
            h = h * (1.0 + gb[:, :self.d_feat]) + gb[:, self.d_feat:]
        parts = [h]
        if self.use_pose:
            parts.append(pose)
        if self.use_user and self.cond == "concat":
            parts.append(z)
        x = self.pred_norm(torch.cat(parts, dim=1))
        pred = self.head(x)
        if self.use_user and self.cond == "bias":
            pred = pred + self.bias(z)
        return pred


def prepare(d, pose_map, support_k, cross_context=False):
    """返回 subjects -> dict(support indices, target indices, pose, scale)."""
    subs = defaultdict(list)
    for i, s in enumerate(d["subject"]):
        subs[s].append(i)
    out = {}
    for s, idx in subs.items():
        idx = np.array(idx, dtype=np.int64)
        idx = idx[np.argsort(d["frame_index"][idx])]
        if len(idx) <= support_k + 5:
            continue
        # 只保留有 pose 的帧
        ok = [i for i in idx if d["path"][i] in pose_map]
        if len(ok) <= support_k + 5:
            continue
        ok = np.array(ok, dtype=np.int64)
        if cross_context:
            ctxs = sorted({pose_map[d["path"][i]][3] for i in ok})
            if len(ctxs) < 2:
                continue
            # 用被试决定 support 上下文，覆盖所有上下文
            pick = ctxs[sum(map(ord, str(s))) % len(ctxs)]
            ok_sup = np.array([i for i in ok if pose_map[d["path"][i]][3] == pick], dtype=np.int64)
            tgt = np.array([i for i in ok if pose_map[d["path"][i]][3] != pick], dtype=np.int64)
            if len(ok_sup) < support_k or len(tgt) <= 5:
                continue
            pos = np.unique(np.linspace(0, len(ok_sup) - 1, support_k).round().astype(int))
            sup = ok_sup[pos]
        else:
            # support 均匀覆盖整个会话，避免只取到同一朝向/活动
            pos = np.unique(np.linspace(0, len(ok) - 1, support_k).round().astype(int))
            sup = ok[pos]
            tgt = np.setdiff1d(ok, sup)
        out[s] = dict(support=sup, target=tgt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--val-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--pose-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--variant", choices=["none", "pose", "user", "both"], default="both")
    ap.add_argument("--pred-feat", choices=["fused", "all"], default="fused",
                    help="预测头用哪段特征：fused=前384维（与原始头一致）/ all=1536")
    ap.add_argument("--enc-feat", choices=["fused", "all"], default="all")
    ap.add_argument("--enc-slice", choices=["all", "fused", "face", "eye"], default="all")
    ap.add_argument("--aux-user-weight", type=float, default=0.0,
                    help="用户分类辅助损失权重（强制 z 编码身份）")
    ap.add_argument("--cond", choices=["concat", "film", "bias"], default="concat")
    ap.add_argument("--cross-context", action="store_true",
                    help="support 取一个上下文（朝向/活动），目标取其他上下文")
    ap.add_argument("--support-k", type=int, default=8)
    ap.add_argument("--z-dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    D = load_npz(a.train_npz)
    V = load_npz(a.val_npz)
    T = load_npz(a.test_npz)
    def slice_feats(d, which):
        if which == "fused":
            return d["feats"][:, :384]
        return d["feats"]
    # 编码器用 enc-feat；预测头用 pred-feat
    D_full, V_full, T_full = D["feats"], V["feats"], T["feats"]
    slices = {"all": slice(None), "fused": slice(0, 384), "face": slice(384, 768),
              "eye": slice(1152, 1536)}
    for d, full in ((D, D_full), (V, V_full), (T, T_full)):
        if a.enc_slice != "all":
            d["enc"] = full[:, slices[a.enc_slice]]
        else:
            d["enc"] = full if a.enc_feat == "all" else full[:, :384]
        d["predf"] = full if a.pred_feat == "all" else full[:, :384]
    print(f"[data] train={len(D['feats'])} val={len(V['feats'])} test={len(T['feats'])} "
          f"d_feat={D['feats'].shape[1]}")
    needed = np.concatenate([D["path"], V["path"], T["path"]])
    pose_map = build_pose_map(a.index_csv, needed, a.pose_npz)
    print(f"[pose] mapped {len(pose_map)} / {len(needed)} paths")
    Dp = prepare(D, pose_map, a.support_k, a.cross_context)
    Vp = prepare(V, pose_map, a.support_k, a.cross_context)
    Tp = prepare(T, pose_map, a.support_k, a.cross_context)
    print(f"[data] subjects train={len(Dp)} val={len(Vp)} test={len(Tp)} "
          f"support_k={a.support_k}")
    for name, sp in (("train", Dp), ("val", Vp), ("test", Tp)):
        ns = [len(x["support"]) for x in sp.values()]; nt = [len(x["target"]) for x in sp.values()]
        if ns:
            print(f"[data] {name}: support/用户 {min(ns)}~{max(ns)}  target/用户 {min(nt)}~{max(nt)}")

    use_pose = a.variant in ("pose", "both")
    use_user = a.variant in ("user", "both")
    d_pred = D["predf"].shape[1]
    d_enc = D["enc"].shape[1]
    train_subs = sorted(Dp)
    sub2id = {s: i for i, s in enumerate(train_subs)}
    n_users = len(train_subs) if a.aux_user_weight > 0 else 0
    model = UserHead(d_pred, 8, a.z_dim, a.hidden, use_pose, use_user, d_enc,
                     a.cond, n_users).to(a.device)
    print(f"[model] variant={a.variant} use_pose={use_pose} use_user={use_user} "
          f"cond={a.cond} d_pred={d_pred} d_enc={d_enc} "
          f"params={sum(p.numel() for p in model.parameters())}")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
    dev = torch.device(a.device)

    def batch_indices(subjects, split, rng=None, max_n=None):
        subs = list(subjects)
        if rng is not None:
            rng.shuffle(subs)
        idxs = []
        for s in subs:
            t = split[s]["target"]
            if max_n:
                t = t[:max_n]
            idxs.append(t)
        return np.concatenate(idxs) if idxs else np.array([], dtype=np.int64)

    def z_for(subjects, split, d, grad=False):
        sf = np.stack([np.stack([d["enc"][i] for i in split[s]["support"]]) for s in subjects])
        sp = np.stack([np.stack([pose_map[d["path"][i]][0] for i in split[s]["support"]])
                       for s in subjects])
        sf = torch.from_numpy(sf).to(dev)
        sp = torch.from_numpy(sp).to(dev)
        if grad:
            return model.encode(sf, sp)
        with torch.no_grad():
            return model.encode(sf, sp)

    def evaluate(split, d, name):
        model.eval()
        subs = list(split)
        zmap = {}
        if use_user:
            for b in range(0, len(subs), 64):
                chunk = subs[b:b+64]
                z = z_for(chunk, split, d)
                for s, zz in zip(chunk, z):
                    zmap[s] = zz
        errs = []; ems = []; base_ems = []
        with torch.no_grad():
            for s in subs:
                t = split[s]["target"]
                h = torch.from_numpy(d["predf"][t]).to(dev)
                p = torch.from_numpy(np.stack([pose_map[d["path"][i]][0] for i in t])).to(dev)
                z = zmap[s].expand(len(t), -1) if use_user else None
                pred = model(h, p, z)
                lab = torch.from_numpy(d["label"][t]).to(dev)
                err = (pred - lab) ** 2
                errs.append(float(err.mean()))
                sx = np.array([pose_map[d["path"][i]][1] for i in t])
                sy = np.array([pose_map[d["path"][i]][2] for i in t])
                em = np.hypot((pred[:,0].cpu().numpy()-lab[:,0].cpu().numpy())*sx,
                              (pred[:,1].cpu().numpy()-lab[:,1].cpu().numpy())*sy)
                ems.append(float(em.mean()))
                if d.get("pred") is not None:
                    bp = d["pred"][t]
                    be = np.hypot((bp[:,0]-lab[:,0].cpu().numpy())*sx,
                                  (bp[:,1]-lab[:,1].cpu().numpy())*sy)
                    base_ems.append(float(be.mean()))
        msg = f"[{name}] mse={np.mean(errs):.6f} EM_cm={np.mean(ems):.4f}"
        if base_ems:
            msg += f" | 原始头 EM_cm={np.mean(base_ems):.4f}"
        print(msg)
        return float(np.mean(errs)), float(np.mean(ems))

    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        subs = list(Dp)
        rng = np.random.RandomState(ep)
        total = 0.0; nb = 0
        for b in range(0, len(subs), a.batch_size):
            chunk = subs[b:b+a.batch_size]
            z_all = z_for(chunk, Dp, D, grad=True) if use_user else None
            hs = []; ps = []; ls = []; zs = []
            for j, s in enumerate(chunk):
                t = Dp[s]["target"]
                k = min(len(t), 32)
                sel = rng.choice(len(t), k, replace=False)
                hs.append(D["predf"][t[sel]])
                ps.append(np.stack([pose_map[D["path"][i]][0] for i in t[sel]]))
                ls.append(D["label"][t[sel]])
                if use_user:
                    zs.append(z_all[j].unsqueeze(0).expand(k, -1))
            h = torch.from_numpy(np.concatenate(hs)).to(dev)
            p = torch.from_numpy(np.concatenate(ps)).to(dev)
            l = torch.from_numpy(np.concatenate(ls)).to(dev)
            z = torch.cat(zs, dim=0) if use_user else None
            pred = model(h, p, z)
            loss = ((pred - l) ** 2).mean()
            if n_users and use_user:
                ids = torch.tensor([sub2id[s] for s in chunk], device=dev)
                loss = loss + a.aux_user_weight * torch.nn.functional.cross_entropy(
                    model.cls(z_all), ids)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss); nb += 1
        if (ep + 1) % 2 == 0 or ep == 0:
            zn = float(z.detach().norm(dim=1).mean()) if (use_user and z is not None) else 0.0
            print(f"[train] ep{ep+1}/{a.epochs} loss={total/max(nb,1):.6f} "
                  f"z_norm={zn:.4f} t={time.time()-t0:.1f}s")
            evaluate(Vp, V, "val")
    evaluate(Tp, T, "test")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(a)}, a.out)
    print(f"[save] {a.out}")


if __name__ == "__main__":
    main()
