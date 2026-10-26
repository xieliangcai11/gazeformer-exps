"""导出「逐帧预测 + 骨干特征」，供特征空间标定 vs 输出空间误差场做同模型同划分对照。

为什么需要：两条标定线此前在不同模型上各测各的（✗ 不可比）：
  · 输出空间误差场：eye224_mgpu，test 1.472 → 1.093（密标签）/ head 一次性弱
  · 特征空间 RidgeCV：g5_nodepth_448_aug，1.676 → 1.407（前 10% 帧）
⇒ 本脚本在旗舰模型上同时导出预测与特征，后续 A/B/C/D 对照全在同一数据上进行。

特征 = [融合CLS(经 norm, 384) ⊕ 骨干face CLS(384) ⊕ face patch 均值(384) ⊕ 双眼中性(384)]
  · 融合 CLS＝head 的输入，是模型"最终内部表示"
  · 骨干 CLS/patch 保留逐帧细节（calibrator.py 记录：融合后 CLS 帧间差异几乎恒定）
  · 眼支流两个 CLS 取均值（与整脸共用同一骨干，故 hook model.rgb 会触发 3 次）
实现：hook model.rgb（3 次/前向：face, eye_l, eye_r）与 model.norm（被 head 消费的 [B,384]），
      直接复用 scripts.train.predict_frames 跑一遍，避免另写一套评估口径。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def _bootstrap() -> Path:
    here = Path(__file__).resolve()
    for d in (here.parent, *here.parents):
        if (d / "src" / "gazelab" / "__init__.py").is_file():
            for p in (str(d), str(d / "src")):
                if p not in sys.path:
                    sys.path.insert(0, p)
            return d
    raise RuntimeError(f"找不到项目根：从 {here} 向上未发现 src/gazelab/__init__.py")


_PROJECT_ROOT = _bootstrap()

from configs.paths import experiment_dirs  # noqa: E402
from gazelab.datasets.table_index import TableIndex  # noqa: E402
from gazelab.models.dinov2_gaze import RGBDGazeDINOv2, load_state_dict_any_res  # noqa: E402
from scripts.train import DATASET_SPECS, build_dataloader, predict_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="导出逐帧预测 + 骨干特征（npz）")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="rgbdgaze")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--orient", choices=["raw", "rot90"], default="raw")
    ap.add_argument("--geom", dest="use_geom", action="store_true", default=None)
    ap.add_argument("--no-geom", dest="use_geom", action="store_false")
    ap.add_argument("--use-eye", dest="use_eye", action="store_true", default=True)
    ap.add_argument("--eye-scale", type=float, default=0.0,
                    help="眼框收紧倍数：必须与训练时一致（此前无法传入 ⇒ 用 eye_scale>0 "
                         "训的权重会被按 0 评估）")
    ap.add_argument("--eye-shuffle", dest="eye_shuffle", action="store_true",
                    help="对照实验：打乱眼图与当前帧的对应（评估 eyeshuf* 权重时必须开启）")
    ap.add_argument("--unfreeze", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--half", action="store_true", help="特征存 float16（省一半空间）")
    ap.add_argument("--per-session", type=int, default=0,
                    help="每个被试最多取多少帧（0=全部；用于快速导出）")
    args = ap.parse_args()

    import torch

    # eye_shuffle 是 scripts/train.py 的模块级全局（predict_frames 读它）
    import scripts.train as _T
    _T.EYE_SHUFFLE = bool(args.eye_shuffle)
    _sc = Path(args.ckpt).parent / "train_args.json"
    _cfg = {}
    if _sc.is_file():
        try:
            import json
            _cfg = json.loads(_sc.read_text(encoding="utf-8"))
            print(f"[ckpt] 训练配置侧车 {_sc}")
            print("       " + json.dumps(_cfg, ensure_ascii=False)[:500])
        except Exception as e:
            print(f"[ckpt] 侧车读取失败（忽略）：{e}")
    if args.use_geom is None:
        args.use_geom = bool(_cfg.get("use_geom", True))
    args.geom_orient = bool(_cfg.get("geom_orient", False)) and args.use_geom
    print(f"[cfg] use_geom={args.use_geom} geom_orient={args.geom_orient}")

    spec = DATASET_SPECS[args.dataset]
    index_dir = Path(args.index_dir) if args.index_dir else Path(spec["default_index"])
    out_npz = Path(args.out) if args.out else (
        experiment_dirs(args.dataset, "analysis")["root"] /
        f"feats_{Path(args.ckpt).parent.name}_{args.split}.npz")
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    dl = build_dataloader(args.dataset, args.split, index_dir, args.batch_size, False,
                          args.num_workers, augment=False, resolution=args.resolution,
                          eye_crops=args.use_eye, orient=args.orient,
                          eye_scale=args.eye_scale, geom_orient=args.geom_orient)
    paths = [ln.strip() for ln in (index_dir / f"{args.split}.txt").read_text(
        encoding="utf-8").splitlines() if ln.strip()]

    model = RGBDGazeDINOv2(unfreeze_last=args.unfreeze, use_geom=args.use_geom,
                           use_geom_orient=args.geom_orient,
                           use_eye=args.use_eye, img_size=args.resolution).to(args.device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    miss, unexp, resampled = load_state_dict_any_res(model, state)
    print(f"[load] {args.ckpt}  缺失键={len(miss)} 多余键={len(unexp)} resampled={resampled}")
    if miss or unexp:
        print(f"  ⚠ 缺失={list(miss)[:4]} 多余={list(unexp)[:4]}（确认配置与训练一致）")
    model.eval()

    # ---- hooks（只保留降维后的特征，并立即搬到 CPU，避免 GPU 累积爆显存） ----
    bag = {"face": [], "eye_l": [], "eye_r": [], "fused": []}
    cnt = {"k": 0}

    def h_rgb(_m, _i, o):
        o = o.detach().float().cpu()
        r = cnt["k"] % 3
        cnt["k"] += 1
        if r == 0:                       # face：CLS + patch 均值（768）
            bag["face"].append(torch.cat([o[:, 0], o[:, 1:].mean(1)], 1))
        elif r == 1:                     # eye_l：CLS（384）
            bag["eye_l"].append(o[:, 0])
        else:                            # eye_r：CLS（384）
            bag["eye_r"].append(o[:, 0])

    def h_norm(_m, _i, o):
        bag["fused"].append(o.detach().float().cpu())

    model.rgb.register_forward_hook(h_rgb)
    model.norm.register_forward_hook(h_norm)

    # 以 loader 实际使用的全局行号为准重建 paths（build_dataloader 会剔除眼框不可用帧）
    base_ds, gidces = dl.dataset.base, list(dl.dataset.idces)
    if args.per_session:
        from torch.utils.data import Subset as _TSubset, DataLoader as _TDL
        from collections import defaultdict as _dd
        per = _dd(list); sel = []
        for pos, gi in enumerate(gidces):
            subj = base_ds.rows[gi].get("subject", "?")
            if len(per[subj]) < args.per_session:
                per[subj].append(pos); sel.append(pos)
        final_gi = [gidces[k] for k in sel]
        dl = _TDL(_TSubset(dl.dataset, sel), batch_size=args.batch_size,
                  shuffle=False, num_workers=args.num_workers)
        print(f"[subset] per-session={args.per_session} ⇒ {len(final_gi)} 帧 / {len(per)} 被试")
    else:
        final_gi = gidces
    paths = [base_ds.rows[gi][spec["path_col"]] for gi in final_gi]
    print(f"[eval] 划分={args.split} 帧数={len(paths)} res={args.resolution} "
          f"eye={args.use_eye} geom={args.use_geom}")
    fr = predict_frames(model, dl, args.device)
    n = len(fr["pred"])
    if n != len(paths):
        raise SystemExit(f"预测数 {n} != 清单帧数 {len(paths)}")

    nb = len(bag["fused"])
    if len(bag["face"]) != nb or len(bag["eye_l"]) != nb or len(bag["eye_r"]) != nb:
        raise SystemExit(f"hook 次数不匹配：fused={nb} face={len(bag['face'])} "
                         f"eye_l={len(bag['eye_l'])} eye_r={len(bag['eye_r'])}")
    fused = torch.cat(bag["fused"], 0).numpy()
    face_all = torch.cat(bag["face"], 0)
    face_cls = face_all[:, :384].numpy()
    face_patch = face_all[:, 384:].numpy()
    eye_mean = ((torch.cat(bag["eye_l"], 0) + torch.cat(bag["eye_r"], 0)) / 2).numpy()
    feats = np.concatenate([fused, face_cls, face_patch, eye_mean], 1).astype(np.float32)
    print(f"[feats] 形状={feats.shape}  "
          f"各段范数均值: 融合={np.linalg.norm(fused, axis=1).mean():.1f} "
          f"骨干CLS={np.linalg.norm(face_cls, axis=1).mean():.1f} "
          f"patch均值={np.linalg.norm(face_patch, axis=1).mean():.1f} "
          f"眼={np.linalg.norm(eye_mean, axis=1).mean():.1f}")

    table = TableIndex(index_dir / "index.csv")
    p2i = table.lookup(spec["path_col"])
    rows = [table[p2i[p]] for p in paths]
    _feats = feats.astype(np.float16) if args.half else feats
    np.savez(out_npz, feats=_feats, pred=fr["pred"].astype(np.float32),
                        label=fr["label"].astype(np.float32),
                        cm_x=fr["cm_x"].astype(np.float32),
                        cm_y=fr["cm_y"].astype(np.float32),
                        ang_deg=fr["ang_deg"].astype(np.float32),
                        path=np.array(paths, dtype=object),
                        subject=np.array([r.get("subject", "?") for r in rows], dtype=object),
                        frame_index=np.array([int(Path(p).stem) if Path(p).stem.isdigit() else -1
                                              for p in paths], dtype=np.int64))
    em = np.hypot(fr["cm_x"], fr["cm_y"])
    print(f"[基线自检] EM均值={em.mean():.3f} 中位={np.median(em):.3f} "
          f">3cm={100 * (em > 3).mean():.2f}%  (应与既有报告一致)")
    csv_path = out_npz.with_suffix(".meta.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "subject", "activity", "device", "frame_index"])
        for p, r in zip(paths, rows):
            w.writerow([p, r.get("subject", "?"), r.get("activity", "?"),
                        r.get("device", "?"), Path(p).stem])
    print(f"\n  NPZ: {out_npz}\n  META: {csv_path}")


if __name__ == "__main__":
    main()
