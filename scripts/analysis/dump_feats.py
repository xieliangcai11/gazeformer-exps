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
    ap.add_argument("--no-geom", dest="use_geom", action="store_false", default=False)
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
    args = ap.parse_args()

    import torch

    # eye_shuffle 是 scripts/train.py 的模块级全局（predict_frames 读它）
    import scripts.train as _T
    _T.EYE_SHUFFLE = bool(args.eye_shuffle)
    _sc = Path(args.ckpt).parent / "train_args.json"
    if _sc.is_file():
        try:
            import json
            print(f"[ckpt] 训练配置侧车 {_sc}")
            print("       " + json.dumps(json.loads(_sc.read_text(encoding="utf-8")),
                                         ensure_ascii=False)[:500])
        except Exception as e:
            print(f"[ckpt] 侧车读取失败（忽略）：{e}")

    spec = DATASET_SPECS[args.dataset]
    index_dir = Path(args.index_dir) if args.index_dir else Path(spec["default_index"])
    out_npz = Path(args.out) if args.out else (
        experiment_dirs(args.dataset, "analysis")["root"] /
        f"feats_{Path(args.ckpt).parent.name}_{args.split}.npz")
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    dl = build_dataloader(args.dataset, args.split, index_dir, args.batch_size, False,
                          args.num_workers, augment=False, resolution=args.resolution,
                          eye_crops=args.use_eye, orient=args.orient,
                          eye_scale=args.eye_scale)
    paths = [ln.strip() for ln in (index_dir / f"{args.split}.txt").read_text(
        encoding="utf-8").splitlines() if ln.strip()]

    model = RGBDGazeDINOv2(unfreeze_last=args.unfreeze, use_geom=args.use_geom,
                           use_eye=args.use_eye, img_size=args.resolution).to(args.device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    miss, unexp, resampled = load_state_dict_any_res(model, state)
    print(f"[load] {args.ckpt}  缺失键={len(miss)} 多余键={len(unexp)} resampled={resampled}")
    if miss or unexp:
        print(f"  ⚠ 缺失={list(miss)[:4]} 多余={list(unexp)[:4]}（确认配置与训练一致）")
    model.eval()

    # ---- hooks ----
    bag = {"rgb": [], "fused": []}

    def h_rgb(_m, _i, o):
        bag["rgb"].append(o.detach())

    def h_norm(_m, _i, o):
        bag["fused"].append(o.detach())

    model.rgb.register_forward_hook(h_rgb)
    model.norm.register_forward_hook(h_norm)

    print(f"[eval] 划分={args.split} 帧数={len(paths)} res={args.resolution} "
          f"eye={args.use_eye} geom={args.use_geom}")
    fr = predict_frames(model, dl, args.device)
    n = len(fr["pred"])
    if n != len(paths):
        raise SystemExit(f"预测数 {n} != 清单帧数 {len(paths)}")

    nb = len(bag["fused"])
    if len(bag["rgb"]) != 3 * nb:
        raise SystemExit(f"rgb hook 触发 {len(bag['rgb'])} 次，期望 {3 * nb} 次"
                         f"（face/eye_l/eye_r 各一次）⇒ 眼支流是否启用？")
    fused = torch.cat(bag["fused"], 0).float().cpu().numpy()
    clss = [torch.cat(bag["rgb"][k::3], 0) for k in range(3)]   # face / eye_l / eye_r
    face_tok = clss[0].float()
    face_cls = face_tok[:, 0].cpu().numpy()
    face_patch = face_tok[:, 1:].mean(1).cpu().numpy()
    eye_mean = ((clss[1][:, 0] + clss[2][:, 0]) / 2).float().cpu().numpy()
    feats = np.concatenate([fused, face_cls, face_patch, eye_mean], 1).astype(np.float32)
    print(f"[feats] 形状={feats.shape}  "
          f"各段范数均值: 融合={np.linalg.norm(fused, axis=1).mean():.1f} "
          f"骨干CLS={np.linalg.norm(face_cls, axis=1).mean():.1f} "
          f"patch均值={np.linalg.norm(face_patch, axis=1).mean():.1f} "
          f"眼={np.linalg.norm(eye_mean, axis=1).mean():.1f}")

    table = TableIndex(index_dir / "index.csv")
    p2i = table.lookup(spec["path_col"])
    rows = [table[p2i[p]] for p in paths]
    np.savez_compressed(out_npz, feats=feats, pred=fr["pred"].astype(np.float32),
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
