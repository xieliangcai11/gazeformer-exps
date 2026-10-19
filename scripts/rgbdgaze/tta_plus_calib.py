"""TTA ⊕ 引导式标定 叠加实验：四个口径在同一批帧上对比。

问题：TTA 修的是"逐用户的输出分布（偏置/收缩）"，多项式标定修的是"残余的空间结构"
      ⇒ 两者**是否互补**？如果互补，就是目前最可能再涨的一步。

四个口径（评估集 = 每会话除标定帧外的全部帧；真值只用于打分 ✓）：
  A 不标定
  B 仅 TTA            ← 零真值（只用图像）✓
  C 仅 9 点标定        ← 9 个真值标定点（3x3 网格选点，模拟"引导用户看满屏"✓）
  D TTA + 9 点标定     ← 先无标签适配，再用 9 个点做多项式标定
标定用 `PolyCalibrator`（多项式 + LOO 选 α + 覆盖检查 ✓），TTA 用 `tta.adapt`（含安全层与护栏 ✓）。
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import json
import numpy as np


def _bootstrap() -> Path:
    here = Path(__file__).resolve()
    for d in (here.parent, *here.parents):
        if (d / "src" / "gazelab" / "__init__.py").is_file():
            for p in (str(d), str(d / "src")):
                if p not in sys.path:
                    sys.path.insert(0, p)
            return d
    raise RuntimeError("找不到项目根")


_ROOT = _bootstrap()

from gazelab.models.dinov2_gaze.calibrator import PolyCalibrator  # noqa: E402
from gazelab.models.dinov2_gaze.tta import TTAConfig, adapt, detach_adapter  # noqa: E402
from scripts.train import DATASET_SPECS, Subset, build_dataloader, predict_frames  # noqa: E402


def pick_grid(P, L, idx, n):
    """按屏幕 n×n 网格格心选最近帧（= 引导用户看满屏的模拟；用真值选点，属协议上界）。"""
    cents = [((i + 0.5) / n, (j + 0.5) / n) for i in range(n) for j in range(n)]
    sel, used = [], set()
    for cx, cy in cents:
        dd = np.hypot(L[idx, 0] - cx, L[idx, 1] - cy)
        for o in np.argsort(dd):
            k = int(o)
            if k not in used:
                used.add(k); sel.append(k); break
    return np.array(sorted(sel))


def main():
    ap = argparse.ArgumentParser(description="TTA ⊕ 9 点标定 叠加实验")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="rgbdgaze")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--orient", choices=["raw", "rot90"], default="raw")
    ap.add_argument("--geom", dest="use_geom", action="store_true", default=None,
                    help="骨干含屏幕几何 token；默认读 ckpt 旁 train_args.json（缺失时 True）")
    ap.add_argument("--no-geom", dest="use_geom", action="store_false",
                    help="骨干不含几何 token（须与 ckpt 一致）")
    ap.add_argument("--use-eye", dest="use_eye", action="store_true", default=None,
                    help="骨干含眼支流；默认读 ckpt 旁 train_args.json（缺失时 True）")
    ap.add_argument("--no-eye", dest="use_eye", action="store_false",
                    help="骨干不含眼支流（须与 ckpt 一致）")
    ap.add_argument("--eye-scale", type=float, default=0.0)
    ap.add_argument("--prior", default=None)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--grid", type=int, default=3, help="标定点数 = grid×grid（默认 3×3=9）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # 以 ckpt 侧车为准解析 geom/eye 开关（避免结构性权重被静默丢弃）
    _side = Path(args.ckpt).parent / "train_args.json"
    _cfg = {}
    if _side.is_file():
        try:
            _cfg = json.loads(_side.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            _cfg = {}
    if args.use_geom is None:
        args.use_geom = bool(_cfg.get("use_geom", True))
        print(f"[cfg] use_geom={args.use_geom}")
    if args.use_eye is None:
        args.use_eye = bool(_cfg.get("use_eye", True))
        print(f"[cfg] use_eye={args.use_eye}")

    import torch
    from gazelab.models.dinov2_gaze import RGBDGazeDINOv2, load_state_dict_any_res
    import json
    from scripts.train import _shuffle_eyes

    spec = DATASET_SPECS[args.dataset]
    index_dir = Path(args.index_dir) if args.index_dir else Path(spec["default_index"])
    dl = build_dataloader(args.dataset, args.split, index_dir, 64, False, 0,
                          augment=False, resolution=args.resolution,
                          eye_crops=args.use_eye, orient=args.orient,
                          eye_scale=args.eye_scale)
    ds = dl.dataset
    paths = [ln.strip() for ln in (index_dir / f"{args.split}.txt").read_text(
        encoding="utf-8").splitlines() if ln.strip()]
    sub2idx = defaultdict(list)
    for k, p in enumerate(paths):
        sub2idx[str(p).rsplit("/", 1)[0]].append(k)

    prior_path = Path(args.prior) if args.prior else (Path(_ROOT) / "out/rgbdgaze/analysis/gaze_prior.json")
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.is_file() \
        else {"mean": [0.5, 0.5], "std": [0.3, 0.3]}
    if not prior_path.is_file():
        print(f"[warn] 缺先验 {prior_path} ⇒ 用占位先验")

    model = RGBDGazeDINOv2(unfreeze_last=0, use_geom=args.use_geom,
                           use_eye=args.use_eye, img_size=args.resolution).to(args.device)
    miss, unexp, _ = load_state_dict_any_res(model, torch.load(args.ckpt, map_location="cpu",
                                                               weights_only=False))
    print(f"[load] {args.ckpt}  缺失键={len(miss)} 多余键={len(unexp)}")
    model.eval()

    def em_of(pred_norm, lab_norm, cm):
        return np.hypot((pred_norm[:, 0] - lab_norm[:, 0]) * cm[:, 0],
                        (pred_norm[:, 1] - lab_norm[:, 1]) * cm[:, 1])

    rows = []
    for si, (s, idx) in enumerate(sorted(sub2idx.items())):
        sub = Subset(ds, idx)
        dl_s = torch.utils.data.DataLoader(sub, batch_size=64, shuffle=False, num_workers=0)
        fr = predict_frames(model, dl_s, args.device)
        Y = fr["label"]
        # cm 换算：用 Subset 自带的逐样本属性（已按子集对齐 ✓；此前误用 ds.cm_px[i] 是"划分内位置" ✗）
        scm = np.stack([sub.cm_px[:, 0].astype(float) * sub.screen_size[:, 0].astype(float),
                        sub.cm_px[:, 1].astype(float) * sub.screen_size[:, 1].astype(float)], 1)
        e_raw = em_of(fr["pred"], Y, scm)
        cal = pick_grid(fr["pred"], Y, np.arange(len(idx)), args.grid)
        ev = np.setdiff1d(np.arange(len(idx)), cal)

        # C 仅标定（用原始预测拟合）
        pc = PolyCalibrator().fit(fr["pred"][cal], Y[cal], screen_cm=scm[cal].mean(0))
        e_cal = em_of(pc.apply(fr["pred"]), Y, scm)

        # B/D 先做无标签 TTA
        frames = []
        for k in range(len(idx)):
            inp, _ = sub[k]
            f = {"face": inp.rgb}
            if args.use_eye:
                f["eye_l"], f["eye_r"] = inp.eye_l, inp.eye_r
            if args.use_geom:
                f["geom"] = inp.geom
            frames.append(f)
        adapt(model, frames, prior, TTAConfig(steps=args.steps, lr=args.lr, rank=args.rank,
                                             batch_size=args.batch_size))
        fr_t = predict_frames(model, dl_s, args.device)
        e_tta = em_of(fr_t["pred"], Y, scm)
        pcD = PolyCalibrator().fit(fr_t["pred"][cal], Y[cal], screen_cm=scm[cal].mean(0))
        e_both = em_of(pcD.apply(fr_t["pred"]), Y, scm)
        detach_adapter(model)

        m = lambda e: float(e[ev].mean())                      # noqa: E731
        rows.append((s, len(idx), m(e_raw), m(e_tta), m(e_cal), m(e_both)))
        print(f"  [{si+1}/{len(sub2idx)}] {s[-38:]}  n={len(idx):>5}  "
              f"A={m(e_raw):.3f}  B(TTA)={m(e_tta):.3f}  C(9点)={m(e_cal):.3f}  "
              f"D(TTA+9点)={m(e_both):.3f}", flush=True)

    a = np.array([[r[2], r[3], r[4], r[5]] for r in rows])
    ws = np.array([r[1] for r in rows], dtype=float)
    w = (a * ws[:, None]).sum(0) / ws.sum()
    print(f"\n  帧加权（{int(ws.sum())} 帧，{len(rows)} 会话）")
    for nm, v in zip(("A 不标定", "B 仅TTA(0真值)", "C 仅9点标定(9真值)", "D TTA+9点"), w):
        print(f"    {nm:<20}{v:.3f}   相对A {100*(v/w[0]-1):+.1f}%")
    if args.out:
        p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as fh:
            wtr = csv.writer(fh); wtr.writerow(["session", "n", "A", "B_tta", "C_cal9", "D_tta_cal9"])
            wtr.writerows(rows)
        print(f"  明细: {p}")


if __name__ == "__main__":
    main()
