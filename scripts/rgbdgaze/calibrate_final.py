"""最终校准脚本：RidgeCV 自动选最优 alpha + G5 骨干模型。

协议（2026-10 审查修正）：
  1. **校准比例只在 val 上选**（以前直接在 test 上遍历 5/10/15% 再取最好 ⇒ 测试集泄漏）
  2. **cm 换算用逐样本屏幕尺寸**（以前写死 7.81x16.08 = iPhone 12 Pro Max）
  3. 每个被试按帧序取前 ratio 帧作校准集，其余帧评估；超参 alpha 由 RidgeCV 内部 CV 选
"""
import sys
import csv
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import RidgeCV


def _bootstrap() -> Path:
    """从本文件向上找到含 src/gazelab/__init__.py 的目录，并加入 sys.path。"""
    here = Path(__file__).resolve()
    for d in (here.parent, *here.parents):
        if (d / "src" / "gazelab" / "__init__.py").is_file():
            for p in (str(d), str(d / "src")):
                if p not in sys.path:
                    sys.path.insert(0, p)
            return d
    raise RuntimeError(f"找不到项目根：从 {here} 向上未发现 src/gazelab/__init__.py")


_PROJECT_ROOT = _bootstrap()

from configs.rgbdgaze_config import RGBDGaze_INDEX_DIR  # noqa: E402
from gazelab.datasets.rgbdgaze import RGBDGazeDataset, rgb_preprocess  # noqa: E402
from gazelab.models.dinov2_gaze.rgbdgaze_dinov2 import RGBDGazeDINOv2  # noqa: E402
from gazelab.models.dinov2_gaze.calibrator import extract_features  # noqa: E402

CKPT = Path("out/rgbdgaze/ablation/checkpoints/g5_nodepth_448_aug/best.pt")
RESOLUTION = 448
OUT_DIR = Path("out/rgbdgaze/calibration")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
RATIOS = (0.05, 0.10, 0.15)

device = "cuda" if torch.cuda.is_available() else "cpu"

# use_geom 必须与 ckpt 一致：模型类默认 use_geom=True，而 g5_nodepth_448_aug 是
# --no-geom 训的、不含 proj_geom，不显式关闭会在 load_state_dict 处直接报错。
model = RGBDGazeDINOv2(unfreeze_last=12, img_size=RESOLUTION, use_geom=False)
model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False))
model.to(device).eval()
print(f"[load] {CKPT}")

rows = list(csv.DictReader(open(RGBDGaze_INDEX_DIR / "index.csv", encoding="utf-8")))
path_to_idx = {r["rgb_path"]: i for i, r in enumerate(rows)}


def read_split(name):
    f = RGBDGaze_INDEX_DIR / f"{name}.txt"
    if not f.is_file():
        raise SystemExit(f"缺少划分清单 {f}")
    return [path_to_idx[p.strip()] for p in
            open(f, encoding="utf-8") if p.strip()]


val_indices = read_split("val")
test_indices = read_split("test")
dataset = RGBDGazeDataset(RGBDGaze_INDEX_DIR / "index.csv",
                          color_transform=rgb_preprocess, image_size=RESOLUTION)

#: 逐样本屏幕物理尺寸（cm）—— 修掉写死 7.81×16.08 的口径错误
CM = {i: (float(r["screen_w"]) * float(r["cm_px_x"]),
          float(r["screen_h"]) * float(r["cm_px_y"])) for i, r in enumerate(rows)}


def _extract(indices):
    feats, preds, gts = {}, {}, {}
    with torch.no_grad():
        for start in range(0, len(indices), 32):
            batch = indices[start:start + 32]
            faces, geoms = [], []
            for idx in batch:
                r = rows[idx]
                inp, _ = dataset[idx]
                gts[idx] = (float(r["gaze_x"]) / float(r["screen_w"]),
                            float(r["gaze_y"]) / float(r["screen_h"]))
                faces.append(inp.rgb)
                geoms.append(inp.geom)
            f = torch.stack(faces).to(device).float()
            g = torch.stack(geoms).to(device).float()
            feat = extract_features(model, f)
            pred = model(f, geom=g)
            for i, idx in enumerate(batch):
                feats[idx] = feat[i].cpu().numpy()
                preds[idx] = pred[i].cpu().numpy()
    return feats, preds, gts


def _eval(indices, feats, preds, gts, ratio, verbose=False):
    """按被试做时序校准；返回 (校准后 cm, 基线 cm, 逐被试明细)。"""
    groups = defaultdict(list)
    for idx in indices:
        r = rows[idx]
        fn = int(r["rgb_path"].replace("\\", "/").split("/")[-1].split(".")[0])
        groups[r["subject"]].append((fn, idx))
    per_sub, tot_c, tot_b = {}, [], []
    for sub in sorted(groups):
        frames = sorted(groups[sub])
        n_cal = max(5, int(len(frames) * ratio))
        calib = [idx for _, idx in frames[:n_cal]]
        test = [idx for _, idx in frames[n_cal:]]
        cf = np.array([feats[i] for i in calib])
        co = np.array([np.array(gts[i]) - np.array(preds[i]) for i in calib])
        tf = np.array([feats[i] for i in test])
        tp = np.array([preds[i] for i in test])
        tg = np.array([gts[i] for i in test])
        ridge = RidgeCV(alphas=ALPHAS).fit(cf, co)
        corr = tp + ridge.predict(tf)
        e_c = [np.hypot((c[0] - g[0]) * CM[i][0], (c[1] - g[1]) * CM[i][1])
               for c, g, i in zip(corr, tg, test)]
        e_b = [np.hypot((p[0] - g[0]) * CM[i][0], (p[1] - g[1]) * CM[i][1])
               for p, g, i in zip(tp, tg, test)]
        tot_c.extend(e_c)
        tot_b.extend(e_b)
        per_sub[sub] = dict(base=float(np.mean(e_b)), corr=float(np.mean(e_c)),
                            alpha=float(ridge.alpha_), n_cal=n_cal, n_test=len(test))
        if verbose:
            print(f"  {sub}: alpha={ridge.alpha_:.0f} | 基准={np.mean(e_b):.2f} "
                  f"校准后={np.mean(e_c):.2f}")
    return float(np.mean(tot_c)), float(np.mean(tot_b)), per_sub


# ---- 比例选择只用 val ----
print("[feat] 提取 val 特征…")
val_feats, val_preds, val_gts = _extract(val_indices)
sel = {}
for ratio in RATIOS:
    c, b, _ = _eval(val_indices, val_feats, val_preds, val_gts, ratio)
    sel[ratio] = dict(val_corr=c, val_base=b, improvement=(1 - c / b) * 100)
    print(f"[select] ratio={ratio:.2f}: val {b:.3f} -> {c:.3f} "
          f"({(1 - c / b) * 100:+.1f}%)")
best_ratio = min(sel, key=lambda r: sel[r]["val_corr"])
print(f"[select] 最优比例（val 选出）={best_ratio:.2f}")

# ---- test 只报告固定协议结果 ----
print("[feat] 提取 test 特征…")
test_feats, test_preds, test_gts = _extract(test_indices)
c, b, per_sub = _eval(test_indices, test_feats, test_preds, test_gts,
                      best_ratio, verbose=True)
imp = (1 - c / b) * 100
print(f"\n[result] test @ ratio={best_ratio:.2f}: {b:.2f} -> {c:.2f} cm ({imp:+.1f}%)"
      f"  ← 比例由 val 选出，test 未参与选择")
print(f"[result] 论文基线 1.89cm: {'超越 ✓' if c < 1.89 else '未达 ✗'}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_json = OUT_DIR / "ridgecv_final.json"
out_json.write_text(json.dumps(
    {"selected_ratio": best_ratio, "selection_on_val": sel,
     "test": dict(overall_corr=c, overall_base=b, improvement=imp,
                  per_subject=per_sub)},
    indent=2, default=str), encoding="utf-8")
print(f"[save] {out_json}")
