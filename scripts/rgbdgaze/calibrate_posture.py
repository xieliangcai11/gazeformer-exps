"""Per-posture calibration comparison script."""
import sys
# ---- 路径引导：把项目根与 src/ 加入 sys.path ----
# 已改为「向上搜索项目根」而不是数 __file__ 的目录层级：后者在文件被移动到
# 不同深度后会静默失效（曾因 scripts/rgbdgaze/ablation.py -> scripts/train.py
# 的移动导致 `No module named 'gazelab'`）。
def _bootstrap() -> Path:
    """从本文件向上找到含 src/gazelab/__init__.py 的目录，并加入 sys.path。"""
    here = Path(__file__).resolve()
    for d in (here.parent, *here.parents):
        if (d / "src" / "gazelab" / "__init__.py").is_file():
            for p in (str(d), str(d / "src")):
                if p not in sys.path:
                    sys.path.insert(0, p)
            return d
    raise RuntimeError(
        f"找不到项目根：从 {here} 向上未发现 src/gazelab/__init__.py")


_PROJECT_ROOT = _bootstrap()

import numpy as np, torch, csv, math, json
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import RidgeCV
from datetime import datetime

from configs.rgbdgaze_config import RGBDGaze_INDEX_DIR
from gazelab.datasets.rgbdgaze import RGBDGazeDataset, rgb_preprocess
from gazelab.models.dinov2_gaze.rgbdgaze_dinov2 import RGBDGazeDINOv2
from gazelab.models.dinov2_gaze.calibrator import extract_features

# use_geom 必须与 ckpt 一致：模型类默认 use_geom=True，而 g5_nodepth_448_aug 是
# --no-geom 训的、不含 proj_geom，不显式关闭会在 load_state_dict 处直接报错。
model = RGBDGazeDINOv2(unfreeze_last=12, img_size=448, use_geom=False)
model.load_state_dict(torch.load(
    Path("out/rgbdgaze/ablation/checkpoints/g5_nodepth_448_aug/best.pt"),
    map_location="cpu", weights_only=False))
model.to('cuda').eval()

rows = list(csv.DictReader(open(RGBDGaze_INDEX_DIR / 'index.csv', encoding='utf-8')))
path_to_idx = {r["rgb_path"]: i for i, r in enumerate(rows)}
test_ids = [path_to_idx[p.strip()] for p in
            open(RGBDGaze_INDEX_DIR / 'test.txt', encoding='utf-8') if p.strip()]
dataset = RGBDGazeDataset(RGBDGaze_INDEX_DIR / 'index.csv',
                          color_transform=rgb_preprocess, image_size=448)

ACTIVITIES = ('sitting', 'standing', 'walking', 'lying')
# 逐样本屏幕物理尺寸（cm）：以前写死 7.81x16.08（iPhone 12 Pro Max），
# 但测试集 34% 的帧来自 iPhone XR / 12 / 12 Pro ⇒ 这些被试的 cm 误差被系统性放大。
CM = [(float(r["screen_w"]) * float(r["cm_px_x"]),
       float(r["screen_h"]) * float(r["cm_px_y"])) for r in rows]
ALPHAS = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

groups = defaultdict(list)
for i in test_ids:
    r = rows[i]
    fn = int(r["rgb_path"].replace("\\", "/").split("/")[-1].split(".")[0])
    groups[r["subject"]].append((fn, i))
subject_frames = {s: sorted(v) for s, v in groups.items()}

# Extract features
print("Extracting features...")
all_feats, all_preds, all_gts, all_acts, all_imu = {}, {}, {}, {}, {}
_imu_missing_warned = False
with torch.no_grad():
    for start in range(0, len(test_ids), 32):
        batch = test_ids[start:start + 32]
        faces, geoms = [], []
        for idx in batch:
            r = rows[idx]
            inp, _ = dataset[idx]
            gx = float(r["gaze_x"]) / float(r["screen_w"])
            gy = float(r["gaze_y"]) / float(r["screen_h"])
            faces.append(inp.rgb)
            geoms.append(inp.geom)
            all_gts[idx] = (gx, gy)
            all_acts[idx] = r["activity"]
            try:
                all_imu[idx] = [float(r["imu_x"]), float(r["imu_y"]),
                                float(r["imu_z"])]
            except (KeyError, ValueError):
                # RGBDGaze 的索引没有 IMU 列（只有 GazeCapture 有）⇒ 策略 C 不可用
                all_imu[idx] = [0.0, 0.0, 0.0]
                if not _imu_missing_warned:
                    print("  [warn] 索引缺少 imu_x/y/z ⇒ 策略 C（IMU）退化为零向量")
                    _imu_missing_warned = True
        f = torch.stack(faces).to('cuda').float()
        g = torch.stack(geoms).to('cuda').float()
        feat = extract_features(model, f)
        pred = model(f, geom=g)
        for i, idx in enumerate(batch):
            all_feats[idx] = feat[i].cpu().numpy()
            all_preds[idx] = pred[i].cpu().numpy()
print(f"Done {len(all_feats)}")

def run_calibration(ratio):
    results = {}
    # Strategy A: global
    total_A, total_base, total_n = [], [], 0
    for sub in sorted(subject_frames):
        frames = subject_frames[sub]
        n_cal = max(4, int(len(frames) * ratio))
        calib = [idx for _, idx in frames[:n_cal]]
        test = [idx for _, idx in frames[n_cal:]]
        cf = np.array([all_feats[i] for i in calib])
        co = np.array([np.array(all_gts[i]) - np.array(all_preds[i]) for i in calib])
        tf = np.array([all_feats[i] for i in test])
        tg = np.array([all_gts[i] for i in test])
        tp = np.array([all_preds[i] for i in test])
        r = RidgeCV(alphas=ALPHAS)
        r.fit(cf, co)
        corr = tp + r.predict(tf)
        for c, g, i in zip(corr, tg, test):
            total_A.append(np.hypot((c[0]-g[0])*CM[i][0], (c[1]-g[1])*CM[i][1]))
        for p, g, i in zip(tp, tg, test):
            total_base.append(np.hypot((p[0]-g[0])*CM[i][0], (p[1]-g[1])*CM[i][1]))
        total_n += len(test)
    results['A_global'] = np.mean(total_A)
    results['baseline'] = np.mean(total_base)

    # Strategy B: per-posture
    total_B = []
    for sub in sorted(subject_frames):
        frames = subject_frames[sub]
        act_frames = defaultdict(list)
        for fn, idx in frames:
            act_frames[all_acts[idx]].append((fn, idx))
        calibrators = {}
        calib_set = set()
        for act in ACTIVITIES:
            if act not in act_frames or len(act_frames[act]) < 5:
                continue
            af = sorted(act_frames[act])
            n_cal = max(2, int(len(af) * ratio))
            calib = [idx for _, idx in af[:n_cal]]
            calib_set.update(calib)
            cf = np.array([all_feats[i] for i in calib])
            co = np.array([np.array(all_gts[i]) - np.array(all_preds[i]) for i in calib])
            r = RidgeCV(alphas=ALPHAS)
            r.fit(cf, co)
            calibrators[act] = r
        for fn, idx in frames:
            if idx in calib_set:
                continue
            act = all_acts[idx]
            feat_i = all_feats[idx].reshape(1, -1)
            pred_i = all_preds[idx].reshape(1, -1)
            g_i = np.array([all_gts[idx]])
            if act in calibrators:
                corr = pred_i + calibrators[act].predict(feat_i)
            else:
                corr = pred_i
            total_B.append(np.hypot((corr[0,0]-g_i[0,0])*CM[idx][0],
                                    (corr[0,1]-g_i[0,1])*CM[idx][1]))
    results['B_per_posture'] = np.mean(total_B)

    # Strategy C: per-posture + IMU
    total_C = []
    for sub in sorted(subject_frames):
        frames = subject_frames[sub]
        act_frames = defaultdict(list)
        for fn, idx in frames:
            act_frames[all_acts[idx]].append((fn, idx))
        calibrators = {}
        calib_set = set()
        for act in ACTIVITIES:
            if act not in act_frames or len(act_frames[act]) < 5:
                continue
            af = sorted(act_frames[act])
            n_cal = max(2, int(len(af) * ratio))
            calib = [idx for _, idx in af[:n_cal]]
            calib_set.update(calib)
            cf = np.array([all_feats[i] for i in calib])
            imu_f = np.array([all_imu[i] for i in calib])
            co = np.array([np.array(all_gts[i]) - np.array(all_preds[i]) for i in calib])
            combined = np.concatenate([cf, imu_f], axis=1)
            r = RidgeCV(alphas=ALPHAS)
            r.fit(combined, co)
            calibrators[act] = r
        for fn, idx in frames:
            if idx in calib_set:
                continue
            act = all_acts[idx]
            if act not in calibrators:
                continue
            feat_i = np.concatenate([
                all_feats[idx].reshape(1, -1),
                np.array([all_imu[idx]]).reshape(1, -1)
            ], axis=1)
            pred_i = all_preds[idx].reshape(1, -1)
            g_i = np.array([all_gts[idx]])
            corr = pred_i + calibrators[act].predict(feat_i)
            total_C.append(np.hypot((corr[0,0]-g_i[0,0])*CM[idx][0],
                                    (corr[0,1]-g_i[0,1])*CM[idx][1]))
    results['C_per_posture_imu'] = np.mean(total_C)
    return results

for ratio in (0.10, 0.15):
    print(f"\n{'='*60}")
    print(f"Calibration ratio: {ratio*100:.0f}%")
    r = run_calibration(ratio)
    base = r['baseline']
    print(f"  Baseline (no calib): {base:.2f}cm")
    print(f"  A (global):          {r['A_global']:.2f}cm ({(1-r['A_global']/base)*100:+.1f}%)")
    print(f"  B (per-posture):     {r['B_per_posture']:.2f}cm ({(1-r['B_per_posture']/base)*100:+.1f}%)")
    print(f"  C (per-posture+IMU): {r['C_per_posture_imu']:.2f}cm ({(1-r['C_per_posture_imu']/base)*100:+.1f}%)")
    print(f"  Paper baseline:      1.89cm | MAC-Gaze: 1.41cm")
    print(f"  Best of ours: {'A' if r['A_global']<r['B_per_posture'] and r['A_global']<r['C_per_posture_imu'] else ('B' if r['B_per_posture']<r['C_per_posture_imu'] else 'C')}: {min(r['A_global'], r['B_per_posture'], r['C_per_posture_imu']):.2f}cm")
    out = Path("out/rgbdgaze/calibration") / f"posture_calib_r{int(ratio*100)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2))
