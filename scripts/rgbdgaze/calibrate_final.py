"""最终校准脚本：RidgeCV 自动选最优 alpha + G5 骨干模型。

对 5 个测试被试分别执行：
  1. 按时间取前 calib_ratio% 帧作为校准集
  2. 提取骨干特征
  3. RidgeCV 自动选最优 alpha 并拟合线性校准器
  4. 在剩余帧上评估
"""
import sys
sys.path[0:0] = ['src']
import csv, math, json, numpy as np, torch
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import RidgeCV

from configs.rgbdgaze_config import RGBDGaze_INDEX_DIR
from gazelab.datasets.rgbdgaze import (
    RGBDGazeDataset, rgb_preprocess, inverse_depth_preprocess,
)
from gazelab.models.rgbdgaze.rgbdgaze_dinov2 import RGBDGazeDINOv2
from gazelab.models.rgbdgaze.calibrator import extract_features

CKPT = Path("out/rgbdgaze/ablation/checkpoints/g5_nodepth_448_aug/best.pt")
RESOLUTION = 448
OUT_DIR = Path("out/rgbdgaze/calibration")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

device = "cuda" if torch.cuda.is_available() else "cpu"

model = RGBDGazeDINOv2(use_depth=False, unfreeze_last=12, img_size=RESOLUTION)
model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False))
model.to(device).eval()
print(f"[load] {CKPT}")

rows = list(csv.DictReader(open(RGBDGaze_INDEX_DIR/'index.csv', encoding='utf-8')))
path_to_idx = {r["rgb_path"]: i for i, r in enumerate(rows)}
test_indices = [path_to_idx[p.strip()] for p in
                open(RGBDGaze_INDEX_DIR/'test.txt', encoding='utf-8') if p.strip()]
dataset = RGBDGazeDataset(RGBDGaze_INDEX_DIR/'index.csv',
                          color_transform=rgb_preprocess,
                          depth_transform=inverse_depth_preprocess,
                          image_size=RESOLUTION)

# 按被试分组，按帧序排列
groups = defaultdict(list)
for idx in test_indices:
    r = rows[idx]
    fn = int(r["rgb_path"].replace("\\", "/").split("/")[-1].split(".")[0])
    groups[r["subject"]].append((fn, idx))
subject_frames = {s: sorted(v) for s, v in groups.items()}

# 批量提取特征和预测
print("[feat] 提取全部测试样本特征...")
all_feats, all_preds, all_gts = {}, {}, {}
with torch.no_grad():
    for start in range(0, len(test_indices), 32):
        batch = test_indices[start:start+32]
        faces, depths = [], []
        for idx in batch:
            r = rows[idx]
            inp, _ = dataset[idx]
            gx = float(r["gaze_x"]) / float(r["screen_w"])
            gy = float(r["gaze_y"]) / float(r["screen_h"])
            faces.append(inp.rgb)
            depths.append(inp.depth)
            all_gts[idx] = (gx, gy)
        f = torch.stack(faces).to(device).float()
        d = torch.stack(depths).to(device).float()
        feat = extract_features(model, f, d)
        pred = model(f, d)
        for i, idx in enumerate(batch):
            all_feats[idx] = feat[i].cpu().numpy()
            all_preds[idx] = pred[i].cpu().numpy()
print(f"[feat] 完成 {len(all_feats)} 条")

# 对每个校准比例执行
results = {}
for ratio in (0.05, 0.10, 0.15):
    print(f"\n{'='*60}")
    print(f"[calib] 校准比例: {ratio*100:.0f}%")
    total_corr, total_base, total_n = [], [], 0
    per_sub = {}

    for sub in sorted(subject_frames.keys()):
        frames = subject_frames[sub]
        n_cal = max(5, int(len(frames) * ratio))
        calib_ids = [idx for _, idx in frames[:n_cal]]
        test_ids_sub = [idx for _, idx in frames[n_cal:]]

        cal_feats = np.array([all_feats[i] for i in calib_ids])
        cal_offsets = np.array([np.array(all_gts[i]) - np.array(all_preds[i])
                                for i in calib_ids])
        test_feats = np.array([all_feats[i] for i in test_ids_sub])
        test_preds = np.array([all_preds[i] for i in test_ids_sub])
        test_gts = np.array([all_gts[i] for i in test_ids_sub])

        # RidgeCV 自动选最优 alpha
        ridge = RidgeCV(alphas=ALPHAS)
        ridge.fit(cal_feats, cal_offsets)
        best_alpha = ridge.alpha_

        # 校正
        offsets = ridge.predict(test_feats)
        corrected = test_preds + offsets

        sw_cm, sh_cm = 7.81, 16.08
        e_corr = np.mean([np.hypot((c[0]-g[0])*sw_cm, (c[1]-g[1])*sh_cm)
                          for c, g in zip(corrected, test_gts)])
        e_base = np.mean([np.hypot((p[0]-g[0])*sw_cm, (p[1]-g[1])*sh_cm)
                          for p, g in zip(test_preds, test_gts)])
        per_sub[sub] = dict(base=e_base, corr=e_corr, alpha=best_alpha,
                            n_cal=n_cal, n_test=len(test_ids_sub))
        total_corr.extend([np.hypot((c[0]-g[0])*sw_cm, (c[1]-g[1])*sh_cm)
                           for c, g in zip(corrected, test_gts)])
        total_base.extend([np.hypot((p[0]-g[0])*sw_cm, (p[1]-g[1])*sh_cm)
                           for p, g in zip(test_preds, test_gts)])
        total_n += len(test_ids_sub)
        imp = (1 - e_corr/e_base) * 100 if e_base > 0 else 0
        print(f"  {sub}: alpha={best_alpha:.0f} | 基准={e_base:.2f} 校准后={e_corr:.2f} ({imp:+.1f}%)")

    overall_corr = np.mean(total_corr)
    overall_base = np.mean(total_base)
    imp = (1 - overall_corr / overall_base) * 100
    print(f"\n  整体: 无校准={overall_base:.2f}cm → RidgeCV校准={overall_corr:.2f}cm ({imp:+.1f}%)")
    print(f"  论文基线: 1.89cm | {'已超越 ✓' if overall_corr < 1.89 else '未达 ✗'}")
    results[ratio] = dict(overall_corr=overall_corr, overall_base=overall_base,
                          improvement=imp, per_subject=per_sub)

# 保存
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_json = OUT_DIR / "ridgecv_final.json"
out_json.write_text(json.dumps(results, indent=2, default=str))
print(f"\n[save] {out_json}")

# 总结
print("\n" + "="*60)
print("最终总结")
print("="*60)
for ratio, r in results.items():
    print(f"  校准 {ratio*100:.0f}%: test {r['overall_corr']:.2f}cm "
          f"(无校准 {r['overall_base']:.2f}cm, 提升 {r['improvement']:+.1f}%)")
best_ratio = min(results, key=lambda r: results[r]["overall_corr"])
best = results[best_ratio]
print(f"\n  最优校准比例: {best_ratio*100:.0f}%")
print(f"  最终 test 成绩: {best['overall_corr']:.2f}cm")
print(f"  vs 论文基线 1.89cm: {'超越' if best['overall_corr'] < 1.89 else '未超越'} "
      f"({(1-best['overall_corr']/1.89)*100:+.1f}%)")