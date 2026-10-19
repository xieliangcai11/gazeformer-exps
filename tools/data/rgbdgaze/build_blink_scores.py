"""生成「闭眼分数」供训练/评估的 --drop-blink 使用。

为什么需要：RGBDGaze 官方数据集的标签是「受试被要求跟随的移动红点位置」，
闭眼帧的真值语义无效（闭眼时不可能在看红点）；论文明确说发布前**已剔除闭眼帧**
（"We pruned the collected dataset by removing data points where the participants
blinked. For this we employed an eye-aspect-ratio…"），而我们的索引**没有**做这一步
⇒ 训练被注入标签噪声、评估数字被压低（实测剔除后 test EM 1.472 → ~1.374，>5cm 尾部 −69%）。

⚠️ 重要：本脚本给出的是**眼裂比(EAR)的代理量**，不是论文所用的原始 EAR
   （仓库里没有 landmark，只有眼框 + 原图）。分数定义（0=睁眼，1=大概率闭眼）：
       对每只眼框取灰度的**逐行均值profile**，量测"中部凹陷深度"：
           dip  = (profile[0] + profile[-1]) / 2 − min(profile)
           s    = 1 − dip / --scale          （默认 scale=40 灰度级，经验值）
       blink = min(s_L, s_R)                  （眨眼 = 双眼同时闭）
   新增的 `dip_L/dip_R/mean_L/mean_R` 列用于人工核对。

⚠️ 使用前**必须**在人工标注的小子集上核对阈值与误检率（不同肤色/光照/分辨率下
   dip 的尺度会漂）。默认阈值 0.5 是这条经验尺度的中点，不是经过验证的判据。

用法：
    # 先小样本冒烟，确认能跑通、看一眼分数分布
    python -m tools.data.rgbdgaze.build_blink_scores --limit 200 --out /tmp/blink_smoke.csv
    # 全量（约 14 万帧，CPU、单进程）
    python -m tools.data.rgbdgaze.build_blink_scores
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
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

from configs.rgbdgaze_config import RGBDGaze_INDEX_DIR  # noqa: E402


def _resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (_PROJECT_ROOT / p)


def _dip(gray_crop: np.ndarray) -> tuple[float, float]:
    """返回 (凹陷深度, 平均灰度)。凹陷越浅 ⇒ 越像闭眼。"""
    prof = gray_crop.mean(axis=1)                 # 逐行均值（上→下）
    edges = 0.5 * (prof[:max(1, len(prof) // 8)].mean()
                   + prof[-max(1, len(prof) // 8):].mean())
    return float(edges - prof.min()), float(gray_crop.mean())


def main():
    ap = argparse.ArgumentParser(description="生成闭眼分数（EAR 代理）")
    ap.add_argument("--index-csv", default=str(Path(RGBDGaze_INDEX_DIR) / "index.csv"))
    ap.add_argument("--eyes-npz", default=None,
                    help="默认 <index dir>/eye_boxes.npz（由 build_eye_boxes.py 生成）")
    ap.add_argument("--out", default=None,
                    help="默认 <index dir>/blink_scores.csv（--drop-blink 的默认读取位置）")
    ap.add_argument("--limit", type=int, default=0, help="只处理 N 帧（冒烟用；与 --start 合用可分片）")
    ap.add_argument("--start", type=int, default=0, help="从第几帧开始（分片并行用：每个分片写自己的 CSV，最后合并）")
    ap.add_argument("--scale", type=float, default=40.0,
                    help="把 dip 归一化到 0~1 的灰度尺度（经验值，需按数据核对）")
    args = ap.parse_args()

    import cv2

    index_csv = Path(args.index_csv)
    eyes_npz = Path(args.eyes_npz) if args.eyes_npz else (index_csv.parent / "eye_boxes.npz")
    out_csv = Path(args.out) if args.out else (index_csv.parent / "blink_scores.csv")
    if not eyes_npz.is_file():
        raise FileNotFoundError(f"缺少眼框表 {eyes_npz}（先跑 tools/data/rgbdgaze/build_eye_boxes.py）")

    paths, subjects = [], []
    with open(index_csv, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            paths.append(r["rgb_path"])
            subjects.append(r.get("subject", "?"))
    boxes = np.load(eyes_npz)["boxes"].astype(np.float32)
    if len(boxes) != len(paths):
        raise ValueError(f"眼框行数 {len(boxes)} 与索引 {len(paths)} 不一致")
    lo = max(0, int(args.start))
    hi = min(len(paths), lo + int(args.limit)) if args.limit else len(paths)
    print(f"[blink] 处理 [{lo}, {hi}) 共 {hi - lo}/{len(paths)} 帧（scale={args.scale}，输出 {out_csv}）")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    hist = defaultdict(int)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "subject", "blink", "dip_L", "dip_R", "mean_L", "mean_R"])
        for i in range(lo, hi):
            img = cv2.imread(str(_resolve(paths[i])), cv2.IMREAD_GRAYSCALE)
            if img is None:
                w.writerow([paths[i], subjects[i], 0.0, "", "", "", ""])
                hist["unreadable"] += 1
                continue
            H, W = img.shape[:2]
            dips, means = [], []
            for k in (0, 4):                      # 左眼 / 右眼（与数据集一致：0/4）
                x, y, bw, bh = boxes[i, k:k + 4]
                x0, y0 = int(max(0, x)), int(max(0, y))
                x1, y1 = int(min(W, x + bw)), int(min(H, y + bh))
                if x1 - x0 < 4 or y1 - y0 < 4:
                    dips.append(0.0); means.append(0.0)
                    continue
                d, m = _dip(img[y0:y1, x0:x1])
                dips.append(d); means.append(m)
            s = [max(0.0, min(1.0, 1.0 - d / args.scale)) for d in dips]
            blink = min(s)
            hist[round(blink, 1)] += 1
            w.writerow([paths[i], subjects[i], f"{blink:.4f}",
                        f"{dips[0]:.1f}", f"{dips[1]:.1f}",
                        f"{means[0]:.1f}", f"{means[1]:.1f}"])
            if (i + 1) % 2000 == 0:
                print(f"  ... {i + 1}/{hi}", flush=True)

    print(f"[blink] 完成 -> {out_csv}")
    print(f"[blink] 分数直方图（0.1 一档）: {dict(sorted(hist.items()))}")
    print("[blink] ⚠️ 下一步：在人工标注子集上核对阈值（默认 0.5）与误检率，再用于训练/评估")


if __name__ == "__main__":
    main()
