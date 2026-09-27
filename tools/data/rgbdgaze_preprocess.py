"""RGBDGaze 预处理：扫描原始数据，生成训练所需的统一索引 CSV。

任务：把 RGBDGaze 官方的分级目录（p*/decoded/{activity}/{rgb,depth,label.csv}）
转成一个扁平索引，每一行对应一个样本：
    rgb_path, depth_path, bbox_x, bbox_y, bbox_w, bbox_h,
    device, screen_w, screen_h, gaze_x, gaze_y

关键处理：
1. 扫描所有 p*/decoded/*/label.csv，逐行读取 bbox 与注视点、设备。
2. label.csv 的 device 名带空格（如 "iPhone 12 Pro Max"），而 iphone_spec.csv
   的 key 无空格（如 "iPhone12 Pro Max"），做去空格匹配。
3. RGB 与 depth 都是 3 通道彩色图；RGB 1440x1080，depth 640x480，二者用
   同一 bbox（按比例缩放后）对齐，供双流模型使用。
4. 输出统一索引到 configs/rgbdgaze_config.py 的 RGBDGaze_INDEX_DIR。
5. 同时生成按人/按样本划分的 train/val/test 清单。
"""

import argparse
import csv
import re
import sys
from collections import OrderedDict
from pathlib import Path

# ---- 路径引导：确保从任意工作目录都能导入 configs / gazelab / tools ----
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from configs.rgbdgaze_config import (
    RGBDGaze_RAW_ROOT,
    RGBDGaze_INDEX_DIR,
    ACTIVITIES,
    IPHONE_SPEC_CSV,
)


def _norm(s: str) -> str:
    """去空格、小写，用于跨表匹配设备名。"""
    return re.sub(r"\s+", "", str(s)).lower()


def load_screen_spec() -> dict:
    """读 iphone_spec.csv -> {归一化设备名: (w_pt, h_pt, w_cm, h_cm)}"""
    spec = {}
    with open(IPHONE_SPEC_CSV, encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            name = row[0].strip()
            try:
                w_pt, h_pt = float(row[1]), float(row[2])
            except (ValueError, IndexError):
                continue
            spec[_norm(name)] = (w_pt, h_pt)
    return spec


def discover_samples():
    """扫描原始目录，返回有序样本列表。

    每个样本 dict: rgb, depth, bbox_xywh, device, screen_w, screen_h,
                   gaze_x, gaze_y, subject, activity
    """
    spec = load_screen_spec()
    samples = []
    subjects = sorted([p.name for p in RGBDGaze_RAW_ROOT.iterdir()
                       if p.is_dir() and re.fullmatch(r"p\d+", p.name)])

    for subj in subjects:
        for act in ACTIVITIES:
            act_dir = RGBDGaze_RAW_ROOT / subj / "decoded" / act
            if not act_dir.is_dir():
                continue
            label_csv = act_dir / "label.csv"
            if not label_csv.exists():
                continue
            with open(label_csv, encoding="utf-8") as f:
                rows = list(csv.reader(f))
            header = rows[0]
            # 定位列索引 (按列名)
            col = {name: i for i, name in enumerate(header)}
            for row in rows[1:]:
                try:
                    uid = row[col["uid"]]
                    bx = float(row[col["bbox_x"]])
                    by = float(row[col["bbox_y"]])
                    bw = float(row[col["bbox_w"]])
                    bh = float(row[col["bbox_h"]])
                    gx = float(row[col["gt_x_pt"]])
                    gy = float(row[col["gt_y_pt"]])
                    device = row[col["device"]]
                except (KeyError, IndexError, ValueError):
                    continue  # 跳过坏行
                key = _norm(device)
                if key not in spec:
                    # 设备不在屏幕规格表（如一些古老机型），跳过
                    continue
                w_pt, h_pt = spec[key]
                rgb = act_dir / "rgb" / f"{uid}.jpg"
                depth = act_dir / "depth" / f"{uid}.jpg"
                if not rgb.exists() or not depth.exists():
                    # RGB 与 depth 数量一致（论文所述），跳过缺任一
                    continue
                samples.append(dict(
                    rgb=str(rgb), depth=str(depth),
                    bbox_xywh=(bx, by, bw, bh),
                    device=device, screen_w=w_pt, screen_h=h_pt,
                    gaze_x=gx, gaze_y=gy,
                    subject=subj, activity=act,
                ))
    return samples


def write_index(samples, index_dir: Path):
    """把样本写入统一索引 csv（含全部列，含 subject/activity 便于划分）。"""
    index_dir.mkdir(parents=True, exist_ok=True)
    out = index_dir / "index.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rgb_path", "depth_path", "bbox_x", "bbox_y", "bbox_w",
                    "bbox_h", "device", "screen_w", "screen_h", "gaze_x",
                    "gaze_y", "subject", "activity"])
        for s in samples:
            bx, by, bw, bh = s["bbox_xywh"]
            w.writerow([s["rgb"], s["depth"], bx, by, bw, bh,
                        s["device"], s["screen_w"], s["screen_h"],
                        s["gaze_x"], s["gaze_y"], s["subject"], s["activity"]])
    subject_set = OrderedDict()
    for s in samples:
        subject_set.setdefault(s["subject"], 0)
        subject_set[s["subject"]] += 1
    print(f"[index] 写入 {len(samples)} 样本 -> {out}")
    print(f"[index] 被试 {len(subject_set)} 人")
    return out


def write_splits(samples, index_dir: Path, mode: str, ratios=(0.8, 0.1, 0.1)):
    """写 train/val/test 清单。

    mode:
      sample    按样本随机（忽略被试）
      subject   按被试 8:1:1
      activity  按前3活动训练、最后1活动测试
    """
    import random
    rng = random.Random(0)
    subjects = sorted({s["subject"] for s in samples})
    tr, va, te = ratios

    if mode == "sample":
        shuffled = samples[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_tr, n_va = int(n * tr), int(n * va)
        train, val = shuffled[:n_tr], shuffled[n_tr:n_tr + n_va]
        test = shuffled[n_tr + n_va:]
        # 去重索引 (默认清单需要可索引) -> 直接写路径集合
    elif mode == "subject":
        rng.shuffle(subjects)
        n = len(subjects)
        n_tr, n_va = int(n * tr), int(n * va)
        subj_tr = set(subjects[:n_tr])
        subj_va = set(subjects[n_tr:n_tr + n_va])
        subj_te = set(subjects[n_tr + n_va:])
        train = [s for s in samples if s["subject"] in subj_tr]
        val = [s for s in samples if s["subject"] in subj_va]
        test = [s for s in samples if s["subject"] in subj_te]
    elif mode == "activity":
        acts = ACTIVITIES
        train = [s for s in samples if s["activity"] in acts[:3]]
        val = [s for s in samples if s["activity"] in acts[3:4]]
        test = [s for s in samples if s["activity"] in acts[4:]]
    else:
        raise ValueError(mode)

    index_dir.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        with open(index_dir / f"{name}.txt", "w", encoding="utf-8") as f:
            for s in part:
                f.write(s["rgb"] + "\n")
        print(f"[split] {name}: {len(part)} 样本")


def main():
    p = argparse.ArgumentParser(description="RGBDGaze 预处理（建索引 + 划分）")
    p.add_argument("--index-dir", default=str(RGBDGaze_INDEX_DIR))
    p.add_argument("--split", default="sample",
                   choices=["sample", "subject", "activity"],
                   help="划分方式：sample随机 / subject按键验 / activity按活动")
    p.add_argument("--no-split", action="store_true",
                   help="只建索引，不写划分清单")
    args = p.parse_args()

    samples = discover_samples()
    index_dir = Path(args.index_dir)
    write_index(samples, index_dir)
    if not args.no_split:
        write_splits(samples, index_dir, args.split)


if __name__ == "__main__":
    main()