"""8 个眼框分片的**完整性体检**（只读 npz ✓ 不重新检测 ✓、不需要 mediapipe ✓）。

检查项：
  ① 每片行数 / idx 区间是否落在它应负责的 [k*N//8, (k+1)*N//8) 内 ✓
  ② 跨片重复（同一行被两片处理 ✗）
  ③ 全局缺口（应有 N 行 ✓ 少了多少 ✗）
  ④ filled（无脸/退化）比例 ✓
  ⑤ 框值合法性：NaN / inf / 负宽高 / 零宽高 / <8px（退化 ✗）

用法：
    python -m tools.data.gazecapture.check_shards
"""
from __future__ import annotations

import csv

import numpy as np

from configs.gazecapture_config import GAZECAPTURE_INDEX_DIR


def main():
    import argparse
    ap = argparse.ArgumentParser(description="8 个眼框分片的完整性体检")
    ap.add_argument("--nshards", type=int, default=8,
                    help="期望的分片数（不按现存文件数推断，否则丢片也会被判正常）")
    args = ap.parse_args()

    index = GAZECAPTURE_INDEX_DIR / "index.csv"
    with open(index, encoding="utf-8", newline="") as f:
        n_rows = sum(1 for _ in csv.DictReader(f))
    files = sorted(GAZECAPTURE_INDEX_DIR.glob("eye_boxes_shard*.npz"))
    nsh = args.nshards
    print(f"[体检] index 行数 N={n_rows}  |  期望分片数 {nsh}  |  实际文件 {len(files)}  "
          f"|  每片应≈{n_rows // max(nsh, 1)} 行")
    if len(files) != nsh:
        print(f"  [WARN] 分片文件数 {len(files)} != {nsh}（缺片 ⇒ 下面会报缺口 ✗）")

    all_idx, total_bad, total_filled, total = [], 0, 0, 0
    for f in files:
        d = np.load(f)
        idx, boxes, filled = d["idx"], d["boxes"], d["filled"]
        n = len(idx)
        total += n
        total_filled += int(filled.sum())
        all_idx.append(idx)
        k = int(f.stem.replace("eye_boxes_shard", ""))
        lo, hi = k * n_rows // max(nsh, 1), (k + 1) * n_rows // max(nsh, 1)

        # 约定：boxes = 左眼(x,y,w,h) + 右眼(x,y,w,h)（与 RGBDGaze 一致 ✓）
        w, h = boxes[:, 2], boxes[:, 3]
        wr, hr = boxes[:, 6], boxes[:, 7]
        nan = int((~np.isfinite(boxes)).any(1).sum())
        invalid = int(((w <= 0) | (h <= 0) | (wr <= 0) | (hr <= 0)).sum())
        tiny = int(((np.minimum(w, wr) < 8) | (np.minimum(h, hr) < 8)).sum())
        total_bad += max(nan, invalid)

        in_range = int(((idx >= lo) & (idx < hi)).sum())
        print(f"  片{k}: {n:>7} 行 | 区间 [{idx.min() if n else -1}, {idx.max() if n else -1}] "
              f"应属[{lo},{hi}) 命中 {in_range}/{n} | filled {int(filled.sum())} "
              f"| NaN {nan} 非法 {invalid} 退化<8px {tiny}")

    cat = np.concatenate(all_idx) if all_idx else np.zeros(0, np.int64)
    uniq = np.unique(cat)
    dup = len(cat) - len(uniq)
    missing = n_rows - len(uniq)
    print(f"[汇总] 总行 {total} | 唯一 {len(uniq)} | 跨片重复 {dup} | 缺口 {missing} "
          f"({100*missing/max(n_rows,1):.2f}%)")
    print(f"[汇总] filled {total_filled} ({100*total_filled/max(total,1):.2f}%) | "
          f"非法框 {total_bad}")
    # 缺口分布（按 10 万行一条）
    miss = np.setdiff1d(np.arange(n_rows, dtype=np.int64), uniq)
    if len(miss):
        bins = {}
        for m in miss:
            bins[int(m // 100000)] = bins.get(int(m // 100000), 0) + 1
        print("[缺口分布] 每 10 万行区间: " + "  ".join(f"{k*10}万:{v}" for k, v in sorted(bins.items())))
    verdict = "正常 ✓" if (dup == 0 and total_bad == 0 and missing == 0) else "需处理 ✗"
    print(f"[判定] {verdict}")


if __name__ == "__main__":
    main()
