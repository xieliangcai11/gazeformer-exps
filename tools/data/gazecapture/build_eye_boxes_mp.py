"""GazeCapture 全量眼框生成（MediaPipe ✓ 分片 + 续跑 + 定期存盘）。

为什么这样设计（本仓库实测 ✓）：
  · 服务器上 MediaPipe **导入要 2–3 分钟** ✗（不是坏 ✓）⇒ 单次运行越长越划算 ✓ ⇒ 用**分片**
    把 149 万帧拆成若干段 ✓，每段一个长期运行 ✓；
  · MediaPipe 与多进程/fork 不稳 ✗ ⇒ **不用多进程** ✓，改用「**多个分片并行**」= 多进程等价 ✓；
  · 每 N 帧存盘 + 启动时读旧 npz 跳过已完成 idx ✓ ⇒ **中断不丢** ✓。

输出与 RGBDGaze **同 schema** ✓（数据集侧可直接复用逻辑 ✓）：
    idx    [N]    int64   index.csv 的**行号** ✓
    boxes  [N,8]  float32 左眼(x,y,w,h)+右眼(x,y,w,h) ✓ **不旋转** ✓（GazeCapture 帧本就正立 ✓）
    filled [N]    bool    是否走了兜底（未检测到 ✓）

用法（服务器 ✓ 建议 nohup ✓；4 个分片并行 ⇒ 149 万帧预计数小时 ✓）：
    for k in 0 1 2 3; do
      nohup python -u -m tools.data.gazecapture.build_eye_boxes_mp \
        --shard $k --nshards 4 --save-every 20000 \
        --out data/GazeCapture/index/eye_boxes_shard$k.npz \
        > out/gazecapture/analysis/eyebox_shard$k.log 2>&1 &
    done
    # 全部完成后合并（把 4 个分片合成一个 npz ✓）
    python -m tools.data.gazecapture.build_eye_boxes_mp --merge \
        data/GazeCapture/index/eye_boxes_shard*.npz --out data/GazeCapture/index/eye_boxes.npz
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from pathlib import Path

import cv2
import numpy as np

#: 关键性能修复 ✓：服务器 88 核 + 高负载时，OpenCV 默认 88 线程会让**读图/解码**
# 从 ~2ms 暴涨到 ~230ms（实测 ✓）。检测器本身单线程 ✓ ⇒ 固定为 1 线程最快 ✓。
cv2.setNumThreads(1)


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
from configs.gazecapture_config import GAZECAPTURE_INDEX_DIR  # noqa: E402
from tools.data.common.eye_detect import detect_mp, make_landmarker  # noqa: E402


def _load_npz(p: Path):
    if not p.is_file():
        return np.zeros(0, np.int64), np.zeros((0, 8), np.float32), np.zeros(0, bool)
    d = np.load(p, allow_pickle=True)
    return d["idx"].astype(np.int64), d["boxes"].astype(np.float32), d["filled"].astype(bool)


def _save_npz(p: Path, idx, boxes, filled):
    """原子写：先写 .tmp 再 os.replace，避免中断留下截断 zip（后续 np.load 会崩）。"""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, idx=idx, boxes=boxes, filled=filled)
    os.replace(tmp, p)


def _is_degenerate(le, re_, face, min_px: float, min_area_ratio: float) -> bool:
    """眼框算退化：短边过小，或面积占人脸框比例过小。"""
    w, h, wr, hr = float(le[2]), float(le[3]), float(re_[2]), float(re_[3])
    if min(w, h, wr, hr) < min_px or w <= 0 or h <= 0 or wr <= 0 or hr <= 0:
        return True
    fa = max(float(face[2]) * float(face[3]), 1e-9)
    return (w * h) / fa < min_area_ratio or (wr * hr) / fa < min_area_ratio


def _detect(img, lm, scales, min_px: float, min_area_ratio: float):
    """检测；若退化且给了重试尺度，则放大重试（坐标已除回原尺度）。"""
    face, le, re_, partial = detect_mp(img, lm)
    if face is None:
        return face, le, re_, partial
    if scales and _is_degenerate(le, re_, face, min_px, min_area_ratio):
        h0, w0 = img.shape[:2]
        for s in scales:
            up = cv2.resize(img, (int(w0 * s), int(h0 * s)),
                            interpolation=cv2.INTER_CUBIC)
            f2, l2, r2, p2 = detect_mp(up, lm)
            if f2 is None:
                continue
            cand = ((l2[0] / s, l2[1] / s, l2[2] / s, l2[3] / s),
                    (r2[0] / s, r2[1] / s, r2[2] / s, r2[3] / s))
            if not _is_degenerate(cand[0], cand[1], face, min_px, min_area_ratio):
                return face, cand[0], cand[1], p2
    return face, le, re_, partial


def _row_count(index: Path) -> int:
    """索引行数（不含表头）。用 sidecar 缓存，避免每个分片都重扫 400MB 索引。"""
    side = index.with_name(index.name + ".rows")
    st = index.stat()
    try:
        parts = side.read_text().split()
        if int(parts[0]) == st.st_mtime_ns and int(parts[1]) == st.st_size:
            return int(parts[2])
    except (OSError, ValueError, IndexError):
        pass
    with open(index, "rb") as f:
        n = sum(1 for _ in f) - 1
    try:
        side.write_text(f"{st.st_mtime_ns} {st.st_size} {n}")
    except OSError:
        pass
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(GAZECAPTURE_INDEX_DIR / "index.csv"))
    ap.add_argument("--out", default=None,
                    help="输出 npz；分片模式下默认 eye_boxes_shard<k>.npz（避免多片写同一文件）")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 行（调试）")
    ap.add_argument("--merge", nargs="*", default=None,
                    help="合并模式：把给定 npz 合并到 --out ✓（不再检测）")
    ap.add_argument("--min-px", type=float, default=8.0,
                    help="眼框短边小于该值判为退化（filled=True，训练时剔除）")
    ap.add_argument("--min-area-ratio", type=float, default=0.003,
                    help="眼框面积/人脸框面积 小于该值判为退化")
    ap.add_argument("--retry-scales", type=float, nargs="*", default=None,
                    help="退化帧的放大重试尺度（如 2.0；默认不重试。实测多数退化=闭眼，"
                         "重试收益很小）")
    args = ap.parse_args()

    if args.out is None:
        # 分片模式必须写各自的文件：以前默认所有片都写 eye_boxes.npz ⇒ 互相覆盖/截断
        name = ("eye_boxes.npz" if args.nshards == 1
                else f"eye_boxes_shard{args.shard}.npz")
        args.out = str(GAZECAPTURE_INDEX_DIR / name)

    if args.merge:
        idxs, boxes, fill = [], [], []
        for f in args.merge:
            i, b, fl = _load_npz(Path(f))
            idxs.append(i); boxes.append(b); fill.append(fl)
        idx = np.concatenate(idxs) if idxs else np.zeros(0, np.int64)
        bx = np.concatenate(boxes) if boxes else np.zeros((0, 8), np.float32)
        fl = np.concatenate(fill) if fill else np.zeros(0, bool)
        order = np.argsort(idx)
        _save_npz(Path(args.out), idx[order], bx[order], fl[order])
        print(f"[merge] {len(args.merge)} 个分片 → {args.out}（{len(idx)} 行 ✓ "
              f"覆盖 {len(np.unique(idx))} ✓ 兜底 {fl.sum()}）", flush=True)
        return

    out = Path(args.out)
    idx_done, boxes_done, fill_done = _load_npz(out)
    done = set(int(i) for i in idx_done)
    print(f"[shard {args.shard}/{args.nshards}] 已有 {len(done)} 行 ✓", flush=True)

    # 流式读取本分片区间：以前 list(DictReader) 把 149 万行 dict 全载入，
    # 单进程 2~3 GB，与"多分片并行"的目的直接冲突。
    n_total = _row_count(Path(args.index))
    if args.limit:
        n_total = min(n_total, args.limit)
    lm = make_landmarker(_ROOT)
    #: **连续区间分片** ✗：步长取行 (i % nshards) 会让多进程在全盘随机寻道，
    # 磁盘顺序读优势全丢（实测 2.2 帧/秒/片）；连续区间则每片顺序读。
    lo = (args.shard * n_total) // args.nshards
    hi = ((args.shard + 1) * n_total) // args.nshards
    print(f"[shard {args.shard}/{args.nshards}] 行区间 [{lo}, {hi}) 共 {hi-lo} 行"
          f"（索引 {n_total} 行）", flush=True)
    reader = csv.DictReader(open(args.index, encoding="utf-8", newline=""))
    rows = itertools.islice(reader, lo, hi)

    new_idx, new_box, new_fill = [], [], []
    processed = 0
    for i, r in enumerate(rows, start=lo):
        if i in done:
            continue
        p = r.get("frame_path")
        if not p:
            continue
        img = cv2.imread(str(_ROOT / p), cv2.IMREAD_COLOR)
        if img is None:
            le = re = (0, 0, 1, 1); filled = True
        else:
            face, le, re, partial = _detect(img, lm, args.retry_scales,
                                            args.min_px, args.min_area_ratio)
            if face is None:
                le = re = (0, 0, 1, 1); filled = True
            else:
                filled = bool(partial) or _is_degenerate(
                    le, re, face, args.min_px, args.min_area_ratio)
        b = [*le, *re]
        new_idx.append(i); new_box.append(b); new_fill.append(filled)
        processed += 1
        if processed % 2000 == 0:
            print(f"  已处理 {processed} ✓（idx 到 {i}）", flush=True)
        if processed % args.save_every == 0:
            _save_npz(out, np.concatenate([idx_done, np.array(new_idx, np.int64)]),
                      np.vstack([boxes_done, np.array(new_box, np.float32)]),
                      np.concatenate([fill_done, np.array(new_fill, bool)]))
            print(f"  [save] {processed} 行 ✓ → {out}", flush=True)

    if new_idx:
        _save_npz(out, np.concatenate([idx_done, np.array(new_idx, np.int64)]),
                  np.vstack([boxes_done, np.array(new_box, np.float32)]),
                  np.concatenate([fill_done, np.array(new_fill, bool)]))
    print(f"[shard {args.shard}] 完成 ✓ 本次 {len(new_idx)} 行 | 兜底 {sum(new_fill)} "
          f"（{100*sum(new_fill)/max(len(new_fill),1):.1f}%）→ {out}", flush=True)


if __name__ == "__main__":
    main()
