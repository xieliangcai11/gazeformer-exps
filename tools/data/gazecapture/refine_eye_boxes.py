"""退化眼框**定向重检**：只对退化行放大图像重新检测，最后把仍失败的行标记 filled。

背景（2026-10-20 实测）：
  · MediaPipe 在侧脸/遮挡/远距离时，眼周轮廓点会退化 ⇒ 框 < 8px（全量 1.81%，27,042 行）；
  · 用"同被试绝对中位框"兜底 ✗ 会落在额头/头发上（手持设备会移动）；
  · 用"相对人脸框的中位偏移"兜底 ✗ 也不稳（自检 IoU 中位仅 0.16）；
  · **把图像放大 2× 再检测，96% 的退化行能恢复出正常眼框** ✓ ⇒ 用重检，不用编造。

流程（分片 + 可续跑 + 原子写）：
    # 1) 分片重检（CPU，建议 4 片并行；不占 GPU）
    for k in 0 1 2 3; do
      nohup python -u -m tools.data.gazecapture.refine_eye_boxes --shard $k --nshards 4 \
        > out/gazecapture/analysis/refine_s$k.log 2>&1 &
    done
    # 2) 合并：以 raw 为底，覆盖恢复行；仍失败的置 filled=True
    python -m tools.data.gazecapture.refine_eye_boxes --merge

约定：
    raw  = eye_boxes.raw.npz（首次运行会自动从 eye_boxes.npz 备份）
    分片 = eye_boxes_recovered_s{k}.npz（idx/boxes 各分片自己的恢复结果）
    最终 = eye_boxes.npz（行数/idx 与 index.csv 严格一致）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
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
    raise RuntimeError("找不到项目根")


_ROOT = _bootstrap()
from configs.gazecapture_config import GAZECAPTURE_INDEX_DIR  # noqa: E402


def _atomic_savez(path: Path, **arrays):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def degenerate_mask(boxes: np.ndarray, face_area: np.ndarray,
                    min_px: float = 8.0, min_area_ratio: float = 0.003) -> np.ndarray:
    w, h = boxes[:, 2], boxes[:, 3]
    wr, hr = boxes[:, 6], boxes[:, 7]
    nan = (~np.isfinite(boxes)).any(1)
    nonpos = (w <= 0) | (h <= 0) | (wr <= 0) | (hr <= 0)
    small = (np.minimum(w, wr) < min_px) | (np.minimum(h, hr) < min_px)
    ok_face = np.isfinite(face_area) & (face_area > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rl = (w * h) / np.where(ok_face, face_area, np.nan)
        rr = (wr * hr) / np.where(ok_face, face_area, np.nan)
    return nan | nonpos | small | (((rl < min_area_ratio) | (rr < min_area_ratio)) & ok_face)


def _load_index(index_csv: Path):
    paths, faces = [], []
    with open(index_csv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            paths.append(r["frame_path"])
            try:
                faces.append([float(r["bbox_x"]), float(r["bbox_y"]),
                              float(r["bbox_w"]), float(r["bbox_h"])])
            except (KeyError, ValueError):
                faces.append([np.nan] * 4)
    return paths, np.asarray(faces, dtype=np.float64)


def _ensure_raw(index_dir: Path) -> Path:
    raw = index_dir / "eye_boxes.raw.npz"
    cur = index_dir / "eye_boxes.npz"
    if not raw.is_file():
        if not cur.is_file():
            raise SystemExit(f"缺少 {cur}（无法建立 raw 备份）")
        d = np.load(cur)
        _atomic_savez(raw, idx=d["idx"], boxes=d["boxes"], filled=d["filled"])
        print(f"[refine] 建立 raw 备份 → {raw}", flush=True)
    return raw


def run_shard(args):
    index_dir = Path(args.index_dir)
    raw_path = _ensure_raw(index_dir)
    raw = np.load(raw_path)
    boxes0 = raw["boxes"].astype(np.float32)
    paths, faces = _load_index(index_dir / "index.csv")
    if len(paths) != len(boxes0):
        raise SystemExit(f"索引 {len(paths)} ≠ 眼框表 {len(boxes0)}")
    deg = degenerate_mask(boxes0, faces[:, 2] * faces[:, 3], args.min_px, args.min_area_ratio)
    todo = np.nonzero(deg)[0]
    print(f"[refine] 退化行 {len(todo)} / {len(boxes0)}", flush=True)

    out = index_dir / f"eye_boxes_recovered_s{args.shard}.npz"
    done = set()
    if out.is_file():
        d = np.load(out)
        done = set(int(i) for i in d["idx"])
        print(f"[refine] 续跑：已有 {len(done)} 行", flush=True)

    import cv2
    cv2.setNumThreads(1)
    from tools.data.common.eye_detect import detect_mp, make_landmarker
    lm = make_landmarker(_ROOT)

    mine = [i for i in todo if (i % args.nshards) == args.shard and i not in done]
    print(f"[refine] 分片 {args.shard}/{args.nshards}：待处理 {len(mine)}", flush=True)
    new_idx, new_box = [], []
    ok = 0
    for k, i in enumerate(mine, 1):
        img = cv2.imread(str(_ROOT / paths[i]) if not Path(paths[i]).is_absolute()
                         else paths[i], cv2.IMREAD_COLOR)
        if img is None:
            continue
        best = None
        for scale in args.scales:
            up = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            face, le, re, _ = detect_mp(up, lm)
            if face is None:
                continue
            cand = np.array([le[0], le[1], le[2], le[3], re[0], re[1], re[2], re[3]],
                            dtype=np.float32) / float(scale)
            if min(cand[2], cand[3], cand[6], cand[7]) >= args.min_px:
                best = cand
                break
        if best is None:
            continue
        new_idx.append(i)
        new_box.append(best)
        ok += 1
        if k % 200 == 0:
            print(f"  [{k}/{len(mine)}] 恢复 {ok}", flush=True)
        if k % args.save_every == 0:
            _append(out, new_idx, new_box)
            new_idx, new_box = [], []
            print(f"  [save] → {out}", flush=True)
    if new_idx:
        _append(out, new_idx, new_box)
    print(f"[refine] 分片 {args.shard} 完成：恢复 {ok}/{len(mine)} → {out}", flush=True)


def _append(path: Path, idx, boxes):
    if path.is_file():
        d = np.load(path)
        idx = np.concatenate([d["idx"].astype(np.int64), np.asarray(idx, np.int64)])
        boxes = np.vstack([d["boxes"].astype(np.float32), np.asarray(boxes, np.float32)])
    _atomic_savez(path, idx=np.asarray(idx, np.int64), boxes=np.asarray(boxes, np.float32))


def merge(args):
    index_dir = Path(args.index_dir)
    raw_path = _ensure_raw(index_dir)
    raw = np.load(raw_path)
    boxes = raw["boxes"].astype(np.float32).copy()
    filled = raw["filled"].astype(bool).copy()
    paths, faces = _load_index(index_dir / "index.csv")
    deg = degenerate_mask(boxes, faces[:, 2] * faces[:, 3], args.min_px, args.min_area_ratio)

    recovered = {}
    if not args.apply_recovered:
        print("[merge] 未指定 --apply-recovered ⇒ 忽略分片重检结果（退化帧一律丢弃）", flush=True)
    for f in (sorted(index_dir.glob("eye_boxes_recovered_s*.npz"))
              if args.apply_recovered else []):
        d = np.load(f)
        for i, b in zip(d["idx"].astype(np.int64), d["boxes"].astype(np.float32)):
            recovered[int(i)] = b
    n_app = 0
    for i, b in recovered.items():
        if 0 <= i < len(boxes):
            boxes[i] = b
            n_app += 1
    still = degenerate_mask(boxes, faces[:, 2] * faces[:, 3], args.min_px, args.min_area_ratio)
    filled_out = filled | still
    out = index_dir / "eye_boxes.npz"
    _atomic_savez(out, idx=np.arange(len(boxes), dtype=np.int64),
                  boxes=boxes, filled=filled_out.astype(bool))
    report = {"rows": int(len(boxes)), "degenerate_raw": int(deg.sum()),
              "recovered_applied": n_app, "still_degenerate": int(still.sum()),
              "filled_after": int(filled_out.sum())}
    print(f"[merge] {report}", flush=True)
    p = _ROOT / "out" / "gazecapture" / "analysis" / "refine_report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[merge] 报告 → {p}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="退化眼框定向重检（放大后重检测）")
    ap.add_argument("--index-dir", default=str(GAZECAPTURE_INDEX_DIR))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--scales", type=float, nargs="+", default=[2.0, 3.0],
                    help="依次尝试的放大倍数（命中即停）")
    ap.add_argument("--min-px", type=float, default=8.0)
    ap.add_argument("--min-area-ratio", type=float, default=0.003)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--merge", action="store_true",
                    help="合并到 eye_boxes.npz：**默认只标记**（退化行 filled=True，训练时丢弃）")
    ap.add_argument("--apply-recovered", action="store_true",
                    help="同时应用分片重检结果。注意：实测退化帧几乎都是闭眼/眯眼，"
                         "重检出的框没有注视信息 ⇒ 默认**不**应用")
    args = ap.parse_args()
    if args.merge:
        merge(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
