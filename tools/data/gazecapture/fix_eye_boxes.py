"""退化眼框修复：用**同被试中位框**替换退化框，并如实标记 filled。

背景（本仓库实测）：MediaPipe 在侧脸/遮挡/眯眼时会给出远侧眼轮廓点退化的框
（< 8 px 甚至 1x1），这类框喂给眼支流等于噪声；而**镜像**另一只眼会把瞳孔
水平翻转，引入系统性偏差，因此不采用。

处理规则（全部可调）：
  退化判定 = 非有限值 / 宽高 ≤ 0 / 宽或高 < min_px / 面积占人脸框比 < min_area_ratio
  替换      = 同被试**有效**框的中位数（左右眼各自取中位）
  兜底      = 该被试有效帧不足 min_subject_valid ⇒ 用全局中位数
  标记      = 被替换的行 filled=True（数据集可据此丢弃/降权）

用法：
    python -m tools.data.gazecapture.fix_eye_boxes --dry-run          # 只看统计
    python -m tools.data.gazecapture.fix_eye_boxes                    # 原地修（自动备份 .raw.npz）
    python -m tools.data.gazecapture.fix_eye_boxes --out fixed.npz --json report.json
    python -m tools.data.gazecapture.fix_eye_boxes --draw 12          # 额外导出前后对比图
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
    """先写 .tmp 再 os.replace：中断不会留下截断 zip。"""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def _load_meta(index_csv: Path):
    """只取 subject 与 bbox 面积所需列（149 万行约数秒）。"""
    subjects, face_area = [], []
    with open(index_csv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            subjects.append((r.get("subject") or "").strip())
            try:
                face_area.append(float(r["bbox_w"]) * float(r["bbox_h"]))
            except (TypeError, ValueError, KeyError):
                face_area.append(float("nan"))
    return np.array(subjects, dtype=object), np.array(face_area, dtype=np.float64)


def degenerate_mask(boxes: np.ndarray, face_area: np.ndarray,
                    min_px: float, min_area_ratio: float) -> np.ndarray:
    w, h = boxes[:, 2], boxes[:, 3]
    wr, hr = boxes[:, 6], boxes[:, 7]
    nan = (~np.isfinite(boxes)).any(1)
    nonpos = (w <= 0) | (h <= 0) | (wr <= 0) | (hr <= 0)
    small = (np.minimum(w, wr) < min_px) | (np.minimum(h, hr) < min_px)
    area = np.concatenate([[w * h], [wr * hr]], axis=0)  # 便于下面分别判断
    small_area = np.zeros(len(boxes), dtype=bool)
    ok_face = np.isfinite(face_area) & (face_area > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rl = (w * h) / np.where(ok_face, face_area, np.nan)
        rr = (wr * hr) / np.where(ok_face, face_area, np.nan)
    small_area = (((rl < min_area_ratio) | (rr < min_area_ratio)) & ok_face)
    del area
    return nan | nonpos | small | small_area


def subject_medians(boxes: np.ndarray, subjects: np.ndarray, valid: np.ndarray):
    """返回 {subject: (median_box_8, n_valid)}（只在有效行上取中位）。"""
    medians = {}
    for s in np.unique(subjects[valid]):
        m = valid & (subjects == s)
        medians[s] = (np.median(boxes[m], axis=0).astype(np.float32), int(m.sum()))
    return medians


def draw_compare(rows_paths, before, after, replaced, out_path: Path, n: int):
    """把"修复前/修复后"画在同一张图上（绿=修复前，黄=修复后）。"""
    import cv2
    cv2.setNumThreads(1)
    pick = [i for i in np.nonzero(replaced)[0][:n]]
    tiles = []
    for i in pick:
        img = cv2.imread(str(_ROOT / rows_paths[i]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        for off, color in ((0, (0, 200, 0)), (4, (0, 200, 0))):
            x, y, w, h = before[i, off:off + 4].astype(int)
            cv2.rectangle(img, (x, y), (x + max(w, 1), y + max(h, 1)), color, 1)
        for off in (0, 4):
            x, y, w, h = after[i, off:off + 4].astype(int)
            cv2.rectangle(img, (x, y), (x + max(w, 1), y + max(h, 1)), (0, 215, 255), 2)
        s = 200 / max(img.shape[:2])
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
        tiles.append(img)
    if not tiles:
        return
    cols = min(6, len(tiles))
    rows_n = (len(tiles) + cols - 1) // cols
    th, tw = tiles[0].shape[:2]
    canvas = np.full((rows_n * th, cols * tw, 3), 255, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        canvas[r * th:r * th + t.shape[0], c * tw:c * tw + t.shape[1]] = t
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    print(f"[fix] 前后对比图 → {out_path}（绿=修复前，黄=修复后）", flush=True)


def main():
    ap = argparse.ArgumentParser(description="退化眼框修复（同被试中位兜底）")
    ap.add_argument("--index-dir", default=str(GAZECAPTURE_INDEX_DIR))
    ap.add_argument("--eye-npz", default=None, help="默认 <index-dir>/eye_boxes.npz")
    ap.add_argument("--out", default=None, help="默认原地写回")
    ap.add_argument("--min-px", type=float, default=8.0)
    ap.add_argument("--min-area-ratio", type=float, default=0.003)
    ap.add_argument("--min-subject-valid", type=int, default=20)
    ap.add_argument("--backup", dest="backup", action="store_true", default=True)
    ap.add_argument("--no-backup", dest="backup", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    eye_npz = Path(args.eye_npz) if args.eye_npz else index_dir / "eye_boxes.npz"
    out_npz = Path(args.out) if args.out else eye_npz
    if not eye_npz.is_file():
        raise SystemExit(f"缺少眼框表: {eye_npz}")

    d = np.load(eye_npz)
    idx, boxes, filled = d["idx"].copy(), d["boxes"].copy(), d["filled"].copy()
    # 行号配对的唯一前提：idx 就是 0..N-1 的顺序。否则 subjects/face_area 会按
    # 错误的行配对（中位兜底与面积比都按错误被试算）。
    if not np.array_equal(idx, np.arange(len(idx))):
        raise SystemExit(f"{eye_npz} 的 idx 不是 0..N-1（可能未合并或乱序）⇒ "
                         f"先跑 build_eye_boxes_mp.py --merge")
    subjects, face_area = _load_meta(index_dir / "index.csv")
    if len(subjects) != len(boxes):
        raise SystemExit(f"索引行数 {len(subjects)} 与眼框表 {len(boxes)} 不一致 ⇒ "
                         f"先跑一次合并（build_eye_boxes_mp --merge）")

    deg = degenerate_mask(boxes, face_area, args.min_px, args.min_area_ratio)
    valid = ~deg
    print(f"[fix] 总行 {len(boxes)} | 退化 {int(deg.sum())} "
          f"({100*deg.mean():.2f}%) | 有效 {int(valid.sum())}", flush=True)

    med = subject_medians(boxes, subjects, valid)
    global_med = np.median(boxes[valid], axis=0).astype(np.float32) if valid.any() else None
    if global_med is None:
        raise SystemExit("没有任何有效眼框，无法兜底")

    after = boxes.copy()
    used_subj = np.zeros(len(boxes), dtype=bool)
    used_global = np.zeros(len(boxes), dtype=bool)
    for s in np.unique(subjects[deg]):
        m = deg & (subjects == s)
        if s in med and med[s][1] >= args.min_subject_valid:
            after[m] = med[s][0]
            used_subj |= m
        else:
            after[m] = global_med
            used_global |= m

    replaced = used_subj | used_global
    filled_out = filled | replaced
    print(f"[fix] 替换 {int(replaced.sum())} 行（同被试中位 {int(used_subj.sum())} | "
          f"全局中位 {int(used_global.sum())}）| filled {int(filled.sum())} → "
          f"{int(filled_out.sum())}", flush=True)

    # 每个被试的退化率（取 Top10，便于定位问题设备/被试）
    per_subj = {}
    for s in np.unique(subjects):
        m = subjects == s
        per_subj[str(s)] = {"frames": int(m.sum()), "degenerate": int((deg & m).sum())}
    worst = sorted(per_subj.items(),
                   key=lambda kv: kv[1]["degenerate"] / max(kv[1]["frames"], 1),
                   reverse=True)[:10]
    for s, v in worst:
        if v["degenerate"]:
            print(f"  被试 {s}: 退化 {v['degenerate']}/{v['frames']} "
                  f"({100*v['degenerate']/v['frames']:.1f}%)", flush=True)

    report = {"rows": len(boxes), "degenerate": int(deg.sum()),
              "replaced_by_subject": int(used_subj.sum()),
              "replaced_by_global": int(used_global.sum()),
              "filled_before": int(filled.sum()), "filled_after": int(filled_out.sum()),
              "min_px": args.min_px, "min_area_ratio": args.min_area_ratio,
              "min_subject_valid": args.min_subject_valid,
              "per_subject_worst": dict(worst)}

    if args.dry_run:
        print("[fix] dry-run：未写入", flush=True)
    else:
        if args.backup and out_npz.resolve() == eye_npz.resolve():
            bak = eye_npz.with_name(eye_npz.stem + ".raw.npz")
            if not bak.is_file():
                _atomic_savez(bak, idx=idx, boxes=boxes, filled=filled)
                print(f"[fix] 备份 → {bak}", flush=True)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        _atomic_savez(out_npz, idx=idx, boxes=after.astype(np.float32),
                      filled=filled_out.astype(bool))
        print(f"[fix] 写出 → {out_npz}", flush=True)

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[fix] 报告 → {p}", flush=True)

    if args.draw > 0 and replaced.any():
        paths = []
        with open(index_dir / "index.csv", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                paths.append((r.get("frame_path") or "").strip())
        draw_compare(paths, boxes, after, replaced,
                     Path("out/gazecapture/analysis/eye_fix_compare.jpg"), args.draw)


if __name__ == "__main__":
    main()
