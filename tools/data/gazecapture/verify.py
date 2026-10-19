"""GazeCapture 数据验证脚本（只读，不修改任何文件）。

回答一个问题：**这份数据集现在能不能直接拿来训练/评估？** 覆盖：

  ① 索引表  : 表头是否齐全、行数、frame_path 是否为空/重复
  ② 划分    : train/val/test 是否存在、互不相交、覆盖索引、与索引 split 列是否一致
  ③ 标签    : gaze 是否落在屏幕内、screen/bbox 是否为正、数值列有无 NaN/inf
  ④ 图像    : 抽样解码（默认 500 张），检查可读性与 bbox 是否越界
  ⑤ 眼框表  : eye_boxes.npz 行数/索引连续性/框合法性/退化率/filled 率
  ⑥ 抽样复检: 可选 --recheck N，用 MediaPipe 现场重检 N 帧并与 npz 比对 IoU

判定：PASS（可训练）/ WARN（可训练但需留意）/ FAIL（不可训练，退出码 1）。

用法：
    python -m tools.data.gazecapture.verify                    # 默认抽样检查
    python -m tools.data.gazecapture.verify --check-images 0   # 跳过图��解码
    python -m tools.data.gazecapture.verify --recheck 50       # 现场重检 50 帧
    python -m tools.data.gazecapture.verify --json out/gazecapture/verify.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
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

#: 索引必须包含的列（缺失任何一列 ⇒ FAIL）
REQUIRED_COLUMNS = [
    "frame_path", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "device", "screen_w", "screen_h", "gaze_x", "gaze_y",
    "cm_px_x", "cm_px_y", "imu_x", "imu_y", "imu_z",
    "subject", "orientation", "split",
    "face_grid_x", "face_grid_y", "face_grid_w", "face_grid_h",
]

_NUMERIC = ["bbox_x", "bbox_y", "bbox_w", "bbox_h", "screen_w", "screen_h",
            "gaze_x", "gaze_y", "cm_px_x", "cm_px_y"]


class Report:
    def __init__(self):
        self.items: list[dict] = []

    def add(self, name: str, status: str, detail: str, data: dict | None = None):
        self.items.append({"check": name, "status": status, "detail": detail,
                           "data": data or {}})
        icon = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}[status]
        print(f"  [{status}] {icon} {name}: {detail}", flush=True)

    @property
    def failed(self) -> bool:
        return any(i["status"] == "FAIL" for i in self.items)

    @property
    def warned(self) -> bool:
        return any(i["status"] == "WARN" for i in self.items)

    def summary(self) -> dict:
        c = Counter(i["status"] for i in self.items)
        return {"pass": c["PASS"], "warn": c["WARN"], "fail": c["FAIL"]}


def _float_or_none(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def check_index(index_csv: Path, rep: Report):
    if not index_csv.is_file():
        rep.add("索引文件", "FAIL", f"不存在: {index_csv}")
        return None
    with open(index_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        rows = list(reader)
    if missing:
        rep.add("索引表头", "FAIL", f"缺少列: {missing}", {"header": header})
    else:
        rep.add("索引表头", "PASS", f"{len(header)} 列齐全")
    empty = sum(1 for r in rows if not (r.get("frame_path") or "").strip())
    paths = [(r.get("frame_path") or "").strip() for r in rows]
    dup = len(paths) - len(set(paths))
    rep.add("索引行数", "PASS" if rows else "FAIL", f"{len(rows)} 行",
            {"rows": len(rows)})
    rep.add("frame_path 完整性",
            "FAIL" if empty or dup else "PASS",
            f"空 {empty} | 重复 {dup}", {"empty": empty, "dup": dup})
    return rows


def check_labels(rows: list[dict], rep: Report):
    nan_cnt = defaultdict(int)
    oob = {"gaze_x": 0, "gaze_y": 0}
    nonpos = {"screen": 0, "bbox": 0}
    for r in rows:
        vals = {c: _float_or_none(r.get(c)) for c in _NUMERIC}
        for c, v in vals.items():
            if v is None:
                nan_cnt[c] += 1
        sw, sh = vals["screen_w"], vals["screen_h"]
        bw, bh = vals["bbox_w"], vals["bbox_h"]
        if sw is None or sh is None or sw <= 0 or sh <= 0 or bw is None or bh is None \
                or bw <= 0 or bh <= 0:
            nonpos["screen" if (sw is None or (sw or 0) <= 0) else "bbox"] += 1
            continue
        gx, gy = vals["gaze_x"], vals["gaze_y"]
        if gx is not None and not (-_TOL * sw <= gx <= sw * (1 + _TOL)):
            oob["gaze_x"] += 1
        if gy is not None and not (-_TOL * sh <= gy <= sh * (1 + _TOL)):
            oob["gaze_y"] += 1

    bad_nan = {k: v for k, v in nan_cnt.items() if v}
    bad_oob = {k: v for k, v in oob.items() if v}
    bad_nonpos = {k: v for k, v in nonpos.items() if v}
    status = "FAIL" if bad_nonpos else ("WARN" if bad_nan or bad_oob else "PASS")
    rep.add("标签合法性", status,
            f"非数值 {bad_nan or '无'} | 越界 {bad_oob or '无'} | 非正 {bad_nonpos or '无'}",
            {"nan": bad_nan, "out_of_screen": bad_oob, "non_positive": bad_nonpos})


_TOL = 1e-3  # 允许 0.1% 屏幕的浮点误差


def check_splits(index_dir: Path, rows: list[dict], rep: Report,
                 splits=("train", "val", "test"), partial: bool = True):
    """partial=True：本地可能只有完整数据集的子集 ⇒ 缺失文件/未入索引只记 WARN。

    结构性错误（同一帧出现在两个划分、划分文件内部重复）在两种模式下都是 FAIL。
    """
    index_paths = {(r.get("frame_path") or "").strip() for r in rows}
    by_split_idx = Counter((r.get("split") or "").strip() for r in rows)
    seen: dict[str, set] = {}
    for s in splits:
        f = index_dir / f"{s}.txt"
        if not f.is_file():
            rep.add(f"划分 {s}", "WARN" if partial else "FAIL", f"不存在: {f}")
            continue
        names = [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
        uniq = set(names)
        missing = len(uniq - index_paths)
        seen[s] = uniq
        if not names:
            rep.add(f"划分 {s}", "WARN" if partial else "FAIL", f"空文件: {f}")
            continue
        rep.add(f"划分 {s}", ("WARN" if partial else "FAIL") if missing else "PASS",
                f"{len(names)} 行（唯一 {len(uniq)}）| 不在索引中 {missing} | "
                f"索引 split 列 {by_split_idx.get(s, 0)} 行",
                {"rows": len(names), "unique": len(uniq), "not_in_index": missing,
                 "index_rows": by_split_idx.get(s, 0)})
    keys = [k for k in seen if seen[k]]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            inter = seen[keys[i]] & seen[keys[j]]
            if inter:
                rep.add(f"划分互斥 {keys[i]}/{keys[j]}", "FAIL",
                        f"重叠 {len(inter)} 条（例: {next(iter(inter))}）")
    if len(keys) == len([s for s in splits if (index_dir / f"{s}.txt").is_file()]):
        union = set().union(*seen.values()) if seen else set()
        rep.add("划分覆盖", "PASS" if union == index_paths else "WARN",
                f"并集 {len(union)} / 索引 {len(index_paths)}"
                f"（差 {len(index_paths - union)} 条）")


def check_images(rows: list[dict], n: int, rep: Report, rng: random.Random,
                 partial: bool = True):
    if n <= 0:
        return
    import cv2
    cv2.setNumThreads(1)
    sample = rng.sample(rows, min(n, len(rows)))
    bad_read, oob, sizes = 0, 0, Counter()
    worst = []
    for r in sample:
        rel = (r.get("frame_path") or "").strip()
        img = cv2.imread(str(_ROOT / rel), cv2.IMREAD_COLOR)
        if img is None:
            bad_read += 1
            if len(worst) < 5:
                worst.append(rel)
            continue
        h, w = img.shape[:2]
        sizes[(w, h)] += 1
        bx, by = _float_or_none(r.get("bbox_x")), _float_or_none(r.get("bbox_y"))
        bw, bh = _float_or_none(r.get("bbox_w")), _float_or_none(r.get("bbox_h"))
        if None in (bx, by, bw, bh) or bx < 0 or by < 0 or bx + bw > w + 1 or by + bh > h + 1:
            oob += 1
    if bad_read > max(2, 0.01 * n):
        st = "WARN" if partial else "FAIL"
    else:
        st = "WARN" if (bad_read or oob) else "PASS"
    rep.add("图像抽样", st,
            f"抽样 {len(sample)} | 无法读取 {bad_read} | bbox 越界 {oob} | "
            f"尺寸分布 top={sizes.most_common(2)}",
            {"sampled": len(sample), "unreadable": bad_read, "bbox_oob": oob,
             "bad_examples": worst})


def check_eye_boxes(eye_npz: Path, n_rows: int, rep: Report):
    if not eye_npz.is_file():
        rep.add("眼框表", "WARN", f"不存在: {eye_npz}（训练不带眼支流时可忽略）")
        return None
    d = np.load(eye_npz)
    need = {"idx", "boxes", "filled"}
    missing = need - set(d.files)
    if missing:
        rep.add("眼框表结构", "FAIL", f"缺少数组: {sorted(missing)}", {"keys": d.files})
        return None
    idx, boxes, filled = d["idx"], d["boxes"], d["filled"]
    ok_shape = boxes.ndim == 2 and boxes.shape[1] == 8
    if not ok_shape:
        rep.add("眼框表结构", "FAIL", f"boxes 形状 {boxes.shape}（应为 [N,8]）")
        return None
    sorted_uniq = bool(len(idx) == len(np.unique(idx)) and (np.diff(idx) > 0).all())
    aligned = bool(len(idx) == n_rows and np.array_equal(idx, np.arange(n_rows)))
    w, h, wr, hr = boxes[:, 2], boxes[:, 3], boxes[:, 6], boxes[:, 7]
    invalid = int(((w <= 0) | (h <= 0) | (wr <= 0) | (hr <= 0)).sum())
    nan = int((~np.isfinite(boxes)).any(1).sum())
    tiny = int(((np.minimum(w, wr) < 8) | (np.minimum(h, hr) < 8)).sum())
    row_ok = len(boxes) == n_rows
    status = "PASS"
    detail = (f"{len(boxes)} 行 | 行数一致 {row_ok} | idx==arange {aligned} | "
              f"非法 {invalid} | NaN {nan} | 退化<8px {tiny} ({100*tiny/max(len(boxes),1):.2f}%) | "
              f"filled {int(filled.sum())}")
    if not ok_shape or not row_ok or not aligned or invalid or nan:
        status = "FAIL"
    elif not sorted_uniq:
        status = "FAIL"
    elif tiny > 0.02 * len(boxes):
        status = "WARN"
    rep.add("眼框表", status, detail,
            {"rows": len(boxes), "aligned": aligned, "sorted_unique": sorted_uniq,
             "invalid": invalid, "nan": nan, "degenerate": tiny,
             "filled": int(filled.sum())})
    return {"idx": idx, "boxes": boxes, "filled": filled}


def check_recheck(rows: list[dict], eye, n: int, rep: Report, rng: random.Random):
    if n <= 0 or eye is None:
        return
    try:
        from tools.data.common.eye_detect import detect_mp, make_landmarker
    except Exception as e:  # noqa: BLE001
        rep.add("眼框复检", "WARN", f"无法导入检测器: {e}")
        return
    import cv2
    cv2.setNumThreads(1)
    lm = make_landmarker(_ROOT)
    sample = rng.sample(range(len(rows)), min(n, len(rows)))
    ious, noface = [], 0
    for i in sample:
        r = rows[i]
        img = cv2.imread(str(_ROOT / (r.get("frame_path") or "").strip()), cv2.IMREAD_COLOR)
        if img is None:
            continue
        face, le, re, _ = detect_mp(img, lm)
        if face is None:
            noface += 1
            continue
        for off, det in ((0, le), (4, re)):
            a = eye["boxes"][i, off:off + 4]
            ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
            bx0, by0, bx1, by1 = det[0], det[1], det[0] + det[2], det[1] + det[3]
            iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
            ih = max(0.0, min(ay1, by1) - max(ay0, by0))
            inter = iw * ih
            union = a[2] * a[3] + det[2] * det[3] - inter
            ious.append(inter / union if union > 0 else 0.0)
    if not ious:
        rep.add("眼框复检", "WARN", "没有可比较的帧")
        return
    med = float(np.median(ious))
    low = float(np.mean(np.array(ious) < 0.5))
    st = "PASS" if med >= 0.8 else ("WARN" if med >= 0.6 else "FAIL")
    rep.add("眼框复检", st,
            f"重检 {len(ious)} 只眼 | IoU 中位 {med:.3f} | IoU<0.5 占比 {100*low:.1f}% | 无脸 {noface}",
            {"iou_median": med, "iou_low_ratio": low})


def main():
    ap = argparse.ArgumentParser(description="GazeCapture 数据验证（只读）")
    ap.add_argument("--index-dir", default=str(GAZECAPTURE_INDEX_DIR))
    ap.add_argument("--check-images", type=int, default=500, help="抽样解码的图像数（0=跳过）")
    ap.add_argument("--recheck", type=int, default=0, help="用 MediaPipe 现场重检的帧数（0=跳过）")
    ap.add_argument("--eye-npz", default=None, help="默认 <index-dir>/eye_boxes.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="把结果写成 JSON")
    ap.add_argument("--strict", action="store_true", help="WARN 也按失败退出")
    ap.add_argument("--strict-data", action="store_true",
                    help="按完整数据集校验（本地多为子集，默认关闭 ⇒ 缺失文件/未入索记为 WARN）")
    args = ap.parse_args()
    partial = not args.strict_data

    index_dir = Path(args.index_dir)
    index_csv = index_dir / "index.csv"
    eye_npz = Path(args.eye_npz) if args.eye_npz else index_dir / "eye_boxes.npz"
    rng = random.Random(args.seed)
    rep = Report()

    print(f"[verify] 索引目录: {index_dir}", flush=True)
    rows = check_index(index_csv, rep)
    if rows:
        check_labels(rows, rep)
        check_splits(index_dir, rows, rep, partial=partial)
        check_images(rows, args.check_images, rep, rng, partial=partial)
        eye = check_eye_boxes(eye_npz, len(rows), rep)
        check_recheck(rows, eye, args.recheck, rep, rng)
        subj = Counter((r.get("subject") or "").strip() for r in rows)
        per = sorted(subj.values())
        rep.add("被试分布", "PASS",
                f"{len(subj)} 名被试 | 每名帧数 p05={per[len(per)//20]} "
                f"p50={per[len(per)//2]} max={per[-1]}")

    s = rep.summary()
    verdict = "FAIL" if rep.failed else ("WARN" if rep.warned else "PASS")
    print(f"\n[verify] 结论: {verdict}  (PASS {s['pass']} / WARN {s['warn']} / FAIL {s['fail']})",
          flush=True)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"verdict": verdict, "summary": s,
                                  "items": rep.items}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"[verify] 详情 → {out}", flush=True)
    return 1 if (rep.failed or (args.strict and rep.warned)) else 0


if __name__ == "__main__":
    sys.exit(main())
