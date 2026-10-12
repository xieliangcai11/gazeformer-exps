"""GazeCapture 预处理：解压归档 -> 扫描元数据 -> 生成统一索引 CSV + 划分清单。

产出（默认写到 configs/gazecapture_config.py 的 GAZECAPTURE_INDEX_DIR）：
    index.csv      每行一个可用样本（人脸框 + 屏幕注视点 + 设备/几何/IMU 等）
    train.txt / val.txt / test.txt
                   官方划分（info.json 的 Dataset 字段）下的帧路径清单
    日志文件       out/gazecapture/preprocess/logs/<时间戳>_gazecapture_preprocess.txt

=====================================================================
坐标系说明（本脚本据此决定"不做任何旋转"，已用样本数据实测）
=====================================================================
GazeCapture 的 screen.json 有 Orientation ∈ {1,2,3,4}（1 竖屏 / 2 竖屏倒置 /
3 横屏 home 键在右 / 4 横屏 home 键在左），直觉上横竖屏之间需要旋转对齐。实测结论：

(a) 图像帧本身始终"人脸正立"。以 README 所述"受试者左眼出现在图像右侧"为准，
    取 (左眼中心 - 右眼中心) 向量的角度作为人脸在图像中的滚转角，四种 orientation
    的中位角分别为 +0.0° / +0.0° / +3.8° / -2.3°，100% 落在 ±30° 内 —— 即
    图像已经是世界正立的，不需要旋转。
(b) 标签（XPts/YPts）与相机/世界坐标系始终对齐。对**同一屏幕尺寸**分别拟合
    (XPts, YPts) -> (XCam, YCam) 的线性变换，四种 orientation 得到的矩阵完全一致
    （M ≈ diag(+s, -s)，两列角度 0° / -90°）。例如 1024x768 屏幕下
    ori3 与 ori4 的 M 都是 [+0.0198 0; 0 -0.0200]。

=> 图像坐标系与标签坐标系在所有 orientation 下天然一致，**无需旋转**。
   这与官方 PyTorch 预处理（pytorch/prepareDataset.py）不旋转的做法一致。

另外两点实现约定：
* appleLeftEye/RightEye 的 X,Y 是相对"人脸框"的，不是相对整帧（官方 README）。
  本脚本只用它们的 IsValid 做可用性过滤，不用其坐标。
* 某帧的可用性 = appleFace ∩ appleLeftEye ∩ appleRightEye ∩ faceGrid 四者的
  IsValid 同时为 1（与官方 prepareDataset.py 的 allValid 一致）。
"""

import argparse
import csv
import json
import math
import re
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

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

from configs.gazecapture_config import (  # noqa: E402
    PROJECT_ROOT,
    GAZECAPTURE_ARCHIVE_DIR,
    GAZECAPTURE_EXTRACT_DIR,
    GAZECAPTURE_INDEX_DIR,
    GAZECAPTURE_DEVICE_CSV,
    DEVICE_ALIASES,
    SPLITS,
    MIN_VALID_FRAMES_PER_SUBJECT,
)

INDEX_FIELDS = [
    "frame_path", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "device", "screen_w", "screen_h",
    "gaze_x", "gaze_y", "cm_px_x", "cm_px_y",
    "imu_x", "imu_y", "imu_z",
    "subject", "orientation", "split",
    "face_grid_x", "face_grid_y", "face_grid_w", "face_grid_h",
    "dot_num", "frame_index",
]


# --------------------------------------------------------------------------
# 日志：同时打印到 stdout 并落盘（沿用项目内 training 脚本的 write_log 约定）
# --------------------------------------------------------------------------
class Logger:
    """极简双写日志：stdout + 文件，行首带时间戳。"""

    def __init__(self, path: Path = None):
        self.path = Path(path) if path else None
        self.lines = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def __call__(self, msg=""):
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}" if msg else ""
        print(line, flush=True)
        self.lines.append(line)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def banner(self, title):
        self("")
        self("=" * 72)
        self(title)
        self("=" * 72)


# --------------------------------------------------------------------------
# 设备规格
# --------------------------------------------------------------------------
def _norm_device(s: str) -> str:
    """设备名归一化：去掉空格/连字符并小写，用于跨表匹配（'iPhone 5S' vs 'iPhone 5s'）。"""
    return re.sub(r"[\s\-_]+", "", str(s)).lower()


def _opt_float(s):
    """解析可空数值列：'NaN'/空 -> None，其余转 float。"""
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def load_device_spec(log) -> dict:
    """读官方 apple_device_data.csv -> {归一化设备名: 规格 dict}。

    规格 dict: {w_mm, h_mm, w_pt, h_pt, w_pt_zoom, h_pt_zoom}。
    zoom 为 iOS「显示缩放(Display Zoom)」下的逻辑点数（部分机型为 None）。
    同一机型在不同被试上可能开启/关闭显示缩放，此时 screen.json 的 W/H 不同、
    但物理屏幕尺寸不变 => 每点厘米不同，必须按帧选择分母（见 cm_per_point）。
    """
    spec = {}
    with open(GAZECAPTURE_DEVICE_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            spec[_norm_device(row["DeviceName"].strip())] = dict(
                w_mm=float(row["DeviceScreenWidthMm"]),
                h_mm=float(row["DeviceScreenHeightMm"]),
                w_pt=float(row["DeviceScreenWidthPoints"]),
                h_pt=float(row["DeviceScreenHeightPoints"]),
                w_pt_zoom=_opt_float(row["DeviceScreenWidthPointsZoomed"]),
                h_pt_zoom=_opt_float(row["DeviceScreenHeightPointsZoomed"]),
            )
    log(f"[device] 载入官方屏幕规格 {len(spec)} 款机型 <- {GAZECAPTURE_DEVICE_CSV}")
    for alias, target in DEVICE_ALIASES.items():
        key, tkey = _norm_device(alias), _norm_device(target)
        if tkey in spec:
            spec[key] = spec[tkey]
            log(f"[device] 别名 {alias!r} -> {target!r}（同面板，复用其屏幕尺寸）")
        else:
            log(f"[device] 警告：别名目标 {target!r} 不在规格表中，忽略别名 {alias!r}")
    return spec


def cm_per_point(spec, device, screen_w=None, screen_h=None):
    """每屏幕点对应的物理厘米 (cm_px_x, cm_px_y)；无法确定时返回 (nan, nan)。

    逻辑点数是"竖屏基准 + 可选的显示缩放"，而屏幕点数在横/竖屏之间会互换，
    物理面板尺寸也随之互换；再叠加 iOS「显示缩放」的两套点数，共 4 种组合。
    故按该帧实际的 (screen_w, screen_h) 在下列组合中匹配：
        标准竖屏 (w_pt, h_pt) / 标准横屏 (h_pt, w_pt)
        缩放竖屏 (w_pt_z, h_pt_z) / 缩放横屏 (h_pt_z, w_pt_z)
    匹配不到时返回 NaN（宁可标记未知，也不静默套用错误尺度）。
    """
    e = spec.get(_norm_device(device))
    if e is None:
        return float("nan"), float("nan")

    pairs = [(e["w_pt"], e["h_pt"])]
    if e["w_pt_zoom"] and e["h_pt_zoom"]:
        pairs.append((e["w_pt_zoom"], e["h_pt_zoom"]))

    if screen_w is None or screen_h is None:
        w_pt, h_pt = pairs[0]
        w_mm, h_mm = e["w_mm"], e["h_mm"]
    else:
        hit = None
        for w_pt, h_pt in pairs:
            for cw, ch, w_mm, h_mm in ((w_pt, h_pt, e["w_mm"], e["h_mm"]),
                                       (h_pt, w_pt, e["h_mm"], e["w_mm"])):
                if abs(screen_w - cw) < 1 and abs(screen_h - ch) < 1:
                    hit = (w_mm, h_mm, cw, ch)
                    break
            if hit:
                break
        if hit is None:
            return float("nan"), float("nan")
        w_mm, h_mm, w_pt, h_pt = hit

    return (w_mm / 10.0) / w_pt, (h_mm / 10.0) / h_pt


# --------------------------------------------------------------------------
# 解压
# --------------------------------------------------------------------------
def extract_archives(archive_dir: Path, extract_dir: Path, log, force=False,
                     subjects=None) -> None:
    """把 archive_dir 下所有 *.tar.gz 解压到 extract_dir（顶层 NNNNN/ 原样释放）。

    subjects: 只解压这些被试的归档（分批解压用，避免一次性占满磁盘）。
    force   : 已存在的被试目录也重新解压。
    """
    archives = sorted(archive_dir.glob("*.tar.gz"))
    if subjects:
        want = set(subjects)
        archives = [a for a in archives if a.name.split(".")[0] in want]
    if not archives:
        log(f"[extract] {archive_dir} 下没有匹配的 *.tar.gz，跳过解压")
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    log(f"[extract] 待解压 {len(archives)} 个归档 -> {extract_dir}")
    done = skipped = 0
    for i, arc in enumerate(archives, 1):
        subject = arc.name.split(".")[0]          # "00002.tar.gz" -> "00002"
        dest = extract_dir / subject
        if dest.is_dir() and any(dest.iterdir()) and not force:
            skipped += 1
            log(f"[extract] ({i}/{len(archives)}) {arc.name} 已解压，跳过")
            continue
        try:
            with tarfile.open(arc, "r:gz") as tf:
                tf.extractall(extract_dir)
            done += 1
            log(f"[extract] ({i}/{len(archives)}) {arc.name} -> {dest}")
        except (tarfile.TarError, OSError) as e:
            log(f"[extract] ({i}/{len(archives)}) {arc.name} 解压失败：{e!r}（跳过）")
    log(f"[extract] 完成：新解压 {done} 个，跳过 {skipped} 个")


# --------------------------------------------------------------------------
# 扫描
# --------------------------------------------------------------------------
def _read_json(path: Path, log):
    """读 JSON；失败返回 None 并记日志（不让单个坏文件中断整轮预处理）。"""
    if not path.is_file():
        log(f"[scan] 警告：缺少 {path}（跳过该被试）")
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log(f"[scan] 警告：解析 {path} 失败：{e!r}（跳过该被试）")
        return None


def _gravity_index(motion):
    """把 motion.json 整理成 {DotNum: [(Time, (gx, gy, gz)), ...]}（按 Time 升序）。"""
    idx = defaultdict(list)
    for s in motion:
        g = s.get("GravityX")
        if not isinstance(g, dict):
            continue
        try:
            idx[s["DotNum"]].append(
                (float(s["Time"]), (float(g["X"]), float(g["Y"]), float(g["Z"]))))
        except (KeyError, TypeError, ValueError):
            continue
    for k in idx:
        idx[k].sort(key=lambda t: t[0])
    return idx


def _nearest_gravity(idx, dot_num, t, tol=0.05):
    """按 (DotNum, 最接近的 Time) 取重力向量；匹配不上返回 None。"""
    lst = idx.get(dot_num)
    if not lst:
        return None
    best = min(lst, key=lambda e: abs(e[0] - t))
    return best[1] if abs(best[0] - t) <= tol else None


def discover_samples(raw_root: Path, device_spec, log, subjects=None):
    """扫描 raw_root 下每个被试目录，返回 (样本列表, 统计 Counter)。

    subjects: 仅处理这些被试 ID（用于小样本自测）；None 表示全部。
    统计量在"整段丢弃"过滤之后统一重算，保证与返回的样本严格一致。
    """
    if not raw_root.is_dir():
        raise FileNotFoundError(f"原始数据目录不存在：{raw_root}")
    sub_dirs = sorted([p for p in raw_root.iterdir()
                       if p.is_dir() and re.fullmatch(r"\d+", p.name)])
    if subjects:
        want = set(subjects)
        sub_dirs = [p for p in sub_dirs if p.name in want]
        missing = want - {p.name for p in sub_dirs}
        if missing:
            log(f"[scan] 警告：请求的被试未找到：{sorted(missing)}")
    log(f"[scan] 待扫描被试 {len(sub_dirs)} 个 <- {raw_root}")

    samples = []
    drops = Counter()                 # 逐帧丢弃原因
    unknown_devices = Counter()
    skipped_subjects = []

    for si, sub_dir in enumerate(sub_dirs, 1):
        subject = sub_dir.name
        info = _read_json(sub_dir / "info.json", log)
        frames = _read_json(sub_dir / "frames.json", log)
        face = _read_json(sub_dir / "appleFace.json", log)
        eye_l = _read_json(sub_dir / "appleLeftEye.json", log)
        eye_r = _read_json(sub_dir / "appleRightEye.json", log)
        grid = _read_json(sub_dir / "faceGrid.json", log)
        dot = _read_json(sub_dir / "dotInfo.json", log)
        screen = _read_json(sub_dir / "screen.json", log)
        motion = _read_json(sub_dir / "motion.json", log)
        if None in (info, frames, face, eye_l, eye_r, grid, dot, screen):
            skipped_subjects.append(subject)
            continue

        split = str(info.get("Dataset", "")).strip().lower()
        device = str(info.get("DeviceName", "")).strip()
        if split not in SPLITS:
            log(f"[scan] {subject}: info.json 的 Dataset={split!r} 非 train/val/test，跳过")
            skipped_subjects.append(subject)
            continue
        cm_px_x, cm_px_y = cm_per_point(device_spec, device)
        if math.isnan(cm_px_x):           # 设备不在规格表
            unknown_devices[device] += 1
        gindex = _gravity_index(motion) if motion else {}
        n = min(len(frames), len(face["X"]), len(face["Y"]), len(face["W"]),
                len(face["H"]), len(dot["XPts"]), len(dot["YPts"]),
                len(screen["W"]), len(screen["H"]), len(screen["Orientation"]))
        n = min(n, len(face["IsValid"]), len(eye_l["IsValid"]),
                len(eye_r["IsValid"]), len(grid["IsValid"]))
        if n == 0:
            log(f"[scan] {subject}: 元数据为空，跳过")
            skipped_subjects.append(subject)
            continue

        # 先按被试缓冲，整段有效帧数达标后才并入总样本
        local = []
        for j in range(n):
            # 四类检测同时有效才可用（与官方 prepareDataset.py 的 allValid 一致）
            if not (face["IsValid"][j] and eye_l["IsValid"][j]
                    and eye_r["IsValid"][j] and grid["IsValid"][j]):
                drops["drop_invalid_detection"] += 1
                continue
            fname = frames[j]
            frame_path = sub_dir / "frames" / fname
            if not frame_path.is_file():
                drops["drop_missing_frame"] += 1
                continue
            bw, bh = float(face["W"][j]), float(face["H"][j])
            if bw <= 1 or bh <= 1:
                drops["drop_degenerate_bbox"] += 1
                continue
            # 路径存"相对项目根 + 前向斜杠"，跨 Windows/Linux 通用
            rel = frame_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
            # 每帧的屏幕点数决定「显示缩放」与否 => 每点厘米须按帧选取
            sw, sh = float(screen["W"][j]), float(screen["H"][j])
            cpx, cpy = cm_per_point(device_spec, device, sw, sh)
            if math.isnan(cpx):
                drops["cm_scale_unmatched"] += 1
            g = _nearest_gravity(gindex, dot["DotNum"][j], float(dot["Time"][j]))
            if g is None:
                drops["imu_missing"] += 1
                g = (0.0, 0.0, 0.0)
            local.append(dict(
                frame_path=rel,
                bbox=(float(face["X"][j]), float(face["Y"][j]), bw, bh),
                device=device,
                screen_w=sw, screen_h=sh,
                gaze_x=float(dot["XPts"][j]), gaze_y=float(dot["YPts"][j]),
                cm_px_x=cpx, cm_px_y=cpy,
                imu=g,
                subject=subject,
                orientation=int(screen["Orientation"][j]),
                split=split,
                face_grid=(int(grid["X"][j]), int(grid["Y"][j]),
                           int(grid["W"][j]), int(grid["H"][j])),
                dot_num=int(dot["DotNum"][j]),
                frame_index=int(re.sub(r"\D", "", fname) or j),
            ))

        if len(local) < MIN_VALID_FRAMES_PER_SUBJECT:
            # 有效帧太少的被试整段丢弃，避免噪声（官方亦说明部分被试没有有效帧）
            log(f"[scan] {subject}: 有效帧仅 {len(local)} < "
                f"{MIN_VALID_FRAMES_PER_SUBJECT}，整段丢弃")
            skipped_subjects.append(subject)
            continue

        samples.extend(local)
        if si % 50 == 0 or si == len(sub_dirs):
            log(f"[scan] ({si}/{len(sub_dirs)}) {subject}: 有效 {len(local)}/{n} 帧")

    # 统计量从最终样本重算，保证与写出的 index.csv 完全一致
    stats = Counter(drops)
    for s in samples:
        stats[f"frame_{s['split']}"] += 1
        stats[f"orient_{s['orientation']}"] += 1
        stats[f"device_{s['device']}"] += 1

    log(f"[scan] 有效样本合计 {len(samples)}")
    if skipped_subjects:
        log(f"[scan] 跳过的被试 {len(skipped_subjects)} 个：{skipped_subjects[:10]}"
            + (" ..." if len(skipped_subjects) > 10 else ""))
    if unknown_devices:
        log(f"[scan] 警告：下列设备不在屏幕规格表中，cm 换算为 NaN，需补 CSV/别名："
            f"{dict(unknown_devices)}")
    return samples, stats


# --------------------------------------------------------------------------
# 写出
# --------------------------------------------------------------------------
def write_index(samples, index_dir: Path, log) -> Path:
    """写统一索引 CSV（列集合见 INDEX_FIELDS）。"""
    index_dir.mkdir(parents=True, exist_ok=True)
    out = index_dir / "index.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(INDEX_FIELDS)
        for s in samples:
            bx, by, bw, bh = s["bbox"]
            gx, gy, gw, gh = s["face_grid"]
            ix, iy, iz = s["imu"]
            w.writerow([
                s["frame_path"], bx, by, bw, bh,
                s["device"], s["screen_w"], s["screen_h"],
                s["gaze_x"], s["gaze_y"], s["cm_px_x"], s["cm_px_y"],
                ix, iy, iz,
                s["subject"], s["orientation"], s["split"],
                gx, gy, gw, gh, s["dot_num"], s["frame_index"],
            ])
    log(f"[index] 写入 {len(samples)} 样本 x {len(INDEX_FIELDS)} 列 -> {out}")
    return out


def write_splits(samples, index_dir: Path, log) -> None:
    """按官方 info.json 的 Dataset 字段写 train/val/test 清单（每行一个帧路径）。"""
    index_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        part = [s for s in samples if s["split"] == name]
        with open(index_dir / f"{name}.txt", "w", encoding="utf-8") as f:
            for s in part:
                f.write(s["frame_path"] + "\n")
        n_sub = len({s["subject"] for s in part})
        log(f"[split] {name}: {len(part)} 帧 / {n_sub} 名被试 -> {index_dir / (name + '.txt')}")


def write_summary(index_dir: Path, samples, stats, log) -> None:
    """把扫描统计写成 summary.txt，便于事后核对（设备/朝向/划分分布）。"""
    lines = []
    lines.append(f"总有效样本: {len(samples)}")
    lines.append(f"被试数: {len({s['subject'] for s in samples})}")
    lines.append("")
    lines.append("== 划分 ==")
    for k in SPLITS:
        lines.append(f"  {k}: {stats.get('frame_' + k, 0)}")
    lines.append("")
    lines.append("== Orientation 分布 ==")
    for o in (1, 2, 3, 4):
        name = {1: "竖屏", 2: "竖屏倒置", 3: "横屏(home 键右)", 4: "横屏(home 键左)"}[o]
        lines.append(f"  {o} ({name}): {stats.get('orient_' + str(o), 0)}")
    lines.append("")
    lines.append("== 设备分布 ==")
    for k, v in sorted(stats.items()):
        if k.startswith("device_"):
            lines.append(f"  {k[len('device_'):]}: {v}")
    lines.append("")
    lines.append("== 丢弃 / 异常计数 ==")
    for k in ("drop_invalid_detection", "drop_missing_frame", "drop_degenerate_bbox",
              "imu_missing", "cm_scale_unmatched"):
        lines.append(f"  {k}: {stats.get(k, 0)}")
    out = index_dir / "summary.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"[summary] 统计摘要 -> {out}")
    for ln in lines:
        log("[summary] " + ln)


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="GazeCapture 预处理：解压 -> 扫描 -> 建索引 + 划分",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--archive-dir", default=str(GAZECAPTURE_ARCHIVE_DIR),
                   help="*.tar.gz 归档所在目录")
    p.add_argument("--extract-dir", default=str(GAZECAPTURE_EXTRACT_DIR),
                   help="解压目标目录")
    p.add_argument("--raw-root", default=None,
                   help="已解压的被试目录所在目录；默认用 --extract-dir")
    p.add_argument("--index-dir", default=str(GAZECAPTURE_INDEX_DIR),
                   help="索引与清单输出目录")
    p.add_argument("--extract", action="store_true",
                   help="先解压 --archive-dir 下的 *.tar.gz")
    p.add_argument("--force-extract", action="store_true",
                   help="配合 --extract：已存在的被试目录也重新解压")
    p.add_argument("--no-extract", action="store_true",
                   help="只扫描已解压目录，不碰归档（默认行为）")
    p.add_argument("--subjects", default=None,
                   help="仅处理这些被试 ID，逗号分隔（小样本自测 / 分批解压用）；"
                        "同时作用于 --extract 的解压范围")
    p.add_argument("--log-file", default=None,
                   help="日志文件路径；默认写到 out/gazecapture/preprocess/logs/")
    args = p.parse_args()

    # 日志
    if args.log_file:
        log_path = Path(args.log_file)
    else:
        log_dir = PROJECT_ROOT / "out" / "gazecapture" / "preprocess" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_gazecapture_preprocess.txt"
    log = Logger(log_path)

    log.banner("GazeCapture 预处理")
    log(f"命令: {' '.join(sys.argv)}")
    log(f"项目根: {PROJECT_ROOT}")
    log(f"日志文件: {log_path}")

    subjects = None
    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
        log(f"限定被试: {subjects}")

    # 1) 可选解压
    if args.extract:
        log.banner("步骤 1/4：解压归档")
        extract_archives(Path(args.archive_dir), Path(args.extract_dir), log,
                         force=args.force_extract, subjects=subjects)
    else:
        log.banner("步骤 1/4：跳过解压（未指定 --extract）")

    raw_root = Path(args.raw_root) if args.raw_root else Path(args.extract_dir)

    # 2) 设备规格
    log.banner("步骤 2/4：载入设备屏幕规格")
    device_spec = load_device_spec(log)

    # 3) 扫描
    log.banner(f"步骤 3/4：扫描被试元数据 <- {raw_root}")
    log("坐标系结论：图像与标签在所有 orientation 下天然对齐，不做任何旋转"
        "（依据见脚本头部说明）")
    samples, stats = discover_samples(raw_root, device_spec, log, subjects=subjects)

    # 4) 写出
    log.banner("步骤 4/4：写索引 / 划分 / 摘要")
    index_dir = Path(args.index_dir)
    if samples:
        write_index(samples, index_dir, log)
        write_splits(samples, index_dir, log)
        write_summary(index_dir, samples, stats, log)
    else:
        log("[index] 没有有效样本，未写出任何文件")

    log.banner("完成")
    log(f"索引目录: {index_dir}")


if __name__ == "__main__":
    main()
