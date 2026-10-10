"""GazeCapture 预处理 / Dataset「全操作留痕」核对工具。

目的：把数据从「原始 JSON」到「模型实际吃到的 tensor」之间**每一步**的
输入/输出数值与图像都落盘，供人工比对，从而判断预处理与 Dataset 是否有错。

产物（默认 out/gazecapture/dataset_check/）：
    report.txt                逐样本完整推导过程（每一步的数值）
    report.csv                逐样本数值表（原始 JSON vs 索引 vs Dataset 返回值）
    summary.txt               全局统计 + 自动一致性校验结果（PASS/FAIL 清单）
    images/NNN_*.png          每样本一张多面板对照图
    images/contact_sheet.png  全部样本缩略总览
    dropped/*.png             被过滤掉的帧（含原因），用于核对过滤逻辑

多面板图内容（每样本一张）：
    A 原始整帧：红=人脸 bbox，绿=实际裁剪框，黄=人脸中心
    B 裁剪结果（Dataset 裁剪后、resize 到 image_size，未归一化）
    C 模型实际输入反归一化还原（应与 B 像素级一致 → 校验归一化/反归一化）
    D 屏幕示意图：按 screen_w:screen_h 比例画屏，标出注视点与归一化坐标
    E 若开启增强：翻转/光度扰动后的输入

用法：
    python tools/data/gazecapture/dump_checks.py                     # 24 个分层抽样样本
    python tools/data/gazecapture/dump_checks.py --num-samples 60 --image-size 448
    python tools/data/gazecapture/dump_checks.py --augment           # 额外导出增强前后对比
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---- 路径引导 ----
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from configs.gazecapture_config import (  # noqa: E402
    PROJECT_ROOT, GAZECAPTURE_EXTRACT_DIR, GAZECAPTURE_INDEX_DIR, IMAGE_SIZE,
    CROP_MARGIN, GEOM_RAND_ASPECT, GEOM_RAND_WIDTH_CM, GEOM_RAND_HEIGHT_CM,
)
from gazelab.datasets.gazecapture import GazeCaptureDataset, _resolve_data_path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import (  # noqa: E402
    cm_per_point, load_device_spec, _norm_device, _nearest_gravity, _gravity_index,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FONT = None
# 优先中文字体，否则面板上的中文标签会变成方块
for _cand in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
              "C:/Windows/Fonts/simsun.ttc", "C:/Windows/Fonts/arial.ttf",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
    try:
        FONT = ImageFont.truetype(_cand, 13)
        break
    except OSError:
        continue
if FONT is None:
    FONT = ImageFont.load_default()


# ==========================================================================
# 小工具
# ==========================================================================
def draw_text(dr, xy, text, fill=(255, 255, 255), font=None):
    dr.text(xy, text, fill=fill, font=font or FONT)


def to_pil_uint8(bgr_or_rgb, is_bgr=True):
    """numpy 图 -> PIL RGB。"""
    a = bgr_or_rgb
    if is_bgr and a.ndim == 3 and a.shape[2] == 3:
        a = a[:, :, ::-1]
    return Image.fromarray(np.ascontiguousarray(a))


def denorm_tensor(t):
    """[3,S,S] ImageNet 归一化 tensor -> HWC uint8（RGB）。"""
    a = t.permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def fit(img: Image.Image, max_w, max_h):
    """等比缩放到不超过 (max_w, max_h)。"""
    s = min(max_w / img.width, max_h / img.height)
    return img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))))


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ==========================================================================
# 单个样本的推导与核对
# ==========================================================================
class SubjectCache:
    """按需缓存被试的 JSON 与 motion 索引，避免重复读盘。"""

    def __init__(self, raw_root: Path):
        self.raw_root = raw_root
        self.cache = {}

    def get(self, subject):
        if subject not in self.cache:
            d = self.raw_root / subject
            js = {name: read_json(d / f"{name}.json") for name in (
                "info", "frames", "appleFace", "appleLeftEye", "appleRightEye",
                "faceGrid", "dotInfo", "screen", "motion")}
            js["_dir"] = d
            js["_gravity"] = _gravity_index(js["motion"]) if js["motion"] else {}
            self.cache[subject] = js
        return self.cache[subject]


def trace_sample(ds, idx, subject_cache, device_spec, log_lines, ds_aug=None):
    """对单个样本执行"完整推导 + 核对"，返回 (面板图, 数值 dict, 校验列表)。

    ds     : 必须为**确定性**数据集（augment=False），
             否则裁剪/翻转随机化，数值报告不可复现、无法人工比对。
    ds_aug : 可选，单独用于导出"增强后输入"面板（RNG 按样本号固定种子）。
    """
    row = ds.rows[idx]
    checks = []
    num = {}

    def rec(k, v):
        num[k] = v
        return v

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        return ok

    # ---------- 1) 索引列原样 ----------
    log_lines.append(f"========== 样本 #{idx} ==========")
    log_lines.append("[1] index.csv 行（预处理写出的原始值）")
    for k, v in row.items():
        log_lines.append(f"      {k} = {v}")
        rec(f"idx_{k}", v)

    # ---------- 2) 回溯原始 JSON ----------
    subject = row["subject"]
    js = subject_cache.get(subject)
    d = js["_dir"]
    # 用 frame_index 定位 JSON 下标（frames.json 顺序 == 元数据数组顺序）
    frames = js["frames"]
    j = frames.index(Path(row["frame_path"]).name)
    info, face, dot, screen, grid = (js["info"], js["appleFace"], js["dotInfo"],
                                     js["screen"], js["faceGrid"])
    log_lines.append(f"[2] 回溯原始 JSON：{d}")
    log_lines.append(f"      info.json      Dataset={info['Dataset']!r} "
                     f"DeviceName={info['DeviceName']!r} TotalFrames={info['TotalFrames']}")
    log_lines.append(f"      frames.json[{j}] = {frames[j]!r}")
    log_lines.append(f"      appleFace[{j}]    X={face['X'][j]:.6f} Y={face['Y'][j]:.6f} "
                     f"W={face['W'][j]:.6f} H={face['H'][j]:.6f} IsValid={face['IsValid'][j]}")
    log_lines.append(f"      dotInfo[{j}]      DotNum={dot['DotNum'][j]} "
                     f"XPts={dot['XPts'][j]:.6f} YPts={dot['YPts'][j]:.6f} "
                     f"XCam={dot['XCam'][j]:.6f} YCam={dot['YCam'][j]:.6f} Time={dot['Time'][j]:.6f}")
    log_lines.append(f"      screen[{j}]       W={screen['W'][j]} H={screen['H'][j]} "
                     f"Orientation={screen['Orientation'][j]}")
    log_lines.append(f"      faceGrid[{j}]     X={grid['X'][j]} Y={grid['Y'][j]} "
                     f"W={grid['W'][j]} H={grid['H'][j]} IsValid={grid['IsValid'][j]}")

    # 校验：索引值 == JSON 原值
    check("bbox_x == appleFace.X", abs(float(row["bbox_x"]) - face["X"][j]) < 1e-6,
          f'{row["bbox_x"]} vs {face["X"][j]}')
    check("bbox_y == appleFace.Y", abs(float(row["bbox_y"]) - face["Y"][j]) < 1e-6)
    check("bbox_w == appleFace.W", abs(float(row["bbox_w"]) - face["W"][j]) < 1e-6)
    check("bbox_h == appleFace.H", abs(float(row["bbox_h"]) - face["H"][j]) < 1e-6)
    check("gaze_x == dotInfo.XPts", abs(float(row["gaze_x"]) - dot["XPts"][j]) < 1e-6)
    check("gaze_y == dotInfo.YPts", abs(float(row["gaze_y"]) - dot["YPts"][j]) < 1e-6)
    check("screen_w/h == screen.json", float(row["screen_w"]) == float(screen["W"][j])
          and float(row["screen_h"]) == float(screen["H"][j]))
    check("orientation == screen.json", int(row["orientation"]) == int(screen["Orientation"][j]))
    check("split == info.Dataset", row["split"] == info["Dataset"].strip().lower())
    check("device == info.DeviceName", row["device"] == info["DeviceName"])
    check("face_grid == faceGrid.json",
          [int(row[f"face_grid_{k}"]) for k in "xywh"]
          == [grid["X"][j], grid["Y"][j], grid["W"][j], grid["H"][j]])
    check("dot_num == dotInfo.DotNum", int(row["dot_num"]) == int(dot["DotNum"][j]))
    check("IsValid 四者皆 1（该帧应被保留）",
          all(js[n]["IsValid"][j] for n in ("appleFace", "appleLeftEye",
                                            "appleRightEye", "faceGrid")))

    # ---------- 3) cm/点 与屏幕物理尺寸 ----------
    dev, sw, sh = row["device"], float(row["screen_w"]), float(row["screen_h"])
    spec = device_spec.get(_norm_device(dev))
    cpx, cpy = cm_per_point(device_spec, dev, sw, sh)
    log_lines.append("[3] 设备屏幕规格匹配（apple_device_data.csv）")
    if spec:
        log_lines.append(f"      规格表命中 {dev!r}: 物理={spec['w_mm']}x{spec['h_mm']}mm "
                         f"点数={spec['w_pt']}x{spec['h_pt']} 缩放点数="
                         f"{spec['w_pt_zoom']}x{spec['h_pt_zoom']}")
    else:
        log_lines.append(f"      规格表未命中 {dev!r} → cm 为 NaN")
    log_lines.append(f"      本帧 screen={sw:.0f}x{sh:.0f} → 每点厘米=({cpx:.8f}, {cpy:.8f})")
    log_lines.append(f"      屏幕物理尺寸 = 点数 x 每点厘米 = "
                     f"({sw * cpx:.4f} cm, {sh * cpy:.4f} cm)")
    check("cm_px_x == 索引值", abs(cpx - float(row["cm_px_x"])) < 1e-12)
    check("cm_px_y == 索引值", abs(cpy - float(row["cm_px_y"])) < 1e-12)
    check("cm_px 非 NaN（设备可换算）", not math.isnan(cpx))
    if spec:
        # 屏幕物理长宽应与规格表一致（横竖屏只是互换）
        got = sorted([round(sw * cpx, 3), round(sh * cpy, 3)])
        exp = sorted([round(spec["w_mm"] / 10, 3), round(spec["h_mm"] / 10, 3)])
        check("屏幕物理尺寸 == 规格表(±0.02cm)", abs(got[0] - exp[0]) < 0.02
              and abs(got[1] - exp[1]) < 0.02, f"{got} vs {exp}")

    # ---------- 4) IMU 对齐 ----------
    g_expect = _nearest_gravity(js["_gravity"], dot["DotNum"][j], float(dot["Time"][j]))
    log_lines.append("[4] IMU 对齐（motion.json 按 DotNum + 最近 Time）")
    if g_expect is None:
        log_lines.append("      未匹配到运动样本 → 索引记 (0,0,0)")
        g_expect = (0.0, 0.0, 0.0)
    else:
        log_lines.append(f"      匹配到重力向量 = {tuple(round(v, 6) for v in g_expect)}")
    check("imu == motion 重力向量(或全 0)",
          all(abs(float(row[k]) - v) < 1e-9 for k, v in
              zip(("imu_x", "imu_y", "imu_z"), g_expect)))

    # ---------- 5) 标签归一化 ----------
    gx, gy = float(row["gaze_x"]), float(row["gaze_y"])
    gxn, gyn = gx / sw, gy / sh
    log_lines.append("[5] 标签归一化 gaze/screen")
    log_lines.append(f"      ({gx:.6f}/{sw:.0f}, {gy:.6f}/{sh:.0f}) = ({gxn:.8f}, {gyn:.8f})")
    check("归一化标签在 [0,1]", 0.0 <= gxn <= 1.0 and 0.0 <= gyn <= 1.0)

    # ---------- 6) 裁剪框（像素级） ----------
    img_bgr = _read_bgr(_resolve_data_path(row["frame_path"]))
    H, W = img_bgr.shape[:2]
    bx, by, bw, bh = (float(row["bbox_x"]), float(row["bbox_y"]),
                      float(row["bbox_w"]), float(row["bbox_h"]))
    cx_, cy_ = bx + bw / 2, by + bh / 2
    side = max(bw, bh) * CROP_MARGIN
    x0 = max(0, int(cx_ - side / 2))
    x1 = min(W, int(cx_ + side / 2))
    y0 = max(0, int(cy_ - side / 2))
    y1 = min(H, int(cy_ + side / 2))
    log_lines.append("[6] 人脸裁剪（crop_face_square）")
    log_lines.append(f"      整帧尺寸 (WxH) = {W}x{H}")
    log_lines.append(f"      bbox 中心=({cx_:.2f},{cy_:.2f})  边长=max(w,h)*margin="
                     f"{max(bw, bh):.2f}*{CROP_MARGIN}={side:.2f}")
    log_lines.append(f"      裁剪框(像素, 已按图像边界截断) = x[{x0}:{x1}] y[{y0}:{y1}] "
                     f"→ {x1 - x0}x{y1 - y0} 像素")
    log_lines.append(f"      resize -> {ds.image_size}x{ds.image_size}")
    check("裁剪框非空", x1 > x0 and y1 > y0)
    check("裁剪框在图像内", 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H)
    clamped = (x0 != int(cx_ - side / 2)) or (y0 != int(cy_ - side / 2)) \
        or (x1 != int(cx_ + side / 2)) or (y1 != int(cy_ + side / 2))
    num["crop_clamped"] = int(clamped)
    num["crop_px"] = f"{x0},{y0},{x1},{y1}"

    # ---------- 7) Dataset 实际返回 ----------
    inp, label = ds[idx]
    log_lines.append("[7] Dataset.__getitem__ 返回值")
    t = inp.rgb
    log_lines.append(f"      rgb   shape={tuple(t.shape)} dtype={t.dtype} "
                     f"min={t.min():.4f} max={t.max():.4f} mean={t.mean():.4f}")
    log_lines.append(f"      label = [{label[0]:.8f}, {label[1]:.8f}]")
    log_lines.append(f"      screen_cm = {[round(v, 4) for v in inp.screen_cm.tolist()]}")
    log_lines.append(f"      imu       = {[round(v, 6) for v in inp.imu.tolist()]}")
    log_lines.append(f"      face_grid = {[int(v) for v in inp.face_grid.tolist()]}")
    rec("ds_label_x", float(label[0]))
    rec("ds_label_y", float(label[1]))
    rec("ds_rgb_shape", str(tuple(t.shape)))
    rec("ds_rgb_min", float(t.min()))
    rec("ds_rgb_max", float(t.max()))
    rec("ds_screen_cm_x", float(inp.screen_cm[0]))
    rec("ds_screen_cm_y", float(inp.screen_cm[1]))

    check("label == gaze/screen", abs(float(label[0]) - gxn) < 1e-6
          and abs(float(label[1]) - gyn) < 1e-6)
    check("rgb shape == (3, S, S)", tuple(t.shape) == (3, ds.image_size, ds.image_size))
    check("rgb 数值范围合理(|x|<6)", float(t.abs().max()) < 6.0)
    check("screen_cm == 点数*每点厘米",
          abs(float(inp.screen_cm[0]) - sw * cpx) < 1e-4
          and abs(float(inp.screen_cm[1]) - sh * cpy) < 1e-4)
    check("imu tensor == 索引 imu",
          all(abs(float(inp.imu[i]) - float(row[k])) < 1e-6 for i, k in
              enumerate(("imu_x", "imu_y", "imu_z"))))

    # ---------- 8) 归一化 / 反归一化往返 ----------
    back = denorm_tensor(t)                       # [S,S,3] uint8 RGB
    crop_pil = to_pil_uint8(_crop_only(img_bgr, (x0, y0, x1, y1), ds.image_size))
    diff = np.abs(back.astype(np.int16) - np.array(crop_pil, dtype=np.int16))
    log_lines.append("[8] 归一化往返校验（tensor 反归一化 vs 直接裁剪图）")
    log_lines.append(f"      最大像素差={diff.max()}  平均像素差={diff.mean():.4f}")
    check("反归一化 == 裁剪图(±1)", diff.max() <= 1, f"max={diff.max()}")

    # ---------- 9) 增强（可选；单开一个数据集的实例，固定随机种子保证可复现） ----------
    aug_arr, aug_label = None, None
    if ds_aug is not None:
        ds_aug._rng.seed(1234 + idx)
        aug_inp, aug_lab = ds_aug[idx]
        aug_arr = denorm_tensor(aug_inp.rgb)          # HWC uint8 (RGB)
        aug_label = (float(aug_lab[0]), float(aug_lab[1]))
        # 增强后标签必须与"水平翻转 + 几何域随机化"的复合变换精确一致。
        # 几何随机化会把标签重投影到合成屏 (W', H')（由 aug_inp.geom 给出）：
        #     u' = 0.5 + s·(u-0.5)·w_cm/W'    s=+1 未翻转 / s=-1 已翻转
        #     v' = 0.5 + (v-0.5)·h_cm/H'
        # 若随机化未触发，则 (W',H') 就是真实屏幕，退化为"只可能水平翻转"。
        Wp = float(math.exp(float(aug_inp.geom[1])))
        Hp = float(math.exp(float(aug_inp.geom[2])))
        # 是否真的被重投影：geom 以 float32 存 log(尺寸)，exp 反解往返误差约 1e-5，
        # 故用**相对**容差判断，否则横屏帧会被误判成"已随机化"。
        w_real, h_real = sw * cpx, sh * cpy
        rel = max(abs(Wp - w_real) / max(w_real, 1e-9),
                  abs(Hp - h_real) / max(h_real, 1e-9))
        randomized = rel > 1e-3
        au, av = float(aug_lab[0]), float(aug_lab[1])
        exp_v = 0.5 + (gyn - 0.5) * (sh * cpy) / Hp
        cand = [0.5 + s * (gxn - 0.5) * (sw * cpx) / Wp for s in (+1.0, -1.0)]
        hit = min(range(2), key=lambda i: abs(cand[i] - au))
        log_lines.append("[9] 增强（水平翻转 + 光度扰动 + 裁剪抖动 + 几何域随机化）")
        log_lines.append(f"      增强后合成屏 W'={Wp:.4f}cm H'={Hp:.4f}cm "
                         f"(宽高比 {Wp / Hp:.4f})"
                         + ("  ← 与真实屏相同，本次未随机化" if rel <= 1e-3
                            else "  ← 已重投影到合成几何"))
        log_lines.append(f"      判定为{'已' if hit == 1 else '未'}水平翻转；"
                         f"增强后 label = [{au:.6f}, {av:.6f}]")
        log_lines.append(f"      期望 = [{cand[hit]:.6f}, {exp_v:.6f}]")
        log_lines.append("      注意：增强含随机裁剪抖动，故 E 面板的裁剪框与 B/C 面板不同")
        check("增强后 label_x == 翻转×几何重投影的期望值", abs(cand[hit] - au) < 1e-5,
              f"期望 {cand[hit]:.6f} 实得 {au:.6f}")
        check("增强后 label_y == 几何重投影的期望值", abs(exp_v - av) < 1e-5,
              f"期望 {exp_v:.6f} 实得 {av:.6f}")
        if randomized:
            check("几何随机化：合成屏尺寸在配置范围内",
                  GEOM_RAND_WIDTH_CM[0] - 1e-6 <= Wp <= GEOM_RAND_WIDTH_CM[1] + 1e-6
                  and GEOM_RAND_HEIGHT_CM[0] - 1e-6 <= Hp <= GEOM_RAND_HEIGHT_CM[1] + 1e-6
                  and GEOM_RAND_ASPECT[0] - 1e-6 <= Wp / Hp <= GEOM_RAND_ASPECT[1] + 1e-6,
                  f"W'={Wp:.3f} H'={Hp:.3f} a={Wp / Hp:.3f}")
        else:
            check("未随机化：geom == 真实屏幕尺寸", rel <= 1e-3,
                  f"W'={Wp:.3f} H'={Hp:.3f} vs 真实 {w_real:.3f}x{h_real:.3f}")
        num["aug_flipped"] = int(hit == 1)
        num["aug_label_x"] = au
        num["aug_Wp"] = Wp
        num["aug_Hp"] = Hp

    # ---------- 组装面板图 ----------
    panel = _make_panel(img_bgr, (x0, y0, x1, y1), (cx_, cy_), (bx, by, bw, bh),
                        crop_pil, back, aug_arr, aug_label, gxn, gyn, sw, sh,
                        cpx, cpy,
                        f"#{idx} {subject} {dev} ori{row['orientation']} "
                        f"split={row['split']}  frame={Path(row['frame_path']).name}",
                        ori=int(row["orientation"]),
                        screen_cm=(sw * cpx, sh * cpy))
    return panel, num, checks, log_lines


def _read_bgr(p: Path):
    import cv2
    img = cv2.imread(str(p))
    if img is None:
        raise FileNotFoundError(f"读不到图像：{p}")
    return img


def _crop_only(img_bgr, rect, size):
    import cv2
    x0, y0, x1, y1 = rect
    return cv2.resize(img_bgr[y0:y1, x0:x1], (size, size))


def _screen_panel(size_w, size_h, sw, sh, dots, title_lines, text_w=200):
    """画一块"设备屏幕"示意图：左侧文字列 + 右侧按 sw:sh(点) 比例的屏幕矩形。

    坐标系与数据集标签完全一致：原点 = 屏幕左上角，x 向右、y 向下，
    归一化后落在 [0,1]。红点位置可直接与 label 对照。
    dots: [(gxn, gyn, color, filled), ...]，多个点可叠加（如"原 vs 增强后"）。
    文字列与屏幕矩形在空间上完全分开，互不遮挡。
    """
    im = Image.new("RGB", (size_w, size_h), (22, 22, 26))
    dr = ImageDraw.Draw(im)

    # ---- 左文字列 ----
    ty = 6
    for txt, col in title_lines:
        if txt:
            draw_text(dr, (6, ty), txt, col)
        ty += 16

    # ---- 屏幕矩形区域（右侧，留出左侧 y 刻度与下方 x 刻度） ----
    ytick_w = 26
    left = text_w + 6 + ytick_w
    right = size_w - 6
    top = 8
    bot = size_h - 22
    avail_w, avail_h = max(20, right - left), max(20, bot - top)

    s = min(avail_w / sw, avail_h / sh)
    rw, rh = max(24, int(sw * s)), max(24, int(sh * s))
    ox = left + (avail_w - rw) // 2
    oy = top + (avail_h - rh) // 2

    dr.rectangle([ox, oy, ox + rw, oy + rh], fill=(40, 43, 52),
                 outline=(165, 168, 178), width=2)
    for f in (0.25, 0.5, 0.75):
        dr.line([ox + rw * f, oy, ox + rw * f, oy + rh], fill=(72, 75, 88))
        dr.line([ox, oy + rh * f, ox + rw, oy + rh * f], fill=(72, 75, 88))
    for f, lab in ((0.0, "0.0"), (0.5, "0.5"), (1.0, "1.0")):
        x = ox + rw * f
        draw_text(dr, (int(x) - (16 if f == 1.0 else 9), oy + rh + 3), lab, (180, 185, 200))
        y = oy + rh * f
        draw_text(dr, (ox - 25, int(y) - 7), lab, (180, 185, 200))

    # ---- 注视点（先辅助线，后画点，保证点不被遮住） ----
    for gxn, gyn, color, filled in dots:
        px, py = ox + rw * gxn, oy + rh * gyn
        for k in range(0, max(1, int(px) - ox), 7):
            dr.line([ox + k, py, min(ox + k + 4, px - 4), py], fill=color)
        for k in range(0, max(1, int(py) - oy), 7):
            dr.line([px, oy + k, px, min(oy + k + 4, py - 4)], fill=color)
    for gxn, gyn, color, filled in dots:
        px, py = ox + rw * gxn, oy + rh * gyn
        if filled:
            dr.ellipse([px - 7, py - 7, px + 7, py + 7], fill=color,
                       outline=(255, 255, 255), width=2)
            dr.line([px - 12, py, px + 12, py], fill=(255, 255, 255))
            dr.line([px, py - 12, px, py + 12], fill=(255, 255, 255))
        else:
            dr.ellipse([px - 7, py - 7, px + 7, py + 7], outline=color, width=3)
    return im


def _make_panel(img_bgr, crop_rect, face_center, bbox, crop_pil, back_arr,
                aug_arr, aug_label, gxn, gyn, sw, sh, cm_cx, cm_cy, title,
                ori=None, screen_cm=None):
    """A 原帧+标注 | B 裁剪 | C 反归一化 | D 屏幕示意 | [E 增强后 + 增强后注视点]"""
    PH = 340                       # 面板高度
    panels = []

    # A: 原始整帧 + bbox(红) + 裁剪框(绿) + 中心(黄)
    a = to_pil_uint8(img_bgr).convert("RGB")
    sc = PH / a.height
    a = a.resize((max(1, int(a.width * sc)), PH))
    dr = ImageDraw.Draw(a)
    k = sc
    x0, y0, x1, y1 = [v * k for v in crop_rect]
    bx, by, bw, bh = [v * k for v in bbox]
    dr.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=2)
    dr.rectangle([bx, by, bx + bw, by + bh], outline=(255, 60, 60), width=2)
    dr.ellipse([face_center[0] * k - 3, face_center[1] * k - 3,
                face_center[0] * k + 3, face_center[1] * k + 3], fill=(255, 255, 0))
    draw_text(dr, (4, 4), "A 原帧: 红=bbox 绿=裁剪框 黄=bbox中心", (255, 255, 0))
    panels.append(a)

    # B/C: 裁剪图 与 反归一化图
    for im, tag in ((crop_pil, "B 裁剪后 resize"), (Image.fromarray(back_arr), "C 反归一化(模型输入)")):
        im = im.convert("RGB").resize((PH, PH))
        draw_text(ImageDraw.Draw(im), (4, 4), tag, (0, 255, 0))
        panels.append(im)

    # D: 屏幕示意图（左文字列 + 右屏幕矩形，互不遮挡）
    d_lines = [
        ("D 设备屏幕", (255, 255, 0)),
        (f"{sw:.0f} x {sh:.0f} 点", (255, 255, 0)),
        ("", None),
        (f"物理 {screen_cm[0]:.2f} x {screen_cm[1]:.2f} cm" if screen_cm else "", (150, 200, 255)),
        ("原点=屏幕左上角", (150, 200, 255)),
        ("x 向右, y 向下", (150, 200, 255)),
        ("", None),
        (f"Orientation={ori}", (180, 185, 200)),
        ("", None),
        ("注视点 归一化", (255, 200, 120)),
        (f"  ({gxn:.4f}, {gyn:.4f})", (255, 200, 120)),
        ("注视点 像素", (255, 200, 120)),
        (f"  ({gxn * sw:.1f}, {gyn * sh:.1f})", (255, 200, 120)),
    ]
    panels.append(_screen_panel(470, PH, sw, sh,
                                [(gxn, gyn, (255, 90, 90), True)],
                                d_lines, text_w=200))

    # E: 增强后输入 + 增强后注视点（叠加"原始位置"空心圈，直观显示翻转镜像）
    if aug_arr is not None:
        im = Image.fromarray(aug_arr).convert("RGB").resize((PH, PH))
        draw_text(ImageDraw.Draw(im), (4, 4), "E 增强后输入", (0, 255, 0))
        panels.append(im)
        agx, agy = float(aug_label[0]), float(aug_label[1])
        e_lines = [
            ("E 增强后注视点", (0, 255, 0)),
            ("", None),
            ("实心 = 增强后", (120, 255, 120)),
            ("空心 = 原始", (255, 160, 160)),
            ("(翻转会镜像 x -> 1-x)", (150, 200, 255)),
            ("", None),
            (f"归一化 ({agx:.4f},", (255, 200, 120)),
            (f"        {agy:.4f})", (255, 200, 120)),
            (f"像素 ({agx * sw:.1f},", (255, 200, 120)),
            (f"      {agy * sh:.1f})", (255, 200, 120)),
        ]
        panels.append(_screen_panel(340, PH, sw, sh,
                                    [(agx, agy, (90, 235, 90), True),
                                     (gxn, gyn, (255, 120, 120), False)],
                                    e_lines, text_w=150))

    W = sum(p.width for p in panels) + 8 * (len(panels) + 1)
    canvas = Image.new("RGB", (W, PH + 46), (15, 15, 15))
    x = 8
    for p in panels:
        canvas.paste(p, (x, 32))
        x += p.width + 8
    draw_text(ImageDraw.Draw(canvas), (8, 8), title, (255, 255, 255))
    return canvas


# ==========================================================================
# 被过滤帧的留痕
# ==========================================================================
def dump_dropped(raw_root: Path, out_dir: Path, log_lines, max_per_reason=4):
    """复现预处理过滤逻辑，把被丢弃的帧连图带值导出，供核对过滤是否正确。"""
    import cv2
    ddir = out_dir / "dropped"
    ddir.mkdir(parents=True, exist_ok=True)
    buckets = defaultdict(list)
    subs = sorted([p for p in raw_root.iterdir()
                   if p.is_dir() and re.fullmatch(r"\d+", p.name)])
    for sub in subs:
        try:
            face = read_json(sub / "appleFace.json")
            el = read_json(sub / "appleLeftEye.json")
            er = read_json(sub / "appleRightEye.json")
            grid = read_json(sub / "faceGrid.json")
            fr = read_json(sub / "frames.json")
            dot = read_json(sub / "dotInfo.json")
        except OSError:
            continue
        n = min(len(fr), len(face["IsValid"]), len(el["IsValid"]),
                len(er["IsValid"]), len(grid["IsValid"]))
        for j in range(n):
            reason = None
            if not (face["IsValid"][j] and el["IsValid"][j] and er["IsValid"][j]
                    and grid["IsValid"][j]):
                which = [nm for nm, arr in (("face", face), ("leftEye", el),
                                            ("rightEye", er), ("faceGrid", grid))
                         if not arr["IsValid"][j]]
                reason = "IsValid=0:" + "+".join(which)
            elif face["W"][j] <= 1 or face["H"][j] <= 1:
                reason = "degenerate_bbox"
            elif not (sub / "frames" / fr[j]).is_file():
                reason = "missing_frame"
            if reason and len(buckets[reason]) < max_per_reason:
                buckets[reason].append((sub, j, fr[j], face, dot))
    log_lines.append("")
    log_lines.append("========== 被过滤的帧（抽样留痕） ==========")
    n_written = 0
    for reason, items in sorted(buckets.items()):
        for (sub, j, fname, face, dot) in items:
            img = cv2.imread(str(sub / "frames" / fname))
            if img is None:
                continue
            x0 = max(0, int(face["X"][j]))
            y0 = max(0, int(face["Y"][j]))
            x1 = min(img.shape[1], int(face["X"][j] + face["W"][j]))
            y1 = min(img.shape[0], int(face["Y"][j] + face["H"][j]))
            a = to_pil_uint8(img).convert("RGB")
            sc = 340 / a.height
            a = a.resize((max(1, int(a.width * sc)), 340))
            dr = ImageDraw.Draw(a)
            if x1 > x0 and y1 > y0:
                dr.rectangle([x0 * sc, y0 * sc, x1 * sc, y1 * sc], outline=(255, 60, 60), width=2)
            draw_text(dr, (4, 4), f"DROPPED {sub}/{fname}", (255, 60, 60))
            draw_text(dr, (4, 22), reason[:60], (255, 200, 120))
            draw_text(dr, (4, 40), f"bbox=({face['X'][j]:.1f},{face['Y'][j]:.1f},"
                                   f"{face['W'][j]:.1f},{face['H'][j]:.1f})", (255, 255, 0))
            name = f"{sub.name}_{Path(fname).stem}_{reason.split(':')[0]}.png"
            a.save(ddir / name)
            n_written += 1
            log_lines.append(f"  {sub}/{fname}: 原因={reason}  "
                             f"bbox=({face['X'][j]:.2f},{face['Y'][j]:.2f},"
                             f"{face['W'][j]:.2f},{face['H'][j]:.2f})  "
                             f"dot=({dot['XPts'][j]:.2f},{dot['YPts'][j]:.2f})")
    log_lines.append(f"  共导出 {n_written} 张被过滤帧图 -> {ddir}")
    return n_written


# ==========================================================================
# 抽样：按 (设备, orientation) 分层，保证覆盖全部情况
# ==========================================================================
def stratified_pick(rows, k):
    """分层抽样：先保证每个设备、再每个 (设备, 朝向) 至少 1 个，然后轮转补齐。

    为什么不能只按 (设备, 朝向, split) 分组后轮转：键数会达到几十个，而
    sorted() 把 'iPad*' 排在 'iPhone*' 前面，导致样本量不大时**全部落在 iPad**
    （实测 60 样本 60 个都是 iPad），验证覆盖不到占比最大的 iPhone。
    """
    by_dev, by_dev_ori = defaultdict(list), defaultdict(list)
    for i, r in enumerate(rows):
        by_dev[r["device"]].append(i)
        by_dev_ori[(r["device"], r["orientation"])].append(i)
    picked, seen = [], set()

    def take(i):
        if i not in seen:
            seen.add(i)
            picked.append(i)

    for dev in sorted(by_dev):                       # 轮 1：每个设备 1 个
        take(by_dev[dev][0])
    for key in sorted(by_dev_ori):                   # 轮 2：每个 (设备, 朝向) 1 个
        if len(picked) >= k:
            break
        take(by_dev_ori[key][0])
    keys = sorted(by_dev_ori)
    r = 1
    while len(picked) < k:                           # 轮 3：轮转补齐
        added = False
        for key in keys:
            if len(by_dev_ori[key]) > r and len(picked) < k:
                take(by_dev_ori[key][r])
                added = True
        if not added:
            break
        r += 1
    return picked


def main():
    ap = argparse.ArgumentParser(description="GazeCapture 预处理/Dataset 全操作留痕核对")
    ap.add_argument("--index-csv", default=str(GAZECAPTURE_INDEX_DIR / "index.csv"))
    ap.add_argument("--raw-root", default=str(GAZECAPTURE_EXTRACT_DIR))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "out" / "gazecapture" / "dataset_check"))
    ap.add_argument("--num-samples", type=int, default=24)
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--augment", action="store_true", help="额外导出增强前后对比")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    log_lines = []
    log_lines.append("=" * 78)
    log_lines.append("GazeCapture 预处理 / Dataset 全操作留痕核对")
    log_lines.append(f"时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log_lines.append(f"索引: {args.index_csv}")
    log_lines.append(f"原始: {args.raw_root}")
    log_lines.append(f"输出: {out_dir}")
    log_lines.append("=" * 78)

    device_spec = load_device_spec(lambda m: log_lines.append("  " + m))
    # 数值核对必须用确定性数据集（augment=False => 几何抖动强制为 0），
    # 否则裁剪抖动/翻转随机化，报告不可复现、无法与人工比对。
    ds = GazeCaptureDataset(index_csv=args.index_csv, augment=False,
                            image_size=args.image_size)
    ds_aug = None
    if args.augment:
        ds_aug = GazeCaptureDataset(index_csv=args.index_csv, augment=True,
                                    image_size=args.image_size)
    log_lines.append(f"Dataset(确定性): {len(ds)} 样本, image_size={ds.image_size}, "
                     f"aug_shift={ds.aug_shift}, crop_margin={CROP_MARGIN}")
    if ds_aug is not None:
        log_lines.append(f"Dataset(增强, 仅用于 E 面板): shift={ds_aug.aug_shift} "
                         f"scale={ds_aug.aug_scale} rotate={ds_aug.aug_rotate}")

    picks = stratified_pick(ds.rows, args.num_samples)
    log_lines.append(f"分层抽样 {len(picks)} 个样本（按 设备 x orientation x split）")
    log_lines.append("")

    cache = SubjectCache(Path(args.raw_root))
    all_checks = []
    csv_rows = []
    sheets = []

    for n, idx in enumerate(picks, 1):
        panel, num, checks, lines = trace_sample(
            ds, idx, cache, device_spec, log_lines, ds_aug=ds_aug)
        panel.save(img_dir / f"{n:03d}_idx{idx}.png")
        sheets.append(panel)
        for name, ok, detail in checks:
            all_checks.append((idx, name, ok, detail))
        num["index"] = idx
        csv_rows.append(num)
        lines.append("")

    # 丢弃帧留痕
    n_dropped = dump_dropped(Path(args.raw_root), out_dir, log_lines)

    # ---- report.csv ----
    keys = []
    for r in csv_rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out_dir / "report.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)

    # ---- contact sheet ----
    if sheets:
        cw = max(s.width for s in sheets)
        ch = sum(s.height + 6 for s in sheets)
        sheet = Image.new("RGB", (cw, ch), (10, 10, 10))
        y = 0
        for s in sheets:
            sheet.paste(s, (0, y))
            y += s.height + 6
        sheet.save(img_dir / "contact_sheet.png")

    # ---- summary：自动校验 ----
    by_name = defaultdict(lambda: [0, 0])
    fails = []
    for idx, name, ok, detail in all_checks:
        by_name[name][0] += 1
        by_name[name][1] += int(ok)
        if not ok:
            fails.append((idx, name, detail))
    log_lines.append("")
    log_lines.append("=" * 78)
    log_lines.append("自动一致性校验汇总")
    log_lines.append("=" * 78)
    for name in sorted(by_name):
        tot, ok = by_name[name]
        log_lines.append(f"  [{'PASS' if ok == tot else 'FAIL'}] {name}: {ok}/{tot}")
    log_lines.append("")
    if fails:
        log_lines.append(f"失败明细（{len(fails)} 条）:")
        for idx, name, detail in fails[:40]:
            log_lines.append(f"  #{idx} {name}  {detail}")
    else:
        log_lines.append("全部校验通过。")
    log_lines.append("")
    log_lines.append(f"被过滤帧导出 {n_dropped} 张 -> {out_dir / 'dropped'}")

    (out_dir / "report.txt").write_text("\n".join(log_lines), encoding="utf-8")
    (out_dir / "summary.txt").write_text(
        "\n".join(log_lines[log_lines.index("=" * 78) + 1:]), encoding="utf-8")

    print(f"报告: {out_dir / 'report.txt'}")
    print(f"数值表: {out_dir / 'report.csv'}")
    print(f"图像: {img_dir}")
    print(f"校验: {sum(1 for _,_,ok,_ in all_checks if ok)}/{len(all_checks)} PASS")
    if fails:
        print(f"失败 {len(fails)} 条，详见 summary.txt")


if __name__ == "__main__":
    main()
