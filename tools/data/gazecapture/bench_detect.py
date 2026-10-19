"""GazeCapture 眼框检测**性能诊断**（定位 0.44 秒/帧的根因 ✓）。

测量（100 帧）：
  · t_read    : cv2.imread（磁盘 + JPEG 解码 ✓）
  · t_detect  : MediaPipe FaceLandmarker.detect（纯推理 ✓）
另记录：图像尺寸、CPU 核数、load average、OpenCV 线程数、数据盘信息 ✓。

用法：
    nohup python -u -m tools.data.gazecapture.bench_detect --n 100 \
        > out/gazecapture/analysis/bench.log 2>&1 &
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import cv2
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[bench] 环境：cpu={os.cpu_count()} load={os.getloadavg() if hasattr(os, 'getloadavg') else 'n/a'}"
          f" cv2_threads={cv2.getNumThreads()}", flush=True)
    try:
        st = os.statvfs(str(_ROOT / "data"))
        print(f"[bench] 数据盘: block={st.f_bsize} 总={st.f_blocks*st.f_bsize/1e9:.0f}GB"
              f" 可用={st.f_bavail*st.f_bsize/1e9:.0f}GB", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[bench] statvfs 失败: {e}", flush=True)

    paths = [ln.strip() for ln in (GAZECAPTURE_INDEX_DIR / f"{args.split}.txt")
             .read_text(encoding="utf-8").splitlines() if ln.strip()]
    rng = random.Random(args.seed)
    sample = rng.sample(paths, min(args.n, len(paths)))

    import mediapipe as mp
    from mediapipe.tasks import python as mp_py
    from mediapipe.tasks.python import vision
    t_imp = time.time()
    base = mp_py.BaseOptions(model_asset_path=str(_ROOT / "assets" / "face_landmarker.task"))
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=base, num_faces=1, running_mode=vision.RunningMode.IMAGE))
    print(f"[bench] landmarker 就绪（含导入）耗时 {time.time()-t_imp:.1f}s", flush=True)

    t_read, t_det, shapes, misses = [], [], [], 0
    first = True
    for p in sample:
        _t = time.time()
        img = cv2.imread(str(_ROOT / p), cv2.IMREAD_COLOR)
        t_read.append(time.time() - _t)
        if img is None:
            continue
        shapes.append(img.shape[:2])
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _t = time.time()
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        t_det.append(time.time() - _t)
        if not res.face_landmarks:
            misses += 1
        if first:
            first = False
            print(f"[bench] 首帧尺寸 {img.shape} | 首次读 {t_read[-1]*1e3:.1f}ms", flush=True)

    def q(a, p):
        return float(np.percentile(a, p)) if a else float("nan")

    print(f"[bench] n={len(t_read)} 读图: mean={np.mean(t_read)*1e3:.1f}ms p50={q(t_read,50)*1e3:.1f}"
          f" p95={q(t_read,95)*1e3:.1f}", flush=True)
    print(f"[bench] n={len(t_det)} 检测: mean={np.mean(t_det)*1e3:.1f}ms p50={q(t_det,50)*1e3:.1f}"
          f" p95={q(t_det,95)*1e3:.1f}", flush=True)
    print(f"[bench] 图像尺寸(min/max): {min(shapes) if shapes else '-'} / {max(shapes) if shapes else '-'}"
          f" | 未检到脸 {misses}/{len(t_det)}", flush=True)
    print(f"[bench] 结论: 读图占比 {100*np.sum(t_read)/(np.sum(t_read)+np.sum(t_det)):.0f}%"
          f" | 检测占比 {100*np.sum(t_det)/(np.sum(t_read)+np.sum(t_det)):.0f}%", flush=True)


if __name__ == "__main__":
    main()
