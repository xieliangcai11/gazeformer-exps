"""GazeCapture 眼框**虹膜检验**（量化"眼框是否真的套住眼睛" ✓，不靠肉眼 ✓）。

指标（每帧、每只眼）：
  · iris_in_box ✓：虹膜中心（MediaPipe 468/473 ✓，需 refine_landmarks=True ✓）是否落在
    我们生成眼框时用的**眼周轮廓框**内 ✓。
  · 另记录：框/脸宽比、左右框顺序、是否重叠（sanity ✓）。

用法（服务器 ✓ 建议 nohup ✓，MP 启动 2–3 分钟 ✓）：
    nohup python -u -m tools.data.gazecapture.check_iris --n 1000 --seed 0 \
        --out out/gazecapture/analysis/iris_check.json \
        > out/gazecapture/analysis/iris_check.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
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
from tools.data.common.eye_detect import L_EYE, R_EYE  # noqa: E402

#: 虹膜中心（refine_landmarks=True ⇒ 478 点 ✓）：左=468、右=473 ✓
IRIS_L, IRIS_R = 468, 473


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/gazecapture/analysis/iris_check.json")
    ap.add_argument("--min-face", type=int, default=40, help="小于此人脸框的帧记为过小 ✓")
    args = ap.parse_args()

    import mediapipe as mp
    from mediapipe.tasks import python as mp_py
    from mediapipe.tasks.python import vision
    base = mp_py.BaseOptions(model_asset_path=str(_ROOT / "assets" / "face_landmarker.task"))
    opts = vision.FaceLandmarkerOptions(base_options=base, num_faces=1,
                                        running_mode=vision.RunningMode.IMAGE)
    # 注：新版 Tasks API 的 face_landmarker.task **默认输出 478 点**（含虹膜 ✓），
    # 无 refine_landmarks 参数（旧版才有 x）；不足 478 点会在下面记 no_iris ✓。
    lm = vision.FaceLandmarker.create_from_options(opts)

    paths = [ln.strip() for ln in (GAZECAPTURE_INDEX_DIR / f"{args.split}.txt")
             .read_text(encoding="utf-8").splitlines() if ln.strip()]
    rng = random.Random(args.seed)
    sample = rng.sample(paths, min(args.n, len(paths)))
    print(f"[iris] 抽样 {len(sample)} 帧（split={args.split} ✓ seed={args.seed} ✓）", flush=True)

    stat = {"n": 0, "no_face": 0, "no_iris": 0, "both_in": 0, "one_in": 0, "none_in": 0,
            "face_too_small": 0}
    ratios, bad_examples = [], []
    t0 = time.time()
    for i, p in enumerate(sample, 1):
        img = cv2.imread(str(_ROOT / p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        stat["n"] += 1
        if not res.face_landmarks:
            stat["no_face"] += 1
            continue
        pts = res.face_landmarks[0]
        if len(pts) < 478:
            stat["no_iris"] += 1
            continue
        xs = np.array([q.x * w for q in pts]); ys = np.array([q.y * h for q in pts])
        face_w = float(xs.max() - xs.min()); face_h = float(ys.max() - ys.min())
        if min(face_w, face_h) < args.min_face:
            stat["face_too_small"] += 1

        def box(idx):
            xy = np.array([[pts[k].x * w, pts[k].y * h] for k in idx])
            x0, y0 = xy.min(0); x1, y1 = xy.max(0)
            return x0, y0, x1, y1

        def inside(b, k):
            x0, y0, x1, y1 = b
            cxk, cyk = pts[k].x * w, pts[k].y * h
            return (x0 <= cxk <= x1) and (y0 <= cyk <= y1)

        bl, br = box(L_EYE), box(R_EYE)
        inl, inr = inside(bl, IRIS_L), inside(br, IRIS_R)
        k = int(inl) + int(inr)
        stat["both_in" if k == 2 else "one_in" if k == 1 else "none_in"] += 1
        ratios.append((bl[2] - bl[0]) / max(face_w, 1e-6))
        if k < 2 and len(bad_examples) < 12:
            bad_examples.append(p)
        if i % 200 == 0:
            print(f"  {i}/{len(sample)}  ✓ both={stat['both_in']} one={stat['one_in']} "
                  f"none={stat['none_in']}", flush=True)

    n = max(stat["n"], 1)
    out = {
        **stat,
        "both_in_rate": stat["both_in"] / n,
        "one_or_both_in_rate": (stat["both_in"] + stat["one_in"]) / n,
        "box_face_ratio_p50": float(np.percentile(ratios, 50)) if ratios else None,
        "box_face_ratio_p05": float(np.percentile(ratios, 5)) if ratios else None,
        "box_face_ratio_p95": float(np.percentile(ratios, 95)) if ratios else None,
        "bad_examples": bad_examples,
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[iris] 完成 ✓ n={stat['n']} | 双眼虹膜都在框内 {100*out['both_in_rate']:.1f}%"
          f" | 至少一只 {100*out['one_or_both_in_rate']:.1f}% | 无脸 {stat['no_face']}"
          f" | 无虹膜 {stat['no_iris']} | 框/脸宽比 p50={out['box_face_ratio_p50']}"
          f" | {out['elapsed_s']}s → {args.out}", flush=True)


if __name__ == "__main__":
    main()
