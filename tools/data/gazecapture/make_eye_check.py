"""GazeCapture 眼框检测**小样本检查**（随机抽帧 + 画框 + 拼图 + 逐帧进度 ✓）。

依赖：只依赖 cv2 ✓（`tools/data/common/eye_detect.py` ✓ 的共享工具）。
不读 149 万行的索引 ✗（那是服务器上最慢的一步 ✓）；人脸框用 Haar 直接画 ✓。

用法（服务器 ✓ 建议 nohup 后台 ✓，该机器前台 python 会假死 ✗）：
    nohup python -u -m tools.data.gazecapture.make_eye_check --n 12 --cols 4 --cell 220 \
        --out out/gazecapture/analysis/eye_check_haar.jpg > out/gazecapture/analysis/eyecheck.log 2>&1 &
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2


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
from tools.data.common.eye_detect import (detect, detect_mp, draw, load_cascades,  # noqa: E402
                                          make_landmarker, montage, to_tile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--cell", type=int, default=220)
    ap.add_argument("--backend", choices=["haar", "mp"], default="haar",
                    help="haar=零依赖但侧脸/倒置差 ✗；mp=MediaPipe（对倒置/侧脸更稳 ✓，需 mediapipe ✓）")
    ap.add_argument("--out", default="out/gazecapture/analysis/eye_check_haar.jpg")
    args = ap.parse_args()

    paths = [ln.strip() for ln in (GAZECAPTURE_INDEX_DIR / f"{args.split}.txt")
             .read_text(encoding="utf-8").splitlines() if ln.strip()]
    rng = random.Random(args.seed)
    sample = rng.sample(paths, min(args.n, len(paths)))
    # ⚠️ 只在 haar 后端才加载级联 ✓：OpenCV 5.x 已移除 `cv2.CascadeClassifier`（本机 cv2 5.0 ✗），
    # 而 mp 后端根本不需要它 ✓ ⇒ 否则 mp 会在本地直接报 AttributeError ✗
    if args.backend == "haar":
        face_cas, eye_cas = load_cascades(_ROOT)
    else:
        face_cas = eye_cas = None
    landmarker = make_landmarker(_ROOT) if args.backend == "mp" else None
    print(f"[eyecheck] 抽样 {len(sample)} 帧（split={args.split} ✓ seed={args.seed} ✓ backend={args.backend} ✓）", flush=True)

    tiles, ok, partial, noface = [], 0, 0, 0
    t_det = 0.0; t_io = 0.0; t0 = time.time()
    for i, p in enumerate(sample, 1):
        _t = time.time()
        img = cv2.imread(str(_ROOT / p), cv2.IMREAD_COLOR)
        t_io += time.time() - _t
        if img is None:
            print(f"  [{i}/{len(sample)}] 读图失败 ✗ {p}", flush=True)
            continue
        _t = time.time()
        if args.backend == "mp":
            face, le, re, is_partial = detect_mp(img, landmarker)
        else:
            face, le, re, is_partial = detect(img, face_cas, eye_cas)
        t_det += time.time() - _t
        if face is None:
            noface += 1
        elif is_partial:
            partial += 1
        else:
            ok += 1
        tiles.append(to_tile(draw(img, face, le, re), args.cell))
        print(f"  [{i}/{len(sample)}] 被试={Path(p).parents[1].name} face={'Y' if face else 'N'}"
              f" eyes={'partial' if is_partial else 'both'}  {Path(p).name}", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), montage(tiles, args.cols, args.cell))
    n = max(len(tiles), 1)
    print(f"[eyecheck] 完成 ✓ 共 {len(tiles)} 帧 | 双眼完整 {ok}（{100*ok/n:.0f}%）"
          f" | partial {partial}（{100*partial/n:.0f}%）| 未检到脸 {noface} | → {out}", flush=True)
    el = time.time() - t0
    print(f"[timing] 总 {el:.1f}s | 读图 {t_io:.1f}s | 检测 {t_det:.1f}s | "
          f"检测吞吐 {len(tiles)/max(t_det,1e-9):.1f} 帧/秒 ✓", flush=True)


if __name__ == "__main__":
    main()
