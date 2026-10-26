"""逐折训练 + 导出**out-of-subject**预测（供源域 C3 学习/选参）。

流程（每折）：
  1. `scripts.train` 训练（与部署配方一致：224 / use-eye / no-geom / augment /
     unfreeze12 / diff-lr / lr 1e-4 / 5 epochs），产物在 out/rgbdgaze/ablation/checkpoints/cv_fold{k}
  2. `scripts.analysis.dump_predictions` 在该折 **test（留出被试）** 上推理
     -> out/rgbdgaze/analysis/context/cv_fold{k}.csv
  3. 已有输出则跳过（可反复重启）

用法（建议 tmux）：
    python -m scripts.analysis.run_group_cv --gpu 3 --folds 5 --epochs 5
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


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
from configs.rgbdgaze_config import RGBDGaze_INDEX_DIR  # noqa: E402


def run(cmd, env):
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env, cwd=str(_ROOT))
    if r.returncode != 0:
        raise SystemExit(f"命令失败（{r.returncode}）: {cmd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--cv-dir", default=str(RGBDGaze_INDEX_DIR / "cv"))
    ap.add_argument("--out-dir", default="out/rgbdgaze/analysis/context")
    ap.add_argument("--start-fold", type=int, default=0)
    ap.add_argument("--only-fold", type=int, default=-1, help="只跑指定折（并行时用）")
    args = ap.parse_args()

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    _folds = [args.only_fold] if args.only_fold >= 0 else list(range(args.start_fold, args.folds))
    for k in _folds:
        fold_dir = Path(args.cv_dir) / f"fold{k}"
        tag = f"cv_fold{k}"
        ckpt = _ROOT / "out" / "rgbdgaze" / "ablation" / "checkpoints" / tag / "best.pt"
        csv_out = out_dir / f"cv_fold{k}.csv"
        if not fold_dir.is_dir():
            raise SystemExit(f"缺少折目录 {fold_dir}（先跑 make_group_cv）")
        if csv_out.is_file():
            print(f"[skip] fold{k} 已有 {csv_out}", flush=True)
            continue
        if not ckpt.is_file():
            run([sys.executable, "-u", "-m", "scripts.train",
                 "--dataset", "rgbdgaze", "--index-dir", str(fold_dir),
                 "--resolution", "224", "--use-eye", "--no-geom",
                 "--augment", "--aug-shift", "0.05", "--aug-scale", "0.05",
                 "--aug-rotate", "0", "--geom-randomize", "0",
                 "--unfreeze", "12", "--diff-lr",
                 "--lr", str(args.lr), "--batch-size", str(args.batch_size),
                 "--num-workers", str(args.num_workers),
                 "--epochs", str(args.epochs), "--seed", "0",
                 "--resume", "off", "--tag", tag], env)
        run([sys.executable, "-u", "-m", "scripts.analysis.dump_predictions",
             "--ckpt", str(ckpt), "--dataset", "rgbdgaze",
             "--index-dir", str(fold_dir), "--split", "test",
             "--resolution", "224", "--use-eye", "--no-geom", "--orient", "raw",
             "--batch-size", "64", "--num-workers", "6",
             "--out", str(csv_out)], env)
        print(f"[done] fold{k} -> {csv_out}", flush=True)


if __name__ == "__main__":
    main()
