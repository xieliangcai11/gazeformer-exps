"""生成 RGBDGaze 的**按被试分组** 5 折交叉验证目录。

每折目录（默认 data/RGBDGaze/index/cv/fold{k}/）：
    index.csv        指向原索引（symlink，失败则复制）
    eye_boxes.npz    同上（训练眼支流需要）
    train.txt        其余 3 折的帧
    val.txt          下一折的帧（用于选模/早停）
    test.txt         本折留出被试的帧（**训练时不可见**）

用法：
    python -m scripts.analysis.make_group_cv --folds 5 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
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
from configs.rgbdgaze_config import RGBDGaze_INDEX_DIR  # noqa: E402


def _link(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        if src.suffix == ".csv":
            shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default=str(RGBDGaze_INDEX_DIR))
    ap.add_argument("--out", default=None, help="默认 <index-dir>/cv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--source-split", default="train",
                    help="只在该官方划分的被试内做折（默认 train；val/test 绝不参与源域）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    out = Path(args.out) if args.out else index_dir / "cv"
    with open(index_dir / "index.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    path_subj = {r["rgb_path"]: r["subject"] for r in rows}
    allow = {path_subj[p.strip()] for p in
             (index_dir / f"{args.source_split}.txt").read_text(encoding="utf-8").splitlines()
             if p.strip()}
    paths_by_subject = defaultdict(list)
    for r in rows:
        if r["subject"] in allow:
            paths_by_subject[r["subject"]].append(r["rgb_path"])
    print(f"[cv] 源域被试 {len(paths_by_subject)}（仅 {args.source_split}.txt）")

    subjects = sorted(paths_by_subject)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(subjects)
    groups = np.array_split(np.array(subjects, dtype=object), args.folds)
    print(f"[cv] {len(subjects)} 被试 -> {args.folds} 折: "
          + ", ".join(f"{len(g)}" for g in groups))

    for k in range(args.folds):
        d = out / f"fold{k}"
        d.mkdir(parents=True, exist_ok=True)
        _link(index_dir / "index.csv", d / "index.csv")
        if (index_dir / "eye_boxes.npz").is_file():
            _link(index_dir / "eye_boxes.npz", d / "eye_boxes.npz")
        test_subj = list(groups[k])
        # train = 其余全部折（最大化数据量，向部署模型靠拢）；val 留空（末轮权重生效）
        train_subj = [s for j, g in enumerate(groups) if j != k for s in g]
        val_subj = []
        for name, subs in (("train", train_subj), ("val", val_subj), ("test", test_subj)):
            lines = [p for s in subs for p in paths_by_subject[s]]
            (d / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"  fold{k} {name}: {len(subs)} 被试 / {len(lines)} 帧")
    print(f"[cv] 输出目录: {out}")


if __name__ == "__main__":
    main()
