# Gazelab

Gaze estimation research repo: **2D on-screen gaze regression** with a **DINOv2 backbone + screen-geometry token**.

Two usage lines sharing one training loop, one model family and one evaluation protocol:

| Dataset | Purpose | Scale |
|---|---|---|
| **GazeCapture** | large-scale gaze-prior pretraining | ~1.49M frames / 1474 subjects |
| **RGBDGaze** | in-domain training, calibration, TTA | ~140k frames / 45 subjects |

## Layout

```
configs/                 # zero-side-effect configs (paths, dataset hyper-params)
src/gazelab/
  datasets/              # gazecapture.py, rgbdgaze.py, common.py, table_index.py
  models/dinov2_gaze/    # dataset-agnostic model family (backbone + calibrator + TTA + VPT)
  models/transformers.py # BlockMoba / RMSNorm
scripts/
  train.py               # unified entry: --dataset rgbdgaze|gazecapture
  cross_eval.py          # cross-dataset / cross-geometry evaluation
  analysis/              # one-off analysis scripts
  rgbdgaze/              # calibration / TTA / bad-case mining
  gazecapture/           # training queue scripts
tools/data/{common,rgbdgaze,gazecapture}/   # preprocessing, eye boxes, validation
assets/ docs/ tests/ experiments/
data/ out/ model/ temp/  # artifacts (gitignored)
```

## Setup

```bash
conda create -n dl python=3.11 -y && conda activate dl
pip install -r requirements.txt
pip install mediapipe          # optional: eye-box generation / QC
```

DINOv2 weights go to `model/dinov2_vits14_pretrain.pth` (not committed).

## Data

```bash
python -m tools.data.gazecapture.preprocess
python -m tools.data.rgbdgaze.preprocess --split subject

# eye boxes (only needed for the eye stream)
python -m tools.data.gazecapture.build_eye_boxes_mp --shard 0 --nshards 8
python -m tools.data.gazecapture.build_eye_boxes_mp --merge \
    data/GazeCapture/index/eye_boxes_shard*.npz --out data/GazeCapture/index/eye_boxes.npz
python -m tools.data.gazecapture.check_shards --nshards 8
python -m tools.data.gazecapture.fix_eye_boxes --dry-run     # degenerate-box stats
python -m tools.data.gazecapture.fix_eye_boxes               # subject-median fallback

# validation
python -m tools.data.gazecapture.verify
```

A partial local dataset is tolerated by default (missing lists/frames are `WARN`);
use `--strict-data` for full-dataset checks.

## Training

```bash
python -m scripts.train --dataset rgbdgaze --resolution 448 --unfreeze 12 --diff-lr \
    --lr 1e-4 --batch-size 24 --epochs 8 --tag g6_nodepth

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 torchrun --standalone --nproc_per_node=6 \
  -m scripts.train --dataset gazecapture --resolution 224 --no-geom --geom-randomize 0 \
  --augment --aug-shift 0.05 --aug-scale 0.05 --aug-rotate 0 --unfreeze 12 --diff-lr \
  --lr 2e-4 --batch-size 32 --num-workers 8 --epochs 5 --seed 0 --tag gc_run1
```

Outputs: `out/<dataset>/ablation/checkpoints/<tag>/{best.pt,train_args.json}` + `logs/`.

## Conventions

- Labels are screen pixels, normalized to `[0,1]` inside datasets; eye boxes are `(x,y,w,h)` in raw-image coordinates.
- Geometry token = `[short/long, log(short_cm), log(long_cm)]` (orientation-independent); geometry randomization reprojects label **and** `screen_cm` together, with sizes derived from label feasibility (~100% realized rate).
- Augmentation is deterministic per `(aug_seed, epoch, idx)`; workers are rebuilt each epoch so `set_epoch` takes effect.
- Degenerate eye boxes (<8px, ~1.8%) are replaced by the subject-median box and flagged `filled=True`; mirroring is never used.

## Notes

- `--loss mse_angle` adds an angular term measured in degrees (MSE is ~1e-4); set `--angle-weight 0.01~0.05` if you want comparable magnitudes.
- Scripts under `scripts/analysis/` are one-off experiment code without a stable interface.

## License

MIT
