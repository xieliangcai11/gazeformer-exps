# Robust Sequential Calibration for 2D Screen Gaze

Label-efficient per-user calibration for RGB / RGB-D screen gaze estimation.
The calibrator is model-agnostic: it takes the base model's 2D screen predictions,
a few labelled calibration points, and optional per-frame pose features.

## Method

For each homogeneous context (phone orientation for GazeCapture, activity for
RGBDGaze) we fit a small affine correction

```
u = sx * x + bx,   v = sy * y + by
```

with a population prior learned from source users and a robust (Huber) fit:

1. **Coverage selection** — calibration frames are spread over a 3×3 grid of
   predicted screen positions.
2. **Ridge prior** — source users provide the prior mean; `lambda` controls its
   strength.
3. **Huber IRLS** — calibration points with large residuals (blinks, missed
   fixations) are down-weighted. This is the main source of gain and of
   worst-case robustness.
4. **Sequential update (optional)** — points can be processed one by one with a
   Bayesian posterior; fixed K and sequential updates are mathematically
   equivalent when no early stopping is used.

No target-domain labels are used outside the K calibration points. Evaluation
labels are used only for scoring.

## Results

Frame-weighted screen error (cm). Baselines: GazeCapture 1.612, RGBDGaze 1.465.
K is per context (per orientation / per activity); 5 calibration-seed average.

### GazeCapture

| K/orientation | B2 (ridge) | Robust sequential |
|---|---|---|
| 5  | −15.86% | **−17.68%** |
| 9  | −19.97% | **−22.10%** |
| 13 | −22.12% | **−23.62%** |
| 26 | −23.23% | **−25.26%** |

At K=9 (≈28 labels/session) error drops from **1.612 → 1.256 cm**.

### RGBDGaze

| K/activity | B2 | Robust sequential |
|---|---|---|
| 5  | −12.71% | **−13.37%** |
| 9  | −16.96% | **−18.19%** |
| 13 | −16.05% | **−17.44%** |
| 26 | −18.87% | **−19.59%** |

At K=9 error drops from **1.465 → 1.201 cm**.

Ablations: quadratic/RBF residual models did not beat affine; active D-optimal
selection was worse than coverage/random; device-family priors and context
means did not help; checkpoint ensembles did not improve the base model. See
`results/k_curve.csv`.

## Layout

```
src/gazelab/              datasets, DINOv2 gaze model, training utilities
configs/                  dataset paths and hyper-parameters
scripts/train.py          model training / evaluation
scripts/calibration/      final calibration method + baselines
  dump_feats.py           export frozen-model features and predictions
  pose_features.py        head-pose / eye-geometry proxy features
  seq_bayes_calib.py      robust sequential calibration (main)
  robust_calib.py         Huber-B2 baseline / comparison
  cc_eb.py                coverage selection + ridge prior (B2)
tools/data/               dataset preprocessing
assets/                   MediaPipe / OpenCV assets
tests/                    unit tests
```

## Install

```bash
pip install -e .
```

Optional: `pip install mediapipe` for eye-box/pose extraction.

## Quickstart

1. Preprocess a dataset with `tools/data/.../preprocess.py` (see `configs/`).
2. Train or obtain a base model with `scripts/train.py`.
3. Export features/predictions:

```bash
python -m scripts.calibration.dump_feats \
  --dataset gazecapture \
  --ckpt out/gazecapture/.../best.pt \
  --index-dir data/GazeCapture/index \
  --split test --resolution 224 --use-eye \
  --out out/gazecapture/analysis/feats_gc_test.npz
```

4. Export pose proxies:

```bash
python -m scripts.calibration.pose_features \
  --dataset gazecapture \
  --index-dir data/GazeCapture/index \
  --eye-boxes data/GazeCapture/index/eye_boxes.npz \
  --out out/gazecapture/analysis/pose_features.npz
```

5. Run robust sequential calibration (K=9 per orientation):

```bash
python -m scripts.calibration.seq_bayes_calib \
  --train-npz out/gazecapture/analysis/feats_gc_train.npz \
  --val-npz   out/gazecapture/analysis/feats_gc_val.npz \
  --test-npz  out/gazecapture/analysis/feats_gc_test.npz \
  --index-csv data/GazeCapture/index/index.csv \
  --pose-npz  out/gazecapture/analysis/pose_features.npz \
  --context-key orientation \
  --select-mode coverage_b2 --noise-mode const --prior-mode ridge \
  --prior-lam 1.0 --huber-delta 2.0 \
  --kmin 3 --kmax 9 --stop-std 0 \
  --out out/gazecapture/analysis/calibrated.csv
```

For RGBDGaze use `--context-key activity --prior-lam 0.1 --huber-delta 1.345`.

## Notes and limitations

- Dataset labels are **displayed target points**, not measured eye position;
  calibration assumes the user fixates the shown targets.
- GazeCapture sessions mix four phone orientations, so calibration is per
  orientation; RGBDGaze sessions are already per activity.
- Default K=9; K=13 helps GazeCapture slightly, K=26 has diminishing returns.
- Optional quality gate: `--gate-quantile 0.9` falls back to the prior when the
  calibration residual is in the top decile. On these benchmarks 0.9 was too
  aggressive; tune on validation before enabling.
- Reported numbers are on the official test splits; no test labels are used for
  fitting or hyper-parameter selection.

## Tests

```bash
python -m unittest tests.test_calibration -v
```

## License

MIT. See `LICENSE`.
