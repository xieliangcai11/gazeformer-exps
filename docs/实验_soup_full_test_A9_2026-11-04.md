# 实验记录：soup 模型 full-test A9 校准（2026-11-04）

> 模型：`soup_best_ep2_5050.pt` = 0.5 × `best.pt` + 0.5 × `epoch_2.pt`
> （同一 `cm_ctx_ft` 训练 run 的权重平均；val cm 1.6169 / norm 0.1983）。
>
> 校准：与 A9 完全相同口径（DINO PCA16 + landmark + 3×3 region，target
> standardization，LOO λ，per-axis Huber，K=50，seeds=3，clip 0.02 norm）。
>
> 主结果：**full test 校准 cm = 1.1383**（raw 1.5817，base 1.1975），
> 比原 best 的 1.1461 好 0.0078 cm（约 0.68%）。

---

## 1. 结果

| 指标 | raw | base（50 点） | A9 校准 |
|---|---:|---:|---:|
| **cm** | 1.5817 | 1.1975 | **1.1383** |
| norm | 0.1836 | 0.1394 | **0.1326** |
| 相对 raw | — | −24.3% | **−28.03%** |
| 相对 base | — | — | **−4.94%** |

- 443 个会话、152,037 个 heldout 帧；
- **434/443 会话变好**；
- worst +10.61%；
- bootstrap 95% CI（calib−raw, cm）：[−0.4845, −0.4033]；
- 运行约 48 s，maxrss 1.22 GB（纯 CPU）。

## 2. 与原 best A9 对比

| 模型 | raw cm | base cm | 校准 cm | 变好 | worst |
|---|---:|---:|---:|---:|---:|
| 原 `cm_ctx_ft/best.pt` | 1.5918 | 1.2067 | 1.1461 | 431/443 | +2.29% |
| **soup`best+epoch2`** | **1.5817** | **1.1975** | **1.1383** | **434/443** | +10.61% |

soup 在 raw、base、calibrated 三个层面都更好；代价是最差会话的退化更大
（+10.61% vs +2.29%），但变好会话数更多。

## 3. 分 orientation

| ori | raw cm | base cm | A9 soup cm | 原 A9 cm |
|---|---:|---:|---:|---:|
| 1 | 1.6832 | 1.2381 | 1.1750 | 1.1750 |
| 2（iPad） | 1.9171 | 1.4399 | 1.3536 | 1.3408 |
| 3 | 1.4798 | 1.1543 | **1.0977** | 1.1080 |
| 4 | 1.4862 | 1.1416 | **1.0900** | 1.1087 |

soup 在 ori3/4 更好，ori2 略差，ori1 持平。

## 4. 计算口径

- 测试集：full test 175,563 帧（8 个 shard 合并，校验 175,563）；
- 会话：按 `session dir || orientation` 分组，443 个会话参与校准评估；
- 每个会话 K=50 支持帧，seeds=0,1,2，指标为 3 个 seed 的 heldout 平均；
- raw/base/calibrated 都在同一 heldout 帧上计算；
- DINO PCA16 用 soup 自己的 val（58,136 帧）拟合，explained=0.8032；
- landmark 使用 `lm_gc_test.npz`；region=3×3；
- 与 A9 原实验唯一不同的是基模型和特征来源（soup 而非 best）。

## 5. 复现

### 5.1 校准（特征已存在）

```bash
bash scripts/calibration/run_soup_full_A9_calib.sh
```

等价命令：

```bash
cd /data1/xieliangcai/code/gazeformer-exps
source /data1/xieliangcai/conda/etc/profile.d/conda.sh
conda activate dl

nice -n 10 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python -u -m scripts.calibration.cm_residual_calib \
  --train-npz out/gazecapture/analysis/feats_soup_val.npz \
  --test-npz out/gazecapture/analysis/feats_soup_test_full.npz \
  --index-csv data/GazeCapture/index/index.csv \
  --dino-npz out/gazecapture/analysis/dino_pca16_soup_full.npz \
  --landmark-npz out/gazecapture/analysis/lm_gc_test.npz \
  --features dino,landmark \
  --regions 3x3 \
  --K 50 --seeds 3 \
  --target-standardize --loo-lam \
  --huber-scale axis \
  --clip-mode fixed --clip-norm 0.02 \
  --out out/gazecapture/analysis/soup_full_A9.csv \
  --detail out/gazecapture/analysis/soup_full_A9_detail.csv
```

### 5.2 特征（如果不存在）

```bash
# soup val
python -m scripts.calibration.dump_feats \
  --ckpt out/gazecapture/analysis/soup_best_ep2_5050.pt \
  --dataset gazecapture --index-dir data/GazeCapture/index --split val \
  --resolution 224 --use-eye --output-space cm --half --amp \
  --log-every 200 --batch-size 16 --num-workers 2 --device cuda:0 \
  --out out/gazecapture/analysis/feats_soup_val.npz

# test：8 个 shard，--shard i --nshards 8；合并后必须是 175,563 帧
# 合并：np.concatenate 各 shard 的 feats/pred/label/cm_x/cm_y/ang_deg/
#       path/subject/frame_index，保存 feats_soup_test_full.npz
```

### 5.3 DINO PCA

```bash
python -m scripts.calibration.dino_extra_features \
  --train-npz out/gazecapture/analysis/feats_soup_val.npz \
  --test-npz out/gazecapture/analysis/feats_soup_test_full.npz \
  --index-csv data/GazeCapture/index/index.csv \
  --components 16 \
  --out out/gazecapture/analysis/dino_pca16_soup_full.npz
```

## 6. 产物

本次提交：

```text
results/calibration/soup_full_A9/soup_full_A9.csv
results/calibration/soup_full_A9/soup_full_A9_detail.csv
results/calibration/soup_full_A9/soup_full_A9.summary.json
results/calibration/soup_full_A9/soup_full_A9.args.json
results/calibration/soup_full_A9/soup_full_A9.log
results/calibration/soup_full_A9/dino_pca16_soup_full.log
results/calibration/soup_full_A9/feats_soup_val.log
```

服务器上的大文件（不入 git）：

```text
out/gazecapture/analysis/soup_best_ep2_5050.pt
out/gazecapture/analysis/feats_soup_test_full.npz   # 531MB
out/gazecapture/analysis/feats_soup_val.npz         # 176MB
out/gazecapture/analysis/dino_pca16_soup_full.npz
```

## 7. 风险与备注

1. soup 的 50/50 权重是在 val 上选的，论文中应注明；
2. 校准的 PCA/prior 使用 soup val，和 test 无重叠；
3. worst +10.61% 集中在少数小会话；如果论文更看重稳定性，可加 gate/收缩；
4. 本次 full-test 特征导出期间服务器多次重启/被其他用户高负载打断，
   最终用 8 个 shard 分段完成；特征文件只在服务器，未入 git；
5. s10 预览（1.0174）与 full test（1.1383）不可直接比较。
