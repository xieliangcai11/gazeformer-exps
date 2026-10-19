# Gaze360 预处理数据 · 训练脚本适配性审计

审计对象：`data/Gaze360/GazeHub/{Image,Label}`（由官方 `preprocess_gaze360.py` 从 `gaze360/` 生成）。
审计工具：`verify_gaze360.py`（本次新增，可重复运行）。
审计范围：**只读**。未修改任何训练脚本、数据集加载器或数据本身。

环境：WSL Ubuntu-26.04，conda base，Python 3.14.7 / torch 2.14.0+cu126 / CUDA True / NVIDIA RTX 2080 Ti。

---

## 1. 结论摘要

**结论：预处理产物与当前训练脚本完全适配，可直接用于训练。验证器 8 类检查全绿（PASS 17 / FAIL 0），
仅 1 条坐标约定 WARNING 和 1 条信息项，均不影响 `train.py` 现有行为。**

| 检查项 | 结果 | 证据 |
|---|---|---|
| 标签格式（6 字段、路径模式、Origin 模式、表头） | PASS | 4 个 split 共 124,831 行全部合法，0 畸形行 |
| split 行数 vs `metadata.mat` | PASS | train 84,902 / val 11,318 / test 16,031 / unused 12,580 = 124,831，与元数据各 split 有效行数逐一相等 |
| 标签 ↔ 元数据一致性 | PASS | 抽样 100 条（每 split 25），Origin 路径与 `gaze_dir` 全部匹配（atol 1e-6） |
| 坐标约定 | WARN | 3DGaze 单位向量（max\|‖g‖−1\| = 3.33e-16）；2DGaze 与 3DGaze 一致（max 偏差 0.00e+00）；但与仓库 `utils.gaze_pitch_yaw_to_ccs` 约定相反（见 P1） |
| 图片完整性 | PASS | 374,493 张全部存在、JPEG SOI/EOI 标记完好；PIL 完整解码 + 尺寸/模式校验通过 |
| 孤立文件 | PASS | 磁盘集合与标签引用集合完全相等（各 374,493 个，双向差集为 0） |
| 裁剪质量 | PASS | 全黑像素占比 p95 ≤ 1.79%，无 > 50% 全黑裁剪；左右眼完全相同 0 例 |
| 训练脚本兼容性 | PASS | `train/val/test.label` 齐备；路径布局与 `config` 一致；`DatasetGaze360ByGazeHub` 解析路径 112,251 行全部可解析 |
| `unused` split | INFO | 12,580 行（占有效数据 10.1%）在盘上但不被训练消费（见 P3） |

验证器退出码：**0**（默认）/**1**（`--strict`，因 1 条 WARN）。负向测试（注入 6 类错误）全部被对应检查捕获，退出码 1。

---

## 2. 数据概览

| split | 标签行数 | 元数据有效行数 | 图片数 | 磁盘占用 |
|---|---|---|---|---|
| train | 84,902 | 84,902 | 254,706 | 1.7 GB |
| val | 11,318 | 11,318 | 33,954 | 233 MB |
| test | 16,031 | 16,031 | 48,093 | 321 MB |
| unused | 12,580 | 12,580 | 37,740 | 272 MB |
| **合计** | **124,831** | **124,831** | **374,493** | **2.6 GB** |

- `gaze360/metadata.mat` 共 197,588 行，其中 `person_face_bbox` 全为 −1 的 72,757 行被官方脚本跳过
  （无法定位人脸，不产出裁剪图与标签），有效行 124,831。
- 标签文件每行 6 字段：`Face Left Right Origin 3DGaze 2DGaze`，以空格分隔；`Face` 形如 `train/Face/7371.jpg`。
- 首行为表头 `Face Left Right Origin 3DGaze 2DGaze`；`gazehub_datasets.py` 用 `splitlines()[1:]` 跳过它，
  因此表头必须存在（已确认 4 个文件均有）。
- 图片尺寸：Face 224×224，Left/Right 60×36，均为 RGB JPEG，无 0 字节文件。
- 文件名编号 = 元数据行号 + 1（`preprocess_gaze360.py:66` 的 `f"{i+1}.jpg"`），验证器据此回查元数据。

---

## 3. 验证器使用方法

```bash
# 默认：全量文件标记检查 + PIL 5% 抽样解码（约 8–10 分钟）
PYTHONPATH=. python verify_gaze360.py

# 全量 PIL 解码（约 12 分钟，最严格）
PYTHONPATH=. python verify_gaze360.py --full

# 提高抽样量 + 落盘 JSON 报告
PYTHONPATH=. python verify_gaze360.py --sample 100 --json report.json

# 严格模式：WARNING 也返回退出码 1（适合 CI）
PYTHONPATH=. python verify_gaze360.py --strict
```

参数：`--root`（默认 `gaze360`）、`--out`（默认 `data/Gaze360/GazeHub`）、`--sample`（默认 25）、
`--full`、`--json`、`--strict`、`--seed`（默认 0，保证可复现）。

退出码：

| 码 | 含义 |
|---|---|
| 0 | 无 FAIL（`--strict` 时也要求无 WARN） |
| 1 | 存在 FAIL；或 `--strict` 下存在 WARN |
| 2 | 用法/环境错误（如完全找不到标签文件） |

性能参考（WSL `/mnt/e`，32 线程）：374,493 次 JPEG 标记检查约 227 s；5% 抽样 PIL 解码约 13 s；
`--full` 全量解码约 41 s。脚本每 20,000 个文件打印进度。

---

## 4. 发现的问题

按严重程度排序。**P1/P2 是已证实的陷阱，P3/P4 是信息项**——均不阻塞当前训练。

### P1 坐标约定不一致（潜在陷阱，当前不触发）

**现象**：标签 `2DGaze` 的 yaw 用 `atan2(x, -z)`（`preprocess_gaze360.py:90` 的 `GazeTo2d`），
而仓库 `utils.gaze_pitch_yaw_to_ccs` 用 `yaw = atan2(x, z)`（`utils.py:66`，`x=cos(pitch)·sin(yaw)`,
`z=cos(pitch)·cos(yaw)`）。两者相差 z 轴符号。

**影响**：当前 `train.py` 只消费 `label[4]`（3DGaze），完全不读 `2DGaze`，因此无影响。
但任何把 `2DGaze` 当 `(pitch, yaw)` 喂给 `gaze_pitch_yaw_to_ccs` 的代码会静默产生错误方向——
实测最大方向误差 **180°**（即完全反向）。`gazehub_datasets.py:195` 的 `DatasetETHXGazeByGazeHub`
正是用 `label[1]`（2DGaze）调 `gaze_pitch_yaw_to_ccs`，但那条路径只服务 ETH-XGaze 数据集，
与 Gaze360 的 `label[4]` 路径互不相干。

**建议**：在文档与代码注释中固化该约定，勿混用。若未来要消费 Gaze360 的 `2DGaze`，
需显式把 yaw 取反后再调用 `gaze_pitch_yaw_to_ccs`，或直接写一个 `ccs_to_pitch_yaw` 反函数
（`train.py:357` 的 `ccs_to_pitchyaw` 已是正确反函数的本地实现，用的是 `atan2(x, z)`）。

### P2 dtype 混用契约脆弱（已复现的错误）

**现象**：CLIP ViT-B/32 权重为 fp16/fp32 混合（`visual.conv1` / `visual.proj` / transformer
`in_proj_weight` 为 fp16，`ln_pre` / `token_embedding` / `logit_scale` 为 fp32），
`encode_image` 输出 fp16，`process_batch` 产出的 `feature_1` / `feature_2` / `token_img_patch` 为 fp16，
而 `TransformerDeepSeek_gaze` 的 `proj_f1` / `proj_f2` / `proj_f3` / `proj_patch` 为 fp32。

在 `autocast()` 之外直接调用 `transformer_model(raw_inputs)` 会抛：

```
RuntimeError: mat1 and mat2 must have the same dtype, but got Half and Float
```

崩溃点：`transformer_models.py:1076` 的 `self.proj_f1(raw_inputs["feature_1"])`
（实际源码位于该函数的 `token_f1 = self.dropout(self.proj_f1(...))` 一行）。

**影响**：`train.py` 训练路径在 `autocast()` 内（`train.py:289`），eval 路径有
`raw_inputs[k].float()` 兜底（`train.py:319-321`），因此不会崩。但该契约是隐式的：
任何新增的推理/评估入口若忘记这两者之一就会立刻炸。

**建议**：任何新增的推理/评估入口显式 `.float()` 或包在 `autocast` 内。更稳的做法是让
`TransformerDeepSeek_gaze.forward` 开头统一 `raw_inputs = {k: v.float() if isinstance(v, torch.Tensor) else v ...}`，
把契约变成显式的。本次审计不落地该修改。

### P3 `unused` split 未被使用（信息项）

**现象**：`unused.label` 12,580 行（占有效数据 10.1%），对应 37,740 张图在盘上，
但 `train.py` 只读 `train.label` / `val.label` / `test.label`，该 split 不参与训练/验证/测试。

**影响**：数据量与 GazeHub 官方用法一致（官方脚本同样产出 `unused`），无正确性问题。

**建议**：若想扩大训练数据量，需显式把 `unused.label` 并入训练集。这属于策略决策，本次不执行。

### P4 `val.label` 依赖（健壮性提示）

**现象**：`train.py:190` 要求 `train/val/test.label` 三者齐备才走官方 split 分支；
缺 `val.label` 会静默走 80/10/10 随机切分，破坏官方划分（`train.py:195-206`）。

**影响**：当前三者齐备，无影响。属于对未来数据布局变更的健壮性提示。

**建议**：删除或重命名 `val.label` 前先确认这是有意行为。若希望缺文件时显式报错而非静默切分，
可修改 `train.py` 的分支逻辑（本次不执行）。

---

## 5. 训练脚本兼容性逐项核对

| 项目 | 状态 | 说明 |
|---|---|---|
| 路径布局 | 一致 | `data/Gaze360/GazeHub/{Image,Label}` 与 `config.DATASETS_PATH = Path("data")` + `TRAIN_DATASET_NAME = "Gaze360"` 推导的 `data/Gaze360/GazeHub/Image`、`Label` 完全一致（`config.py:29,33-34`，`train.py:179-185`） |
| 标签字段索引 | 正确 | `DatasetGaze360ByGazeHub = DatasetEyeDiapByGazeHub`（`gazehub_datasets.py:163`）消费 `label[0]`（Face 相对路径）与 `label[4]`（3DGaze，× `coefficients=[1,1,1]`，因 `TRAIN=Gaze360, TEST=Gaze360`，`gazehub_datasets.py:102-103`）。112,251 行同构解析全部成功，Face 图片均存在 |
| 标签表头 | 正确 | 4 个 `.label` 首行均为表头；`splitlines()[1:]` 跳过后数据行数与磁盘图片数一致 |
| split 使用 | 官方划分 | `train/val/test.label` 齐备 → `train.py:190` 走显式 split 分支，不用随机切分 |
| 图片尺寸（CLIP） | 满足 | Face 224×224，`CLIP_PREPROCESS` 与 `CNN_PREPROCESS` 均要求 224 输入 |
| 图片尺寸（CNN） | 满足 | 同上，`DatasetGaze360ByGazeHub.__getitem__` 用 Face 过 `transform`（CLIP）与 `other_transform`（CNN） |
| Left/Right 眼图 | 未消费 | 60×36，`DatasetGaze360ByGazeHub` 不读取；保留在盘上供其他用途 |
| 训练前向 | 通过 | 数据集加载、`process_batch`、`TransformerDeepSeek_gaze` 前向、autocast 训练步实测全部通过 |
| 峰值显存 | 11.18 GB | `BATCH_SIZE = 64`，CLIP ViT-B/32 + ResNet-50 + TransformerDeepSeek（12 层，d_model 768） |
| 吞吐参考 | — | `NUM_WORKERS = 0`（`config.py:53`），单进程加载；瓶颈在 CLIP 前向而非数据加载 |
| dtype | 契约脆弱 | 见 P2。训练路径在 `autocast` 内、eval 路径有 `.float()`，当前不触发 |

---

## 6. 复现命令

WSL Ubuntu-26.04，conda base 环境（`~/miniconda3/bin/python`，torch 2.14.0+cu126，cuda True）：

```bash
# 1. 正样本验证（默认抽样解码，约 8–10 分钟；预期 PASS 17 / FAIL 0 / WARN 1 / INFO 1，退出码 0）
wsl -d Ubuntu-26.04 -- bash -lc "cd /mnt/e/code/gazeformer-exps && \
  PYTHONPATH=. ~/miniconda3/bin/python verify_gaze360.py"

# 2. 全量 PIL 解码（最严格，约 12 分钟）
wsl -d Ubuntu-26.04 -- bash -lc "cd /mnt/e/code/gazeformer-exps && \
  PYTHONPATH=. ~/miniconda3/bin/python verify_gaze360.py --full"

# 3. 严格模式（WARN 也失败，预期退出码 1，因 P1 坐标约定 WARNING）
wsl -d Ubuntu-26.04 -- bash -lc "cd /mnt/e/code/gazeformer-exps && \
  PYTHONPATH=. ~/miniconda3/bin/python verify_gaze360.py --strict"

# 4. 落盘 JSON 报告
wsl -d Ubuntu-26.04 -- bash -lc "cd /mnt/e/code/gazeformer-exps && \
  PYTHONPATH=. ~/miniconda3/bin/python verify_gaze360.py --json report.json"
```

### 负向验证（证明验证器不是恒真）

在 `/tmp` 下构造坏副本（不触碰 `data/`）：截断 `train.label` 到 30 行，并注入 6 类错误——
篡改 3DGaze（非单位向量 + 与 `gaze_dir` 不符）、Face 引用不存在的编号、Left 引用不存在的编号、
字段数改为 5、Face 路径 split 写成 `val`、额外放置孤立文件。预期 8 类检查中 8 类 FAIL、退出码 1：

```bash
# 构造（示例，需按实际路径调整）
mkdir -p /tmp/gz_bad/data/Gaze360/GazeHub/{Image,Label}
ln -s /mnt/e/code/gazeformer-exps/gaze360/metadata.mat /tmp/gz_bad/gaze360/metadata.mat
# ... 拷贝 Label 与引用的 Image，然后按上述 6 类错误篡改 train.label

# 验证（预期 FAIL 8 / PASS 10，退出码 1）
wsl -d Ubuntu-26.04 -- bash -lc "cd /mnt/e/code/gazeformer-exps && \
  PYTHONPATH=. ~/miniconda3/bin/python verify_gaze360.py \
  --root /tmp/gz_bad/gaze360 --out /tmp/gz_bad/data/Gaze360/GazeHub"
```

实测负向结果（2026-09-21）：`label_format` 捕获字段数/split/编号错配、`split_counts` 捕获行数不符、
`label_metadata_consistency` 捕获 gaze_dir 与 index 越界、`gaze_conventions` 捕获非单位向量与 2D↔3D 偏差、
`image_integrity` 捕获缺图、`orphans` 捕获双向差集、`crop_quality` 捕获解码失败、
`compat_dataset_parse` 捕获 Face 图片不存在——全部 FAIL，退出码 1。
