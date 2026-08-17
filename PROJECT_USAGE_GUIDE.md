# DACG-IR 项目结构、CDD-11 一键训练与服务器使用手册

本文是当前仓库的统一使用入口，重点说明 CDD-11 上 `DACG → Difix` 的 scene-level 5-Fold OOF 流程，同时区分仓库中的原始训练入口和 AIO3-v1 公共 runner。

## 1. 当前项目包含什么

```text
DACG-IR-code/
├── src/                         # 原仓库训练、测试、网络与传统 Dataset
│   ├── train.py                 # 原始 DACG-IR 训练入口
│   ├── test.py                  # 原始测试入口
│   ├── net/model.py             # DACG-IR / DACG-IR-S 网络
│   ├── data/                    # 原始数据集与退化逻辑
│   ├── MulWeatherData/          # 多天气数据入口
│   └── utils/                   # loss、指标、调度器等
│
├── aio3_runner/                 # AIO3-v1 三任务公平比较 runner
├── AIO3_SERVER_GUIDE.md         # AIO3-v1 服务器说明
│
├── cdd11_oof/                   # CDD-11 scene-level 5-Fold OOF runner
│   ├── protocol.py              # 固定 seed、退化列表、划分及指纹
│   ├── data.py                  # 标准 CDD-11 索引、配对、裁剪和增强
│   ├── split.py                 # 首次生成 5 folds + validation
│   ├── verify.py                # 数据规模、11 类配对、泄漏检查
│   ├── model.py                 # DACG Lightning 包装、训练/验证指标
│   ├── tracking.py              # W&B、global_step、媒体、Artifact、本地日志
│   ├── train.py                 # 单个 fold 或 DACG-final 训练
│   ├── infer.py                 # OOF、Validation、Test coarse 推理
│   └── merge_oof.py             # 合并并验证 11715 条 Difix 三元组
│
├── train_cdd11_oof.py           # CDD-11 DACG 侧一键启动入口
├── CDD11_OOF_SERVER_GUIDE.md    # 分阶段详细命令
└── tests/test_cdd11_oof.py      # 划分、配对、泄漏单元测试
```

不要混用三种入口：

- 复现原论文传统实验时使用 `src/train.py`、`src/test.py`；
- 做 AIO3-v1 去噪/去雨/去雾公平比较时使用 `aio3_runner`；
- 做本文的 CDD-11 两阶段 OOF 实验时只使用 `train_cdd11_oof.py` 和 `cdd11_oof`。

## 2. CDD-11 实验协议

官方训练集 1183 个 clear scene 按固定 `seed=42` 划分：

```text
1183 = 1065 OOF pool + 118 validation
1065 = 5 folds × 213 scenes
```

每个 scene 对应 11 种退化：

```text
low
haze
rain
snow
low_haze
low_rain
low_snow
haze_rain
haze_snow
low_haze_rain
low_haze_snow
```

同一 clear scene 的 11 种退化永远进入同一个 split。每个 OOF DACG 仅在其他四折训练，再对未见过的目标折生成 coarse：

```text
DACG-fold1: train F2+F3+F4+F5 → infer F1
DACG-fold2: train F1+F3+F4+F5 → infer F2
...
DACG-fold5: train F1+F2+F3+F4 → infer F5
```

最终生成：

```text
1065 scenes × 11 degradations = 11715 OOF triples
(degraded, coarse, gt)
```

Difix 只能使用这 11715 条 OOF 三元组训练。118 个 Validation scene 和官方 200 个 Test scene 不进入任何训练集。

## 3. 标准 CDD-11 数据目录

`--data-root` 必须直接指向包含 `train` 和 `test` 的 CDD11 目录：

```text
CDD11/
├── train/
│   ├── clear/
│   ├── low/
│   ├── haze/
│   ├── rain/
│   ├── snow/
│   ├── low_haze/
│   ├── low_rain/
│   ├── low_snow/
│   ├── haze_rain/
│   ├── haze_snow/
│   ├── low_haze_rain/
│   └── low_haze_snow/
└── test/
    └── 与 train 相同的 12 个子目录
```

图片通过不含扩展名的 scene ID 配对。例如：

```text
train/clear/000123.png
train/rain/000123.png
train/low_haze_rain/000123.png
```

支持 `png/jpg/jpeg/bmp/tif/tiff`。一键脚本会严格检查：

- train 是否为 1183 个 scene；
- test 是否为 200 个 scene；
- 每个 clear 是否都有全部 11 种退化；
- fold 和 validation 是否完全互斥；
- split 是否仍是固定 seed=42 的结果。

## 4. 创建 Conda 环境

```bash
conda create -n DACG_CDD11 python=3.10 pip -y
conda activate DACG_CDD11
```

推荐安装 PyTorch 2.5.1 + CUDA 11.8：

```bash
conda install \
  pytorch==2.5.1 \
  torchvision==0.20.1 \
  pytorch-cuda=11.8 \
  -c pytorch \
  -c nvidia \
  -y
```

进入仓库并安装其余依赖：

```bash
cd /path/to/DACG-IR-code
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

验证 GPU、BF16、Lightning 和 W&B：

```bash
python - <<'PY'
import torch
import lightning
import wandb

print("torch:", torch.__version__)
print("lightning:", lightning.__version__)
print("wandb:", wandb.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda runtime:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("bf16:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
assert wandb.__version__ == "0.25.1"
PY
```

## 5. W&B 登录与监控约定

在线监控前登录一次：

```bash
wandb login
```

CDD-11 使用独立的 W&B 空间，避免混入 AIO3-v1：

```text
project: cdd11-restoration
group: cdd11-dacg-difix-5fold-oof-v1
```

`--wandb-entity` 填自己的 W&B 用户名或团队名。每个 `fold1...fold5` 和 `final` 对应一个独立 run。监控内容包括：

- 总 loss、L1、FFT loss、学习率和裁剪前梯度范数；
- GPU allocated/reserved 显存、step 时间和吞吐量；
- signed residual 均值、标准差、极值和正负比例；
- prediction 超出 `[0,1]` 的比例；
- 11 类退化各自的 Validation PSNR/SSIM；
- Validation macro PSNR/SSIM；
- 固定验证样本的 input、prediction、target、absolute error 和 signed residual；
- split Artifact 和最佳 checkpoint Artifact。

所有指标绑定统一 `global_step`。每个 run 还会在服务器本地保存：

```text
wandb_run_id.txt
wandb_state.json
train_metrics.jsonl
validation_metrics.jsonl
logs/wandb_errors.jsonl
wandb/
```

W&B 单次日志错误不会删除本地 checkpoint。在线初始化失败则会在训练开始前退出。

## 6. A800 80GB 一键启动推荐配置

进入 tmux，避免 SSH 断开导致前台进程丢失：

```bash
tmux new -s cdd11
```

激活环境并启动：

```bash
conda activate DACG_CDD11
cd /path/to/DACG-IR-code

python train_cdd11_oof.py \
  --data-root /path/to/CDD11 \
  --output-root /path/to/outputs/CDD11_A800_WANDB \
  --model DACG_IR \
  --epochs 120 \
  --num-gpus 1 \
  --batch-size 32 \
  --accumulate-grad-batches 1 \
  --patch-size 128 \
  --lr 2e-4 \
  --precision bf16-mixed \
  --num-workers 8 \
  --infer-workers 4 \
  --val-every 5 \
  --wandb-media-every 40 \
  --wandb-mode online \
  --wandb-entity YOUR_ENTITY
```

将 `YOUR_ENTITY` 替换为实际 W&B 用户名或团队名。

一键脚本依次自动完成：

1. 首次生成固定 split；
2. 完整数据和泄漏校验；
3. 训练 DACG-fold1；
4. 对 F1 生成 OOF coarse；
5. 依次完成 fold2 至 fold5；
6. 合并并验证 11715 条 Difix JSONL；
7. 用完整 1065-scene OOF pool 训练 DACG-final；
8. 用 DACG-final 生成 Validation coarse；
9. 停止，不自动运行 Official Test。

退出 tmux 而不停止训练：

```text
Ctrl+B，然后按 D
```

重新进入：

```bash
tmux attach -t cdd11
```

## 7. 断点恢复

中断后重新执行完全相同的一键命令即可。

脚本的恢复规则：

- 已存在 `best_macro_psnr.ckpt`：认为该 role 已完成并跳过训练；
- 只有 `last.ckpt`：自动恢复该 role；
- OOF manifest 完整：跳过对应推理；
- OOF manifest 不完整：重新生成对应输出；
- W&B 使用原 `wandb_run_id.txt` 和 `resume=must` 接回同一个 run；
- 参数与 `pipeline_config.json` 不一致：拒绝恢复。

因此不要手工删除：

```text
pipeline_config.json
wandb_run_id.txt
wandb_state.json
checkpoints/last.ckpt
```

## 8. 显存不足时怎么处理

### 8.1 训练阶段 CUDA OOM

全局有效 batch 的计算方式：

```text
global effective batch = batch-size × num-gpus × accumulate-grad-batches
```

A800 推荐有效 batch=32。显存不足时依次尝试：

| 每卡 batch | 梯度累积 | 有效 batch | 使用场景 |
|---:|---:|---:|---|
| 32 | 1 | 32 | A800 80GB 首选 |
| 16 | 2 | 32 | 第一次 OOM |
| 8 | 4 | 32 | 仍然 OOM |
| 4 | 8 | 32 | 保守配置 |
| 2 | 16 | 32 | 最后显存兜底 |

例如降到 batch 16：

```bash
python train_cdd11_oof.py \
  --data-root /path/to/CDD11 \
  --output-root /path/to/outputs/CDD11_A800_BS16_ACC2 \
  --model DACG_IR \
  --epochs 120 \
  --num-gpus 1 \
  --batch-size 16 \
  --accumulate-grad-batches 2 \
  --patch-size 128 \
  --lr 2e-4 \
  --precision bf16-mixed \
  --num-workers 8 \
  --wandb-mode online \
  --wandb-entity YOUR_ENTITY
```

改变 batch、累积、GPU 数、patch、精度或学习率后必须使用新的 `--output-root`。这些参数会写入冻结的 `pipeline_config.json`，不能用新配置恢复旧 checkpoint。

### 8.2 batch=1 仍然 OOM

最后才减小 patch，例如：

```bash
--patch-size 96
```

或：

```bash
--patch-size 64
```

减小 patch 会改变训练数据分布，属于新实验，必须使用新输出目录；五个 fold 和 final 必须统一采用相同 patch。

### 8.3 coarse 推理阶段 OOM

推理 OOM 不需要修改训练 batch。依次尝试：

```bash
--tile-size 1024 --tile-overlap 32
```

```bash
--tile-size 512 --tile-overlap 32
```

```bash
--tile-size 256 --tile-overlap 32
```

`tile-size=0` 表示整图推理。tile 参数不写入训练冻结配置，因此可以在同一输出目录重新运行一键命令；脚本会保留训练 checkpoint，只重做不完整推理。

注意：`--tile-size` 只作用于 OOF/Validation/Test coarse 推理，不作用于训练期间的原分辨率 Validation。如果标准 CDD-11 Validation 在 batch size 1 下仍然 OOM，不应静默改变评价方式，需要先为所有比较模型统一设计 tiled-validation 协议。

### 8.4 GPU 利用率低但显存充足

先增加 DataLoader worker：

```bash
--num-workers 12 --infer-workers 8
```

CPU 和内存充足时可以尝试：

```bash
--num-workers 16 --infer-workers 8
```

若出现 `DataLoader worker killed`、共享内存不足或系统 RAM 紧张，降为：

```bash
--num-workers 4 --infer-workers 2
```

worker 数和 tile 大小可以在恢复时调整；它们不改变模型优化配置。

### 8.5 BF16 不可用

A800 支持 BF16。若换成不支持 BF16 的旧 GPU，使用：

```bash
--precision 16-mixed
```

精度模式变化必须使用新输出目录。

## 9. 无网络时使用离线 W&B

将在线参数改为：

```bash
--wandb-mode offline
```

离线模式不需要 `--wandb-entity`，其余监控和本地文件仍然生成。网络恢复后只同步指定 run：

```bash
wandb sync /path/to/run/wandb/offline-run-*
```

不要在同一个 `--output-root` 中途切换 online/offline。需要切换时使用新的输出目录。

## 10. 一键运行后的输出结构

```text
CDD11_A800_WANDB/
├── pipeline_config.json             # 冻结的一键运行配置
├── splits/
│   ├── fold1.txt ... fold5.txt
│   ├── val.txt
│   └── split_info.json
├── dacg_fold1/ ... dacg_fold5/
│   ├── checkpoints/
│   │   ├── best_macro_psnr.ckpt
│   │   └── last.ckpt
│   ├── wandb_run_id.txt
│   ├── wandb_state.json
│   ├── train_metrics.jsonl
│   ├── validation_metrics.jsonl
│   ├── logs/wandb_errors.jsonl
│   └── wandb/
├── oof_outputs/
│   ├── fold1/ ... fold5/
│   │   ├── manifest.jsonl
│   │   └── <degradation>/<scene_id>.png
├── difix_train_oof.jsonl            # 11715 条 Difix 训练三元组
├── dacg_final/
│   ├── checkpoints/
│   └── W&B 与本地指标文件
└── val_coarse/
    ├── manifest.jsonl
    └── <degradation>/<scene_id>.png
```

## 11. Difix 如何接入

本仓库当前不包含 Difix 实现。一键脚本提供：

```text
<output-root>/difix_train_oof.jsonl
```

每行格式：

```json
{
  "scene_id": "000123",
  "degradation": "rain",
  "degraded": "/path/to/CDD11/train/rain/000123.png",
  "coarse": "/path/to/output/oof_outputs/foldX/rain/000123.png",
  "gt": "/path/to/CDD11/train/clear/000123.png",
  "source_checkpoint": "/path/to/dacg_foldX/checkpoints/best_macro_psnr.ckpt"
}
```

Difix Dataset 直接读取 `degraded`、`coarse`、`gt`。不得用 DACG 在自身训练 scene 上生成的 coarse 替换 OOF coarse。

Validation 使用：

```text
<output-root>/val_coarse/manifest.jsonl
```

## 12. Official Test

一键脚本故意不自动运行 Official Test。只有 DACG、Difix checkpoint、采样步数、loss 权重和所有推理参数全部冻结后，才执行一次：

```bash
python -m cdd11_oof.infer \
  --checkpoint /path/to/output/dacg_final/checkpoints/best_macro_psnr.ckpt \
  --data-root /path/to/CDD11 \
  --split-dir /path/to/output/splits \
  --target test \
  --output-dir /path/to/output/test_coarse \
  --num-workers 4
```

如果整图推理确实 OOM，可以增加统一的：

```bash
--tile-size 512 --tile-overlap 32
```

随后把 `test_coarse/manifest.jsonl` 输入已经冻结的 Difix，报告 11 类退化和总体 PSNR、SSIM、LPIPS，以及 DACG → Difix 的提升。Official Test 不得用于调参或 checkpoint 选择。

## 13. 常用检查命令

检查进程：

```bash
pgrep -af 'python.*train_cdd11_oof'
pgrep -af 'python.*cdd11_oof.train'
```

观察 GPU：

```bash
watch -n 1 nvidia-smi
```

检查最近本地训练指标：

```bash
tail -n 1 /path/to/run/train_metrics.jsonl | python -m json.tool
```

检查 W&B 状态：

```bash
cat /path/to/run/wandb_state.json
cat /path/to/run/wandb_run_id.txt
```

重新校验数据和 split：

```bash
python -m cdd11_oof.verify \
  --data-root /path/to/CDD11 \
  --split-dir /path/to/output/splits
```

运行协议单元测试：

```bash
python -m unittest -v tests.test_cdd11_oof
```

## 14. 最短操作清单

```bash
conda activate DACG_CDD11
cd /path/to/DACG-IR-code
wandb login

python train_cdd11_oof.py \
  --data-root /path/to/CDD11 \
  --output-root /path/to/outputs/CDD11_A800_WANDB \
  --num-gpus 1 \
  --batch-size 32 \
  --accumulate-grad-batches 1 \
  --precision bf16-mixed \
  --wandb-mode online \
  --wandb-entity YOUR_ENTITY
```

中断后再次执行同一条命令；训练 OOM 时改为 `batch-size=16, accumulate=2` 并换新输出目录；推理 OOM 时保持原输出目录并增加 `--tile-size 512 --tile-overlap 32`。
