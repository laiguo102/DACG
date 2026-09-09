# CCDD-11 原生选择性监督 Difix3D 训练

本流程只使用 `CCDD-11/half_train`：固定 seed 42 在 1183 个 scene 上划分
1065 个训练 scene 和 118 个验证 scene。五种双退化各展开两个方向，主 target 始终是：

```text
remove A, preserve B
-> half_train/sub_data/A_B/<scene>/<scene>_B_.png
```

训练读取每条有向记录时还会以 `--negative-train-probability` 动态切换到
preserve-both 负样本，默认概率为 `0.2`。负样本使用
`preserve A, preserve B`，两路 condition 为双退化原图及有符号
`原图-DACG coarse` 纹理，target 为同一张双退化原图。该参数只属于
`train_ccdd11_difix.py`；设为 `0` 可恢复原正样本训练行为。

`_half_` 不参与监督，`half_test` 不参与训练或验证。第一阶段复用现有 CDD-trained
full-restoration DACG checkpoint；不在本实验中重训 DACG。

## 1. 环境变量

```bash
export CCDD_ROOT=/path/to/CCDD-11
export CCDD_COARSE_ROOT=/path/to/CCDD11-DACG-coarse
export DACG_CKPT=/path/to/dacg/checkpoints/final.pth
export RUN_ROOT=/path/to/ccdd11-difix-runs
export FORMAL_RUN="$RUN_ROOT/ccdd-all5-bs4-100k-lucid-seed42-v1"
```

## 2. 数据审计

```bash
python -u verify_ccdd11_selective.py \
  --data-root "$CCDD_ROOT" \
  --degradation-pairs 1 2 3 4 5 \
  --audit-output "$RUN_ROOT/ccdd11-audit"
```

PASS 时应报告 1183 个 scene、10 个 directed task，并在
`$RUN_ROOT/ccdd11-audit/target_semantics` 生成 10 张四联图。必须人工确认 target
方向正确；其中 target-vs-main-preserve PSNR 只是诊断，不能据此替换 native target。

## 3. 生成 DACG coarse

```bash
python -u prepare_ccdd11_coarse.py \
  --data-root "$CCDD_ROOT" \
  --split half_train \
  --dacg-checkpoint "$DACG_CKPT" \
  --coarse-root "$CCDD_COARSE_ROOT" \
  --degradation-pairs 1 2 3 4 5 \
  --device cuda
```

显存不足时增加 `--tile-size 512 --tile-overlap 64`。相同命令可断点续跑，已有图会
跳过。完成后 `$CCDD_COARSE_ROOT/half_train` 必须包含 5915 张 coarse 及
`coarse_preparation.json`。

## 4. Manifest-only 检查

```bash
python -u train_ccdd11_difix.py \
  --data-root "$CCDD_ROOT" \
  --coarse-root "$CCDD_COARSE_ROOT/half_train" \
  --output-dir "$RUN_ROOT/manifest-check" \
  --degradation-pairs 1 2 3 4 5 \
  --prepare-only
```

预期输出：

```text
Indexed 5915 coarse images
train samples = 10650
validation samples = 1180
```

该模式不会加载 SD-Turbo、LPIPS、VAE 或 UNet。

## 5. 两步 smoke

```bash
accelerate launch --mixed_precision=bf16 train_ccdd11_difix.py \
  --data-root "$CCDD_ROOT" \
  --coarse-root "$CCDD_COARSE_ROOT/half_train" \
  --output-dir "$RUN_ROOT/ccdd-selective-smoke" \
  --degradation-pairs 1 2 3 4 5 \
  --negative-train-probability 0.2 \
  --resolution 512 \
  --max-train-steps 2 \
  --train-batch-size 4 \
  --dataloader-num-workers 0 \
  --checkpointing-steps 2 \
  --latest-checkpointing-steps 2 \
  --eval-freq 1 \
  --full-eval-freq 2 \
  --viz-freq 1 \
  --num-validation-samples 10 \
  --num-validation-visualizations 10 \
  --lambda-l2 1 \
  --lambda-lpips 1 \
  --lambda-gram 0 \
  --seed 42 \
  --report-to wandb \
  --tracker-project-name difix-ccdd11-selective \
  --tracker-run-name ccdd-all5-selective-smoke
```

检查 overall 与 10 个 `validation[/_full]/tasks/...` 方向指标、五联图、`latest`、
milestone 和 `best_psnr`。训练 loss 同时提供旧键 `train/l2`,`train/lpips` 和规范键
`train/loss_l2`,`train/loss_lpips`，并以 `train/negative_fraction` 记录实际负样本比例。
另检查 `validation_negative[/_full]` 的 PSNR、SSIM、LPIPS、
`mean_absolute_change`、五个 pair 分组指标及负样本五联图。负验证按 scene/pair
去重为 590 条，不参与 `best_psnr` 选择。

## 6. 5000-step pilot 与恢复

```bash
accelerate launch --mixed_precision=bf16 train_ccdd11_difix.py \
  --data-root "$CCDD_ROOT" \
  --coarse-root "$CCDD_COARSE_ROOT/half_train" \
  --output-dir "$RUN_ROOT/ccdd-all5-bs4-5k-lucid-seed42-v1" \
  --degradation-pairs 1 2 3 4 5 \
  --negative-train-probability 0.2 \
  --resolution 512 \
  --max-train-steps 5000 \
  --train-batch-size 4 \
  --dataloader-num-workers 8 \
  --learning-rate 5e-6 \
  --lr-scheduler linear \
  --lr-warmup-steps 500 \
  --lambda-l2 1 --lambda-lpips 1 --lambda-gram 0 \
  --enable-xformers-memory-efficient-attention \
  --eval-freq 500 --full-eval-freq 5000 --viz-freq 1000 \
  --num-validation-samples 100 \
  --num-validation-visualizations 10 \
  --latest-checkpointing-steps 1000 \
  --milestone-steps 5000 \
  --seed 42 \
  --report-to wandb \
  --tracker-project-name difix-ccdd11-selective \
  --tracker-run-name ccdd-all5-bs4-5k-lucid-seed42-v1
```

中断后保持所有数据、模型和优化器参数一致，并增加：

```bash
--resume "$RUN_ROOT/ccdd-all5-bs4-5k-lucid-seed42-v1/checkpoints/latest.pkl"
```

先检查 10 个方向、rain target、显存、dataloader、resume 与 best PSNR；异常时先排查
target mapping、coarse、prompt 和 sub_data，不能直接进入正式训练。

## 7. 100000-step 正式训练（只准备命令）

```bash
accelerate launch --mixed_precision=bf16 train_ccdd11_difix.py \
  --data-root "$CCDD_ROOT" \
  --coarse-root "$CCDD_COARSE_ROOT/half_train" \
  --output-dir "$RUN_ROOT/ccdd-all5-bs4-100k-lucid-seed42-v1" \
  --degradation-pairs 1 2 3 4 5 \
  --negative-train-probability 0.2 \
  --resolution 512 \
  --max-train-steps 100000 \
  --train-batch-size 4 \
  --dataloader-num-workers 8 \
  --learning-rate 5e-6 \
  --lr-scheduler linear \
  --lr-warmup-steps 500 \
  --lambda-l2 1 \
  --lambda-lpips 1 \
  --lambda-gram 0 \
  --enable-xformers-memory-efficient-attention \
  --eval-freq 500 \
  --full-eval-freq 5000 \
  --viz-freq 1000 \
  --num-validation-samples 100 \
  --num-validation-visualizations 10 \
  --latest-checkpointing-steps 1000 \
  --milestone-steps 10000 \
  --seed 42 \
  --report-to wandb \
  --tracker-project-name difix-ccdd11-selective \
  --tracker-run-name ccdd-all5-bs4-100k-lucid-seed42-v1
```

只有 audit、manifest、smoke 和 5k pilot 均通过并人工确认后才运行此命令。

## 8. 100k 完成后的 half_test 旧式正式评估

### 8.0 在 validation 上扫描正负潜变量 CFG

在读取 `half_test` 前，可先对训练时冻结的 1180 条 validation 记录扫描 LUCID 风格
CFG。每个样本只计算一次 positive/negative 分支，然后在 VAE 解码前使用：

```text
z_cfg = z_negative + beta * (z_positive - z_negative)
skip_cfg[i] = skip_negative[i] + beta * (skip_positive[i] - skip_negative[i])
```

`beta=0` 是 preserve-both negative 分支，`beta=1` 是选择性去除 positive 分支，
`beta>1` 是远离 negative 分支的外插。潜变量和四层 VAE skip features 使用相同 beta
同步插值，因此两个端点分别严格等于完整 negative/positive mode。评估输出同时以
选择性 target、原双退化图和 clean GT 为参照计算
L2、LPIPS-VGG、L2+LPIPS、PSNR、SSIM 和 DISTS。

先把 `FORMAL_RUN` 指向实际完成的 100k CFG 训练目录：

```bash
export FORMAL_RUN="$RUN_ROOT/ccdd-all5-bs4-100k-lucid-seed42-v1"
```

```bash
python -u evaluate_ccdd11_cfg.py \
  --checkpoint "$FORMAL_RUN/checkpoints/best_psnr.pkl" \
  --output-dir "$FORMAL_RUN/cfg_validation_best_psnr_state_beta_sweep_v2" \
  --betas 0 0.25 0.5 0.75 1.0 1.05 1.1 1.2 \
  --num-gallery-samples 20 \
  --workers 8 \
  --device cuda \
  --mixed-precision bf16 \
  --seed 42 \
  --enable-xformers-memory-efficient-attention \
  --report-to wandb \
  --wandb-entity c14150591-sjtu \
  --wandb-project difix-ccdd11-selective \
  --wandb-run-name ccdd-all5-100k-best-state-cfg-validation-beta-sweep-v2
```

正式运行前可用独立输出目录做两个分层样本的 GPU smoke：

```bash
python -u evaluate_ccdd11_cfg.py \
  --checkpoint "$FORMAL_RUN/checkpoints/best_psnr.pkl" \
  --output-dir "$FORMAL_RUN/cfg_validation_state_smoke_v2" \
  --betas 0 1 1.2 \
  --num-gallery-samples 2 \
  --max-samples 2 \
  --workers 0 \
  --device cuda \
  --mixed-precision bf16 \
  --seed 42 \
  --enable-xformers-memory-efficient-attention \
  --report-to none
```

结果包含 `per_image_metrics.csv`、`summary.csv`、`metrics.json`、可恢复的
`records/` 与 `state.json`，以及 20 组带 beta 标签的 `gallery/` 对比图。完整扫描
不会读取或写入 `half_test`。

W&B 的 task-macro 和 micro 标量曲线显式使用 `cfg/beta` 作为横轴；此外
`cfg/task_curves/` 下为 18 个质量指标各生成一张多线图，每张图包含 10 条有向任务
曲线。已有完整结果无需重新推理，可创建一个新 W&B run：

```bash
python -u upload_ccdd11_cfg_wandb.py \
  --results-dir "$FORMAL_RUN/cfg_validation_best_psnr_state_beta_sweep_v2" \
  --wandb-entity c14150591-sjtu \
  --wandb-project difix-ccdd11-selective \
  --wandb-run-name ccdd-all5-100k-best-step90000-cfg-task-curves
```

#### 8.0.1 与像素级 beta 加权做同图对照

以下程序直接读取上一节 CFG gallery 中同一样本的两张端点图：`beta_0.png`
（negative mode）与 `beta_1.png`（positive mode），然后在 RGB `[0,1]` 像素域使用

```text
I_pixel = clip(I_negative + beta * (I_positive - I_negative), 0, 1)
```

因此输入图、样本和 beta 均与 latent/skip CFG 严格相同，且不需要再次运行模型。
程序会从 `records/` 和 `gallery/` 自动确认两个完整 scene；每个 scene 必须同时具有
5 类双退化的两个去除方向，共生成 2 × 5 × 2 = 20 张对比拼图。每张拼图和 CFG
gallery 一样，第一行是 4 张参考图，后两行是 8 个 beta 输出。

新开终端后先恢复路径：

```bash
cd /path/to/DACG
export RUN_ROOT=/path/to/ccdd11-difix-runs
export FORMAL_RUN="$RUN_ROOT/ccdd-all5-bs4-100k-lucid-seed42-v1"
export CFG_DIR="$FORMAL_RUN/cfg_validation_best_psnr_state_beta_sweep_v2"
```

默认自动识别两个完整 scene，并在同一 W&B project 新建独立对比 run：

```bash
python -u compare_ccdd11_pixel_cfg.py \
  --cfg-dir "$CFG_DIR" \
  --report-to wandb \
  --wandb-entity c14150591-sjtu \
  --wandb-project difix-ccdd11-selective \
  --wandb-run-name ccdd-all5-100k-best-state-pixel-beta-comparison-v2
```

W&B 的 `pixel_cfg/gallery` 表固定为 20 行；每行包含 scene、pair、remove、preserve，
以及并排的原 CFG 拼图和像素级拼图。若 gallery 中不止两个完整 scene，必须显式指定
两个 scene ID：

```bash
python -u compare_ccdd11_pixel_cfg.py \
  --cfg-dir "$CFG_DIR" \
  --scene-ids 000001 000002 \
  --report-to wandb
```

输出位于 `$CFG_DIR/pixel_beta_comparison/`，包括每个任务的 8 张独立像素混合图、
`pixel_contact_sheet.png`、记录所选 scene 和全部图片路径的 `comparison.json`，以及
可检查完成状态的 `state.json`。本程序只生成视觉对照，不重复计算定量指标。

#### 8.0.2 Latent–skip 独立归因

以下诊断保持 checkpoint、正负分支顺序和 VAE posterior `sample()` 不变，只把
latent beta 与四层 VAE skip beta 拆成独立变量。结果写入独立目录，不改变上一节的
CFG-v2 输出。先运行 20 个分层样本的 smoke：

```bash
export ATTR_DIR="$FORMAL_RUN/latent_skip_attribution_v1"
python -u evaluate_ccdd11_latent_skip.py \
  --stage smoke \
  --checkpoint "$FORMAL_RUN/checkpoints/best_psnr.pkl" \
  --output-dir "$ATTR_DIR/smoke" \
  --workers 0 --device cuda --mixed-precision bf16 \
  --enable-xformers-memory-efficient-attention --report-to none
```

smoke 完成后，在 200 个分层样本上扫描固定 5×5 网格：

```bash
python -u evaluate_ccdd11_latent_skip.py \
  --stage screen \
  --checkpoint "$FORMAL_RUN/checkpoints/best_psnr.pkl" \
  --output-dir "$ATTR_DIR/screen" \
  --workers 8 --device cuda --mixed-precision bf16 \
  --enable-xformers-memory-efficient-attention \
  --report-to wandb \
  --wandb-entity c14150591-sjtu \
  --wandb-project difix-ccdd11-selective \
  --wandb-run-name ccdd-all5-best-latent-skip-screen-v1
```

最后在完整 1180 个 validation 样本上确认核心条件、四层消融、两条响应轴，以及
screen 最优点的一阶 Manhattan 邻域。confirm 会核对 screen 与当前 checkpoint、
manifest、seed 和协议版本，任何不一致都会停止：

```bash
python -u evaluate_ccdd11_latent_skip.py \
  --stage confirm \
  --checkpoint "$FORMAL_RUN/checkpoints/best_psnr.pkl" \
  --screening-results "$ATTR_DIR/screen/metrics.json" \
  --output-dir "$ATTR_DIR/confirm" \
  --workers 8 --device cuda --mixed-precision bf16 \
  --enable-xformers-memory-efficient-attention \
  --report-to wandb \
  --wandb-entity c14150591-sjtu \
  --wandb-project difix-ccdd11-selective \
  --wandb-run-name ccdd-all5-best-latent-skip-confirm-v1
```

每个阶段均可从相同目录的 `state.json` 和 `records/` 恢复。主要结果见
`condition_summary.csv`、`paired_effects.csv`、`interaction_summary.csv` 和
`influence_report.json`；配对置信区间按 scene 聚类并使用固定 10000 次 bootstrap。

以下流程只能在 100k 正式训练完成后执行。先确认 `best_validation.json`，并记录仅由
`half_train` 的 `validation_full/psnr` 选出的 checkpoint 身份：

```bash
cat "$FORMAL_RUN/best_validation.json"
sha256sum "$FORMAL_RUN/checkpoints/best_psnr.pkl" \
  | tee "$FORMAL_RUN/best_psnr.sha256"
```

不得用 half_test 比较 milestone、调整超参、选择 checkpoint 或继续训练。

### 8.1 生成 1000 张 half_test DACG coarse

必须使用与 half_train 完全相同的 `$DACG_CKPT`：

```bash
python -u prepare_ccdd11_coarse.py \
  --data-root "$CCDD_ROOT" \
  --split half_test \
  --dacg-checkpoint "$DACG_CKPT" \
  --coarse-root "$CCDD_COARSE_ROOT" \
  --degradation-pairs 1 2 3 4 5 \
  --device cuda
```

完成后 `$CCDD_COARSE_ROOT/half_test` 应含 5 × 200 = 1000 张图及状态为
`completed` 的 `coarse_preparation.json`。生成器会拒绝混用 half_train、不同数据根或
不同 DACG checkpoint 的断点目录。

### 8.2 trained best 的 2000 条评估

```bash
python -u evaluate_ccdd11_difix.py \
  --data-root "$CCDD_ROOT" \
  --test-coarse-root "$CCDD_COARSE_ROOT/half_test" \
  --checkpoint "$FORMAL_RUN/checkpoints/best_psnr.pkl" \
  --output-dir "$FORMAL_RUN/half_test_best_psnr" \
  --degradation-pairs 1 2 3 4 5 \
  --resolution 512 \
  --lora-rank-vae 4 \
  --timestep 199 \
  --workers 8 \
  --mixed-precision bf16 \
  --seed 42 \
  --enable-xformers-memory-efficient-attention
```

target 固定为 `half_test/main_data/<preserve>/<scene>.png`；不会读取 `sub_data`、
`_half_` 或 `half_train`。默认保存全部 prediction，并可从 `records/` 断点续跑。

### 8.3 同协议 step-0 initialization

```bash
python -u evaluate_ccdd11_difix.py \
  --model-source initialization \
  --data-root "$CCDD_ROOT" \
  --test-coarse-root "$CCDD_COARSE_ROOT/half_test" \
  --output-dir "$FORMAL_RUN/half_test_initialization_seed42" \
  --degradation-pairs 1 2 3 4 5 \
  --resolution 512 \
  --lora-rank-vae 4 \
  --timestep 199 \
  --workers 8 \
  --mixed-precision bf16 \
  --seed 42 \
  --enable-xformers-memory-efficient-attention
```

### 8.4 逐图配对与 scene-cluster bootstrap

```bash
python -u compare_ccdd11_difix.py \
  --trained-dir "$FORMAL_RUN/half_test_best_psnr" \
  --initialization-dir "$FORMAL_RUN/half_test_initialization_seed42" \
  --output-dir "$FORMAL_RUN/half_test_trained_vs_initialization" \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 42
```

比较器会先校验 dataset、manifest SHA256、数据/coarse 根、sample ID、prompt、
target/degraded/coarse 路径及 degraded/coarse 指标一致，再按 scene 聚类计算 95% CI。
正式报告应包含 task-macro、10 个有向任务、final-vs-coarse 提升和胜率，以及
trained-vs-initialization 的 bootstrap CI。

## 9. 兼容性和实验边界

`train_cdd11_difix.py` 的默认 `--dataset-format` 仍为 `cdd11`，target 仍来自
`CDD11/train/<preserve>`。CCDD 入口默认 `ccdd11`，但模型、两路视觉输入、prompt、
L2+LPIPS loss、validation 和 checkpoint 代码完全共享。本阶段不做 `_half_`、identity、
ratio control、三退化训练、架构修改、联合 DACG+Difix 或 CCDD-native DACG 训练。
