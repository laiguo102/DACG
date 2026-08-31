# CCDD-11 原生选择性监督 Difix3D 训练

本流程只使用 `CCDD-11/half_train`：固定 seed 42 在 1183 个 scene 上划分
1065 个训练 scene 和 118 个验证 scene。五种双退化各展开两个方向，主 target 始终是：

```text
remove A, preserve B
-> half_train/sub_data/A_B/<scene>/<scene>_B_.png
```

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
`train/loss_l2`,`train/loss_lpips`。

## 6. 5000-step pilot 与恢复

```bash
accelerate launch --mixed_precision=bf16 train_ccdd11_difix.py \
  --data-root "$CCDD_ROOT" \
  --coarse-root "$CCDD_COARSE_ROOT/half_train" \
  --output-dir "$RUN_ROOT/ccdd-all5-bs4-5k-lucid-seed42-v1" \
  --degradation-pairs 1 2 3 4 5 \
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
