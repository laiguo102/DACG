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

## 8. 兼容性和实验边界

`train_cdd11_difix.py` 的默认 `--dataset-format` 仍为 `cdd11`，target 仍来自
`CDD11/train/<preserve>`。CCDD 入口默认 `ccdd11`，但模型、两路视觉输入、prompt、
L2+LPIPS loss、validation 和 checkpoint 代码完全共享。本阶段不做 `_half_`、identity、
ratio control、三退化训练、架构修改、联合 DACG+Difix 或 CCDD-native DACG 训练。
