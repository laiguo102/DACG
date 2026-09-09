# SelectiveDifix DAEM-lite

本改造在现有 VAE decoder 的四层 primary/coarse skip 上增加轻量细节专家。它不改变
UNet、两路视觉输入、数据 manifest、target、loss、validation 主指标或输出 clamp。
默认 `--no-detail-enabled`，原训练命令的前向路径保持不变。

## 结构

每层先执行原有 `skip_conv_l` 投影，随后计算：

```text
p = skip_conv_l(encoder_skip * gamma)
e = NAFBlockLite(p)
g = sigmoid(G(concat(decoder_feature, p)))
refined_skip = p + alpha_l * (2 * g - 1) * e
decoder_feature = up_block(decoder_feature + refined_skip)
```

四层输出通道依次为 `512, 512, 512, 256`。默认每层一个 NAFBlockLite，gate 是
spatial-channel map，reduction 为 4，`alpha_l=0.1`。gate 最后一层卷积权重和偏置为
零，因此初始化时 `g=0.5`、centered residual 严格为零，启用后的 step-0 输出与原
skip 路径一致。

| decoder level | projected skip | gate 输入（无 prompt） | NAFBlockLite |
|---|---:|---:|---:|
| 0 / deepest | 512 | 1024 | 1 |
| 1 | 512 | 1024 | 1 |
| 2 | 512 | 1024 | 1 |
| 3 / shallowest | 256 | 512 | 1 |

默认配置新增 `6,668,612` 个 detail 参数。实际运行会在控制台和 W&B 的
`model/parameters/*` 中记录 `detail`、`vae_adaptation`、`unet`、`trainable` 和总参数量，
从而给出当前 diffusers/PEFT 版本下 `detail+vae` 与 `all` 的精确计数。

首轮实验只使用 primary/coarse skip。`--detail-gate-use-prompt` 是后续消融开关，默认
关闭；不开启 reference skip 或 hard gate。

## 训练范围与学习率

- `--train-scope detail`：只训练四层 DAEM-lite；UNet、VAE LoRA 和原 skip conv 冻结。
- `--train-scope detail+vae`：训练 DAEM-lite、VAE decoder LoRA 和原 skip conv；UNet 冻结。
- `--train-scope all`：训练原有 UNet/VAE adaptation 和 DAEM-lite。

`--learning-rate` 控制 UNet/VAE adaptation，`--detail-learning-rate` 只控制 DAEM-lite。
不开启 detail 时仍使用原来的单 optimizer parameter group。

## Checkpoint 语义

`--init-checkpoint` 只载入模型权重，`global_step`、optimizer 和 scheduler 从零开始，适合
从现有 `best_psnr.pkl` 启动 stage-2。`--resume` 恢复完整训练状态。两者互斥。

新 checkpoint 单独保存 `detail_config` 和 `state_dict_detail`。载入新 checkpoint 时要求
detail 结构配置一致；老 checkpoint 没有这两个字段时继续载入已有 UNet/VAE 权重，
DAEM-lite 保持新建时的 zero-gate 初始化。

## CCDD-11 两步 smoke

```bash
export DAEM_RUN="$RUN_ROOT/ccdd-daem-lite-detail-only-v1"

accelerate launch --mixed_precision=bf16 train_ccdd11_difix.py \
  --data-root "$CCDD_ROOT" \
  --coarse-root "$CCDD_COARSE_ROOT/half_train" \
  --output-dir "$DAEM_RUN/smoke" \
  --degradation-pairs 1 2 3 4 5 \
  --negative-train-probability 0.2 \
  --resolution 512 \
  --max-train-steps 2 \
  --train-batch-size 4 \
  --dataloader-num-workers 0 \
  --detail-enabled \
  --detail-num-blocks 1 \
  --detail-gate-reduction 4 \
  --detail-alpha-init 0.1 \
  --train-scope detail \
  --detail-learning-rate 1e-4 \
  --init-checkpoint "$FORMAL_RUN/checkpoints/best_psnr.pkl" \
  --lr-scheduler linear \
  --lr-warmup-steps 0 \
  --lambda-l2 1 --lambda-lpips 1 --lambda-gram 0 \
  --eval-freq 1 --full-eval-freq 2 --viz-freq 1 \
  --num-validation-samples 2 \
  --num-validation-visualizations 2 \
  --latest-checkpointing-steps 2 \
  --milestone-steps 2 \
  --seed 42 \
  --report-to wandb \
  --tracker-project-name difix-ccdd11-selective-daem \
  --tracker-run-name ccdd-daem-lite-detail-only-smoke-v1
```

检查控制台参数量、两步 loss、bf16 显存和 checkpoint；W&B 应出现
`train/detail/l0..l3/{gate_mean,gate_std,gate_min,gate_max,alpha,residual_ratio}`、吞吐及
峰值显存。初始化 gate 应为 0.5；一个优化 step 后应开始偏离常数。

## 首个 10k stage-2 实验

smoke 通过后使用同一个 baseline checkpoint warm-start：

```bash
accelerate launch --mixed_precision=bf16 train_ccdd11_difix.py \
  --data-root "$CCDD_ROOT" \
  --coarse-root "$CCDD_COARSE_ROOT/half_train" \
  --output-dir "$DAEM_RUN/detail-only-10k" \
  --degradation-pairs 1 2 3 4 5 \
  --negative-train-probability 0.2 \
  --resolution 512 \
  --max-train-steps 10000 \
  --train-batch-size 4 \
  --dataloader-num-workers 8 \
  --detail-enabled \
  --detail-num-blocks 1 \
  --detail-gate-reduction 4 \
  --detail-alpha-init 0.1 \
  --train-scope detail \
  --learning-rate 5e-6 \
  --detail-learning-rate 1e-4 \
  --init-checkpoint "$FORMAL_RUN/checkpoints/best_psnr.pkl" \
  --lr-scheduler linear \
  --lr-warmup-steps 200 \
  --lambda-l2 1 --lambda-lpips 1 --lambda-gram 0 \
  --enable-xformers-memory-efficient-attention \
  --eval-freq 250 --full-eval-freq 2000 --viz-freq 1000 \
  --num-validation-samples 100 \
  --num-validation-visualizations 10 \
  --latest-checkpointing-steps 1000 \
  --milestone-steps 5000 \
  --seed 42 \
  --report-to wandb \
  --tracker-project-name difix-ccdd11-selective-daem \
  --tracker-run-name ccdd-daem-lite-detail-only-10k-v1
```

对 CDD-11 使用同一组 detail 参数并换成 `train_cdd11_difix.py`、CDD 数据路径；不要添加
CCDD 专属的 `--negative-train-probability`。

## 结果判定

checkpoint 仍只由 selective positive validation PSNR 选择。优先比较同一个 baseline
checkpoint 下的 task-macro PSNR 和 10 个 directed task，并同时检查 LPIPS/DISTS、
negative identity 和图像。重点排查 rain/snow re-injection、gate 饱和、residual ratio
过大，以及 PSNR 与感知指标之间的取舍。
