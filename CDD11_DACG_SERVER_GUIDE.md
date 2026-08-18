# DACG 在 CDD-11-v1 上的训练与测试

本流程训练原版完整 `DACG_IR`（30,861,200 参数），不执行 OOF，也不改变网络结构。数据、采样、优化器、学习率、验证、检查点、W&B 与此前 UNet/Uformer 的 `cdd11-v1` 规范一致；唯一例外是保留论文原始目标函数：

```text
L = RGB L1 + 0.1 × Fourier L1
```

因此该结果属于“统一训练规范、各模型使用原生目标函数”的比较，不能表述为相同 loss 下的纯架构比较。

## 1. 更新代码与环境

```bash
export DACG_ROOT=/home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/all-in-one-model/DACG
cd "$DACG_ROOT"
git fetch origin cdd11-dacg-paper-loss-training-testing
git switch --track -c cdd11-dacg-paper-loss-training-testing \
  origin/cdd11-dacg-paper-loss-training-testing

conda activate dacg-cdd11
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m pip install -r requirements.txt
```

若本地分支已存在，改用：

```bash
git switch cdd11-dacg-paper-loss-training-testing
git pull --ff-only origin cdd11-dacg-paper-loss-training-testing
```

## 2. 冻结路径并审计数据

这里必须直接复用 UNet/Uformer 已生成并通过审计的 `cdd11-v1` manifest 目录，不重新划分数据。请把变量改成服务器上的真实路径。

```bash
export CDD11_MANIFEST_DIR=/path/to/existing/cdd11-v1-manifests
export CDD11_OUTPUT_ROOT=/home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/all-in-one-model/experiments/cdd11-v1-runs

cd "$DACG_ROOT"
python verify_cdd11.py --manifest-dir "$CDD11_MANIFEST_DIR"
```

审计应报告 train=11913、val=1100、test=2200，且每个 split 的 11 类数量一致。

## 3. 100-step smoke（建议先执行）

```bash
python -u train_cdd11.py \
  --manifest-dir "$CDD11_MANIFEST_DIR" \
  --output-root "$CDD11_OUTPUT_ROOT" \
  --run-kind smoke \
  --run-name dacg-smoke-native-wandb-seed3407-v1 \
  --inference-mode native \
  --microbatch-size 1 \
  --num-workers 8 \
  --wandb-mode online \
  --wandb-entity c14150591-sjtu \
  --wandb-project cdd11-restoration
```

确认 `run_state.json` 为 `completed`、step=100，存在 `latest.pth` 与 `best_macro_psnr.pth`，验证图像数为 1100、固定可视样本数为 22，且 `wandb_state.json` 的 errors=0。

## 4. 5000-step pilot

```bash
python -u train_cdd11.py \
  --manifest-dir "$CDD11_MANIFEST_DIR" \
  --output-root "$CDD11_OUTPUT_ROOT" \
  --run-kind pilot \
  --run-name dacg-pilot-native-wandb-seed3407-v1 \
  --inference-mode native \
  --microbatch-size 1 \
  --num-workers 8 \
  --wandb-mode online \
  --wandb-entity c14150591-sjtu
```

## 5. 200000-step formal

```bash
python -u train_cdd11.py \
  --manifest-dir "$CDD11_MANIFEST_DIR" \
  --output-root "$CDD11_OUTPUT_ROOT" \
  --run-kind formal \
  --run-name dacg-formal-native-wandb-seed3407-v1 \
  --inference-mode native \
  --microbatch-size 1 \
  --num-workers 8 \
  --wandb-mode online \
  --wandb-entity c14150591-sjtu
```

训练使用 patch=256、有效 batch=11（每类恰好一张）、BF16 网络前向与 FP32 Fourier loss、AdamW(lr=2e-4, betas=.9/.999, weight_decay=1e-4)、2000-step warmup 后 cosine 到 1e-6、梯度裁剪 1.0。每 5000 step 验证并按 macro PSNR 保存最佳模型。

## 6. 安全暂停与精确恢复

启动时可增加 `--pause-at-step 5000`；只允许在日志边界安全暂停。恢复只接受该运行的 `latest.pth`：

```bash
export DACG_FORMAL_RUN="$CDD11_OUTPUT_ROOT/dacg/dacg-formal-native-wandb-seed3407-v1"
python -u train_cdd11.py --resume "$DACG_FORMAL_RUN/checkpoints/latest.pth"
```

恢复时不要再传 manifest、输出目录、run name 或 W&B 参数；这些内容全部从冻结配置恢复，并继续同一个 W&B run ID。

## 7. 正式测试

只有 formal 运行达到 200000 step 且状态为 `completed` 后才允许读取 official test：

```bash
python -u evaluate_cdd11.py \
  --checkpoint "$DACG_FORMAL_RUN/checkpoints/best_macro_psnr.pth" \
  --num-workers 4
```

测试输出包括 2200 张预测、逐图 CSV、按退化和 single/double/triple/macro 汇总指标，以及两组完整场景共 22 个五联图。评估器拒绝覆盖已有正式测试结果。
