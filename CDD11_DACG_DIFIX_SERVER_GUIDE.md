# CDD-11 DACG + Difix3D 服务器指南

本实现位于分支 `codex/difix3d-degradation-lora-cdd11`。流程固定为：冻结 DACG
生成主图（coarse），原退化图作为参考图，冻结 DACG 的 DAM 产生 96 维
`P_global`，再训练 Difix3D adapter。VAE 编码器仍接收两视角，但解码器只使用
主视角的四级 skip，不把退化参考图的 skip 注入输出。

> 许可证：Difix3D 及其衍生代码受 NVIDIA 非商业许可证约束，只能用于非商业
> 研究或评估。完整条款见 `dacg_difix/LICENSE_DIFIX3D.txt`。

## 1. 环境

建议 Python 3.10、CUDA 12.1。依赖全部固定在 `requirements_difix.txt`：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.1.2 torchvision==0.16.2
python -m pip install -r requirements_difix.txt
accelerate config
```

首次运行需要从 Hugging Face 读取 `stabilityai/sd-turbo`。服务器已缓存模型时，
后续命令可加 `--local-files-only`，避免训练节点访问网络。

## 2. 用已训练 DACG 预生成主图

`MANIFEST_DIR` 包含 CDD-11 的 `train.jsonl`、`val.jsonl`、`test.jsonl`；
`DACG_CKPT` 是在 `cdd11-dacg-paper-loss-training-testing` 代码上训练的完整权重。

```bash
python prepare_cdd11_difix.py \
  --manifest-dir /data/cdd11/manifests \
  --checkpoint /checkpoints/dacg_cdd11_best.pt \
  --output-dir /data/cdd11_difix_prepared \
  --device cuda \
  --tile-size 512 \
  --tile-overlap 128
```

结果包含原尺寸 DACG 图像、`train.jsonl`、`val.jsonl`、`test.jsonl` 和
`prepare_metadata.json`。元数据记录 DACG 权重 SHA-256。显存足够时可用
`--tile-size 0` 做整图 DACG 推理。

## 3. 训练

默认实验配置正是 MSE/LPIPS/Gram 权重 `1/1/1`、Gram 从第 2000 步启用、
10000 个优化步、BF16、固定 diffusion timestep 199、随机种子 42。默认 LoRA 条件模式为
`p-global-layer-id`：`P_global` 与 UNet stage ID 共同生成每层 rank×rank 矩阵。

```bash
accelerate launch train_cdd11_difix.py \
  --train-manifest /data/cdd11_difix_prepared/train.jsonl \
  --validation-manifest /data/cdd11_difix_prepared/val.jsonl \
  --dam-checkpoint /checkpoints/dacg_cdd11_best.pt \
  --output-dir /runs/cdd11_difix_pglobal_layer \
  --pretrained-model stabilityai/sd-turbo \
  --resolution 512 \
  --max-train-steps 10000 \
  --mixed-precision bf16 \
  --timestep 199 \
  --lambda-mse 1 \
  --lambda-lpips 1 \
  --lambda-gram 1 \
  --gram-loss-warmup-steps 2000 \
  --lora-mode p-global-layer-id \
  --lora-rank-unet 32 \
  --lora-rank-vae 16 \
  --gradient-checkpointing \
  --enable-xformers-memory-efficient-attention
```

正式训练前先做两步 GPU smoke，验证真实 SD-Turbo、PEFT、DAM 权重和数据路径：

```bash
accelerate launch train_cdd11_difix.py \
  --train-manifest /data/cdd11_difix_prepared/train.jsonl \
  --validation-manifest /data/cdd11_difix_prepared/val.jsonl \
  --dam-checkpoint /checkpoints/dacg_cdd11_best.pt \
  --output-dir /runs/cdd11_difix_smoke \
  --max-train-steps 2 \
  --checkpointing-steps 2 \
  --validation-steps 2 \
  --validation-limit 2 \
  --dataloader-num-workers 0 \
  --mixed-precision bf16
```

训练期不会加载完整 DACG restoration 网络，只从同一 checkpoint 提取并冻结
DAM。checkpoint 只包含 Difix LoRA、条件生成器、主视角 VAE skip conv，以及
恢复训练所需的 optimizer/scheduler/RNG 状态；不会重复保存 SD-Turbo 或 DACG
基座权重。

断点续训时保留原实验参数并传入 `latest.pt`：

```bash
accelerate launch train_cdd11_difix.py \
  --train-manifest /data/cdd11_difix_prepared/train.jsonl \
  --validation-manifest /data/cdd11_difix_prepared/val.jsonl \
  --dam-checkpoint /checkpoints/dacg_cdd11_best.pt \
  --output-dir /runs/cdd11_difix_pglobal_layer \
  --resume /runs/cdd11_difix_pglobal_layer/checkpoints/latest.pt \
  --lora-mode p-global-layer-id
```

建议做三组明确消融：

- `static`：标准静态 LoRA，等价于条件矩阵 `C=I`。
- `p-global`：每张图由 `P_global` 产生一个矩阵，同一组件的所有层共享。
- `p-global-layer-id`：默认方案，由 `P_global + stage embedding` 产生分层矩阵。

三组运行只改变条件模式，其他超参数保持一致：

```bash
for MODE in static p-global p-global-layer-id; do
  accelerate launch train_cdd11_difix.py \
    --train-manifest /data/cdd11_difix_prepared/train.jsonl \
    --validation-manifest /data/cdd11_difix_prepared/val.jsonl \
    --dam-checkpoint /checkpoints/dacg_cdd11_best.pt \
    --output-dir "/runs/cdd11_difix_${MODE}" \
    --lora-mode "${MODE}" \
    --max-train-steps 10000 \
    --mixed-precision bf16
done
```

首版只注入全局退化向量，避免同时改变太多变量。DACG 的四级 layer prompts
适合后续做 stage-aligned 条件；空间退化特征则更适合通过低分辨率 spatial gate
调制 LoRA 输出，而不是直接生成每像素 rank×rank 矩阵。

## 4. 单图级联推理

该入口先运行完整 DACG，再复用同一次 DAM 计算得到的 `P_global` 运行 Difix，
最终裁回输入原尺寸：

```bash
python infer_cdd11_difix.py \
  --input /data/example/degraded.png \
  --output /data/example/restored.png \
  --coarse-output /data/example/dacg_coarse.png \
  --dacg-checkpoint /checkpoints/dacg_cdd11_best.pt \
  --difix-checkpoint /runs/cdd11_difix_pglobal_layer/checkpoints/latest.pt \
  --lora-mode p-global-layer-id \
  --precision bf16 \
  --device cuda
```

部署时必须同时保留 SD-Turbo 基座、DACG checkpoint 和 Difix adapter。

## 5. 原尺寸验证/测试

评估读取预生成 manifest 的 coarse/reference/target，DAM 仍从 DACG checkpoint
加载；batch size 固定为 1，因此不会把不同原尺寸图像强行堆叠。

```bash
python evaluate_cdd11_difix.py \
  --manifest /data/cdd11_difix_prepared/test.jsonl \
  --dam-checkpoint /checkpoints/dacg_cdd11_best.pt \
  --difix-checkpoint /runs/cdd11_difix_pglobal_layer/checkpoints/latest.pt \
  --output-dir /runs/cdd11_difix_pglobal_layer/test \
  --lora-mode p-global-layer-id \
  --precision bf16 \
  --device cuda
```

输出为：

- `images/`：每张原尺寸恢复结果；
- `per_image.csv`：逐图同时记录 DACG coarse 与 Difix 的 RGB PSNR、RGB SSIM、LPIPS；
- `summary.json` / `summary.csv`：11 个退化类别、single/double/triple arity、
  image-micro 与 11 类等权 macro 汇总；两套输出的指标均完整保留。

## 6. 离线运行时测试

这些测试全部使用小型 mock，不会下载 SD-Turbo、VGG 或 LPIPS 权重：

```bash
python -m unittest discover -s tests -v
```
