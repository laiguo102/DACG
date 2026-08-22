# CDD11 选择性退化 Difix3D 训练

该流程分为两个独立阶段。首先使用 `cdd11-full-dacg-training-testing` 产生的
`final.pth`/`latest.pth`，把 CDD11 训练集的五类双退化全部推理到独立 coarse
文件夹；之后 Difix 训练只读取该文件夹，不再加载或运行 DACG。流程只读取
`CDD11/train`：固定 seed 42 划分 1065 个训练 scene 和 118 个验证 scene，
`CDD11/test` 不参与训练或验证。

Difix3D 源码及衍生代码受 NVIDIA 非商业许可证约束，详见
`difix3d_selective/LICENSE_DIFIX3D.txt`。

## 1. 环境

建议为 Difix 单独建立环境，避免其固定版本覆盖原 DACG 环境：

```bash
conda create -n dacg-difix python=3.10 pip -y
conda activate dacg-difix
python -m pip install -r requirements_difix.txt
```

首次运行会从 Hugging Face 下载 `stabilityai/sd-turbo`，并下载 LPIPS/VGG 权重。

## 2. 双退化编号

`--degradation-pairs` 接受一个或多个编号，并同时限定训练和验证：

1. `low_haze`
2. `low_rain`
3. `low_snow`
4. `haze_rain`
5. `haze_snow`

每张双退化图会展开为两个方向。例如编号 1 同时产生
`remove low light, preserve haze` 和 `remove haze, preserve low light`。

## 3. 一次性生成全部 coarse

如果服务器上的 `background` 已经是 DACG 对五类双退化的初步恢复结果，且目录为
`background/<pair>/<scene>.<suffix>`，不要再次运行 DACG。训练时直接令
`--coarse-root /path/to/CDD11/background`。开始前应确认它至少包含：

```text
background/
├── low_haze/<scene>.png
├── low_rain/<scene>.png
├── low_snow/<scene>.png
├── haze_rain/<scene>.png
└── haze_snow/<scene>.png
```

每个选中 pair 必须与 `CDD11/train/clear` 的 1183 个 scene 文件名 stem 完全一致。
以下生成命令只用于服务器尚无这些结果的情况。

```bash
python prepare_cdd11_coarse.py \
  --data-root /path/to/CDD11 \
  --dacg-checkpoint /path/to/cdd11-full-run/checkpoints/final.pth \
  --coarse-root /path/to/CDD11-DACG-coarse
```

默认生成 1–5 五类，共 `1183×5=5915` 张图。显存不足时增加
`--tile-size 512 --tile-overlap 64`。脚本会跳过已有输出，因此中断后可直接用相同
命令继续。输出目录固定为：

```text
CDD11-DACG-coarse/
├── low_haze/<scene>.png
├── low_rain/<scene>.png
├── low_snow/<scene>.png
├── haze_rain/<scene>.png
├── haze_snow/<scene>.png
└── coarse_preparation.json
```

如果只需预生成部分类型，可显式添加 `--degradation-pairs 1 3 5`。

## 4. 单卡训练（LUCID 风格主实验）

主实验采用更适合有像素级监督恢复任务的 LUCID 配置：batch 4、`5e-6`、500 步
warmup 后 linear decay、L2+LPIPS，并默认关闭 Difix3D 的 Gram loss。训练长度跟随
LUCID 设为 100000 步。BF16 保留不变；Gram/VGG 在主实验中不会加载。

```bash
accelerate launch --mixed_precision=bf16 train_cdd11_difix.py \
  --data-root /path/to/CDD11 \
  --coarse-root /path/to/CDD11/background \
  --output-dir /path/to/difix-selective-run \
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
  --report-to wandb \
  --tracker-project-name difix-cdd11-selective \
  --tracker-run-name difix-selective-lucid-100k
```

训练启动时索引 coarse、CDD 原双退化图、单退化 target 和 clean GT，并生成
manifest。每 500 步在按 10 个有向任务分层固定的 100 张验证图上上传 RGB PSNR、
RGB SSIM 和辅助 LPIPS；每 5000 步对全部 1180 张验证图上传同名
`validation_full/*` 指标。主指标始终是最终图相对单退化 target，另记录相对 clean
GT 和 DACG coarse 相对 target 的诊断指标。

每 1000 步上传 10 张固定验证样本的横向五联图：双退化图、DACG 初步去除图、
单退化 target、Difix 最终图、clean GT。可视化频率与快验证、全量验证相互独立。

正式跑 10000 步前，建议先用真实权重和 `background` 做两步 smoke：

```bash
accelerate launch --mixed_precision=bf16 train_cdd11_difix.py \
  --data-root /path/to/CDD11 \
  --coarse-root /path/to/CDD11/background \
  --output-dir /path/to/difix-selective-smoke \
  --degradation-pairs 1 2 3 4 5 \
  --max-train-steps 2 \
  --train-batch-size 4 \
  --dataloader-num-workers 0 \
  --checkpointing-steps 2 \
  --latest-checkpointing-steps 2 \
  --eval-freq 1 \
  --full-eval-freq 2 \
  --viz-freq 1 \
  --num-validation-samples 2 \
  --num-validation-visualizations 2 \
  --report-to wandb \
  --tracker-project-name difix-cdd11-selective \
  --tracker-run-name difix-selective-smoke
```

确认 W&B 中出现 `validation/psnr`、`validation/ssim` 和
`validation/degraded_coarse_target_final_gt` 后，再启动正式训练。

### NVIDIA Difix3D 风格消融

如需复现发布版 Difix3D 的训练策略，只改变以下参数。batch 固定为 1，避免其原始
Gram 定义在多样本间耦合；本仓库的实现即使 batch 大于 1 也会逐图计算 Gram。

```bash
--max-train-steps 10000 \
--train-batch-size 1 \
--learning-rate 2e-5 \
--lr-scheduler constant \
--lambda-gram 1 \
--gram-loss-warmup-steps 2000
```

## 5. 多卡训练

```bash
accelerate launch --mixed_precision=bf16 --multi_gpu --num_processes 4 \
  train_cdd11_difix.py \
  --data-root /path/to/CDD11 \
  --coarse-root /path/to/CDD11-DACG-coarse \
  --output-dir /path/to/difix-selective-run \
  --degradation-pairs 1 3 5 \
  --resolution 512 \
  --max-train-steps 100000 \
  --train-batch-size 4 \
  --gradient-accumulation-steps 2 \
  --enable-xformers-memory-efficient-attention \
  --gradient-checkpointing
```

仅 global rank 0 生成 manifest；其余进程等待完成后共同进入 Difix 训练。

## 6. 输出和恢复

```text
difix-selective-run/
├── prepared/
│   ├── manifests/{train,validation}.jsonl
│   └── split_and_preparation.json
└── checkpoints/
    ├── latest.pkl
    ├── best_psnr.pkl
    ├── model_<10k-step>.pkl
    └── final.pkl
```

manifest 中的 `image` 字段直接指向 `--coarse-root` 下的对应图像。恢复训练时保持
数据根目录和类别选择一致并增加：

```bash
--resume /path/to/difix-selective-run/checkpoints/latest.pkl
```

`latest.pkl` 每 1000 步原子覆盖；永久里程碑每 10000 步保存；`best_psnr.pkl` 按
全量验证 PSNR 更新，并在根目录 `best_validation.json` 记录 step、PSNR 和 SSIM。

## 7. 回填已有 10k run 的 W&B

旧训练无需重跑。以下命令按 checkpoint step 去重并评估现有 `model_*.pkl` 与
`final.pkl`，将完整验证 PSNR/SSIM、五联图和 clean/coarse 诊断上传到新的 companion
run；不会改写原 W&B 历史。

```bash
python -m difix3d_selective.backfill \
  --data-root /path/to/CDD11 \
  --coarse-root /path/to/CDD11/background \
  --run-dir /path/to/old-difix-selective-run \
  --degradation-pairs 1 2 3 4 5 \
  --report-to wandb \
  --tracker-project-name difix-cdd11-selective \
  --tracker-run-name all5-bs4-10k-seed42-v1-backfill \
  --source-wandb-run ENTITY/PROJECT/RUN_ID
```

本地结果写入 `<run-dir>/backfill/{metrics.jsonl,summary.json,media/}`。只有已保存的
checkpoint 才能恢复，因此旧 run 默认只能得到每 1000 步的历史点，不能重建 500
步时的模型输出。

选择 K 类时应得到 `1183×K` 张 coarse、`1065×K×2` 条训练记录和
`118×K×2` 条验证记录。
