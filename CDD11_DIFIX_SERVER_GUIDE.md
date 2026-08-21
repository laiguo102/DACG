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

## 4. 单卡训练

```bash
accelerate launch --mixed_precision=bf16 train_cdd11_difix.py \
  --data-root /path/to/CDD11 \
  --coarse-root /path/to/CDD11-DACG-coarse \
  --output-dir /path/to/difix-selective-run \
  --degradation-pairs 1 2 3 4 5 \
  --resolution 512 \
  --max-train-steps 10000 \
  --train-batch-size 1 \
  --dataloader-num-workers 8 \
  --enable-xformers-memory-efficient-attention \
  --tracker-run-name difix-selective-all-five
```

训练启动时只索引 coarse、CDD 原退化图和单退化 target，并生成 manifest。

## 5. 多卡训练

```bash
accelerate launch --mixed_precision=bf16 --multi_gpu --num_processes 4 \
  train_cdd11_difix.py \
  --data-root /path/to/CDD11 \
  --coarse-root /path/to/CDD11-DACG-coarse \
  --output-dir /path/to/difix-selective-run \
  --degradation-pairs 1 3 5 \
  --resolution 512 \
  --max-train-steps 10000 \
  --train-batch-size 1 \
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
    ├── model_<step>.pkl
    └── final.pkl
```

manifest 中的 `image` 字段直接指向 `--coarse-root` 下的对应图像。恢复训练时保持
数据根目录和类别选择一致并增加：

```bash
--resume /path/to/difix-selective-run/checkpoints/model_5000.pkl
```

选择 K 类时应得到 `1183×K` 张 coarse、`1065×K×2` 条训练记录和
`118×K×2` 条验证记录。
