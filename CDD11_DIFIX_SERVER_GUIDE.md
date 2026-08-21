# CDD11 选择性退化 Difix3D 训练

该流程读取 CDD11 和 `cdd11-full-dacg-training-testing` 产生的
`final.pth`/`latest.pth`。程序先对所选双退化生成 DACG coarse，再训练只解码主视角的
Difix。只读取 `CDD11/train`：固定 seed 42 划分 1065 个训练 scene 和 118 个验证
scene，`CDD11/test` 不参与训练或验证。

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

## 3. 单卡训练

```bash
accelerate launch --mixed_precision=bf16 train_cdd11_difix.py \
  --data-root /path/to/CDD11 \
  --dacg-checkpoint /path/to/cdd11-full-run/checkpoints/final.pth \
  --output-dir /path/to/difix-selective-run \
  --degradation-pairs 1 2 3 4 5 \
  --resolution 512 \
  --max-train-steps 10000 \
  --train-batch-size 1 \
  --dataloader-num-workers 8 \
  --enable-xformers-memory-efficient-attention \
  --tracker-run-name difix-selective-all-five
```

DACG 原图推理显存不足时增加 `--dacg-tile-size 512 --dacg-tile-overlap 64`。

## 4. 多卡训练

```bash
accelerate launch --mixed_precision=bf16 --multi_gpu --num_processes 4 \
  train_cdd11_difix.py \
  --data-root /path/to/CDD11 \
  --dacg-checkpoint /path/to/cdd11-full-run/checkpoints/final.pth \
  --output-dir /path/to/difix-selective-run \
  --degradation-pairs 1 3 5 \
  --resolution 512 \
  --max-train-steps 10000 \
  --train-batch-size 1 \
  --gradient-accumulation-steps 2 \
  --enable-xformers-memory-efficient-attention \
  --gradient-checkpointing
```

仅 global rank 0 生成 coarse；其余进程等待准备完成后共同进入 Difix 训练。

## 5. 输出和恢复

```text
difix-selective-run/
├── prepared/
│   ├── coarse/{train,validation}/<pair>/<scene>.png
│   ├── manifests/{train,validation}.jsonl
│   └── split_and_preparation.json
└── checkpoints/
    ├── model_<step>.pkl
    └── final.pkl
```

同一输出目录再次运行时会复用已经存在的 coarse。恢复训练时保持其余数据参数一致并增加：

```bash
--resume /path/to/difix-selective-run/checkpoints/model_5000.pkl
```

选择 K 类时应得到 `1183×K` 张 coarse、`1065×K×2` 条训练记录和
`118×K×2` 条验证记录。
