# CDD-11：DACG → Difix 的 5-Fold OOF 服务器运行方法

这套入口只读取标准 CDD-11 目录，不复制原始图片：

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
    └──（同样的 12 个子目录）
```

文件扩展名可以是 png/jpg/jpeg/bmp/tif/tiff；配对依据是不含扩展名的 scene ID。

以下假设：

```bash
CODE=/path/to/DACG-IR-code
DATA=/path/to/CDD11
OUT=/path/to/outputs/CDD11_5fold_oof
cd "$CODE"
bash install.sh
```

## 一条命令完成 DACG 的全部训练与 OOF 生成（推荐）

配置好 Conda 环境并切换到代码目录后，只需启动：

```bash
python train_cdd11_oof.py \
  --data-root /path/to/CDD11 \
  --output-root /path/to/outputs/CDD11_5fold_oof \
  --num-gpus 1
```

该命令会自动完成：数据校验与首次划分、5 个 DACG fold 的训练及 OOF 推理、11715 条 Difix 三元组合并、DACG-final 训练、Validation coarse 生成。中断后重新执行完全相同的命令即可：已有最佳 checkpoint 的阶段会跳过，只有 `last.ckpt` 的训练会自动恢复。

多卡只改为 `--num-gpus 4`。默认模型为 `DACG_IR`、120 epochs、patch 128、BF16。脚本不会自动运行 Official Test。

训练的全局有效 batch 为：

```text
batch-size（每卡） × num-gpus × accumulate-grad-batches
```

单卡默认是 `8 × 1 × 1 = 8`。若显存不足，可保持有效 batch=8：

```bash
python train_cdd11_oof.py \
  --data-root /path/to/CDD11 \
  --output-root /path/to/outputs/CDD11_5fold_oof_bs8 \
  --num-gpus 1 \
  --batch-size 2 \
  --accumulate-grad-batches 4 \
  --tile-size 512
```

四卡保持有效 batch=8 时使用 `--num-gpus 4 --batch-size 2`。`--tile-size` 只影响 coarse 推理，不改变训练。五折和 final 必须使用相同的 batch/GPU/累积配置；一键脚本会自动保证这一点。

下面是各阶段的等价手动命令，仅用于排错或单独重跑。

## 1. 首次且仅首次生成划分

```bash
python -m cdd11_oof.split --data-root "$DATA"
python -m cdd11_oof.verify --data-root "$DATA"
```

生成 `$DATA/splits/fold1.txt` 到 `fold5.txt`、`val.txt` 和 `split_info.json`。命令会严格检查 1183 个 train scene、200 个 test scene、每个 scene 的 11 种退化，并拒绝覆盖已有划分。后续所有实验复用这个 `splits` 目录。

## 2. 训练 5 个 OOF DACG，并推理各自未见的 fold

五次训练必须使用完全相同的参数。单卡示例：

```bash
for FOLD in 1 2 3 4 5; do
  python -m cdd11_oof.train \
    --data-root "$DATA" \
    --output-dir "$OUT/dacg_fold${FOLD}" \
    --role "fold${FOLD}" \
    --model DACG_IR \
    --epochs 120 \
    --batch-size 8 \
    --patch-size 128 \
    --num-workers 8 \
    --num-gpus 1

  python -m cdd11_oof.infer \
    --checkpoint "$OUT/dacg_fold${FOLD}/checkpoints/best_macro_psnr.ckpt" \
    --data-root "$DATA" \
    --target "fold${FOLD}" \
    --output-dir "$OUT/oof_outputs/fold${FOLD}" \
    --num-workers 4
done
```

`fold1` 模型只训练 F2+F3+F4+F5；其余同理。程序会核对 checkpoint 中的 role，阻止用错误 fold 的模型生成 coarse。

多卡时只改 `--num-gpus 4`。显存不够时减小 `--batch-size`；若整图推理显存不足，在 infer 命令加 `--tile-size 512 --tile-overlap 32`。

断点续训示例：

```bash
python -m cdd11_oof.train \
  --data-root "$DATA" \
  --output-dir "$OUT/dacg_fold1" \
  --role fold1 \
  --resume "$OUT/dacg_fold1/checkpoints/last.ckpt"
```

## 3. 合并并验证 Difix 训练三元组

```bash
python -m cdd11_oof.merge_oof \
  --oof-root "$OUT/oof_outputs" \
  --split-dir "$DATA/splits" \
  --output "$OUT/difix_train_oof.jsonl"
```

成功时得到严格的 11715 条 JSONL。每行字段为：

```json
{"scene_id":"000123","degradation":"rain","degraded":"...","coarse":"...","gt":"...","source_checkpoint":"..."}
```

Difix 训练数据集应直接读取 `degraded`、`coarse`、`gt` 三个路径。这个仓库不包含 Difix 实现，因此这里只生成可直接接入 Difix 的无泄漏训练清单，不虚构第二阶段训练入口。

## 4. 训练 DACG-final

在 OOF 数据生成完、Difix 配置确定后，用 1065 个 OOF-pool scene 训练最终 DACG：

```bash
python -m cdd11_oof.train \
  --data-root "$DATA" \
  --output-dir "$OUT/dacg_final" \
  --role final \
  --model DACG_IR \
  --epochs 120 \
  --batch-size 8 \
  --patch-size 128 \
  --num-workers 8 \
  --num-gpus 1
```

## 5. 为 Validation 生成 coarse

```bash
python -m cdd11_oof.infer \
  --checkpoint "$OUT/dacg_final/checkpoints/best_macro_psnr.ckpt" \
  --data-root "$DATA" \
  --target val \
  --output-dir "$OUT/val_coarse"
```

用 `val_coarse/manifest.jsonl` 跑 Difix 验证并选择 Difix checkpoint/采样参数。Validation 从未进入任一训练集。

## 6. 最终一次 Official Test

模型和参数全部冻结后再执行：

```bash
python -m cdd11_oof.infer \
  --checkpoint "$OUT/dacg_final/checkpoints/best_macro_psnr.ckpt" \
  --data-root "$DATA" \
  --target test \
  --output-dir "$OUT/test_coarse"
```

随后将 `test_coarse/manifest.jsonl` 输入已经冻结的 Difix，统计各退化和总体 PSNR/SSIM/LPIPS。不要用 Official Test 反复选择参数。
