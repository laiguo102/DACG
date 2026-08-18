# 原版 DACG 在完整 CDD-11 上的训练与测试

该流程固定使用官方 `train` 的 1183 个场景及 11 类退化，共 13013 个唯一配对。它不划分训练折或验证集，不用 official test 选择模型；完成全部训练步数后产生唯一的 `final.pth`，随后才在 200×11 张 official test 图像上评估一次。

这与使用 1083/100 train/validation 划分的实验数据协议不同，结果应单独报告。

## 1. 进入环境与仓库

```bash
conda activate dacg-cdd11

export DACG_ROOT=/home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/all-in-one-model/DACG
export CDD11_ROOT=/home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/data/CDD11
export DACG_RUN_ROOT=/home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/all-in-one-model/experiments/cdd11-dacg-full
export DACG_RUN="$DACG_RUN_ROOT/dacg-ir-full-seed3407-v1"

cd "$DACG_ROOT"
```

该环境的 PyTorch wheel 需要优先加载其自带的 nvJitLink。若已经通过 `conda env config vars set` 固化，重新激活环境即可；检查如下：

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
assert torch.__version__ == "2.5.1+cu124"
assert torch.cuda.is_available()
PY
```

## 2. 更新分支与数据审计

```bash
git fetch origin cdd11-full-dacg-training-testing
git switch --track -c cdd11-full-dacg-training-testing \
  origin/cdd11-full-dacg-training-testing

python -m cdd11_full.verify \
  --data-root "$CDD11_ROOT" \
  --output "$DACG_RUN_ROOT/data_audit.json"
```

已有同名本地分支时执行：

```bash
git switch cdd11-full-dacg-training-testing
git pull --ff-only origin cdd11-full-dacg-training-testing
```

审计必须得到 train=1183 scenes/13013 pairs、test=200 scenes/2200 pairs。

## 3. 正式训练与 W&B

先完成 `wandb login`，再启动：

```bash
mkdir -p "$DACG_RUN_ROOT"

python -u train_cdd11_full.py \
  --data-root "$CDD11_ROOT" \
  --output-dir "$DACG_RUN" \
  --model DACG_IR \
  --epochs 120 \
  --batch-size 8 \
  --accumulate-grad-batches 1 \
  --patch-size 128 \
  --lr 2e-4 \
  --num-workers 8 \
  --precision fp32 \
  --log-interval-steps 50 \
  --checkpoint-interval-steps 500 \
  --wandb-mode online \
  --wandb-entity c14150591-sjtu \
  2>&1 | tee "$DACG_RUN_ROOT/dacg-ir-full-seed3407-v1.console.log"
```

该命令保留原版 DACG-IR（30,861,200 参数）、AdamW、L1 + 0.1×FFT 损失以及 15/150 的 epoch 级 warmup-cosine 调度。默认使用 FP32，与已通过的原版网络前反向审计一致。

每 500 optimizer steps 及每个 epoch 边界原子保存 `checkpoints/latest.pth`；`train_metrics.jsonl` 是本地持久监控记录。W&B 使用项目 `cdd11-restoration`、组 `cdd11-dacg-full-v1`。

## 4. 中断后精确恢复

只传入同一运行目录的 latest checkpoint；采样、优化器、调度器、随机数和 W&B run ID 均从 checkpoint 恢复：

```bash
python -u train_cdd11_full.py \
  --resume "$DACG_RUN/checkpoints/latest.pth" \
  --wandb-mode online \
  --wandb-entity c14150591-sjtu \
  2>&1 | tee -a "$DACG_RUN_ROOT/dacg-ir-full-seed3407-v1.console.log"
```

不要使用 `final.pth` 继续训练，也不要修改已有运行目录中的 `config.json`。

## 5. 完成门禁

```bash
python - <<'PY'
import json, os
from pathlib import Path
import torch

run = Path(os.environ["DACG_RUN"])
state = json.loads((run / "run_state.json").read_text())
checkpoint = torch.load(run / "checkpoints/final.pth", map_location="cpu", weights_only=False)
print("state:", state)
print("checkpoint:", checkpoint["status"], checkpoint["global_step"], checkpoint["config"]["total_steps"])
assert state["status"] == "completed"
assert checkpoint["status"] == "completed"
assert checkpoint["global_step"] == checkpoint["config"]["total_steps"]
print("CDD-11 full DACG completion gate: PASS")
PY
```

## 6. Official test（训练完成后仅执行一次）

```bash
python -u evaluate_cdd11_full.py \
  --checkpoint "$DACG_RUN/checkpoints/final.pth" \
  --data-root "$CDD11_ROOT" \
  --output-dir "$DACG_RUN/official_test" \
  --tile-size 512 \
  --tile-overlap 64 \
  --precision fp32 \
  --wandb-mode online \
  --wandb-entity c14150591-sjtu \
  2>&1 | tee "$DACG_RUN/official_test.console.log"
```

输出包括 2200 张恢复图、`metrics.csv` 和 `summary.json`；宏平均 PSNR/SSIM 也追加到同一个 W&B run。评估器拒绝覆盖非空测试目录。
