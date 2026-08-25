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

## 8. 在官方 test 上评估

测试只使用训练期间从未读取的 `CDD11/test`。应优先测试由验证集选择出的
`best_psnr.pkl`；不要根据 test 指标在 `final.pkl` 和多个 milestone 之间反复选择，
否则 test 会事实上变成验证集。

### 8.1 生成独立的 test coarse

使用与训练 coarse 完全相同的 DACG checkpoint，但必须写入新的目录。不要把
`CDD11/background`（训练 scene 的 coarse）传给测试器。

```bash
python -u prepare_cdd11_coarse.py \
  --split test \
  --data-root /path/to/CDD11 \
  --dacg-checkpoint /path/to/cdd11-full-run/checkpoints/final.pth \
  --coarse-root /path/to/CDD11-DACG-coarse-test \
  --degradation-pairs 1 2 3 4 5 \
  --device cuda
```

选择 K 个 pair 时应生成 `200×K` 张 coarse。目录根部的
`coarse_preparation.json` 会明确记录 `split=test` 和 `status=completed`；评测器会
检查该标记，防止训练 coarse 与测试 coarse 混用。显存不足时同样可增加
`--tile-size 512 --tile-overlap 64`，中断后用同一命令续跑。

### 8.2 运行选择性 DiFix 测试

```bash
python -u evaluate_cdd11_difix.py \
  --data-root /path/to/CDD11 \
  --test-coarse-root /path/to/CDD11-DACG-coarse-test \
  --checkpoint /path/to/difix-selective-run/checkpoints/best_psnr.pkl \
  --resolution 512 \
  --workers 4 \
  --mixed-precision bf16 \
  --enable-xformers-memory-efficient-attention
```

默认从 checkpoint 所属 run 的 `prepared/split_and_preparation.json` 推断训练时选择的
pair；旧 run 若缺少该文件，可显式增加 `--degradation-pairs 1 2 3 4 5`。若训练时
修改过 `--lora-rank-vae` 或 `--timestep`，测试命令必须传入相同值。

五类完整测试共 `200×5×2=2000` 个有向任务样本。每个样本以“需要保留的单退化图”
为主参考，分别评估：

- `degraded`：原始双退化输入；
- `coarse`：DACG 初步恢复；
- `final`：选择性 DiFix 输出。

指标包括 RGB PSNR、RGB SSIM、LPIPS-VGG 和 DISTS。PSNR/SSIM 越高越好，
LPIPS-VGG/DISTS 越低越好。所有 `improvement_*` 都统一成“正值表示 final 优于
coarse”：PSNR/SSIM 使用 `final-coarse`，LPIPS/DISTS 使用 `coarse-final`；同时输出
逐图胜率。指标在训练协议相同的 512×512 bicubic 输入/参考图上计算，不应与
native-resolution DACG 正式测试数字混为一列。

默认结果目录为 `<run-dir>/test_best_psnr/`：

```text
test_best_psnr/
├── state.json
├── manifest.jsonl
├── metrics.json
├── summary.csv
├── per_image_metrics.csv
├── records/                 # 每图原子记录，用于断点续跑
├── predictions/             # 2000 张 final；可用 --no-save-predictions 关闭
└── gallery/                 # 固定分层五联图
```

`summary.csv` 按 10 个有向任务、5 个 pair、micro 和 task-macro 分组。同一命令在
中断后会跳过已有 `records/` 并继续；完成目录不会被覆盖，需要重测时请指定新的
`--output-dir`。首次使用 DISTS/LPIPS 时会下载相应的感知网络权重。

### 8.3 与训练前 step-0 初始化做配对比较

为了隔离本项目训练本身的作用，评测器可以构造与训练开始时完全相同的模型初始化，
但不加载任何训练 checkpoint。该基线仍使用 `stabilityai/sd-turbo` 权重、本项目的
双视图 UNet、VAE skip/LoRA、相同 timestep 和提示词；训练 seed 为 42 时必须显式
保持 `--seed 42`。除是否加载训练权重外，其余测试输入和设置均与正式测试相同：

```bash
python -u evaluate_cdd11_difix.py \
  --model-source initialization \
  --data-root /path/to/CDD11 \
  --test-coarse-root /path/to/CDD11-DACG-coarse-test \
  --output-dir /path/to/difix-selective-run/test_initialization_seed42 \
  --degradation-pairs 1 2 3 4 5 \
  --resolution 512 \
  --workers 4 \
  --mixed-precision bf16 \
  --seed 42 \
  --enable-xformers-memory-efficient-attention
```

初始化基线不接受 `--checkpoint`。输出仍使用 `final_*` 字段表示该 step-0 模型的
输出指标，`metrics.json` 会记录 `model_source=initialization`、初始化来源与 seed。

完成后，对已有训练结果和初始化结果执行逐图配对比较：

```bash
python -u compare_cdd11_difix.py \
  --trained-dir /path/to/difix-selective-run/test_best_psnr_step080000 \
  --initialization-dir /path/to/difix-selective-run/test_initialization_seed42 \
  --output-dir /path/to/difix-selective-run/compare_step080000_vs_initialization
```

比较器会先确认两侧 2000 个 `sample_id`、任务、prompt，以及 degraded/coarse 指标
完全对应，然后输出 `per_image_comparison.csv`、`summary.csv` 和
`comparison.json`。`trained_advantage` 统一为正值代表训练后更好；95% 置信区间
使用 10000 次按 `scene_id` 聚类的 bootstrap，以避免把同一 scene 的 10 个有向任务
错误地视为相互独立。如果某指标的 `ci95_low > 0`，对应结论为
`trained_better`；区间跨 0 时标记为 `inconclusive`。

若已有可信的 test coarse 但没有元数据，可显式传
`--allow-unverified-test-coarse`；这会在结果配置中标记
`coarse_metadata_verified=false`，不建议用于正式报告。

### 8.4 双退化训练权重的三退化 OOD 测试

CDD11 官方 test 还包含两种三退化组合：

| ID | 目录 | 三个选择性任务示例 |
|---:|---|---|
| 1 | `low_haze_rain` | `preserve low light, remove haze and rain` |
| 2 | `low_haze_snow` | `preserve snow, remove low light and haze` |

该实验不修改或继续训练模型。每张三退化输入分别选择一个退化 `A` 保留，并以对应的
单退化图作为参考目标；另外两个退化 `B、C` 同时移除。因此共有
`200 scenes × 2 triples × 3 preservation tasks = 1200` 个样本。

先用与双退化训练 coarse **完全相同**的 DACG checkpoint 生成 400 张三退化 coarse：

```bash
python -u prepare_cdd11_triple_coarse.py \
  --data-root /path/to/CDD11 \
  --dacg-checkpoint /path/to/cdd11-full-run/checkpoints/best_macro_psnr.pth \
  --coarse-root /path/to/CDD11-DACG-coarse-test-triple \
  --triple-combinations 1 2 \
  --device cuda
```

再评估已经训练好的选择性 DiFix。主实验固定采用 `preserve A, remove B and C`：

```bash
python -u evaluate_cdd11_difix.py \
  --data-root /path/to/CDD11 \
  --test-coarse-root /path/to/CDD11-DACG-coarse-test-triple \
  --checkpoint /path/to/difix-selective-run/checkpoints/best_psnr.pkl \
  --output-dir /path/to/difix-selective-run/test_triple_preserve_first \
  --triple-combinations 1 2 \
  --triple-prompt-template preserve-first \
  --resolution 512 \
  --workers 4 \
  --mixed-precision bf16 \
  --seed 42 \
  --enable-xformers-memory-efficient-attention
```

输出指标和断点续跑机制与 8.2 相同；`summary.csv` 分为 6 个选择性任务、2 个 triple、
micro 和 task-macro。正式结论优先看 task-macro，并同时报告 `final` 相对 `coarse` 的
提升和胜率。`final` 优于 `coarse` 说明选择性 DiFix 在 DACG 之后仍有增益；它并不
等同于证明整个 DACG+DiFix 流水线都未见过三退化数据：如果 DACG 按 CDD11 全 11 类
训练，三退化只对选择性 DiFix 阶段是 OOD。

若要继续验证“训练本身”是否在三退化上产生收益，以完全相同的三退化输入和 prompt
运行 seed-matched step-0 基线：

```bash
python -u evaluate_cdd11_difix.py \
  --model-source initialization \
  --data-root /path/to/CDD11 \
  --test-coarse-root /path/to/CDD11-DACG-coarse-test-triple \
  --output-dir /path/to/difix-selective-run/test_triple_initialization_seed42 \
  --triple-combinations 1 2 \
  --triple-prompt-template preserve-first \
  --resolution 512 \
  --workers 4 \
  --mixed-precision bf16 \
  --seed 42 \
  --enable-xformers-memory-efficient-attention

python -u compare_cdd11_difix.py \
  --trained-dir /path/to/difix-selective-run/test_triple_preserve_first \
  --initialization-dir /path/to/difix-selective-run/test_triple_initialization_seed42 \
  --output-dir /path/to/difix-selective-run/compare_triple_trained_vs_initialization
```

这里最有力的训练有效性证据是：`trained_advantage` 的 scene-cluster bootstrap 95% CI
下界大于 0，同时训练输出相对 DACG coarse 的 `improvement_*` 也为正。

可选的提示词对照只需换一个全新的输出目录，并增加
`--triple-prompt-template remove-first`。它生成训练格式更接近的
`remove B and C, preserve A`，用于区分三元组合 OOD 与提示词顺序变化；不要在看到
test 结果后选择表现更好的模板作为唯一主结果。
