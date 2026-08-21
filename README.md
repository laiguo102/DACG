# Degradation-Aware Adaptive Context Gating for Unified Image Restoration (DACG-IR)

**Authors:** Lei He, Jielei Chu*, Fengmao Lv, Weide Liu, Tianrui Li, Jun Cheng, Yuming Fang  
**\* Corresponding Author**

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2605.01236)

<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Unified image restoration aims to handle diverse degradation types using a single model. However, the significant variability across different degradations often leads to severe task interference and suboptimal performance. Existing methods often struggle to balance task-specific discriminability with inter-task generalization, leading to either negative interference in complex environments or suboptimal performance on specific degradations. To overcome these challenges, we propose a Degradation-Aware Adaptive Context Gating (DACG-IR), which enables the restoration model to explicitly perceive degradation characteristics and dynamically modulate feature representations conditioned on the input image. The core idea is to construct degradation-aware contextual representations directly from the input image and utilize them to modulate attention distribution, frequency-domain modulation, and feature aggregation throughout the model. This design enables the model to suppress degradation-induced noise and interference while preserving informative image structures. Specifically, we design a lightweight multi-scale degradation-aware module to extract coarse degradation information and generate layer-wise degradation prompts, which guide the attention temperature and attention output gating in different blocks of the encoder and decoder, enabling adaptive feature extraction and fusion across scales. The generated global feature prompts is further used to dynamically modulate high-dimensional latent features. Furthermore, a spatial-channel dual-gated adaptive fusion mechanism is designed to refine encoder features and suppress the propagation of noise or irrelevant background information from shallow layers to deeper representations, thereby promoting high-fidelity reconstruction in the decoder. Extensive experiments on multiple benchmark datasets show that DACG-IR consistently outperforms state-of-the-art image restoration methods under single-task, all-in-one, adverse weather removal, and composite degradation settings. 
</details>



# 🏗️DACG-IR Architecture 

![](fig/arch.png)

# 🛠️Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/HlHomes/DACG-IR-code.git
    cd DACG-IR-code
    ```

2.  **Create a Conda environment:**
    ```bash
    ENV_NAME="DACG-IR"
    conda create -n $ENV_NAME python==3.10
    conda activate $ENV_NAME
    ```

3.  **Install dependencies:**
    ```bash
    bash install.sh
    ```

# ⚙️Data Preparation

## All-in-One Dataset

All the datasets for 5 tasks used in the paper can be downloaded from the following locations:

Denoising: [BSD400](https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view?usp=sharing), [WED](https://drive.google.com/file/d/1e62XGdi5c6IbvkZ70LFq0KLRhFvih7US/view?usp=sharing), [Urban100](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u), [Kodak24](https://r0k.us/graphics/kodak/), [BSD68](https://github.com/clausmichele/CBSD68-dataset/tree/master/CBSD68/original)

Deraining: [Train100L&Rain100L](https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f?usp=sharing)

Dehazing: Train[ RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2), Test [SOTS-Outdoor](https://sites.google.com/view/reside-dehaze-datasets/reside-v0)

Deblur: [GoPro](https://drive.google.com/file/d/1y_wQ5G5B65HS_mdIjxKYTcnRys_AGh5v/view?usp=sharing)

Low-light Enhancement: [LOL-V1](https://daooshee.github.io/BMVC2018website/)

The training data should be placed in ``` data/Train/{task_name}``` directory where ```task_name``` can be Denoise, Derain, Dehaze, Deblur, or Enhance.

## Single-Degradation Dataset

Deraining:  [Train100L&Rain100L](https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f?usp=sharing) , [SPAData](https://pan.baidu.com/s/1lPn3MWckHxh1uBYYucoWVQ?pwd=4fwo)

Dehazing: [ RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2)

Desnowing: [ Snow100K]( https://sites.google.com/view/yunfuliu/desnownet), [SRRS](https://drive.google.com/file/d/1GX3e0ziUzBXnDtgeB5sHkgeWP1wUg5td/view?usp=sharing)

Deblur: [GoPro](https://drive.google.com/file/d/1y_wQ5G5B65HS_mdIjxKYTcnRys_AGh5v/view?usp=sharing)

Denoising: [BSD400](https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view?usp=sharing), [WED](https://drive.google.com/file/d/1e62XGdi5c6IbvkZ70LFq0KLRhFvih7US/view?usp=sharing), [Urban100](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u), [Kodak24](https://r0k.us/graphics/kodak/), [BSD68](https://github.com/clausmichele/CBSD68-dataset/tree/master/CBSD68/original)

Low-light Enhancement: [LOL-V1](https://daooshee.github.io/BMVC2018website/), [LOL-V2](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view), [LOL-Blur](https://github.com/sczhou/LEDNet)

## Multi-Weather Dataset

Train: [Allweather](https://github.com/jeya-maria-jose/TransWeather)

Test: [Rain+fog](https://github.com/liruoteng/HeavyRainRemoval), [Snow100K-L]( https://sites.google.com/view/yunfuliu/desnownet), [Raindrop](https://github.com/rui1996/DeRaindrop)

## Composite Degradations

[CDD11](https://1drv.ms/f/s!As3rCDROnrbLgqpezG4sao-u9ddDhw?e=A0REHx)

# 🔍 Results

<br>

<details>
  <summary>
  <font>Three-task All-in-One Restoration.</font>
  </summary>
  <p align="center">
  <img src = "fig/result_task_3.png">
  </p>
</details>

<br>

<details>
  <summary>
  <font> Five-task All-in-One Restoration.</font>
  </summary>
  <p align="center">
  <img src = "fig/result_task_5.png">
  </p>
</details>

<br>

<details>
  <summary>
  <font> Mul-Weather Restoration.</font>
  </summary>
  <p align="center">
  <img src = "fig/result_mul_weather.png">
  </p>
</details>

  <br>

<details>
  <summary>
  <font> Composite Restoration.</font>
  </summary>
  <p align="center">
  <img src = "fig/result_composite.png">
  </p>
</details>

# 🧪Testing

Performance results are generated using the `src/test.py` script. 
*   `--model`: Set to `DACG-IR` or `DACG-IR-S`.
*   `--benchmarks`: Accepts a list of strings to iterate over defined test sets.
*   `--checkpoint_id`: Path to the pre-trained model.
*   `--de_type`: Denotes degradation type.
*   `--data_file_dir`: Your dataset directory.

## All-in-One Testing

**Three Tasks:**
```bash
python src/test.py --model "model" --benchmarks benchmarks --checkpoint_id "checkpoint_id" --de_type denoise_15 denoise_25 denoise_50 dehaze derain --data_file_dir "data_file_dir"
```

**Five Tasks:**
```bash
python src/test.py --model "$model" --benchmarks $benchmarks --checkpoint_id "${checkpoint_id}" --de_type denoise_15 denoise_25 denoise_50 dehaze derain deblur synllie --data_file_dir "data_file_dir"
```

## Multi-Weather and Signal Degradation
```bash
python src/test.py --dataset "dataset"
```
*(For specific settings, please refer to `./test.py`)*

## CDD11: Composited Degradations
Replace `[DEG_CONFIG]` with the desired configuration:
*   **Single:** `low`, `haze`, `rain`, `snow`
*   **Double:** `low_haze`, `low_rain`, `low_snow`, `haze_rain`, `haze_snow`
*   **Triple:** `low_haze_rain`, `low_haze_snow`

```bash
python src/test.py --model [MODEL] --checkpoint_id [MODEL]_CDD11 --trainset CDD11_[DEG_CONFIG] --benchmarks cdd11 --de_type denoise_15 denoise_25 denoise_50 dehaze derain deblur synllie --data_file_dir "data_file_dir"
```

# 🚀 Training

You can train the lightweight (`DACG-IR-S`) or heavy (`DACG-IR`) versions on three or five degradations.
*   `--gpus`: Specify number of GPUs (`1` for single, `>1` for multiple).
*   `--batch_size`: Defines batch size **per GPU**.

All-in-One Training

```bash
python src/train.py --model model --batch_size 8 --de_type synllie --trainset standard --num_gpus 4 --data_file_dir data_file_dir
```

### Multi-Weather or Signal Degradation Training
```bash
python src/train.py --dataset allweather --batch_size 8 --num_gpus 4 --data_file_dir data_file_dir
```

### CDD11: Composited Degradations Training
Train from scratch on the [CDD11](https://github.com/gy65896/OneRestore) dataset:
*   `CDD_single`: Low light (L), Haze (H), Rain (R), Snow (S)
*   `CDD_double`: L+H, L+R, L+S, H+R, H+S
*   `CDD_triple`: L+H+R, L+H+S
*   `--trainset CDD_all`: Combines Single + Double + Triple

```bash
python src/train.py --model model_s --batch_size 8 --de_type denoise_15 denoise_25 denoise_50 dehaze derain --trainset CDD11_all --num_gpus 4 --data_file_dir data_file_dir
```

## AIO3-v1 frozen-manifest input and formal output

The original `src/train.py` and `src/test.py` reproduce the repository's legacy
experiments. Protocol-comparable AIO3-v1 experiments use the separate
`aio3_runner` package so the legacy data and metric behavior cannot be selected
accidentally.

The complete no-environment-variable server workflow is documented in
[`AIO3_SERVER_GUIDE.md`](AIO3_SERVER_GUIDE.md).

Verify the shared manifests before a run:

```bash
python -m aio3_runner.verify_data --manifest-dir /path/to/outputs/AIO3/aio3-v1/manifests
```

Use `aio3_runner.data.make_training_loader` (or `AIO3ManifestDataset` together
with `BalancedTaskBatchSampler`) for training.
It reads JSONL records directly, creates deterministic 128 x 128 synchronized
patches, generates Gaussian denoising inputs online, and emits every optimizer
batch as exactly `4 denoise + 4 derain + 4 dehaze`. Recreate the sampler with
`start_step=<checkpoint global_step>` to reproduce the next batch after resume.

The DACG-IR AIO3 adapter is `aio3_runner.adapter.build_model`. It preserves the
input's exact spatial dimensions and returns an unclamped raw restoration.

The protocol runner is launched with `python -m aio3_runner.train`. It owns the
fixed L1/AdamW/warmup-cosine training loop, validation selection, atomic
checkpoints, exact resume state, W&B logging, and local JSONL records. Supported
run kinds are `smoke`, `pilot`, and `formal`; their lengths and intervals are
frozen by AIO3-v1 rather than exposed as tunable arguments.

After a completed formal run, evaluate only its selected validation checkpoint:

```bash
python -m aio3_runner.evaluate \
  --checkpoint /path/to/run/checkpoints/best_macro_psnr.pth \
  --data-root /path/to/data/AIO3 \
  --num-workers 4
```

The evaluator validates the frozen manifest hashes and formal run state, performs
native-resolution batch-size-1 inference, uses the protocol RGB PSNR/SSIM, and
creates `test/state.json`, both summary tables, per-image metrics, 804 uniquely
named predictions, and the deterministic 14-sample/70-image gallery. It refuses
to overwrite an existing `test` directory.

## DACG-conditioned Difix3D for CDD11

This branch adds a two-view, degradation-conditioned Difix3D pipeline for CDD11:

```text
[DACG coarse restoration, original degraded reference] -> clear target
```

The frozen DACG degradation-aware module extracts `P_global` from the reference
view. It conditions trainable UNet and VAE-decoder LoRA rank matrices, while the
SD-Turbo bases, text encoder, and DAM remain frozen. Both views participate in
the multi-view UNet, but only view 0 is decoded and only view-0 VAE encoder skips
are used. The prompt is fixed to `remove degradation`.

Install the pinned optional environment with `requirements_difix.txt`. The full
server workflow, including coarse generation, smoke training, resume, ablations,
cascade inference, and CDD11 evaluation, is in
[`CDD11_DACG_DIFIX_SERVER_GUIDE.md`](CDD11_DACG_DIFIX_SERVER_GUIDE.md).

The Difix3D-derived components are based on upstream commit
`c76edc595586e16732c91ddee82f3a6d83a8a9cc`. See `dacg_difix/UPSTREAM.md` and
`dacg_difix/LICENSE_DIFIX3D.txt`. Those components remain subject to NVIDIA's
non-commercial research/evaluation license.

## CDD-11-v1 DACG training and test

The CDD-11 runner trains the original full `DACG_IR` network under the same
frozen data split, one-per-degradation batch sampler, optimizer, schedule,
validation, checkpoint, native-resolution inference, and W&B conventions used
for the UNet/Uformer comparison. It reuses the existing cdd11-v1 manifests:
1083 train scenes, 100 validation scenes, and the untouched 200-scene official
test split. OOF training is not part of this workflow.

The deliberate protocol exception is the objective: DACG retains its paper
RGB L1 + 0.1 Fourier L1 loss. Every config and result records the objective as
`dacg-paper-rgb-fourier-l1`; therefore comparisons with RGB-L1-only models must
be described as architecture-plus-native-objective comparisons. See
[`CDD11_DACG_SERVER_GUIDE.md`](CDD11_DACG_SERVER_GUIDE.md) for audited commands.

```bash
python verify_cdd11.py --manifest-dir /path/to/cdd11-v1-manifests

python train_cdd11.py \
  --manifest-dir /path/to/cdd11-v1-manifests \
  --output-root /path/to/experiments/cdd11-v1-runs \
  --run-kind formal \
  --run-name dacg-formal-native-seed3407-v1 \
  --wandb-mode online \
  --wandb-entity YOUR_ENTITY
```

# 📋Acknowledgements

This code is built upon:

*   [PromptIR](https://github.com/va1shn9v/PromptIR)
*   [AirNet](https://github.com/XLearning-SCU/2022-CVPR-AirNet)
*   [MoCE-IR](https://github.com/eduardzamfir/MoCE-IR)
*   [Restormer](https://github.com/swz30/Restormer)

# 📖 Citation

```
@misc{he2026degradationawareadaptivecontextgating,
      title={Degradation-Aware Adaptive Context Gating for Unified Image Restoration}, 
      author={Lei He and Jielei Chu and Fengmao Lv and Weide Liu and Tianrui Li and Jun Cheng and Yuming Fang},
      year={2026},
      eprint={2605.01236},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.01236}, 
}
```

