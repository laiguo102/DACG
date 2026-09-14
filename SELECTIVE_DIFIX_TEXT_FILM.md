# SelectiveDifix B-20k Text-FiLM runbook

Text-FiLM is a zero-initialized delta on the trained B-20k DAEM-lite gate. The
UNet, VAE base, VAE LoRA, and original skip convolutions stay frozen in E1/E2.
Do not start E1 unless E0 passes, and do not start E2 unless the E1 semantic
controls pass.

## Server paths

```bash
export REPO_ROOT=/home/bml/.storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/all-in-one-model/DACG
export CCDD_DATA_ROOT=/home/bml/.storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/data/CCDD-11
export CCDD_COARSE_ROOT=/home/bml/.storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/data/CCDD11-DACG-coarse
export RUN_ROOT=/home/bml/.storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/all-in-one-model/experiments/ccdd11-difix-runs
export B20_CHECKPOINT="$RUN_ROOT/ccdd-daem-lite-detail-only-v1/detail-only-20k-extension-v1/checkpoints/best_psnr.pkl"
export B20_PAIRED_RESULTS="$RUN_ROOT/ccdd-daem-lite-detail-only-v1/detail-only-20k-extension-v1/paired_vs_detail_only_10k_v1/full"
export TORCH_HOME=/home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/model-cache/torch
export TEXT_FILM_ROOT="$RUN_ROOT/ccdd-daem-lite-text-film-v1"
cd "$REPO_ROOT"
```

## Tests and E0

```bash
python -m pytest -q \
  tests/test_daem_lite.py \
  tests/test_daem_checkpoint.py \
  tests/test_text_film_evaluate.py \
  tests/test_paired_validation.py \
  tests/test_difix3d_selective.py \
  tests/test_ccdd11_prepare.py

python -u verify_ccdd11_text_film_e0.py \
  --b20-checkpoint "$B20_CHECKPOINT" \
  --output-dir "$TEXT_FILM_ROOT/e0" \
  --samples 4 --device cuda:0 \
  --precisions no bf16 --gradient-precision bf16
```

`$TEXT_FILM_ROOT/e0/report.json` must report zero delta logits, no gradients in
the frozen B-20k modules, non-zero `delta_out` and second-step upstream text
gradients, successful checkpoint round-trip, fp32 max error below `1e-6`, and
bf16 max error below `2e-3`.

## E1-full

```bash
accelerate launch --mixed_precision=bf16 train_ccdd11_difix.py \
  --data-root "$CCDD_DATA_ROOT" \
  --coarse-root "$CCDD_COARSE_ROOT/half_train" \
  --output-dir "$TEXT_FILM_ROOT/e1-full" \
  --degradation-pairs 1 2 3 4 5 \
  --negative-train-probability 0.2 \
  --resolution 512 --max-train-steps 3000 \
  --train-batch-size 4 --dataloader-num-workers 8 \
  --detail-enabled --detail-num-blocks 1 --detail-gate-reduction 4 \
  --detail-text-mode full --detail-text-proj-dim 128 \
  --detail-film-hidden-ratio 4 --detail-text-condition-scale 1.0 \
  --train-scope film --text-learning-rate 1e-4 \
  --init-checkpoint "$B20_CHECKPOINT" \
  --lr-scheduler linear --lr-warmup-steps 100 \
  --lambda-l2 1 --lambda-lpips 1 --lambda-gram 0 \
  --eval-freq 250 --full-eval-freq 500 --viz-freq 500 \
  --latest-checkpointing-steps 500 --milestone-steps 1000 \
  --enable-xformers-memory-efficient-attention \
  --report-to wandb \
  --tracker-project-name difix-cdd11-selective-text-film \
  --tracker-run-name b20-text-film-e1-full
```

## E1-constant

Run the same budget and seed as E1-full; only the text mode and run names differ.

```bash
accelerate launch --mixed_precision=bf16 train_ccdd11_difix.py \
  --data-root "$CCDD_DATA_ROOT" \
  --coarse-root "$CCDD_COARSE_ROOT/half_train" \
  --output-dir "$TEXT_FILM_ROOT/e1-constant" \
  --degradation-pairs 1 2 3 4 5 \
  --negative-train-probability 0.2 \
  --resolution 512 --max-train-steps 3000 \
  --train-batch-size 4 --dataloader-num-workers 8 \
  --detail-enabled --detail-num-blocks 1 --detail-gate-reduction 4 \
  --detail-text-mode constant --detail-text-proj-dim 128 \
  --detail-film-hidden-ratio 4 --detail-text-condition-scale 1.0 \
  --train-scope film --text-learning-rate 1e-4 \
  --init-checkpoint "$B20_CHECKPOINT" \
  --lr-scheduler linear --lr-warmup-steps 100 \
  --lambda-l2 1 --lambda-lpips 1 --lambda-gram 0 \
  --eval-freq 250 --full-eval-freq 500 --viz-freq 500 \
  --latest-checkpointing-steps 500 --milestone-steps 1000 \
  --enable-xformers-memory-efficient-attention \
  --report-to wandb \
  --tracker-project-name difix-cdd11-selective-text-film \
  --tracker-run-name b20-text-film-e1-constant
```

## E1 evaluation

```bash
python -u evaluate_ccdd11_paired.py \
  --comparison-profile b20-vs-text-film \
  --reuse-baseline-results "$B20_PAIRED_RESULTS" \
  --reuse-baseline-role candidate \
  --baseline-label B20 --candidate-label E1_full \
  --baseline-checkpoint "$B20_CHECKPOINT" \
  --candidate-checkpoint "$TEXT_FILM_ROOT/e1-full/checkpoints/best_psnr.pkl" \
  --output-dir "$TEXT_FILM_ROOT/e1-full/paired_vs_b20_v1/full" \
  --workers 0 --device cuda --mixed-precision bf16 \
  --bootstrap-resamples 10000 --bootstrap-seed 42 \
  --enable-xformers-memory-efficient-attention

python -u evaluate_ccdd11_paired.py \
  --comparison-profile b20-vs-text-film \
  --reuse-baseline-results "$B20_PAIRED_RESULTS" \
  --reuse-baseline-role candidate \
  --baseline-label B20 --candidate-label E1_constant \
  --baseline-checkpoint "$B20_CHECKPOINT" \
  --candidate-checkpoint "$TEXT_FILM_ROOT/e1-constant/checkpoints/best_psnr.pkl" \
  --output-dir "$TEXT_FILM_ROOT/e1-constant/paired_vs_b20_v1/full" \
  --workers 0 --device cuda --mixed-precision bf16 \
  --bootstrap-resamples 10000 --bootstrap-seed 42 \
  --enable-xformers-memory-efficient-attention

python -u evaluate_ccdd11_text_film.py \
  --candidate-checkpoint "$TEXT_FILM_ROOT/e1-full/checkpoints/best_psnr.pkl" \
  --constant-checkpoint "$TEXT_FILM_ROOT/e1-constant/checkpoints/best_psnr.pkl" \
  --b20-paired-results "$B20_PAIRED_RESULTS" \
  --output-dir "$TEXT_FILM_ROOT/e1-full/text_semantics_v1/full" \
  --workers 0 --device cuda --mixed-precision bf16 \
  --bootstrap-resamples 10000 --bootstrap-seed 42 \
  --num-gallery-samples 20 \
  --enable-xformers-memory-efficient-attention
```

Proceed only if E1-full improves over B-20k or holds PSNR while improving the
perceptual metrics, beats the trained constant control, affects swapped-prompt
gates/output in the expected direction, keeps at least 7/10 directed tasks from
regressing, and loses no more than 0.05 dB identity PSNR.

## E2 hybrid

```bash
accelerate launch --mixed_precision=bf16 train_ccdd11_difix.py \
  --data-root "$CCDD_DATA_ROOT" \
  --coarse-root "$CCDD_COARSE_ROOT/half_train" \
  --output-dir "$TEXT_FILM_ROOT/e2-hybrid" \
  --degradation-pairs 1 2 3 4 5 \
  --negative-train-probability 0.2 \
  --resolution 512 --max-train-steps 5000 \
  --train-batch-size 4 --dataloader-num-workers 8 \
  --detail-enabled --detail-num-blocks 1 --detail-gate-reduction 4 \
  --detail-text-mode full --detail-text-proj-dim 128 \
  --detail-film-hidden-ratio 4 --detail-text-condition-scale 1.0 \
  --train-scope film+detail \
  --text-learning-rate 5e-5 --detail-learning-rate 1e-5 \
  --detail-alpha-learning-rate 5e-6 \
  --init-checkpoint "$TEXT_FILM_ROOT/e1-full/checkpoints/best_psnr.pkl" \
  --lr-scheduler linear --lr-warmup-steps 100 \
  --lambda-l2 1 --lambda-lpips 1 --lambda-gram 0 \
  --eval-freq 250 --full-eval-freq 1000 --viz-freq 500 \
  --latest-checkpointing-steps 500 --milestone-steps 1000 \
  --enable-xformers-memory-efficient-attention \
  --report-to wandb \
  --tracker-project-name difix-cdd11-selective-text-film \
  --tracker-run-name b20-text-film-e2-hybrid
```

Use `--init-checkpoint`, not `--resume`, between E1 and E2. `--resume` is only
for continuing an interrupted run with the same scope and optimizer groups.

After E2 completes, run the same two evaluators against B-20k:

```bash
python -u evaluate_ccdd11_paired.py \
  --comparison-profile b20-vs-text-film \
  --reuse-baseline-results "$B20_PAIRED_RESULTS" \
  --reuse-baseline-role candidate \
  --baseline-label B20 --candidate-label E2_hybrid \
  --baseline-checkpoint "$B20_CHECKPOINT" \
  --candidate-checkpoint "$TEXT_FILM_ROOT/e2-hybrid/checkpoints/best_psnr.pkl" \
  --output-dir "$TEXT_FILM_ROOT/e2-hybrid/paired_vs_b20_v1/full" \
  --workers 0 --device cuda --mixed-precision bf16 \
  --bootstrap-resamples 10000 --bootstrap-seed 42 \
  --enable-xformers-memory-efficient-attention

python -u evaluate_ccdd11_text_film.py \
  --candidate-checkpoint "$TEXT_FILM_ROOT/e2-hybrid/checkpoints/best_psnr.pkl" \
  --constant-checkpoint "$TEXT_FILM_ROOT/e1-constant/checkpoints/best_psnr.pkl" \
  --b20-paired-results "$B20_PAIRED_RESULTS" \
  --output-dir "$TEXT_FILM_ROOT/e2-hybrid/text_semantics_v1/full" \
  --workers 0 --device cuda --mixed-precision bf16 \
  --bootstrap-resamples 10000 --bootstrap-seed 42 \
  --num-gallery-samples 20 \
  --enable-xformers-memory-efficient-attention
```
