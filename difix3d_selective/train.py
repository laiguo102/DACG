"""Train selective Difix from precomputed DACG coarse images."""

from __future__ import annotations

import argparse
import gc
import json
import random
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn.functional as F
from .loss import gram_loss


VISUALIZATION_KEY = "validation/degraded_coarse_target_final_gt"
NEGATIVE_VISUALIZATION_KEY = (
    "validation_negative/double_degraded_dacg_signed_texture_final_abs_error"
)


def trigger_schedule(step: int, max_steps: int, frequency: int) -> bool:
    """Return whether a periodic action is due, always including the final step."""

    return step == max_steps or (frequency > 0 and step % frequency == 0)


def gram_is_enabled(weight: float, step: int, warmup_steps: int) -> bool:
    """Match released Difix3D: enable Gram only after its warmup phase."""

    return weight > 0 and step >= warmup_steps


def probability(value: str) -> float:
    """Parse a closed-interval probability for argparse."""

    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("probability must be a number") from error
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("probability must be in [0, 1]")
    return parsed


def _make_loader(dataset, *, batch_size: int, shuffle: bool, workers: int):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _best_state(output_dir: Path) -> dict[str, float | int]:
    path = output_dir / "best_validation.json"
    if not path.is_file():
        return {"global_step": 0, "psnr": float("-inf"), "ssim": float("-inf")}
    return json.loads(path.read_text(encoding="utf-8"))


def _experiment_metadata(args: argparse.Namespace) -> dict:
    dataset_format = getattr(args, "dataset_format", "cdd11")
    metadata = {
        "dataset": "CCDD-11" if dataset_format == "ccdd11" else "CDD-11",
        "target": (
            "native sub_data selective target"
            if dataset_format == "ccdd11"
            else "single-degradation preserve target"
        ),
        "seed": int(args.seed),
        "pairs": list(args.degradation_pairs),
    }
    if dataset_format == "ccdd11":
        metadata.update(
            {
                "negative_train_probability": float(
                    getattr(args, "negative_train_probability", 0.0)
                ),
                "training_mode_sampling": "dynamic_per_record_read",
                "negative_prompt": "preserve A, preserve B",
                "negative_condition": [
                    "original double-degradation image",
                    "signed original-minus-DACG texture",
                ],
                "negative_target": "original double-degradation image",
            }
        )
    return metadata


def _prepare_manifests(args: argparse.Namespace, output_dir: Path) -> dict:
    dataset_format = getattr(args, "dataset_format", "cdd11")
    if dataset_format == "ccdd11":
        from .ccdd_prepare import prepare_ccdd11_selective_manifests

        prepare = prepare_ccdd11_selective_manifests
    elif dataset_format == "cdd11":
        from .prepare import prepare_selective_manifests

        prepare = prepare_selective_manifests
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")
    prepared = prepare(
        data_root=args.data_root,
        coarse_root=args.coarse_root,
        output_dir=output_dir,
        pair_ids=args.degradation_pairs,
    )
    print(
        f"Indexed {prepared['coarse_images']} coarse images\n"
        f"train samples = {prepared['train_samples']}\n"
        f"validation samples = {prepared['validation_samples']}",
        flush=True,
    )
    return prepared


def _log_validation(
    accelerator,
    metrics: dict[str, float],
    *,
    prefix: str,
    step: int,
    visualizations=None,
    visualization_key: str = VISUALIZATION_KEY,
) -> None:
    from .validation import wandb_images

    payload = {f"{prefix}/{key}": value for key, value in metrics.items()}
    if visualizations and accelerator.is_main_process:
        payload[visualization_key] = wandb_images(visualizations)
    accelerator.log(payload, step=step)


def run(args: argparse.Namespace) -> None:
    import lpips
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs, set_seed
    from diffusers.optimization import get_scheduler
    from diffusers.utils.import_utils import is_xformers_available
    from torch.utils.data import Subset
    from torchvision import transforms
    from torchvision.transforms.functional import crop
    from tqdm.auto import tqdm

    from .data import SelectiveDifixDataset
    from .model import SelectiveDifix, load_training_checkpoint, save_training_checkpoint
    from .validation import (
        stratified_indices,
        validate,
        validate_negative,
        wandb_images,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=24))],
    )
    set_seed(args.seed)
    output_dir = args.output_dir.resolve()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        _prepare_manifests(args, output_dir)
    accelerator.wait_for_everyone()
    train_manifest = output_dir / "prepared" / "manifests" / "train.jsonl"
    validation_manifest = output_dir / "prepared" / "manifests" / "validation.jsonl"

    with accelerator.main_process_first():
        model = SelectiveDifix(lora_rank_vae=args.lora_rank_vae, timestep=args.timestep)
        perceptual_model = lpips.LPIPS(net="vgg").eval().requires_grad_(False)
        vgg = None
        if args.lambda_gram > 0:
            import torchvision

            vgg = torchvision.models.vgg16(
                weights=torchvision.models.VGG16_Weights.DEFAULT
            ).features.eval().requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if not is_xformers_available():
            raise RuntimeError("xformers is not installed")
        model.unet.enable_xformers_memory_efficient_attention()
    if args.gradient_checkpointing:
        model.unet.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    trainable_parameters = model.trainable_parameters()
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    if args.resume is not None:
        global_step = load_training_checkpoint(
            model, optimizer, lr_scheduler, args.resume.resolve()
        )

    negative_validation_enabled = (
        args.dataset_format == "ccdd11"
        and hasattr(args, "negative_train_probability")
    )
    negative_train_probability = (
        float(args.negative_train_probability) if negative_validation_enabled else 0.0
    )
    train_dataset = SelectiveDifixDataset(
        train_manifest,
        model.tokenizer,
        resolution=args.resolution,
        negative_probability=negative_train_probability,
    )
    validation_dataset = SelectiveDifixDataset(
        validation_manifest,
        model.tokenizer,
        resolution=args.resolution,
        training_mode="positive",
    )
    fast_indices = stratified_indices(
        validation_dataset.records, args.num_validation_samples
    )
    visualization_indices = stratified_indices(
        validation_dataset.records, args.num_validation_visualizations
    )
    train_loader = _make_loader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        workers=args.dataloader_num_workers,
    )
    fast_validation_loader = _make_loader(
        Subset(validation_dataset, fast_indices), batch_size=1, shuffle=False,
        workers=min(args.dataloader_num_workers, 4),
    )
    full_validation_loader = _make_loader(
        validation_dataset, batch_size=1, shuffle=False,
        workers=min(args.dataloader_num_workers, 4),
    )
    visualization_loader = _make_loader(
        Subset(validation_dataset, visualization_indices), batch_size=1, shuffle=False,
        workers=min(args.dataloader_num_workers, 4),
    )

    negative_fast_validation_loader = None
    negative_full_validation_loader = None
    negative_visualization_loader = None
    if negative_validation_enabled:
        negative_validation_dataset = SelectiveDifixDataset(
            validation_manifest,
            model.tokenizer,
            resolution=args.resolution,
            training_mode="negative",
            deduplicate_negative=True,
        )
        negative_fast_indices = stratified_indices(
            negative_validation_dataset.records, args.num_validation_samples
        )
        negative_visualization_indices = stratified_indices(
            negative_validation_dataset.records, args.num_validation_visualizations
        )
        negative_fast_validation_loader = _make_loader(
            Subset(negative_validation_dataset, negative_fast_indices),
            batch_size=1,
            shuffle=False,
            workers=min(args.dataloader_num_workers, 4),
        )
        negative_full_validation_loader = _make_loader(
            negative_validation_dataset,
            batch_size=1,
            shuffle=False,
            workers=min(args.dataloader_num_workers, 4),
        )
        negative_visualization_loader = _make_loader(
            Subset(negative_validation_dataset, negative_visualization_indices),
            batch_size=1,
            shuffle=False,
            workers=min(args.dataloader_num_workers, 4),
        )

    if negative_validation_enabled:
        (
            model,
            optimizer,
            train_loader,
            fast_validation_loader,
            full_validation_loader,
            visualization_loader,
            negative_fast_validation_loader,
            negative_full_validation_loader,
            negative_visualization_loader,
            lr_scheduler,
        ) = accelerator.prepare(
            model,
            optimizer,
            train_loader,
            fast_validation_loader,
            full_validation_loader,
            visualization_loader,
            negative_fast_validation_loader,
            negative_full_validation_loader,
            negative_visualization_loader,
            lr_scheduler,
        )
    else:
        (
            model,
            optimizer,
            train_loader,
            fast_validation_loader,
            full_validation_loader,
            visualization_loader,
            lr_scheduler,
        ) = accelerator.prepare(
            model,
            optimizer,
            train_loader,
            fast_validation_loader,
            full_validation_loader,
            visualization_loader,
            lr_scheduler,
        )
    perceptual_model = perceptual_model.to(accelerator.device)
    if vgg is not None:
        vgg = vgg.to(accelerator.device)
    renormalize_for_vgg = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )

    if args.report_to != "none":
        tracker_config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        accelerator.init_trackers(
            args.tracker_project_name,
            config=tracker_config,
            init_kwargs={"wandb": {"name": args.tracker_run_name, "dir": str(output_dir)}},
        )
        if args.report_to == "wandb" and accelerator.is_main_process:
            run_tracker = accelerator.get_tracker("wandb", unwrap=True)
            run_tracker.define_metric("validation_full/psnr", summary="max")
            run_tracker.define_metric("validation_full/ssim", summary="max")
            if negative_validation_enabled:
                run_tracker.define_metric(
                    "validation_negative_full/psnr", summary="max"
                )
                run_tracker.define_metric(
                    "validation_negative_full/ssim", summary="max"
                )
                run_tracker.define_metric(
                    "validation_negative_full/mean_absolute_change", summary="min"
                )

    best = _best_state(output_dir) if accelerator.is_main_process else None
    experiment_metadata = _experiment_metadata(args)
    progress = tqdm(
        total=args.max_train_steps,
        initial=global_step,
        desc="Difix steps",
        disable=not accelerator.is_local_main_process,
    )
    train_iterator = iter(train_loader)
    negative_samples_since_sync = 0
    training_samples_since_sync = 0
    while global_step < args.max_train_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        source = batch["conditioning_pixel_values"]
        target = batch["output_pixel_values"]
        negative_samples_since_sync += int(batch["is_negative"].sum().item())
        training_samples_since_sync += int(batch["is_negative"].numel())
        with accelerator.accumulate(model):
            prediction = model(source, prompt_tokens=batch["input_ids"])
            loss_l2 = F.mse_loss(prediction.float(), target.float()) * args.lambda_l2
            loss_lpips = (
                perceptual_model(prediction.float(), target.float()).mean()
                * args.lambda_lpips
            )
            loss_gram = torch.zeros((), device=accelerator.device)
            loss = loss_l2 + loss_lpips
            if gram_is_enabled(
                args.lambda_gram, global_step, args.gram_loss_warmup_steps
            ):
                prediction_vgg = renormalize_for_vgg(prediction * 0.5 + 0.5)
                target_vgg = renormalize_for_vgg(target * 0.5 + 0.5)
                crop_size = min(400, prediction.shape[-2], prediction.shape[-1])
                top = random.randint(0, prediction.shape[-2] - crop_size)
                left = random.randint(0, prediction.shape[-1] - crop_size)
                prediction_vgg = crop(prediction_vgg, top, left, crop_size, crop_size)
                target_vgg = crop(target_vgg, top, left, crop_size, crop_size)
                loss_gram = gram_loss(prediction_vgg, target_vgg, vgg) * args.lambda_gram
                loss = loss + loss_gram

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if not accelerator.sync_gradients:
            continue
        global_step += 1
        progress.update(1)
        local_mode_counts = torch.tensor(
            [negative_samples_since_sync, training_samples_since_sync],
            device=accelerator.device,
            dtype=torch.long,
        )
        global_mode_counts = accelerator.gather(local_mode_counts).reshape(-1, 2).sum(0)
        realized_negative_fraction = float(
            global_mode_counts[0].float().div(global_mode_counts[1]).item()
        )
        negative_samples_since_sync = 0
        training_samples_since_sync = 0
        logs = {
            "train/l2": float(loss_l2.detach()),
            "train/lpips": float(loss_lpips.detach()),
            "train/loss_l2": float(loss_l2.detach()),
            "train/loss_lpips": float(loss_lpips.detach()),
            "train/gram": float(loss_gram.detach()),
            "train/loss": float(loss.detach()),
            "train/learning_rate": lr_scheduler.get_last_lr()[0],
            "train/negative_fraction": realized_negative_fraction,
        }
        progress.set_postfix(loss=logs["train/loss"])
        if args.report_to != "none":
            accelerator.log(logs, step=global_step)

        fast_due = trigger_schedule(global_step, args.max_train_steps, args.eval_freq)
        full_due = trigger_schedule(global_step, args.max_train_steps, args.full_eval_freq)
        visualization_due = trigger_schedule(global_step, args.max_train_steps, args.viz_freq)
        visualizations = []
        negative_visualizations = []
        if fast_due:
            fast_metrics, visualizations = validate(
                model,
                fast_validation_loader,
                perceptual_model,
                accelerator,
                visualization_limit=(
                    args.num_validation_visualizations if visualization_due else 0
                ),
            )
            _log_validation(
                accelerator,
                fast_metrics,
                prefix="validation",
                step=global_step,
                visualizations=visualizations,
            )
            if accelerator.is_main_process:
                print(f"validation step={global_step}: {fast_metrics}", flush=True)
            if negative_validation_enabled:
                negative_fast_metrics, negative_visualizations = validate_negative(
                    model,
                    negative_fast_validation_loader,
                    perceptual_model,
                    accelerator,
                    visualization_limit=(
                        args.num_validation_visualizations if visualization_due else 0
                    ),
                )
                _log_validation(
                    accelerator,
                    negative_fast_metrics,
                    prefix="validation_negative",
                    step=global_step,
                    visualizations=negative_visualizations,
                    visualization_key=NEGATIVE_VISUALIZATION_KEY,
                )
                if accelerator.is_main_process:
                    print(
                        f"negative validation step={global_step}: "
                        f"{negative_fast_metrics}",
                        flush=True,
                    )

        if visualization_due and not visualizations:
            _, visualizations = validate(
                model,
                visualization_loader,
                perceptual_model,
                accelerator,
                visualization_limit=args.num_validation_visualizations,
            )
            if args.report_to != "none" and accelerator.is_main_process:
                accelerator.log(
                    {VISUALIZATION_KEY: wandb_images(visualizations)}, step=global_step
                )

        if (
            negative_validation_enabled
            and visualization_due
            and not negative_visualizations
        ):
            _, negative_visualizations = validate_negative(
                model,
                negative_visualization_loader,
                perceptual_model,
                accelerator,
                visualization_limit=args.num_validation_visualizations,
            )
            if args.report_to != "none" and accelerator.is_main_process:
                accelerator.log(
                    {
                        NEGATIVE_VISUALIZATION_KEY: wandb_images(
                            negative_visualizations
                        )
                    },
                    step=global_step,
                )

        if full_due:
            full_metrics, _ = validate(
                model, full_validation_loader, perceptual_model, accelerator
            )
            _log_validation(
                accelerator, full_metrics, prefix="validation_full", step=global_step
            )
            if accelerator.is_main_process:
                print(f"full validation step={global_step}: {full_metrics}", flush=True)
                if full_metrics["psnr"] > float(best["psnr"]):
                    best = {
                        "global_step": global_step,
                        "psnr": full_metrics["psnr"],
                        "ssim": full_metrics["ssim"],
                    }
                    checkpoint_dir = output_dir / "checkpoints"
                    save_training_checkpoint(
                        accelerator.unwrap_model(model), optimizer, lr_scheduler,
                        checkpoint_dir / "best_psnr.pkl", global_step,
                        experiment_metadata,
                    )
                    _write_json(output_dir / "best_validation.json", best)
            if negative_validation_enabled:
                negative_full_metrics, _ = validate_negative(
                    model,
                    negative_full_validation_loader,
                    perceptual_model,
                    accelerator,
                )
                _log_validation(
                    accelerator,
                    negative_full_metrics,
                    prefix="validation_negative_full",
                    step=global_step,
                )
                if accelerator.is_main_process:
                    print(
                        f"negative full validation step={global_step}: "
                        f"{negative_full_metrics}",
                        flush=True,
                    )

        if fast_due or full_due or visualization_due:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        permanent_frequency = args.checkpointing_steps or args.milestone_steps
        latest_due = trigger_schedule(
            global_step, args.max_train_steps, args.latest_checkpointing_steps
        )
        milestone_due = trigger_schedule(
            global_step, args.max_train_steps, permanent_frequency
        )
        if accelerator.is_main_process and (latest_due or milestone_due):
            checkpoint_dir = output_dir / "checkpoints"
            unwrapped = accelerator.unwrap_model(model)
            if latest_due:
                save_training_checkpoint(
                    unwrapped, optimizer, lr_scheduler,
                    checkpoint_dir / "latest.pkl", global_step,
                    experiment_metadata,
                )
            if milestone_due:
                save_training_checkpoint(
                    unwrapped, optimizer, lr_scheduler,
                    checkpoint_dir / f"model_{global_step:06d}.pkl", global_step,
                    experiment_metadata,
                )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_training_checkpoint(
            accelerator.unwrap_model(model), optimizer, lr_scheduler,
            output_dir / "checkpoints" / "final.pkl", global_step,
            experiment_metadata,
        )
    accelerator.end_training()


def parser(default_dataset_format: str = "cdd11") -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--coarse-root", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument(
        "--dataset-format", choices=("cdd11", "ccdd11"),
        default=default_dataset_format,
    )
    value.add_argument(
        "--prepare-only", action="store_true",
        help="Build and validate manifests without loading Difix or perceptual models.",
    )
    value.add_argument(
        "--degradation-pairs", nargs="+", type=int, choices=range(1, 6), required=True
    )
    if default_dataset_format == "ccdd11":
        value.add_argument(
            "--negative-train-probability",
            type=probability,
            default=0.2,
            help=(
                "Probability of dynamically replacing a directed positive CCDD-11 "
                "record with a preserve-both negative record on each read."
            ),
        )
    value.add_argument("--resolution", type=int, default=512)
    value.add_argument("--max-train-steps", type=int, default=100_000)
    value.add_argument("--train-batch-size", type=int, default=4)
    value.add_argument("--gradient-accumulation-steps", type=int, default=1)
    value.add_argument("--dataloader-num-workers", type=int, default=8)
    value.add_argument("--learning-rate", type=float, default=5e-6)
    value.add_argument("--lr-scheduler", default="linear")
    value.add_argument("--lr-warmup-steps", type=int, default=500)
    value.add_argument("--adam-beta1", type=float, default=0.9)
    value.add_argument("--adam-beta2", type=float, default=0.999)
    value.add_argument("--adam-weight-decay", type=float, default=1e-2)
    value.add_argument("--adam-epsilon", type=float, default=1e-8)
    value.add_argument("--max-grad-norm", type=float, default=1.0)
    value.add_argument("--lambda-l2", type=float, default=1.0)
    value.add_argument("--lambda-lpips", type=float, default=1.0)
    value.add_argument("--lambda-gram", type=float, default=0.0)
    value.add_argument("--gram-loss-warmup-steps", type=int, default=2000)
    value.add_argument("--lora-rank-vae", type=int, default=4)
    value.add_argument("--timestep", type=int, default=199)
    value.add_argument(
        "--checkpointing-steps", type=int,
        help="Compatibility override for permanent checkpoint frequency.",
    )
    value.add_argument("--latest-checkpointing-steps", type=int, default=1000)
    value.add_argument("--milestone-steps", type=int, default=10_000)
    value.add_argument("--eval-freq", type=int, default=500)
    value.add_argument("--full-eval-freq", type=int, default=5000)
    value.add_argument("--viz-freq", type=int, default=1000)
    value.add_argument("--num-validation-samples", type=int, default=100)
    value.add_argument("--num-validation-visualizations", type=int, default=10)
    value.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    value.add_argument("--gradient-checkpointing", action="store_true")
    value.add_argument("--enable-xformers-memory-efficient-attention", action="store_true")
    value.add_argument("--allow-tf32", action="store_true")
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--resume", type=Path)
    value.add_argument("--report-to", choices=("wandb", "none"), default="wandb")
    value.add_argument("--tracker-project-name", default="difix-cdd11-selective")
    value.add_argument("--tracker-run-name", default="difix-selective-lucid")
    return value


def main(default_dataset_format: str = "cdd11") -> None:
    args = parser(default_dataset_format).parse_args()
    if args.prepare_only:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        _prepare_manifests(args, output_dir)
        return
    run(args)


if __name__ == "__main__":
    main()
