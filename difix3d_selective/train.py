"""Train selective Difix from precomputed DACG coarse images."""

from __future__ import annotations

import argparse
import gc
import random
from datetime import timedelta
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F
import torchvision
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm

from cdd11_full.metrics import rgb_psnr, rgb_ssim

from .data import SelectiveDifixDataset
from .loss import gram_loss
from .model import (
    SelectiveDifix,
    load_training_checkpoint,
    save_training_checkpoint,
)
from .prepare import prepare_selective_manifests


def _display(value: torch.Tensor) -> torch.Tensor:
    return (value.detach().float().cpu() * 0.5 + 0.5).clamp(0, 1)


def _comparison_image(
    source: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor
) -> torch.Tensor:
    """Join degraded, DACG coarse, Difix output and GT from left to right."""

    degraded = _display(source[0, 1])
    coarse = _display(source[0, 0])
    final = _display(prediction[0])
    ground_truth = _display(target[0])
    return torch.cat((degraded, coarse, final, ground_truth), dim=-1)


def _validation(
    model,
    loader,
    lpips_model,
    accelerator,
    limit: int,
    num_visualizations: int = 0,
) -> tuple[dict[str, float], list[tuple[torch.Tensor, str]]]:
    model.eval()
    psnr_values, ssim_values, lpips_values = [], [], []
    visualizations: list[tuple[torch.Tensor, str]] = []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= limit:
                break
            target = batch["output_pixel_values"]
            prediction = model(
                batch["conditioning_pixel_values"],
                prompt_tokens=batch["input_ids"],
            )
            prediction_01 = prediction.float().add(1).mul(0.5).clamp(0, 1)
            target_01 = target.float().add(1).mul(0.5).clamp(0, 1)
            psnr = prediction.new_tensor([rgb_psnr(prediction_01, target_01)]).float()
            ssim = prediction.new_tensor([rgb_ssim(prediction_01, target_01)]).float()
            perceptual = lpips_model(prediction.float(), target.float()).flatten()
            psnr_values.append(accelerator.gather_for_metrics(psnr))
            ssim_values.append(accelerator.gather_for_metrics(ssim))
            lpips_values.append(accelerator.gather_for_metrics(perceptual))
            if accelerator.is_main_process and len(visualizations) < num_visualizations:
                caption = (
                    f"{batch['sample_id'][0]} | {batch['prompt'][0]} | "
                    "left to right: degraded | DACG coarse | Difix final | GT"
                )
                visualizations.append((
                    _comparison_image(batch["conditioning_pixel_values"], target, prediction),
                    caption,
                ))
    accelerator.unwrap_model(model).set_train()
    metrics = {
        "psnr": float(torch.cat(psnr_values).mean()),
        "ssim": float(torch.cat(ssim_values).mean()),
        "lpips": float(torch.cat(lpips_values).mean()),
    }
    return metrics, visualizations


def _wandb_validation_images(visualizations):
    import wandb

    return [wandb.Image(image, caption=caption) for image, caption in visualizations]


def run(args: argparse.Namespace) -> None:
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=None if args.report_to == "none" else args.report_to,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=24))],
    )
    set_seed(args.seed)
    output_dir = args.output_dir.resolve()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        prepared = prepare_selective_manifests(
            data_root=args.data_root,
            coarse_root=args.coarse_root,
            output_dir=output_dir,
            pair_ids=args.degradation_pairs,
        )
        print(
            f"Indexed {prepared['coarse_images']} coarse images, "
            f"wrote {prepared['train_samples']} train and {prepared['validation_samples']} validation samples",
            flush=True,
        )
    accelerator.wait_for_everyone()
    train_manifest = output_dir / "prepared" / "manifests" / "train.jsonl"
    validation_manifest = output_dir / "prepared" / "manifests" / "validation.jsonl"

    with accelerator.main_process_first():
        model = SelectiveDifix(lora_rank_vae=args.lora_rank_vae, timestep=args.timestep)
        perceptual_model = lpips.LPIPS(net="vgg").eval().requires_grad_(False)
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

    train_dataset = SelectiveDifixDataset(
        train_manifest, model.tokenizer, resolution=args.resolution
    )
    validation_dataset = SelectiveDifixDataset(
        validation_manifest, model.tokenizer, resolution=args.resolution
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=min(args.dataloader_num_workers, 4),
        pin_memory=True,
    )

    model, optimizer, train_loader, validation_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, validation_loader, lr_scheduler
    )
    perceptual_model = perceptual_model.to(accelerator.device)
    vgg = vgg.to(accelerator.device)
    renormalize_for_vgg = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )

    if accelerator.is_main_process and args.report_to != "none":
        tracker_config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        accelerator.init_trackers(
            args.tracker_project_name,
            config=tracker_config,
            init_kwargs={"wandb": {"name": args.tracker_run_name, "dir": str(output_dir)}},
        )

    progress = tqdm(
        total=args.max_train_steps,
        initial=global_step,
        desc="Difix steps",
        disable=not accelerator.is_local_main_process,
    )
    train_iterator = iter(train_loader)
    while global_step < args.max_train_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        source = batch["conditioning_pixel_values"]
        target = batch["output_pixel_values"]
        with accelerator.accumulate(model):
            prediction = model(source, prompt_tokens=batch["input_ids"])
            loss_l2 = F.mse_loss(prediction.float(), target.float()) * args.lambda_l2
            loss_lpips = perceptual_model(prediction.float(), target.float()).mean() * args.lambda_lpips
            loss_gram = torch.zeros((), device=accelerator.device)
            loss = loss_l2 + loss_lpips
            if args.lambda_gram > 0 and global_step >= args.gram_loss_warmup_steps:
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
        logs = {
            "train/l2": float(loss_l2.detach()),
            "train/lpips": float(loss_lpips.detach()),
            "train/gram": float(loss_gram.detach()),
            "train/loss": float(loss.detach()),
            "train/learning_rate": lr_scheduler.get_last_lr()[0],
        }
        progress.set_postfix(loss=logs["train/loss"])

        if args.report_to != "none":
            accelerator.log(logs, step=global_step)

        if global_step % args.eval_freq == 0 or global_step == args.max_train_steps:
            should_visualize = global_step % args.viz_freq == 0 or global_step == args.max_train_steps
            validation_metrics, visualizations = _validation(
                model,
                validation_loader,
                perceptual_model,
                accelerator,
                args.num_validation_samples,
                args.num_validation_visualizations if should_visualize else 0,
            )
            if args.report_to != "none":
                validation_logs = {
                    f"validation/{key}": value for key, value in validation_metrics.items()
                }
                if accelerator.is_main_process and args.report_to == "wandb" and visualizations:
                    validation_logs["validation/degraded_coarse_final_gt"] = (
                        _wandb_validation_images(visualizations)
                    )
                accelerator.log(validation_logs, step=global_step)
            if accelerator.is_main_process:
                print(f"validation step={global_step}: {validation_metrics}", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_training_checkpoint(
                accelerator.unwrap_model(model),
                optimizer,
                lr_scheduler,
                checkpoint_dir / f"model_{global_step}.pkl",
                global_step,
            )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_training_checkpoint(
            accelerator.unwrap_model(model),
            optimizer,
            lr_scheduler,
            checkpoint_dir / "final.pkl",
            global_step,
        )
    accelerator.end_training()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--coarse-root", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument(
        "--degradation-pairs", nargs="+", type=int, choices=range(1, 6), required=True
    )
    value.add_argument("--resolution", type=int, default=512)
    value.add_argument("--max-train-steps", type=int, default=10_000)
    value.add_argument("--train-batch-size", type=int, default=1)
    value.add_argument("--gradient-accumulation-steps", type=int, default=1)
    value.add_argument("--dataloader-num-workers", type=int, default=8)
    value.add_argument("--learning-rate", type=float, default=2e-5)
    value.add_argument("--lr-scheduler", default="constant")
    value.add_argument("--lr-warmup-steps", type=int, default=500)
    value.add_argument("--adam-beta1", type=float, default=0.9)
    value.add_argument("--adam-beta2", type=float, default=0.999)
    value.add_argument("--adam-weight-decay", type=float, default=1e-2)
    value.add_argument("--adam-epsilon", type=float, default=1e-8)
    value.add_argument("--max-grad-norm", type=float, default=1.0)
    value.add_argument("--lambda-l2", type=float, default=1.0)
    value.add_argument("--lambda-lpips", type=float, default=1.0)
    value.add_argument("--lambda-gram", type=float, default=1.0)
    value.add_argument("--gram-loss-warmup-steps", type=int, default=2000)
    value.add_argument("--lora-rank-vae", type=int, default=4)
    value.add_argument("--timestep", type=int, default=199)
    value.add_argument("--checkpointing-steps", type=int, default=1000)
    value.add_argument("--eval-freq", type=int, default=1000)
    value.add_argument("--viz-freq", type=int, default=1000)
    value.add_argument("--num-validation-samples", type=int, default=100)
    value.add_argument("--num-validation-visualizations", type=int, default=4)
    value.add_argument("--gradient-checkpointing", action="store_true")
    value.add_argument("--enable-xformers-memory-efficient-attention", action="store_true")
    value.add_argument("--allow-tf32", action="store_true")
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--resume", type=Path)
    value.add_argument("--report-to", choices=("wandb", "none"), default="wandb")
    value.add_argument("--tracker-project-name", default="difix-cdd11-selective")
    value.add_argument("--tracker-run-name", default="difix-selective")
    return value


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
