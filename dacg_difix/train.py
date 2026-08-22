"""Accelerate training for degradation-guided DACG + Difix3D restoration."""

from __future__ import annotations

import argparse
import json
import math
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cdd11_full.metrics import rgb_psnr, rgb_ssim
from cdd11_full.runtime import atomic_json, atomic_torch_save, restore_rng, rng_state

from .loss import build_restoration_loss


CHECKPOINT_FORMAT = "dacg-difix-adapter-v1"


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def adapter_checkpoint(
    model: torch.nn.Module,
    global_step: int,
    args: argparse.Namespace | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler: Any | None = None,
) -> dict[str, Any]:
    """Create a checkpoint without frozen SD-Turbo or DACG base weights."""

    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "global_step": int(global_step),
        "adapter": _cpu_tree(model.adapter_state_dict()),
    }
    if args is not None:
        payload["training_config"] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
    if optimizer is not None:
        payload["optimizer"] = _cpu_tree(optimizer.state_dict())
    if lr_scheduler is not None:
        payload["lr_scheduler"] = _cpu_tree(lr_scheduler.state_dict())
    if optimizer is not None:
        payload["rng"] = rng_state()
    return payload


def save_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    global_step: int,
    args: argparse.Namespace | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler: Any | None = None,
) -> None:
    atomic_torch_save(
        path,
        adapter_checkpoint(model, global_step, args, optimizer, lr_scheduler),
    )


def load_adapter_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler: Any | None = None,
    restore_random_state: bool = False,
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported Difix checkpoint format in {path}")
    model.load_adapter_state_dict(checkpoint["adapter"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    if restore_random_state and "rng" in checkpoint:
        restore_rng(checkpoint["rng"])
    return int(checkpoint["global_step"])


def _pad_to_multiple(images: torch.Tensor, multiple: int = 8) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = images.shape[-2:]
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if pad_h or pad_w:
        mode = "reflect" if height > pad_h and width > pad_w else "replicate"
        images = F.pad(images, (0, pad_w, 0, pad_h), mode=mode)
    return images, (height, width)


def _native_prediction(
    model: torch.nn.Module,
    main: torch.Tensor,
    reference: torch.Tensor,
    prompt_tokens: torch.Tensor,
    p_global: torch.Tensor,
) -> torch.Tensor:
    padded_main, original_size = _pad_to_multiple(main)
    padded_reference, _ = _pad_to_multiple(reference)
    prediction = model(
        padded_main,
        padded_reference,
        prompt_tokens,
        p_global=p_global,
    )
    height, width = original_size
    return prediction[..., :height, :width]


def _display_image(value: torch.Tensor) -> torch.Tensor:
    """Convert one normalized CHW image to a CPU [0, 1] tensor for logging."""

    return value.detach().float().cpu().add(1).mul(0.5).clamp(0, 1)


def _comparison_image(
    degraded: torch.Tensor,
    coarse: torch.Tensor,
    final: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Join degraded, DACG coarse, Difix final and GT from left to right."""

    panels = (
        _display_image(degraded[0]),
        _display_image(coarse[0]),
        _display_image(final[0]),
        _display_image(target[0]),
    )
    return torch.cat(panels, dim=-1)


def _wandb_validation_images(visualizations: list[tuple[torch.Tensor, str]]) -> list[Any]:
    import wandb

    return [wandb.Image(image, caption=caption) for image, caption in visualizations]


def _build_model(args: argparse.Namespace) -> torch.nn.Module:
    from .model import DACGDifix

    return DACGDifix(
        pretrained_model_name_or_path=args.pretrained_model,
        local_files_only=args.local_files_only,
        lora_mode=args.lora_mode,
        lora_rank_unet=args.lora_rank_unet,
        lora_rank_vae=args.lora_rank_vae,
        timestep=args.timestep,
    )


def _make_loader(
    manifest: Path,
    model: torch.nn.Module,
    *,
    resolution: int | None,
    prompt: str,
    batch_size: int,
    shuffle: bool,
    workers: int,
) -> DataLoader:
    from .data import PreparedCDD11Dataset

    dataset = PreparedCDD11Dataset(
        manifest,
        resolution=resolution,
        tokenizer=model.tokenizer,
        prompt=prompt,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


@torch.inference_mode()
def _validate(
    model: torch.nn.Module,
    dam: torch.nn.Module,
    loader: DataLoader,
    lpips_model: torch.nn.Module,
    accelerator: Any,
    limit: int,
    visualization_limit: int = 0,
) -> tuple[dict[str, float], list[tuple[torch.Tensor, str]]]:
    model.eval()
    rows: list[torch.Tensor] = []
    visualizations: list[tuple[torch.Tensor, str]] = []
    for index, batch in enumerate(loader):
        if limit and index >= limit:
            break
        p_global = dam(batch["ref_01"])
        prediction = _native_prediction(
            model,
            batch["main"],
            batch["ref"],
            batch["prompt_tokens"],
            p_global,
        )
        target = batch["target"]
        prediction_01 = prediction.float().add(1).mul(0.5).clamp(0, 1)
        coarse_01 = batch["main"].float().add(1).mul(0.5).clamp(0, 1)
        target_01 = target.float().add(1).mul(0.5).clamp(0, 1)
        perceptual = lpips_model(prediction.float(), target.float()).mean()
        row = prediction.new_tensor(
            [
                rgb_psnr(prediction_01, target_01),
                rgb_ssim(prediction_01, target_01),
                rgb_psnr(coarse_01, target_01),
                rgb_ssim(coarse_01, target_01),
            ]
        ).float()
        rows.append(torch.cat((row, perceptual.reshape(1).float())))
        if accelerator.is_main_process and len(visualizations) < visualization_limit:
            caption = (
                f"{batch['id'][0]} | {batch['degradation'][0]} | "
                "left to right: degraded | DACG coarse | Difix final | GT"
            )
            visualizations.append((
                _comparison_image(batch["ref"], batch["main"], prediction, target),
                caption,
            ))
    if not rows:
        metrics = {
            "psnr": math.nan,
            "ssim": math.nan,
            "coarse_psnr": math.nan,
            "coarse_ssim": math.nan,
            "lpips": math.nan,
        }
        model.train()
        return metrics, visualizations
    gathered = accelerator.gather_for_metrics(torch.stack(rows))
    means = gathered.mean(0).cpu().tolist()
    model.train()
    metric_names = ("psnr", "ssim", "coarse_psnr", "coarse_ssim", "lpips")
    return dict(zip(metric_names, means, strict=True)), visualizations


def run(args: argparse.Namespace) -> None:
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs, set_seed
    from diffusers.optimization import get_scheduler
    from tqdm.auto import tqdm

    from cdd11_full.model import load_dam_encoder

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

    if args.data_root is not None:
        if accelerator.is_main_process:
            from .prepare import prepare_cdd11_folder_manifests

            prepared = prepare_cdd11_folder_manifests(
                args.data_root,
                args.coarse_root,
                output_dir,
            )
            print(f"prepared CDD-11 folder split: {prepared['counts']}", flush=True)
        accelerator.wait_for_everyone()
        train_manifest = output_dir / "prepared" / "manifests" / "train.jsonl"
        validation_manifest = output_dir / "prepared" / "manifests" / "validation.jsonl"
    else:
        train_manifest = args.train_manifest
        validation_manifest = args.validation_manifest

    with accelerator.main_process_first():
        model = _build_model(args)
        loss_model = build_restoration_loss(
            lambda_mse=args.lambda_mse,
            lambda_lpips=args.lambda_lpips,
            lambda_gram=args.lambda_gram,
            gram_warmup_steps=args.gram_loss_warmup_steps,
        )
    dam, _ = load_dam_encoder(str(args.dam_checkpoint), accelerator.device)
    dam.requires_grad_(False).eval()

    if args.gradient_checkpointing:
        model.unet.enable_gradient_checkpointing()
    if args.enable_xformers_memory_efficient_attention:
        model.unet.enable_xformers_memory_efficient_attention()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    trainable_parameters = list(model.trainable_parameters())
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
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    global_step = 0
    best_state_path = output_dir / "best_validation.json"
    best_psnr = -math.inf
    if best_state_path.is_file():
        best_psnr = float(json.loads(best_state_path.read_text(encoding="utf-8"))["psnr"])
    if args.resume is not None:
        global_step = load_adapter_checkpoint(
            args.resume,
            model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            restore_random_state=True,
        )

    train_loader = _make_loader(
        train_manifest,
        model,
        resolution=args.resolution,
        prompt=args.prompt,
        batch_size=args.train_batch_size,
        shuffle=True,
        workers=args.dataloader_num_workers,
    )
    validation_loader = _make_loader(
        validation_manifest,
        model,
        resolution=None,
        prompt=args.prompt,
        batch_size=1,
        shuffle=False,
        workers=min(args.dataloader_num_workers, 4),
    )
    model, optimizer, train_loader, validation_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, validation_loader, lr_scheduler
    )
    loss_model = loss_model.to(accelerator.device)

    if accelerator.is_main_process and args.report_to != "none":
        config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        init_kwargs = {"wandb": {"dir": str(output_dir)}}
        if args.tracker_run_name:
            init_kwargs["wandb"]["name"] = args.tracker_run_name
        accelerator.init_trackers(
            args.tracker_project_name,
            config=config,
            init_kwargs=init_kwargs,
        )

    progress = tqdm(
        total=args.max_train_steps,
        initial=global_step,
        desc="Difix steps",
        disable=not accelerator.is_local_main_process,
    )
    iterator = iter(train_loader)
    model.train()
    while global_step < args.max_train_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)

        with accelerator.accumulate(model):
            with torch.no_grad():
                p_global = dam(batch["ref_01"])
            with accelerator.autocast():
                prediction = model(
                    batch["main"],
                    batch["ref"],
                    batch["prompt_tokens"],
                    p_global=p_global,
                )
                loss, components = loss_model(prediction, batch["target"], global_step)
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
            f"train/{name}": float(value)
            for name, value in components.items()
        }
        logs["train/learning_rate"] = lr_scheduler.get_last_lr()[0]
        progress.set_postfix(loss=logs["train/total"])
        if args.report_to != "none":
            accelerator.log(logs, step=global_step)

        should_validate = global_step % args.validation_steps == 0 or global_step == args.max_train_steps
        if should_validate:
            metrics, visualizations = _validate(
                model,
                dam,
                validation_loader,
                loss_model.lpips_model,
                accelerator,
                args.validation_limit,
                args.validation_visualizations,
            )
            if args.report_to != "none":
                validation_logs = {
                    f"validation/{key}": value for key, value in metrics.items()
                }
                if accelerator.is_main_process and args.report_to == "wandb" and visualizations:
                    validation_logs["validation/degraded_coarse_final_gt"] = (
                        _wandb_validation_images(visualizations)
                    )
                accelerator.log(validation_logs, step=global_step)
            if accelerator.is_main_process:
                print(f"validation step={global_step}: {metrics}", flush=True)
                if math.isfinite(metrics["psnr"]) and metrics["psnr"] > best_psnr:
                    best_psnr = metrics["psnr"]
                    checkpoint_dir = output_dir / "checkpoints"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    save_training_checkpoint(
                        checkpoint_dir / "best_psnr.pt",
                        accelerator.unwrap_model(model),
                        global_step,
                        args,
                        optimizer,
                        lr_scheduler,
                    )
                    atomic_json(best_state_path, {
                        "global_step": global_step,
                        "psnr": best_psnr,
                        "ssim": metrics["ssim"],
                    })
                    print(
                        f"new best validation PSNR={best_psnr:.4f} at step={global_step}",
                        flush=True,
                    )

        should_checkpoint = global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps
        if accelerator.is_main_process and should_checkpoint:
            unwrapped = accelerator.unwrap_model(model)
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            payload_args = args
            save_training_checkpoint(
                checkpoint_dir / f"adapter_{global_step:06d}.pt",
                unwrapped,
                global_step,
                payload_args,
                optimizer,
                lr_scheduler,
            )
            save_training_checkpoint(
                checkpoint_dir / "latest.pt",
                unwrapped,
                global_step,
                payload_args,
                optimizer,
                lr_scheduler,
            )

    accelerator.wait_for_everyone()
    accelerator.end_training()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--train-manifest", type=Path)
    value.add_argument("--validation-manifest", type=Path)
    value.add_argument("--data-root", type=Path)
    value.add_argument("--coarse-root", type=Path)
    value.add_argument("--dam-checkpoint", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--pretrained-model", default="stabilityai/sd-turbo")
    value.add_argument("--local-files-only", action="store_true")
    value.add_argument(
        "--lora-mode",
        choices=("static", "p-global", "p-global-layer-id"),
        default="p-global-layer-id",
    )
    value.add_argument("--lora-rank-unet", type=int, default=32)
    value.add_argument("--lora-rank-vae", type=int, default=16)
    value.add_argument("--timestep", type=int, default=199)
    value.add_argument("--prompt", default="remove degradation")
    value.add_argument("--resolution", type=int, default=512)
    value.add_argument("--max-train-steps", type=int, default=100_000)
    value.add_argument("--train-batch-size", type=int, default=1)
    value.add_argument("--gradient-accumulation-steps", type=int, default=1)
    value.add_argument("--dataloader-num-workers", type=int, default=8)
    value.add_argument("--learning-rate", type=float, default=2e-5)
    value.add_argument("--lr-scheduler", default="constant")
    value.add_argument("--lr-warmup-steps", type=int, default=0)
    value.add_argument("--adam-beta1", type=float, default=0.9)
    value.add_argument("--adam-beta2", type=float, default=0.999)
    value.add_argument("--adam-weight-decay", type=float, default=1e-2)
    value.add_argument("--adam-epsilon", type=float, default=1e-8)
    value.add_argument("--max-grad-norm", type=float, default=1.0)
    value.add_argument("--lambda-mse", type=float, default=1.0)
    value.add_argument("--lambda-lpips", type=float, default=1.0)
    value.add_argument("--lambda-gram", type=float, default=1.0)
    value.add_argument("--gram-loss-warmup-steps", type=int, default=2_000)
    value.add_argument("--checkpointing-steps", type=int, default=1_000)
    value.add_argument("--validation-steps", type=int, default=1_000)
    value.add_argument("--validation-limit", type=int, default=100)
    value.add_argument("--validation-visualizations", type=int, default=4)
    value.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    value.add_argument("--gradient-checkpointing", action="store_true")
    value.add_argument("--enable-xformers-memory-efficient-attention", action="store_true")
    value.add_argument("--allow-tf32", action="store_true")
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--resume", type=Path)
    value.add_argument("--report-to", choices=("wandb", "none"), default="none")
    value.add_argument("--tracker-project-name", default="cdd11-dacg-difix")
    value.add_argument("--tracker-run-name")
    return value


def main() -> None:
    args = parser().parse_args()
    folder_mode = args.data_root is not None or args.coarse_root is not None
    manifest_mode = args.train_manifest is not None or args.validation_manifest is not None
    if folder_mode == manifest_mode:
        raise SystemExit(
            "choose exactly one input mode: --data-root with --coarse-root, or "
            "--train-manifest with --validation-manifest"
        )
    if folder_mode and (args.data_root is None or args.coarse_root is None):
        raise SystemExit("folder mode requires both --data-root and --coarse-root")
    if manifest_mode and (args.train_manifest is None or args.validation_manifest is None):
        raise SystemExit("manifest mode requires both training and validation manifests")
    run(args)


if __name__ == "__main__":
    main()
