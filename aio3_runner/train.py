"""Complete single-GPU AIO3-v1 training runner for DACG-IR."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .adapter import DEFAULT_MODEL_CONFIG, build_model
from .data import (
    EXPECTED_MANIFEST_SHA256, AIO3ManifestDataset, make_training_loader, manifest_statistics, sha256_file,
    verify_frozen_manifests,
)
from .protocol import (
    BETAS, EFFECTIVE_BATCH_SIZE, GRAD_CLIP_NORM, PROTOCOL, RUN_KINDS, WEIGHT_DECAY,
    AIO3LRScheduler, resolved_config,
)
from .runtime import (
    append_jsonl, atomic_json, atomic_torch_save, capture_rng_state, environment_info,
    git_state, path_is_within, restore_rng_state, seed_everything,
)
from .tracking import WandbTracker
from .validation import validate


def _write_config(path: Path, config: dict[str, Any]) -> None:
    # JSON is valid YAML 1.2 and avoids adding a second configuration parser.
    atomic_json(path, config)


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _state(run_dir: Path, status: str, step: int, config: dict[str, Any], **extra: Any) -> None:
    atomic_json(run_dir / "run_state.json", {
        "protocol": PROTOCOL, "model_id": config["model_id"], "run_kind": config["run_kind"],
        "seed": config["seed"], "precision": "bf16", "status": status,
        "global_step": step, "max_steps": config["training"]["max_steps"], **extra,
    })


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: AIO3LRScheduler,
    step: int,
    best: dict[str, Any],
    config: dict[str, Any],
    manifest_hashes: dict[str, str],
    git: dict[str, Any],
    environment: dict[str, Any],
    tracker: WandbTracker,
    run_dir: Path,
    metric_window: list[dict[str, float]],
) -> dict[str, Any]:
    parameters_total = sum(parameter.numel() for parameter in model.parameters())
    parameters_trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "protocol": PROTOCOL, "run_kind": config["run_kind"], "global_step": step,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "best": best, "rng_state": capture_rng_state(),
        "metric_window": metric_window,
        "config": config, "model_config": config["model_config"], "manifest_sha256": manifest_hashes,
        "wandb_run_id": tracker.run_id, "run_dir": str(run_dir),
        "metadata": {
            "protocol": PROTOCOL, "run_kind": config["run_kind"],
            "model_config": config["model_config"], "manifest_sha256": manifest_hashes,
            "parameters_total": parameters_total, "parameters_trainable": parameters_trainable,
            "initialization": "DACG-IR native random initialization", "git": git,
            "environment": environment, "precision": "bf16", "special_fp32_boundaries": ["torch.fft"],
            "pretrained": False, "ema": False, "tta": False,
            "wandb": {
                "entity": config["monitoring"].get("entity"), "project": "aio3-restoration",
                "run_id": tracker.run_id, "run_url": getattr(tracker.run, "url", None),
                "run_dir": str(run_dir),
            },
        },
    }


def _save_checkpoint(path: Path, **kwargs: Any) -> None:
    atomic_torch_save(path, _checkpoint_payload(**kwargs))


def _model_preflight(model: torch.nn.Module, device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError("AIO3-v1 BF16 training requires a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected GPU does not support BF16")
    for parameter in model.parameters():
        if parameter.dtype != torch.float32:
            raise RuntimeError("all model parameters must remain FP32")
    model.to(device).train()
    sample = torch.randn(1, 3, 127, 191, device=device)
    target = torch.rand(1, 3, 127, 191, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(sample)
        loss = torch.mean(torch.abs(output - target))
    if output.shape != sample.shape or not torch.isfinite(loss):
        raise RuntimeError("arbitrary-resolution BF16 forward preflight failed")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"non-finite or missing preflight gradient: {name}")
    model.zero_grad(set_to_none=True)


def _data_preflight(run_dir: Path, config: dict[str, Any]) -> None:
    loader = make_training_loader(
        run_dir / "manifests" / "train.jsonl", data_root=config["paths"]["data_root"],
        max_steps=1, seed=int(config["seed"]), num_workers=0,
    )
    batch = next(iter(loader))
    if tuple(batch["input"].shape) != (12, 3, 128, 128) or batch["input"].shape != batch["target"].shape:
        raise RuntimeError("training data preflight expected paired [12,3,128,128] tensors")
    if Counter(batch["task"]) != {"denoise": 4, "derain": 4, "dehaze": 4}:
        raise RuntimeError("training data preflight failed the 4/4/4 task invariant")
    validation = AIO3ManifestDataset(
        run_dir / "manifests" / "val.jsonl", split="val", data_root=config["paths"]["data_root"]
    )
    denoise_index = next(index for index, record in enumerate(validation.records) if record["task"] == "denoise")
    if not torch.equal(validation[denoise_index]["input"], validation[denoise_index]["input"]):
        raise RuntimeError("fixed validation noise is not exactly reproducible")


def _copy_run_inputs(
    *, run_dir: Path, manifest_dir: Path, protocol_document: Path, config: dict[str, Any]
) -> dict[str, str]:
    destination = run_dir / "manifests"
    destination.mkdir(parents=True, exist_ok=False)
    for name in EXPECTED_MANIFEST_SHA256:
        shutil.copy2(manifest_dir / name, destination / name)
    hashes = verify_frozen_manifests(destination)
    shutil.copy2(protocol_document, run_dir / "AIO3_TRAINING_EVALUATION_PROTOCOL.md")
    config["protocol_document_sha256"] = sha256_file(run_dir / "AIO3_TRAINING_EVALUATION_PROTOCOL.md")
    config["manifest_sha256"] = hashes
    config["data_summary"] = manifest_statistics(destination)
    _write_config(run_dir / "config.yaml", config)
    return hashes


def _new_run(args: argparse.Namespace, device: torch.device) -> tuple[Path, dict[str, Any], dict[str, str]]:
    repo_root = Path(__file__).resolve().parents[1]
    output_root, data_root = Path(args.output_root).resolve(), Path(args.data_root).resolve()
    if path_is_within(output_root, repo_root) or path_is_within(output_root, data_root):
        raise ValueError("output-root must be outside both the source repository and raw data directory")
    manifest_dir = Path(args.manifest_dir).resolve()
    protocol_document = Path(args.protocol_document).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"data-root does not exist: {data_root}")
    if not protocol_document.is_file():
        raise FileNotFoundError(f"protocol document does not exist: {protocol_document}")
    if args.run_kind == "formal" and args.wandb_mode == "online" and not args.wandb_entity:
        raise ValueError("formal online training requires --wandb-entity")
    verify_frozen_manifests(manifest_dir)
    git = git_state(repo_root)
    if args.run_kind == "formal" and git["dirty"]:
        raise RuntimeError("formal training requires a clean Git worktree")
    model_config = dict(DEFAULT_MODEL_CONFIG)
    if args.model_size == "small":
        model_config["dim"] = 32
    model_id = "dacg_ir_s" if args.model_size == "small" else "dacg_ir"
    config = resolved_config(
        model_id=model_id,
        run_kind=args.run_kind, seed=args.seed, num_workers=args.num_workers,
        micro_batch_size=args.micro_batch_size, model_config=model_config,
        wandb_mode=args.wandb_mode, wandb_entity=args.wandb_entity,
    )
    config["paths"] = {
        "data_root": str(data_root), "output_root": str(output_root),
        "manifest_source": str(manifest_dir),
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"{model_id.replace('_', '-')}-{args.run_kind}-seed{args.seed}-{timestamp}"
    run_dir = output_root / model_id / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints", "logs", "validation", "wandb"):
        (run_dir / name).mkdir()
    hashes = _copy_run_inputs(
        run_dir=run_dir, manifest_dir=manifest_dir,
        protocol_document=protocol_document, config=config,
    )
    return run_dir, config, hashes


def _resume_run(args: argparse.Namespace) -> tuple[Path, dict[str, Any], dict[str, str], dict[str, Any]]:
    checkpoint_path = Path(args.resume).resolve()
    if checkpoint_path.name != "latest.pth":
        raise ValueError("resume accepts only the same run's checkpoints/latest.pth")
    run_dir = checkpoint_path.parent.parent
    config = _read_config(run_dir / "config.yaml")
    hashes = verify_frozen_manifests(run_dir / "manifests")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != PROTOCOL or checkpoint.get("config") != config:
        raise ValueError("resume protocol/config mismatch")
    if checkpoint.get("manifest_sha256") != hashes:
        raise ValueError("resume manifest hash mismatch")
    if Path(args.data_root).resolve() != Path(config["paths"]["data_root"]).resolve():
        raise ValueError("resume data-root differs from the frozen run configuration")
    current_git = git_state(Path(__file__).resolve().parents[1])
    if current_git["commit"] != checkpoint["metadata"]["git"]["commit"] or current_git["dirty"]:
        raise ValueError("resume requires the clean Git commit recorded in the checkpoint")
    return run_dir, config, hashes, checkpoint


def _training_step(
    model: torch.nn.Module,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    micro_batch_size: int,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    degraded, target = batch["input"].to(device, non_blocking=True), batch["target"].to(device, non_blocking=True)
    tasks = list(batch["task"])
    sample_losses: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []
    for start in range(0, EFFECTIVE_BATCH_SIZE, micro_batch_size):
        stop = start + micro_batch_size
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            restored_raw = model(degraded[start:stop])
            difference = torch.abs(restored_raw - target[start:stop])
            micro_loss = difference.mean()
        (micro_loss * ((stop - start) / EFFECTIVE_BATCH_SIZE)).backward()
        sample_losses.append(difference.detach().float().mean(dim=(1, 2, 3)).cpu())
        residuals.append((restored_raw.detach().float() - degraded[start:stop].float()).cpu())
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError(f"non-finite gradient norm: {grad_norm}")
    optimizer.step()
    losses = torch.cat(sample_losses)
    residual = torch.cat(residuals)
    result = {
        "loss": float(losses.mean()), "grad_norm": float(grad_norm),
        "residual_mean": float(residual.mean()),
        "residual_second_moment": float(residual.square().mean()),
        "residual_min": float(residual.min()), "residual_max": float(residual.max()),
        "residual_negative_fraction": float((residual < 0).float().mean()),
        "residual_positive_fraction": float((residual > 0).float().mean()),
        "residual_near_zero_fraction": float((residual.abs() <= 1e-6).float().mean()),
        "prediction_below_zero_fraction": float(((residual + degraded.cpu()) < 0).float().mean()),
        "prediction_above_one_fraction": float(((residual + degraded.cpu()) > 1).float().mean()),
    }
    for task in ("denoise", "derain", "dehaze"):
        indices = [index for index, value in enumerate(tasks) if value == task]
        if len(indices) != 4:
            raise RuntimeError(f"balanced batch invariant failed for {task}: {len(indices)}")
        result[f"{task}_l1_sum"] = float(losses[indices].sum())
        result[f"samples_{task}"] = len(indices)
    return result


def train(args: argparse.Namespace) -> Path:
    if args.micro_batch_size not in (1, 2, 3, 4, 6, 12):
        raise ValueError("micro-batch-size must divide the effective batch size 12")
    if args.num_workers != 8 and not args.resume:
        raise ValueError("AIO3-v1 fixes training num-workers at 8")
    device = torch.device(args.device or "cuda")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("AIO3-v1 training requires an available CUDA GPU")
    resume_checkpoint = None
    if args.resume:
        run_dir, config, manifest_hashes, resume_checkpoint = _resume_run(args)
    else:
        run_dir, config, manifest_hashes = _new_run(args, device)
    environment = environment_info(device)
    atomic_json(run_dir / "environment.json", environment)
    git = git_state(Path(__file__).resolve().parents[1])
    start_step = int(resume_checkpoint["global_step"]) if resume_checkpoint else 0
    best = dict(resume_checkpoint.get("best", {})) if resume_checkpoint else {
        "val_macro_psnr": None, "val_macro_ssim": None, "global_step": None,
    }
    max_steps = int(config["training"]["max_steps"])
    pause_at = args.pause_at_step
    scalar_interval = int(config["monitoring"]["scalar_interval_steps"])
    if pause_at is not None and (pause_at <= start_step or pause_at >= max_steps or pause_at % scalar_interval):
        raise ValueError("pause-at-step must be after the current step, below max_steps, and align with scalar logging")

    _state(run_dir, "running", start_step, config, phase="preflight")
    try:
        seed_everything(int(config["seed"]))
        model = build_model(config["model_config"])
        _model_preflight(model, device)
        if not resume_checkpoint:
            _data_preflight(run_dir, config)
        seed_everything(int(config["seed"]))
        model = build_model(config["model_config"]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=2e-4, betas=BETAS, weight_decay=WEIGHT_DECAY,
        )
        scheduler = AIO3LRScheduler(
            optimizer, max_steps=max_steps, warmup_steps=int(config["scheduler"]["warmup_steps"]),
        )
        if resume_checkpoint:
            model.load_state_dict(resume_checkpoint["model"], strict=True)
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
            for state in optimizer.state.values():
                for name, value in state.items():
                    if torch.is_tensor(value):
                        state[name] = value.to(device)
            scheduler.load_state_dict(resume_checkpoint["scheduler"])
            if scheduler.completed_steps != start_step:
                raise ValueError("checkpoint scheduler/global_step mismatch")
            restore_rng_state(resume_checkpoint["rng_state"])
            recorded_environment = resume_checkpoint["metadata"]["environment"]
            for name in ("python", "pytorch", "torchvision", "cuda_runtime", "gpu", "wandb"):
                if environment.get(name) != recorded_environment.get(name):
                    raise ValueError(f"resume environment mismatch for {name}")
        else:
            config["reproducibility"] = {
                "git": git, "environment": environment,
                "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
                "parameters_trainable": sum(
                    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                ),
                "initialization": "DACG-IR native random initialization",
                "special_fp32_boundaries": ["torch.fft"],
                "launch_command": list(sys.argv),
            }
            _write_config(run_dir / "config.yaml", config)
        tracker = WandbTracker(
            run_dir=run_dir, config=config,
            run_id=resume_checkpoint.get("wandb_run_id") if resume_checkpoint else None,
            resume=resume_checkpoint is not None,
        )
    except BaseException as setup_error:
        _state(
            run_dir, "failed", start_step, config, phase="preflight_or_wandb_init",
            error_type=type(setup_error).__name__, error=str(setup_error),
        )
        (run_dir / "logs" / "fatal_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    (run_dir / "wandb_run_id.txt").write_text(tracker.run_id + "\n", encoding="utf-8")
    atomic_json(run_dir / "wandb_state.json", {
        "mode": config["monitoring"]["mode"], "run_id": tracker.run_id,
        "entity": config["monitoring"].get("entity"), "project": "aio3-restoration",
        "group": PROTOCOL, "status": "running", "url": getattr(tracker.run, "url", None),
    })
    if not resume_checkpoint:
        tracker.log_dataset_artifact(
            [run_dir / "AIO3_TRAINING_EVALUATION_PROTOCOL.md", run_dir / "config.yaml"]
            + [run_dir / "manifests" / name for name in EXPECTED_MANIFEST_SHA256],
            {"protocol": PROTOCOL, "manifest_sha256": manifest_hashes},
        )

    loader = make_training_loader(
        run_dir / "manifests" / "train.jsonl", data_root=config["paths"]["data_root"],
        max_steps=max_steps, seed=int(config["seed"]), start_step=start_step,
        num_workers=int(config["data"]["num_workers"]),
    )
    window: list[dict[str, float]] = list(resume_checkpoint.get("metric_window", [])) if resume_checkpoint else []
    _state(run_dir, "running", start_step, config, wandb_run_id=tracker.run_id)
    exit_code = 1
    try:
        loader_iterator = iter(loader)
        while True:
            tick = time.perf_counter()
            try:
                batch = next(loader_iterator)
            except StopIteration:
                break
            current_lr = float(optimizer.param_groups[0]["lr"])
            values = _training_step(
                model, batch, optimizer, device, int(config["data"]["micro_batch_size"])
            )
            scheduler.step()
            step = scheduler.completed_steps
            elapsed = time.perf_counter() - tick
            values.update({"learning_rate": current_lr, "step_time_seconds": elapsed, "images_per_second": 12 / elapsed})
            window.append(values)

            if step % scalar_interval == 0:
                count = len(window)
                payload: dict[str, Any] = {
                    "global_step": step,
                    "train/loss": sum(item["loss"] for item in window) / count,
                    "train/learning_rate": window[-1]["learning_rate"],
                    "train/grad_norm": sum(item["grad_norm"] for item in window) / count,
                    "train/step_time_seconds": sum(item["step_time_seconds"] for item in window) / count,
                    "train/images_per_second": sum(item["images_per_second"] for item in window) / count,
                }
                for task in ("denoise", "derain", "dehaze"):
                    samples = int(sum(item[f"samples_{task}"] for item in window))
                    payload[f"train/{task}_l1"] = sum(item[f"{task}_l1_sum"] for item in window) / samples
                    payload[f"train/samples_{task}"] = samples
                    if samples != 4 * scalar_interval:
                        raise RuntimeError(f"task-balance logging invariant failed for {task}: {samples}")
                for name in (
                    "residual_mean", "residual_negative_fraction", "residual_positive_fraction",
                    "residual_near_zero_fraction", "prediction_below_zero_fraction",
                    "prediction_above_one_fraction",
                ):
                    payload[f"diagnostics/{name}"] = sum(item[name] for item in window) / count
                residual_mean = payload["diagnostics/residual_mean"]
                second_moment = sum(item["residual_second_moment"] for item in window) / count
                payload["diagnostics/residual_std"] = math.sqrt(max(0.0, second_moment - residual_mean**2))
                payload["diagnostics/residual_min"] = min(item["residual_min"] for item in window)
                payload["diagnostics/residual_max"] = max(item["residual_max"] for item in window)
                payload["system/gpu_memory_allocated_gib"] = torch.cuda.memory_allocated(device) / (1024**3)
                payload["system/gpu_memory_reserved_gib"] = torch.cuda.memory_reserved(device) / (1024**3)
                append_jsonl(run_dir / "train_metrics.jsonl", payload)
                tracker.log(payload)
                window.clear()

            checkpoint_due = step % int(config["checkpoint"]["interval_steps"]) == 0
            if checkpoint_due or (pause_at is not None and step == pause_at):
                checkpoint_kwargs = dict(
                    model=model, optimizer=optimizer, scheduler=scheduler, step=step, best=best,
                    config=config, manifest_hashes=manifest_hashes, git=git, environment=environment,
                    tracker=tracker, run_dir=run_dir, metric_window=window,
                )
                _save_checkpoint(run_dir / "checkpoints" / "latest.pth", **checkpoint_kwargs)
                if config["run_kind"] == "formal" and step % 50000 == 0:
                    _save_checkpoint(run_dir / "checkpoints" / f"step_{step:06d}.pth", **checkpoint_kwargs)

            if pause_at is not None and step == pause_at:
                _state(
                    run_dir, "paused", step, config, wandb_run_id=tracker.run_id,
                    wandb_url=getattr(tracker.run, "url", None), best=best,
                )
                atomic_json(run_dir / "wandb_state.json", {"status": "paused", "run_id": tracker.run_id})
                tracker.finish(0)
                return run_dir

            if step % int(config["validation"]["interval_steps"]) == 0:
                _state(run_dir, "validating", step, config, wandb_run_id=tracker.run_id, best=best)
                rng_before_validation = capture_rng_state()
                media = step % int(config["monitoring"]["media_interval_steps"]) == 0
                try:
                    metrics, rows, table, histogram = validate(
                        model, manifest=run_dir / "manifests" / "val.jsonl",
                        visual_samples=run_dir / "manifests" / "visual_samples.json",
                        data_root=config["paths"]["data_root"], device=device,
                        num_workers=min(4, int(config["data"]["num_workers"])),
                        global_step=step, media=media, output_dir=run_dir / "validation",
                        wandb_module=tracker.wandb if media else None,
                    )
                    validation_payload = {"global_step": step, **metrics}
                    validation_file = run_dir / "validation" / f"metrics_step_{step:06d}.json"
                    atomic_json(validation_file, {"global_step": step, "metrics": metrics, "per_image": rows})
                    append_jsonl(run_dir / "validation_metrics.jsonl", validation_payload)
                    if table is not None:
                        validation_payload["val/fixed_samples"] = table
                    if histogram is not None:
                        validation_payload["diagnostics/fixed_samples/residual_histogram"] = tracker.histogram(histogram)
                finally:
                    restore_rng_state(rng_before_validation)
                tracker.log(validation_payload)
                psnr, ssim = metrics["val/macro/psnr"], metrics["val/macro/ssim"]
                if best["val_macro_psnr"] is None or psnr > float(best["val_macro_psnr"]):
                    best = {"val_macro_psnr": psnr, "val_macro_ssim": ssim, "global_step": step}
                    checkpoint_kwargs["best"] = best
                    _save_checkpoint(run_dir / "checkpoints" / "best_macro_psnr.pth", **checkpoint_kwargs)
                    tracker.update_best(step=step, psnr=psnr, ssim=ssim)
                checkpoint_kwargs["best"] = best
                _save_checkpoint(run_dir / "checkpoints" / "latest.pth", **checkpoint_kwargs)
                _state(run_dir, "running", step, config, wandb_run_id=tracker.run_id, best=best)

        final_step = scheduler.completed_steps
        if final_step != max_steps:
            raise RuntimeError(f"training ended at {final_step}, expected {max_steps}")
        _state(
            run_dir, "completed", final_step, config, wandb_run_id=tracker.run_id, best=best,
            wandb_url=getattr(tracker.run, "url", None), best_validation_step=best["global_step"],
        )
        atomic_json(run_dir / "wandb_state.json", {"status": "completed", "run_id": tracker.run_id})
        tracker.log_model_artifact([
            run_dir / "checkpoints" / "best_macro_psnr.pth", run_dir / "config.yaml",
            run_dir / "environment.json", run_dir / "manifests" / "data_audit.json",
            run_dir / "validation" / f"metrics_step_{final_step:06d}.json",
        ], int(config["seed"]))
        exit_code = 0
        tracker.finish(0)
        print(f"RUN_DIR={run_dir}", flush=True)
        return run_dir
    except BaseException as error:
        step = scheduler.completed_steps
        status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        try:
            _save_checkpoint(
                run_dir / "checkpoints" / "latest.pth", model=model, optimizer=optimizer,
                scheduler=scheduler, step=step, best=best, config=config,
                manifest_hashes=manifest_hashes, git=git, environment=environment,
                tracker=tracker, run_dir=run_dir, metric_window=window,
            )
        except Exception as checkpoint_error:
            append_jsonl(run_dir / "logs" / "checkpoint_errors.jsonl", {
                "global_step": step, "error_type": type(checkpoint_error).__name__,
                "error": str(checkpoint_error),
            })
        _state(run_dir, status, step, config, error_type=type(error).__name__, error=str(error), best=best)
        (run_dir / "logs" / "fatal_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        atomic_json(run_dir / "wandb_state.json", {"status": "failed", "run_id": tracker.run_id})
        tracker.finish(exit_code)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DACG-IR under the frozen AIO3-v1 protocol")
    parser.add_argument("--manifest-dir")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--protocol-document")
    parser.add_argument("--run-kind", choices=sorted(RUN_KINDS), default="formal")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=12)
    parser.add_argument("--model-size", choices=("base", "small"), default="base")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--pause-at-step", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.resume:
        if args.manifest_dir or args.output_root or args.protocol_document:
            parser.error("--resume reads manifest/output/protocol settings from the existing run")
    elif not all((args.manifest_dir, args.output_root, args.protocol_document)):
        parser.error("new runs require --manifest-dir, --data-root, --output-root and --protocol-document")
    train(args)


if __name__ == "__main__":
    main()
