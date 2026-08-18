"""Train original DACG on every pair in the official CDD-11 train partition."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.utils.schedulers import LinearWarmupCosineAnnealingLR

from .data import CDD11FullDataset, DeterministicStepBatchSampler, dataset_fingerprint, validate_partition
from .model import OriginalDACGLoss, build_model
from .protocol import NUM_TRAIN_SCENES, PROTOCOL_NAME, SEED, sha256_payload, write_protocol
from .runtime import atomic_json, atomic_torch_save, git_state, restore_rng, rng_state, seed_all
from .tracking import Tracker


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config(args: argparse.Namespace, fingerprint: str) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_NAME,
        "model": args.model,
        "seed": SEED,
        "data_root": str(args.data_root.resolve()),
        "train_fingerprint": fingerprint,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "effective_batch_size": args.batch_size * args.accumulate_grad_batches,
        "patch_size": args.patch_size,
        "optimizer": "AdamW",
        "lr": args.lr,
        "scheduler": {"type": "LinearWarmupCosineAnnealingLR", "warmup_epochs": 15, "max_epochs": 150},
        "loss": "L1 + 0.1 * mean_abs_real_imag_rfft2",
        "precision": args.precision,
        "gradient_clip_norm": args.gradient_clip_norm,
        "num_workers": args.num_workers,
        "log_interval_steps": args.log_interval_steps,
        "checkpoint_interval_steps": args.checkpoint_interval_steps,
        "checkpoint_selection": "final_completed_training_checkpoint",
        "validation": None,
    }


def _checkpoint(
    *, model, optimizer, scheduler, config: dict[str, Any], config_sha256: str,
    global_step: int, status: str, tracker: Tracker,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "protocol": PROTOCOL_NAME,
        "status": status,
        "global_step": global_step,
        "epoch": global_step // config["steps_per_epoch"],
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_state": rng_state(),
        "config": config,
        "config_sha256": config_sha256,
        "wandb_run_id": tracker.run_id,
        "saved_at_utc": _utc_now(),
    }


def _state(run_dir: Path, status: str, step: int, total: int, message: str | None = None) -> None:
    payload = {"status": status, "global_step": step, "total_steps": total, "updated_at_utc": _utc_now()}
    if message:
        payload["message"] = message
    atomic_json(run_dir / "run_state.json", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="CDD11 directory containing train/ and test/")
    parser.add_argument("--output-dir", type=Path, help="New run directory")
    parser.add_argument("--resume", type=Path, help="Path to this run's checkpoints/latest.pth")
    parser.add_argument("--model", choices=("DACG_IR", "DACG_IR_S"), default="DACG_IR")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--log-interval-steps", type=int, default=50)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=500,
                        help="Save every N optimizer steps and at epoch boundaries; 0 means epoch boundaries only")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-entity")
    args = parser.parse_args()
    if args.resume:
        args.resume = args.resume.resolve()
        inferred = args.resume.parent.parent
        args.output_dir = (args.output_dir or inferred).resolve()
        if args.output_dir != inferred:
            parser.error("--output-dir must match the run directory inferred from --resume")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved = checkpoint["config"]
        args.data_root = (args.data_root or Path(saved["data_root"])).resolve()
        # Exact resume owns all optimization and sampling arguments.
        for key in ("model", "epochs", "batch_size", "accumulate_grad_batches", "patch_size", "lr",
                    "gradient_clip_norm", "num_workers", "precision", "log_interval_steps",
                    "checkpoint_interval_steps"):
            setattr(args, key, saved[key])
    elif not args.data_root or not args.output_dir:
        parser.error("new training requires --data-root and --output-dir")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Full DACG training requires a CUDA GPU")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not support BF16")
    if min(args.epochs, args.batch_size, args.accumulate_grad_batches, args.patch_size) < 1:
        raise ValueError("epochs, batch size, accumulation and patch size must be positive")

    run_dir = args.output_dir.resolve()
    resume = args.resume is not None
    if not resume and run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty run directory: {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    train_ids = validate_partition(args.data_root, "train", NUM_TRAIN_SCENES)
    fingerprint = dataset_fingerprint(args.data_root, "train", train_ids)
    config = _config(args, fingerprint)
    dataset = CDD11FullDataset(args.data_root, "train", train_ids, patch_size=args.patch_size, augment=True)
    probe = DeterministicStepBatchSampler(
        len(dataset), batch_size=args.batch_size, accumulate=args.accumulate_grad_batches,
        epochs=args.epochs, seed=SEED,
    )
    config.update({"train_pairs": len(dataset), "steps_per_epoch": probe.steps_per_epoch,
                   "total_steps": probe.total_steps, "repository": git_state()})
    reproducibility_config = {key: value for key, value in config.items() if key != "repository"}
    config_sha256 = sha256_payload(reproducibility_config)

    if resume:
        saved_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved_checkpoint.get("protocol") != PROTOCOL_NAME:
            raise RuntimeError("Checkpoint does not belong to the full CDD-11 DACG protocol")
        if saved_checkpoint["config_sha256"] != config_sha256:
            raise RuntimeError("Resume configuration or dataset fingerprint differs from checkpoint")
        start_step = int(saved_checkpoint["global_step"])
    else:
        start_step, saved_checkpoint = 0, None
        write_protocol(run_dir / "protocol.json")
        atomic_json(run_dir / "config.json", {**config, "config_sha256": config_sha256})

    sampler = DeterministicStepBatchSampler(
        len(dataset), batch_size=args.batch_size, accumulate=args.accumulate_grad_batches,
        epochs=args.epochs, seed=SEED, start_step=start_step,
    )
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model = build_model(args.model).to(device)
    expected = 30_861_200 if args.model == "DACG_IR" else None
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if expected is not None and parameters != expected:
        raise RuntimeError(f"Original DACG_IR identity changed: {parameters} != {expected}")
    criterion = OriginalDACGLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=15, max_epochs=150)
    if saved_checkpoint:
        model.load_state_dict(saved_checkpoint["model"], strict=True)
        optimizer.load_state_dict(saved_checkpoint["optimizer"])
        scheduler.load_state_dict(saved_checkpoint["scheduler"])
        restore_rng(saved_checkpoint["rng_state"])

    tracker = Tracker(run_dir=run_dir, config={**config, "config_sha256": config_sha256},
                      entity=args.wandb_entity, mode=args.wandb_mode, resume=resume)
    _state(run_dir, "running", start_step, sampler.total_steps)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = start_step
    window = {"loss": 0.0, "l1": 0.0, "fft": 0.0, "grad_norm": 0.0, "seconds": 0.0, "count": 0}
    iterator = iter(loader)
    try:
        while global_step < sampler.total_steps:
            started = time.perf_counter()
            total_loss = total_l1 = total_fft = 0.0
            for _ in range(args.accumulate_grad_batches):
                batch = next(iterator)
                lq = batch["lq"].to(device, non_blocking=True)
                gt = batch["gt"].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                    restored = model(lq)
                    loss, parts = criterion(restored, gt)
                (loss / args.accumulate_grad_batches).backward()
                total_loss += float(loss.detach())
                total_l1 += float(parts["l1"])
                total_fft += float(parts["fft"])
            grad_norm = float(clip_grad_norm_(model.parameters(), args.gradient_clip_norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % sampler.steps_per_epoch == 0:
                scheduler.step()
            elapsed = time.perf_counter() - started
            divisor = args.accumulate_grad_batches
            window["loss"] += total_loss / divisor
            window["l1"] += total_l1 / divisor
            window["fft"] += total_fft / divisor
            window["grad_norm"] += grad_norm
            window["seconds"] += elapsed
            window["count"] += 1

            if global_step % args.log_interval_steps == 0 or global_step == sampler.total_steps:
                count = window["count"]
                metrics = {
                    "train/loss": window["loss"] / count,
                    "train/l1": window["l1"] / count,
                    "train/fft": window["fft"] / count,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "diagnostics/grad_norm": window["grad_norm"] / count,
                    "system/step_seconds": window["seconds"] / count,
                    "system/max_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    "system/max_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                    "train/epoch": global_step / sampler.steps_per_epoch,
                }
                tracker.log(metrics, global_step)
                print(f"step={global_step}/{sampler.total_steps} loss={metrics['train/loss']:.6f} "
                      f"lr={metrics['train/lr']:.8g} seconds={metrics['system/step_seconds']:.4f}", flush=True)
                window = {key: 0 if key == "count" else 0.0 for key in window}

            epoch_boundary = global_step % sampler.steps_per_epoch == 0
            interval = args.checkpoint_interval_steps > 0 and global_step % args.checkpoint_interval_steps == 0
            if epoch_boundary or interval:
                payload = _checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, config=config,
                                      config_sha256=config_sha256, global_step=global_step,
                                      status="running", tracker=tracker)
                atomic_torch_save(run_dir / "checkpoints" / "latest.pth", payload)
                _state(run_dir, "running", global_step, sampler.total_steps)

        final = _checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, config=config,
                            config_sha256=config_sha256, global_step=global_step,
                            status="completed", tracker=tracker)
        atomic_torch_save(run_dir / "checkpoints" / "latest.pth", final)
        atomic_torch_save(run_dir / "checkpoints" / "final.pth", final)
        _state(run_dir, "completed", global_step, sampler.total_steps)
        tracker.finish("completed")
        print(f"CDD-11 full DACG training completed: {run_dir}", flush=True)
    except KeyboardInterrupt:
        payload = _checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, config=config,
                              config_sha256=config_sha256, global_step=global_step,
                              status="interrupted", tracker=tracker)
        atomic_torch_save(run_dir / "checkpoints" / "latest.pth", payload)
        _state(run_dir, "interrupted", global_step, sampler.total_steps, "KeyboardInterrupt")
        tracker.finish("interrupted")
        raise
    except Exception as error:
        _state(run_dir, "failed", global_step, sampler.total_steps, f"{type(error).__name__}: {error}")
        tracker.finish("failed")
        raise


if __name__ == "__main__":
    main()
