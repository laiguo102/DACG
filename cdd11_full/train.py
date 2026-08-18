"""Train original DACG-IR under CDD-11-v1 with its paper RGB+FFT loss."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
import numpy as np
from PIL import Image

from .data import build_eval_loader, build_train_loader
from .metrics import rgb_psnr, rgb_ssim, summarize
from .model import OriginalDACGLoss, architecture_metadata, build_model
from .protocol import DEGRADATIONS, OBJECTIVE_VARIANT, PROTOCOL_NAME, RUN_PROFILES, SEED
from .runtime import (append_jsonl, atomic_json, atomic_torch_save, atomic_yaml,
                      file_sha256, git_state, restore_rng, rng_state, seed_all)
from .tracking import Tracker, WANDB_VERSION

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FILES = ("train.jsonl", "val.jsonl", "test.jsonl")
AUXILIARY_FILES = ("data_audit.json", "visual_samples.json")


class WarmupCosineScheduler:
    def __init__(self, optimizer, base_lr: float, min_lr: float, warmup_steps: int, max_steps: int):
        self.optimizer, self.base_lr, self.min_lr = optimizer, base_lr, min_lr
        self.warmup_steps, self.max_steps, self.completed_steps = warmup_steps, max_steps, 0
        self._apply()

    def _lr(self):
        step = self.completed_steps
        if step == self.max_steps:
            return self.min_lr
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        if self.warmup_steps == self.max_steps:
            return self.base_lr
        if self.warmup_steps == 0:
            progress = 1.0 if self.max_steps == 1 else step / (self.max_steps - 1)
        else:
            progress = (step - self.warmup_steps + 1) / (self.max_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def _apply(self):
        value = self._lr()
        for group in self.optimizer.param_groups:
            group["lr"] = value

    def step(self):
        if self.completed_steps >= self.max_steps:
            raise RuntimeError("Scheduler was stepped beyond max_steps")
        self.completed_steps += 1
        self._apply()

    def state_dict(self):
        return {"scheduler_type": "warmup_cosine", "base_lr": self.base_lr,
                "min_lr": self.min_lr, "warmup_steps": self.warmup_steps,
                "max_steps": self.max_steps, "completed_steps": self.completed_steps}

    def load_state_dict(self, value):
        for key in ("scheduler_type", "base_lr", "min_lr", "warmup_steps", "max_steps", "completed_steps"):
            if key not in value:
                raise RuntimeError(f"Scheduler checkpoint misses {key}")
        if value["scheduler_type"] != "warmup_cosine":
            raise RuntimeError("Scheduler type mismatch")
        if any(value[key] != getattr(self, key) for key in ("base_lr", "min_lr", "warmup_steps", "max_steps")):
            raise RuntimeError("Scheduler configuration mismatch")
        self.completed_steps = int(value["completed_steps"])
        self._apply()


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _verify_manifests(directory: Path):
    directory = directory.expanduser().resolve()
    audit_path = directory / "data_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("protocol") != PROTOCOL_NAME or audit.get("status") != "pass":
        raise RuntimeError("Manifest bundle did not pass cdd11-v1 audit")
    hashes = {}
    for filename in MANIFEST_FILES:
        path = directory / filename
        actual = file_sha256(path)
        expected = audit.get("manifests", {}).get(filename, {}).get("sha256")
        if actual != expected:
            raise RuntimeError(f"Manifest SHA256 mismatch: {filename}")
        hashes[filename] = actual
    visual = directory / "visual_samples.json"
    hashes["visual_samples.json"] = file_sha256(visual)
    if hashes["visual_samples.json"] != audit.get("visual_samples", {}).get("sha256"):
        raise RuntimeError("visual_samples.json SHA256 mismatch")
    hashes["data_audit.json"] = file_sha256(audit_path)
    return directory, hashes


def _config(args, run_dir: Path, hashes, repository):
    profile = RUN_PROFILES[args.run_kind]
    return {
        "protocol": PROTOCOL_NAME, "run_kind": args.run_kind, "run_name": args.run_name,
        "seed": SEED,
        "model": {**architecture_metadata("DACG_IR"), "id": "dacg"},
        "data": {"patch_size": 256, "effective_batch_size": 11,
                 "samples_per_degradation": {value: 1 for value in DEGRADATIONS},
                 "microbatch_size": args.microbatch_size, "num_workers": args.num_workers,
                 "manifest_sha256": dict(hashes)},
        "training": {"max_steps": profile["max_steps"], "precision": "bf16",
                     "loss": "rgb_l1_plus_0.1_fourier_l1", "objective_variant": OBJECTIVE_VARIANT,
                     "protocol_exception": "loss differs from UNet/Uformer cdd11-v1 RGB-L1 objective",
                     "grad_clip_norm": 1.0, "cudnn_benchmark": True,
                     "deterministic_algorithms": False, "pretrained": False, "ema": False,
                     "labels_as_model_input": False, "clamp_before_loss": False},
        "optimizer": {"name": "adamw", "learning_rate": 2e-4,
                      "betas": [0.9, 0.999], "weight_decay": 1e-4},
        "scheduler": {"name": "linear_warmup_cosine", "warmup_steps": min(2000, profile["max_steps"]),
                      "min_learning_rate": 1e-6, "unit": "optimizer_step"},
        "validation": {"interval_steps": profile["validation_interval"],
                       "selection_metric": "macro/psnr", "inference_mode": args.inference_mode,
                       "tile_size": 512, "tile_overlap": 128, "tta": False},
        "checkpoint": {"interval_steps": profile["validation_interval"], "milestone_interval_steps": 50_000},
        "monitoring": {"provider": "wandb", "version": WANDB_VERSION, "mode": args.wandb_mode,
                       "entity": args.wandb_entity, "project": args.wandb_project,
                       "group": f"{PROTOCOL_NAME}-{args.run_kind}-{args.inference_mode}-seed{SEED}",
                       "wandb_run_id": uuid.uuid4().hex[:16] if args.wandb_mode != "disabled" else None,
                       "tags": [PROTOCOL_NAME, args.run_kind, "dacg", OBJECTIVE_VARIANT,
                                args.inference_mode, f"seed-{SEED}"],
                       "local_backend": "jsonl", "scalar_interval_steps": profile["scalar_interval"]},
        "paths": {"output_root": str(args.output_root.resolve()), "run_dir": str(run_dir),
                  "manifest_dir": str((run_dir / "manifests").resolve())},
        "source": {"repository_commit": repository["commit"],
                   "repository_dirty_at_creation": repository["dirty"]},
    }


def _prepare(args):
    repository = git_state(REPOSITORY_ROOT)
    if repository["dirty"]:
        raise RuntimeError("Commit or stash DACG changes before creating a cdd11-v1 run")
    source, hashes = _verify_manifests(args.manifest_dir)
    run_dir = (args.output_root / "dacg" / args.run_name).resolve()
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    for name in ("checkpoints", "logs", "validation", "wandb"):
        (run_dir / name).mkdir()
    destination = run_dir / "manifests"
    destination.mkdir()
    for filename in (*MANIFEST_FILES, *AUXILIARY_FILES):
        shutil.copy2(source / filename, destination / filename)
    config = _config(args, run_dir, hashes, repository)
    atomic_yaml(run_dir / "config.yaml", config)
    atomic_json(run_dir / "run_state.json", {"status": "created", "global_step": 0, "created_at_utc": _utc()})
    if config["monitoring"]["wandb_run_id"]:
        (run_dir / "wandb_run_id.txt").write_text(config["monitoring"]["wandb_run_id"] + "\n", encoding="utf-8")
    return run_dir, config, None


def _resume(path: Path):
    path = path.resolve()
    if path.name != "latest.pth":
        raise RuntimeError("Exact resume requires checkpoints/latest.pth")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    run_dir = Path(checkpoint["run_dir"]).resolve()
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if checkpoint["config"] != config:
        raise RuntimeError("Checkpoint/config mismatch")
    return run_dir, config, checkpoint


def _state(run_dir, status, step, best, message=None):
    value = {"status": status, "global_step": step, "best_macro_psnr": best.get("macro_psnr"),
             "best_macro_ssim": best.get("macro_ssim"), "best_global_step": best.get("global_step"),
             "updated_at_utc": _utc()}
    if message:
        value["message"] = message
    atomic_json(run_dir / "run_state.json", value)


def _checkpoint(model, optimizer, scheduler, step, best, config, run_dir):
    if scheduler.completed_steps != step:
        raise RuntimeError("Scheduler/global-step mismatch")
    return {"checkpoint_version": 1, "protocol": PROTOCOL_NAME, "objective_variant": OBJECTIVE_VARIANT,
            "global_step": step, "model": model.state_dict(), "architecture": architecture_metadata("DACG_IR"),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "best_metrics": dict(best), "rng_state": rng_state(), "config": config,
            "manifest_sha256": config["data"]["manifest_sha256"],
            "repository_commit": config["source"]["repository_commit"],
            "wandb_run_id": config["monitoring"]["wandb_run_id"],
            "run_dir": str(run_dir), "run_name": config["run_name"]}


def _restore_image(model, value, mode, tile_size=512, overlap=128):
    def forward(item):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(item).float()
    if mode == "native":
        return forward(value)
    height, width = value.shape[-2:]
    stride = tile_size - overlap
    starts = lambda length: ([0] if length <= tile_size else list(range(0, length - tile_size + 1, stride)))
    tops, lefts = starts(height), starts(width)
    if tops[-1] != max(height - tile_size, 0): tops.append(max(height - tile_size, 0))
    if lefts[-1] != max(width - tile_size, 0): lefts.append(max(width - tile_size, 0))
    output = torch.zeros_like(value, dtype=torch.float32)
    weights = torch.zeros((1, 1, height, width), device=value.device)
    for top in tops:
        for left in lefts:
            tile = value[..., top:min(top + tile_size, height), left:min(left + tile_size, width)]
            restored = forward(tile)
            window = torch.outer(torch.hann_window(restored.shape[-2], periodic=False, device=value.device),
                                 torch.hann_window(restored.shape[-1], periodic=False, device=value.device)).clamp_min(1e-3)[None, None]
            output[..., top:top + restored.shape[-2], left:left + restored.shape[-1]] += restored * window
            weights[..., top:top + restored.shape[-2], left:left + restored.shape[-1]] += window
    return output / weights


def _save_display(tensor, path: Path):
    array = tensor.detach().float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(array * 255).astype(np.uint8), "RGB").save(path)


def _validate(model, loader, device, step, config, run_dir):
    visual_spec = json.loads((Path(config["paths"]["manifest_dir"]) / "visual_samples.json").read_text(encoding="utf-8"))
    ordered_visual_ids = [str(value) for value in visual_spec["ordered_sample_ids"]]
    if len(ordered_visual_ids) != 22 or len(set(ordered_visual_ids)) != 22:
        raise RuntimeError("CDD-11-v1 validation requires 22 frozen visual samples")
    requested_visuals, visuals_by_id = set(ordered_visual_ids), {}
    rows, model_was_training = [], model.training
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            degraded, target = batch["degraded"].to(device), batch["target"].to(device)
            prediction = _restore_image(model, degraded, config["validation"]["inference_mode"],
                                        config["validation"]["tile_size"], config["validation"]["tile_overlap"])
            sample_id, degradation = str(batch["sample_id"][0]), str(batch["degradation"][0])
            psnr, ssim = rgb_psnr(prediction, target), rgb_ssim(prediction, target)
            rows.append({"sample_id": sample_id, "degradation": degradation, "psnr": psnr, "ssim": ssim})
            if sample_id in requested_visuals:
                root = run_dir / "validation" / "media" / f"step_{step:06d}" / degradation / sample_id.replace("/", "__")
                paths = {name: root / f"{name}.png" for name in
                         ("input", "prediction", "target", "absolute_error", "signed_residual")}
                _save_display(degraded, paths["input"]); _save_display(prediction, paths["prediction"])
                _save_display(target, paths["target"]); _save_display((prediction - target).abs() / 0.25,
                                                                      paths["absolute_error"])
                _save_display(((prediction - degraded).clamp(-0.25, 0.25) + 0.25) / 0.5,
                              paths["signed_residual"])
                visuals_by_id[sample_id] = {"sample_id": sample_id, "degradation": degradation,
                    "arity": int(batch["arity"][0]), "psnr": psnr, "ssim": ssim,
                    **{f"{name}_path": str(path) for name, path in paths.items()}}
    model.train(model_was_training)
    missing = [value for value in ordered_visual_ids if value not in visuals_by_id]
    if missing:
        raise RuntimeError(f"Frozen validation visual samples not found: {missing}")
    visuals = [visuals_by_id[value] for value in ordered_visual_ids]
    summary = summarize(rows, DEGRADATIONS)
    result = {"protocol": PROTOCOL_NAME, "objective_variant": OBJECTIVE_VARIANT,
              "global_step": step, "summary": summary, "images": len(rows), "visuals": visuals}
    atomic_json(run_dir / "validation" / f"step_{step:06d}.json", result)
    append_jsonl(run_dir / "validation_metrics.jsonl",
                 {"global_step": step, **{f"val/{key}": value for key, value in summary.items()}})
    return summary, visuals


def _backward(model, criterion, batch, microbatch, device):
    degradations, values = list(batch["degradation"]), []
    totals = {"loss": 0.0, "l1": 0.0, "fft": 0.0}
    for start in range(0, 11, microbatch):
        end = min(start + microbatch, 11)
        degraded = batch["degraded"][start:end].to(device, non_blocking=True)
        target = batch["target"][start:end].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(degraded)
        per_loss, per_l1, per_fft = criterion.per_sample(prediction, target)
        if not torch.isfinite(per_loss).all():
            raise FloatingPointError("Non-finite DACG paper loss")
        (per_loss.sum() / 11).backward()
        totals["loss"] += float(per_loss.sum()) / 11
        totals["l1"] += float(per_l1.sum()) / 11
        totals["fft"] += float(per_fft.sum()) / 11
        values.extend(float(value) for value in per_loss.detach().cpu())
    return totals, degradations, values


def run(run_dir, config, checkpoint=None, pause_at_step=None):
    device, max_steps = torch.device("cuda", 0), int(config["training"]["max_steps"])
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    repository = git_state(REPOSITORY_ROOT)
    if repository["dirty"] or repository["commit"] != config["source"]["repository_commit"]:
        raise RuntimeError("Current DACG source differs from frozen run source")
    manifest_dir = Path(config["paths"]["manifest_dir"])
    _, hashes = _verify_manifests(manifest_dir)
    if hashes != config["data"]["manifest_sha256"]:
        raise RuntimeError("Frozen manifest bundle changed")
    model = build_model("DACG_IR").to(device)
    parameters = sum(value.numel() for value in model.parameters())
    if parameters != 30_861_200:
        raise RuntimeError(f"DACG-IR identity mismatch: {parameters}")
    criterion = OriginalDACGLoss()
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_config["learning_rate"],
                                  betas=tuple(optimizer_config["betas"]),
                                  weight_decay=optimizer_config["weight_decay"])
    scheduler = WarmupCosineScheduler(optimizer, optimizer_config["learning_rate"],
                                      config["scheduler"]["min_learning_rate"],
                                      config["scheduler"]["warmup_steps"], max_steps)
    step, best = 0, {"macro_psnr": None, "macro_ssim": None, "global_step": None}
    if checkpoint:
        if checkpoint["repository_commit"] != repository["commit"] or checkpoint["manifest_sha256"] != hashes:
            raise RuntimeError("Resume identity mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng(checkpoint["rng_state"])
        step, best = int(checkpoint["global_step"]), dict(checkpoint["best_metrics"])
    target_step = max_steps if pause_at_step is None else int(pause_at_step)
    if step == max_steps and pause_at_step is None:
        target_step = max_steps
    elif not step < target_step <= max_steps:
        raise ValueError(f"pause target must satisfy {step} < target <= {max_steps}")
    scalar_interval = config["monitoring"]["scalar_interval_steps"]
    if pause_at_step is not None and target_step % scalar_interval:
        raise ValueError("--pause-at-step must align with scalar logging interval")
    loader, train_dataset, sampler = build_train_loader(manifest_dir / "train.jsonl", 256, step,
                                                        target_step - step, SEED,
                                                        config["data"]["num_workers"])
    train_counts = Counter(record.degradation for record in train_dataset.records)
    if len(train_dataset) != 11913 or train_counts != Counter({value: 1083 for value in DEGRADATIONS}):
        raise RuntimeError(f"CDD-11-v1 training manifest identity mismatch: {dict(train_counts)}")
    if sampler.batch_size != 11:
        raise RuntimeError("CDD-11-v1 effective batch must contain 11 samples")
    validation_loader, validation_dataset = build_eval_loader(manifest_dir / "val.jsonl", "val",
                                                               min(config["data"]["num_workers"], 4))
    if len(validation_dataset) != 1100:
        raise RuntimeError("CDD-11-v1 validation requires 1100 images")
    tracker = Tracker(run_dir, config, resume=checkpoint is not None)
    if checkpoint is not None and step > 0 and step % config["validation"]["interval_steps"] == 0:
        result_path = run_dir / "validation" / f"step_{step:06d}.json"
        if result_path.is_file():
            summary = json.loads(result_path.read_text(encoding="utf-8"))["summary"]
        else:
            _state(run_dir, "validating", step, best, "Recovering validation after resume")
            summary, visuals = _validate(model, validation_loader, device, step, config, run_dir)
            tracker.log_validation(summary, visuals, step)
        if best["global_step"] is None or step > int(best["global_step"]):
            if best["macro_psnr"] is None or summary["macro/psnr"] > best["macro_psnr"]:
                best = {"macro_psnr": summary["macro/psnr"], "macro_ssim": summary["macro/ssim"],
                        "global_step": step}
                tracker.update_best(best)
                atomic_torch_save(run_dir / "checkpoints" / "best_macro_psnr.pth",
                                  _checkpoint(model, optimizer, scheduler, step, best, config, run_dir))
            atomic_torch_save(run_dir / "checkpoints" / "latest.pth",
                              _checkpoint(model, optimizer, scheduler, step, best, config, run_dir))
        if best.get("global_step") == step and not (run_dir / "checkpoints" / "best_macro_psnr.pth").is_file():
            atomic_torch_save(run_dir / "checkpoints" / "best_macro_psnr.pth",
                              _checkpoint(model, optimizer, scheduler, step, best, config, run_dir))
    if step == max_steps:
        _state(run_dir, "completed", step, best, "Audited completion recovery")
        tracker.finish()
        return
    _state(run_dir, "running", step, best)
    model.train()
    window = {"loss": 0.0, "l1": 0.0, "fft": 0.0, "grad": 0.0, "seconds": 0.0, "steps": 0,
              **{f"{value}_loss": 0.0 for value in DEGRADATIONS}}
    safe = True
    try:
        for batch in loader:
            safe = False
            if Counter(batch["degradation"]) != Counter({value: 1 for value in DEGRADATIONS}):
                raise RuntimeError("Unbalanced CDD-11 effective batch")
            torch.cuda.synchronize(device)
            started, learning_rate = time.perf_counter(), optimizer.param_groups[0]["lr"]
            optimizer.zero_grad(set_to_none=True)
            losses, degradations, per_sample = _backward(
                model, criterion, batch, config["data"]["microbatch_size"], device)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); step += 1; safe = True
            torch.cuda.synchronize(device)
            window["loss"] += losses["loss"]; window["l1"] += losses["l1"]; window["fft"] += losses["fft"]
            window["grad"] += float(grad); window["seconds"] += time.perf_counter() - started; window["steps"] += 1
            for degradation, value in zip(degradations, per_sample):
                window[f"{degradation}_loss"] += value
            if step % scalar_interval == 0 or step == target_step:
                count = window["steps"]
                metrics = {"train/loss": window["loss"] / count, "train/rgb_l1": window["l1"] / count,
                           "train/fourier_l1_weighted": window["fft"] / count,
                           "train/learning_rate": learning_rate, "train/grad_norm": window["grad"] / count,
                           "train/step_time_seconds": window["seconds"] / count,
                           "system/gpu_memory_allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
                           "system/gpu_memory_reserved_gib": torch.cuda.memory_reserved(device) / 2**30}
                metrics.update({f"train/{value}_loss": window[f"{value}_loss"] / count
                                for value in DEGRADATIONS})
                append_jsonl(run_dir / "train_metrics.jsonl", {"global_step": step, **metrics})
                tracker.log(metrics, step)
                print(f"step={step}/{max_steps} loss={metrics['train/loss']:.6f} "
                      f"rgb={metrics['train/rgb_l1']:.6f} fft={metrics['train/fourier_l1_weighted']:.6f} "
                      f"lr={learning_rate:.8g}", flush=True)
                window = {key: 0 if key == "steps" else 0.0 for key in window}
            should_validate = step % config["validation"]["interval_steps"] == 0 or step == max_steps
            if should_validate:
                atomic_torch_save(run_dir / "checkpoints" / "latest.pth",
                                  _checkpoint(model, optimizer, scheduler, step, best, config, run_dir))
                _state(run_dir, "validating", step, best)
                summary, visuals = _validate(model, validation_loader, device, step, config, run_dir)
                tracker.log_validation(summary, visuals, step)
                if best["macro_psnr"] is None or summary["macro/psnr"] > best["macro_psnr"]:
                    best = {"macro_psnr": summary["macro/psnr"], "macro_ssim": summary["macro/ssim"],
                            "global_step": step}
                    tracker.update_best(best)
                    atomic_torch_save(run_dir / "checkpoints" / "best_macro_psnr.pth",
                                      _checkpoint(model, optimizer, scheduler, step, best, config, run_dir))
                atomic_torch_save(run_dir / "checkpoints" / "latest.pth",
                                  _checkpoint(model, optimizer, scheduler, step, best, config, run_dir))
            if step % config["checkpoint"]["milestone_interval_steps"] == 0:
                atomic_torch_save(run_dir / "checkpoints" / f"step_{step:06d}.pth",
                                  _checkpoint(model, optimizer, scheduler, step, best, config, run_dir))
            if step % scalar_interval == 0 or should_validate:
                _state(run_dir, "completed" if step == max_steps else "running", step, best)
        if pause_at_step is not None:
            atomic_torch_save(run_dir / "checkpoints" / "latest.pth",
                              _checkpoint(model, optimizer, scheduler, step, best, config, run_dir))
            _state(run_dir, "paused", step, best, "Requested safe pause at optimizer boundary")
        elif step == max_steps:
            _state(run_dir, "completed", step, best)
    except KeyboardInterrupt:
        if safe and step > 0:
            atomic_torch_save(run_dir / "checkpoints" / "latest.pth",
                              _checkpoint(model, optimizer, scheduler, step, best, config, run_dir))
        _state(run_dir, "interrupted", step, best, "KeyboardInterrupt")
        raise
    except Exception as error:
        _state(run_dir, "failed", step, best, f"{type(error).__name__}: {error}")
        raise
    finally:
        tracker.finish()


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest-dir", type=Path)
    value.add_argument("--output-root", type=Path)
    value.add_argument("--resume", type=Path)
    value.add_argument("--run-kind", choices=tuple(RUN_PROFILES), default="smoke")
    value.add_argument("--run-name")
    value.add_argument("--num-workers", type=int, default=8)
    value.add_argument("--microbatch-size", type=int, default=1)
    value.add_argument("--inference-mode", choices=("native", "tiled"), default="native")
    value.add_argument("--pause-at-step", type=int)
    value.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    value.add_argument("--wandb-entity")
    value.add_argument("--wandb-project", default="cdd11-restoration")
    return value


def main():
    args = parser().parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CDD-11-v1 DACG training requires a BF16-capable CUDA GPU")
    if args.resume:
        if any(value is not None for value in (args.manifest_dir, args.output_root, args.run_name)):
            raise SystemExit("--resume cannot be combined with manifest/output/run-name")
        run_dir, config, checkpoint = _resume(args.resume)
    else:
        if not args.manifest_dir or not args.output_root or not args.run_name:
            raise SystemExit("New runs require --manifest-dir, --output-root and --run-name")
        if not 1 <= args.microbatch_size <= 11:
            raise SystemExit("--microbatch-size must be between 1 and 11")
        run_dir, config, checkpoint = _prepare(args)
    seed_all(int(config["seed"]))
    print(f"CDD-11 DACG run directory: {run_dir}", flush=True)
    run(run_dir, config, checkpoint, args.pause_at_step)


if __name__ == "__main__":
    main()
