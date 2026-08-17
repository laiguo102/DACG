"""Native-resolution AIO3 validation and fixed-sample media."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import AIO3ManifestDataset
from .metrics import image_metrics
from .results import aggregate_metrics


def _extract_ids(value: Any, *, strings_are_ids: bool = False) -> list[str]:
    output: list[str] = []
    if isinstance(value, str) and strings_are_ids:
        output.append(value)
    elif isinstance(value, list):
        for item in value:
            output.extend(_extract_ids(item, strings_are_ids=True))
    elif isinstance(value, dict):
        if "id" in value:
            output.append(str(value["id"]))
        elif "sample_id" in value:
            output.append(str(value["sample_id"]))
        else:
            preferred = [key for key in ("samples", "sample_ids", "visual_samples") if key in value]
            items = (value[key] for key in preferred) if preferred else (
                item for item in value.values() if isinstance(item, (list, dict))
            )
            for item in items:
                output.extend(_extract_ids(item, strings_are_ids=True))
    return output


def visual_sample_ids(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        ids = list(dict.fromkeys(_extract_ids(json.load(handle))))
    if len(ids) != 14:
        raise ValueError(f"visual_samples.json must identify exactly 14 samples, found {len(ids)}")
    return ids


def _display_rgb(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()


def _display_abs_error(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().float().mean(0).clamp(0, 0.25).cpu().numpy() / 0.25
    return np.repeat(value[..., None], 3, axis=2)


def _display_signed(tensor: torch.Tensor) -> np.ndarray:
    value = (tensor.detach().float().mean(0).clamp(-0.25, 0.25).cpu().numpy() / 0.25 + 1) / 2
    return np.stack((value, (1 - np.abs(value - 0.5) * 2) * 0.75, 1 - value), axis=-1)


def _save_display(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8), mode="RGB").save(path)


def validate(
    model: torch.nn.Module,
    *,
    manifest: str | Path,
    visual_samples: str | Path,
    data_root: str | Path | None,
    device: torch.device,
    num_workers: int,
    global_step: int,
    media: bool,
    output_dir: str | Path,
    wandb_module: Any | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]], Any | None, np.ndarray | None]:
    dataset = AIO3ManifestDataset(manifest, split="val", data_root=data_root)
    if len(dataset) != 420:
        raise ValueError(f"AIO3 validation manifest must contain 420 records, found {len(dataset)}")
    fixed_ids = set(visual_sample_ids(visual_samples)) if media else set()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    rows: list[dict[str, Any]] = []
    negative: dict[str, list[float]] = defaultdict(list)
    table_rows: list[list[Any]] = []
    histogram_values: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            sample_id, task = batch["id"][0], batch["task"][0]
            sigma = int(batch["sigma"][0])
            degraded, target = batch["input"].to(device), batch["target"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                restored_raw = model(degraded)
            restored = restored_raw.float()
            score = image_metrics(restored, target.float())
            residual = restored - degraded.float()
            neg = float((residual < 0).float().mean())
            negative[task].append(neg)
            rows.append({"task": task, "sigma": sigma if task == "denoise" else "", "sample_id": sample_id, **score})
            if sample_id in fixed_ids:
                histogram_values.append(residual.detach().float().cpu().numpy().reshape(-1)[::64])
                arrays = {
                    "input": _display_rgb(degraded[0]), "prediction": _display_rgb(restored[0]),
                    "target": _display_rgb(target[0]),
                    "absolute_error": _display_abs_error((restored - target).abs()[0]),
                    "signed_residual": _display_signed(residual[0]),
                }
                safe_id = sample_id.replace("/", "_").replace("\\", "_")
                sample_root = Path(output_dir) / f"media_step_{global_step:06d}" / safe_id
                for name, array in arrays.items():
                    _save_display(array, sample_root.with_name(sample_root.name + f"_{name}.png"))
                images = [wandb_module.Image(array) for array in arrays.values()] if wandb_module else list(arrays.values())
                table_rows.append([
                    global_step, task, sigma if task == "denoise" else None, sample_id, *images,
                    score["psnr"], score["ssim"], float(residual.mean()), neg,
                ])
    summary = aggregate_metrics(rows)
    metrics: dict[str, float] = {}
    for sigma in (15, 25, 50):
        for name in ("psnr", "ssim"):
            metrics[f"val/denoise/sigma{sigma}/{name}"] = float(summary[f"denoise/sigma{sigma}"][name])
    for source, target_name in (("denoise/mean", "denoise/mean"), ("derain", "derain"), ("dehaze", "dehaze"), ("macro", "macro")):
        for name in ("psnr", "ssim"):
            metrics[f"val/{target_name}/{name}"] = float(summary[source][name])
    for task in ("denoise", "derain", "dehaze"):
        metrics[f"diagnostics/{task}/residual_negative_fraction"] = sum(negative[task]) / len(negative[task])
    table = None
    if media:
        if len(table_rows) != 14:
            raise ValueError(f"fixed validation table must contain 14 rows, found {len(table_rows)}")
        if wandb_module:
            table = wandb_module.Table(columns=[
                "global_step", "task", "sigma", "sample_id", "input", "prediction", "target",
                "absolute_error", "signed_residual", "psnr", "ssim", "residual_mean",
                "residual_negative_fraction",
            ], data=table_rows)
    model.train()
    histogram = np.concatenate(histogram_values) if histogram_values else None
    return metrics, rows, table, histogram
