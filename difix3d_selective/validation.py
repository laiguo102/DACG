"""Deterministic selective-Difix validation, metrics, and W&B media."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from cdd11_full.metrics import rgb_psnr, rgb_ssim
from .protocol import PAIR_FOLDERS, directed_tasks


METRIC_NAMES = (
    "psnr",
    "ssim",
    "lpips",
    "psnr_vs_clean",
    "ssim_vs_clean",
    "coarse_psnr",
    "coarse_ssim",
)


def stratified_indices(records: list[dict], total: int) -> list[int]:
    """Round-robin fixed records across all directed degradation tasks."""

    if total <= 0 or total >= len(records):
        return list(range(len(records)))
    groups: dict[tuple, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        key = (record.get("pair_id"), record.get("remove"), record.get("preserve"))
        groups[key].append(index)
    ordered_groups = [groups[key] for key in sorted(groups, key=lambda item: tuple(map(str, item)))]
    selected: list[int] = []
    offset = 0
    while len(selected) < total:
        added = False
        for group in ordered_groups:
            if offset < len(group):
                selected.append(group[offset])
                added = True
                if len(selected) == total:
                    break
        if not added:
            break
        offset += 1
    return selected


def display_image(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().cpu().add(1).mul(0.5).clamp(0, 1)


def comparison_image(
    source: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
) -> torch.Tensor:
    """Join degraded, DACG, selective target, final output, and clean GT."""

    panels = (
        display_image(source[0, 1]),
        display_image(source[0, 0]),
        display_image(target[0]),
        display_image(prediction[0]),
        display_image(ground_truth[0]),
    )
    return torch.cat(panels, dim=-1)


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader,
    lpips_model: torch.nn.Module,
    accelerator: Any,
    *,
    limit: int = 0,
    visualization_limit: int = 0,
) -> tuple[dict[str, float], list[tuple[torch.Tensor, str]]]:
    """Evaluate final-vs-target primary metrics and clean/coarse diagnostics."""

    model.eval()
    rows: list[torch.Tensor] = []
    visualizations: list[tuple[torch.Tensor, str]] = []
    try:
        for index, batch in enumerate(loader):
            if limit > 0 and index >= limit:
                break
            source = batch["conditioning_pixel_values"]
            target = batch["output_pixel_values"]
            ground_truth = batch["ground_truth_pixel_values"]
            prediction = model(source, prompt_tokens=batch["input_ids"])

            prediction_01 = prediction.float().add(1).mul(0.5).clamp(0, 1)
            coarse_01 = source[:, 0].float().add(1).mul(0.5).clamp(0, 1)
            target_01 = target.float().add(1).mul(0.5).clamp(0, 1)
            ground_truth_01 = ground_truth.float().add(1).mul(0.5).clamp(0, 1)
            perceptual = lpips_model(prediction.float(), target.float()).mean()
            metric_values = (
                    rgb_psnr(prediction_01, target_01),
                    rgb_ssim(prediction_01, target_01),
                    perceptual.detach().item(),
                    rgb_psnr(prediction_01, ground_truth_01),
                    rgb_ssim(prediction_01, ground_truth_01),
                    rgb_psnr(coarse_01, target_01),
                    rgb_ssim(coarse_01, target_01),
                )
            pair_id = int(batch["pair_id"][0]) if "pair_id" in batch else -1
            direction = -1
            if pair_id in PAIR_FOLDERS and "remove" in batch:
                remove = batch["remove"][0]
                for task_index, task in enumerate(directed_tasks(pair_id)):
                    if task.remove == remove:
                        direction = task_index
                        break
            row = prediction.new_tensor(
                (*metric_values, pair_id, direction), dtype=torch.float32
            )
            rows.append(row)
            if accelerator.is_main_process and len(visualizations) < visualization_limit:
                caption = (
                    f"{batch['sample_id'][0]} | {batch['prompt'][0]} | "
                    "left to right: degraded | DACG coarse | selective target | "
                    "Difix final | clean GT"
                )
                visualizations.append(
                    (comparison_image(source, target, prediction, ground_truth), caption)
                )
    finally:
        model.train()
        accelerator.unwrap_model(model).set_train()

    if not rows:
        raise ValueError("validation loader produced no samples")
    gathered = accelerator.gather_for_metrics(torch.stack(rows))
    metric_rows = gathered[:, : len(METRIC_NAMES)].float()
    means = metric_rows.mean(0).cpu().tolist()
    if len(METRIC_NAMES) != len(means):
        raise RuntimeError(
            f"Expected {len(METRIC_NAMES)} validation metrics, got {len(means)}"
        )
    metrics = dict(zip(METRIC_NAMES, means))
    task_codes = gathered[:, len(METRIC_NAMES) : len(METRIC_NAMES) + 2].long()
    for pair_id, pair in sorted(PAIR_FOLDERS.items()):
        for direction, task in enumerate(directed_tasks(pair_id)):
            mask = (task_codes[:, 0] == pair_id) & (task_codes[:, 1] == direction)
            if not bool(mask.any()):
                continue
            task_means = metric_rows[mask].mean(0).cpu().tolist()
            task_prefix = f"tasks/{pair}/remove_{task.remove}"
            for name, value in zip(METRIC_NAMES, task_means):
                metrics[f"{task_prefix}/{name}"] = value
    return metrics, visualizations


def wandb_images(visualizations: list[tuple[torch.Tensor, str]]) -> list[Any]:
    import wandb

    return [wandb.Image(image, caption=caption) for image, caption in visualizations]
