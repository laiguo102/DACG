"""Aggregation and protocol-shaped result files for AIO3-v1."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def _finite_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot aggregate an empty metric group")
    if all(math.isinf(value) and value > 0 for value in values):
        return float("inf")
    return sum(values) / len(values)


def aggregate_metrics(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Aggregate per-image rows by the frozen task-macro definition."""

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        task = str(row["task"])
        if task == "denoise":
            groups[f"denoise/sigma{int(row['sigma'])}"].append(row)
        else:
            groups[task].append(row)

    required = ["denoise/sigma15", "denoise/sigma25", "denoise/sigma50", "derain", "dehaze"]
    missing = [name for name in required if not groups[name]]
    if missing:
        raise ValueError(f"cannot produce AIO3 macro metrics; missing groups: {missing}")

    output: dict[str, dict[str, float | int]] = {}
    for name in required:
        values = groups[name]
        output[name] = {
            "count": len(values),
            "psnr": _finite_mean(float(row["psnr"]) for row in values),
            "ssim": _finite_mean(float(row["ssim"]) for row in values),
        }
    output["denoise/mean"] = {
        "count": sum(int(output[f"denoise/sigma{s}"]["count"]) for s in (15, 25, 50)),
        "psnr": _finite_mean(float(output[f"denoise/sigma{s}"]["psnr"]) for s in (15, 25, 50)),
        "ssim": _finite_mean(float(output[f"denoise/sigma{s}"]["ssim"]) for s in (15, 25, 50)),
    }
    output["macro"] = {
        "count": len(rows),
        "psnr": _finite_mean(float(output[name]["psnr"]) for name in ("denoise/mean", "derain", "dehaze")),
        "ssim": _finite_mean(float(output[name]["ssim"]) for name in ("denoise/mean", "derain", "dehaze")),
    }
    return output


def write_result_tables(
    test_dir: str | Path,
    rows: list[Mapping[str, Any]],
    summary: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> None:
    root = Path(test_dir)
    root.mkdir(parents=True, exist_ok=True)
    per_image_fields = ["dataset", "task", "sigma", "sample_id", "psnr", "ssim", "inference_seconds", "prediction"]
    with (root / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_image_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with (root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric_group", "count", "psnr", "ssim"])
        writer.writeheader()
        for name, values in summary.items():
            writer.writerow({"metric_group": name, **values})

    payload = dict(metadata)
    payload["metrics"] = summary
    with (root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)
        handle.write("\n")


def write_compliance_report(
    path: str | Path,
    *,
    run: Mapping[str, Any],
    result: Mapping[str, Any],
    summary: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
    gallery_png_count: int,
) -> None:
    """Write a compact human-readable report matching the comparison template."""

    def pair(name: str) -> str:
        values = summary[name]
        return f"{float(values['psnr']):.6f} / {float(values['ssim']):.6f}"

    mean_inference = _finite_mean(float(row["inference_seconds"]) for row in rows)
    lines = [
        "# AIO3-v1 model compliance report",
        "",
        f"- protocol: {result.get('protocol', 'aio3-v1')}",
        f"- model_id: {run.get('model_id', 'dacg_ir')}",
        f"- model_description: Degradation-Aware Adaptive Context Gating for Unified Image Restoration",
        f"- model_commit: {result.get('git', {}).get('commit', '')}",
        f"- runner_commit: {result.get('git', {}).get('commit', '')}",
        f"- seed: {run.get('seed', '')}",
        f"- parameters_total: {result.get('parameters_total', '')}",
        f"- parameters_trainable: {result.get('parameters_trainable', '')}",
        f"- precision: {run.get('precision', 'bf16')}",
        "- special_fp32_boundaries: torch.fft",
        "- pretrained: false",
        "- ema: false",
        "- tta: false",
        f"- best_validation_step: {run.get('best_validation_step', '')}",
        f"- best_validation_macro_psnr: {run.get('best', {}).get('val_macro_psnr', '')}",
        f"- best_validation_macro_ssim: {run.get('best', {}).get('val_macro_ssim', '')}",
        f"- best_checkpoint_sha256: {result.get('checkpoint_sha256', '')}",
        f"- manifest_sha256: {result.get('manifest_sha256', {})}",
        f"- run_dir: {result.get('run_dir', '')}",
        f"- wandb_url: {result.get('wandb_url', run.get('wandb_url', ''))}",
        f"- mean_inference_seconds: {mean_inference:.6f}",
        f"- test_predictions_count: {len(rows)}",
        f"- test_per_image_csv_lines: {len(rows) + 1}",
        f"- test_gallery_png_count: {gallery_png_count}",
        "- known_limitations: none declared",
        "",
        "| Test set | PSNR | SSIM | Count |",
        "|---|---:|---:|---:|",
    ]
    labels = [
        ("BSD68 sigma15", "denoise/sigma15"), ("BSD68 sigma25", "denoise/sigma25"),
        ("BSD68 sigma50", "denoise/sigma50"), ("BSD68 mean", "denoise/mean"),
        ("Rain100L", "derain"), ("SOTS Outdoor", "dehaze"), ("AIO3 task macro", "macro"),
    ]
    for label, name in labels:
        psnr, ssim = pair(name).split(" / ")
        lines.append(f"| {label} | {psnr} | {ssim} | {summary[name]['count']} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
