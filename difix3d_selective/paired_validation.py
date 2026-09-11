"""Fixed-seed paired validation for baseline and DAEM-lite CCDD-11 checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from cdd11_full.metrics import rgb_psnr, rgb_ssim

from .cfg_evaluate import validate_full_manifest
from .data import SelectiveDifixDataset, load_records
from .evaluate import (
    _atomic_json,
    _autocast,
    _checkpoint_identity,
    _file_sha256,
    _now,
    _package_version,
    _read_json,
    _resolve_device,
    _safe_id,
    _save_image,
)
from .protocol import selected_pairs
from .validation import stratified_indices


PROTOCOL = "ccdd11-daem-fixed-seed-paired-validation-v1"
QUALITY_METRICS = ("mse_l2", "psnr", "ssim", "lpips_vgg", "dists")
POSITIVE_METRICS = tuple(
    f"{reference}_{metric}"
    for reference in ("target", "clean_gt")
    for metric in QUALITY_METRICS
)
COARSE_METRICS = tuple(f"coarse_target_{metric}" for metric in QUALITY_METRICS)
NEGATIVE_METRICS = ("identity_mae",) + tuple(
    f"identity_{metric}" for metric in QUALITY_METRICS
)
LOWER_SUFFIXES = ("mae", "mse_l2", "lpips_vgg", "dists")


def sample_seed(seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(
        f"ccdd11-daem-paired-v1:{seed}:{sample_id}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _lower_is_better(metric: str) -> bool:
    return metric.endswith(LOWER_SUFFIXES)


class PairedMetricSuite:
    """Measure one prediction against an arbitrary set of references."""

    def __init__(
        self,
        device: torch.device,
        *,
        lpips_model: torch.nn.Module | None = None,
        dists_model: torch.nn.Module | None = None,
    ) -> None:
        if lpips_model is None:
            try:
                import lpips
            except ImportError as error:
                raise RuntimeError(
                    "LPIPS is required; install requirements_difix.txt"
                ) from error
            lpips_model = lpips.LPIPS(net="vgg")
        if dists_model is None:
            try:
                import piq
            except ImportError as error:
                raise RuntimeError(
                    "DISTS is required; install requirements_difix.txt (piq==0.8.0)"
                ) from error
            dists_model = piq.DISTS(reduction="none")
        self.lpips = lpips_model.to(device).eval().requires_grad_(False)
        self.dists = dists_model.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def __call__(
        self, prediction: torch.Tensor, references: Mapping[str, torch.Tensor]
    ) -> dict[str, float]:
        if prediction.shape[0] != 1 or not references:
            raise ValueError("Paired metrics require one prediction and references")
        names = list(references)
        ordered = [references[name] for name in names]
        if any(reference.shape != prediction.shape for reference in ordered):
            raise ValueError("prediction/reference shape mismatch")

        prediction = prediction.float().clamp(-1, 1)
        stacked_prediction = prediction.repeat(len(ordered), 1, 1, 1)
        stacked_references = torch.cat(
            [reference.float().clamp(-1, 1) for reference in ordered], dim=0
        )
        lpips_values = self.lpips(stacked_prediction, stacked_references).reshape(-1)
        prediction_01 = prediction.add(1).mul(0.5).clamp(0, 1)
        references_01 = stacked_references.add(1).mul(0.5).clamp(0, 1)
        dists_values = self.dists(
            prediction_01.repeat(len(ordered), 1, 1, 1), references_01
        ).reshape(-1)
        if lpips_values.numel() != len(ordered) or dists_values.numel() != len(ordered):
            raise RuntimeError("A learned metric did not return one value per reference")

        result: dict[str, float] = {}
        for index, name in enumerate(names):
            reference = ordered[index].float().clamp(-1, 1)
            reference_01 = reference.add(1).mul(0.5).clamp(0, 1)
            result.update(
                {
                    f"{name}_mse_l2": float(F.mse_loss(prediction, reference).item()),
                    f"{name}_psnr": rgb_psnr(prediction_01, reference_01),
                    f"{name}_ssim": rgb_ssim(prediction_01, reference_01),
                    f"{name}_lpips_vgg": float(lpips_values[index].item()),
                    f"{name}_dists": float(dists_values[index].item()),
                }
            )
        return result


def _training_run(checkpoint: Path) -> tuple[Path, Path, dict]:
    run_dir = checkpoint.parent.parent
    preparation_path = run_dir / "prepared" / "split_and_preparation.json"
    manifest = run_dir / "prepared" / "manifests" / "validation.jsonl"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not preparation_path.is_file():
        raise FileNotFoundError(preparation_path)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    preparation = _read_json(preparation_path)
    if preparation.get("dataset") != "CCDD-11":
        raise ValueError("Paired validation requires CCDD-11 training runs")
    if preparation.get("target_kind") != "native_selective_sub_data":
        raise ValueError("Paired validation requires native CCDD-11 selective targets")
    pair_ids = [int(value["id"]) for value in preparation["degradation_pairs"]]
    if [pair_id for pair_id, _ in selected_pairs(pair_ids)] != [1, 2, 3, 4, 5]:
        raise ValueError("Paired validation requires all five degradation pairs")
    return run_dir, manifest, preparation


def _selected_records(records: list[dict], max_samples: int | None) -> list[dict]:
    if max_samples is None:
        return records
    if max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    return [records[index] for index in stratified_indices(records, max_samples)]


def _negative_ids(records: list[dict]) -> list[str]:
    keys = sorted({(str(row["pair"]), str(row["scene_id"])) for row in records})
    return [f"validation/{pair}/{scene}/preserve-both" for pair, scene in keys]


def _record_path(root: Path, role: str, mode: str, record: Mapping) -> Path:
    if mode == "positive":
        task = f"remove-{record['remove']}-preserve-{record['preserve']}"
        return root / role / mode / str(record["pair"]) / task / f"{record['scene_id']}.json"
    return root / role / mode / str(record["pair"]) / f"{record['scene_id']}.json"


def _load_role_records(root: Path, role: str, mode: str) -> dict[str, dict]:
    directory = root / role / mode
    result: dict[str, dict] = {}
    if not directory.is_dir():
        return result
    for path in sorted(directory.rglob("*.json")):
        if path.stat().st_size == 0:
            warnings.warn(
                f"Ignoring empty partial record so it can be recomputed: {path}",
                stacklevel=2,
            )
            continue
        try:
            record = _read_json(path)
        except json.JSONDecodeError:
            if path.read_bytes().strip():
                raise
            warnings.warn(
                f"Ignoring empty partial record so it can be recomputed: {path}",
                stacklevel=2,
            )
            continue
        sample_id = str(record["sample_id"])
        if sample_id in result:
            raise ValueError(f"Duplicate {role}/{mode} record: {sample_id}")
        result[sample_id] = record
    return result


def _write_record_json(path: Path, value: dict) -> None:
    """Persist an independent record using close-to-open friendly direct I/O."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _progress(records_root: Path) -> dict[str, int]:
    return {
        f"{role}_{mode}": len(_load_role_records(records_root, role, mode))
        for role in ("baseline", "candidate")
        for mode in ("positive", "negative")
    }


def _write_state(
    path: Path,
    config: dict,
    progress: Mapping[str, int],
    *,
    status: str,
    phase: str,
    positive_total: int,
    negative_total: int,
    checkpoint_steps: Mapping[str, int] | None = None,
) -> None:
    _atomic_json(
        path,
        {
            "status": status,
            "phase": phase,
            "progress": dict(progress),
            "positive_samples_per_model": positive_total,
            "negative_samples_per_model": negative_total,
            "checkpoint_steps": dict(checkpoint_steps or {}),
            "config": config,
            "updated_at_utc": _now(),
        },
    )


def _tensor_to_pil(value: torch.Tensor) -> Image.Image:
    tensor = value.detach().float().clamp(0, 1)
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = tensor.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.asarray(array), "RGB")


def _save_sheet(panels: list[tuple[str, Image.Image]], path: Path) -> None:
    width, height = panels[0][1].size
    columns = 4
    header = 24
    rows = math.ceil(len(panels) / columns)
    sheet = Image.new("RGB", (columns * width, rows * (height + header)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (label, panel) in enumerate(panels):
        left = (index % columns) * width
        top = (index // columns) * (height + header)
        draw.text((left + 5, top + 5), label, fill="black", font=font)
        sheet.paste(panel, (left, top + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _pil_tensor(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _positive_gallery(
    batch: Mapping,
    candidate_01: torch.Tensor,
    baseline_path: str,
    destination: Path,
) -> None:
    baseline = _pil_tensor(baseline_path)
    candidate = candidate_01[0].detach().float().cpu().clamp(0, 1)
    source = batch["conditioning_pixel_values"]
    target = batch["output_pixel_values"]
    clean = batch["ground_truth_pixel_values"]
    panels = [
        ("double degraded", _tensor_to_pil(source[:, 1].add(1).mul(0.5))),
        ("DACG coarse", _tensor_to_pil(source[:, 0].add(1).mul(0.5))),
        ("selective target", _tensor_to_pil(target.add(1).mul(0.5))),
        ("baseline", _tensor_to_pil(baseline)),
        ("DAEM", _tensor_to_pil(candidate)),
        ("abs model diff", _tensor_to_pil((candidate - baseline).abs())),
        ("clean GT", _tensor_to_pil(clean.add(1).mul(0.5))),
    ]
    _save_sheet(panels, destination)


def _negative_gallery(
    batch: Mapping,
    candidate_01: torch.Tensor,
    baseline_path: str,
    destination: Path,
) -> None:
    baseline = _pil_tensor(baseline_path)
    candidate = candidate_01[0].detach().float().cpu().clamp(0, 1)
    degraded = batch["output_pixel_values"][0].add(1).mul(0.5).float().cpu()
    panels = [
        ("double degraded", _tensor_to_pil(degraded)),
        ("baseline", _tensor_to_pil(baseline)),
        ("DAEM", _tensor_to_pil(candidate)),
        ("baseline abs change", _tensor_to_pil((baseline - degraded).abs())),
        ("DAEM abs change", _tensor_to_pil((candidate - degraded).abs())),
    ]
    _save_sheet(panels, destination)


def _run_prediction(
    model: torch.nn.Module,
    batch: Mapping,
    device: torch.device,
    mixed_precision: str,
    seed: int,
) -> tuple[torch.Tensor, float]:
    source = batch["conditioning_pixel_values"].to(device, non_blocking=True)
    tokens = batch["input_ids"].to(device, non_blocking=True)
    devices = [device.index] if device.type == "cuda" else []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        with torch.inference_mode(), _autocast(device, mixed_precision):
            prediction = model(source, prompt_tokens=tokens)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return prediction, time.perf_counter() - started


def _evaluate_checkpoint(
    *,
    role: str,
    checkpoint: Path,
    manifest: Path,
    selected_positive: list[dict],
    selected_negative_ids: list[str],
    gallery_ids: set[str],
    records_root: Path,
    gallery_root: Path,
    metric_suite: PairedMetricSuite,
    device: torch.device,
    args: argparse.Namespace,
    state_path: Path,
    config: dict,
    checkpoint_steps: dict[str, int],
    progress_counts: dict[str, int],
) -> int:
    from torch.utils.data import DataLoader, Subset
    from tqdm.auto import tqdm

    positive_records = _load_role_records(records_root, role, "positive")
    negative_records = _load_role_records(records_root, role, "negative")
    expected_positive = {str(row["id"]) for row in selected_positive}
    expected_negative = set(selected_negative_ids)
    if set(positive_records) == expected_positive and set(negative_records) == expected_negative:
        return int(next(iter(positive_records.values()))["global_step"])

    torch.manual_seed(args.seed)
    from .model import load_model_from_checkpoint

    model, global_step, metadata = load_model_from_checkpoint(
        checkpoint, expected_dataset="CCDD-11", expected_seed=42
    )
    if role == "baseline" and model.detail_enabled:
        raise ValueError("The baseline checkpoint must be detail-disabled or legacy")
    if role == "candidate" and not model.detail_enabled:
        raise ValueError("The candidate checkpoint must enable DAEM-lite detail blocks")
    if [int(value) for value in metadata.get("pairs", [])] != [1, 2, 3, 4, 5]:
        raise ValueError(f"The {role} checkpoint does not cover all five pairs")
    if float(metadata.get("negative_train_probability", 0.0)) <= 0:
        raise ValueError(f"The {role} checkpoint lacks negative CCDD-11 training")
    model = model.to(device)
    if args.enable_xformers_memory_efficient_attention:
        model.unet.enable_xformers_memory_efficient_attention()
    model.set_eval()
    checkpoint_steps[role] = global_step
    baseline_positive_records = (
        _load_role_records(records_root, "baseline", "positive")
        if role == "candidate"
        else {}
    )
    baseline_negative_records = (
        _load_role_records(records_root, "baseline", "negative")
        if role == "candidate"
        else {}
    )

    positive_dataset = SelectiveDifixDataset(
        manifest, model.tokenizer, resolution=512, training_mode="positive"
    )
    positive_index = {
        str(record["id"]): index for index, record in enumerate(positive_dataset.records)
    }
    pending_positive = sorted(expected_positive - set(positive_records))
    loader = DataLoader(
        Subset(positive_dataset, [positive_index[value] for value in pending_positive]),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    progress = tqdm(
        loader,
        total=len(pending_positive),
        desc=f"{role} positive",
    )
    manifest_by_id = {str(row["id"]): row for row in selected_positive}
    for index, batch in enumerate(progress, start=1):
        sample_id = str(batch["sample_id"][0])
        inference_seed = sample_seed(args.seed, sample_id)
        prediction, elapsed = _run_prediction(
            model, batch, device, args.mixed_precision, inference_seed
        )
        target = batch["output_pixel_values"].to(device, non_blocking=True)
        clean = batch["ground_truth_pixel_values"].to(device, non_blocking=True)
        metrics = metric_suite(prediction, {"target": target, "clean_gt": clean})
        source = batch["conditioning_pixel_values"].to(device, non_blocking=True)
        coarse = source[:, 0]
        if role == "baseline":
            coarse_metrics = metric_suite(coarse, {"coarse_target": target})
        else:
            coarse_metrics = {
                key: float(baseline_positive_records[sample_id][key])
                for key in COARSE_METRICS
            }
        prediction_path = ""
        if sample_id in gallery_ids:
            prediction_01 = prediction.float().add(1).mul(0.5).clamp(0, 1)
            destination = gallery_root / "predictions" / role / "positive" / f"{_safe_id(sample_id)}.png"
            _save_image(prediction_01, destination)
            prediction_path = str(destination)
            if role == "candidate":
                baseline_path = baseline_positive_records[sample_id]["prediction_path"]
                _positive_gallery(
                    batch,
                    prediction_01,
                    baseline_path,
                    gallery_root / "positive" / f"{_safe_id(sample_id)}.png",
                )
        source_record = manifest_by_id[sample_id]
        record = {
            "sample_id": sample_id,
            "scene_id": str(batch["scene_id"][0]),
            "pair_id": int(batch["pair_id"].item()),
            "pair": str(batch["pair"][0]),
            "remove": str(batch["remove"][0]),
            "preserve": str(batch["preserve"][0]),
            "prompt": str(batch["prompt"][0]),
            "coarse_path": source_record["image"],
            "degraded_path": source_record["ref_image"],
            "target_path": source_record["target_image"],
            "clean_gt_path": source_record["clear_image"],
            "role": role,
            "mode": "positive",
            "global_step": global_step,
            "inference_seed": inference_seed,
            "inference_time_seconds": elapsed,
            "prediction_path": prediction_path,
            **metrics,
            **coarse_metrics,
        }
        _write_record_json(_record_path(records_root, role, "positive", record), record)
        positive_records[sample_id] = record
        progress_counts[f"{role}_positive"] = len(positive_records)
        if index % 10 == 0 or index == len(pending_positive):
            _write_state(
                state_path,
                config,
                progress_counts,
                status="evaluating",
                phase=f"{role}_positive",
                positive_total=len(expected_positive),
                negative_total=len(expected_negative),
                checkpoint_steps=checkpoint_steps,
            )

    negative_dataset = SelectiveDifixDataset(
        manifest,
        model.tokenizer,
        resolution=512,
        training_mode="negative",
        deduplicate_negative=True,
    )
    negative_index = {
        (
            f"{record.get('split', '')}/{record.get('pair', '')}/"
            f"{record.get('scene_id', '')}/preserve-both"
        ).lstrip("/"): index
        for index, record in enumerate(negative_dataset.records)
    }
    pending_negative = sorted(expected_negative - set(negative_records))
    negative_loader = DataLoader(
        Subset(negative_dataset, [negative_index[value] for value in pending_negative]),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    progress = tqdm(
        negative_loader,
        total=len(pending_negative),
        desc=f"{role} negative",
    )
    negative_gallery_ids = {
        f"validation/{row['pair']}/{row['scene_id']}/preserve-both"
        for row in selected_positive
        if str(row["id"]) in gallery_ids
    }
    for index, batch in enumerate(progress, start=1):
        sample_id = str(batch["sample_id"][0])
        inference_seed = sample_seed(args.seed, sample_id)
        prediction, elapsed = _run_prediction(
            model, batch, device, args.mixed_precision, inference_seed
        )
        identity = batch["output_pixel_values"].to(device, non_blocking=True)
        measured = metric_suite(prediction, {"identity": identity})
        prediction_01 = prediction.float().add(1).mul(0.5).clamp(0, 1)
        identity_01 = identity.float().add(1).mul(0.5).clamp(0, 1)
        measured["identity_mae"] = float((prediction_01 - identity_01).abs().mean().item())
        prediction_path = ""
        if sample_id in negative_gallery_ids:
            destination = gallery_root / "predictions" / role / "negative" / f"{_safe_id(sample_id)}.png"
            _save_image(prediction_01, destination)
            prediction_path = str(destination)
            if role == "candidate":
                baseline_path = baseline_negative_records[sample_id]["prediction_path"]
                _negative_gallery(
                    batch,
                    prediction_01,
                    baseline_path,
                    gallery_root / "negative" / f"{_safe_id(sample_id)}.png",
                )
        record = {
            "sample_id": sample_id,
            "scene_id": str(batch["scene_id"][0]),
            "pair_id": int(batch["pair_id"].item()),
            "pair": str(batch["pair"][0]),
            "prompt": str(batch["prompt"][0]),
            "role": role,
            "mode": "negative",
            "global_step": global_step,
            "inference_seed": inference_seed,
            "inference_time_seconds": elapsed,
            "prediction_path": prediction_path,
            **measured,
        }
        _write_record_json(_record_path(records_root, role, "negative", record), record)
        negative_records[sample_id] = record
        progress_counts[f"{role}_negative"] = len(negative_records)
        if index % 10 == 0 or index == len(pending_negative):
            _write_state(
                state_path,
                config,
                progress_counts,
                status="evaluating",
                phase=f"{role}_negative",
                positive_total=len(expected_positive),
                negative_total=len(expected_negative),
                checkpoint_steps=checkpoint_steps,
            )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return global_step


def paired_rows(
    baseline: Mapping[str, dict],
    candidate: Mapping[str, dict],
    *,
    metrics: tuple[str, ...],
    mode: str,
) -> list[dict]:
    if set(baseline) != set(candidate):
        raise ValueError(f"The {mode} sample sets differ")
    rows = []
    identity_fields = ("scene_id", "pair_id", "pair", "prompt", "inference_seed")
    if mode == "positive":
        identity_fields += ("remove", "preserve", "coarse_path", "degraded_path", "target_path", "clean_gt_path")
    for sample_id in sorted(baseline):
        left = baseline[sample_id]
        right = candidate[sample_id]
        for field in identity_fields:
            if left[field] != right[field]:
                raise ValueError(f"Paired protocol mismatch for {sample_id}: {field}")
        row = {"mode": mode, "sample_id": sample_id}
        for field in identity_fields:
            row[field] = left[field]
        for metric in metrics:
            baseline_value = float(left[metric])
            candidate_value = float(right[metric])
            raw_delta = candidate_value - baseline_value
            advantage = -raw_delta if _lower_is_better(metric) else raw_delta
            row[f"baseline_{metric}"] = baseline_value
            row[f"candidate_{metric}"] = candidate_value
            row[f"raw_delta_{metric}"] = raw_delta
            row[f"candidate_advantage_{metric}"] = advantage
            row[f"candidate_wins_{metric}"] = int(advantage > 0)
        if mode == "positive":
            for metric in COARSE_METRICS:
                row[metric] = float(left[metric])
        rows.append(row)
    return rows


def _point_estimate(rows: list[dict], field: str, *, task_macro: bool) -> float:
    if not task_macro:
        return math.fsum(float(row[field]) for row in rows) / len(rows)
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = f"{row['pair']}:remove-{row['remove']}:preserve-{row['preserve']}"
        by_task[key].append(float(row[field]))
    return math.fsum(math.fsum(values) / len(values) for values in by_task.values()) / len(by_task)


def _cluster_bootstrap_ci(
    rows: list[dict],
    field: str,
    *,
    resamples: int,
    rng: np.random.Generator,
    task_macro: bool = False,
) -> tuple[float, float]:
    scenes = sorted({str(row["scene_id"]) for row in rows})
    scene_index = {scene: index for index, scene in enumerate(scenes)}
    if task_macro:
        tasks = sorted(
            {
                f"{row['pair']}:remove-{row['remove']}:preserve-{row['preserve']}"
                for row in rows
            }
        )
    else:
        tasks = ["all"]
    task_index = {task: index for index, task in enumerate(tasks)}
    sums = np.zeros((len(scenes), len(tasks)), dtype=np.float64)
    counts = np.zeros_like(sums)
    for row in rows:
        task = (
            f"{row['pair']}:remove-{row['remove']}:preserve-{row['preserve']}"
            if task_macro
            else "all"
        )
        i = scene_index[str(row["scene_id"])]
        j = task_index[task]
        sums[i, j] += float(row[field])
        counts[i, j] += 1
    estimates = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 512):
        stop = min(start + 512, resamples)
        indices = rng.integers(0, len(scenes), size=(stop - start, len(scenes)))
        sampled_sums = sums[indices].sum(axis=1)
        sampled_counts = counts[indices].sum(axis=1)
        estimates[start:stop] = np.nanmean(
            np.divide(
                sampled_sums,
                sampled_counts,
                out=np.full_like(sampled_sums, np.nan),
                where=sampled_counts > 0,
            ),
            axis=1,
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def summarize_effects(
    positive_rows: list[dict],
    negative_rows: list[dict],
    *,
    resamples: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    if resamples < 100:
        raise ValueError("--bootstrap-resamples must be at least 100")
    rng = np.random.default_rng(seed)
    effects: list[dict] = []
    conditions: list[dict] = []

    def add_group(mode: str, group: str, condition: str, rows: list[dict], metrics: tuple[str, ...], *, task_macro: bool = False) -> None:
        for metric in metrics:
            baseline_field = f"baseline_{metric}"
            candidate_field = f"candidate_{metric}"
            advantage_field = f"candidate_advantage_{metric}"
            baseline_mean = _point_estimate(rows, baseline_field, task_macro=task_macro)
            candidate_mean = _point_estimate(rows, candidate_field, task_macro=task_macro)
            raw_delta = candidate_mean - baseline_mean
            advantage = _point_estimate(rows, advantage_field, task_macro=task_macro)
            low, high = _cluster_bootstrap_ci(
                rows,
                advantage_field,
                resamples=resamples,
                rng=rng,
                task_macro=task_macro,
            )
            conclusion = "inconclusive"
            if low > 0:
                conclusion = "improved"
            elif high < 0:
                conclusion = "degraded"
            effects.append(
                {
                    "mode": mode,
                    "group": group,
                    "condition": condition,
                    "metric": metric,
                    "better": "lower" if _lower_is_better(metric) else "higher",
                    "images": len(rows),
                    "scenes": len({str(row["scene_id"]) for row in rows}),
                    "baseline_mean": baseline_mean,
                    "candidate_mean": candidate_mean,
                    "raw_delta_mean": raw_delta,
                    "candidate_advantage_mean": advantage,
                    "ci95_low": low,
                    "ci95_high": high,
                    "candidate_win_rate": _point_estimate(
                        rows, f"candidate_wins_{metric}", task_macro=task_macro
                    ),
                    "conclusion": conclusion,
                }
            )
            for model_name, value in (("baseline", baseline_mean), ("candidate", candidate_mean)):
                conditions.append(
                    {
                        "mode": mode,
                        "group": group,
                        "condition": condition,
                        "model": model_name,
                        "metric": metric,
                        "images": len(rows),
                        "scenes": len({str(row["scene_id"]) for row in rows}),
                        "mean": value,
                    }
                )

    by_task: dict[str, list[dict]] = defaultdict(list)
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for row in positive_rows:
        task = f"{row['pair']}:remove-{row['remove']}:preserve-{row['preserve']}"
        by_task[task].append(row)
        by_pair[str(row["pair"])].append(row)
    for name in sorted(by_task):
        add_group("positive", "directed_task", name, by_task[name], POSITIVE_METRICS)
    for name in sorted(by_pair):
        add_group("positive", "pair", name, by_pair[name], POSITIVE_METRICS)
    add_group("positive", "overall", "micro", positive_rows, POSITIVE_METRICS)
    add_group(
        "positive",
        "overall",
        "task_macro",
        positive_rows,
        POSITIVE_METRICS,
        task_macro=True,
    )

    def add_coarse_reference(
        group: str,
        condition: str,
        rows: list[dict],
        *,
        task_macro: bool = False,
    ) -> None:
        for metric in COARSE_METRICS:
            conditions.append(
                {
                    "mode": "positive",
                    "group": group,
                    "condition": condition,
                    "model": "coarse",
                    "metric": metric.removeprefix("coarse_"),
                    "images": len(rows),
                    "scenes": len({str(row["scene_id"]) for row in rows}),
                    "mean": _point_estimate(rows, metric, task_macro=task_macro),
                }
            )

    for name in sorted(by_task):
        add_coarse_reference("directed_task", name, by_task[name])
    for name in sorted(by_pair):
        add_coarse_reference("pair", name, by_pair[name])
    add_coarse_reference("overall", "micro", positive_rows)
    add_coarse_reference("overall", "task_macro", positive_rows, task_macro=True)

    negative_by_pair: dict[str, list[dict]] = defaultdict(list)
    for row in negative_rows:
        negative_by_pair[str(row["pair"])].append(row)
    for name in sorted(negative_by_pair):
        add_group("negative", "pair", name, negative_by_pair[name], NEGATIVE_METRICS)
    add_group("negative", "overall", "micro", negative_rows, NEGATIVE_METRICS)
    return effects, conditions


def build_decision(effects: list[dict]) -> dict:
    def find(mode: str, group: str, condition: str, metric: str) -> dict:
        return next(
            row
            for row in effects
            if row["mode"] == mode
            and row["group"] == group
            and row["condition"] == condition
            and row["metric"] == metric
        )

    primary = find("positive", "overall", "task_macro", "target_psnr")
    task_psnr = [
        row
        for row in effects
        if row["mode"] == "positive"
        and row["group"] == "directed_task"
        and row["metric"] == "target_psnr"
    ]
    perceptual = [
        find("positive", "overall", "task_macro", metric)
        for metric in ("target_lpips_vgg", "target_dists")
    ]
    negative = [
        row
        for row in effects
        if row["mode"] == "negative" and row["group"] == "overall"
    ]
    checks = {
        "task_macro_psnr_improved": primary["conclusion"] == "improved",
        "at_least_six_tasks_positive": sum(
            float(row["candidate_advantage_mean"]) > 0 for row in task_psnr
        )
        >= 6,
        "no_task_psnr_clearly_degraded": not any(
            row["conclusion"] == "degraded" for row in task_psnr
        ),
        "perceptual_metrics_not_clearly_degraded": not any(
            row["conclusion"] == "degraded" for row in perceptual
        ),
        "negative_identity_not_clearly_degraded": not any(
            row["conclusion"] == "degraded" for row in negative
        ),
    }
    return {
        "primary_metric": primary,
        "positive_psnr_tasks": len(task_psnr),
        "positive_mean_psnr_task_count": sum(
            float(row["candidate_advantage_mean"]) > 0 for row in task_psnr
        ),
        "checks": checks,
        "statistical_gate": "passed" if all(checks.values()) else "failed",
        "final_gate": "visual_review_required" if all(checks.values()) else "failed",
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    if args.seed != 42:
        raise ValueError("CCDD-11 paired validation requires --seed 42")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.num_gallery_samples < 0:
        raise ValueError("--num-gallery-samples must be non-negative")

    baseline_checkpoint = args.baseline_checkpoint.resolve()
    candidate_checkpoint = args.candidate_checkpoint.resolve()
    _, baseline_manifest, baseline_preparation = _training_run(baseline_checkpoint)
    _, candidate_manifest, candidate_preparation = _training_run(candidate_checkpoint)
    baseline_manifest_sha = _file_sha256(baseline_manifest)
    candidate_manifest_sha = _file_sha256(candidate_manifest)
    if baseline_manifest_sha != candidate_manifest_sha:
        raise ValueError("Baseline and candidate validation manifest SHA256 differ")
    manifest_records = load_records(baseline_manifest)
    validate_full_manifest(manifest_records)
    selected_positive = _selected_records(manifest_records, args.max_samples)
    selected_negative_ids = _negative_ids(selected_positive)
    gallery_ids = {
        str(selected_positive[index]["id"])
        for index in stratified_indices(selected_positive, args.num_gallery_samples)
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    records_root = output_dir / "records"
    gallery_root = output_dir / "gallery"
    config = {
        "protocol": PROTOCOL,
        "baseline_checkpoint": {
            **_checkpoint_identity(baseline_checkpoint),
            "sha256": _file_sha256(baseline_checkpoint),
        },
        "candidate_checkpoint": {
            **_checkpoint_identity(candidate_checkpoint),
            "sha256": _file_sha256(candidate_checkpoint),
        },
        "validation_manifest": str(baseline_manifest.resolve()),
        "validation_manifest_sha256": baseline_manifest_sha,
        "seed": args.seed,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "resolution": 512,
        "mixed_precision": args.mixed_precision,
        "device": args.device,
        "workers": args.workers,
        "xformers_memory_efficient_attention": args.enable_xformers_memory_efficient_attention,
        "max_samples": args.max_samples,
        "num_gallery_samples": args.num_gallery_samples,
        "positive_samples": len(selected_positive),
        "negative_samples": len(selected_negative_ids),
    }
    if (output_dir / "metrics.json").is_file():
        raise RuntimeError(f"Completed paired results already exist in {output_dir}")
    if state_path.is_file():
        previous = _read_json(state_path)
        if previous.get("config") != config:
            raise RuntimeError("Existing paired state uses a different configuration")
    elif records_root.exists() and any(records_root.rglob("*.json")):
        raise RuntimeError("Partial paired records exist without state metadata")

    progress_counts = _progress(records_root)
    _write_state(
        state_path,
        config,
        progress_counts,
        status="evaluating",
        phase="initializing",
        positive_total=len(selected_positive),
        negative_total=len(selected_negative_ids),
    )
    device = _resolve_device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.cuda.set_device(device.index)
    elif args.mixed_precision != "no":
        raise ValueError("CPU evaluation requires --mixed-precision no")
    metric_suite = PairedMetricSuite(device)
    checkpoint_steps: dict[str, int] = {}
    for role, checkpoint in (
        ("baseline", baseline_checkpoint),
        ("candidate", candidate_checkpoint),
    ):
        checkpoint_steps[role] = _evaluate_checkpoint(
            role=role,
            checkpoint=checkpoint,
            manifest=baseline_manifest,
            selected_positive=selected_positive,
            selected_negative_ids=selected_negative_ids,
            gallery_ids=gallery_ids,
            records_root=records_root,
            gallery_root=gallery_root,
            metric_suite=metric_suite,
            device=device,
            args=args,
            state_path=state_path,
            config=config,
            checkpoint_steps=checkpoint_steps,
            progress_counts=progress_counts,
        )

    baseline_positive = _load_role_records(records_root, "baseline", "positive")
    candidate_positive = _load_role_records(records_root, "candidate", "positive")
    baseline_negative = _load_role_records(records_root, "baseline", "negative")
    candidate_negative = _load_role_records(records_root, "candidate", "negative")
    positive_rows = paired_rows(
        baseline_positive,
        candidate_positive,
        metrics=POSITIVE_METRICS,
        mode="positive",
    )
    negative_rows = paired_rows(
        baseline_negative,
        candidate_negative,
        metrics=NEGATIVE_METRICS,
        mode="negative",
    )
    effects, conditions = summarize_effects(
        positive_rows,
        negative_rows,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    decision = build_decision(effects)
    _write_csv(output_dir / "positive_per_image.csv", positive_rows)
    _write_csv(output_dir / "negative_per_image.csv", negative_rows)
    _write_csv(output_dir / "paired_effects.csv", effects)
    _write_csv(output_dir / "condition_summary.csv", conditions)
    metrics_payload = {
        "metadata": {
            "protocol": PROTOCOL,
            "dataset": "CCDD-11",
            "split": "validation",
            "created_at_utc": _now(),
            "config": config,
            "checkpoint_steps": checkpoint_steps,
            "baseline_training_preparation": baseline_preparation,
            "candidate_training_preparation": candidate_preparation,
            "posterior_sampling": "sample() with a fixed SHA256-derived seed per sample",
            "bootstrap": {
                "unit": "scene_id cluster",
                "resamples": args.bootstrap_resamples,
                "seed": args.bootstrap_seed,
                "confidence_interval": "percentile 95%",
            },
            "advantage_definition": "positive always means candidate DAEM is better",
            "metric_semantics": {
                "mse_scale": "prediction and reference tensors in [-1, 1]",
                "psnr_ssim_scale": "prediction and reference tensors mapped to [0, 1]",
                "lpips_backbone": "VGG",
            },
            "software": {
                name: _package_version(name)
                for name in ("torch", "torchvision", "diffusers", "lpips", "piq")
            },
        },
        "counts": {
            "positive_per_model": len(positive_rows),
            "negative_per_model": len(negative_rows),
            "total_model_forwards": 2 * (len(positive_rows) + len(negative_rows)),
        },
        "decision": decision,
        "paired_effects": effects,
    }
    _atomic_json(output_dir / "metrics.json", metrics_payload)
    _write_state(
        state_path,
        config,
        progress_counts,
        status="completed",
        phase="completed",
        positive_total=len(positive_rows),
        negative_total=len(negative_rows),
        checkpoint_steps=checkpoint_steps,
    )
    print(
        f"Completed paired validation: {len(positive_rows)} positive and "
        f"{len(negative_rows)} negative samples per model. Results: {output_dir}",
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--baseline-checkpoint", type=Path, required=True)
    value.add_argument("--candidate-checkpoint", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--workers", type=int, default=8)
    value.add_argument("--device", default="cuda")
    value.add_argument(
        "--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16"
    )
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--bootstrap-resamples", type=int, default=10_000)
    value.add_argument("--bootstrap-seed", type=int, default=42)
    value.add_argument("--num-gallery-samples", type=int, default=20)
    value.add_argument("--max-samples", type=int)
    value.add_argument(
        "--enable-xformers-memory-efficient-attention", action="store_true"
    )
    return value


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
