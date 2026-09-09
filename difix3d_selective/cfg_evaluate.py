"""LUCID-style latent CFG sweep on the frozen CCDD-11 validation split."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from cdd11_full.metrics import rgb_psnr, rgb_ssim
from .cfg import blend_cfg_latents, blend_cfg_skips
from .data import load_records, negative_conditioning
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
from .protocol import directed_tasks, preserve_pair_prompt, selected_pairs
from .validation import stratified_indices


REFERENCES = ("positive_target", "negative_identity", "clean_gt")
REFERENCE_METRICS = (
    "mse_l2",
    "lpips_vgg",
    "objective_l2_lpips",
    "psnr",
    "ssim",
    "dists",
)
CFG_METRIC_FIELDS = tuple(
    f"{reference}_{metric}"
    for reference in REFERENCES
    for metric in REFERENCE_METRICS
)


def normalize_betas(values: list[float]) -> list[float]:
    """Validate, deduplicate and sort a CFG beta grid."""

    if not values:
        raise ValueError("At least one beta value is required")
    betas = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0 for value in betas):
        raise ValueError("beta values must be finite and non-negative")
    if len(set(betas)) != len(betas):
        raise ValueError("beta values must be unique")
    return sorted(betas)


def beta_label(value: float) -> str:
    return f"{float(value):g}"


class CfgMetricSuite:
    """Compute training losses and full-reference metrics for three targets."""

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
        self,
        prediction: torch.Tensor,
        references: Mapping[str, torch.Tensor],
    ) -> dict[str, float]:
        if set(references) != set(REFERENCES):
            raise ValueError(
                f"Expected references {list(REFERENCES)}, got {sorted(references)}"
            )
        if prediction.shape[0] != 1:
            raise ValueError("CFG metric evaluation currently requires batch size 1")
        ordered = [references[name] for name in REFERENCES]
        if any(reference.shape != prediction.shape for reference in ordered):
            raise ValueError("prediction/reference shape mismatch")

        prediction_float = prediction.float().clamp(-1, 1)
        stacked_prediction = prediction_float.repeat(len(REFERENCES), 1, 1, 1)
        stacked_references = torch.cat(
            [reference.float().clamp(-1, 1) for reference in ordered], dim=0
        )
        lpips_values = self.lpips(stacked_prediction, stacked_references).reshape(-1)
        prediction_01 = prediction_float.add(1).mul(0.5).clamp(0, 1)
        references_01 = stacked_references.add(1).mul(0.5).clamp(0, 1)
        stacked_prediction_01 = prediction_01.repeat(len(REFERENCES), 1, 1, 1)
        dists_values = self.dists(stacked_prediction_01, references_01).reshape(-1)
        if lpips_values.numel() != len(REFERENCES):
            raise RuntimeError("LPIPS did not return one value per reference")
        if dists_values.numel() != len(REFERENCES):
            raise RuntimeError("DISTS did not return one value per reference")

        result: dict[str, float] = {}
        for index, name in enumerate(REFERENCES):
            reference = ordered[index].float().clamp(-1, 1)
            reference_01 = reference.add(1).mul(0.5).clamp(0, 1)
            mse = float(F.mse_loss(prediction_float, reference).item())
            perceptual = float(lpips_values[index].item())
            result.update(
                {
                    f"{name}_mse_l2": mse,
                    f"{name}_lpips_vgg": perceptual,
                    f"{name}_objective_l2_lpips": mse + perceptual,
                    f"{name}_psnr": rgb_psnr(prediction_01, reference_01),
                    f"{name}_ssim": rgb_ssim(prediction_01, reference_01),
                    f"{name}_dists": float(dists_values[index].item()),
                }
            )
        return result


def _mean(rows: list[dict], field: str) -> float:
    return math.fsum(float(row[field]) for row in rows) / len(rows)


def _summarize_group(rows: list[dict]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("Cannot summarize an empty CFG group")
    return {
        "images": len(rows),
        **{field: _mean(rows, field) for field in CFG_METRIC_FIELDS},
        "mean_branch_inference_time_seconds": _mean(
            rows, "branch_inference_time_seconds"
        ),
        "mean_decode_time_seconds": _mean(rows, "decode_time_seconds"),
    }


def summarize_cfg_rows(rows: list[dict]) -> dict:
    """Summarize every beta by task, pair, micro mean and task macro mean."""

    if not rows:
        raise ValueError("Cannot summarize an empty CFG sweep")
    by_beta: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        by_beta[float(row["beta"])].append(row)

    beta_summaries: dict[str, dict] = {}
    for beta in sorted(by_beta):
        beta_rows = by_beta[beta]
        by_task: dict[str, list[dict]] = defaultdict(list)
        by_pair: dict[str, list[dict]] = defaultdict(list)
        for row in beta_rows:
            task = (
                f"{row['pair']}:remove-{row['remove']}:"
                f"preserve-{row['preserve']}"
            )
            by_task[task].append(row)
            by_pair[str(row["pair"])].append(row)
        tasks = {key: _summarize_group(by_task[key]) for key in sorted(by_task)}
        pairs = {key: _summarize_group(by_pair[key]) for key in sorted(by_pair)}
        micro = _summarize_group(beta_rows)
        macro_fields = (
            *CFG_METRIC_FIELDS,
            "mean_branch_inference_time_seconds",
            "mean_decode_time_seconds",
        )
        macro: dict[str, float | int] = {
            "images": len(beta_rows),
            "tasks": len(tasks),
        }
        for field in macro_fields:
            macro[field] = math.fsum(float(task[field]) for task in tasks.values()) / len(
                tasks
            )
        beta_summaries[beta_label(beta)] = {
            "beta": beta,
            "overall": {"micro": micro, "macro": macro},
            "pairs": pairs,
            "tasks": tasks,
        }

    macro_by_beta = {
        float(summary["beta"]): summary["overall"]["macro"]
        for summary in beta_summaries.values()
    }
    best_psnr = max(
        macro_by_beta,
        key=lambda beta: macro_by_beta[beta]["positive_target_psnr"],
    )
    best_objective = min(
        macro_by_beta,
        key=lambda beta: macro_by_beta[beta]["positive_target_objective_l2_lpips"],
    )
    pareto: list[float] = []
    for beta, candidate in sorted(macro_by_beta.items()):
        dominated = any(
            other_beta != beta
            and other["positive_target_objective_l2_lpips"]
            <= candidate["positive_target_objective_l2_lpips"]
            and other["negative_identity_objective_l2_lpips"]
            <= candidate["negative_identity_objective_l2_lpips"]
            and (
                other["positive_target_objective_l2_lpips"]
                < candidate["positive_target_objective_l2_lpips"]
                or other["negative_identity_objective_l2_lpips"]
                < candidate["negative_identity_objective_l2_lpips"]
            )
            for other_beta, other in macro_by_beta.items()
        )
        if not dominated:
            pareto.append(beta)
    return {
        "betas": beta_summaries,
        "selection": {
            "best_positive_target_psnr_beta": best_psnr,
            "best_positive_target_objective_beta": best_objective,
            "positive_identity_objective_pareto_betas": pareto,
        },
    }


def summary_csv_rows(summary: dict) -> list[dict]:
    rows: list[dict] = []
    for beta_key, beta_summary in summary["betas"].items():
        beta = beta_summary["beta"]
        for condition, values in beta_summary["tasks"].items():
            rows.append({
                "beta": beta,
                "group": "directed_task",
                "condition": condition,
                **values,
            })
        for condition, values in beta_summary["pairs"].items():
            rows.append({
                "beta": beta,
                "group": "pair",
                "condition": condition,
                **values,
            })
        for condition, values in beta_summary["overall"].items():
            rows.append({
                "beta": beta,
                "group": "overall",
                "condition": condition,
                **values,
            })
    return rows


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


def _tensor_to_pil(value: torch.Tensor) -> Image.Image:
    tensor = value.detach().float().clamp(0, 1)
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = tensor.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.asarray(array), "RGB")


def save_contact_sheet(
    references: Mapping[str, torch.Tensor],
    predictions: Mapping[float, torch.Tensor],
    path: Path,
) -> None:
    """Save a labeled 4-column reference and beta comparison sheet."""

    panels = [
        ("double degraded", references["negative_identity"]),
        ("DACG coarse", references["coarse"]),
        ("selective target", references["positive_target"]),
        ("clean GT", references["clean_gt"]),
    ]
    panels.extend(
        (f"beta={beta_label(beta)}", value) for beta, value in predictions.items()
    )
    pil_panels = [(label, _tensor_to_pil(value)) for label, value in panels]
    width, height = pil_panels[0][1].size
    columns = 4
    rows = math.ceil(len(pil_panels) / columns)
    header = 24
    sheet = Image.new("RGB", (columns * width, rows * (height + header)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (label, panel) in enumerate(pil_panels):
        left = (index % columns) * width
        top = (index // columns) * (height + header)
        draw.text((left + 5, top + 5), label, fill="black", font=font)
        sheet.paste(panel, (left, top + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _record_path(records_dir: Path, record: dict) -> Path:
    task = f"remove-{record['remove']}-preserve-{record['preserve']}"
    return records_dir / str(record["pair"]) / task / f"{record['scene_id']}.json"


def _load_partial_records(records_dir: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    if not records_dir.is_dir():
        return result
    for path in sorted(records_dir.rglob("*.json")):
        record = _read_json(path)
        sample_id = str(record["sample_id"])
        if sample_id in result:
            raise ValueError(f"Duplicate CFG record for {sample_id}")
        result[sample_id] = record
    return result


def _validate_training_run(checkpoint: Path) -> tuple[Path, Path, dict]:
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
        raise ValueError("CFG validation requires a CCDD-11 training run")
    if preparation.get("target_kind") != "native_selective_sub_data":
        raise ValueError("CFG validation requires native CCDD-11 selective targets")
    pair_ids = [int(value["id"]) for value in preparation["degradation_pairs"]]
    if [pair_id for pair_id, _ in selected_pairs(pair_ids)] != [1, 2, 3, 4, 5]:
        raise ValueError("CFG validation requires all five CCDD-11 degradation pairs")
    return run_dir, manifest, preparation


def validate_full_manifest(records: list[dict]) -> None:
    """Require the frozen 118-scene, ten-direction CCDD validation split."""

    if len(records) != 1180:
        raise ValueError(f"Expected 1180 frozen validation records, got {len(records)}")
    if {record.get("split") for record in records} != {"validation"}:
        raise ValueError("CFG sweep may only read the validation split")
    identifiers = [str(record.get("id", "")) for record in records]
    if any(not value for value in identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise ValueError("Validation manifest IDs must be non-empty and unique")
    expected_tasks = {
        (pair_id, task.remove, task.preserve)
        for pair_id in range(1, 6)
        for task in directed_tasks(pair_id)
    }
    counts = Counter(
        (int(record["pair_id"]), str(record["remove"]), str(record["preserve"]))
        for record in records
    )
    if set(counts) != expected_tasks or set(counts.values()) != {118}:
        raise ValueError(
            "Validation manifest must contain exactly 118 records for each of the "
            "ten directed tasks"
        )


def _flatten_records(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        common = {
            key: record[key]
            for key in (
                "sample_id",
                "scene_id",
                "pair_id",
                "pair",
                "remove",
                "preserve",
                "positive_prompt",
                "negative_prompt",
                "coarse_path",
                "degraded_path",
                "positive_target_path",
                "clean_gt_path",
                "inference_seed",
                "branch_inference_time_seconds",
                "contact_sheet_path",
            )
        }
        for beta_result in record["results"]:
            rows.append({**common, **beta_result})
    return rows


def wandb_task_curves(summary: dict) -> dict[str, dict]:
    """Build beta-indexed multi-line data for all ten directed tasks."""

    beta_summaries = sorted(
        summary["betas"].values(), key=lambda value: float(value["beta"])
    )
    if not beta_summaries:
        raise ValueError("Cannot build W&B task curves without beta summaries")
    task_names = sorted(beta_summaries[0]["tasks"])
    expected_tasks = set(task_names)
    for beta_summary in beta_summaries:
        if set(beta_summary["tasks"]) != expected_tasks:
            raise ValueError("Directed task set differs between beta summaries")
    betas = [float(value["beta"]) for value in beta_summaries]
    return {
        field: {
            "betas": betas,
            "tasks": task_names,
            "values": [
                [
                    float(beta_summary["tasks"][task][field])
                    for beta_summary in beta_summaries
                ]
                for task in task_names
            ],
        }
        for field in CFG_METRIC_FIELDS
    }


def _log_wandb(
    args: argparse.Namespace, config: dict, summary: dict, records: list[dict]
) -> None:
    if args.report_to == "none":
        return
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("wandb is required when --report-to wandb") from error
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        dir=str(args.output_dir),
        config={**config, "wandb_schema": "beta-axis-task-curves-v1"},
        job_type="cfg-validation",
    )
    run.define_metric("cfg/beta")
    summary_rows = summary_csv_rows(summary)
    scalar_payloads = []
    for beta_summary in summary["betas"].values():
        beta = float(beta_summary["beta"])
        macro = beta_summary["overall"]["macro"]
        micro = beta_summary["overall"]["micro"]
        payload = {"cfg/beta": beta}
        payload.update(
            {f"cfg/task_macro/{key}": value for key, value in macro.items()}
        )
        payload.update({f"cfg/micro/{key}": value for key, value in micro.items()})
        scalar_payloads.append(payload)
    scalar_metric_names = sorted(
        {
            name
            for payload in scalar_payloads
            for name in payload
            if name != "cfg/beta"
        }
    )
    for name in scalar_metric_names:
        run.define_metric(name, step_metric="cfg/beta")
    for payload in scalar_payloads:
        run.log(payload)

    task_plots = {}
    for field, curve in wandb_task_curves(summary).items():
        task_plots[f"cfg/task_curves/{field}"] = wandb.plot.line_series(
            xs=curve["betas"],
            ys=curve["values"],
            keys=curve["tasks"],
            title=f"{field.replace('_', ' ')} by directed task",
            xname="beta",
        )
    run.log(task_plots)
    table_fields = list(dict.fromkeys(key for row in summary_rows for key in row))
    run.log(
        {
            "cfg/summary_table": wandb.Table(
                columns=table_fields,
                data=[[row.get(field) for field in table_fields] for row in summary_rows],
            )
        }
    )
    gallery_data = []
    for record in records:
        if record.get("contact_sheet_path"):
            gallery_data.append(
                [
                    record["sample_id"],
                    record["pair"],
                    record["remove"],
                    record["preserve"],
                    wandb.Image(record["contact_sheet_path"]),
                ]
            )
    if gallery_data:
        run.log(
            {
                "cfg/gallery": wandb.Table(
                    columns=[
                        "sample_id",
                        "pair",
                        "remove",
                        "preserve",
                        "contact_sheet",
                    ],
                    data=gallery_data,
                )
            }
        )
    for key, value in summary["selection"].items():
        run.summary[f"cfg/{key}"] = value
    run.finish()


def upload_wandb_results(args: argparse.Namespace) -> None:
    """Create a new W&B run from completed local CFG results, without inference."""

    results_dir = args.results_dir.resolve()
    metrics_path = results_dir / "metrics.json"
    records_dir = results_dir / "records"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics = _read_json(metrics_path)
    metadata = metrics.get("metadata")
    summary = metrics.get("summary")
    if not isinstance(metadata, dict) or not isinstance(summary, dict):
        raise ValueError("metrics.json lacks metadata or summary")
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError("metrics.json lacks metadata.config")
    records_by_id = _load_partial_records(records_dir)
    expected = int(metrics.get("validation_sample_count", -1))
    if len(records_by_id) != expected:
        raise ValueError(
            f"Expected {expected} sample records, found {len(records_by_id)}"
        )
    args.output_dir = results_dir
    args.report_to = "wandb"
    records = [records_by_id[key] for key in sorted(records_by_id)]
    _log_wandb(args, config, summary, records)
    print(
        f"Uploaded {expected} completed CFG samples to W&B from {results_dir}",
        flush=True,
    )


def run(args: argparse.Namespace) -> None:
    from torch.utils.data import DataLoader, Subset
    from tqdm.auto import tqdm

    from .data import SelectiveDifixDataset
    from .model import SelectiveDifix, load_model_checkpoint

    args.betas = normalize_betas(list(args.betas))
    fixed_protocol = {
        "resolution": 512,
        "lora_rank_vae": 4,
        "timestep": 199,
        "seed": 42,
    }
    mismatches = {
        key: (getattr(args, key), expected)
        for key, expected in fixed_protocol.items()
        if getattr(args, key) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {expected!r})"
            for key, (actual, expected) in mismatches.items()
        )
        raise ValueError(f"CCDD-11 CFG validation protocol mismatch: {details}")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.num_gallery_samples < 0:
        raise ValueError("--num-gallery-samples must be non-negative")
    checkpoint = args.checkpoint.resolve()
    run_dir, manifest, preparation = _validate_training_run(checkpoint)
    all_manifest_records = load_records(manifest)
    validate_full_manifest(all_manifest_records)

    if args.max_samples is None:
        selected_indices = list(range(len(all_manifest_records)))
    else:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        selected_indices = stratified_indices(all_manifest_records, args.max_samples)
    selected_records = [all_manifest_records[index] for index in selected_indices]
    expected_ids = {str(record["id"]) for record in selected_records}
    output_dir = (
        args.output_dir
        or run_dir / f"cfg_validation_{checkpoint.stem}_state_beta_sweep_v2"
    ).resolve()
    args.output_dir = output_dir
    if (output_dir / "metrics.json").is_file():
        raise RuntimeError(f"Completed CFG results already exist in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    experiment_metadata = checkpoint_payload.get("experiment_metadata", {})
    if experiment_metadata.get("dataset") != "CCDD-11":
        raise ValueError("Checkpoint metadata does not identify CCDD-11 training")
    if float(experiment_metadata.get("negative_train_probability", 0.0)) <= 0:
        raise ValueError("Checkpoint was not trained with negative CCDD-11 samples")
    del checkpoint_payload

    config = {
        "protocol": "ccdd11-endpoint-correct-state-cfg-validation-v2",
        "checkpoint": {
            **_checkpoint_identity(checkpoint),
            "sha256": _file_sha256(checkpoint),
        },
        "validation_manifest": str(manifest.resolve()),
        "validation_manifest_sha256": _file_sha256(manifest),
        "betas": args.betas,
        "resolution": args.resolution,
        "lora_rank_vae": args.lora_rank_vae,
        "timestep": args.timestep,
        "seed": args.seed,
        "device": args.device,
        "mixed_precision": args.mixed_precision,
        "xformers_memory_efficient_attention": (
            args.enable_xformers_memory_efficient_attention
        ),
        "num_gallery_samples": args.num_gallery_samples,
        "max_samples": args.max_samples,
        "validation_samples": len(selected_records),
        "negative_train_probability": experiment_metadata["negative_train_probability"],
        "latent_formula": "z_negative + beta * (z_positive - z_negative)",
        "decoder_skip_formula": (
            "skip_negative + beta * (skip_positive - skip_negative)"
        ),
    }
    state_path = output_dir / "state.json"
    records_dir = output_dir / "records"
    partial_records = _load_partial_records(records_dir)
    if state_path.is_file():
        previous = _read_json(state_path)
        if previous.get("config") != config:
            raise RuntimeError("Existing partial CFG state uses a different configuration")
    elif partial_records:
        raise RuntimeError("Partial CFG records exist without state metadata")
    unknown = set(partial_records) - expected_ids
    if unknown:
        raise ValueError(
            f"Partial CFG records contain unknown samples: {sorted(unknown)[:3]}"
        )
    _atomic_json(
        state_path,
        {
            "status": "evaluating",
            "processed_samples": len(partial_records),
            "total_samples": len(selected_records),
            "config": config,
            "updated_at_utc": _now(),
        },
    )

    device = _resolve_device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.cuda.set_device(device.index)
    elif args.mixed_precision != "no":
        raise ValueError("CPU evaluation requires --mixed-precision no")
    torch.manual_seed(args.seed)
    model = SelectiveDifix(
        lora_rank_vae=args.lora_rank_vae, timestep=args.timestep
    ).to(device)
    if args.enable_xformers_memory_efficient_attention:
        model.unet.enable_xformers_memory_efficient_attention()
    global_step = load_model_checkpoint(
        model, checkpoint, expected_dataset="CCDD-11", expected_seed=42
    )
    model.set_eval()
    metric_suite = CfgMetricSuite(device)
    dataset = SelectiveDifixDataset(
        manifest, model.tokenizer, resolution=args.resolution, training_mode="positive"
    )
    gallery_ids = set()
    if args.num_gallery_samples > 0:
        gallery_ids = {
            str(dataset.records[index]["id"])
            for index in stratified_indices(dataset.records, args.num_gallery_samples)
        } & expected_ids
    pending = [
        index
        for index in selected_indices
        if str(dataset.records[index]["id"]) not in partial_records
    ]
    loader = DataLoader(
        Subset(dataset, pending),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    manifest_by_id = {str(record["id"]): record for record in all_manifest_records}
    progress = tqdm(
        total=len(selected_records),
        initial=len(partial_records),
        desc="CCDD-11 validation CFG sweep",
    )
    for batch in loader:
        sample_id = str(batch["sample_id"][0])
        positive_source = batch["conditioning_pixel_values"].to(
            device, non_blocking=True
        )
        positive_target = batch["output_pixel_values"].to(device, non_blocking=True)
        clean_gt = batch["ground_truth_pixel_values"].to(device, non_blocking=True)
        coarse = positive_source[:, 0]
        degraded = positive_source[:, 1]
        negative_source = negative_conditioning(coarse, degraded)
        pair_id = int(batch["pair_id"].item())
        negative_prompt = preserve_pair_prompt(pair_id)
        negative_tokens = model.tokenizer(
            [negative_prompt],
            max_length=model.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        devices = [device.index] if device.type == "cuda" else []
        inference_seed = _sample_seed(args.seed, sample_id)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        branch_started = time.perf_counter()
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(inference_seed)
            with torch.inference_mode(), _autocast(device, args.mixed_precision):
                (
                    z_positive,
                    z_negative,
                    positive_skips,
                    negative_skips,
                ) = model.cfg_latents(
                    positive_source,
                    negative_source,
                    positive_prompt_tokens=batch["input_ids"].to(device),
                    negative_prompt_tokens=negative_tokens,
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        branch_elapsed = time.perf_counter() - branch_started
        references = {
            "positive_target": positive_target,
            "negative_identity": degraded,
            "clean_gt": clean_gt,
        }
        predictions_for_gallery: dict[float, torch.Tensor] = {}
        results = []
        for beta in args.betas:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            decode_started = time.perf_counter()
            with torch.inference_mode(), _autocast(device, args.mixed_precision):
                prediction = model.decode_main_latent(
                    blend_cfg_latents(z_positive, z_negative, beta),
                    blend_cfg_skips(positive_skips, negative_skips, beta),
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            decode_elapsed = time.perf_counter() - decode_started
            metrics = metric_suite(prediction, references)
            prediction_path = ""
            if sample_id in gallery_ids:
                prediction_01 = prediction.float().add(1).mul(0.5).clamp(0, 1)
                predictions_for_gallery[beta] = prediction_01.cpu()
                destination = (
                    output_dir
                    / "gallery"
                    / _safe_id(sample_id)
                    / f"beta_{beta_label(beta)}.png"
                )
                _save_image(prediction_01, destination)
                prediction_path = str(destination)
            results.append(
                {
                    "beta": beta,
                    **metrics,
                    "branch_inference_time_seconds": branch_elapsed,
                    "decode_time_seconds": decode_elapsed,
                    "prediction_path": prediction_path,
                }
            )
        contact_sheet_path = ""
        if sample_id in gallery_ids:
            contact_sheet = (
                output_dir / "gallery" / _safe_id(sample_id) / "contact_sheet.png"
            )
            save_contact_sheet(
                {
                    **references,
                    "coarse": coarse.float().add(1).mul(0.5).clamp(0, 1),
                    "positive_target": (
                        positive_target.float().add(1).mul(0.5).clamp(0, 1)
                    ),
                    "negative_identity": (
                        degraded.float().add(1).mul(0.5).clamp(0, 1)
                    ),
                    "clean_gt": clean_gt.float().add(1).mul(0.5).clamp(0, 1),
                },
                predictions_for_gallery,
                contact_sheet,
            )
            contact_sheet_path = str(contact_sheet)
        source_record = manifest_by_id[sample_id]
        record = {
            "sample_id": sample_id,
            "scene_id": str(batch["scene_id"][0]),
            "pair_id": pair_id,
            "pair": str(batch["pair"][0]),
            "remove": str(batch["remove"][0]),
            "preserve": str(batch["preserve"][0]),
            "positive_prompt": str(batch["prompt"][0]),
            "negative_prompt": negative_prompt,
            "coarse_path": source_record["image"],
            "degraded_path": source_record["ref_image"],
            "positive_target_path": source_record["target_image"],
            "clean_gt_path": source_record["clear_image"],
            "inference_seed": inference_seed,
            "branch_inference_time_seconds": branch_elapsed,
            "contact_sheet_path": contact_sheet_path,
            "results": results,
        }
        destination = _record_path(records_dir, record)
        _atomic_json(destination, record)
        partial_records[sample_id] = record
        progress.update(1)
        if len(partial_records) % 10 == 0 or len(partial_records) == len(
            selected_records
        ):
            _atomic_json(
                state_path,
                {
                    "status": "evaluating",
                    "processed_samples": len(partial_records),
                    "total_samples": len(selected_records),
                    "config": config,
                    "updated_at_utc": _now(),
                },
            )
        del z_positive, z_negative, positive_skips, negative_skips
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    progress.close()

    if set(partial_records) != expected_ids:
        raise RuntimeError(
            f"CFG validation incomplete: missing {len(expected_ids - set(partial_records))} samples"
        )
    ordered_records = [
        partial_records[str(record["id"])] for record in selected_records
    ]
    rows = _flatten_records(ordered_records)
    summary = summarize_cfg_rows(rows)
    _write_csv(output_dir / "per_image_metrics.csv", rows)
    _write_csv(output_dir / "summary.csv", summary_csv_rows(summary))
    metadata = {
        "protocol": config["protocol"],
        "dataset": "CCDD-11",
        "split": "validation",
        "global_step": global_step,
        "created_at_utc": _now(),
        "config": config,
        "training_preparation": preparation,
        "metrics": {
            "mse_l2": "MSE on [-1,1], matching the training L2 term",
            "lpips_vgg": "lpips.LPIPS(net='vgg') on [-1,1]",
            "objective_l2_lpips": "mse_l2 + lpips_vgg",
            "psnr": "CDD11 RGB PSNR on [0,1]",
            "ssim": "CDD11 RGB SSIM on [0,1]",
            "dists": "piq.DISTS(reduction='none') on [0,1]",
        },
        "references": {
            "positive_target": "native selective sub_data target",
            "negative_identity": "original double-degradation input",
            "clean_gt": "fully clean scene",
        },
        "software": {
            name: _package_version(name)
            for name in ("torch", "torchvision", "diffusers", "lpips", "piq", "wandb")
        },
    }
    metrics_payload = {
        "metadata": metadata,
        "summary": summary,
        "per_image_count": len(rows),
        "validation_sample_count": len(ordered_records),
    }
    _atomic_json(
        state_path,
        {
            "status": "local_completed_reporting_pending",
            "processed_samples": len(ordered_records),
            "total_samples": len(ordered_records),
            "config": config,
            "global_step": global_step,
            "selection": summary["selection"],
            "updated_at_utc": _now(),
        },
    )
    _log_wandb(args, config, summary, ordered_records)
    _atomic_json(output_dir / "metrics.json", metrics_payload)
    _atomic_json(
        state_path,
        {
            "status": "completed",
            "processed_samples": len(ordered_records),
            "total_samples": len(ordered_records),
            "config": config,
            "global_step": global_step,
            "selection": summary["selection"],
            "updated_at_utc": _now(),
        },
    )
    print(
        f"Completed {len(ordered_records)} validation samples x {len(args.betas)} betas "
        f"at step {global_step}. Results: {output_dir}",
        flush=True,
    )


def _sample_seed(seed: int, sample_id: str) -> int:
    import hashlib

    digest = hashlib.sha256(f"ccdd11-cfg:{seed}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--output-dir", type=Path)
    value.add_argument("--betas", nargs="+", type=float, required=True)
    value.add_argument("--num-gallery-samples", type=int, default=20)
    value.add_argument("--max-samples", type=int)
    value.add_argument("--resolution", type=int, default=512)
    value.add_argument("--lora-rank-vae", type=int, default=4)
    value.add_argument("--timestep", type=int, default=199)
    value.add_argument("--workers", type=int, default=8)
    value.add_argument("--device", default="cuda")
    value.add_argument(
        "--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16"
    )
    value.add_argument("--seed", type=int, default=42)
    value.add_argument(
        "--enable-xformers-memory-efficient-attention", action="store_true"
    )
    value.add_argument("--report-to", choices=("wandb", "none"), default="wandb")
    value.add_argument("--wandb-entity", default="c14150591-sjtu")
    value.add_argument("--wandb-project", default="difix-ccdd11-selective")
    value.add_argument(
        "--wandb-run-name",
        default="ccdd-all5-100k-best-state-cfg-validation-beta-sweep-v2",
    )
    return value


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
