"""Evaluate semantic effects of SelectiveDifix Text-FiLM detail gates."""

from __future__ import annotations

import argparse
import csv
import gc
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import torch

from .data import SelectiveDifixDataset
from .evaluate import (
    _atomic_json,
    _autocast,
    _file_sha256,
    _read_json,
    _resolve_device,
    _safe_id,
    _save_image,
)
from .paired_validation import (
    NEGATIVE_METRICS,
    POSITIVE_METRICS,
    PairedMetricSuite,
    _load_role_records,
    _negative_ids,
    _selected_records,
    _training_run,
    paired_rows,
    sample_seed,
    summarize_effects,
)
from .protocol import PROMPT_NAMES
from .validation import stratified_indices


def swapped_prompts(batch: Mapping) -> list[str]:
    return [
        f"remove {PROMPT_NAMES[preserve]}, preserve {PROMPT_NAMES[remove]}"
        for remove, preserve in zip(batch["remove"], batch["preserve"])
    ]


def _seeded(device: torch.device, seed: int):
    devices = [device.index] if device.type == "cuda" else []
    return torch.random.fork_rng(devices=devices)


@torch.inference_mode()
def text_variant_predictions(
    model,
    batch: Mapping,
    device: torch.device,
    mixed_precision: str,
    seed: int,
    *,
    capture_gates: bool = False,
) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Decode correct/swapped/no-text variants from one correct-prompt latent."""

    source = batch["conditioning_pixel_values"].to(device, non_blocking=True)
    tokens = batch["input_ids"].to(device, non_blocking=True)
    mask = batch["attention_mask"].to(device, non_blocking=True)
    with _seeded(device, seed):
        torch.manual_seed(seed)
        with _autocast(device, mixed_precision):
            latent, skips = model.denoised_main_latent(
                source,
                prompt_tokens=tokens,
                prompt_attention_mask=mask,
            )
            correct_condition = model._last_detail_condition
            swapped_condition = model.detail_condition(
                source, prompt=swapped_prompts(batch)
            )
            predictions = {
                "correct": model.decode_main_latent(
                    latent, skips, prompt_condition=correct_condition
                ),
                "swapped": model.decode_main_latent(
                    latent, skips, prompt_condition=swapped_condition
                ),
                "no_text": model.decode_main_latent(
                    latent,
                    skips,
                    prompt_condition=correct_condition,
                    text_condition_scale=0.0,
                ),
            }
            analysis = (
                model.analyze_detail_gates(
                    latent, skips, correct_condition, swapped_condition
                )
                if capture_gates
                else []
            )
    return predictions, analysis


@torch.inference_mode()
def constant_prediction(
    model,
    batch: Mapping,
    device: torch.device,
    mixed_precision: str,
    seed: int,
) -> torch.Tensor:
    source = batch["conditioning_pixel_values"].to(device, non_blocking=True)
    with _seeded(device, seed):
        torch.manual_seed(seed)
        with _autocast(device, mixed_precision):
            return model(
                source,
                prompt_tokens=batch["input_ids"].to(device, non_blocking=True),
                prompt_attention_mask=batch["attention_mask"].to(
                    device, non_blocking=True
                ),
            )


def _metric_record(
    baseline: Mapping,
    prediction: torch.Tensor,
    metric_suite: PairedMetricSuite,
    references: Mapping[str, torch.Tensor],
) -> dict:
    return {**baseline, **metric_suite(prediction, references), "prediction_path": ""}


def _map_image(value: torch.Tensor) -> torch.Tensor:
    value = value[0].float()
    low = value.amin()
    high = value.amax()
    normalized = (value - low) / (high - low).clamp_min(1e-8)
    return normalized.unsqueeze(0).repeat(3, 1, 1)


def _save_gate_maps(analysis: list[dict[str, torch.Tensor]], destination: Path) -> None:
    for index, layer in enumerate(analysis):
        for name in (
            "base_gate",
            "correct_gate",
            "swapped_gate",
            "gate_difference",
            "delta_logits",
        ):
            _save_image(
                _map_image(layer[name]),
                destination / f"l{index}_{name}.png",
            )


def _record_path(root: Path, variant: str, mode: str, sample_id: str) -> Path:
    return root / variant / mode / f"{_safe_id(sample_id)}.json"


def _load_records(root: Path, variant: str, mode: str) -> dict[str, dict]:
    directory = root / variant / mode
    if not directory.is_dir():
        return {}
    records = {}
    for path in directory.glob("*.json"):
        record = _read_json(path)
        records[str(record["sample_id"])] = record
    return records


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _semantic_summary(records: Mapping[str, dict]) -> dict:
    fields = [
        "prompt_swap_output_delta_l1",
        "correct_vs_swapped_target_psnr_gap",
        *[f"gate_difference_l{index}" for index in range(4)],
    ]
    by_task = defaultdict(list)
    for record in records.values():
        by_task[f"{record['pair']}/remove_{record['remove']}"].append(record)

    def means(rows):
        return {
            field: math.fsum(float(row[field]) for row in rows if field in row)
            / sum(field in row for row in rows)
            for field in fields
            if any(field in row for row in rows)
        }

    return {
        "overall": means(list(records.values())),
        "tasks": {task: means(rows) for task, rows in sorted(by_task.items())},
    }


def _evaluate_full_checkpoint(
    model,
    positive_loader,
    negative_loader,
    baseline_positive: Mapping[str, dict],
    baseline_negative: Mapping[str, dict],
    metric_suite: PairedMetricSuite,
    records_root: Path,
    gallery_ids: set[str],
    gallery_root: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, tuple[dict[str, dict], dict[str, dict]]]:
    variants = {
        name: (
            _load_records(records_root, name, "positive"),
            _load_records(records_root, name, "negative"),
        )
        for name in ("correct", "swapped", "no_text")
    }
    for batch in positive_loader:
        sample_id = str(batch["sample_id"][0])
        records_complete = all(sample_id in variants[name][0] for name in variants)
        gate_maps_complete = (
            sample_id not in gallery_ids
            or (gallery_root / _safe_id(sample_id) / "l3_gate_difference.png").is_file()
        )
        if records_complete and gate_maps_complete:
            continue
        predictions, analysis = text_variant_predictions(
            model,
            batch,
            device,
            args.mixed_precision,
            sample_seed(args.seed, sample_id),
            capture_gates=sample_id in gallery_ids,
        )
        target = batch["output_pixel_values"].to(device)
        clean = batch["ground_truth_pixel_values"].to(device)
        correct_psnr = None
        swapped_psnr = None
        for variant, prediction in predictions.items():
            record = _metric_record(
                baseline_positive[sample_id],
                prediction,
                metric_suite,
                {"target": target, "clean_gt": clean},
            )
            record["variant"] = variant
            if variant == "correct":
                correct_psnr = record["target_psnr"]
            elif variant == "swapped":
                swapped_psnr = record["target_psnr"]
            if variant != "no_text":
                record["output_delta_l1_vs_no_text"] = float(
                    (prediction.float() - predictions["no_text"].float()).abs().mean()
                )
            for index, layer in enumerate(analysis):
                record[f"gate_difference_l{index}"] = float(
                    layer["gate_difference_mean"]
                )
            _atomic_json(
                _record_path(records_root, variant, "positive", sample_id), record
            )
            variants[variant][0][sample_id] = record
        variants["correct"][0][sample_id]["prompt_swap_output_delta_l1"] = float(
            (predictions["correct"].float() - predictions["swapped"].float())
            .abs()
            .mean()
        )
        variants["correct"][0][sample_id]["correct_vs_swapped_target_psnr_gap"] = float(
            correct_psnr - swapped_psnr
        )
        _atomic_json(
            _record_path(records_root, "correct", "positive", sample_id),
            variants["correct"][0][sample_id],
        )
        if analysis:
            _save_gate_maps(analysis, gallery_root / _safe_id(sample_id))

    for batch in negative_loader:
        sample_id = str(batch["sample_id"][0])
        if all(sample_id in variants[name][1] for name in ("correct", "no_text")):
            continue
        source = batch["conditioning_pixel_values"].to(device)
        with _seeded(device, sample_seed(args.seed, sample_id)):
            torch.manual_seed(sample_seed(args.seed, sample_id))
            with _autocast(device, args.mixed_precision):
                latent, skips = model.denoised_main_latent(
                    source,
                    prompt_tokens=batch["input_ids"].to(device),
                    prompt_attention_mask=batch["attention_mask"].to(device),
                )
                condition = model._last_detail_condition
                correct = model.decode_main_latent(
                    latent, skips, prompt_condition=condition
                )
                no_text = model.decode_main_latent(
                    latent,
                    skips,
                    prompt_condition=condition,
                    text_condition_scale=0.0,
                )
        identity = batch["output_pixel_values"].to(device)
        for variant, prediction in (("correct", correct), ("no_text", no_text)):
            record = _metric_record(
                baseline_negative[sample_id],
                prediction,
                metric_suite,
                {"identity": identity},
            )
            prediction_01 = prediction.float().add(1).mul(0.5).clamp(0, 1)
            identity_01 = identity.float().add(1).mul(0.5).clamp(0, 1)
            record["identity_mae"] = float((prediction_01 - identity_01).abs().mean())
            record["variant"] = variant
            _atomic_json(
                _record_path(records_root, variant, "negative", sample_id), record
            )
            variants[variant][1][sample_id] = record
    return variants


def _evaluate_constant_checkpoint(
    model,
    positive_loader,
    negative_loader,
    baseline_positive,
    baseline_negative,
    metric_suite,
    records_root,
    device,
    args,
) -> tuple[dict[str, dict], dict[str, dict]]:
    positive = _load_records(records_root, "constant", "positive")
    negative = _load_records(records_root, "constant", "negative")
    for mode, loader, baseline in (
        ("positive", positive_loader, baseline_positive),
        ("negative", negative_loader, baseline_negative),
    ):
        destination = positive if mode == "positive" else negative
        for batch in loader:
            sample_id = str(batch["sample_id"][0])
            if sample_id in destination:
                continue
            prediction = constant_prediction(
                model,
                batch,
                device,
                args.mixed_precision,
                sample_seed(args.seed, sample_id),
            )
            if mode == "positive":
                references = {
                    "target": batch["output_pixel_values"].to(device),
                    "clean_gt": batch["ground_truth_pixel_values"].to(device),
                }
            else:
                identity = batch["output_pixel_values"].to(device)
                references = {"identity": identity}
            record = _metric_record(
                baseline[sample_id], prediction, metric_suite, references
            )
            if mode == "negative":
                record["identity_mae"] = float(
                    (
                        prediction.float().add(1).mul(0.5).clamp(0, 1)
                        - identity.float().add(1).mul(0.5).clamp(0, 1)
                    )
                    .abs()
                    .mean()
                )
            record["variant"] = "constant"
            _atomic_json(
                _record_path(records_root, "constant", mode, sample_id), record
            )
            destination[sample_id] = record
    return positive, negative


def run(args: argparse.Namespace) -> None:
    from torch.utils.data import DataLoader, Subset

    from .model import load_model_from_checkpoint

    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_root = output_dir / "records"
    gallery_root = output_dir / "gate_maps"

    _, manifest, _ = _training_run(args.candidate_checkpoint.resolve())
    model, _, metadata = load_model_from_checkpoint(
        args.candidate_checkpoint.resolve(),
        expected_dataset="CCDD-11",
        expected_seed=42,
    )
    if model.detail_text_mode != "full":
        raise ValueError("--candidate-checkpoint must use detail_text_mode='full'")
    candidate_text_config = model.text_film_config()
    candidate_detail_config = model.detail_config()
    root_checkpoint_sha = metadata.get("root_checkpoint_sha256")
    paired_metrics = _read_json(args.b20_paired_results.resolve() / "metrics.json")
    paired_b20_sha = (
        paired_metrics.get("metadata", {})
        .get("config", {})
        .get("candidate_checkpoint", {})
        .get("sha256")
    )
    if root_checkpoint_sha != paired_b20_sha:
        raise ValueError(
            "The semantic candidate and reused paired results do not share B-20k"
        )
    paired_config = paired_metrics.get("metadata", {}).get("config", {})
    reuse_identity = {
        "seed": args.seed,
        "mixed_precision": args.mixed_precision,
        "xformers_memory_efficient_attention": (
            args.enable_xformers_memory_efficient_attention
        ),
        "validation_manifest_sha256": _file_sha256(manifest),
    }
    if any(paired_config.get(key) != value for key, value in reuse_identity.items()):
        raise ValueError("The reused B-20k results use a different evaluation setup")
    if paired_config.get("max_samples") is not None:
        raise ValueError("Semantic evaluation requires full B-20k paired results")
    model = model.to(device)
    model.set_eval()
    if args.enable_xformers_memory_efficient_attention:
        model.unet.enable_xformers_memory_efficient_attention()

    positive_dataset = SelectiveDifixDataset(
        manifest, model.tokenizer, resolution=512, training_mode="positive"
    )
    selected = _selected_records(positive_dataset.records, args.max_samples)
    selected_ids = {str(record["id"]) for record in selected}
    indices = [
        index
        for index, record in enumerate(positive_dataset.records)
        if str(record["id"]) in selected_ids
    ]
    negative_dataset = SelectiveDifixDataset(
        manifest,
        model.tokenizer,
        resolution=512,
        training_mode="negative",
        deduplicate_negative=True,
    )
    negative_ids = set(_negative_ids(selected))
    negative_indices = [
        index
        for index, record in enumerate(negative_dataset.records)
        if (
            f"{record.get('split', '')}/{record.get('pair', '')}/"
            f"{record.get('scene_id', '')}/preserve-both"
        ).lstrip("/")
        in negative_ids
    ]
    loader_args = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    positive_loader = DataLoader(Subset(positive_dataset, indices), **loader_args)
    negative_loader = DataLoader(
        Subset(negative_dataset, negative_indices), **loader_args
    )

    baseline_root = args.b20_paired_results.resolve() / "records"
    baseline_positive = _load_role_records(baseline_root, "candidate", "positive")
    baseline_negative = _load_role_records(baseline_root, "candidate", "negative")
    baseline_positive = {
        key: value for key, value in baseline_positive.items() if key in selected_ids
    }
    baseline_negative = {
        key: value for key, value in baseline_negative.items() if key in negative_ids
    }
    metric_suite = PairedMetricSuite(device)
    gallery_indices = stratified_indices(selected, args.num_gallery_samples)
    gallery_ids = {str(selected[index]["id"]) for index in gallery_indices}
    variants = _evaluate_full_checkpoint(
        model,
        positive_loader,
        negative_loader,
        baseline_positive,
        baseline_negative,
        metric_suite,
        records_root,
        gallery_ids,
        gallery_root,
        device,
        args,
    )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.constant_checkpoint is not None:
        constant_model, _, constant_metadata = load_model_from_checkpoint(
            args.constant_checkpoint.resolve(),
            expected_dataset="CCDD-11",
            expected_seed=42,
        )
        if constant_model.detail_text_mode != "constant":
            raise ValueError("--constant-checkpoint must use constant text mode")
        if constant_metadata.get("root_checkpoint_sha256") != root_checkpoint_sha:
            raise ValueError("The full and constant checkpoints must share B-20k")
        constant_text_config = constant_model.text_film_config()
        for key in (
            "pooling",
            "clip_hidden_dim",
            "text_proj_dim",
            "film_hidden_ratio",
            "condition_scale",
        ):
            if constant_text_config[key] != candidate_text_config[key]:
                raise ValueError(f"Full/constant Text-FiLM config mismatch: {key}")
        if constant_model.detail_config() != candidate_detail_config:
            raise ValueError("Full/constant DAEM-lite configuration mismatch")
        constant_model = constant_model.to(device)
        constant_model.set_eval()
        if args.enable_xformers_memory_efficient_attention:
            constant_model.unet.enable_xformers_memory_efficient_attention()
        variants["constant"] = _evaluate_constant_checkpoint(
            constant_model,
            positive_loader,
            negative_loader,
            baseline_positive,
            baseline_negative,
            metric_suite,
            records_root,
            device,
            args,
        )

    per_image = []
    effects = []
    conditions = []
    for variant, (positive, negative) in variants.items():
        if variant == "swapped":
            negative = variants["correct"][1]
        positive_rows = paired_rows(
            baseline_positive,
            positive,
            metrics=POSITIVE_METRICS,
            mode="positive",
            baseline_label="B20",
            candidate_label=variant,
        )
        negative_rows = paired_rows(
            baseline_negative,
            negative,
            metrics=NEGATIVE_METRICS,
            mode="negative",
            baseline_label="B20",
            candidate_label=variant,
        )
        for row in positive_rows:
            source = positive[row["sample_id"]]
            for key, value in source.items():
                if key.startswith("gate_difference_l") or key in (
                    "prompt_swap_output_delta_l1",
                    "correct_vs_swapped_target_psnr_gap",
                    "output_delta_l1_vs_no_text",
                ):
                    row[key] = value
        for row in positive_rows + negative_rows:
            row["variant"] = variant
        variant_effects, variant_conditions = summarize_effects(
            positive_rows,
            negative_rows,
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
        )
        for row in variant_effects + variant_conditions:
            row["variant"] = variant
        per_image.extend(positive_rows + negative_rows)
        effects.extend(variant_effects)
        conditions.extend(variant_conditions)

    _write_csv(output_dir / "per_image.csv", per_image)
    _write_csv(output_dir / "paired_effects.csv", effects)
    _write_csv(output_dir / "condition_summary.csv", conditions)
    _atomic_json(
        output_dir / "metrics.json",
        {
            "protocol": "ccdd11-text-film-semantics-v1",
            "candidate_checkpoint": str(args.candidate_checkpoint.resolve()),
            "constant_checkpoint": (
                str(args.constant_checkpoint.resolve())
                if args.constant_checkpoint is not None
                else None
            ),
            "b20_paired_results": str(args.b20_paired_results.resolve()),
            "candidate_metadata": metadata,
            "variants": list(variants),
            "positive_samples": len(selected_ids),
            "negative_samples": len(negative_ids),
            "semantic_summary": _semantic_summary(variants["correct"][0]),
            "paired_effects": effects,
        },
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--candidate-checkpoint", type=Path, required=True)
    value.add_argument("--constant-checkpoint", type=Path)
    value.add_argument("--b20-paired-results", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--workers", type=int, default=0)
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
