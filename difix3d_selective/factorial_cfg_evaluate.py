"""Factorial latent/skip attribution for the frozen CCDD-11 validation split."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .cfg import blend_cfg_latents, blend_cfg_skips
from .cfg_evaluate import (
    CFG_METRIC_FIELDS,
    CfgMetricSuite,
    REFERENCES,
    _load_partial_records,
    _record_path,
    _validate_training_run,
    validate_full_manifest,
)
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
from .protocol import preserve_pair_prompt
from .validation import stratified_indices


PROTOCOL = "ccdd11-latent-skip-factorial-v1"
LATENT_GRID = (0.0, 0.5, 1.0, 1.1, 1.2)
SKIP_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
STAGE_SAMPLE_COUNTS = {"smoke": 20, "screen": 200, "confirm": None}
STAGE_GALLERY_COUNTS = {"smoke": 2, "screen": 20, "confirm": 20}
LOWER_IS_BETTER = {
    field
    for field in CFG_METRIC_FIELDS
    if field.endswith(("_mse_l2", "_lpips_vgg", "_objective_l2_lpips", "_dists"))
}
# Encoder activations are stored shallow-to-deep; the decoder consumes them reversed.
SKIP_NAMES_BY_ENCODER_ORDER = (
    "skip_conv_4",
    "skip_conv_3",
    "skip_conv_2",
    "skip_conv_1",
)


def number_label(value: float) -> str:
    return f"{float(value):g}"


def condition_filename(condition_id: str) -> str:
    return condition_id.replace(":", "__").replace("/", "_").replace("\\", "_")


@dataclass(frozen=True)
class Condition:
    condition_id: str
    latent_beta: float
    skip_kind: str
    skip_beta: float | None = None
    active_skips: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        value = asdict(self)
        value["active_skips"] = list(self.active_skips)
        return value


def grid_condition(latent_beta: float, skip_beta: float) -> Condition:
    z = float(latent_beta)
    s = float(skip_beta)
    return Condition(
        condition_id=f"grid:z{number_label(z)}:s{number_label(s)}",
        latent_beta=z,
        skip_kind="blend",
        skip_beta=s,
        active_skips=SKIP_NAMES_BY_ENCODER_ORDER,
    )


def zero_skip_condition() -> Condition:
    return Condition("z1:skip-zero", 1.0, "zero")


def masked_skip_condition(name: str, *, only: bool) -> Condition:
    if name not in SKIP_NAMES_BY_ENCODER_ORDER:
        raise ValueError(f"Unknown VAE skip name: {name}")
    active = (name,) if only else tuple(
        candidate for candidate in SKIP_NAMES_BY_ENCODER_ORDER if candidate != name
    )
    mode = "only" if only else "drop"
    return Condition(
        f"z1:{mode}-{name}",
        1.0,
        "positive_masked",
        1.0,
        active,
    )


def deduplicate_conditions(conditions: Iterable[Condition]) -> list[Condition]:
    result: list[Condition] = []
    seen: set[str] = set()
    for condition in conditions:
        if condition.condition_id in seen:
            continue
        seen.add(condition.condition_id)
        result.append(condition)
    return result


def core_conditions() -> list[Condition]:
    conditions = [
        grid_condition(0, 0),
        grid_condition(1, 0),
        grid_condition(0, 1),
        grid_condition(1, 1),
        zero_skip_condition(),
    ]
    conditions.extend(
        masked_skip_condition(name, only=True)
        for name in reversed(SKIP_NAMES_BY_ENCODER_ORDER)
    )
    conditions.extend(
        masked_skip_condition(name, only=False)
        for name in reversed(SKIP_NAMES_BY_ENCODER_ORDER)
    )
    return conditions


def screen_conditions() -> list[Condition]:
    return [grid_condition(z, s) for z in LATENT_GRID for s in SKIP_GRID]


def _condition_summary_map(screening_payload: Mapping) -> Mapping[str, Mapping]:
    summary = screening_payload.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("Screening metrics lack summary")
    conditions = summary.get("conditions")
    if not isinstance(conditions, Mapping):
        raise ValueError("Screening metrics lack summary.conditions")
    return conditions


def select_best_screen_cell(screening_payload: Mapping) -> tuple[float, float]:
    candidates: list[tuple[float, float, float]] = []
    for condition_id, summary in _condition_summary_map(screening_payload).items():
        condition = summary.get("condition", {})
        if condition.get("skip_kind") != "blend":
            continue
        try:
            z = float(condition["latent_beta"])
            s = float(condition["skip_beta"])
            score = float(summary["overall"]["macro"]["positive_target_psnr"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid screen summary for {condition_id}") from error
        if not math.isfinite(score):
            raise ValueError(f"Non-finite screen PSNR for {condition_id}: {score}")
        if z in LATENT_GRID and s in SKIP_GRID:
            candidates.append((score, z, s))
    if len(candidates) != len(LATENT_GRID) * len(SKIP_GRID):
        raise ValueError(
            "Screening summary must contain the complete 5x5 latent/skip grid"
        )
    score, z, s = min(
        candidates,
        key=lambda value: (
            -value[0],
            abs(value[1] - 1.0) + abs(value[2] - 1.0),
            value[1],
            value[2],
        ),
    )
    del score
    return z, s


def confirm_conditions(screening_payload: Mapping) -> list[Condition]:
    best_z, best_s = select_best_screen_cell(screening_payload)
    z_index = LATENT_GRID.index(best_z)
    s_index = SKIP_GRID.index(best_s)
    neighbours = []
    for current_z_index, z in enumerate(LATENT_GRID):
        for current_s_index, s in enumerate(SKIP_GRID):
            if abs(current_z_index - z_index) + abs(current_s_index - s_index) <= 1:
                neighbours.append(grid_condition(z, s))
    return deduplicate_conditions(
        [
            *core_conditions(),
            *(grid_condition(z, 1.0) for z in LATENT_GRID),
            *(grid_condition(1.0, s) for s in SKIP_GRID),
            *neighbours,
        ]
    )


def condition_skips(
    condition: Condition,
    positive_skips: list[torch.Tensor],
    negative_skips: list[torch.Tensor],
) -> list[torch.Tensor]:
    if len(positive_skips) != len(SKIP_NAMES_BY_ENCODER_ORDER):
        raise ValueError(
            f"Expected four positive skips, got {len(positive_skips)}"
        )
    if len(negative_skips) != len(positive_skips):
        raise ValueError("Positive/negative skip count mismatch")
    if condition.skip_kind == "blend":
        assert condition.skip_beta is not None
        return blend_cfg_skips(positive_skips, negative_skips, condition.skip_beta)
    if condition.skip_kind == "zero":
        return [torch.zeros_like(value) for value in positive_skips]
    if condition.skip_kind == "positive_masked":
        active = set(condition.active_skips)
        return [
            value if name in active else torch.zeros_like(value)
            for name, value in zip(SKIP_NAMES_BY_ENCODER_ORDER, positive_skips)
        ]
    raise ValueError(f"Unknown skip kind: {condition.skip_kind}")


def _mean_metrics(rows: list[dict]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("Cannot summarize an empty row group")
    return {
        "images": len(rows),
        **{
            field: math.fsum(float(row[field]) for row in rows) / len(rows)
            for field in CFG_METRIC_FIELDS
        },
        "mean_decode_time_seconds": math.fsum(
            float(row["decode_time_seconds"]) for row in rows
        )
        / len(rows),
    }


def summarize_conditions(rows: list[dict], conditions: list[Condition]) -> dict:
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition_id"])].append(row)
    result = {}
    for condition in conditions:
        condition_rows = by_condition.get(condition.condition_id, [])
        if not condition_rows:
            raise ValueError(f"No rows for condition {condition.condition_id}")
        by_task: dict[str, list[dict]] = defaultdict(list)
        by_pair: dict[str, list[dict]] = defaultdict(list)
        for row in condition_rows:
            task = (
                f"{row['pair']}:remove-{row['remove']}:preserve-{row['preserve']}"
            )
            by_task[task].append(row)
            by_pair[str(row["pair"])].append(row)
        tasks = {key: _mean_metrics(by_task[key]) for key in sorted(by_task)}
        pairs = {key: _mean_metrics(by_pair[key]) for key in sorted(by_pair)}
        macro = {
            "images": len(condition_rows),
            "tasks": len(tasks),
            **{
                field: math.fsum(float(task[field]) for task in tasks.values())
                / len(tasks)
                for field in (*CFG_METRIC_FIELDS, "mean_decode_time_seconds")
            },
        }
        result[condition.condition_id] = {
            "condition": condition.to_dict(),
            "overall": {"micro": _mean_metrics(condition_rows), "macro": macro},
            "pairs": pairs,
            "tasks": tasks,
        }
    return {"conditions": result}


def condition_summary_rows(summary: Mapping) -> list[dict]:
    rows = []
    for condition_id, value in summary["conditions"].items():
        common = {"condition_id": condition_id, **value["condition"]}
        common["active_skips"] = ";".join(common["active_skips"])
        for group, grouped in (("directed_task", value["tasks"]), ("pair", value["pairs"])):
            for name, metrics in grouped.items():
                rows.append({**common, "group": group, "condition": name, **metrics})
        for name, metrics in value["overall"].items():
            rows.append({**common, "group": "overall", "condition": name, **metrics})
    return rows


def _index_rows(rows: list[dict]) -> dict[tuple[str, str], dict]:
    result = {}
    for row in rows:
        key = (str(row["sample_id"]), str(row["condition_id"]))
        if key in result:
            raise ValueError(f"Duplicate result row: {key}")
        result[key] = row
    return result


def _cluster_bootstrap(
    paired: list[dict], field: str, *, resamples: int, rng: np.random.Generator
) -> tuple[float, float]:
    by_scene: dict[str, list[float]] = defaultdict(list)
    for row in paired:
        by_scene[str(row["scene_id"])].append(float(row[field]))
    scenes = sorted(by_scene)
    sums = np.asarray([math.fsum(by_scene[scene]) for scene in scenes])
    counts = np.asarray([len(by_scene[scene]) for scene in scenes])
    estimates = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 512):
        stop = min(start + 512, resamples)
        indices = rng.integers(0, len(scenes), size=(stop - start, len(scenes)))
        estimates[start:stop] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def effect_definitions(available: set[str]) -> list[dict[str, str]]:
    definitions = [
        {
            "effect": "skip_positive_vs_negative_at_z1",
            "candidate": "grid:z1:s1",
            "reference": "grid:z1:s0",
        },
        {
            "effect": "latent_positive_vs_negative_at_s1",
            "candidate": "grid:z1:s1",
            "reference": "grid:z0:s1",
        },
        {
            "effect": "positive_skip_vs_zero_at_z1",
            "candidate": "grid:z1:s1",
            "reference": "z1:skip-zero",
        },
        {
            "effect": "negative_skip_vs_zero_at_z1",
            "candidate": "grid:z1:s0",
            "reference": "z1:skip-zero",
        },
    ]
    for name in reversed(SKIP_NAMES_BY_ENCODER_ORDER):
        definitions.extend(
            [
                {
                    "effect": f"marginal_{name}_all_vs_drop",
                    "candidate": "grid:z1:s1",
                    "reference": f"z1:drop-{name}",
                },
                {
                    "effect": f"standalone_{name}_only_vs_zero",
                    "candidate": f"z1:only-{name}",
                    "reference": "z1:skip-zero",
                },
            ]
        )
    return [
        definition
        for definition in definitions
        if definition["candidate"] in available and definition["reference"] in available
    ]


def paired_effects(
    rows: list[dict], *, resamples: int, seed: int
) -> tuple[list[dict], list[dict]]:
    indexed = _index_rows(rows)
    available = {key[1] for key in indexed}
    summaries = []
    differences = []
    rng = np.random.default_rng(seed)
    sample_ids = sorted({key[0] for key in indexed})
    for definition in effect_definitions(available):
        candidate_id = definition["candidate"]
        reference_id = definition["reference"]
        for metric in CFG_METRIC_FIELDS:
            paired = []
            for sample_id in sample_ids:
                candidate = indexed[(sample_id, candidate_id)]
                reference = indexed[(sample_id, reference_id)]
                raw_delta = float(candidate[metric]) - float(reference[metric])
                advantage = -raw_delta if metric in LOWER_IS_BETTER else raw_delta
                item = {
                    "effect": definition["effect"],
                    "candidate": candidate_id,
                    "reference": reference_id,
                    "sample_id": sample_id,
                    "scene_id": candidate["scene_id"],
                    "metric": metric,
                    "raw_delta": raw_delta,
                    "advantage": advantage,
                }
                paired.append(item)
                differences.append(item)
            low, high = _cluster_bootstrap(
                paired, "advantage", resamples=resamples, rng=rng
            )
            conclusion = "inconclusive"
            if low > 0:
                conclusion = "improved"
            elif high < 0:
                conclusion = "degraded"
            summaries.append(
                {
                    "effect": definition["effect"],
                    "candidate": candidate_id,
                    "reference": reference_id,
                    "metric": metric,
                    "better": "lower" if metric in LOWER_IS_BETTER else "higher",
                    "images": len(paired),
                    "scenes": len({item["scene_id"] for item in paired}),
                    "raw_delta_mean": math.fsum(item["raw_delta"] for item in paired)
                    / len(paired),
                    "advantage_mean": math.fsum(item["advantage"] for item in paired)
                    / len(paired),
                    "ci95_low": low,
                    "ci95_high": high,
                    "win_rate": math.fsum(item["advantage"] > 0 for item in paired)
                    / len(paired),
                    "conclusion": conclusion,
                }
            )
    return summaries, differences


def interaction_summary(
    rows: list[dict], *, resamples: int, seed: int
) -> list[dict]:
    indexed = _index_rows(rows)
    required = (
        "grid:z0:s0",
        "grid:z1:s0",
        "grid:z0:s1",
        "grid:z1:s1",
    )
    available = {key[1] for key in indexed}
    if not set(required).issubset(available):
        return []
    rng = np.random.default_rng(seed)
    result = []
    for metric in CFG_METRIC_FIELDS:
        paired = []
        for sample_id in sorted({key[0] for key in indexed}):
            value = (
                float(indexed[(sample_id, "grid:z1:s1")][metric])
                - float(indexed[(sample_id, "grid:z1:s0")][metric])
                - float(indexed[(sample_id, "grid:z0:s1")][metric])
                + float(indexed[(sample_id, "grid:z0:s0")][metric])
            )
            paired.append(
                {
                    "scene_id": indexed[(sample_id, "grid:z1:s1")]["scene_id"],
                    "interaction": value,
                }
            )
        low, high = _cluster_bootstrap(
            paired, "interaction", resamples=resamples, rng=rng
        )
        conclusion = "inconclusive"
        if low > 0:
            conclusion = "positive"
        elif high < 0:
            conclusion = "negative"
        result.append(
            {
                "metric": metric,
                "images": len(paired),
                "interaction_mean": math.fsum(item["interaction"] for item in paired)
                / len(paired),
                "ci95_low": low,
                "ci95_high": high,
                "conclusion": conclusion,
            }
        )
    return result


def build_influence_report(
    summary: Mapping,
    effects: list[dict],
    interactions: list[dict],
) -> dict:
    best_z, best_s = select_best_screen_cell({"summary": summary}) if len(
        [
            value
            for value in summary["conditions"].values()
            if value["condition"]["skip_kind"] == "blend"
        ]
    ) == len(LATENT_GRID) * len(SKIP_GRID) else (None, None)
    primary_effects = [
        row
        for row in effects
        if row["metric"] in ("positive_target_mse_l2", "positive_target_psnr")
    ]
    layer_rows = [
        row
        for row in effects
        if row["metric"] == "positive_target_psnr"
        and row["effect"].startswith(("marginal_", "standalone_"))
    ]
    layer_rows.sort(key=lambda row: (-float(row["advantage_mean"]), row["effect"]))
    primary_interaction = next(
        (row for row in interactions if row["metric"] == "positive_target_mse_l2"),
        None,
    )
    return {
        "primary_metric": "positive_target task-macro PSNR",
        "best_screen_cell": (
            None if best_z is None else {"latent_beta": best_z, "skip_beta": best_s}
        ),
        "primary_paired_effects": primary_effects,
        "positive_target_mse_interaction": primary_interaction,
        "skip_layer_psnr_ranking": layer_rows,
        "interpretation_note": (
            "Hybrid latent/skip conditions are diagnostic counterfactuals, not deployment modes. "
            "Latent differences include the current sequential VAE posterior sampling path."
        ),
    }


def _flatten_records(records: list[dict]) -> list[dict]:
    rows = []
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
        for result in record["results"]:
            rows.append({**common, **result})
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


def save_condition_sheet(
    references: Mapping[str, torch.Tensor],
    predictions: Mapping[str, torch.Tensor],
    path: Path,
) -> None:
    panels = [
        ("double degraded", references["negative_identity"]),
        ("DACG coarse", references["coarse"]),
        ("selective target", references["positive_target"]),
        ("clean GT", references["clean_gt"]),
        *predictions.items(),
    ]
    images = [(label, _tensor_to_pil(value)) for label, value in panels]
    width, height = images[0][1].size
    columns = 4
    header = 24
    sheet = Image.new(
        "RGB", (columns * width, math.ceil(len(images) / columns) * (height + header)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (label, image) in enumerate(images):
        left = index % columns * width
        top = index // columns * (height + header)
        draw.text((left + 5, top + 5), label, fill="black", font=font)
        sheet.paste(image, (left, top + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _validate_screening_payload(
    payload: Mapping, *, checkpoint_sha256: str, manifest_sha256: str, seed: int
) -> None:
    metadata = payload.get("metadata", {})
    config = metadata.get("config", {}) if isinstance(metadata, Mapping) else {}
    expected = {
        "protocol": PROTOCOL,
        "stage": "screen",
        "checkpoint_sha256": checkpoint_sha256,
        "validation_manifest_sha256": manifest_sha256,
        "seed": seed,
    }
    actual = {
        "protocol": metadata.get("protocol") if isinstance(metadata, Mapping) else None,
        "stage": config.get("stage"),
        "checkpoint_sha256": config.get("checkpoint", {}).get("sha256"),
        "validation_manifest_sha256": config.get("validation_manifest_sha256"),
        "seed": config.get("seed"),
    }
    mismatches = {
        key: (actual[key], value) for key, value in expected.items() if actual[key] != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={observed!r} expected={required!r}"
            for key, (observed, required) in mismatches.items()
        )
        raise ValueError(f"Screening results do not match confirm protocol: {details}")
    select_best_screen_cell(payload)


def _load_conditions(
    args: argparse.Namespace, *, checkpoint_sha256: str, manifest_sha256: str
) -> tuple[list[Condition], dict | None]:
    if args.stage == "smoke":
        if args.screening_results is not None:
            raise ValueError("--screening-results is only valid for --stage confirm")
        return core_conditions(), None
    if args.stage == "screen":
        if args.screening_results is not None:
            raise ValueError("--screening-results is only valid for --stage confirm")
        return screen_conditions(), None
    if args.screening_results is None:
        raise ValueError("--stage confirm requires --screening-results")
    path = args.screening_results.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _read_json(path)
    _validate_screening_payload(
        payload,
        checkpoint_sha256=checkpoint_sha256,
        manifest_sha256=manifest_sha256,
        seed=args.seed,
    )
    return confirm_conditions(payload), payload


def _validate_partial_record(record: Mapping, condition_ids: list[str]) -> None:
    actual = [str(result.get("condition_id")) for result in record.get("results", [])]
    if actual != condition_ids:
        raise ValueError(
            f"Partial record {record.get('sample_id')} uses a different condition set"
        )


def _log_wandb(
    args: argparse.Namespace,
    config: dict,
    summary: Mapping,
    effects: list[dict],
    records: list[dict],
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
        config={**config, "wandb_schema": "latent-skip-factorial-v1"},
        job_type="latent-skip-attribution",
    )
    surface_rows = []
    for value in summary["conditions"].values():
        condition = value["condition"]
        if condition["skip_kind"] != "blend":
            continue
        macro = value["overall"]["macro"]
        surface_rows.append(
            [
                condition["latent_beta"],
                condition["skip_beta"],
                macro["positive_target_psnr"],
                macro["positive_target_mse_l2"],
                macro["negative_identity_psnr"],
            ]
        )
    run.log(
        {
            "factorial/surface": wandb.Table(
                columns=[
                    "latent_beta",
                    "skip_beta",
                    "positive_target_psnr",
                    "positive_target_mse_l2",
                    "negative_identity_psnr",
                ],
                data=surface_rows,
            ),
            "factorial/effects": wandb.Table(
                columns=list(effects[0]) if effects else ["effect"],
                data=(
                    [[row.get(key) for key in effects[0]] for row in effects]
                    if effects
                    else []
                ),
            ),
        }
    )
    grid = {
        (
            float(value["condition"]["latent_beta"]),
            float(value["condition"]["skip_beta"]),
        ): value["overall"]["macro"]
        for value in summary["conditions"].values()
        if value["condition"]["skip_kind"] == "blend"
    }
    latent_xs = [z for z in LATENT_GRID if any((z, s) in grid for s in SKIP_GRID)]
    latent_series = [
        (s, [grid[(z, s)]["positive_target_psnr"] for z in latent_xs])
        for s in SKIP_GRID
        if all((z, s) in grid for z in latent_xs)
    ]
    if latent_xs and latent_series:
        run.log(
            {
                "factorial/latent_axis_psnr": wandb.plot.line_series(
                    xs=latent_xs,
                    ys=[values for _, values in latent_series],
                    keys=[f"skip_beta={number_label(s)}" for s, _ in latent_series],
                    title="Positive-target PSNR by latent beta",
                    xname="latent_beta",
                )
            }
        )
    skip_xs = [s for s in SKIP_GRID if any((z, s) in grid for z in LATENT_GRID)]
    skip_series = [
        (z, [grid[(z, s)]["positive_target_psnr"] for s in skip_xs])
        for z in LATENT_GRID
        if all((z, s) in grid for s in skip_xs)
    ]
    if skip_xs and skip_series:
        run.log(
            {
                "factorial/skip_axis_psnr": wandb.plot.line_series(
                    xs=skip_xs,
                    ys=[values for _, values in skip_series],
                    keys=[f"latent_beta={number_label(z)}" for z, _ in skip_series],
                    title="Positive-target PSNR by skip beta",
                    xname="skip_beta",
                )
            }
        )
    galleries = [
        [record["sample_id"], record["pair"], record["remove"], record["preserve"], wandb.Image(record["contact_sheet_path"])]
        for record in records
        if record.get("contact_sheet_path")
    ]
    if galleries:
        run.log(
            {
                "factorial/gallery": wandb.Table(
                    columns=["sample_id", "pair", "remove", "preserve", "sheet"],
                    data=galleries,
                )
            }
        )
    run.finish()


def run(args: argparse.Namespace) -> None:
    from torch.utils.data import DataLoader, Subset
    from tqdm.auto import tqdm

    from .data import SelectiveDifixDataset
    from .model import SelectiveDifix, load_model_checkpoint

    fixed_protocol = {"resolution": 512, "lora_rank_vae": 4, "timestep": 199, "seed": 42}
    mismatches = {
        key: (getattr(args, key), expected)
        for key, expected in fixed_protocol.items()
        if getattr(args, key) != expected
    }
    if mismatches:
        raise ValueError(f"CCDD-11 factorial protocol mismatch: {mismatches}")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.bootstrap_resamples < 100:
        raise ValueError("--bootstrap-resamples must be at least 100")
    checkpoint = args.checkpoint.resolve()
    run_dir, manifest, preparation = _validate_training_run(checkpoint)
    records = load_records(manifest)
    validate_full_manifest(records)
    checkpoint_sha256 = _file_sha256(checkpoint)
    manifest_sha256 = _file_sha256(manifest)
    conditions, screening_payload = _load_conditions(
        args,
        checkpoint_sha256=checkpoint_sha256,
        manifest_sha256=manifest_sha256,
    )
    requested_count = STAGE_SAMPLE_COUNTS[args.stage]
    indices = (
        list(range(len(records)))
        if requested_count is None
        else stratified_indices(records, requested_count)
    )
    selected_records = [records[index] for index in indices]
    expected_ids = {str(record["id"]) for record in selected_records}
    output_dir = (
        args.output_dir
        or run_dir / "latent_skip_attribution_v1" / args.stage
    ).resolve()
    args.output_dir = output_dir
    if (output_dir / "metrics.json").is_file():
        raise RuntimeError(f"Completed factorial results already exist in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    experiment_metadata = checkpoint_payload.get("experiment_metadata", {})
    if experiment_metadata.get("dataset") != "CCDD-11":
        raise ValueError("Checkpoint metadata does not identify CCDD-11 training")
    if float(experiment_metadata.get("negative_train_probability", 0.0)) <= 0:
        raise ValueError("Checkpoint was not trained with negative CCDD-11 samples")
    del checkpoint_payload
    gallery_count = (
        STAGE_GALLERY_COUNTS[args.stage]
        if args.num_gallery_samples is None
        else args.num_gallery_samples
    )
    if gallery_count < 0:
        raise ValueError("--num-gallery-samples must be non-negative")
    condition_payload = [condition.to_dict() for condition in conditions]
    config = {
        "protocol": PROTOCOL,
        "stage": args.stage,
        "checkpoint": {**_checkpoint_identity(checkpoint), "sha256": checkpoint_sha256},
        "validation_manifest": str(manifest.resolve()),
        "validation_manifest_sha256": manifest_sha256,
        "conditions": condition_payload,
        "latent_grid": list(LATENT_GRID),
        "skip_grid": list(SKIP_GRID),
        "resolution": args.resolution,
        "lora_rank_vae": args.lora_rank_vae,
        "timestep": args.timestep,
        "seed": args.seed,
        "posterior_sampling": "current sequential latent_dist.sample() path",
        "device": args.device,
        "mixed_precision": args.mixed_precision,
        "workers": args.workers,
        "xformers_memory_efficient_attention": args.enable_xformers_memory_efficient_attention,
        "gallery_samples": gallery_count,
        "validation_samples": len(selected_records),
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "negative_train_probability": experiment_metadata["negative_train_probability"],
        "latent_formula": "z_negative + latent_beta * (z_positive - z_negative)",
        "skip_formula": "skip_negative + skip_beta * (skip_positive - skip_negative)",
    }
    if screening_payload is not None:
        best_z, best_s = select_best_screen_cell(screening_payload)
        config["screening_results"] = {
            "path": str(args.screening_results.resolve()),
            "sha256": _file_sha256(args.screening_results.resolve()),
            "selected_best_cell": {"latent_beta": best_z, "skip_beta": best_s},
        }
    state_path = output_dir / "state.json"
    records_dir = output_dir / "records"
    partial = _load_partial_records(records_dir)
    if state_path.is_file():
        previous = _read_json(state_path)
        if previous.get("config") != config:
            raise RuntimeError("Existing partial factorial state uses a different configuration")
    elif partial:
        raise RuntimeError("Partial factorial records exist without state metadata")
    unknown = set(partial) - expected_ids
    if unknown:
        raise ValueError(f"Partial records contain unknown samples: {sorted(unknown)[:3]}")
    condition_ids = [condition.condition_id for condition in conditions]
    for record in partial.values():
        _validate_partial_record(record, condition_ids)
    _atomic_json(
        state_path,
        {
            "status": "evaluating",
            "processed_samples": len(partial),
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
    model = SelectiveDifix(lora_rank_vae=args.lora_rank_vae, timestep=args.timestep).to(device)
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
    gallery_ids = {
        str(dataset.records[index]["id"])
        for index in stratified_indices(dataset.records, gallery_count)
    } & expected_ids if gallery_count else set()
    pending = [index for index in indices if str(dataset.records[index]["id"]) not in partial]
    loader = DataLoader(
        Subset(dataset, pending),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    by_id = {str(record["id"]): record for record in records}
    progress = tqdm(total=len(selected_records), initial=len(partial), desc=f"factorial-{args.stage}")
    for batch in loader:
        sample_id = str(batch["sample_id"][0])
        positive_source = batch["conditioning_pixel_values"].to(device, non_blocking=True)
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
                z_positive, z_negative, positive_skips, negative_skips = model.cfg_latents(
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
        gallery_predictions = {}
        results = []
        for condition in conditions:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            decode_started = time.perf_counter()
            with torch.inference_mode(), _autocast(device, args.mixed_precision):
                latent = blend_cfg_latents(z_positive, z_negative, condition.latent_beta)
                skips = condition_skips(condition, positive_skips, negative_skips)
                prediction = model.decode_main_latent(latent, skips)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            decode_elapsed = time.perf_counter() - decode_started
            metrics = metric_suite(prediction, references)
            prediction_path = ""
            if sample_id in gallery_ids:
                prediction_01 = prediction.float().add(1).mul(0.5).clamp(0, 1)
                gallery_predictions[condition.condition_id] = prediction_01.cpu()
                destination = (
                    output_dir
                    / "gallery"
                    / _safe_id(sample_id)
                    / f"{condition_filename(condition.condition_id)}.png"
                )
                _save_image(prediction_01, destination)
                prediction_path = str(destination)
            item = condition.to_dict()
            item["active_skips"] = ";".join(item["active_skips"])
            results.append(
                {
                    **item,
                    **metrics,
                    "decode_time_seconds": decode_elapsed,
                    "prediction_path": prediction_path,
                }
            )
            del latent, skips, prediction
        contact_sheet_path = ""
        if sample_id in gallery_ids:
            contact_sheet = output_dir / "gallery" / _safe_id(sample_id) / "contact_sheet.png"
            save_condition_sheet(
                {
                    **references,
                    "coarse": coarse.float().add(1).mul(0.5).clamp(0, 1),
                    "positive_target": positive_target.float().add(1).mul(0.5).clamp(0, 1),
                    "negative_identity": degraded.float().add(1).mul(0.5).clamp(0, 1),
                    "clean_gt": clean_gt.float().add(1).mul(0.5).clamp(0, 1),
                },
                gallery_predictions,
                contact_sheet,
            )
            contact_sheet_path = str(contact_sheet)
        source_record = by_id[sample_id]
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
        _atomic_json(_record_path(records_dir, record), record)
        partial[sample_id] = record
        progress.update(1)
        if len(partial) % 10 == 0 or len(partial) == len(selected_records):
            _atomic_json(
                state_path,
                {
                    "status": "evaluating",
                    "processed_samples": len(partial),
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
    if set(partial) != expected_ids:
        raise RuntimeError(f"Factorial evaluation incomplete: missing {len(expected_ids - set(partial))} samples")
    ordered = [partial[str(record["id"])] for record in selected_records]
    rows = _flatten_records(ordered)
    summary = summarize_conditions(rows, conditions)
    effects, paired_differences = paired_effects(
        rows, resamples=args.bootstrap_resamples, seed=args.bootstrap_seed
    )
    interactions = interaction_summary(
        rows, resamples=args.bootstrap_resamples, seed=args.bootstrap_seed
    )
    influence = build_influence_report(summary, effects, interactions)
    if screening_payload is not None:
        best_z, best_s = select_best_screen_cell(screening_payload)
        influence["best_screen_cell"] = {
            "latent_beta": best_z,
            "skip_beta": best_s,
        }
    _write_csv(output_dir / "per_image_metrics.csv", rows)
    _write_csv(output_dir / "condition_summary.csv", condition_summary_rows(summary))
    _write_csv(output_dir / "paired_effects.csv", effects)
    _write_csv(output_dir / "paired_differences.csv", paired_differences)
    _write_csv(output_dir / "interaction_summary.csv", interactions)
    _atomic_json(output_dir / "influence_report.json", influence)
    metadata = {
        "protocol": PROTOCOL,
        "dataset": "CCDD-11",
        "split": "validation",
        "global_step": global_step,
        "created_at_utc": _now(),
        "config": config,
        "training_preparation": preparation,
        "metrics": {
            "mse_l2": "MSE on [-1,1]",
            "lpips_vgg": "lpips.LPIPS(net='vgg') on [-1,1]",
            "objective_l2_lpips": "mse_l2 + lpips_vgg",
            "psnr": "CDD11 RGB PSNR on [0,1]",
            "ssim": "CDD11 RGB SSIM on [0,1]",
            "dists": "piq.DISTS(reduction='none') on [0,1]",
        },
        "references": list(REFERENCES),
        "software": {
            name: _package_version(name)
            for name in ("torch", "torchvision", "diffusers", "lpips", "piq", "wandb")
        },
    }
    metrics_payload = {
        "metadata": metadata,
        "summary": summary,
        "paired_effects": effects,
        "interactions": interactions,
        "influence_report": influence,
        "per_image_count": len(rows),
        "validation_sample_count": len(ordered),
        "condition_count": len(conditions),
    }
    _atomic_json(
        state_path,
        {
            "status": "local_completed_reporting_pending",
            "processed_samples": len(ordered),
            "total_samples": len(ordered),
            "config": config,
            "global_step": global_step,
            "updated_at_utc": _now(),
        },
    )
    _log_wandb(args, config, summary, effects, ordered)
    _atomic_json(output_dir / "metrics.json", metrics_payload)
    _atomic_json(
        state_path,
        {
            "status": "completed",
            "processed_samples": len(ordered),
            "total_samples": len(ordered),
            "config": config,
            "global_step": global_step,
            "updated_at_utc": _now(),
        },
    )
    print(
        f"Completed factorial {args.stage}: {len(ordered)} samples x {len(conditions)} conditions at step {global_step}. Results: {output_dir}",
        flush=True,
    )


def _sample_seed(seed: int, sample_id: str) -> int:
    import hashlib

    digest = hashlib.sha256(f"ccdd11-cfg:{seed}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--stage", choices=("smoke", "screen", "confirm"), required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--screening-results", type=Path)
    value.add_argument("--output-dir", type=Path)
    value.add_argument("--num-gallery-samples", type=int)
    value.add_argument("--bootstrap-resamples", type=int, default=10_000)
    value.add_argument("--bootstrap-seed", type=int, default=42)
    value.add_argument("--resolution", type=int, default=512)
    value.add_argument("--lora-rank-vae", type=int, default=4)
    value.add_argument("--timestep", type=int, default=199)
    value.add_argument("--workers", type=int, default=8)
    value.add_argument("--device", default="cuda")
    value.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--enable-xformers-memory-efficient-attention", action="store_true")
    value.add_argument("--report-to", choices=("wandb", "none"), default="wandb")
    value.add_argument("--wandb-entity", default="c14150591-sjtu")
    value.add_argument("--wandb-project", default="difix-ccdd11-selective")
    value.add_argument("--wandb-run-name")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.wandb_run_name is None:
        args.wandb_run_name = f"ccdd-all5-best-latent-skip-{args.stage}-v1"
    run(args)


if __name__ == "__main__":
    main()
