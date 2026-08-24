"""Paired comparison of trained and step-0 selective Difix test outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

QUALITY_METRICS = ("psnr", "ssim", "lpips_vgg", "dists")
LOWER_IS_BETTER = frozenset(("lpips_vgg", "dists"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _index_rows(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in indexed:
            raise ValueError(f"Duplicate {label} sample_id: {sample_id}")
        indexed[sample_id] = row
    return indexed


def _validate_protocol(
    trained: dict[str, dict[str, str]],
    initialization: dict[str, dict[str, str]],
) -> None:
    if set(trained) != set(initialization):
        missing_initialization = sorted(set(trained) - set(initialization))
        missing_trained = sorted(set(initialization) - set(trained))
        raise ValueError(
            "Test sample sets differ: "
            f"missing initialization={missing_initialization[:3]}, "
            f"missing trained={missing_trained[:3]}"
        )
    identity_fields = ("scene_id", "pair_id", "pair", "remove", "preserve", "prompt")
    reference_fields = tuple(
        f"{stage}_{metric}"
        for stage in ("degraded", "coarse")
        for metric in QUALITY_METRICS
    )
    for sample_id in sorted(trained):
        left = trained[sample_id]
        right = initialization[sample_id]
        for field in identity_fields:
            if left[field] != right[field]:
                raise ValueError(
                    f"Protocol mismatch for {sample_id} field {field}: "
                    f"{left[field]!r} != {right[field]!r}"
                )
        for field in reference_fields:
            if not math.isclose(
                float(left[field]),
                float(right[field]),
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    f"Baseline metric mismatch for {sample_id} field {field}: "
                    f"{left[field]} != {right[field]}"
                )


def paired_rows(
    trained_rows: list[dict[str, str]],
    initialization_rows: list[dict[str, str]],
) -> list[dict[str, str | float | int]]:
    trained = _index_rows(trained_rows, "trained")
    initialization = _index_rows(initialization_rows, "initialization")
    _validate_protocol(trained, initialization)
    result: list[dict[str, str | float | int]] = []
    for sample_id in sorted(trained):
        left = trained[sample_id]
        right = initialization[sample_id]
        row: dict[str, str | float | int] = {
            field: left[field]
            for field in (
                "sample_id",
                "scene_id",
                "pair_id",
                "pair",
                "remove",
                "preserve",
                "prompt",
            )
        }
        for metric in QUALITY_METRICS:
            trained_value = float(left[f"final_{metric}"])
            initialization_value = float(right[f"final_{metric}"])
            advantage = (
                initialization_value - trained_value
                if metric in LOWER_IS_BETTER
                else trained_value - initialization_value
            )
            row[f"trained_{metric}"] = trained_value
            row[f"initialization_{metric}"] = initialization_value
            row[f"trained_advantage_{metric}"] = advantage
            row[f"trained_wins_{metric}"] = int(advantage > 0)
        result.append(row)
    return result


def _cluster_bootstrap_ci(
    rows: list[dict[str, str | float | int]],
    field: str,
    *,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    by_scene: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_scene[str(row["scene_id"])].append(float(row[field]))
    scenes = sorted(by_scene)
    sums = np.asarray([math.fsum(by_scene[scene]) for scene in scenes])
    counts = np.asarray([len(by_scene[scene]) for scene in scenes])
    estimates = np.empty(resamples, dtype=np.float64)
    chunk_size = 512
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        indices = rng.integers(0, len(scenes), size=(stop - start, len(scenes)))
        estimates[start:stop] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _summarize_group(
    rows: list[dict[str, str | float | int]],
    *,
    group: str,
    condition: str,
    resamples: int,
    rng: np.random.Generator,
) -> list[dict[str, str | float | int]]:
    result = []
    for metric in QUALITY_METRICS:
        trained_values = [float(row[f"trained_{metric}"]) for row in rows]
        initialization_values = [float(row[f"initialization_{metric}"]) for row in rows]
        advantages = [float(row[f"trained_advantage_{metric}"]) for row in rows]
        low, high = _cluster_bootstrap_ci(
            rows,
            f"trained_advantage_{metric}",
            resamples=resamples,
            rng=rng,
        )
        conclusion = "inconclusive"
        if low > 0:
            conclusion = "trained_better"
        elif high < 0:
            conclusion = "initialization_better"
        result.append(
            {
                "group": group,
                "condition": condition,
                "images": len(rows),
                "scenes": len({str(row["scene_id"]) for row in rows}),
                "metric": metric,
                "better": "lower" if metric in LOWER_IS_BETTER else "higher",
                "trained": math.fsum(trained_values) / len(rows),
                "initialization": math.fsum(initialization_values) / len(rows),
                "trained_advantage": math.fsum(advantages) / len(rows),
                "ci95_low": low,
                "ci95_high": high,
                "trained_win_rate": math.fsum(
                    float(row[f"trained_wins_{metric}"]) for row in rows
                )
                / len(rows),
                "conclusion": conclusion,
            }
        )
    return result


def summarize_comparison(
    rows: list[dict[str, str | float | int]],
    *,
    resamples: int,
    seed: int,
) -> list[dict[str, str | float | int]]:
    if resamples < 100:
        raise ValueError("--bootstrap-resamples must be at least 100")
    rng = np.random.default_rng(seed)
    by_task: dict[str, list[dict[str, str | float | int]]] = defaultdict(list)
    by_pair: dict[str, list[dict[str, str | float | int]]] = defaultdict(list)
    for row in rows:
        task = f"{row['pair']}:remove-{row['remove']}:preserve-{row['preserve']}"
        by_task[task].append(row)
        by_pair[str(row["pair"])].append(row)
    result: list[dict[str, str | float | int]] = []
    for condition in sorted(by_task):
        result.extend(
            _summarize_group(
                by_task[condition],
                group="directed_task",
                condition=condition,
                resamples=resamples,
                rng=rng,
            )
        )
    for condition in sorted(by_pair):
        result.extend(
            _summarize_group(
                by_pair[condition],
                group="pair",
                condition=condition,
                resamples=resamples,
                rng=rng,
            )
        )
    result.extend(
        _summarize_group(
            rows,
            group="overall",
            condition="micro",
            resamples=resamples,
            rng=rng,
        )
    )
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    trained_dir = args.trained_dir.resolve()
    initialization_dir = args.initialization_dir.resolve()
    output_dir = args.output_dir.resolve()
    trained_metrics = json.loads(
        (trained_dir / "metrics.json").read_text(encoding="utf-8")
    )
    initialization_metrics = json.loads(
        (initialization_dir / "metrics.json").read_text(encoding="utf-8")
    )
    initialization_source = initialization_metrics["metadata"].get("model_source")
    if initialization_source != "initialization":
        raise ValueError(
            "The initialization result metadata must contain "
            "model_source='initialization'"
        )
    trained_step = int(trained_metrics["metadata"].get("global_step", 0))
    if trained_step <= 0:
        raise ValueError("The trained result must have global_step > 0")
    paired = paired_rows(
        _read_csv(trained_dir / "per_image_metrics.csv"),
        _read_csv(initialization_dir / "per_image_metrics.csv"),
    )
    summary = summarize_comparison(
        paired,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_image_comparison.csv", paired)
    _write_csv(output_dir / "summary.csv", summary)
    payload = {
        "protocol": "cdd11-selective-difix-trained-vs-initialization-v1",
        "trained_dir": str(trained_dir),
        "initialization_dir": str(initialization_dir),
        "trained_global_step": trained_step,
        "images": len(paired),
        "bootstrap": {
            "unit": "scene_id cluster",
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "confidence_interval": "percentile 95%",
        },
        "advantage_definition": (
            "positive means trained is better than initialization; trained-initialization "
            "for PSNR/SSIM and initialization-trained for LPIPS-VGG/DISTS"
        ),
        "overall": [
            row
            for row in summary
            if row["group"] == "overall" and row["condition"] == "micro"
        ],
    }
    temporary = output_dir / "comparison.json.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(output_dir / "comparison.json")
    print(
        f"Compared {len(paired)} paired samples at trained step {trained_step}. "
        f"Results: {output_dir}",
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--trained-dir", type=Path, required=True)
    value.add_argument("--initialization-dir", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--bootstrap-resamples", type=int, default=10_000)
    value.add_argument("--bootstrap-seed", type=int, default=42)
    return value


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
