"""Full-reference metrics and deterministic summaries for selective Difix tests."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping

import torch

from cdd11_full.metrics import rgb_psnr, rgb_ssim


STAGES = ("degraded", "coarse", "final")
QUALITY_METRICS = ("psnr", "ssim", "lpips_vgg", "dists")
LOWER_IS_BETTER = frozenset(("lpips_vgg", "dists"))


class FullReferenceMetricSuite:
    """Compute classical and learned perceptual metrics for several candidates."""

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

    @staticmethod
    def _per_image(values: torch.Tensor, count: int, name: str) -> list[float]:
        flattened = values.detach().float().reshape(-1).cpu()
        if flattened.numel() != count:
            raise RuntimeError(
                f"{name} returned {flattened.numel()} values for a batch of {count}"
            )
        return [float(value) for value in flattened]

    @torch.inference_mode()
    def __call__(
        self,
        candidates: Mapping[str, torch.Tensor],
        target: torch.Tensor,
    ) -> dict[str, dict[str, float]]:
        missing = set(STAGES) - set(candidates)
        if missing:
            raise ValueError(f"Missing metric candidates: {sorted(missing)}")
        ordered = [candidates[stage].float().clamp(0, 1) for stage in STAGES]
        if any(value.shape != target.shape for value in ordered):
            shapes = {stage: tuple(candidates[stage].shape) for stage in STAGES}
            raise ValueError(
                f"Candidate/target metric shape mismatch: candidates={shapes}, "
                f"target={tuple(target.shape)}"
            )
        stacked = torch.cat(ordered, dim=0)
        targets = target.float().clamp(0, 1).expand(len(STAGES), -1, -1, -1)
        lpips_values = self._per_image(
            self.lpips(stacked.mul(2).sub(1), targets.mul(2).sub(1)),
            len(STAGES),
            "LPIPS-VGG",
        )
        dists_values = self._per_image(
            self.dists(stacked, targets), len(STAGES), "DISTS"
        )

        result: dict[str, dict[str, float]] = {}
        for index, (stage, value) in enumerate(zip(STAGES, ordered)):
            result[stage] = {
                "psnr": rgb_psnr(value, target),
                "ssim": rgb_ssim(value, target),
                "lpips_vgg": lpips_values[index],
                "dists": dists_values[index],
            }
        return result


def flatten_sample_metrics(
    metrics: Mapping[str, Mapping[str, float]],
) -> dict[str, float | int]:
    """Flatten stage metrics and make every improvement positive when Difix wins."""

    result: dict[str, float | int] = {}
    for stage in STAGES:
        for metric in QUALITY_METRICS:
            result[f"{stage}_{metric}"] = float(metrics[stage][metric])
    for metric in QUALITY_METRICS:
        final = float(metrics["final"][metric])
        coarse = float(metrics["coarse"][metric])
        improvement = coarse - final if metric in LOWER_IS_BETTER else final - coarse
        result[f"improvement_{metric}"] = improvement
        result[f"final_wins_{metric}"] = int(improvement > 0)
    return result


def _mean(rows: list[dict], field: str) -> float:
    return math.fsum(float(row[field]) for row in rows) / len(rows)


def summarize_group(rows: list[dict]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("Cannot summarize an empty metric group")
    result: dict[str, float | int] = {"images": len(rows)}
    for stage in STAGES:
        for metric in QUALITY_METRICS:
            field = f"{stage}_{metric}"
            result[field] = _mean(rows, field)
    for metric in QUALITY_METRICS:
        result[f"improvement_{metric}"] = _mean(rows, f"improvement_{metric}")
        result[f"win_rate_{metric}"] = _mean(rows, f"final_wins_{metric}")
    result["mean_inference_time_seconds"] = _mean(rows, "inference_time_seconds")
    return result


def summarize_test_rows(rows: list[dict]) -> dict[str, dict]:
    """Summarize by directed task, degradation pair, micro, and task macro."""

    if not rows:
        raise ValueError("Cannot summarize an empty selective-Difix test")
    by_task: dict[str, list[dict]] = defaultdict(list)
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        task = f"{row['pair']}:remove-{row['remove']}:preserve-{row['preserve']}"
        by_task[task].append(row)
        by_pair[str(row["pair"])].append(row)
    tasks = {key: summarize_group(by_task[key]) for key in sorted(by_task)}
    pairs = {key: summarize_group(by_pair[key]) for key in sorted(by_pair)}
    micro = summarize_group(rows)
    macro: dict[str, float | int] = {"images": len(rows), "tasks": len(tasks)}
    numeric_fields = [key for key in micro if key != "images"]
    for field in numeric_fields:
        macro[field] = math.fsum(float(value[field]) for value in tasks.values()) / len(
            tasks
        )
    return {"overall": {"micro": micro, "macro": macro}, "pairs": pairs, "tasks": tasks}


def summary_csv_rows(summary: dict[str, dict]) -> list[dict]:
    rows = []
    for condition, values in summary["tasks"].items():
        rows.append({"group": "directed_task", "condition": condition, **values})
    for condition, values in summary["pairs"].items():
        rows.append({"group": "pair", "condition": condition, **values})
    for condition, values in summary["overall"].items():
        rows.append({"group": "overall", "condition": condition, **values})
    return rows
