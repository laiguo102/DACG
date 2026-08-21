"""Native-resolution CDD-11 validation/test evaluation for DACG + Difix3D."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

from cdd11_full.metrics import rgb_psnr, rgb_ssim
from cdd11_full.protocol import ARITY_GROUPS, DEGRADATIONS
from cdd11_full.runtime import atomic_json

from .inference import save_tensor_image
from .train import _native_prediction, load_adapter_checkpoint


OUTPUTS = ("coarse", "difix")
BASE_METRICS = ("psnr", "ssim", "lpips")
METRICS = tuple(
    f"{output}_{metric}" for output in OUTPUTS for metric in BASE_METRICS
)


def _metric_mean(rows: Iterable[dict[str, Any]], metric: str) -> float:
    values = [float(row[metric]) for row in rows]
    return math.fsum(values) / len(values)


def summarize_metrics(
    rows: list[dict[str, Any]],
    degradations: tuple[str, ...] = DEGRADATIONS,
) -> dict[str, Any]:
    """Summarize DACG coarse and Difix metrics by class, arity and macro."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["degradation"])].append(row)
    missing = [name for name in degradations if not grouped[name]]
    if missing:
        raise ValueError(f"missing CDD-11 metric categories: {missing}")

    by_degradation: dict[str, dict[str, float | int]] = {}
    for name in degradations:
        values = grouped[name]
        by_degradation[name] = {
            "images": len(values),
            **{metric: _metric_mean(values, metric) for metric in METRICS},
        }

    by_arity: dict[str, dict[str, float | int]] = {}
    for group, categories in ARITY_GROUPS.items():
        by_arity[group] = {
            "categories": len(categories),
            "images": sum(int(by_degradation[name]["images"]) for name in categories),
            **{
                metric: math.fsum(float(by_degradation[name][metric]) for name in categories)
                / len(categories)
                for metric in METRICS
            },
        }

    macro = {
        metric: math.fsum(float(by_degradation[name][metric]) for name in degradations)
        / len(degradations)
        for metric in METRICS
    }
    micro = {metric: _metric_mean(rows, metric) for metric in METRICS}
    return {
        "images": len(rows),
        "micro": micro,
        "macro": macro,
        "by_degradation": by_degradation,
        "by_arity": by_arity,
    }


def _summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {"scope": "overall", "name": "micro", "images": summary["images"], **summary["micro"]},
        {"scope": "overall", "name": "macro", "images": summary["images"], **summary["macro"]},
    ]
    for name, values in summary["by_degradation"].items():
        rows.append({"scope": "degradation", "name": name, **values})
    for name, values in summary["by_arity"].items():
        rows.append({"scope": "arity", "name": name, **values})
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metadata(batch: dict[str, Any], key: str) -> Any:
    value = batch[key]
    if torch.is_tensor(value):
        return value.flatten()[0].item()
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def _safe_filename(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace("\\", "__") + ".png"


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import lpips

    from cdd11_full.model import load_dam_encoder

    from .data import PreparedCDD11Dataset
    from .model import DACGDifix

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    model = DACGDifix(
        pretrained_model_name_or_path=args.pretrained_model,
        local_files_only=args.local_files_only,
        lora_mode=args.lora_mode,
        lora_rank_unet=args.lora_rank_unet,
        lora_rank_vae=args.lora_rank_vae,
        timestep=args.timestep,
    ).to(device)
    load_adapter_checkpoint(args.difix_checkpoint, model)
    model.requires_grad_(False).eval()
    dam, _ = load_dam_encoder(str(args.dam_checkpoint), device)
    dam.requires_grad_(False).eval()
    lpips_model = lpips.LPIPS(net=args.lpips_backbone).to(device).eval().requires_grad_(False)

    dataset = PreparedCDD11Dataset(
        args.manifest,
        resolution=None,
        tokenizer=model.tokenizer,
        prompt=args.prompt,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.dataloader_num_workers > 0,
    )
    output_dir = args.output_dir.resolve()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.precision == "bf16" and device.type == "cuda"
        else nullcontext()
    )
    for batch in loader:
        tensor_batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        p_global = dam(tensor_batch["ref_01"])
        with autocast:
            prediction = _native_prediction(
                model,
                tensor_batch["main"],
                tensor_batch["ref"],
                tensor_batch["prompt_tokens"],
                p_global,
            )
        target = tensor_batch["target"]
        coarse = tensor_batch["main"]
        prediction_01 = prediction.float().add(1).mul(0.5).clamp(0, 1)
        coarse_01 = coarse.float().add(1).mul(0.5).clamp(0, 1)
        target_01 = target.float().add(1).mul(0.5).clamp(0, 1)
        sample_id = str(_metadata(batch, "id"))
        save_tensor_image(prediction_01, image_dir / _safe_filename(sample_id))
        rows.append(
            {
                "id": sample_id,
                "scene_id": str(_metadata(batch, "scene_id")),
                "degradation": str(_metadata(batch, "degradation")),
                "arity": int(_metadata(batch, "arity")),
                "coarse_psnr": rgb_psnr(coarse_01, target_01),
                "coarse_ssim": rgb_ssim(coarse_01, target_01),
                "coarse_lpips": float(lpips_model(coarse.float(), target.float()).mean()),
                "difix_psnr": rgb_psnr(prediction_01, target_01),
                "difix_ssim": rgb_ssim(prediction_01, target_01),
                "difix_lpips": float(lpips_model(prediction.float(), target.float()).mean()),
            }
        )

    summary = summarize_metrics(rows)
    atomic_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "per_image.csv", rows)
    _write_csv(output_dir / "summary.csv", _summary_rows(summary))
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--dam-checkpoint", type=Path, required=True)
    value.add_argument("--difix-checkpoint", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--pretrained-model", default="stabilityai/sd-turbo")
    value.add_argument("--local-files-only", action="store_true")
    value.add_argument(
        "--lora-mode",
        choices=("static", "p-global", "p-global-layer-id"),
        default="p-global-layer-id",
    )
    value.add_argument("--lora-rank-unet", type=int, default=32)
    value.add_argument("--lora-rank-vae", type=int, default=16)
    value.add_argument("--timestep", type=int, default=199)
    value.add_argument("--prompt", default="remove degradation")
    value.add_argument("--lpips-backbone", choices=("alex", "vgg"), default="vgg")
    value.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    value.add_argument("--device", default="auto")
    value.add_argument("--dataloader-num-workers", type=int, default=4)
    return value


def main() -> None:
    summary = evaluate(parser().parse_args())
    print(summary)


if __name__ == "__main__":
    main()
