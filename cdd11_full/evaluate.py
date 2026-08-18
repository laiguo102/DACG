"""Evaluate one completed full-training DACG checkpoint on official CDD-11 test."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import CDD11FullDataset, dataset_fingerprint, validate_partition
from .metrics import rgb_psnr, rgb_ssim
from .model import load_network
from .protocol import DEGRADATIONS, NUM_TEST_SCENES, PROTOCOL_NAME
from .runtime import atomic_json
from .tracking import WANDB_GROUP, WANDB_PROJECT, WANDB_VERSION


def tiled_forward(model, image: torch.Tensor, tile: int, overlap: int) -> torch.Tensor:
    if tile < 16 or tile % 16:
        raise ValueError("--tile-size must be a positive multiple of 16")
    if not 0 <= overlap < tile:
        raise ValueError("--tile-overlap must be within [0, tile-size)")
    height, width = image.shape[-2:]
    if height <= tile and width <= tile:
        return model(image)
    stride = tile - overlap
    rows = list(range(0, max(height - tile, 0) + 1, stride))
    cols = list(range(0, max(width - tile, 0) + 1, stride))
    if not rows or rows[-1] != height - tile:
        rows.append(max(height - tile, 0))
    if not cols or cols[-1] != width - tile:
        cols.append(max(width - tile, 0))
    result = torch.zeros_like(image)
    weight = torch.zeros_like(image)
    for top in rows:
        for left in cols:
            patch = image[..., top:top + tile, left:left + tile]
            prediction = model(patch)
            result[..., top:top + patch.shape[-2], left:left + patch.shape[-1]] += prediction
            weight[..., top:top + patch.shape[-2], left:left + patch.shape[-1]] += 1
    return result / weight.clamp_min(1)


def _save_image(tensor: torch.Tensor, path: Path) -> None:
    array = tensor.detach().float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(array * 255).astype(np.uint8), mode="RGB").save(path)


def _wandb_test(checkpoint: dict, summary: dict, mode: str, entity: str | None, output_dir: Path) -> dict:
    if mode == "disabled":
        return {"status": "disabled"}
    import wandb

    if wandb.__version__ != WANDB_VERSION:
        raise RuntimeError(f"CDD-11 requires wandb=={WANDB_VERSION}, found {wandb.__version__}")
    if mode == "online" and not entity:
        raise ValueError("Online W&B requires --wandb-entity")
    run_id = checkpoint.get("wandb_run_id")
    if not run_id:
        raise RuntimeError("Completed checkpoint has no W&B run ID")
    run = wandb.init(entity=entity, project=WANDB_PROJECT, group=WANDB_GROUP, id=run_id,
                     resume="must", mode=mode, dir=str(output_dir / "wandb"))
    step = int(checkpoint["global_step"])
    payload = {"global_step": step, "test/macro_psnr": summary["macro"]["psnr"],
               "test/macro_ssim": summary["macro"]["ssim"]}
    for degradation, values in summary["by_degradation"].items():
        payload[f"test/{degradation}/psnr"] = values["psnr"]
        payload[f"test/{degradation}/ssim"] = values["ssim"]
    run.log(payload, step=step)
    url = getattr(run, "url", None)
    run.finish()
    return {"status": "logged", "run_id": run_id, "url": url}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-entity")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Official DACG evaluation requires a CUDA GPU")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty evaluation directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model, checkpoint = load_network(str(args.checkpoint), device)
    if checkpoint.get("protocol") != PROTOCOL_NAME or checkpoint.get("status") != "completed":
        raise RuntimeError("Official test requires a completed full CDD-11 DACG checkpoint")
    if int(checkpoint["global_step"]) != int(checkpoint["config"]["total_steps"]):
        raise RuntimeError("Checkpoint did not finish every configured optimizer step")

    scene_ids = validate_partition(args.data_root, "test", NUM_TEST_SCENES)
    fingerprint = dataset_fingerprint(args.data_root, "test", scene_ids)
    dataset = CDD11FullDataset(args.data_root, "test", scene_ids)
    rows: list[dict] = []
    values = defaultdict(lambda: {"psnr": [], "ssim": []})
    csv_path = args.output_dir / "metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("scene_id", "degradation", "psnr", "ssim",
                                                        "input_path", "target_path", "prediction_path"))
        writer.writeheader()
        with torch.inference_mode():
            for index in range(len(dataset)):
                sample = dataset[index]
                image = sample["lq"].unsqueeze(0).to(device)
                target = sample["gt"].unsqueeze(0).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                    prediction = tiled_forward(model, image, args.tile_size, args.tile_overlap)
                psnr, ssim = rgb_psnr(prediction, target), rgb_ssim(prediction, target)
                degradation, scene_id = sample["degradation"], sample["scene_id"]
                prediction_path = args.output_dir / "predictions" / degradation / f"{scene_id}.png"
                _save_image(prediction, prediction_path)
                row = {"scene_id": scene_id, "degradation": degradation, "psnr": f"{psnr:.10f}",
                       "ssim": f"{ssim:.10f}", "input_path": sample["lq_path"],
                       "target_path": sample["gt_path"], "prediction_path": str(prediction_path)}
                writer.writerow(row)
                stream.flush()
                rows.append(row)
                values[degradation]["psnr"].append(psnr)
                values[degradation]["ssim"].append(ssim)
                if (index + 1) % 50 == 0:
                    print(f"official test {index + 1}/{len(dataset)}", flush=True)

    by_degradation = {}
    for degradation in DEGRADATIONS:
        by_degradation[degradation] = {
            "images": len(values[degradation]["psnr"]),
            "psnr": sum(values[degradation]["psnr"]) / len(values[degradation]["psnr"]),
            "ssim": sum(values[degradation]["ssim"]) / len(values[degradation]["ssim"]),
        }
    summary = {
        "status": "completed", "protocol": PROTOCOL_NAME,
        "checkpoint": str(args.checkpoint.resolve()), "global_step": checkpoint["global_step"],
        "test_fingerprint": fingerprint, "images": len(rows),
        "metric_domain": "RGB [0,1], prediction clamped for metrics",
        "inference": {"mode": "tiled", "tile_size": args.tile_size, "tile_overlap": args.tile_overlap,
                      "precision": args.precision},
        "by_degradation": by_degradation,
        "macro": {
            "psnr": sum(item["psnr"] for item in by_degradation.values()) / len(DEGRADATIONS),
            "ssim": sum(item["ssim"] for item in by_degradation.values()) / len(DEGRADATIONS),
        },
    }
    summary["wandb"] = _wandb_test(checkpoint, summary, args.wandb_mode, args.wandb_entity, args.output_dir)
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
