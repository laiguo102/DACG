"""Evaluate a completed formal DACG run on frozen CDD-11-v1 official test."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from .data import build_eval_loader
from .metrics import rgb_psnr, rgb_ssim, summarize
from .model import architecture_metadata, build_model
from .protocol import ARITY_GROUPS, DEGRADATIONS, OBJECTIVE_VARIANT, PROTOCOL_NAME, deterministic_seed
from .runtime import atomic_json, file_sha256, git_state, seed_all
from .train import REPOSITORY_ROOT, _restore_image, _verify_manifests


def _save(tensor, path):
    array = tensor.detach().float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(array * 255).astype(np.uint8), "RGB").save(path)


def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Formal CDD-11 test requires CUDA")
    checkpoint_path = args.checkpoint.resolve()
    if checkpoint_path.name != "best_macro_psnr.pth":
        raise RuntimeError("Formal test requires checkpoints/best_macro_psnr.pth")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run_dir = Path(checkpoint["run_dir"]).resolve()
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if config["protocol"] != PROTOCOL_NAME or config["run_kind"] != "formal":
        raise RuntimeError("Official test is restricted to formal cdd11-v1 runs")
    if checkpoint.get("objective_variant") != OBJECTIVE_VARIANT:
        raise RuntimeError("Checkpoint is not the frozen DACG paper-loss variant")
    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    if state.get("status") != "completed" or int(state["global_step"]) != 200_000:
        raise RuntimeError(f"Formal training is incomplete: {state}")
    repository = git_state(REPOSITORY_ROOT)
    if repository["dirty"] or repository["commit"] != config["source"]["repository_commit"]:
        raise RuntimeError("Evaluation source differs from frozen training source")
    manifest_dir = Path(config["paths"]["manifest_dir"])
    _, hashes = _verify_manifests(manifest_dir)
    if hashes != checkpoint["manifest_sha256"]:
        raise RuntimeError("Test manifest identity mismatch")
    test_dir = run_dir / "test"
    if any((test_dir / name).exists() for name in ("metrics.json", "predictions", "gallery")):
        raise RuntimeError("Refusing to overwrite formal test artifacts")
    test_dir.mkdir(exist_ok=True)
    device = torch.device("cuda", 0); torch.cuda.set_device(device); seed_all(config["seed"])
    model = build_model("DACG_IR").to(device).eval()
    if sum(value.numel() for value in model.parameters()) != 30_861_200:
        raise RuntimeError("DACG-IR parameter identity mismatch")
    if checkpoint["architecture"] != architecture_metadata("DACG_IR"):
        raise RuntimeError("DACG-IR architecture metadata mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    loader, dataset = build_eval_loader(manifest_dir / "test.jsonl", "test", args.num_workers)
    if len(dataset) != 2200:
        raise RuntimeError(f"CDD-11-v1 test requires 2200 images, got {len(dataset)}")
    scenes = sorted({record.scene_id for record in dataset.records},
                    key=lambda value: deterministic_seed(f"{PROTOCOL_NAME}:test-gallery-scene:{value}"))[:2]
    gallery_lookup = {(record.scene_id, record.degradation): record.sample_id for record in dataset.records}
    ordered_gallery_ids = [gallery_lookup[(scene, degradation)]
                           for scene in scenes for degradation in DEGRADATIONS]
    gallery_ids = set(ordered_gallery_ids)
    if len(gallery_ids) != 22:
        raise RuntimeError("Formal gallery requires two complete scenes x 11 degradations")
    atomic_json(test_dir / "gallery_selection.json", {"scene_ids": scenes,
                "ordered_sample_ids": ordered_gallery_ids})
    rows, visuals, processed = [], [], 0
    atomic_json(test_dir / "state.json", {"status": "evaluating", "processed_images": 0,
                "total_images": 2200, "global_step": checkpoint["global_step"],
                "updated_at_utc": datetime.now(timezone.utc).isoformat()})
    with torch.inference_mode():
        for batch in loader:
            degraded, target = batch["degraded"].to(device), batch["target"].to(device)
            torch.cuda.synchronize(device)
            started = __import__("time").perf_counter()
            prediction = _restore_image(model, degraded, config["validation"]["inference_mode"],
                                        config["validation"]["tile_size"], config["validation"]["tile_overlap"])
            torch.cuda.synchronize(device)
            elapsed = __import__("time").perf_counter() - started
            sample_id, degradation = batch["sample_id"][0], batch["degradation"][0]
            prediction_path = test_dir / "predictions" / degradation / f"{sample_id.replace('/', '__')}.png"
            _save(prediction, prediction_path)
            row = {"sample_id": sample_id, "degradation": degradation, "arity": int(batch["arity"][0]),
                   "psnr": rgb_psnr(prediction, target), "ssim": rgb_ssim(prediction, target),
                   "inference_time_seconds": elapsed, "prediction_path": str(prediction_path)}
            rows.append(row)
            if sample_id in gallery_ids:
                root = test_dir / "gallery" / degradation / sample_id.replace("/", "__")
                paths = {name: root / f"{name}.png" for name in
                         ("input", "prediction", "target", "absolute_error", "signed_residual")}
                _save(degraded, paths["input"]); _save(prediction, paths["prediction"]); _save(target, paths["target"])
                _save((prediction - target).abs() / 0.25, paths["absolute_error"])
                _save(((prediction - degraded).clamp(-0.25, 0.25) + 0.25) / 0.5, paths["signed_residual"])
                visuals.append({"sample_id": sample_id, "degradation": degradation,
                                "arity": int(batch["arity"][0]), "psnr": row["psnr"], "ssim": row["ssim"],
                                **{f"{name}_path": str(path) for name, path in paths.items()}})
            processed += 1
            if processed % 25 == 0 or processed == 2200:
                atomic_json(test_dir / "state.json", {"status": "evaluating", "processed_images": processed,
                            "total_images": 2200, "global_step": checkpoint["global_step"],
                            "updated_at_utc": datetime.now(timezone.utc).isoformat()})
                print(f"CDD-11 formal test: {processed}/2200", flush=True)
    summary = summarize(rows, DEGRADATIONS)
    for group, categories in ARITY_GROUPS.items():
        for metric in ("psnr", "ssim"):
            summary[f"{group}/{metric}"] = math.fsum(summary[f"{value}/{metric}"] for value in categories) / len(categories)
        summary[f"{group}/categories"] = float(len(categories))
    metadata = {"protocol": PROTOCOL_NAME, "objective_variant": OBJECTIVE_VARIANT,
                "checkpoint": {"path": str(checkpoint_path), "sha256": file_sha256(checkpoint_path),
                               "global_step": checkpoint["global_step"], "best_metrics": checkpoint["best_metrics"]},
                "manifest_sha256": hashes, "repository_commit": repository["commit"],
                "precision": "bf16", "inference_mode": config["validation"]["inference_mode"],
                "batch_size": 1, "tta": False, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(test_dir / "metrics.json", {"metadata": metadata, "summary": summary, "per_image_count": len(rows)})
    summary_rows = []
    for degradation in DEGRADATIONS:
        summary_rows.append({"group": "degradation", "condition": degradation,
                             "images": int(summary[f"{degradation}/images"]),
                             "psnr": summary[f"{degradation}/psnr"], "ssim": summary[f"{degradation}/ssim"]})
    for group in ("single", "double", "triple", "macro"):
        summary_rows.append({"group": "aggregate", "condition": group,
                             "images": 2200 if group == "macro" else 200 * len(ARITY_GROUPS[group]),
                             "psnr": summary[f"{group}/psnr"], "ssim": summary[f"{group}/ssim"]})
    _write_csv(test_dir / "metrics.csv", ("group", "condition", "images", "psnr", "ssim"), summary_rows)
    _write_csv(test_dir / "per_image_metrics.csv",
               ("sample_id", "degradation", "arity", "psnr", "ssim", "inference_time_seconds", "prediction_path"), rows)
    visuals_by_id = {value["sample_id"]: value for value in visuals}
    if set(visuals_by_id) != set(ordered_gallery_ids):
        raise RuntimeError("Formal gallery is incomplete")
    atomic_json(test_dir / "gallery.json", {"global_step": checkpoint["global_step"],
                "samples": [visuals_by_id[value] for value in ordered_gallery_ids]})
    prediction_count = sum(1 for path in (test_dir / "predictions").rglob("*.png"))
    if prediction_count != 2200:
        raise RuntimeError(f"Expected 2200 predictions, got {prediction_count}")
    atomic_json(test_dir / "state.json", {"status": "completed", "processed_images": 2200,
                "total_images": 2200, "global_step": checkpoint["global_step"],
                "macro_psnr": summary["macro/psnr"], "macro_ssim": summary["macro/ssim"],
                "updated_at_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
