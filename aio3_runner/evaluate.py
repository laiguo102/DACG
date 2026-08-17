"""Formal AIO3-v1 evaluation for DACG-IR.

Usage:
    python -m aio3_runner.evaluate --checkpoint RUN/checkpoints/best_macro_psnr.pth
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .adapter import build_model
from .data import AIO3ManifestDataset, PROTOCOL, sha256_file, stable_seed, verify_frozen_manifests
from .metrics import image_metrics
from .results import aggregate_metrics, write_compliance_report, write_result_tables
from .tracking import WandbTracker


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _validate_formal_run(checkpoint: Path, run_dir: Path) -> dict[str, Any]:
    if checkpoint.name != "best_macro_psnr.pth":
        raise ValueError("formal evaluation accepts only checkpoints/best_macro_psnr.pth")
    state_path = run_dir / "run_state.json"
    if not state_path.exists():
        raise ValueError(f"missing run state: {state_path}")
    state = _read_json(state_path)
    if (
        state.get("run_kind") != "formal" or state.get("status") != "completed"
        or int(state.get("global_step", -1)) != 200000
    ):
        raise ValueError("formal evaluation requires run_kind=formal, status=completed and global_step=200000")
    return state


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a mapping with model metadata")
    config = checkpoint.get("model_config") or checkpoint.get("metadata", {}).get("model_config") or {}
    model = build_model(config)
    state = checkpoint.get("model") or checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state is None:
        raise ValueError("checkpoint has no model/model_state_dict/state_dict")
    # Accept protocol adapter states and older Lightning 'net.' states explicitly.
    if any(key.startswith("net.") for key in state):
        state = {f"model.{key[4:]}": value for key, value in state.items() if key.startswith("net.")}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, checkpoint


def _validate_checkpoint_metadata(checkpoint: dict[str, Any], manifest_hashes: dict[str, str]) -> None:
    metadata = checkpoint.get("metadata", {})
    protocol = checkpoint.get("protocol", metadata.get("protocol"))
    run_kind = checkpoint.get("run_kind", metadata.get("run_kind"))
    if protocol != PROTOCOL or run_kind != "formal":
        raise ValueError("checkpoint metadata must identify protocol=aio3-v1 and run_kind=formal")
    if int(checkpoint.get("global_step", -1)) <= 0:
        raise ValueError("best checkpoint must record its validation selection step")
    saved_hashes = checkpoint.get("manifest_sha256", metadata.get("manifest_sha256"))
    if saved_hashes != manifest_hashes:
        raise ValueError("checkpoint manifest hashes do not match the frozen manifests")
    git = metadata.get("git", checkpoint.get("git", {}))
    if git.get("dirty") is not False or not git.get("commit"):
        raise ValueError("checkpoint must record a clean Git commit")
    try:
        current_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            check=True, capture_output=True, text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("formal evaluation requires an accessible Git worktree") from error
    if dirty or current_commit != git["commit"]:
        raise ValueError("formal evaluation requires the clean Git commit recorded by the checkpoint")


def _save_rgb(tensor: torch.Tensor, path: Path) -> None:
    array = tensor.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(array * 255).astype(np.uint8), mode="RGB").save(path)


def _save_error(tensor: torch.Tensor, path: Path, signed: bool) -> None:
    value = tensor.detach().float().cpu()
    if signed:
        # Fixed [-0.25, 0.25] diverging map: negative=blue, zero=gray, positive=red.
        normalized = (value.mean(0).clamp(-0.25, 0.25) / 0.25 + 1.0) / 2.0
        red, blue = normalized, 1.0 - normalized
        green = 1.0 - (normalized - 0.5).abs() * 2.0
        array = torch.stack((red, green * 0.75, blue), -1).numpy()
    else:
        # Fixed [0, 0.25] grayscale map.
        normalized = value.mean(0).clamp(0, 0.25) / 0.25
        array = normalized.unsqueeze(-1).expand(-1, -1, 3).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(array * 255).astype(np.uint8), mode="RGB").save(path)


def _save_gallery_composite(paths: dict[str, Path], destination: Path) -> None:
    order = ("input", "prediction", "target", "absolute_error", "signed_residual")
    images = []
    for name in order:
        with Image.open(paths[name]) as image:
            images.append(image.convert("RGB").copy())
    height = max(image.height for image in images)
    width = sum(image.width for image in images)
    canvas = Image.new("RGB", (width, height))
    left = 0
    for image in images:
        canvas.paste(image, (left, 0))
        left += image.width
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def _prediction_subdir(task: str, sigma: int | None) -> str:
    if task == "denoise":
        return f"BSD68_sigma{sigma}"
    return "Rain100L" if task == "derain" else "SOTS-outdoor"


def _gallery_ids(records: list[dict[str, Any]]) -> set[str]:
    groups: dict[str, list[str]] = {}
    for record in records:
        metadata = record.get("metadata", {})
        sigma = metadata.get("sigma", record.get("sigma"))
        key = f"denoise-{sigma}" if record["task"] == "denoise" else record["task"]
        groups.setdefault(key, []).append(record["id"])
    selected: set[str] = set()
    for key, ids in groups.items():
        count = 2 if key.startswith("denoise-") else 4
        selected.update(sorted(ids, key=lambda value: stable_seed(f"{PROTOCOL}:gallery:{value}"))[:count])
    return selected


def evaluate(
    checkpoint_path: str | Path,
    *,
    manifest_dir: str | Path | None = None,
    data_root: str | Path | None = None,
    num_workers: int = 4,
    device_name: str | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).resolve()
    run_dir = checkpoint_path.parent.parent
    run_state = _validate_formal_run(checkpoint_path, run_dir)
    run_config = _read_json(run_dir / "config.yaml")
    frozen_data_root = run_config.get("paths", {}).get("data_root")
    if data_root is None:
        data_root = frozen_data_root
    elif frozen_data_root and Path(data_root).resolve() != Path(frozen_data_root).resolve():
        raise ValueError("evaluation data-root differs from the frozen training configuration")
    manifest_root = Path(manifest_dir) if manifest_dir else run_dir / "manifests"
    manifest_hashes = verify_frozen_manifests(manifest_root)
    test_dir = run_dir / "test"
    if test_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing formal test output: {test_dir}")
    test_dir.mkdir(parents=True)
    _write_json(test_dir / "state.json", {"status": "running", "completed": 0, "total": 804})

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    gallery: list[dict[str, Any]] = []
    try:
        dataset = AIO3ManifestDataset(manifest_root / "test.jsonl", split="test", data_root=data_root)
        if len(dataset) != 804:
            raise ValueError(f"formal test manifest must contain 804 records, found {len(dataset)}")
        selected = _gallery_ids(dataset.records)
        _write_json(test_dir / "gallery_selection.json", {"protocol": PROTOCOL, "sample_ids": sorted(selected)})
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        model, checkpoint = _load_checkpoint(checkpoint_path, device)
        _validate_checkpoint_metadata(checkpoint, manifest_hashes)

        with torch.inference_mode():
            for position, batch in enumerate(loader, 1):
                sample_id = batch["id"][0]
                task = batch["task"][0]
                sigma_value = batch["sigma"]
                sigma = int(sigma_value[0]) if torch.is_tensor(sigma_value) else -1
                degraded, target = batch["input"].to(device), batch["target"].to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                tick = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    restored_raw = model(degraded)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                inference_seconds = time.perf_counter() - tick
                score = image_metrics(restored_raw.float(), target.float())
                subdir = _prediction_subdir(task, sigma)
                safe_id = sample_id.replace("/", "_").replace("\\", "_")
                prediction_relative = Path("predictions") / subdir / f"{safe_id}.png"
                _save_rgb(restored_raw[0], test_dir / prediction_relative)
                metadata = dataset.records[position - 1].get("metadata", {})
                rows.append({
                    "dataset": metadata.get("dataset", subdir), "task": task,
                    "sigma": sigma if task == "denoise" else "",
                    "sample_id": sample_id, **score, "inference_seconds": inference_seconds,
                    "prediction": prediction_relative.as_posix(),
                })
                if sample_id in selected:
                    prefix = test_dir / "gallery" / safe_id
                    paths = {
                        "input": prefix.with_name(prefix.name + "_input.png"),
                        "prediction": prefix.with_name(prefix.name + "_prediction.png"),
                        "target": prefix.with_name(prefix.name + "_target.png"),
                        "absolute_error": prefix.with_name(prefix.name + "_absolute_error.png"),
                        "signed_residual": prefix.with_name(prefix.name + "_signed_residual.png"),
                    }
                    _save_rgb(degraded[0], paths["input"])
                    _save_rgb(restored_raw[0], paths["prediction"])
                    _save_rgb(target[0], paths["target"])
                    _save_error((restored_raw - target).abs()[0], paths["absolute_error"], False)
                    residual = (restored_raw - degraded)[0]
                    _save_error(residual, paths["signed_residual"], True)
                    artifact_composite = test_dir / "artifact_gallery" / f"{safe_id}.png"
                    _save_gallery_composite(paths, artifact_composite)
                    gallery.append({
                        "sample_id": sample_id, "task": task,
                        "sigma": sigma if task == "denoise" else None, **score,
                        "residual_mean": float(residual.mean()),
                        "residual_negative_fraction": float((residual < 0).float().mean()),
                        "artifact_composite": artifact_composite.relative_to(test_dir).as_posix(),
                        **{name: path.relative_to(test_dir).as_posix() for name, path in paths.items()},
                    })
                if position % 25 == 0 or position == len(dataset):
                    _write_json(test_dir / "state.json", {"status": "running", "completed": position, "total": len(dataset)})

        summary = aggregate_metrics(rows)
        if len(gallery) != 14:
            raise ValueError(f"formal gallery must contain 14 samples, selected {len(gallery)}")
        duration = time.perf_counter() - started
        checkpoint_metadata = checkpoint.get("metadata", {})
        result_metadata = {
            "protocol": PROTOCOL,
            "checkpoint": str(checkpoint_path),
            "run_dir": str(run_dir),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "manifest_sha256": manifest_hashes,
            "git": checkpoint_metadata.get("git", {}),
            "wandb_url": checkpoint_metadata.get("wandb", {}).get("run_url"),
            "sample_count": len(rows),
            "total_runtime_seconds": duration,
            "tiled_inference": False,
            "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
            "parameters_trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "peak_gpu_memory_gib": (
                torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
            ),
        }
        write_result_tables(test_dir, rows, summary, result_metadata)
        _write_json(test_dir / "gallery.json", gallery)
        write_compliance_report(
            test_dir / "compliance_report.md", run=run_state, result=result_metadata,
            summary=summary, rows=rows, gallery_png_count=len(gallery) * 5,
        )
        try:
            config = run_config
            run_id = (run_dir / "wandb_run_id.txt").read_text(encoding="utf-8").strip()
            tracker = WandbTracker(run_dir=run_dir, config=config, run_id=run_id, resume=True)
            test_payload: dict[str, Any] = {
                "global_step": 200000,
                "test/bsd68/sigma15/psnr": summary["denoise/sigma15"]["psnr"],
                "test/bsd68/sigma15/ssim": summary["denoise/sigma15"]["ssim"],
                "test/bsd68/sigma25/psnr": summary["denoise/sigma25"]["psnr"],
                "test/bsd68/sigma25/ssim": summary["denoise/sigma25"]["ssim"],
                "test/bsd68/sigma50/psnr": summary["denoise/sigma50"]["psnr"],
                "test/bsd68/sigma50/ssim": summary["denoise/sigma50"]["ssim"],
                "test/bsd68/mean/psnr": summary["denoise/mean"]["psnr"],
                "test/bsd68/mean/ssim": summary["denoise/mean"]["ssim"],
                "test/rain100l/psnr": summary["derain"]["psnr"],
                "test/rain100l/ssim": summary["derain"]["ssim"],
                "test/sots_outdoor/psnr": summary["dehaze"]["psnr"],
                "test/sots_outdoor/ssim": summary["dehaze"]["ssim"],
                "test/macro/psnr": summary["macro"]["psnr"],
                "test/macro/ssim": summary["macro"]["ssim"],
                "test/total_runtime_seconds": duration,
            }
            table = tracker.wandb.Table(
                columns=["dataset", "task", "sigma", "sample_id", "psnr", "ssim", "inference_seconds"],
                data=[[row[name] for name in ("dataset", "task", "sigma", "sample_id", "psnr", "ssim", "inference_seconds")] for row in rows],
            )
            test_payload["test/per_image_metrics"] = table
            tracker.log(test_payload)
            tracker.log_evaluation_artifact([
                test_dir / "metrics.json", test_dir / "metrics.csv",
                test_dir / "per_image_metrics.csv", test_dir / "gallery.json",
                test_dir / "gallery_selection.json", test_dir / "compliance_report.md",
            ], test_dir / "artifact_gallery")
            tracker.finish(0)
            _write_json(run_dir / "wandb_state.json", {"status": "completed_with_test", "run_id": run_id})
        except Exception as wandb_error:
            from .runtime import append_jsonl
            append_jsonl(run_dir / "logs" / "wandb_errors.jsonl", {
                "operation": "formal_test_finalize", "error_type": type(wandb_error).__name__,
                "error": str(wandb_error),
            })
        _write_json(test_dir / "state.json", {"status": "completed", "completed": len(rows), "total": len(dataset)})
        return {**result_metadata, "metrics": summary}
    except Exception as error:
        _write_json(test_dir / "state.json", {
            "status": "failed", "completed": len(rows), "total": 804,
            "error_type": type(error).__name__, "error": str(error),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a completed AIO3-v1 formal DACG-IR run")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest-dir")
    parser.add_argument("--data-root")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device")
    args = parser.parse_args()
    result = evaluate(
        args.checkpoint, manifest_dir=args.manifest_dir, data_root=args.data_root,
        num_workers=args.num_workers, device_name=args.device,
    )
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
