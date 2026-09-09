"""Evaluate trained or initialization-only selective Difix on an official test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .metrics import (
    STAGES,
    FullReferenceMetricSuite,
    flatten_sample_metrics,
    summarize_test_rows,
    summary_csv_rows,
)
from .protocol import (
    expected_test_count,
    expected_triple_test_count,
    selected_pairs,
    selected_triples,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)

    # Some distributed mounts expose the renamed directory entry before the
    # file data is visible, so an immediate reader can observe a zero-byte JSON
    # file even though the temporary file was fsynced. Sync the directory when
    # supported and verify the destination. A direct, fsynced rewrite is a
    # compatibility fallback for filesystems whose rename is not coherently
    # visible to the process that performed it.
    directory_fd = None
    try:
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        if directory_fd is not None:
            os.close(directory_fd)

    for delay_seconds in (0.0, 0.01, 0.05):
        if delay_seconds:
            time.sleep(delay_seconds)
        try:
            if path.read_bytes() == payload:
                return
        except OSError:
            pass

    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if path.read_bytes() != payload:
        raise OSError(f"JSON write verification failed for {path}")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _save_image(tensor: torch.Tensor, path: Path) -> None:
    value = tensor.detach().float().clamp(0, 1)
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError(f"Expected one image, got {tuple(value.shape)}")
        value = value[0]
    array = value.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array), "RGB").save(path)


def _safe_id(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace("\\", "__")


def _sample_seed(seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(
        f"selective-difix-test:{seed}:{sample_id}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_identity(path: Path) -> dict[str, str | int]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_time_ns": stat.st_mtime_ns,
    }


def _infer_pair_ids(checkpoint: Path, explicit: list[int] | None) -> list[int]:
    if explicit:
        selected_pairs(explicit)
        return explicit
    preparation = checkpoint.parent.parent / "prepared" / "split_and_preparation.json"
    if not preparation.is_file():
        raise ValueError(
            "Cannot infer training degradation pairs. Pass --degradation-pairs explicitly."
        )
    info = json.loads(preparation.read_text(encoding="utf-8"))
    pair_ids = [int(value["id"]) for value in info["degradation_pairs"]]
    selected_pairs(pair_ids)
    return pair_ids


def _verify_training_dataset(checkpoint: Path, dataset_format: str) -> None:
    """Reject using a CDD-trained checkpoint as the formal CCDD test model."""

    if dataset_format != "ccdd11":
        return
    preparation = checkpoint.parent.parent / "prepared" / "split_and_preparation.json"
    if not preparation.is_file():
        raise ValueError(
            "CCDD-11 evaluation requires the training split_and_preparation.json "
            f"next to the checkpoint: {preparation}"
        )
    info = json.loads(preparation.read_text(encoding="utf-8"))
    if info.get("dataset") != "CCDD-11" or info.get("target_kind") != (
        "native_selective_sub_data"
    ):
        raise ValueError(
            "The checkpoint run is not identified as CCDD-11 native selective training"
        )


def _autocast(device: torch.device, mixed_precision: str):
    if mixed_precision == "no" or device.type != "cuda":
        return nullcontext()
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _resolve_device(value: str) -> torch.device:
    """Return an explicitly indexed CUDA device for older PyTorch releases."""
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", 0)
    return device


def _load_partial_rows(records_dir: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not records_dir.is_dir():
        return rows
    for path in sorted(records_dir.rglob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        sample_id = str(row["sample_id"])
        if sample_id in rows:
            raise ValueError(f"Duplicate partial test row for {sample_id}")
        rows[sample_id] = row
    return rows


def _record_path(records_dir: Path, row: dict) -> Path:
    task = f"remove-{row['remove']}-preserve-{row['preserve']}"
    return records_dir / str(row["pair"]) / task / f"{row['scene_id']}.json"


def _state_payload(
    config: dict, status: str, processed: int, total: int, **extra
) -> dict:
    return {
        "status": status,
        "processed_samples": processed,
        "total_samples": total,
        "config": config,
        "updated_at_utc": _now(),
        **extra,
    }


def run(args: argparse.Namespace) -> None:
    from torch.utils.data import DataLoader, Subset
    from tqdm.auto import tqdm

    from .data import SelectiveDifixDataset
    from .model import SelectiveDifix, load_model_checkpoint
    from .prepare import (
        prepare_selective_test_manifest,
        prepare_selective_triple_test_manifest,
    )
    from .validation import comparison_image, stratified_indices

    model_source = getattr(args, "model_source", "trained")
    dataset_format = getattr(args, "dataset_format", "cdd11")
    triple_ids = getattr(args, "triple_combinations", None)
    triple_mode = triple_ids is not None
    if dataset_format == "ccdd11" and triple_mode:
        raise ValueError("CCDD-11 half_test adapter currently supports pair tasks only")
    if dataset_format == "ccdd11":
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
            raise ValueError(f"CCDD-11 official evaluation protocol mismatch: {details}")
    prompt_template = getattr(args, "triple_prompt_template", "preserve-first")
    triple_task_mode = getattr(args, "triple_task_mode", "preserve-one")
    if not triple_mode and triple_task_mode != "preserve-one":
        raise ValueError("--triple-task-mode requires --triple-combinations")
    checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else None
    if model_source == "trained":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for --model-source trained")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        _verify_training_dataset(checkpoint, dataset_format)
        if triple_mode:
            selection_ids = [triple_id for triple_id, _ in selected_triples(triple_ids)]
            task_prefix = (
                "" if triple_task_mode == "preserve-one" else f"{triple_task_mode}_"
            )
            default_output_dir = (
                checkpoint.parent.parent
                / f"test_triple_{task_prefix}{prompt_template}_{checkpoint.stem}"
            )
        else:
            selection_ids = _infer_pair_ids(checkpoint, args.degradation_pairs)
            default_output_dir = checkpoint.parent.parent / f"test_{checkpoint.stem}"
    else:
        if checkpoint is not None:
            raise ValueError(
                "Do not pass --checkpoint with --model-source initialization"
            )
        if triple_mode:
            selection_ids = [triple_id for triple_id, _ in selected_triples(triple_ids)]
        else:
            if not args.degradation_pairs:
                raise ValueError(
                    "--degradation-pairs is required for initialization evaluation"
                )
            selection_ids = [
                pair_id for pair_id, _ in selected_pairs(args.degradation_pairs)
            ]
        if args.output_dir is None:
            raise ValueError("--output-dir is required for initialization evaluation")
        default_output_dir = args.output_dir
    output_dir = (args.output_dir or default_output_dir).resolve()
    if (output_dir / "metrics.json").exists():
        raise RuntimeError(
            f"Completed metrics already exist in {output_dir}; choose a new --output-dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    if triple_mode:
        prepared = prepare_selective_triple_test_manifest(
            data_root=args.data_root,
            coarse_root=args.test_coarse_root,
            output_dir=output_dir,
            triple_ids=selection_ids,
            prompt_template=prompt_template,
            task_mode=triple_task_mode,
            require_coarse_metadata=not args.allow_unverified_test_coarse,
        )
        total = expected_triple_test_count(selection_ids)
    else:
        if dataset_format == "ccdd11":
            from .ccdd_prepare import prepare_ccdd11_selective_test_manifest

            prepare_test_manifest = prepare_ccdd11_selective_test_manifest
        else:
            prepare_test_manifest = prepare_selective_test_manifest
        prepared = prepare_test_manifest(
            data_root=args.data_root,
            coarse_root=args.test_coarse_root,
            output_dir=output_dir,
            pair_ids=selection_ids,
            require_coarse_metadata=not args.allow_unverified_test_coarse,
        )
        total = expected_test_count(selection_ids)
    if prepared["test_samples"] != total:
        raise RuntimeError(
            f"Expected {total} directed test samples, got {prepared['test_samples']}"
        )
    manifest_records = {
        str(record["id"]): record
        for record in (
            json.loads(line)
            for line in Path(prepared["manifest"])
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }

    config = {
        "dataset_format": dataset_format,
        "data_root": str(args.data_root.resolve()),
        "test_coarse_root": str(args.test_coarse_root.resolve()),
        "test_manifest_sha256": _file_sha256(prepared["manifest"]),
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
        "save_predictions": args.save_predictions,
        "coarse_metadata_verified": prepared["coarse_metadata_verified"],
    }
    if triple_mode:
        config.update(
            {
                "task_family": "triple",
                "triple_combinations": selection_ids,
                "triple_prompt_template": prompt_template,
            }
        )
        if triple_task_mode == "remove-one":
            config["triple_task_mode"] = triple_task_mode
    else:
        config["degradation_pairs"] = selection_ids
    if model_source == "trained":
        config["checkpoint"] = _checkpoint_identity(checkpoint)
    else:
        config.update(
            {
                "model_source": "initialization",
                "initialization": "stabilityai/sd-turbo plus selective Difix adapters",
                "initialization_seed": args.seed,
            }
        )
    state_path = output_dir / "state.json"
    records_dir = output_dir / "records"
    partial_rows = _load_partial_rows(records_dir)
    if state_path.is_file():
        previous_state = json.loads(state_path.read_text(encoding="utf-8"))
        if previous_state.get("config") != config:
            raise RuntimeError(
                "Existing partial test state uses a different configuration; "
                "choose a new --output-dir"
            )
    elif partial_rows:
        raise RuntimeError(
            f"Partial records exist without state metadata in {output_dir}"
        )
    _atomic_json(
        state_path,
        _state_payload(config, "evaluating", len(partial_rows), total),
    )

    device = _resolve_device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.cuda.set_device(device.index)
    elif args.mixed_precision != "no":
        raise ValueError("CPU evaluation requires --mixed-precision no")

    # Training calls accelerate.set_seed(args.seed) before constructing this model.
    # Reusing the same seed makes the initialization baseline the exact step-0 model.
    torch.manual_seed(args.seed)
    model = SelectiveDifix(lora_rank_vae=args.lora_rank_vae, timestep=args.timestep).to(
        device
    )
    if args.enable_xformers_memory_efficient_attention:
        model.unet.enable_xformers_memory_efficient_attention()
    if model_source == "trained":
        checkpoint_requirements = (
            {"expected_dataset": "CCDD-11", "expected_seed": 42}
            if dataset_format == "ccdd11"
            else {}
        )
        global_step = load_model_checkpoint(
            model, checkpoint, **checkpoint_requirements
        )
    else:
        global_step = 0
    model.set_eval()
    metric_suite = FullReferenceMetricSuite(device)

    dataset = SelectiveDifixDataset(
        prepared["manifest"], model.tokenizer, resolution=args.resolution
    )
    gallery_ids = set()
    if args.num_gallery_samples > 0:
        gallery_ids = {
            dataset.records[index]["id"]
            for index in stratified_indices(dataset.records, args.num_gallery_samples)
        }
    expected_ids = {record["id"] for record in dataset.records}
    unknown_partial = set(partial_rows) - expected_ids
    if unknown_partial:
        raise ValueError(
            f"Partial records do not belong to this test manifest: {sorted(unknown_partial)[:3]}"
        )
    pending_indices = [
        index
        for index, record in enumerate(dataset.records)
        if record["id"] not in partial_rows
    ]
    loader = DataLoader(
        Subset(dataset, pending_indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    progress = tqdm(
        total=total,
        initial=len(partial_rows),
        desc="CDD11 selective Difix test",
    )
    for batch in loader:
        sample_id = batch["sample_id"][0]
        source = batch["conditioning_pixel_values"].to(device, non_blocking=True)
        target = batch["output_pixel_values"].to(device, non_blocking=True)
        ground_truth = batch["ground_truth_pixel_values"].to(device, non_blocking=True)
        devices = []
        if device.type == "cuda":
            devices = [
                (
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                )
            ]
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        inference_seed = _sample_seed(args.seed, sample_id)
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(inference_seed)
            with torch.inference_mode(), _autocast(device, args.mixed_precision):
                prediction = model(source, prompt_tokens=batch["input_ids"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

        prediction_01 = prediction.float().add(1).mul(0.5).clamp(0, 1)
        candidates = {
            "degraded": source[:, 1].float().add(1).mul(0.5).clamp(0, 1),
            "coarse": source[:, 0].float().add(1).mul(0.5).clamp(0, 1),
            "final": prediction_01,
        }
        target_01 = target.float().add(1).mul(0.5).clamp(0, 1)
        stage_metrics = metric_suite(candidates, target_01)
        pair = batch["pair"][0]
        remove = batch["remove"][0]
        preserve = batch["preserve"][0]
        scene_id = batch["scene_id"][0]
        prediction_path = ""
        if args.save_predictions:
            destination = (
                output_dir
                / "predictions"
                / pair
                / f"remove-{remove}-preserve-{preserve}"
                / f"{scene_id}.png"
            )
            _save_image(prediction_01, destination)
            prediction_path = str(destination)
        gallery_path = ""
        if sample_id in gallery_ids:
            gallery = comparison_image(source, target, prediction, ground_truth)
            destination = output_dir / "gallery" / pair / f"{_safe_id(sample_id)}.png"
            _save_image(gallery, destination)
            gallery_path = str(destination)

        row = {
            "sample_id": sample_id,
            "scene_id": scene_id,
            "pair_id": int(batch["pair_id"].item()),
            "pair": pair,
            "remove": remove,
            "preserve": preserve,
            "prompt": batch["prompt"][0],
            "coarse_path": manifest_records[sample_id]["image"],
            "degraded_path": manifest_records[sample_id]["ref_image"],
            "target_path": manifest_records[sample_id]["target_image"],
            "inference_seed": inference_seed,
            **flatten_sample_metrics(stage_metrics),
            "inference_time_seconds": elapsed,
            "prediction_path": prediction_path,
            "gallery_path": gallery_path,
        }
        if triple_mode:
            row.update(
                {
                    "task_family": "triple",
                    "prompt_template": prompt_template,
                }
            )
            if triple_task_mode == "remove-one":
                row["triple_task_mode"] = triple_task_mode
        _atomic_json(_record_path(records_dir, row), row)
        partial_rows[sample_id] = row
        progress.update(1)
        if len(partial_rows) % 25 == 0 or len(partial_rows) == total:
            _atomic_json(
                state_path,
                _state_payload(config, "evaluating", len(partial_rows), total),
            )
    progress.close()

    if set(partial_rows) != expected_ids:
        missing = sorted(expected_ids - set(partial_rows))
        raise RuntimeError(f"Test is incomplete; missing {len(missing)} samples")
    rows = [partial_rows[record["id"]] for record in dataset.records]
    summary = summarize_test_rows(
        rows, combination_group="triple" if triple_mode else "pair"
    )
    metadata = {
        "protocol": (
            "cdd11-selective-difix-triple-remove-one-ood-test-v1"
            if triple_mode and triple_task_mode == "remove-one"
            else (
                "cdd11-selective-difix-triple-ood-test-v1"
                if triple_mode
                else (
                    "ccdd11-old-style-selective-difix-half-test-v1"
                    if dataset_format == "ccdd11"
                    else "cdd11-selective-difix-test-v1"
                )
            )
        ),
        "task_family": "triple" if triple_mode else "pair",
        "dataset": "CCDD-11" if dataset_format == "ccdd11" else "CDD-11",
        "model_source": model_source,
        "global_step": global_step,
        "created_at_utc": _now(),
        "config": config,
        "software": {
            name: _package_version(name)
            for name in ("torch", "torchvision", "diffusers", "lpips", "piq")
        },
        "primary_reference": (
            "double-degradation target containing both preserved degradations"
            if triple_mode and triple_task_mode == "remove-one"
            else "single-degradation target to be preserved"
        ),
        "baselines": {
            "degraded": (
                "original triple-degradation input"
                if triple_mode
                else "original double-degradation input"
            ),
            "coarse": "DACG preliminary restoration",
            "final": (
                "trained selective Difix output"
                if model_source == "trained"
                else "step-0 selective Difix initialization output"
            ),
        },
        "metrics": {
            "psnr": {
                "better": "higher",
                "implementation": "CDD11 RGB PSNR on [0,1]",
            },
            "ssim": {
                "better": "higher",
                "implementation": "CDD11 RGB SSIM, 11x11 Gaussian window",
            },
            "lpips_vgg": {
                "better": "lower",
                "implementation": "lpips.LPIPS(net='vgg') on [-1,1]",
            },
            "dists": {
                "better": "lower",
                "implementation": "piq.DISTS(reduction='none') on [0,1]",
            },
        },
        "improvement_definition": (
            "positive means final Difix is better than DACG coarse; final-coarse for "
            "PSNR/SSIM and coarse-final for LPIPS-VGG/DISTS"
        ),
        "inference_time_scope": "Difix model forward only; excludes metrics and file I/O",
        "stages": list(STAGES),
    }
    if triple_mode and triple_task_mode == "remove-one":
        metadata["triple_task_mode"] = triple_task_mode
    per_image_fields = list(rows[0])
    _write_csv(output_dir / "per_image_metrics.csv", rows, per_image_fields)
    compact_rows = summary_csv_rows(summary)
    summary_fields = list(compact_rows[0])
    if any("tasks" in row for row in compact_rows):
        summary_fields.append("tasks")
    summary_fields = list(dict.fromkeys(summary_fields))
    _write_csv(output_dir / "summary.csv", compact_rows, summary_fields)
    macro = summary["overall"]["macro"]
    _atomic_json(
        state_path,
        _state_payload(
            config,
            "completed",
            total,
            total,
            global_step=global_step,
            macro_final_psnr=macro["final_psnr"],
            macro_final_ssim=macro["final_ssim"],
            macro_final_lpips_vgg=macro["final_lpips_vgg"],
            macro_final_dists=macro["final_dists"],
        ),
    )
    _atomic_json(
        output_dir / "metrics.json",
        {"metadata": metadata, "summary": summary, "per_image_count": len(rows)},
    )
    print(
        f"Completed {total} {model_source} "
        f"{'triple ' + triple_task_mode if triple_mode else 'pair'} samples "
        f"at step {global_step}. "
        f"PSNR={macro['final_psnr']:.4f}, SSIM={macro['final_ssim']:.6f}, "
        f"LPIPS-VGG={macro['final_lpips_vgg']:.6f}, "
        f"DISTS={macro['final_dists']:.6f}. Results: {output_dir}",
        flush=True,
    )


def parser(default_dataset_format: str = "cdd11") -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--test-coarse-root", type=Path, required=True)
    value.add_argument(
        "--model-source", choices=("trained", "initialization"), default="trained"
    )
    value.add_argument("--checkpoint", type=Path)
    value.add_argument("--output-dir", type=Path)
    value.add_argument(
        "--dataset-format", choices=("cdd11", "ccdd11"),
        default=default_dataset_format,
    )
    selection = value.add_mutually_exclusive_group()
    selection.add_argument(
        "--degradation-pairs", nargs="+", type=int, choices=range(1, 6)
    )
    selection.add_argument("--triple-combinations", nargs="+", type=int, choices=(1, 2))
    value.add_argument(
        "--triple-prompt-template",
        choices=("preserve-first", "remove-first"),
        default="preserve-first",
    )
    value.add_argument(
        "--triple-task-mode",
        choices=("preserve-one", "remove-one"),
        default="preserve-one",
    )
    value.add_argument("--resolution", type=int, default=512)
    value.add_argument("--lora-rank-vae", type=int, default=4)
    value.add_argument("--timestep", type=int, default=199)
    value.add_argument("--workers", type=int, default=4)
    value.add_argument("--device", default="cuda")
    value.add_argument(
        "--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16"
    )
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--num-gallery-samples", type=int, default=20)
    value.add_argument(
        "--enable-xformers-memory-efficient-attention", action="store_true"
    )
    value.add_argument("--allow-unverified-test-coarse", action="store_true")
    value.add_argument(
        "--no-save-predictions", dest="save_predictions", action="store_false"
    )
    value.set_defaults(save_predictions=True)
    return value


def main(default_dataset_format: str = "cdd11") -> None:
    run(parser(default_dataset_format).parse_args())


if __name__ == "__main__":
    main()
