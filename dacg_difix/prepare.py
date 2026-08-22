"""Create Difix3D manifests and native-resolution DACG coarse images."""

from __future__ import annotations

import hashlib
import json
import random
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from cdd11_full.protocol import DEGRADATIONS
from cdd11_full.runtime import file_sha256


SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
FOLDER_SPLIT_SEED = 42
NUM_TRAIN_SCENES = 1183
NUM_OPTIMIZATION_SCENES = 1065
NUM_VALIDATION_SCENES = 118
NUM_TEST_SCENES = 200


def _load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div_(255.0).unsqueeze(0)


def _save_rgb(value: torch.Tensor, path: Path) -> None:
    array = value.detach().float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(array * 255.0).astype(np.uint8), "RGB").save(path)


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _forward(model: torch.nn.Module, value: torch.Tensor, device: torch.device) -> torch.Tensor:
    with _autocast(device):
        return model(value).float()


def restore_native(
    model: torch.nn.Module,
    degraded: torch.Tensor,
    device: torch.device,
    tile_size: int | None = None,
    tile_overlap: int = 128,
) -> torch.Tensor:
    """Restore an image at native resolution, optionally using overlapping tiles."""

    degraded = degraded.to(device)
    if tile_size is None:
        return _forward(model, degraded, device)

    stride = tile_size - tile_overlap
    if stride <= 0:
        raise ValueError("tile_size must be greater than tile_overlap")
    height, width = degraded.shape[-2:]

    def starts(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        values = list(range(0, length - tile_size + 1, stride))
        if values[-1] != length - tile_size:
            values.append(length - tile_size)
        return values

    output = torch.zeros_like(degraded, dtype=torch.float32)
    weights = torch.zeros((1, 1, height, width), device=device, dtype=torch.float32)
    for top in starts(height):
        for left in starts(width):
            tile = degraded[..., top:min(top + tile_size, height), left:min(left + tile_size, width)]
            restored = _forward(model, tile, device)
            window = torch.outer(
                torch.hann_window(restored.shape[-2], periodic=False, device=device),
                torch.hann_window(restored.shape[-1], periodic=False, device=device),
            ).clamp_min(1e-3)[None, None]
            bottom, right = top + restored.shape[-2], left + restored.shape[-1]
            output[..., top:bottom, left:right] += restored * window
            weights[..., top:bottom, left:right] += window
    return output / weights


def _coarse_path(output_dir: Path, record: dict[str, Any], split: str) -> Path:
    sample_id = str(record["id"])
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(sample_id.replace("\\", "/")).stem)
    suffix = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:12]
    return output_dir / "coarse" / split / str(record["degradation"]) / f"{stem}_{suffix}.png"


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def index_images(directory: Path) -> dict[str, Path]:
    """Index one flat CDD-11 image directory by filename stem."""

    if not directory.is_dir():
        raise FileNotFoundError(directory)
    images = {
        path.stem: path.resolve()
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if not images:
        raise ValueError(f"no images found in {directory}")
    return images


def make_folder_scene_split(scene_ids: Iterable[str]) -> dict[str, list[str]]:
    """Reproduce the selective branch's seed-42 1065/118 scene split."""

    values = list(scene_ids)
    if len(values) != NUM_TRAIN_SCENES or len(set(values)) != NUM_TRAIN_SCENES:
        raise ValueError(
            f"CDD-11 train must contain {NUM_TRAIN_SCENES} unique scenes, got {len(values)}"
        )
    shuffled = sorted(values)
    random.Random(FOLDER_SPLIT_SEED).shuffle(shuffled)
    return {
        "train": shuffled[:NUM_OPTIMIZATION_SCENES],
        "validation": shuffled[NUM_OPTIMIZATION_SCENES:],
    }


def _coarse_directory(coarse_root: Path, split: str, degradation: str) -> Path:
    candidates = (
        coarse_root / split / "background" / degradation,
        coarse_root / "background" / split / degradation,
        coarse_root / "background" / degradation,
        coarse_root / split / degradation,
        coarse_root / degradation,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "missing DACG background directory; expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def prepare_cdd11_folder_manifests(
    data_root: str | Path,
    coarse_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Auto-index CDD-11 like selective and keep the official test untouched."""

    data_root = Path(data_root).resolve()
    coarse_root = Path(coarse_root).resolve()
    prepared_root = Path(output_dir).resolve() / "prepared"
    manifest_root = prepared_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)

    train_clear = index_images(data_root / "train" / "clear")
    scene_splits = make_folder_scene_split(train_clear)
    test_clear = index_images(data_root / "test" / "clear")
    if len(test_clear) != NUM_TEST_SCENES:
        raise ValueError(
            f"CDD-11 official test must contain {NUM_TEST_SCENES} scenes, got {len(test_clear)}"
        )

    rows_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [], "validation": [], "test": [],
    }
    for physical_split, clear_images in (("train", train_clear), ("test", test_clear)):
        expected_scenes = set(clear_images)
        degraded_by_type: dict[str, dict[str, Path]] = {}
        coarse_by_type: dict[str, dict[str, Path]] = {}
        for degradation in DEGRADATIONS:
            degraded_dir = data_root / physical_split / degradation
            coarse_dir = _coarse_directory(coarse_root, physical_split, degradation)
            degraded = index_images(degraded_dir)
            coarse = index_images(coarse_dir)
            if set(degraded) != expected_scenes:
                raise ValueError(f"CDD-11 scene mismatch in {degraded_dir}")
            if expected_scenes - set(coarse):
                raise ValueError(f"CDD-11 background scene mismatch in {coarse_dir}")
            degraded_by_type[degradation] = degraded
            coarse_by_type[degradation] = coarse

        logical_splits = (
            (("train", scene_splits["train"]), ("validation", scene_splits["validation"]))
            if physical_split == "train"
            else (("test", sorted(test_clear)),)
        )
        for logical_split, scene_ids in logical_splits:
            for scene_id in scene_ids:
                for degradation in DEGRADATIONS:
                    rows_by_split[logical_split].append({
                        "coarse": str(coarse_by_type[degradation][scene_id]),
                        "degraded": str(degraded_by_type[degradation][scene_id]),
                        "target": str(clear_images[scene_id]),
                        "degradation": degradation,
                        "split": logical_split,
                        "id": f"{logical_split}/{degradation}/{scene_id}",
                        "scene_id": scene_id,
                        "arity": degradation.count("_") + 1,
                    })

    manifests: dict[str, str] = {}
    for split, rows in rows_by_split.items():
        path = manifest_root / f"{split}.jsonl"
        _write_prepared_manifest(path, rows)
        manifests[split] = str(path)
    metadata = {
        "protocol": "selective-style-folder-split",
        "seed": FOLDER_SPLIT_SEED,
        "data_root": str(data_root),
        "coarse_root": str(coarse_root),
        "train_scene_ids": scene_splits["train"],
        "validation_scene_ids": scene_splits["validation"],
        "test_scene_ids": sorted(test_clear),
        "counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "manifests": manifests,
    }
    (prepared_root / "split_and_preparation.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _existing_coarse_path(
    coarse_root: Path, record: dict[str, Any], split: str
) -> Path:
    """Resolve a precomputed background image using common CDD-11 layouts."""

    degradation = str(record["degradation"])
    input_path = Path(str(record["input"]))
    directories = (
        coarse_root / split / degradation,
        coarse_root / degradation,
        coarse_root / split,
        coarse_root,
    )
    names = (input_path.name,) + tuple(
        f"{input_path.stem}{suffix}" for suffix in IMAGE_SUFFIXES
    )
    checked: list[Path] = []
    for directory in directories:
        for name in dict.fromkeys(names):
            candidate = directory / name
            checked.append(candidate)
            if candidate.is_file():
                return candidate.resolve()
    preview = ", ".join(str(path) for path in checked[:4])
    raise FileNotFoundError(
        f"no existing DACG background for record {record['id']!r}; checked {preview}, ..."
    )


def _prepared_row(
    record: dict[str, Any], split: str, coarse_path: Path
) -> dict[str, Any]:
    degraded_path = Path(str(record["input"]))
    target_path = Path(str(record["target"]))
    arity = int(record.get("arity", record.get("metadata", {}).get(
        "arity", str(record["degradation"]).count("_") + 1
    )))
    return {
        "coarse": str(coarse_path.resolve()),
        "degraded": str(degraded_path.resolve()),
        "target": str(target_path.resolve()),
        "degradation": str(record["degradation"]),
        "split": split,
        "id": str(record["id"]),
        "scene_id": str(record["scene_id"]),
        "arity": arity,
    }


def _write_prepared_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_existing_cdd11_manifests(
    source_manifest_dir: str | Path,
    coarse_root: str | Path,
    output_dir: str | Path,
    *,
    splits: Iterable[str] = SPLITS,
) -> dict[str, Any]:
    """Build Difix manifests around DACG background images that already exist."""

    source_manifest_dir = Path(source_manifest_dir)
    coarse_root = Path(coarse_root).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in tuple(splits):
        records = _read_manifest(source_manifest_dir / f"{split}.jsonl")
        rows = [
            _prepared_row(record, split, _existing_coarse_path(coarse_root, record, split))
            for record in records
        ]
        _write_prepared_manifest(output_dir / f"{split}.jsonl", rows)
        counts[split] = len(rows)
    metadata = {
        "source_manifest_dir": str(source_manifest_dir.resolve()),
        "coarse_root": str(coarse_root),
        "coarse_source": "existing-background",
        "counts": counts,
    }
    (output_dir / "prepare_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def prepare_cdd11_manifests(
    source_manifest_dir: str | Path,
    output_dir: str | Path,
    model: torch.nn.Module,
    device: torch.device,
    checkpoint_path: str | Path,
    *,
    splits: Iterable[str] = SPLITS,
    tile_size: int | None = None,
    tile_overlap: int = 128,
) -> dict[str, Any]:
    """Run frozen DACG over CDD11 manifests and write prepared JSONL files."""

    source_manifest_dir = Path(source_manifest_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_sha256 = file_sha256(checkpoint_path)
    selected_splits = tuple(splits)
    model.eval()
    counts: dict[str, int] = {}

    with torch.inference_mode():
        for split in selected_splits:
            source_path = source_manifest_dir / f"{split}.jsonl"
            prepared_path = output_dir / f"{split}.jsonl"
            rows = []
            for record in _read_manifest(source_path):
                coarse_path = _coarse_path(output_dir, record, split).resolve()
                coarse = restore_native(
                    model,
                    _load_rgb(Path(str(record["input"]))),
                    device,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                )
                _save_rgb(coarse, coarse_path)
                row = _prepared_row(record, split, coarse_path)
                row["dacg_checkpoint_sha256"] = checkpoint_sha256
                rows.append(row)
            _write_prepared_manifest(prepared_path, rows)
            counts[split] = len(rows)

    metadata = {
        "source_manifest_dir": str(source_manifest_dir.resolve()),
        "dacg_checkpoint": str(checkpoint_path.resolve()),
        "dacg_checkpoint_sha256": checkpoint_sha256,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap if tile_size is not None else None,
        "counts": counts,
    }
    (output_dir / "prepare_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata
