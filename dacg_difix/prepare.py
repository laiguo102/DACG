"""Create Difix3D manifests and native-resolution DACG coarse images."""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from cdd11_full.runtime import file_sha256


SPLITS = ("train", "val", "test")


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
                degraded_path = Path(str(record["input"]))
                target_path = Path(str(record["target"]))
                coarse_path = _coarse_path(output_dir, record, split).resolve()
                coarse = restore_native(
                    model,
                    _load_rgb(degraded_path),
                    device,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                )
                _save_rgb(coarse, coarse_path)
                arity = int(record.get("arity", record.get("metadata", {}).get(
                    "arity", str(record["degradation"]).count("_") + 1
                )))
                rows.append({
                    "coarse": str(coarse_path),
                    "degraded": str(degraded_path.resolve()),
                    "target": str(target_path.resolve()),
                    "degradation": str(record["degradation"]),
                    "split": split,
                    "id": str(record["id"]),
                    "scene_id": str(record["scene_id"]),
                    "arity": arity,
                    "dacg_checkpoint_sha256": checkpoint_sha256,
                })
            with prepared_path.open("w", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
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
