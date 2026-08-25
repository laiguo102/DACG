"""Stable scene and mask-pool split helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

SPLITS = ("train", "val", "test")


def canonical_split(name: str) -> str:
    value = str(name).lower()
    if value == "validation":
        value = "val"
    if value not in SPLITS:
        raise ValueError(f"Unsupported split: {name}")
    return value


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{int(seed)}:{value}".encode("utf-8")).hexdigest()


def deterministic_scene_split(
    scene_ids: Iterable[str],
    *,
    seed: int,
    train_ratio: float = 0.80,
    val_ratio: float = 0.05,
    test_ratio: float = 0.15,
) -> dict[str, list[str]]:
    scenes = sorted(set(str(scene) for scene in scene_ids))
    if not scenes:
        raise ValueError("Cannot split an empty scene list")
    ratios = (float(train_ratio), float(val_ratio), float(test_ratio))
    if any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("Scene split ratios must be non-negative and sum to 1")

    ordered = sorted(scenes, key=lambda scene: _stable_key(seed, f"scene:{scene}"))
    raw = [len(ordered) * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remaining = len(ordered) - sum(counts)
    fractional_order = sorted(
        range(3), key=lambda index: (raw[index] - counts[index], -index), reverse=True
    )
    for index in fractional_order[:remaining]:
        counts[index] += 1

    # Keep tiny fixtures useful while preserving the requested order.  A
    # one-scene smoke fixture belongs to train; larger fixtures get one scene
    # in every non-zero split when possible.
    if len(ordered) >= 3:
        for index, ratio in enumerate(ratios):
            if ratio <= 0 or counts[index] > 0:
                continue
            donor = max(
                (candidate for candidate in range(3) if counts[candidate] > 1),
                key=lambda candidate: counts[candidate],
                default=None,
            )
            if donor is not None:
                counts[donor] -= 1
                counts[index] += 1

    train_end = counts[0]
    val_end = train_end + counts[1]
    result = {
        "train": ordered[:train_end],
        "val": ordered[train_end:val_end],
        "test": ordered[val_end:],
    }
    validate_no_scene_leakage(result)
    return result


def load_split_file(path: str | Path) -> dict[str, list[str]]:
    split_path = Path(path).expanduser().resolve()
    value = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Split file must contain an object: {split_path}")
    result = {split: [str(item) for item in value.get(split, [])] for split in SPLITS}
    validate_no_scene_leakage(result)
    return result


def validate_no_scene_leakage(splits: dict[str, Iterable[str]]) -> None:
    normalized = {split: set(str(item) for item in splits.get(split, [])) for split in SPLITS}
    conflicts = []
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = sorted(normalized[left] & normalized[right])
            if overlap:
                conflicts.append(f"{left}/{right}: {overlap[:5]}")
    if conflicts:
        raise RuntimeError("Scene leakage across splits: " + "; ".join(conflicts))


def mask_pool_directory(root: str | Path, split: str, separate: bool) -> Path:
    base = Path(root).expanduser().resolve()
    selected = base / canonical_split(split) if separate else base
    if not selected.is_dir():
        raise FileNotFoundError(f"Missing mask pool directory: {selected}")
    return selected


def validate_disjoint_mask_pools(
    roots: dict[str, str | Path],
) -> None:
    names = tuple(SPLITS)
    normalized = {
        split: {str(Path(path).expanduser().resolve()) for path in paths}
        for split, paths in roots.items()
    }
    conflicts = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = sorted(normalized.get(left, set()) & normalized.get(right, set()))
            if overlap:
                conflicts.append(f"{left}/{right}: {overlap[:5]}")
    if conflicts:
        raise RuntimeError("Mask leakage across splits: " + "; ".join(conflicts))
