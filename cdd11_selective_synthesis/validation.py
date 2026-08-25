"""Artifact, manifest, leakage, and distribution validation."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

import cv2

from .manifest import read_jsonl
from .params import FamilyParams
from .states import VALID_STATE_TUPLES, state_name
from .split import validate_no_scene_leakage


def _expected_image_paths(directory: Path) -> set[Path]:
    return {directory / f"{state_name(state)}.png" for state in VALID_STATE_TUPLES}


def is_complete_family(directory: str | Path) -> bool:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir() or not (root / "meta.json").is_file():
        return False
    return all(path.is_file() for path in _expected_image_paths(root))


def load_family_meta(directory: str | Path) -> dict:
    root = Path(directory).expanduser().resolve()
    return json.loads((root / "meta.json").read_text(encoding="utf-8"))


def validate_family(directory: str | Path) -> dict:
    root = Path(directory).expanduser().resolve()
    if not is_complete_family(root):
        missing = sorted(
            str(path.name)
            for path in _expected_image_paths(root) | {root / "meta.json"}
            if not path.is_file()
        )
        raise RuntimeError(f"Incomplete family {root}; missing {missing}")
    metadata = load_family_meta(root)
    for path in _expected_image_paths(root):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Unable to decode generated image: {path}")
    params = FamilyParams.from_dict(metadata)
    if params.scene_id != root.parent.name:
        raise RuntimeError(
            f"Family metadata scene mismatch: directory={root.parent.name}, "
            f"metadata={params.scene_id}"
        )
    try:
        directory_family_id = int(root.name.removeprefix("family_"))
    except ValueError as error:
        raise RuntimeError(f"Invalid family directory name: {root.name}") from error
    if int(params.family_id) != directory_family_id:
        raise RuntimeError(
            f"Family metadata id mismatch: directory={directory_family_id}, "
            f"metadata={params.family_id}"
        )
    return metadata


def validate_manifest(records: Iterable[dict], expected_families: int | None = None) -> None:
    rows = list(records)
    by_family: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        key = (str(row["scene_id"]), int(row["family_id"]))
        by_family[key].append(row)
        input_state = str(row["input_state"])
        target_state = str(row["target_state"])
        remove = str(row["remove"])
        if remove not in input_state.split("_"):
            raise RuntimeError(f"Manifest remove value is not in input state: {row}")
        values = [value for value in input_state.split("_") if value and value != remove]
        expected_target = "_".join(values) if values else "clean"
        if target_state != expected_target:
            raise RuntimeError(f"Manifest target mismatch: {row}")
        if not Path(row["input_path"]).is_file() or not Path(row["target_path"]).is_file():
            raise RuntimeError(f"Manifest points to a missing image: {row}")
    if expected_families is not None and len(by_family) != expected_families:
        raise RuntimeError(
            f"Expected {expected_families} families in manifest, got {len(by_family)}"
        )
    invalid = {key: len(value) for key, value in by_family.items() if len(value) != 20}
    if invalid:
        raise RuntimeError(f"Each family must have 20 selective pairs: {invalid}")


def validate_split_manifests(manifest_paths: dict[str, str | Path]) -> None:
    scene_sets: dict[str, set[str]] = {}
    for split, path in manifest_paths.items():
        rows = read_jsonl(path)
        validate_manifest(rows)
        scene_sets[split] = {str(row["scene_id"]) for row in rows}
    validate_no_scene_leakage(scene_sets)


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
    }


def build_dataset_stats(metas: Iterable[dict], records: Iterable[dict]) -> dict:
    metadata = list(metas)
    rows = list(records)
    state_counts = Counter()
    for meta in metadata:
        state_counts.update(state_name(state) for state in VALID_STATE_TUPLES)
    remove_counts = Counter(str(row["remove"]) for row in rows)
    mask_usage = {
        "rain": dict(Counter(str(meta["rain"]["mask"]) for meta in metadata)),
        "snow": dict(Counter(str(meta["snow"]["mask"]) for meta in metadata)),
    }
    parameter_values = {
        "gamma": [float(meta["low"]["gamma"]) for meta in metadata],
        "sigma": [float(meta["low"]["noise_sigma"]) for meta in metadata],
        "beta": [float(meta["haze"]["beta"]) for meta in metadata],
        "atmospheric_light": [
            float(meta["haze"]["atmospheric_light"]) for meta in metadata
        ],
    }
    return {
        "scenes": len({str(meta["scene_id"]) for meta in metadata}),
        "families": len(metadata),
        "images": len(metadata) * len(VALID_STATE_TUPLES),
        "selective_pairs": len(rows),
        "state_counts": dict(state_counts),
        "remove_counts": dict(remove_counts),
        "mask_usage": mask_usage,
        "parameters": {name: _summary(values) for name, values in parameter_values.items()},
    }


def build_distribution_report(metas: Iterable[dict]) -> dict:
    metadata = list(metas)
    report: dict = {
        "families": len(metadata),
        "degradations": {},
        "all_reuse_checks_passed": True,
    }
    for degradation in ("low", "haze", "rain", "snow"):
        states = [
            state_name(state)
            for state in VALID_STATE_TUPLES
            if degradation in state
        ]
        fields = {
            "low": ("gamma", "noise_sigma", "noise_seed"),
            "haze": ("beta", "atmospheric_light"),
            "rain": ("mask",),
            "snow": ("mask",),
        }[degradation]
        checks = {}
        for field in fields:
            values = [
                metadata_for_state[degradation][field]
                for metadata_for_state in metadata
            ]
            by_state = {state: list(values) for state in states}
            baseline = list(values)
            checks[field] = {
                "states": states,
                "by_state": by_state,
                "exactly_equal": all(values_for_state == baseline for values_for_state in by_state.values()),
            }
            report["all_reuse_checks_passed"] &= checks[field]["exactly_equal"]
        report["degradations"][degradation] = checks
    return report


def write_json(path: str | Path, value: dict) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
