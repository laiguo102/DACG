"""Constants and split helpers for the frozen CDD-11 OOF protocol."""

from __future__ import annotations

import json
import hashlib
import random
from pathlib import Path

PROTOCOL_NAME = "cdd11-dacg-difix-5fold-oof-v1"
SEED = 42
NUM_TRAIN_SCENES = 1183
NUM_OOF_SCENES = 1065
NUM_VAL_SCENES = 118
NUM_FOLDS = 5
SCENES_PER_FOLD = 213
NUM_TEST_SCENES = 200

DEGRADATIONS = (
    "low", "haze", "rain", "snow",
    "low_haze", "low_rain", "low_snow", "haze_rain", "haze_snow",
    "low_haze_rain", "low_haze_snow",
)


def make_split(scene_ids: list[str], seed: int = SEED) -> dict[str, list[str]]:
    """Return the deterministic scene-level split defined by the protocol."""
    if seed != SEED:
        raise ValueError(f"This protocol fixes seed={SEED}; got {seed}")
    if len(scene_ids) != NUM_TRAIN_SCENES:
        raise ValueError(
            f"Expected {NUM_TRAIN_SCENES} official train scenes, got {len(scene_ids)}"
        )
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError("Duplicate train scene IDs were found")
    shuffled = sorted(scene_ids)
    random.Random(seed).shuffle(shuffled)
    oof = shuffled[:NUM_OOF_SCENES]
    result = {
        f"fold{i + 1}": oof[i * SCENES_PER_FOLD:(i + 1) * SCENES_PER_FOLD]
        for i in range(NUM_FOLDS)
    }
    result["val"] = shuffled[NUM_OOF_SCENES:]
    return result


def read_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate scene IDs in {path}")
    return ids


def load_splits(split_dir: Path) -> dict[str, list[str]]:
    result = {name: read_ids(split_dir / f"{name}.txt") for name in [*(f"fold{i}" for i in range(1, 6)), "val"]}
    sets = {name: set(ids) for name, ids in result.items()}
    names = list(sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise ValueError(f"Scene leakage between {left} and {right}: {sorted(overlap)[:5]}")
    if any(len(result[f"fold{i}"]) != SCENES_PER_FOLD for i in range(1, 6)):
        raise ValueError(f"Every fold must contain {SCENES_PER_FOLD} scenes")
    if len(result["val"]) != NUM_VAL_SCENES:
        raise ValueError(f"Validation must contain {NUM_VAL_SCENES} scenes")
    return result


def split_fingerprint(splits: dict[str, list[str]]) -> str:
    payload = json.dumps(splits, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_frozen_split(scene_ids: list[str], splits: dict[str, list[str]]) -> None:
    expected = make_split(scene_ids)
    if splits != expected:
        raise ValueError(
            "The split files are valid and disjoint, but do not match the frozen "
            "seed-42 split for this dataset. Do not regenerate or edit splits per experiment."
        )


def write_splits(split_dir: Path, splits: dict[str, list[str]]) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (split_dir / f"{name}.txt").write_text("".join(f"{x}\n" for x in ids), encoding="utf-8")
    info = {
        "dataset": "CDD-11",
        "protocol": PROTOCOL_NAME,
        "seed": SEED,
        "scene_level_split": True,
        "num_official_train_scenes": NUM_TRAIN_SCENES,
        "num_oof_scenes": NUM_OOF_SCENES,
        "num_validation_scenes": NUM_VAL_SCENES,
        "num_folds": NUM_FOLDS,
        "scenes_per_fold": SCENES_PER_FOLD,
        "degradations": list(DEGRADATIONS),
        "degradations_per_scene": len(DEGRADATIONS),
        "num_oof_training_pairs": NUM_OOF_SCENES * len(DEGRADATIONS),
        "num_official_test_scenes": NUM_TEST_SCENES,
        "num_official_test_pairs": NUM_TEST_SCENES * len(DEGRADATIONS),
    }
    (split_dir / "split_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
