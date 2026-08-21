"""Fixed CDD11 selective-degradation task definitions."""

from __future__ import annotations

import random
from dataclasses import dataclass

SEED = 42
NUM_SCENES = 1183
NUM_TRAIN_SCENES = 1065
NUM_VALIDATION_SCENES = 118

PROMPT_NAMES = {
    "low": "low light",
    "haze": "haze",
    "rain": "rain",
    "snow": "snow",
}

PAIR_FOLDERS = {
    1: "low_haze",
    2: "low_rain",
    3: "low_snow",
    4: "haze_rain",
    5: "haze_snow",
}


@dataclass(frozen=True)
class DirectedTask:
    pair_id: int
    pair: str
    remove: str
    preserve: str
    prompt: str


def selected_pairs(pair_ids: list[int] | tuple[int, ...]) -> list[tuple[int, str]]:
    if not pair_ids:
        raise ValueError("At least one degradation pair must be selected")
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("Duplicate degradation pair IDs are not allowed")
    unknown = sorted(set(pair_ids) - set(PAIR_FOLDERS))
    if unknown:
        raise ValueError(f"Unknown degradation pair IDs: {unknown}")
    return [(pair_id, PAIR_FOLDERS[pair_id]) for pair_id in pair_ids]


def directed_tasks(pair_id: int) -> tuple[DirectedTask, DirectedTask]:
    pair = PAIR_FOLDERS[pair_id]
    first, second = pair.split("_")
    return tuple(
        DirectedTask(
            pair_id=pair_id,
            pair=pair,
            remove=remove,
            preserve=preserve,
            prompt=f"remove {PROMPT_NAMES[remove]}, preserve {PROMPT_NAMES[preserve]}",
        )
        for remove, preserve in ((first, second), (second, first))
    )


def make_scene_split(scene_ids: list[str]) -> dict[str, list[str]]:
    if len(scene_ids) != NUM_SCENES or len(set(scene_ids)) != NUM_SCENES:
        raise ValueError(f"CDD11 train must contain {NUM_SCENES} unique scenes")
    shuffled = sorted(scene_ids)
    random.Random(SEED).shuffle(shuffled)
    return {
        "train": shuffled[:NUM_TRAIN_SCENES],
        "validation": shuffled[NUM_TRAIN_SCENES:],
    }


def expected_counts(pair_ids: list[int] | tuple[int, ...]) -> dict[str, int]:
    pair_count = len(selected_pairs(pair_ids))
    return {
        "coarse": NUM_SCENES * pair_count,
        "train": NUM_TRAIN_SCENES * pair_count * 2,
        "validation": NUM_VALIDATION_SCENES * pair_count * 2,
    }
