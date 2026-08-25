"""Fixed CDD11 selective-degradation task definitions."""

from __future__ import annotations

import random
from dataclasses import dataclass

SEED = 42
NUM_SCENES = 1183
NUM_TRAIN_SCENES = 1065
NUM_VALIDATION_SCENES = 118
NUM_TEST_SCENES = 200

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

TRIPLE_FOLDERS = {
    1: "low_haze_rain",
    2: "low_haze_snow",
}


@dataclass(frozen=True)
class DirectedTask:
    pair_id: int
    pair: str
    remove: str
    preserve: str
    prompt: str


@dataclass(frozen=True)
class TripleTask:
    triple_id: int
    triple: str
    remove: tuple[str, str]
    preserve: str
    prompt: str


@dataclass(frozen=True)
class TripleRemoveOneTask:
    triple_id: int
    triple: str
    remove: str
    preserve: tuple[str, str]
    target: str
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


def selected_triples(
    triple_ids: list[int] | tuple[int, ...],
) -> list[tuple[int, str]]:
    if not triple_ids:
        raise ValueError("At least one triple-degradation combination must be selected")
    if len(set(triple_ids)) != len(triple_ids):
        raise ValueError("Duplicate triple-degradation IDs are not allowed")
    unknown = sorted(set(triple_ids) - set(TRIPLE_FOLDERS))
    if unknown:
        raise ValueError(f"Unknown triple-degradation IDs: {unknown}")
    return [(triple_id, TRIPLE_FOLDERS[triple_id]) for triple_id in triple_ids]


def triple_tasks(
    triple_id: int, prompt_template: str = "preserve-first"
) -> tuple[TripleTask, TripleTask, TripleTask]:
    triple = TRIPLE_FOLDERS[triple_id]
    degradations = tuple(triple.split("_"))
    if prompt_template not in ("preserve-first", "remove-first"):
        raise ValueError(f"Unknown triple prompt template: {prompt_template}")
    tasks = []
    for preserve in degradations:
        remove = tuple(value for value in degradations if value != preserve)
        remove_text = " and ".join(PROMPT_NAMES[value] for value in remove)
        preserve_text = PROMPT_NAMES[preserve]
        if prompt_template == "preserve-first":
            prompt = f"preserve {preserve_text}, remove {remove_text}"
        else:
            prompt = f"remove {remove_text}, preserve {preserve_text}"
        tasks.append(TripleTask(triple_id, triple, remove, preserve, prompt))
    return tuple(tasks)


def triple_remove_one_tasks(
    triple_id: int, prompt_template: str = "remove-first"
) -> tuple[TripleRemoveOneTask, TripleRemoveOneTask, TripleRemoveOneTask]:
    """Remove one degradation and preserve the other two from a CDD11 triple."""

    triple = TRIPLE_FOLDERS[triple_id]
    degradations = tuple(triple.split("_"))
    if prompt_template not in ("preserve-first", "remove-first"):
        raise ValueError(f"Unknown triple prompt template: {prompt_template}")
    tasks = []
    for remove in degradations:
        preserve = tuple(value for value in degradations if value != remove)
        target = "_".join(preserve)
        if target not in PAIR_FOLDERS.values():
            raise ValueError(f"CDD11 has no double-degradation target folder: {target}")
        remove_text = PROMPT_NAMES[remove]
        preserve_text = " and ".join(PROMPT_NAMES[value] for value in preserve)
        if prompt_template == "preserve-first":
            prompt = f"preserve {preserve_text}, remove {remove_text}"
        else:
            prompt = f"remove {remove_text}, preserve {preserve_text}"
        tasks.append(
            TripleRemoveOneTask(
                triple_id,
                triple,
                remove,
                preserve,
                target,
                prompt,
            )
        )
    return tuple(tasks)


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


def expected_test_count(pair_ids: list[int] | tuple[int, ...]) -> int:
    """Return the number of directed samples in the official CDD11 test split."""

    return NUM_TEST_SCENES * len(selected_pairs(pair_ids)) * 2


def expected_triple_test_count(triple_ids: list[int] | tuple[int, ...]) -> int:
    """Return the number of selective tasks in the CDD11 triple OOD test."""

    return NUM_TEST_SCENES * len(selected_triples(triple_ids)) * 3
