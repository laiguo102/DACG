"""Selective-removal manifest generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .states import VALID_STATE_TUPLES, state_name


def family_records(
    *,
    split: str,
    scene_id: str,
    family_id: int,
    family_directory: str | Path,
) -> list[dict]:
    root = Path(family_directory).expanduser().resolve()
    records: list[dict] = []
    for state in VALID_STATE_TUPLES[1:]:
        input_state = state_name(state)
        input_path = root / f"{input_state}.png"
        for remove in state:
            target_state_tuple = tuple(value for value in state if value != remove)
            target_state = state_name(target_state_tuple)
            target_path = root / f"{target_state}.png"
            records.append(
                {
                    "id": (
                        f"{split}/{scene_id}/family_{int(family_id):03d}/"
                        f"{input_state}/remove-{remove}"
                    ),
                    "split": split,
                    "scene_id": str(scene_id),
                    "family_id": int(family_id),
                    "input_state": input_state,
                    "remove": remove,
                    "target_state": target_state,
                    "input_path": str(input_path),
                    "target_path": str(target_path),
                    "is_noop": False,
                }
            )
    if len(records) != 20:
        raise AssertionError(f"Expected 20 selective pairs, got {len(records)}")
    return records


def write_jsonl(path: str | Path, records: Iterable[dict]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return destination


def read_jsonl(path: str | Path) -> list[dict]:
    source = Path(path).expanduser().resolve()
    return [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
