"""Frozen constants for full CDD-11 DACG training without OOF or holdout splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PROTOCOL_NAME = "cdd11-dacg-full-v1"
PROTOCOL_VERSION = 1
SEED = 3407
NUM_TRAIN_SCENES = 1183
NUM_TEST_SCENES = 200
DEGRADATIONS = (
    "low", "haze", "rain", "snow",
    "low_haze", "low_rain", "low_snow", "haze_rain", "haze_snow",
    "low_haze_rain", "low_haze_snow",
)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def protocol_metadata() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "seed": SEED,
        "training_partition": "official train",
        "training_scenes": NUM_TRAIN_SCENES,
        "training_pairs": NUM_TRAIN_SCENES * len(DEGRADATIONS),
        "validation_partition": None,
        "checkpoint_selection": "final_completed_training_checkpoint",
        "test_partition": "official test",
        "test_scenes": NUM_TEST_SCENES,
        "test_pairs": NUM_TEST_SCENES * len(DEGRADATIONS),
        "degradations": list(DEGRADATIONS),
        "oof": False,
    }


def write_protocol(path: Path) -> None:
    payload = protocol_metadata()
    payload["sha256"] = sha256_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
