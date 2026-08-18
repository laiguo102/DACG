"""Frozen CDD-11-v1 constants for DACG with the paper objective."""

from __future__ import annotations

import hashlib
from typing import Mapping

PROTOCOL_NAME = "cdd11-v1"
OBJECTIVE_VARIANT = "dacg-paper-rgb-fourier-l1"
SEED = 3407
DEGRADATIONS = (
    "low", "haze", "rain", "snow", "low_haze", "low_rain", "low_snow",
    "haze_rain", "haze_snow", "low_haze_rain", "low_haze_snow",
)
DEGRADATION_ARITY: Mapping[str, int] = {
    value: value.count("_") + 1 for value in DEGRADATIONS
}
ARITY_GROUPS = {
    name: tuple(value for value in DEGRADATIONS if DEGRADATION_ARITY[value] == arity)
    for name, arity in (("single", 1), ("double", 2), ("triple", 3))
}
RUN_PROFILES = {
    "smoke": {"max_steps": 100, "scalar_interval": 10, "validation_interval": 100},
    "pilot": {"max_steps": 5_000, "scalar_interval": 50, "validation_interval": 5_000},
    "formal": {"max_steps": 200_000, "scalar_interval": 50, "validation_interval": 5_000},
}


def deterministic_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)
