"""Deterministic family parameter sampling and serialization."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LowParams:
    gamma: float
    noise_sigma: float
    noise_seed: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "gamma": float(self.gamma),
            "noise_sigma": float(self.noise_sigma),
            "noise_seed": int(self.noise_seed),
        }


@dataclass(frozen=True)
class HazeParams:
    beta: float
    atmospheric_light: float

    def to_dict(self) -> dict[str, float]:
        return {
            "beta": float(self.beta),
            "atmospheric_light": float(self.atmospheric_light),
        }


@dataclass(frozen=True)
class RainParams:
    mask_path: str

    def to_dict(self) -> dict[str, str]:
        return {"mask": self.mask_path}


@dataclass(frozen=True)
class SnowParams:
    mask_path: str

    def to_dict(self) -> dict[str, str]:
        return {"mask": self.mask_path}


@dataclass(frozen=True)
class FamilyParams:
    scene_id: str
    family_id: int
    family_seed: int
    low: LowParams
    haze: HazeParams
    rain: RainParams
    snow: SnowParams

    sampler_name: str = "sha256+numpy.default_rng"
    sampler_version: int = 1

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "family_id": int(self.family_id),
            "family_seed": int(self.family_seed),
            "sampler": {
                "name": self.sampler_name,
                "version": int(self.sampler_version),
            },
            "low": self.low.to_dict(),
            "haze": self.haze.to_dict(),
            "rain": self.rain.to_dict(),
            "snow": self.snow.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "FamilyParams":
        sampler = value.get("sampler", {})
        return cls(
            scene_id=str(value["scene_id"]),
            family_id=int(value["family_id"]),
            family_seed=int(value["family_seed"]),
            sampler_name=str(sampler.get("name", "sha256+numpy.default_rng")),
            sampler_version=int(sampler.get("version", 1)),
            low=LowParams(
                gamma=float(value["low"]["gamma"]),
                noise_sigma=float(value["low"]["noise_sigma"]),
                noise_seed=int(value["low"]["noise_seed"]),
            ),
            haze=HazeParams(
                beta=float(value["haze"]["beta"]),
                atmospheric_light=float(value["haze"]["atmospheric_light"]),
            ),
            rain=RainParams(mask_path=str(value["rain"]["mask"])),
            snow=SnowParams(mask_path=str(value["snow"]["mask"])),
        )


def make_deterministic_seed(global_seed: int, scene_id: str, family_id: int) -> int:
    """Derive a stable 64-bit seed without Python's process-randomized hash."""

    key = f"{int(global_seed)}:{scene_id}:{int(family_id)}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _sorted_paths(paths: Iterable[str | Path]) -> list[Path]:
    values = sorted((Path(path).resolve() for path in paths), key=lambda path: str(path))
    if not values:
        raise ValueError("A mask pool must contain at least one image")
    return values


def sample_family_params(
    *,
    scene_id: str,
    family_id: int,
    rain_mask_pool: Iterable[str | Path],
    snow_mask_pool: Iterable[str | Path],
    global_seed: int,
    gamma_min: float = 2.0,
    gamma_max: float = 3.0,
    noise_sigma_min: float = 0.03,
    noise_sigma_max: float = 0.08,
    beta_min: float = 1.0,
    beta_max: float = 2.0,
    atmospheric_light_min: float = 0.6,
    atmospheric_light_max: float = 0.9,
) -> FamilyParams:
    """Sample all family latent variables exactly once.

    The draw order is part of sampler version 1: gamma, noise sigma, haze
    beta, atmospheric light, rain mask index, snow mask index, noise seed.
    """

    if gamma_min >= gamma_max or noise_sigma_min >= noise_sigma_max:
        raise ValueError("Low-light parameter ranges must be increasing")
    if beta_min >= beta_max or atmospheric_light_min >= atmospheric_light_max:
        raise ValueError("Haze parameter ranges must be increasing")

    rain_paths = _sorted_paths(rain_mask_pool)
    snow_paths = _sorted_paths(snow_mask_pool)
    family_seed = make_deterministic_seed(global_seed, scene_id, family_id)
    rng = np.random.default_rng(family_seed)

    gamma = float(rng.uniform(gamma_min, gamma_max))
    noise_sigma = float(rng.uniform(noise_sigma_min, noise_sigma_max))
    beta = float(rng.uniform(beta_min, beta_max))
    atmospheric_light = float(
        rng.uniform(atmospheric_light_min, atmospheric_light_max)
    )
    rain_index = int(rng.integers(0, len(rain_paths)))
    snow_index = int(rng.integers(0, len(snow_paths)))
    noise_seed = int(rng.integers(0, 2**63 - 1))

    return FamilyParams(
        scene_id=str(scene_id),
        family_id=int(family_id),
        family_seed=family_seed,
        low=LowParams(gamma, noise_sigma, noise_seed),
        haze=HazeParams(beta, atmospheric_light),
        rain=RainParams(str(rain_paths[rain_index])),
        snow=SnowParams(str(snow_paths[snow_index])),
    )


def make_noise_map(params: LowParams, shape: tuple[int, ...]) -> np.ndarray:
    """Create the one Gaussian realization reused by every low-light state."""

    rng = np.random.default_rng(int(params.noise_seed))
    return rng.normal(0.0, params.noise_sigma, size=shape).astype(np.float64)
