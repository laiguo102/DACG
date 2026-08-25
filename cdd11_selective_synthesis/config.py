"""YAML configuration parsing for the synthesis CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .split import SPLITS, canonical_split


@dataclass(frozen=True)
class SynthesisConfig:
    seed: int
    clean_root: Path
    light_root: Path
    depth_root: Path
    rain_mask_root: Path
    snow_mask_root: Path
    output_root: Path
    split_file: Path | None
    families_per_scene: dict[str, int]
    train_ratio: float
    val_ratio: float
    test_ratio: float
    separate_mask_pools: bool
    gamma_min: float
    gamma_max: float
    noise_sigma_min: float
    noise_sigma_max: float
    beta_min: float
    beta_max: float
    atmospheric_light_min: float
    atmospheric_light_max: float
    overwrite: bool
    save_meta: bool
    image_ext: str


def _resolve_path(value: str | None, base: Path) -> Path | None:
    if value in (None, "", "null"):
        return None
    path = Path(str(value)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path: str | Path) -> SynthesisConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base = config_path.parent
    paths = raw.get("paths", {})
    split = raw.get("split", {})
    families = {
        canonical_split(name): int(value)
        for name, value in raw.get("families_per_scene", {}).items()
    }
    for name in SPLITS:
        families.setdefault(name, 0)
    if any(value < 0 for value in families.values()):
        raise ValueError("families_per_scene values must be non-negative")
    output = raw.get("output", {})
    low = raw.get("low", {})
    haze = raw.get("haze", {})
    return SynthesisConfig(
        seed=int(raw.get("seed", 20260825)),
        clean_root=_resolve_path(paths.get("clean_root"), base),
        light_root=_resolve_path(paths.get("light_root"), base),
        depth_root=_resolve_path(paths.get("depth_root"), base),
        rain_mask_root=_resolve_path(paths.get("rain_mask_root"), base),
        snow_mask_root=_resolve_path(paths.get("snow_mask_root"), base),
        output_root=_resolve_path(paths.get("output_root"), base),
        split_file=_resolve_path(split.get("file"), base),
        families_per_scene=families,
        train_ratio=float(split.get("train_ratio", 0.80)),
        val_ratio=float(split.get("val_ratio", 0.05)),
        test_ratio=float(split.get("test_ratio", 0.15)),
        separate_mask_pools=bool(
            raw.get("mask_pool", {}).get("separate_train_val_test", True)
        ),
        gamma_min=float(low.get("gamma_min", 2.0)),
        gamma_max=float(low.get("gamma_max", 3.0)),
        noise_sigma_min=float(low.get("noise_sigma_min", 0.03)),
        noise_sigma_max=float(low.get("noise_sigma_max", 0.08)),
        beta_min=float(haze.get("beta_min", 1.0)),
        beta_max=float(haze.get("beta_max", 2.0)),
        atmospheric_light_min=float(haze.get("atmospheric_light_min", 0.6)),
        atmospheric_light_max=float(haze.get("atmospheric_light_max", 0.9)),
        overwrite=bool(output.get("overwrite", False)),
        save_meta=bool(output.get("save_meta", True)),
        image_ext=str(output.get("image_ext", "png")).lstrip("."),
    )


def validate_config(config: SynthesisConfig) -> None:
    required = {
        "clean_root": config.clean_root,
        "light_root": config.light_root,
        "depth_root": config.depth_root,
        "rain_mask_root": config.rain_mask_root,
        "snow_mask_root": config.snow_mask_root,
        "output_root": config.output_root,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("Missing required config paths: " + ", ".join(missing))
    if not config.save_meta:
        raise ValueError("output.save_meta must remain true for reproducible families")
