from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cdd11_selective_synthesis.degradations import apply_haze, apply_low
from cdd11_selective_synthesis.manifest import family_records
from cdd11_selective_synthesis.params import (
    HazeParams,
    LowParams,
    make_deterministic_seed,
    make_noise_map,
    sample_family_params,
)
from cdd11_selective_synthesis.renderer import render_family, render_state
from cdd11_selective_synthesis.split import (
    deterministic_scene_split,
    validate_disjoint_mask_pools,
    validate_no_scene_leakage,
)
from cdd11_selective_synthesis.states import VALID_STATE_TUPLES, state_name
from cdd11_selective_synthesis.validation import (
    build_distribution_report,
    is_complete_family,
    validate_family,
    validate_manifest,
)


def _fixture(tmp_path: Path):
    clean = np.zeros((12, 16, 3), dtype=np.float64)
    clean[..., 0] = np.linspace(0.1, 0.8, 16)[None, :]
    clean[..., 1] = 0.3
    clean[..., 2] = np.linspace(0.8, 0.1, 12)[:, None]
    light = np.full((12, 16), 0.5, dtype=np.float64)
    depth = np.full((12, 16, 3), 0.25, dtype=np.float64)
    rain = np.zeros((5, 7, 3), dtype=np.float64)
    rain[::2, :, :] = 0.08
    snow = np.zeros((7, 5, 3), dtype=np.float64)
    snow[:, ::2, :] = 0.35
    rain_path = tmp_path / "rain.png"
    snow_path = tmp_path / "snow.png"
    cv2.imwrite(str(rain_path), np.rint(rain * 255).astype(np.uint8))
    cv2.imwrite(str(snow_path), np.rint(snow * 255).astype(np.uint8))
    return clean, light, depth, [rain_path], [snow_path]


def test_seed_and_family_sampling_are_stable(tmp_path):
    _, _, _, rain_pool, snow_pool = _fixture(tmp_path)
    first = sample_family_params(
        scene_id="scene-1",
        family_id=2,
        rain_mask_pool=rain_pool,
        snow_mask_pool=snow_pool,
        global_seed=20260825,
    )
    second = sample_family_params(
        scene_id="scene-1",
        family_id=2,
        rain_mask_pool=rain_pool,
        snow_mask_pool=snow_pool,
        global_seed=20260825,
    )
    assert first == second
    assert make_deterministic_seed(1, "a", 0) == make_deterministic_seed(1, "a", 0)
    assert 2.0 <= first.low.gamma <= 3.0
    assert 0.03 <= first.low.noise_sigma <= 0.08
    assert 1.0 <= first.haze.beta <= 2.0
    assert 0.6 <= first.haze.atmospheric_light <= 0.9


def test_render_family_has_all_states_and_shared_noise(tmp_path):
    clean, light, depth, rain_pool, snow_pool = _fixture(tmp_path)
    params = sample_family_params(
        scene_id="scene-1",
        family_id=0,
        rain_mask_pool=rain_pool,
        snow_mask_pool=snow_pool,
        global_seed=7,
    )
    rendered = render_family(
        clean=clean,
        light_map=light,
        depth_map=depth,
        rain_mask=cv2.imread(str(rain_pool[0]), cv2.IMREAD_COLOR) / 255.0,
        snow_mask=cv2.imread(str(snow_pool[0]), cv2.IMREAD_COLOR) / 255.0,
        params=params,
    )
    assert tuple(rendered) == VALID_STATE_TUPLES
    assert set(rendered) == set(VALID_STATE_TUPLES)
    assert rendered[()].shape == clean.shape
    assert np.array_equal(
        make_noise_map(params.low, clean.shape), make_noise_map(params.low, clean.shape)
    )
    with pytest.raises(ValueError, match="Rain and snow"):
        render_state(
            clean,
            ("rain", "snow"),
            light,
            depth,
            np.zeros_like(clean),
            np.zeros_like(clean),
            params,
            make_noise_map(params.low, clean.shape),
        )


def test_render_order_is_low_then_rain_then_haze(tmp_path):
    clean, light, depth, rain_pool, snow_pool = _fixture(tmp_path)
    params = sample_family_params(
        scene_id="scene-1",
        family_id=0,
        rain_mask_pool=rain_pool,
        snow_mask_pool=snow_pool,
        global_seed=9,
    )
    rain = cv2.imread(str(rain_pool[0]), cv2.IMREAD_COLOR) / 255.0
    snow = cv2.imread(str(snow_pool[0]), cv2.IMREAD_COLOR) / 255.0
    rendered = render_family(
        clean=clean,
        light_map=light,
        depth_map=depth,
        rain_mask=rain,
        snow_mask=snow,
        params=params,
    )
    noise = make_noise_map(params.low, clean.shape)
    gray = cv2.cvtColor(np.rint(clean * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    manual = apply_low(clean, light, gray, params.low, noise)
    manual = manual + cv2.resize(rain, (clean.shape[1], clean.shape[0]))
    manual = apply_haze(manual, depth, params.haze)
    assert np.array_equal(rendered[("low", "haze", "rain")], np.clip(manual, 0, 1))


def test_manifest_has_twenty_pairs_and_target_states(tmp_path):
    family = tmp_path / "family_000"
    family.mkdir()
    for state in VALID_STATE_TUPLES:
        path = family / f"{state_name(state)}.png"
        cv2.imwrite(str(path), np.zeros((2, 2, 3), dtype=np.uint8))
    records = family_records(
        split="train", scene_id="scene-1", family_id=0, family_directory=family
    )
    validate_manifest(records, expected_families=1)
    assert len(records) == 20
    assert {row["remove"] for row in records} == {"low", "haze", "rain", "snow"}


def test_family_validation_and_distribution_report(tmp_path):
    clean, light, depth, rain_pool, snow_pool = _fixture(tmp_path)
    params = sample_family_params(
        scene_id="scene-1",
        family_id=0,
        rain_mask_pool=rain_pool,
        snow_mask_pool=snow_pool,
        global_seed=11,
    )
    rendered = render_family(
        clean=clean,
        light_map=light,
        depth_map=depth,
        rain_mask=cv2.imread(str(rain_pool[0]), cv2.IMREAD_COLOR) / 255.0,
        snow_mask=cv2.imread(str(snow_pool[0]), cv2.IMREAD_COLOR) / 255.0,
        params=params,
    )
    family = tmp_path / "scene-1" / "family_000"
    family.mkdir(parents=True)
    for state, image in rendered.items():
        cv2.imwrite(str(family / f"{state_name(state)}.png"), np.rint(image * 255).astype(np.uint8))
    metadata = params.to_dict()
    metadata.update({"schema_version": 1, "states": [state_name(s) for s in VALID_STATE_TUPLES]})
    (family / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert is_complete_family(family)
    assert validate_family(family)["scene_id"] == "scene-1"
    report = build_distribution_report([metadata])
    assert report["all_reuse_checks_passed"]


def test_scene_and_mask_splits_are_disjoint():
    splits = deterministic_scene_split([f"scene-{i}" for i in range(12)], seed=42)
    validate_no_scene_leakage(splits)
    assert all(splits.values())
    validate_disjoint_mask_pools(
        {
            "train": ["/m/train/a.png"],
            "val": ["/m/val/a.png"],
            "test": ["/m/test/a.png"],
        }
    )
