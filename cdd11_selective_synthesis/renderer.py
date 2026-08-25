"""Family rendering with explicit degradation switches."""

from __future__ import annotations

import cv2
import numpy as np

from .degradations import apply_haze, apply_low, apply_rain, apply_snow
from .params import FamilyParams, make_noise_map
from .states import RENDER_ORDER, VALID_STATE_TUPLES, canonical_state
from .io import resize_mask


def _image_gray(image: np.ndarray) -> np.ndarray:
    encoded = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(encoded, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0


def render_state(
    clean: np.ndarray,
    state: tuple[str, ...] | list[str] | set[str],
    light_map: np.ndarray,
    depth_map: np.ndarray,
    rain_mask: np.ndarray,
    snow_mask: np.ndarray,
    params: FamilyParams,
    noise_map: np.ndarray,
    image_gray: np.ndarray | None = None,
) -> np.ndarray:
    """Render one state; operator order is independent of tuple order."""

    state_tuple = canonical_state(state)
    if state_tuple not in VALID_STATE_TUPLES:
        raise ValueError(f"Unsupported state: {state_tuple}")
    if clean.ndim != 3 or clean.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 clean image, got {clean.shape}")
    if image_gray is None:
        image_gray = _image_gray(clean)

    enabled = set(state_tuple)
    x = clean.astype(np.float64, copy=True)
    for degradation in RENDER_ORDER:
        if degradation not in enabled:
            continue
        if degradation == "low":
            x = apply_low(x, light_map, image_gray, params.low, noise_map)
        elif degradation == "rain":
            x = apply_rain(x, rain_mask)
        elif degradation == "snow":
            x = apply_snow(x, snow_mask)
        elif degradation == "haze":
            x = apply_haze(x, depth_map, params.haze)
    return np.clip(x, 0.0, 1.0)


def render_family(
    *,
    clean: np.ndarray,
    light_map: np.ndarray,
    depth_map: np.ndarray,
    rain_mask: np.ndarray,
    snow_mask: np.ndarray,
    params: FamilyParams,
) -> dict[tuple[str, ...], np.ndarray]:
    """Render all 12 states from one shared family parameter realization."""

    height, width, channels = clean.shape
    if channels != 3:
        raise ValueError(f"Expected HxWx3 clean image, got {clean.shape}")
    if light_map.shape[:2] != (height, width):
        raise ValueError("Light map dimensions do not match clean image")
    if depth_map.shape[:2] != (height, width):
        raise ValueError("Depth map dimensions do not match clean image")

    resized_rain = resize_mask(rain_mask, height, width)
    resized_snow = resize_mask(snow_mask, height, width)
    noise_map = make_noise_map(params.low, clean.shape)
    image_gray = _image_gray(clean)
    rendered: dict[tuple[str, ...], np.ndarray] = {}
    for state in VALID_STATE_TUPLES:
        rendered[state] = render_state(
            clean,
            state,
            light_map,
            depth_map,
            resized_rain,
            resized_snow,
            params,
            noise_map,
            image_gray,
        )
    return rendered
