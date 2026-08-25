"""Deterministic CDD-11 degradation operators.

All random values are sampled by :mod:`params` and passed into these
functions.  The formulas intentionally follow the released OneRestore
``syn_data.py`` behavior, including its depth preprocessing expression.
"""

from __future__ import annotations

import cv2
import numpy as np

from .params import HazeParams, LowParams


def guide_filter(
    image: np.ndarray, guide: np.ndarray, window: tuple[int, int], epsilon: float
) -> np.ndarray:
    mean_image = cv2.blur(image, window)
    mean_guide = cv2.blur(guide, window)
    mean_image_sq = cv2.blur(image * image, window)
    mean_image_guide = cv2.blur(image * guide, window)
    variance = mean_image_sq - mean_image * mean_image
    covariance = mean_image_guide - mean_image * mean_guide
    coefficient = covariance / (variance + epsilon)
    intercept = mean_guide - coefficient * mean_image
    mean_coefficient = cv2.blur(coefficient, window)
    mean_intercept = cv2.blur(intercept, window)
    return mean_coefficient * image + mean_intercept


def apply_low(
    image: np.ndarray,
    light_map: np.ndarray,
    image_gray: np.ndarray,
    params: LowParams,
    noise: np.ndarray,
) -> np.ndarray:
    """Apply the official low-light operator using an explicit noise map."""

    if image.shape != noise.shape:
        raise ValueError(
            f"Low-light noise shape {noise.shape} does not match image {image.shape}"
        )
    guided_light = guide_filter(light_map, image_gray, (3, 3), 0.01)[
        ..., np.newaxis
    ]
    reflectance = image / (guided_light + 1e-7)
    illumination = (guided_light + 1e-7) ** params.gamma
    # The paper calls [0.03, 0.08] a variance range, but the official code
    # passes the sampled value as np.random.normal(..., scale=n).  It is
    # therefore treated as noise_sigma (standard deviation) here.
    return np.clip(reflectance * illumination + noise, 0.0, 1.0)


def apply_rain(image: np.ndarray, rain_mask: np.ndarray) -> np.ndarray:
    """Apply the official additive rain composition without intermediate clip."""

    return image + rain_mask


def apply_snow(image: np.ndarray, snow_mask: np.ndarray) -> np.ndarray:
    """Apply the official white snow alpha-compositing operator."""

    return image * (1.0 - snow_mask) + snow_mask


def apply_haze(
    image: np.ndarray, depth_map: np.ndarray, params: HazeParams
) -> np.ndarray:
    """Apply the official atmospheric-scattering expression."""

    depth = depth_map
    if depth.ndim == 2:
        depth = depth[..., np.newaxis]
    blurred_depth = cv2.blur(depth, (22, 22))
    transmission = np.exp(
        -np.minimum(1.0 - blurred_depth, 0.7) * params.beta
    )
    return np.clip(
        image * transmission
        + params.atmospheric_light * (1.0 - transmission),
        0.0,
        1.0,
    )
