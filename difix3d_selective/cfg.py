"""Endpoint-correct positive/negative CFG blending operations."""

from __future__ import annotations

import numpy as np
import torch


def blend_cfg_latents(
    z_positive: torch.Tensor, z_negative: torch.Tensor, beta: float
) -> torch.Tensor:
    """Interpolate denoised latents, preserving the exact endpoint tensors."""

    if z_positive.shape != z_negative.shape:
        raise ValueError(
            "positive/negative latent shape mismatch: "
            f"positive={tuple(z_positive.shape)}, negative={tuple(z_negative.shape)}"
        )
    beta = float(beta)
    if beta == 0.0:
        return z_negative
    if beta == 1.0:
        return z_positive
    return z_negative + beta * (z_positive - z_negative)


def blend_cfg_skips(
    positive_skips: list[torch.Tensor],
    negative_skips: list[torch.Tensor],
    beta: float,
) -> list[torch.Tensor]:
    """Interpolate every VAE skip so CFG endpoints are complete model modes."""

    if len(positive_skips) != len(negative_skips):
        raise ValueError(
            "positive/negative skip count mismatch: "
            f"positive={len(positive_skips)}, negative={len(negative_skips)}"
        )
    return [
        blend_cfg_latents(positive, negative, beta)
        for positive, negative in zip(positive_skips, negative_skips)
    ]


def blend_cfg_pixels(
    positive: np.ndarray, negative: np.ndarray, beta: float
) -> np.ndarray:
    """Blend decoded RGB images with the same endpoint-correct CFG formula.

    Inputs must be equally shaped floating-point RGB arrays in ``[0, 1]``.  The
    unclipped affine blend is useful for ``beta > 1`` extrapolation; clipping is
    applied only after the blend so that it matches normal image export.
    """

    if positive.shape != negative.shape:
        raise ValueError(
            "positive/negative pixel shape mismatch: "
            f"positive={positive.shape}, negative={negative.shape}"
        )
    if positive.ndim != 3 or positive.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB images, got {positive.shape}")
    if not np.issubdtype(positive.dtype, np.floating) or not np.issubdtype(
        negative.dtype, np.floating
    ):
        raise TypeError("Pixel inputs must be floating-point arrays in [0, 1]")
    beta = float(beta)
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and non-negative")
    if beta == 0.0:
        return negative
    if beta == 1.0:
        return positive
    return np.clip(negative + beta * (positive - negative), 0.0, 1.0)
