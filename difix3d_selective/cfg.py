"""Pure tensor operations for endpoint-correct positive/negative CFG."""

from __future__ import annotations

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
