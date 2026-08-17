"""DACG-IR adapter implementing the AIO3 model contract."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from src.net.model import DACG_IR


DEFAULT_MODEL_CONFIG = {
    "dim": 48,
    "num_blocks": [4, 6, 6, 8],
    "num_refinement_blocks": 4,
    "heads": [1, 2, 4, 8],
    "num_scales": 3,
}


class DACGIRAdapter(nn.Module):
    """Expose a strict ``[B,3,H,W] -> [B,3,H,W]`` raw restoration API."""

    def __init__(self, model_config: Mapping[str, Any] | None = None):
        super().__init__()
        config = dict(DEFAULT_MODEL_CONFIG)
        if model_config:
            config.update(model_config)
        self.model_config = config
        self.model = DACG_IR(**config)

    def forward(self, degraded: torch.Tensor) -> torch.Tensor:
        if degraded.ndim != 4 or degraded.shape[1] != 3:
            raise ValueError(f"expected [B,3,H,W], got {tuple(degraded.shape)}")
        restored_raw = self.model(degraded)
        if restored_raw.shape != degraded.shape:
            raise RuntimeError(
                f"adapter shape contract failed: {tuple(degraded.shape)} -> "
                f"{tuple(restored_raw.shape)}"
            )
        return restored_raw


def build_model(model_config: Mapping[str, Any] | None = None) -> DACGIRAdapter:
    """Build a randomly initialized DACG-IR model; no pretrained weights are read."""

    return DACGIRAdapter(model_config)
