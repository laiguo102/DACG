"""Tensor helpers for retaining only the primary Difix view."""

from __future__ import annotations

import torch


def select_main_view(tensor: torch.Tensor, batch_size: int, num_views: int = 2) -> torch.Tensor:
    if tensor.shape[0] != batch_size * num_views:
        raise ValueError(
            f"Expected flattened batch {batch_size * num_views}, got {tensor.shape[0]}"
        )
    return tensor.reshape(batch_size, num_views, *tensor.shape[1:])[:, 0].contiguous()


def select_main_view_skips(
    activations: list[torch.Tensor], batch_size: int, num_views: int = 2
) -> list[torch.Tensor]:
    return [select_main_view(value, batch_size, num_views) for value in activations]
