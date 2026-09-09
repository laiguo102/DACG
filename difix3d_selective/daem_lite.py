"""Lightweight decoder-conditioned detail refinement for VAE skips."""

from __future__ import annotations

import torch
from torch import nn


class LayerNorm2d(nn.Module):
    """Channel-wise layer normalization for NCHW feature maps."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        mean = value.mean(dim=1, keepdim=True)
        variance = (value - mean).square().mean(dim=1, keepdim=True)
        normalized = (value - mean) * torch.rsqrt(variance + self.eps)
        return normalized * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first, second = value.chunk(2, dim=1)
        return first * second


class NAFBlockLite(nn.Module):
    """A self-contained, identity-initialized NAF-style refinement block."""

    def __init__(self, channels: int):
        super().__init__()
        expanded = channels * 2
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, expanded, 1)
        self.depthwise = nn.Conv2d(
            expanded, expanded, 3, padding=1, groups=expanded
        )
        self.simple_gate1 = SimpleGate()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
        )
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, expanded, 1)
        self.simple_gate2 = SimpleGate()
        self.conv4 = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first = self.conv1(self.norm1(value))
        first = self.simple_gate1(self.depthwise(first))
        first = self.conv2(first * self.channel_attention(first))
        intermediate = value + self.beta * first

        second = self.simple_gate2(self.conv3(self.norm2(intermediate)))
        second = self.conv4(second)
        return intermediate + self.gamma * second


class GatedDetailSkip(nn.Module):
    """Refine a projected encoder skip with a centered spatial-channel gate."""

    def __init__(
        self,
        channels: int,
        num_naf_blocks: int = 1,
        gate_reduction: int = 4,
        alpha_init: float = 0.1,
        prompt_dim: int | None = None,
        prompt_proj_dim: int = 32,
    ):
        super().__init__()
        if num_naf_blocks < 1:
            raise ValueError("num_naf_blocks must be at least 1")
        if gate_reduction < 1:
            raise ValueError("gate_reduction must be at least 1")

        self.channels = channels
        self.use_prompt = prompt_dim is not None
        self.refiner = nn.Sequential(
            *(NAFBlockLite(channels) for _ in range(num_naf_blocks))
        )
        gate_input_channels = channels * 2
        if self.use_prompt:
            self.prompt_projection = nn.Linear(prompt_dim, prompt_proj_dim)
            gate_input_channels += prompt_proj_dim
        else:
            self.prompt_projection = None

        hidden_channels = max(channels // gate_reduction, 32)
        self.gate = nn.Sequential(
            nn.Conv2d(gate_input_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                3,
                padding=1,
                groups=hidden_channels,
            ),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, channels, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        self.last_gate_shape: tuple[int, ...] | None = None
        self.last_gate_mean: torch.Tensor | None = None
        self.last_gate_std: torch.Tensor | None = None
        self.last_gate_min: torch.Tensor | None = None
        self.last_gate_max: torch.Tensor | None = None
        self.last_residual_ratio: torch.Tensor | None = None

    def forward(
        self,
        decoder_feature: torch.Tensor,
        projected_skip: torch.Tensor,
        prompt_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if decoder_feature.shape != projected_skip.shape:
            raise ValueError(
                "decoder feature and projected skip must have the same shape: "
                f"{tuple(decoder_feature.shape)} != {tuple(projected_skip.shape)}"
            )

        expert_feature = self.refiner(projected_skip)
        gate_parts = [decoder_feature, projected_skip]
        if self.use_prompt:
            if prompt_condition is None:
                raise ValueError("prompt_condition is required by this detail gate")
            prompt_feature = self.prompt_projection(prompt_condition)
            prompt_feature = prompt_feature[:, :, None, None].expand(
                -1, -1, decoder_feature.shape[-2], decoder_feature.shape[-1]
            )
            gate_parts.append(prompt_feature)

        gate = torch.sigmoid(self.gate(torch.cat(gate_parts, dim=1)))
        residual = self.alpha * (2.0 * gate - 1.0) * expert_feature
        denominator = projected_skip.detach().abs().mean() + 1e-6
        detached_gate = gate.detach()
        self.last_gate_shape = tuple(gate.shape)
        self.last_gate_mean = detached_gate.mean()
        self.last_gate_std = detached_gate.std(unbiased=False)
        self.last_gate_min = detached_gate.amin()
        self.last_gate_max = detached_gate.amax()
        self.last_residual_ratio = residual.detach().abs().mean() / denominator
        return projected_skip + residual

