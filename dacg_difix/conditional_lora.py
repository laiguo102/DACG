"""Degradation-conditioned LoRA used by the DACG/Difix adapter.

The design follows S3Diff's insertion of a sample-dependent rank-by-rank
matrix between LoRA A and B.  Unlike the original implementation, the matrix
is residual, ``C = I + delta``, so a zero-initialized conditioner starts as an
ordinary static LoRA.
"""

from __future__ import annotations

import re
from types import MethodType
from typing import Callable, Iterator

import torch
from torch import nn


LORA_MODES = ("static", "p-global", "p-global-layer-id")


def unet_stage_id(layer_name: str) -> int:
    """Map a Stable-Diffusion UNet LoRA layer to one of ten stages."""

    match = re.search(r"(?:^|\.)down_blocks\.(\d+)(?:\.|$)", layer_name)
    if match:
        return int(match.group(1))
    if re.search(r"(?:^|\.)mid_block(?:\.|$)", layer_name):
        return 4
    match = re.search(r"(?:^|\.)up_blocks\.(\d+)(?:\.|$)", layer_name)
    if match:
        return 5 + int(match.group(1))
    return 9


def vae_decoder_stage_id(layer_name: str) -> int:
    """Map a VAE decoder LoRA layer to six ordered decoder stages.

    Stage 0 is the decoder input/mid block, stages 1--4 are up blocks 0--3,
    and stage 5 is the post/output block.  The four skip convolutions feed the
    matching up block and therefore share its stage id.
    """

    match = re.search(r"(?:^|\.)(?:decoder\.)?skip_conv_(\d+)(?:\.|$)", layer_name)
    if match:
        return int(match.group(1))

    match = re.search(r"(?:^|\.)(?:decoder\.)?up_blocks\.(\d+)(?:\.|$)", layer_name)
    if match:
        return 1 + int(match.group(1))
    if re.search(r"(?:^|\.)(?:decoder\.)?(?:conv_in|mid_block)(?:\.|$)", layer_name):
        return 0
    return 5


class ConditionalLoRAConditioner(nn.Module):
    """Generate residual LoRA mixing matrices from DACG ``P_global``.

    ``p-global`` produces one matrix per sample. ``p-global-layer-id`` also
    conditions on a learned stage embedding. ``static`` has no trainable
    conditioner and corresponds to the identity matrix between LoRA A and B.
    """

    def __init__(
        self,
        rank: int,
        num_stages: int,
        mode: str,
        p_global_dim: int = 96,
        hidden_dim: int = 256,
        stage_embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        if mode not in LORA_MODES:
            raise ValueError(f"Unknown LoRA conditioning mode: {mode}")
        self.rank = int(rank)
        self.num_stages = int(num_stages)
        self.mode = mode

        if mode != "static":
            self.p_global_norm = nn.LayerNorm(p_global_dim)
            self.p_global_proj = nn.Linear(p_global_dim, hidden_dim)
            self.activation = nn.GELU()
            if mode == "p-global-layer-id":
                self.stage_embedding = nn.Embedding(num_stages, stage_embedding_dim)
                head_input_dim = hidden_dim + stage_embedding_dim
            else:
                head_input_dim = hidden_dim
            self.to_delta = nn.Linear(head_input_dim, self.rank * self.rank)
            nn.init.zeros_(self.to_delta.weight)
            nn.init.zeros_(self.to_delta.bias)

    def forward(
        self,
        p_global: torch.Tensor | None,
        *,
        batch_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return ``C`` with shape ``[B, S, rank, rank]``."""

        if self.mode == "static":
            if p_global is not None:
                batch_size = p_global.shape[0]
                device, dtype = p_global.device, p_global.dtype
            if batch_size is None:
                raise ValueError("batch_size is required for a static conditioner without p_global")
            eye = torch.eye(self.rank, device=device, dtype=dtype)
            return eye.view(1, 1, self.rank, self.rank).expand(batch_size, 1, -1, -1)

        global_embedding = self.activation(self.p_global_proj(self.p_global_norm(p_global)))
        if self.mode == "p-global":
            delta = self.to_delta(global_embedding).unsqueeze(1)
        else:
            stages = self.stage_embedding.weight
            global_embedding = global_embedding[:, None, :].expand(-1, self.num_stages, -1)
            stages = stages[None, :, :].expand(p_global.shape[0], -1, -1)
            delta = self.to_delta(torch.cat((global_embedding, stages), dim=-1))

        delta = delta.reshape(p_global.shape[0], -1, self.rank, self.rank)
        eye = torch.eye(self.rank, device=delta.device, dtype=delta.dtype)
        return delta + eye.view(1, 1, self.rank, self.rank)


def _apply_rank_matrix(value: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.to(device=value.device, dtype=value.dtype)
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0)
    value_batch = value.shape[0]
    condition_batch = matrix.shape[0]
    if condition_batch != value_batch:
        if value_batch % condition_batch:
            raise ValueError(
                "LoRA condition batch cannot be aligned with the layer input: "
                f"condition={condition_batch}, input={value_batch}"
            )
        # The multi-view UNet merges and re-expands views inside attention
        # blocks.  Repeat each image condition to match the runtime layer
        # batch instead of assuming one fixed expansion at assignment time.
        matrix = matrix.repeat_interleave(value_batch // condition_batch, dim=0)
    if isinstance(value, torch.Tensor) and value.ndim == 4:
        return torch.einsum("brhw,brs->bshw", value, matrix)
    return torch.einsum("b...r,brs->b...s", value, matrix)


def conditional_lora_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """PEFT-compatible LoRA forward with a per-sample matrix between A/B."""

    if hasattr(self, "_check_forward_args"):
        self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)

    if getattr(self, "disable_adapters", False):
        if getattr(self, "merged", False):
            self.unmerge()
        return self.base_layer(x, *args, **kwargs)
    if adapter_names is not None:
        return self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
    if getattr(self, "merged", False):
        return self.base_layer(x, *args, **kwargs)

    result = self.base_layer(x, *args, **kwargs)
    result_dtype = result.dtype
    active_adapters = getattr(self, "active_adapters", ())
    if isinstance(active_adapters, str):
        active_adapters = (active_adapters,)

    for active_adapter in active_adapters:
        if active_adapter not in self.lora_A:
            continue
        lora_a = self.lora_A[active_adapter]
        lora_b = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]
        use_dora = getattr(self, "use_dora", {}).get(active_adapter, False)
        if use_dora:
            raise NotImplementedError("Conditional LoRA does not support DoRA")

        lora_input = x.to(lora_a.weight.dtype)
        low_rank = lora_a(dropout(lora_input))
        matrix = getattr(self, "condition_matrix", None)
        if matrix is not None:
            low_rank = _apply_rank_matrix(low_rank, matrix)
        result = result + lora_b(low_rank) * scaling

    return result.to(result_dtype)


def iter_lora_modules(module: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    """Yield PEFT-style modules containing ``base_layer``, ``lora_A`` and ``lora_B``."""

    for name, child in module.named_modules():
        if hasattr(child, "base_layer") and hasattr(child, "lora_A") and hasattr(child, "lora_B"):
            yield name, child


def install_conditional_lora_forward(module: nn.Module) -> list[str]:
    """Install :func:`conditional_lora_forward` on all PEFT LoRA layers."""

    layer_names = []
    for name, child in iter_lora_modules(module):
        child.forward = MethodType(conditional_lora_forward, child)
        layer_names.append(name)
    return layer_names


def clear_condition_matrices(module: nn.Module) -> None:
    for _, child in iter_lora_modules(module):
        child.condition_matrix = None


def assign_condition_matrices(
    module: nn.Module,
    matrices: torch.Tensor,
    stage_id: Callable[[str], int],
    repeat_interleave: int = 1,
) -> None:
    """Assign matrices to LoRA layers without detaching the conditioner graph."""

    for name, child in iter_lora_modules(module):
        stage = stage_id(name) if matrices.shape[1] > 1 else 0
        matrix = matrices[:, stage]
        if repeat_interleave > 1:
            matrix = matrix.repeat_interleave(repeat_interleave, dim=0)
        child.condition_matrix = matrix


__all__ = [
    "ConditionalLoRAConditioner",
    "LORA_MODES",
    "assign_condition_matrices",
    "clear_condition_matrices",
    "conditional_lora_forward",
    "install_conditional_lora_forward",
    "iter_lora_modules",
    "unet_stage_id",
    "vae_decoder_stage_id",
]
