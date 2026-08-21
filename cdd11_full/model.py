"""Original DACG-IR construction and unmodified repository training objective."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.net.model import DACG_IR, Degradation_Aware_Module

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "DACG_IR": {
        "dim": 48, "num_blocks": [4, 6, 6, 8], "heads": [1, 2, 4, 8],
        "num_refinement_blocks": 4, "num_scales": 3,
    },
}


def build_model(model_name: str = "DACG_IR") -> DACG_IR:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model {model_name!r}; choose from {sorted(MODEL_CONFIGS)}")
    return DACG_IR(**MODEL_CONFIGS[model_name])


def _checkpoint_model_config(checkpoint: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    model_value = checkpoint.get("config", {}).get("model", "DACG_IR")
    if isinstance(model_value, str):
        model_name = model_value
        overrides: Mapping[str, Any] = {}
    else:
        model_name = str(model_value.get("model_name", "DACG_IR"))
        overrides = model_value
    config = dict(MODEL_CONFIGS[model_name])
    config.update({key: overrides[key] for key in config if key in overrides})
    return model_name, config


class FrozenDAMEncoder(nn.Module):
    """Frozen degradation encoder extracted from a trained DACG checkpoint."""

    def __init__(self, context_net: Degradation_Aware_Module, padder_size: int):
        super().__init__()
        self.context_net = context_net
        self.padder_size = padder_size
        self.requires_grad_(False)
        self.eval()

    def forward(self, degraded_01: torch.Tensor) -> torch.Tensor:
        _, _, height, width = degraded_01.shape
        pad_h = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - width % self.padder_size) % self.padder_size
        mode = "reflect" if height > pad_h and width > pad_w and height > 1 and width > 1 else "replicate"
        padded = F.pad(degraded_01, (0, pad_w, 0, pad_h), mode=mode)
        _, p_global = self.context_net(padded)
        return p_global


class OriginalDACGLoss(nn.Module):
    """Repository L1 + 0.1 times mean absolute real/imaginary FFT error."""

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        l1 = self.l1(prediction, target)
        # CUDA FFT is deliberately kept in FP32 even when network autocast is BF16.
        pred_fft = torch.fft.rfft2(prediction.float(), dim=(-2, -1))
        target_fft = torch.fft.rfft2(target.float(), dim=(-2, -1))
        fft = torch.view_as_real(pred_fft).sub(torch.view_as_real(target_fft)).abs().mean() * 0.1
        loss = l1.float() + fft
        return loss, {"l1": l1.detach().float(), "fft": fft.detach().float()}

    def per_sample(self, prediction: torch.Tensor, target: torch.Tensor):
        """Paper loss for each item, enabling exact 11-sample microbatching."""
        l1 = (prediction.float() - target.float()).abs().flatten(1).mean(1)
        pred_fft = torch.view_as_real(torch.fft.rfft2(prediction.float(), dim=(-2, -1)))
        target_fft = torch.view_as_real(torch.fft.rfft2(target.float(), dim=(-2, -1)))
        fft = (pred_fft - target_fft).abs().flatten(1).mean(1) * 0.1
        return l1 + fft, l1, fft


def architecture_metadata(model_name: str = "DACG_IR") -> dict[str, object]:
    config = MODEL_CONFIGS[model_name]
    return {
        "model_name": model_name,
        "implementation": "original_dacg_ir",
        "expected_trainable_parameters": 30_861_200,
        **config,
    }


def load_network(checkpoint_path: str, device: torch.device) -> tuple[DACG_IR, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _, model_config = _checkpoint_model_config(checkpoint)
    network = DACG_IR(**model_config)
    network.load_state_dict(checkpoint["model"], strict=True)
    return network.to(device).eval(), checkpoint


def load_dam_encoder(
    checkpoint_path: str, device: torch.device
) -> tuple[FrozenDAMEncoder, dict[str, Any]]:
    """Load only ``context_net`` weights and leave the restoration network unbuilt."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _, model_config = _checkpoint_model_config(checkpoint)
    dim = int(model_config["dim"])
    context_net = Degradation_Aware_Module(
        dim=dim,
        num_scales=int(model_config["num_scales"]),
        dim_list=[int(dim * 2 ** index) for index in range(4)],
    )
    prefix = "context_net."
    context_state = {
        key[len(prefix):]: value
        for key, value in checkpoint["model"].items()
        if key.startswith(prefix)
    }
    context_net.load_state_dict(context_state, strict=True)
    encoder = FrozenDAMEncoder(context_net, 2 ** len(model_config["num_blocks"]))
    return encoder.to(device).eval(), checkpoint
