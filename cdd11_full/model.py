"""Original DACG-IR construction and unmodified repository training objective."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.net.model import DACG_IR

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "DACG_IR": {
        "dim": 48, "num_blocks": [4, 6, 6, 8], "heads": [1, 2, 4, 8],
        "num_refinement_blocks": 4, "num_scales": 3,
    },
    "DACG_IR_S": {
        "dim": 32, "num_blocks": [4, 6, 6, 8], "heads": [1, 2, 4, 8],
        "num_refinement_blocks": 4, "num_scales": 3,
    },
}


def build_model(model_name: str = "DACG_IR") -> DACG_IR:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model {model_name!r}; choose from {sorted(MODEL_CONFIGS)}")
    return DACG_IR(**MODEL_CONFIGS[model_name])


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


def load_network(checkpoint_path: str, device: torch.device) -> tuple[DACG_IR, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = checkpoint.get("config", {}).get("model", "DACG_IR")
    network = build_model(model_name)
    network.load_state_dict(checkpoint["model"], strict=True)
    return network.to(device).eval(), checkpoint
