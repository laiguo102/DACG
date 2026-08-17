"""The frozen AIO3-v1 RGB PSNR/SSIM implementation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _as_image(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
        raise ValueError(f"expected one RGB image [1,3,H,W], got {tuple(tensor.shape)}")
    return tensor.float()


def rgb_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction, target = _as_image(prediction).clamp(0.0, 1.0), _as_image(target)
    mse = torch.mean((prediction - target) ** 2).item()
    return float("inf") if mse == 0.0 else -10.0 * math.log10(mse)


def _gaussian_window(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(11, device=device, dtype=dtype) - 5
    gaussian = torch.exp(-(coords**2) / (2 * 1.5**2))
    gaussian /= gaussian.sum()
    window = torch.outer(gaussian, gaussian)
    return window.expand(3, 1, 11, 11).contiguous()


def rgb_ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction, target = _as_image(prediction).clamp(0.0, 1.0), _as_image(target)
    if min(prediction.shape[-2:]) < 11:
        raise ValueError("AIO3 SSIM requires height and width >= 11 for its valid 11x11 window")
    window = _gaussian_window(prediction.device, prediction.dtype)
    mu_x = F.conv2d(prediction, window, groups=3)
    mu_y = F.conv2d(target, window, groups=3)
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x2 = F.conv2d(prediction.square(), window, groups=3) - mu_x2
    sigma_y2 = F.conv2d(target.square(), window, groups=3) - mu_y2
    sigma_xy = F.conv2d(prediction * target, window, groups=3) - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return float(score.mean().item())


def image_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {"psnr": rgb_psnr(prediction, target), "ssim": rgb_ssim(prediction, target)}
