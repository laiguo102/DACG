"""CDD-11 RGB PSNR and SSIM used consistently for official-test reporting."""

from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn.functional as F

from .protocol import ARITY_GROUPS


def _image(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
        raise ValueError(f"expected one RGB image [1,3,H,W], got {tuple(tensor.shape)}")
    return tensor.float()


def rgb_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction, target = _image(prediction).clamp(0, 1), _image(target)
    mse = torch.mean((prediction - target) ** 2).item()
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def rgb_ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction, target = _image(prediction).clamp(0, 1), _image(target)
    if min(prediction.shape[-2:]) < 11:
        raise ValueError("SSIM requires image height and width >= 11")
    coords = torch.arange(11, device=prediction.device, dtype=prediction.dtype) - 5
    gaussian = torch.exp(-(coords**2) / (2 * 1.5**2))
    gaussian /= gaussian.sum()
    window = torch.outer(gaussian, gaussian).expand(3, 1, 11, 11).contiguous()
    mu_x, mu_y = F.conv2d(prediction, window, groups=3), F.conv2d(target, window, groups=3)
    sigma_x = F.conv2d(prediction.square(), window, groups=3) - mu_x.square()
    sigma_y = F.conv2d(target.square(), window, groups=3) - mu_y.square()
    sigma_xy = F.conv2d(prediction * target, window, groups=3) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return float(score.mean().item())


def summarize(rows, degradations):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["degradation"]].append(row)
    missing = [value for value in degradations if not grouped[value]]
    if missing:
        raise ValueError(f"Missing metric categories: {missing}")
    result = {"images": float(len(rows))}
    for degradation in degradations:
        values = grouped[degradation]
        result[f"{degradation}/images"] = float(len(values))
        for metric in ("psnr", "ssim"):
            result[f"{degradation}/{metric}"] = math.fsum(row[metric] for row in values) / len(values)
    for group, categories in ARITY_GROUPS.items():
        for metric in ("psnr", "ssim"):
            result[f"{group}/{metric}"] = math.fsum(
                result[f"{value}/{metric}"] for value in categories) / len(categories)
        result[f"{group}/categories"] = float(len(categories))
    for metric in ("psnr", "ssim"):
        result[f"macro/{metric}"] = math.fsum(
            result[f"{value}/{metric}"] for value in degradations) / len(degradations)
    return result
