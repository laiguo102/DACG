"""Training losses for DACG-conditioned Difix3D restoration."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


VGG_FEATURE_LAYERS = (3, 8, 15, 22, 29)
STYLE_WEIGHTS = (
    1.0 / 2.6,
    1.0 / 4.8,
    1.0 / 3.7,
    1.0 / 5.6,
    10.0 / 1.5,
)


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Return one unnormalized channel Gram matrix per image."""

    if features.ndim != 4:
        raise ValueError(f"expected [B,C,H,W] features, got {tuple(features.shape)}")
    batch, channels, height, width = features.shape
    flattened = features.reshape(batch, channels, height * width)
    return torch.bmm(flattened, flattened.transpose(1, 2))


def imagenet_normalize(images: torch.Tensor) -> torch.Tensor:
    """Convert Difix tensors in ``[-1, 1]`` to VGG/ImageNet input space."""

    images = images.float().add(1.0).mul(0.5).clamp(0.0, 1.0)
    mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (images - mean) / std


def aligned_center_crop(
    prediction: torch.Tensor,
    target: torch.Tensor,
    max_size: int = 400,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the same centered square crop to a prediction/target pair."""

    crop_size = min(max_size, prediction.shape[-2], prediction.shape[-1])
    top = (prediction.shape[-2] - crop_size) // 2
    left = (prediction.shape[-1] - crop_size) // 2
    slices = (..., slice(top, top + crop_size), slice(left, left + crop_size))
    return prediction[slices], target[slices]


class VGGGramLoss(nn.Module):
    """Style loss over VGG relu1_2 through relu5_2 activations."""

    def __init__(
        self,
        features: nn.Sequential,
        layer_ids: Sequence[int] = VGG_FEATURE_LAYERS,
        style_weights: Sequence[float] = STYLE_WEIGHTS,
        crop_size: int = 400,
    ) -> None:
        super().__init__()
        self.features = features.eval().requires_grad_(False)
        self.layer_ids = tuple(int(value) for value in layer_ids)
        self.style_weights = tuple(float(value) for value in style_weights)
        if len(self.layer_ids) != len(self.style_weights):
            raise ValueError("one style weight is required for each VGG feature layer")
        self.crop_size = int(crop_size)

    def _activations(self, images: torch.Tensor) -> list[torch.Tensor]:
        result: list[torch.Tensor] = []
        last_layer = self.layer_ids[-1]
        for index, layer in enumerate(self.features):
            images = layer(images)
            if index in self.layer_ids:
                result.append(images)
            if index >= last_layer:
                break
        return result

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction, target = aligned_center_crop(prediction, target, self.crop_size)
        prediction = imagenet_normalize(prediction)
        target = imagenet_normalize(target)
        prediction_features = self._activations(prediction)
        with torch.no_grad():
            target_features = self._activations(target)
        losses = []
        for pred, reference, weight in zip(
            prediction_features,
            target_features,
            self.style_weights,
            strict=True,
        ):
            _, channels, height, width = reference.shape
            layer_loss = weight * F.mse_loss(
                gram_matrix(pred),
                gram_matrix(reference),
            )
            losses.append(layer_loss / (channels * height * width))
        return torch.stack(losses).sum()


class DifixRestorationLoss(nn.Module):
    """Weighted main-view MSE, LPIPS, and delayed VGG Gram loss."""

    def __init__(
        self,
        lpips_model: nn.Module,
        gram_model: nn.Module,
        lambda_mse: float = 1.0,
        lambda_lpips: float = 1.0,
        lambda_gram: float = 1.0,
        gram_warmup_steps: int = 2_000,
    ) -> None:
        super().__init__()
        self.lpips_model = lpips_model.eval().requires_grad_(False)
        self.gram_model = gram_model.eval().requires_grad_(False)
        self.lambda_mse = float(lambda_mse)
        self.lambda_lpips = float(lambda_lpips)
        self.lambda_gram = float(lambda_gram)
        self.gram_warmup_steps = int(gram_warmup_steps)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction_float = prediction.float()
        target_float = target.float()
        mse = F.mse_loss(prediction_float, target_float)
        perceptual = self.lpips_model(prediction_float, target_float).mean()
        gram = prediction_float.new_zeros(())
        if self.lambda_gram and global_step >= self.gram_warmup_steps:
            gram = self.gram_model(prediction_float, target_float)

        total = (
            self.lambda_mse * mse
            + self.lambda_lpips * perceptual
            + self.lambda_gram * gram
        )
        return total, {
            "mse": mse.detach(),
            "lpips": perceptual.detach(),
            "gram": gram.detach(),
            "total": total.detach(),
        }


def build_restoration_loss(
    *,
    lambda_mse: float = 1.0,
    lambda_lpips: float = 1.0,
    lambda_gram: float = 1.0,
    gram_warmup_steps: int = 2_000,
) -> DifixRestorationLoss:
    """Build pretrained loss networks; called only by the training entry point."""

    import lpips

    lpips_model = lpips.LPIPS(net="vgg")
    if lambda_gram:
        from torchvision.models import VGG16_Weights, vgg16

        vgg_features = vgg16(weights=VGG16_Weights.DEFAULT).features
        gram_model: nn.Module = VGGGramLoss(vgg_features)
    else:
        gram_model = nn.Identity()
    return DifixRestorationLoss(
        lpips_model,
        gram_model,
        lambda_mse=lambda_mse,
        lambda_lpips=lambda_lpips,
        lambda_gram=lambda_gram,
        gram_warmup_steps=gram_warmup_steps,
    )
