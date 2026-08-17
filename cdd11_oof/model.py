"""Lightning wrapper and checkpoint loading shared by train and inference."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import lightning.pytorch as pl

from src.net.model import DACG_IR


MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "DACG_IR": {"dim": 48, "num_blocks": [4, 6, 6, 8], "heads": [1, 2, 4, 8], "num_refinement_blocks": 4, "num_scales": 3},
    "DACG_IR_S": {"dim": 32, "num_blocks": [4, 6, 6, 8], "heads": [1, 2, 4, 8], "num_refinement_blocks": 4, "num_scales": 3},
}


class DACGLitModel(pl.LightningModule):
    def __init__(self, *, model_name: str = "DACG_IR", lr: float = 2e-4, epochs: int = 120, role: str = "unknown", split_dir: str = "", split_fingerprint: str = ""):
        super().__init__()
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model {model_name!r}")
        self.save_hyperparameters()
        self.net = DACG_IR(**MODEL_CONFIGS[model_name])
        self.l1 = nn.L1Loss()
        self._val_sums = None
        self._val_counts = None

    def forward(self, x):
        prediction = self.net(x)
        return prediction[..., :x.shape[-2], :x.shape[-1]]

    def training_step(self, batch, batch_idx):
        prediction = self(batch["lq"])
        l1 = self.l1(prediction, batch["gt"])
        pred_fft = torch.fft.rfft2(prediction, dim=(-2, -1))
        gt_fft = torch.fft.rfft2(batch["gt"], dim=(-2, -1))
        fft = torch.view_as_real(pred_fft).sub(torch.view_as_real(gt_fft)).abs().mean() * 0.1
        loss = l1 + fft
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        from .protocol import DEGRADATIONS
        self._val_sums = torch.zeros(len(DEGRADATIONS), device=self.device, dtype=torch.float64)
        self._val_counts = torch.zeros(len(DEGRADATIONS), device=self.device, dtype=torch.float64)

    def validation_step(self, batch, batch_idx):
        from .protocol import DEGRADATIONS
        prediction = self(batch["lq"]).clamp(0, 1)
        mse = (prediction - batch["gt"]).square().mean().clamp_min(1e-12)
        psnr = -10.0 * torch.log10(mse)
        index = DEGRADATIONS.index(batch["degradation"][0])
        self._val_sums[index] += psnr.double()
        self._val_counts[index] += 1

    def on_validation_epoch_end(self):
        sums = self.trainer.strategy.reduce(self._val_sums, reduce_op="sum")
        counts = self.trainer.strategy.reduce(self._val_counts, reduce_op="sum")
        macro = (sums / counts.clamp_min(1)).mean().float()
        self.log("val_macro_psnr", macro, prog_bar=True, sync_dist=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.epochs, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def load_network(checkpoint_path: str, device: torch.device) -> tuple[DACG_IR, dict[str, Any]]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only keyword
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hparams = dict(checkpoint.get("hyper_parameters", {}))
    model_name = hparams.get("model_name", "DACG_IR")
    net = DACG_IR(**MODEL_CONFIGS[model_name])
    state = checkpoint.get("state_dict", checkpoint)
    state = {key.removeprefix("net."): value for key, value in state.items() if key.startswith("net.")}
    if not state:
        raise ValueError("Checkpoint does not contain Lightning 'net.*' model weights")
    net.load_state_dict(state, strict=True)
    return net.to(device).eval(), hparams
