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
    def __init__(self, *, model_name: str = "DACG_IR", lr: float = 2e-4, epochs: int = 120, role: str = "unknown", split_dir: str = "", split_fingerprint: str = "", wandb_media_every: int = 40):
        super().__init__()
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model {model_name!r}")
        self.save_hyperparameters()
        self.net = DACG_IR(**MODEL_CONFIGS[model_name])
        self.l1 = nn.L1Loss()
        self._val_sums = None
        self._val_ssim_sums = None
        self._val_counts = None
        self.register_buffer("_best_psnr_value", torch.tensor(float("-inf")), persistent=True)
        self._val_media = {}

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
        residual = prediction.detach() - batch["lq"]
        learning_rate = self.optimizers().param_groups[0]["lr"]
        metrics = {
            "train/loss": loss,
            "train/l1": l1,
            "train/fft": fft,
            "train/learning_rate": learning_rate,
            "train/samples": float(batch["lq"].shape[0]),
            "diagnostics/residual_mean": residual.mean(),
            "diagnostics/residual_std": residual.std(),
            "diagnostics/residual_min": residual.min(),
            "diagnostics/residual_max": residual.max(),
            "diagnostics/residual_negative_fraction": (residual < 0).float().mean(),
            "diagnostics/residual_positive_fraction": (residual > 0).float().mean(),
            "diagnostics/residual_near_zero_fraction": (residual.abs() <= 1e-6).float().mean(),
            "diagnostics/prediction_below_zero_fraction": (prediction.detach() < 0).float().mean(),
            "diagnostics/prediction_above_one_fraction": (prediction.detach() > 1).float().mean(),
        }
        if torch.cuda.is_available():
            metrics["system/gpu_memory_allocated_gib"] = torch.cuda.memory_allocated() / 2**30
            metrics["system/gpu_memory_reserved_gib"] = torch.cuda.memory_reserved() / 2**30
        self.log_dict(metrics, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=batch["lq"].shape[0])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False, sync_dist=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        norms = [parameter.grad.detach().float().norm(2) for parameter in self.parameters() if parameter.grad is not None]
        if norms:
            grad_norm = torch.stack(norms).norm(2)
            self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False, sync_dist=True)

    def on_validation_epoch_start(self):
        from .protocol import DEGRADATIONS
        self._val_sums = torch.zeros(len(DEGRADATIONS), device=self.device, dtype=torch.float64)
        self._val_ssim_sums = torch.zeros(len(DEGRADATIONS), device=self.device, dtype=torch.float64)
        self._val_counts = torch.zeros(len(DEGRADATIONS), device=self.device, dtype=torch.float64)
        self._val_media = {}

    def validation_step(self, batch, batch_idx):
        from .protocol import DEGRADATIONS
        prediction = self(batch["lq"]).clamp(0, 1)
        mse = (prediction - batch["gt"]).square().mean().clamp_min(1e-12)
        psnr = -10.0 * torch.log10(mse)
        from aio3_runner.metrics import rgb_ssim
        ssim = rgb_ssim(prediction, batch["gt"])
        index = DEGRADATIONS.index(batch["degradation"][0])
        self._val_sums[index] += psnr.double()
        self._val_ssim_sums[index] += ssim
        self._val_counts[index] += 1
        should_log_media = (
            not self.trainer.sanity_checking
            and ((self.current_epoch + 1) % self.hparams.wandb_media_every == 0 or self.current_epoch + 1 == self.trainer.max_epochs)
        )
        if should_log_media and batch["degradation"][0] not in self._val_media:
            self._val_media[batch["degradation"][0]] = {
                "scene_id": batch["scene_id"][0],
                "input": batch["lq"][0].detach().float().cpu(),
                "prediction": prediction[0].detach().float().cpu(),
                "target": batch["gt"][0].detach().float().cpu(),
                "psnr": float(psnr),
                "ssim": float(ssim),
            }

    def on_validation_epoch_end(self):
        sums = self.trainer.strategy.reduce(self._val_sums, reduce_op="sum")
        ssim_sums = self.trainer.strategy.reduce(self._val_ssim_sums, reduce_op="sum")
        counts = self.trainer.strategy.reduce(self._val_counts, reduce_op="sum")
        per_psnr = sums / counts.clamp_min(1)
        per_ssim = ssim_sums / counts.clamp_min(1)
        macro_psnr, macro_ssim = per_psnr.mean().float(), per_ssim.mean().float()
        metrics = {"val/macro/psnr": macro_psnr, "val/macro/ssim": macro_ssim}
        for index, degradation in enumerate(DEGRADATIONS):
            metrics[f"val/{degradation}/psnr"] = per_psnr[index].float()
            metrics[f"val/{degradation}/ssim"] = per_ssim[index].float()
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=False)
        self.log("val_macro_psnr", macro_psnr, prog_bar=True, logger=False, sync_dist=False)
        if self.trainer.is_global_zero and float(macro_psnr) > float(self._best_psnr_value):
            self._best_psnr_value.fill_(float(macro_psnr))
            if hasattr(self.logger, "update_best"):
                self.logger.update_best(step=self.global_step, psnr=float(macro_psnr), ssim=float(macro_ssim))
        if self.trainer.is_global_zero and self._val_media and hasattr(self.logger, "log_table"):
            import wandb

            columns = ["global_step", "degradation", "scene_id", "input", "prediction", "target", "absolute_error", "signed_residual", "psnr", "ssim", "residual_mean", "residual_negative_fraction"]
            table = wandb.Table(columns=columns)
            for degradation in DEGRADATIONS:
                if degradation not in self._val_media:
                    continue
                item = self._val_media[degradation]
                input_tensor, prediction, target = item["input"], item["prediction"], item["target"]
                residual = prediction - input_tensor
                def image_array(tensor):
                    return tensor.clamp(0, 1).permute(1, 2, 0).numpy()
                absolute_error = (prediction - target).abs().clamp(0, 0.25) / 0.25
                signed_display = (residual.clamp(-0.25, 0.25) + 0.25) / 0.5
                table.add_data(
                    self.global_step, degradation, item["scene_id"],
                    wandb.Image(image_array(input_tensor)), wandb.Image(image_array(prediction)),
                    wandb.Image(image_array(target)), wandb.Image(image_array(absolute_error)),
                    wandb.Image(image_array(signed_display)), item["psnr"], item["ssim"],
                    float(residual.mean()), float((residual < 0).float().mean()),
                )
            self.logger.log_table(key="val/fixed_samples", table=table, step=self.global_step)

    def on_save_checkpoint(self, checkpoint):
        if hasattr(self.logger, "_run_id"):
            checkpoint["wandb_run_id"] = self.logger._run_id
            checkpoint["wandb_project"] = "cdd11-restoration"

    def on_load_checkpoint(self, checkpoint):
        saved_id = checkpoint.get("wandb_run_id")
        active_id = getattr(self.logger, "_run_id", saved_id)
        if saved_id and active_id and saved_id != active_id:
            raise RuntimeError(f"Checkpoint W&B run ID {saved_id} does not match active run {active_id}")

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
