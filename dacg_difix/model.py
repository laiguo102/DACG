"""DACG-conditioned, two-view Difix model.

The VAE skip path and two-view flow are adapted from NVIDIA Difix3D commit
``c76edc595586e16732c91ddee82f3a6d83a8a9cc``.  Difix3D is restricted to
non-commercial research/evaluation by its NVIDIA License (see the upstream
``LICENSE.txt`` at that commit).  The conditional
rank-mixing design follows S3Diff, with the local residual ``C = I + delta``
parameterization implemented in :mod:`dacg_difix.conditional_lora`.
"""

from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Any, Mapping

import torch
from torch import nn

from .conditional_lora import (
    ConditionalLoRAConditioner,
    LORA_MODES,
    assign_condition_matrices,
    clear_condition_matrices,
    install_conditional_lora_forward,
    iter_lora_modules,
    unet_stage_id,
    vae_decoder_stage_id,
)


SD_TURBO = "stabilityai/sd-turbo"
NUM_VIEWS = 2

UNET_LORA_TARGETS = (
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    "conv",
    "conv1",
    "conv2",
    "conv_shortcut",
    "conv_out",
    "proj_in",
    "proj_out",
    "ff.net.2",
    "ff.net.0.proj",
)

VAE_DECODER_LORA_SUFFIXES = (
    "conv1",
    "conv2",
    "conv_in",
    "conv_shortcut",
    "conv",
    "conv_out",
    "skip_conv_1",
    "skip_conv_2",
    "skip_conv_3",
    "skip_conv_4",
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
)

SKIP_CONV_NAMES = ("skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4")


def vae_encoder_forward(self, sample: torch.Tensor) -> torch.Tensor:
    """Difix encoder forward that records the four pre-downsample features."""

    sample = self.conv_in(sample)
    down_activations = []
    for down_block in self.down_blocks:
        down_activations.append(sample)
        sample = down_block(sample)
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = down_activations
    return sample


def vae_decoder_forward(self, sample: torch.Tensor, latent_embeds=None) -> torch.Tensor:
    """Difix decoder forward with four learned 1x1 encoder skips."""

    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = self.mid_block(sample, latent_embeds).to(upscale_dtype)
    skip_convs = [getattr(self, name) for name in SKIP_CONV_NAMES]
    for index, up_block in enumerate(self.up_blocks):
        skip = skip_convs[index](self.incoming_skip_acts[::-1][index] * self.gamma)
        sample = up_block(sample + skip, latent_embeds)
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    return self.conv_out(self.conv_act(sample))


def select_main_view(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Select view zero from a flattened ``[B*2, ...]`` tensor."""

    return tensor.reshape(batch_size, NUM_VIEWS, *tensor.shape[1:])[:, 0].contiguous()


def select_main_view_skips(activations: list[torch.Tensor], batch_size: int) -> list[torch.Tensor]:
    return [select_main_view(activation, batch_size) for activation in activations]


def _sample_from_output(output: Any) -> torch.Tensor:
    if hasattr(output, "sample"):
        return output.sample
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _latent_sample(encoded: Any) -> torch.Tensor:
    distribution = encoded.latent_dist
    return distribution.sample()


class DACGDifix(nn.Module):
    """Two-view SD-Turbo adapter conditioned by DACG degradation features.

    Inputs and outputs use the diffusion convention ``[-1, 1]``.  ``main`` is
    the DACG restoration and ``ref`` is the original degraded image.  Both
    views enter the VAE encoder and multi-view UNet, while only view zero's
    denoised latent and VAE skips are decoded.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str | Path = SD_TURBO,
        *,
        local_files_only: bool = False,
        lora_mode: str = "p-global-layer-id",
        lora_rank_unet: int = 32,
        lora_rank_vae: int = 16,
        timestep: int = 199,
        dam_encoder: nn.Module | None = None,
        tokenizer: Any | None = None,
        text_encoder: nn.Module | None = None,
        vae: nn.Module | None = None,
        unet: nn.Module | None = None,
        scheduler: Any | None = None,
    ) -> None:
        super().__init__()
        if lora_mode not in LORA_MODES:
            raise ValueError(f"Unknown LoRA conditioning mode: {lora_mode}")

        self.pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        self.lora_mode = lora_mode
        self.lora_rank_unet = int(lora_rank_unet)
        self.lora_rank_vae = int(lora_rank_vae)

        tokenizer, text_encoder, vae, unet, scheduler = self._load_missing_components(
            tokenizer,
            text_encoder,
            vae,
            unet,
            scheduler,
            local_files_only,
        )
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler
        if hasattr(self.scheduler, "set_timesteps"):
            self.scheduler.set_timesteps(1)
        self.dam_encoder = dam_encoder
        self.register_buffer("timesteps", torch.tensor([timestep], dtype=torch.long))

        self._configure_vae_skip_path()
        self._configure_lora_adapters()

        self.unet_conditioner = ConditionalLoRAConditioner(
            rank=self.lora_rank_unet,
            num_stages=10,
            mode=lora_mode,
        )
        self.vae_conditioner = ConditionalLoRAConditioner(
            rank=self.lora_rank_vae,
            num_stages=6,
            mode=lora_mode,
        )
        self.unet_lora_layers = install_conditional_lora_forward(self.unet)
        self.vae_lora_layers = install_conditional_lora_forward(self.vae)
        self._configure_trainable_parameters()

    def _load_missing_components(
        self,
        tokenizer,
        text_encoder,
        vae,
        unet,
        scheduler,
        local_files_only: bool,
    ):
        if all(component is not None for component in (tokenizer, text_encoder, vae, unet, scheduler)):
            return tokenizer, text_encoder, vae, unet, scheduler

        from diffusers import AutoencoderKL, DDPMScheduler
        from transformers import AutoTokenizer, CLIPTextModel

        from .mv_unet import UNet2DConditionModel

        load_kwargs = {"local_files_only": local_files_only}
        path = self.pretrained_model_name_or_path
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(path, subfolder="tokenizer", **load_kwargs)
        text_encoder = text_encoder or CLIPTextModel.from_pretrained(
            path, subfolder="text_encoder", **load_kwargs
        )
        vae = vae or AutoencoderKL.from_pretrained(path, subfolder="vae", **load_kwargs)
        unet = unet or UNet2DConditionModel.from_pretrained(path, subfolder="unet", **load_kwargs)
        scheduler = scheduler or DDPMScheduler.from_pretrained(path, subfolder="scheduler", **load_kwargs)
        scheduler.set_timesteps(1)
        return tokenizer, text_encoder, vae, unet, scheduler

    def _configure_vae_skip_path(self) -> None:
        encoder = self.vae.encoder
        decoder = self.vae.decoder
        if hasattr(encoder, "down_blocks") and hasattr(decoder, "up_blocks"):
            encoder.forward = MethodType(vae_encoder_forward, encoder)
            decoder.forward = MethodType(vae_decoder_forward, decoder)
            skip_specs = ((512, 512), (256, 512), (128, 512), (128, 256))
            for name, (in_channels, out_channels) in zip(SKIP_CONV_NAMES, skip_specs):
                if not hasattr(decoder, name):
                    convolution = nn.Conv2d(in_channels, out_channels, 1, bias=False)
                    nn.init.constant_(convolution.weight, 1e-5)
                    setattr(decoder, name, convolution)
            decoder.gamma = 1.0

    def _configure_lora_adapters(self) -> None:
        if not any(True for _ in iter_lora_modules(self.unet)) and hasattr(self.unet, "add_adapter"):
            from peft import LoraConfig

            self.unet.add_adapter(
                LoraConfig(
                    r=self.lora_rank_unet,
                    init_lora_weights="gaussian",
                    target_modules=list(UNET_LORA_TARGETS),
                )
            )

        self.target_modules_vae = [
            name
            for name, _ in self.vae.named_modules()
            if "decoder" in name and any(name.endswith(suffix) for suffix in VAE_DECODER_LORA_SUFFIXES)
        ]
        if not any(True for _ in iter_lora_modules(self.vae)) and hasattr(self.vae, "add_adapter"):
            from peft import LoraConfig

            self.vae.add_adapter(
                LoraConfig(
                    r=self.lora_rank_vae,
                    init_lora_weights="gaussian",
                    target_modules=self.target_modules_vae,
                ),
                adapter_name="vae_skip",
            )

        self.target_modules_unet = list(UNET_LORA_TARGETS)

    def _configure_trainable_parameters(self) -> None:
        self.requires_grad_(False)
        for name, parameter in self.unet.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(True)
        for name, parameter in self.vae.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(True)
        for name in SKIP_CONV_NAMES:
            skip_conv = getattr(self.vae.decoder, name, None)
            if skip_conv is not None:
                skip_conv.requires_grad_(True)
        self.unet_conditioner.requires_grad_(True)
        self.vae_conditioner.requires_grad_(True)
        if self.dam_encoder is not None:
            self.dam_encoder.eval().requires_grad_(False)
        self.text_encoder.eval().requires_grad_(False)
        if hasattr(self.vae, "encoder"):
            self.vae.encoder.eval().requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.text_encoder.eval()
        if self.dam_encoder is not None:
            self.dam_encoder.eval()
        if hasattr(self.vae, "encoder"):
            self.vae.encoder.eval()
        return self

    def set_train(self) -> None:
        self.train(True)

    def set_eval(self) -> None:
        self.eval()

    def _resolve_p_global(self, ref: torch.Tensor, p_global: torch.Tensor | None) -> torch.Tensor | None:
        if self.lora_mode == "static":
            return None
        if p_global is None:
            if self.dam_encoder is None:
                raise ValueError("p_global is required when no frozen DAM encoder is attached")
            with torch.no_grad():
                dam_output = self.dam_encoder(ref.mul(0.5).add(0.5))
            p_global = dam_output[-1] if isinstance(dam_output, (tuple, list)) else dam_output
        conditioner_parameter = next(self.unet_conditioner.parameters())
        return p_global.to(device=conditioner_parameter.device, dtype=conditioner_parameter.dtype)

    def _set_lora_conditions(self, p_global: torch.Tensor | None) -> None:
        if self.lora_mode == "static":
            clear_condition_matrices(self.unet)
            clear_condition_matrices(self.vae)
            return

        unet_matrices = self.unet_conditioner(p_global)
        vae_matrices = self.vae_conditioner(p_global)
        assign_condition_matrices(
            self.unet,
            unet_matrices,
            unet_stage_id,
        )
        assign_condition_matrices(self.vae, vae_matrices, vae_decoder_stage_id)

    def forward(
        self,
        main: torch.Tensor,
        ref: torch.Tensor,
        prompt_tokens: torch.Tensor,
        *,
        p_global: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the restored main view as ``[B, 3, H, W]`` in ``[-1, 1]``."""

        batch_size = main.shape[0]
        p_global = self._resolve_p_global(ref, p_global)
        self._set_lora_conditions(p_global)

        images = torch.stack((main, ref), dim=1)
        flat_images = images.reshape(batch_size * NUM_VIEWS, *images.shape[2:])
        encoded = self.vae.encode(flat_images)
        latents = _latent_sample(encoded) * self.vae.config.scaling_factor

        text_output = self.text_encoder(prompt_tokens.to(main.device))
        text_embedding = _sample_from_output(text_output)
        text_embedding = text_embedding.repeat_interleave(NUM_VIEWS, dim=0)

        timesteps = self.timesteps.to(main.device)
        model_prediction = _sample_from_output(
            self.unet(latents, timesteps, encoder_hidden_states=text_embedding)
        )
        if hasattr(self.scheduler, "alphas_cumprod"):
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(latents.device)
        denoised = self.scheduler.step(
            model_prediction,
            timesteps,
            latents,
            return_dict=True,
        ).prev_sample

        main_latent = select_main_view(denoised, batch_size)
        self.vae.decoder.incoming_skip_acts = select_main_view_skips(
            self.vae.encoder.current_down_blocks,
            batch_size,
        )
        decoded = self.vae.decode(main_latent / self.vae.config.scaling_factor)
        return _sample_from_output(decoded).clamp(-1, 1)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def adapter_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    @staticmethod
    def _is_adapter_state_key(key: str) -> bool:
        return (
            (key.startswith("unet.") and ".lora_" in key)
            or (
                key.startswith("vae.")
                and (".lora_" in key or any(f"decoder.{name}" in key for name in SKIP_CONV_NAMES))
            )
            or key.startswith("unet_conditioner.")
            or key.startswith("vae_conditioner.")
        )

    def adapter_state_dict(self) -> dict[str, Any]:
        state = {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if self._is_adapter_state_key(key)
        }
        return {
            "format": "dacg-difix-adapter-v1",
            "config": {
                "pretrained_model_name_or_path": self.pretrained_model_name_or_path,
                "lora_mode": self.lora_mode,
                "lora_rank_unet": self.lora_rank_unet,
                "lora_rank_vae": self.lora_rank_vae,
                "timestep": int(self.timesteps.item()),
                "target_modules_unet": self.target_modules_unet,
                "target_modules_vae": self.target_modules_vae,
            },
            "state_dict": state,
        }

    def load_adapter_state_dict(self, adapter_state: Mapping[str, Any], strict: bool = True) -> None:
        saved = adapter_state.get("state_dict", adapter_state)
        current = self.state_dict()
        unexpected = [key for key in saved if key not in current]
        missing = [
            key for key in current if self._is_adapter_state_key(key) and key not in saved
        ]
        if strict and (unexpected or missing):
            raise RuntimeError(f"Adapter state mismatch; missing={missing}, unexpected={unexpected}")
        for key, value in saved.items():
            if key in current:
                current[key] = value
        self.load_state_dict(current, strict=True)


DACGDifixModel = DACGDifix


def export_adapter_state(model: DACGDifix) -> dict[str, Any]:
    return model.adapter_state_dict()


def load_adapter_state(
    model: DACGDifix,
    adapter_state: Mapping[str, Any],
    strict: bool = True,
) -> DACGDifix:
    model.load_adapter_state_dict(adapter_state, strict=strict)
    return model


__all__ = [
    "DACGDifix",
    "DACGDifixModel",
    "export_adapter_state",
    "load_adapter_state",
    "select_main_view",
    "select_main_view_skips",
    "vae_decoder_forward",
    "vae_encoder_forward",
]
