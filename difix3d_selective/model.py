"""Difix3D model adapted to decode only the coarse primary view."""

from __future__ import annotations

import errno
import os
import time
import uuid
import warnings
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDPMScheduler
from einops import rearrange, repeat
from peft import LoraConfig
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel

from .daem_lite import (
    GatedDetailSkip,
    TextConditionProjection,
    TextFiLMDeltaGate,
    masked_mean_pool,
)
from .main_view import select_main_view, select_main_view_skips

SD_TURBO = "stabilityai/sd-turbo"


def make_1step_scheduler() -> DDPMScheduler:
    scheduler = DDPMScheduler.from_pretrained(SD_TURBO, subfolder="scheduler")
    scheduler.set_timesteps(1)
    return scheduler


def vae_encoder_forward(self, sample):
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


def vae_decoder_forward(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = self.mid_block(sample, latent_embeds).to(upscale_dtype)
    skip_convs = [
        self.skip_conv_1,
        self.skip_conv_2,
        self.skip_conv_3,
        self.skip_conv_4,
    ]
    projected_text = None
    if self.detail_text_enabled and self.incoming_detail_condition is not None:
        projected_text = self.detail_text_projection(self.incoming_detail_condition)
    analysis_text = None
    if self.incoming_detail_analysis_conditions is not None:
        analysis_text = tuple(
            self.detail_text_projection(condition)
            for condition in self.incoming_detail_analysis_conditions
        )
        self.last_detail_analysis = []
    for index, up_block in enumerate(self.up_blocks):
        skip = skip_convs[index](self.incoming_skip_acts[::-1][index] * self.gamma)
        if self.detail_enabled:
            if analysis_text is not None:
                base_logits = self.detail_blocks[index].gate_logits(sample, skip)
                first_delta = self.detail_text_gates[index](
                    sample, skip, analysis_text[0]
                )
                second_delta = self.detail_text_gates[index](
                    sample, skip, analysis_text[1]
                )
                base_gate = 2.0 * torch.sigmoid(base_logits) - 1.0
                first_gate = 2.0 * torch.sigmoid(base_logits + first_delta) - 1.0
                second_gate = 2.0 * torch.sigmoid(base_logits + second_delta) - 1.0
                difference = (first_gate - second_gate).abs()
                self.last_detail_analysis.append(
                    {
                        "base_gate": base_gate.detach().mean(dim=1),
                        "correct_gate": first_gate.detach().mean(dim=1),
                        "swapped_gate": second_gate.detach().mean(dim=1),
                        "gate_difference": difference.detach().mean(dim=1),
                        "delta_logits": first_delta.detach().mean(dim=1),
                        "gate_difference_mean": difference.detach().mean(),
                    }
                )
            delta_logits = None
            if projected_text is not None and self.incoming_text_condition_scale != 0:
                delta_logits = self.detail_text_gates[index](
                    sample, skip, projected_text
                )
            skip = self.detail_blocks[index](
                decoder_feature=sample,
                projected_skip=skip,
                prompt_condition=(
                    self.incoming_detail_condition
                    if self.detail_gate_use_prompt
                    else None
                ),
                delta_logits=delta_logits,
                text_condition_scale=self.incoming_text_condition_scale,
            )
        sample = up_block(sample + skip, latent_embeds)
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    return self.conv_out(self.conv_act(sample))


class SelectiveDifix(torch.nn.Module):
    def __init__(
        self,
        lora_rank_vae: int = 4,
        timestep: int = 199,
        *,
        detail_enabled: bool = False,
        detail_num_blocks: int = 1,
        detail_gate_reduction: int = 4,
        detail_alpha_init: float = 0.1,
        detail_gate_use_prompt: bool = False,
        detail_prompt_proj_dim: int = 32,
        detail_text_mode: str = "none",
        detail_text_proj_dim: int = 128,
        detail_film_hidden_ratio: int = 4,
        detail_text_condition_scale: float = 1.0,
    ):
        super().__init__()
        if detail_text_mode not in ("none", "full", "constant"):
            raise ValueError(f"Unknown detail text mode: {detail_text_mode}")
        if detail_text_mode != "none" and not detail_enabled:
            raise ValueError("detail text mode requires detail_enabled")
        if detail_gate_use_prompt and detail_text_mode != "none":
            raise ValueError(
                "--detail-gate-use-prompt cannot be combined with --detail-text-mode"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(SD_TURBO, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            SD_TURBO, subfolder="text_encoder"
        )
        self.scheduler = make_1step_scheduler()

        vae = AutoencoderKL.from_pretrained(SD_TURBO, subfolder="vae")
        vae.encoder.forward = vae_encoder_forward.__get__(
            vae.encoder, vae.encoder.__class__
        )
        vae.decoder.forward = vae_decoder_forward.__get__(
            vae.decoder, vae.decoder.__class__
        )
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, 1, bias=False)
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, 1, bias=False)
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, 1, bias=False)
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, 1, bias=False)
        vae.decoder.gamma = 1
        for skip_conv in (
            vae.decoder.skip_conv_1,
            vae.decoder.skip_conv_2,
            vae.decoder.skip_conv_3,
            vae.decoder.skip_conv_4,
        ):
            torch.nn.init.constant_(skip_conv.weight, 1e-5)

        target_suffixes = [
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
        ]
        self.target_modules_vae = [
            name
            for name, _ in vae.named_modules()
            if "decoder" in name
            and any(name.endswith(suffix) for suffix in target_suffixes)
        ]
        self.lora_rank_vae = lora_rank_vae
        vae.add_adapter(
            LoraConfig(
                r=lora_rank_vae,
                init_lora_weights="gaussian",
                target_modules=self.target_modules_vae,
            ),
            adapter_name="vae_skip",
        )

        detail_blocks = []
        if detail_enabled:
            prompt_dim = (
                int(self.text_encoder.config.hidden_size)
                if detail_gate_use_prompt
                else None
            )
            detail_blocks = [
                GatedDetailSkip(
                    channels,
                    num_naf_blocks=detail_num_blocks,
                    gate_reduction=detail_gate_reduction,
                    alpha_init=detail_alpha_init,
                    prompt_dim=prompt_dim,
                    prompt_proj_dim=detail_prompt_proj_dim,
                )
                for channels in (512, 512, 512, 256)
            ]
        vae.decoder.detail_blocks = torch.nn.ModuleList(detail_blocks)
        vae.decoder.detail_enabled = bool(detail_enabled)
        text_hidden_size = int(self.text_encoder.config.hidden_size)
        if detail_enabled and detail_text_mode != "none":
            vae.decoder.detail_text_projection = TextConditionProjection(
                text_hidden_size, detail_text_proj_dim
            )
            vae.decoder.detail_text_gates = torch.nn.ModuleList(
                TextFiLMDeltaGate(
                    channels,
                    detail_text_proj_dim,
                    hidden_ratio=detail_film_hidden_ratio,
                )
                for channels in (512, 512, 512, 256)
            )
        else:
            vae.decoder.detail_text_projection = None
            vae.decoder.detail_text_gates = torch.nn.ModuleList()
        vae.decoder.detail_text_enabled = bool(
            detail_enabled and detail_text_mode != "none"
        )
        vae.decoder.detail_gate_use_prompt = bool(detail_gate_use_prompt)
        vae.decoder.incoming_detail_condition = None
        vae.decoder.incoming_detail_analysis_conditions = None
        vae.decoder.last_detail_analysis = []
        vae.decoder.incoming_text_condition_scale = float(detail_text_condition_scale)
        self.detail_enabled = bool(detail_enabled)
        self.detail_num_blocks = int(detail_num_blocks)
        self.detail_gate_reduction = int(detail_gate_reduction)
        self.detail_alpha_init = float(detail_alpha_init)
        self.detail_gate_use_prompt = bool(detail_gate_use_prompt)
        self.detail_prompt_proj_dim = int(detail_prompt_proj_dim)
        self.detail_text_mode = detail_text_mode
        self.detail_text_proj_dim = int(detail_text_proj_dim)
        self.detail_film_hidden_ratio = int(detail_film_hidden_ratio)
        self.detail_text_condition_scale = float(detail_text_condition_scale)
        self._last_detail_condition = None

        constant_condition = None
        if detail_text_mode == "constant":
            constant_tokens = self.tokenizer(
                "",
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                constant_hidden = self.text_encoder(
                    constant_tokens.input_ids, return_dict=True
                ).last_hidden_state
            constant_condition = masked_mean_pool(
                constant_hidden, constant_tokens.attention_mask
            )
        self.register_buffer(
            "_constant_detail_condition", constant_condition, persistent=False
        )

        from .mv_unet import UNet2DConditionModel

        self.unet = UNet2DConditionModel.from_pretrained(SD_TURBO, subfolder="unet")
        self.vae = vae
        self.register_buffer("timesteps", torch.tensor([timestep], dtype=torch.long))
        self.text_encoder.requires_grad_(False)
        self.train_scope = "all"
        self.set_train("all")

    def set_train(self, train_scope: str | None = None) -> None:
        train_scope = self.train_scope if train_scope is None else train_scope
        if train_scope not in (
            "all",
            "detail",
            "detail+vae",
            "film",
            "film+detail",
        ):
            raise ValueError(f"Unknown train scope: {train_scope}")
        if train_scope != "all" and not self.detail_enabled:
            raise ValueError(f"train scope {train_scope!r} requires detail_enabled")
        if train_scope in ("film", "film+detail") and not self.text_film_enabled:
            raise ValueError(f"train scope {train_scope!r} requires detail text mode")
        self.train_scope = train_scope

        train_all = train_scope == "all"
        train_base_detail = train_scope in (
            "all",
            "detail",
            "detail+vae",
            "film+detail",
        )
        train_text = self.text_film_enabled and train_scope in (
            "all",
            "film",
            "film+detail",
        )
        self.unet.train(train_all).requires_grad_(train_all)
        self.vae.train().requires_grad_(False)
        if train_scope in ("all", "detail+vae"):
            for parameter in self.vae_adaptation_parameters():
                parameter.requires_grad = True
        if self.detail_enabled:
            self.vae.decoder.detail_blocks.train(train_base_detail).requires_grad_(
                train_base_detail
            )
        if self.text_film_enabled:
            self.vae.decoder.detail_text_projection.train(train_text).requires_grad_(
                train_text
            )
            self.vae.decoder.detail_text_gates.train(train_text).requires_grad_(
                train_text
            )
        self.text_encoder.eval().requires_grad_(False)

    def set_eval(self) -> None:
        self.unet.eval().requires_grad_(False)
        self.vae.eval().requires_grad_(False)
        self.text_encoder.eval().requires_grad_(False)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def detail_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.vae.decoder.detail_blocks.parameters())

    @property
    def text_film_enabled(self) -> bool:
        return self.detail_enabled and self.detail_text_mode != "none"

    def text_film_parameters(self) -> list[torch.nn.Parameter]:
        if not self.text_film_enabled:
            return []
        return list(self.vae.decoder.detail_text_projection.parameters()) + list(
            self.vae.decoder.detail_text_gates.parameters()
        )

    def detail_alpha_parameters(self) -> list[torch.nn.Parameter]:
        return [block.alpha for block in self.vae.decoder.detail_blocks]

    def naf_and_base_gate_parameters(self) -> list[torch.nn.Parameter]:
        parameters = []
        alpha_ids = {id(parameter) for parameter in self.detail_alpha_parameters()}
        for parameter in self.detail_parameters():
            if id(parameter) not in alpha_ids:
                parameters.append(parameter)
        return parameters

    def vae_adaptation_parameters(self) -> list[torch.nn.Parameter]:
        return [
            parameter
            for name, parameter in self.vae.named_parameters()
            if ("lora" in name and "vae_skip" in name)
            or name.startswith("decoder.skip_conv_")
        ]

    def parameter_counts(self) -> dict[str, int]:
        detail = sum(parameter.numel() for parameter in self.detail_parameters())
        vae_adaptation = sum(
            parameter.numel() for parameter in self.vae_adaptation_parameters()
        )
        text_film = sum(parameter.numel() for parameter in self.text_film_parameters())
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "total": total,
            "baseline_total": total - detail - text_film,
            "detail": detail,
            "text_film": text_film,
            "vae_adaptation": vae_adaptation,
            "unet": sum(parameter.numel() for parameter in self.unet.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
        }

    def detail_diagnostics(self, prefix: str = "train/detail") -> dict[str, float]:
        if not self.detail_enabled:
            return {}
        values = {}
        for index, block in enumerate(self.vae.decoder.detail_blocks):
            if block.last_gate_mean is None:
                continue
            layer_prefix = f"{prefix}/l{index}"
            values.update(
                {
                    f"{layer_prefix}/gate_mean": float(block.last_gate_mean),
                    f"{layer_prefix}/gate_std": float(block.last_gate_std),
                    f"{layer_prefix}/gate_min": float(block.last_gate_min),
                    f"{layer_prefix}/gate_max": float(block.last_gate_max),
                    f"{layer_prefix}/base_gate_mean": float(block.last_base_gate_mean),
                    f"{layer_prefix}/gate_change_abs_mean": float(
                        block.last_gate_change_abs_mean
                    ),
                    f"{layer_prefix}/alpha": float(block.alpha.detach()),
                    f"{layer_prefix}/residual_ratio": float(block.last_residual_ratio),
                }
            )
        return values

    def text_film_diagnostics(
        self, prefix: str = "train/text_film"
    ) -> dict[str, float]:
        if not self.text_film_enabled:
            return {}
        values = {}
        for index, gate in enumerate(self.vae.decoder.detail_text_gates):
            if gate.last_delta_logits_mean is None:
                continue
            layer_prefix = f"{prefix}/l{index}"
            values.update(
                {
                    f"{layer_prefix}/delta_logits_mean": float(
                        gate.last_delta_logits_mean
                    ),
                    f"{layer_prefix}/delta_logits_abs_mean": float(
                        gate.last_delta_logits_abs_mean
                    ),
                    f"{layer_prefix}/delta_logits_std": float(
                        gate.last_delta_logits_std
                    ),
                    f"{layer_prefix}/gamma_abs_mean": float(gate.last_gamma_abs_mean),
                    f"{layer_prefix}/beta_abs_mean": float(gate.last_beta_abs_mean),
                }
            )
        return values

    def text_film_gradient_norms(
        self, prefix: str = "train/text_film"
    ) -> dict[str, float]:
        if not self.text_film_enabled:
            return {}
        values = {}
        modules = [
            ("shared", self.vae.decoder.detail_text_projection),
            *[
                (f"l{index}", gate)
                for index, gate in enumerate(self.vae.decoder.detail_text_gates)
            ],
        ]
        for name, module in modules:
            squared = sum(
                parameter.grad.detach().float().square().sum()
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            if not isinstance(squared, int):
                values[f"{prefix}/{name}/text_grad_norm"] = float(squared.sqrt())
        return values

    def detail_config(self) -> dict[str, bool | int]:
        return {
            "enabled": self.detail_enabled,
            "num_naf_blocks": self.detail_num_blocks,
            "gate_reduction": self.detail_gate_reduction,
            "prompt_condition": self.detail_gate_use_prompt,
            "prompt_proj_dim": self.detail_prompt_proj_dim,
        }

    def detail_state_dict(self) -> dict[str, torch.Tensor]:
        return self.vae.decoder.detail_blocks.state_dict()

    def text_film_config(self) -> dict[str, bool | float | int | str]:
        return {
            "enabled": self.text_film_enabled,
            "mode": self.detail_text_mode,
            "pooling": "masked_mean",
            "clip_hidden_dim": int(self.text_encoder.config.hidden_size),
            "text_proj_dim": self.detail_text_proj_dim,
            "film_hidden_ratio": self.detail_film_hidden_ratio,
            "condition_scale": self.detail_text_condition_scale,
        }

    def text_film_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        if not self.text_film_enabled:
            return {}
        return {
            "projection": self.vae.decoder.detail_text_projection.state_dict(),
            "gates": self.vae.decoder.detail_text_gates.state_dict(),
        }

    def _prompt_embeddings(
        self,
        images: torch.Tensor,
        *,
        prompt: list[str] | str | None,
        prompt_tokens: torch.Tensor | None,
        prompt_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (prompt is None) == (prompt_tokens is None):
            raise ValueError("Provide exactly one of prompt or prompt_tokens")
        if prompt is not None:
            tokenized = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            tokens = tokenized.input_ids.to(images.device)
            attention_mask = tokenized.attention_mask.to(images.device)
        else:
            tokens = prompt_tokens.to(images.device)
            attention_mask = (
                self._attention_mask(tokens)
                if prompt_attention_mask is None
                else prompt_attention_mask.to(images.device)
            )
        if not self.detail_gate_use_prompt and not self.text_film_enabled:
            self._last_detail_condition = None
            return self.text_encoder(tokens)[0]
        text_output = self.text_encoder(tokens, return_dict=True)
        if self.detail_gate_use_prompt:
            self._last_detail_condition = text_output.pooler_output
        elif self.detail_text_mode == "constant":
            self._last_detail_condition = self._constant_detail_condition.expand(
                tokens.shape[0], -1
            )
        else:
            self._last_detail_condition = masked_mean_pool(
                text_output.last_hidden_state, attention_mask
            )
        return text_output.last_hidden_state

    def _attention_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)[None, :]
        first_eos = (
            (tokens == self.tokenizer.eos_token_id)
            .to(torch.int64)
            .argmax(dim=1, keepdim=True)
        )
        return (positions <= first_eos).to(torch.long)

    def detail_condition(
        self,
        images: torch.Tensor,
        *,
        prompt: list[str] | str | None = None,
        prompt_tokens: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Encode a decoder-only condition without changing the UNet prompt."""

        if not self.detail_gate_use_prompt and not self.text_film_enabled:
            return None
        if (prompt is None) == (prompt_tokens is None):
            raise ValueError("Provide exactly one of prompt or prompt_tokens")
        if prompt is not None:
            tokenized = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            tokens = tokenized.input_ids.to(images.device)
            attention_mask = tokenized.attention_mask.to(images.device)
        else:
            tokens = prompt_tokens.to(images.device)
            attention_mask = (
                self._attention_mask(tokens)
                if prompt_attention_mask is None
                else prompt_attention_mask.to(images.device)
            )
        if self.detail_text_mode == "constant":
            return self._constant_detail_condition.expand(tokens.shape[0], -1)
        text_output = self.text_encoder(tokens, return_dict=True)
        if self.detail_gate_use_prompt:
            return text_output.pooler_output
        return masked_mean_pool(text_output.last_hidden_state, attention_mask)

    def denoised_main_latent(
        self,
        images: torch.Tensor,
        *,
        prompt: list[str] | str | None = None,
        prompt_tokens: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return the denoised primary-view latent and its encoder skips."""

        batch_size, num_views = images.shape[:2]
        if num_views != 2:
            raise ValueError(f"Selective Difix requires two views, got {num_views}")
        flat_images = rearrange(images, "b v c h w -> (b v) c h w")
        latents = self.vae.encode(flat_images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        main_skips = select_main_view_skips(
            self.vae.encoder.current_down_blocks, batch_size, num_views
        )

        text = self._prompt_embeddings(
            images,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            prompt_attention_mask=prompt_attention_mask,
        )
        text = repeat(text, "b n c -> (b v) n c", v=num_views)
        current_timesteps = self.timesteps if timesteps is None else timesteps
        model_prediction = self.unet(
            latents,
            current_timesteps,
            encoder_hidden_states=text,
        ).sample
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(latents.device)
        denoised = self.scheduler.step(
            model_prediction,
            current_timesteps,
            latents,
            return_dict=True,
        ).prev_sample
        return select_main_view(denoised, batch_size, num_views), main_skips

    def decode_main_latent(
        self,
        latent: torch.Tensor,
        encoder_skips: list[torch.Tensor],
        *,
        prompt_condition: torch.Tensor | None = None,
        text_condition_scale: float | None = None,
    ) -> torch.Tensor:
        """Decode one primary-view latent using explicitly captured VAE skips."""

        self.vae.decoder.incoming_skip_acts = encoder_skips
        self.vae.decoder.incoming_detail_condition = prompt_condition
        self.vae.decoder.incoming_text_condition_scale = (
            self.detail_text_condition_scale
            if text_condition_scale is None
            else float(text_condition_scale)
        )
        output = self.vae.decode(latent / self.vae.config.scaling_factor).sample
        return output.clamp(-1, 1)

    def analyze_detail_gates(
        self,
        latent: torch.Tensor,
        encoder_skips: list[torch.Tensor],
        correct_condition: torch.Tensor,
        swapped_condition: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Compare two text gates against the same no-text decoder states."""

        previous_condition = self.vae.decoder.incoming_detail_condition
        previous_scale = self.vae.decoder.incoming_text_condition_scale
        previous_analysis = self.vae.decoder.incoming_detail_analysis_conditions
        self.vae.decoder.incoming_detail_analysis_conditions = (
            correct_condition,
            swapped_condition,
        )
        try:
            self.decode_main_latent(
                latent,
                encoder_skips,
                prompt_condition=correct_condition,
                text_condition_scale=0.0,
            )
            return [
                {key: value.detach().cpu() for key, value in layer.items()}
                for layer in self.vae.decoder.last_detail_analysis
            ]
        finally:
            self.vae.decoder.incoming_detail_condition = previous_condition
            self.vae.decoder.incoming_text_condition_scale = previous_scale
            self.vae.decoder.incoming_detail_analysis_conditions = previous_analysis

    def cfg_latents(
        self,
        positive_images: torch.Tensor,
        negative_images: torch.Tensor,
        *,
        positive_prompt_tokens: torch.Tensor,
        negative_prompt_tokens: torch.Tensor,
        positive_prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        """Compute denoised latents and VAE skips for both complete CCDD modes."""

        z_positive, positive_skips = self.denoised_main_latent(
            positive_images,
            prompt_tokens=positive_prompt_tokens,
            prompt_attention_mask=positive_prompt_attention_mask,
            timesteps=timesteps,
        )
        z_negative, negative_skips = self.denoised_main_latent(
            negative_images,
            prompt_tokens=negative_prompt_tokens,
            prompt_attention_mask=negative_prompt_attention_mask,
            timesteps=timesteps,
        )
        return z_positive, z_negative, positive_skips, negative_skips

    def forward(
        self,
        images: torch.Tensor,
        *,
        prompt: list[str] | str | None = None,
        prompt_tokens: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        detail_prompt: list[str] | str | None = None,
        detail_prompt_tokens: torch.Tensor | None = None,
        detail_prompt_attention_mask: torch.Tensor | None = None,
        text_condition_scale: float | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        main_latent, main_skips = self.denoised_main_latent(
            images,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            prompt_attention_mask=prompt_attention_mask,
            timesteps=timesteps,
        )
        prompt_condition = self._last_detail_condition
        if detail_prompt is not None or detail_prompt_tokens is not None:
            prompt_condition = self.detail_condition(
                images,
                prompt=detail_prompt,
                prompt_tokens=detail_prompt_tokens,
                prompt_attention_mask=detail_prompt_attention_mask,
            )
        return self.decode_main_latent(
            main_latent,
            main_skips,
            prompt_condition=prompt_condition,
            text_condition_scale=text_condition_scale,
        )

    @torch.inference_mode()
    def sample(
        self,
        image: Image.Image,
        reference: Image.Image,
        prompt: str,
        resolution: int = 512,
    ) -> Image.Image:
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (resolution, resolution),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        images = torch.stack(
            [transform(image.convert("RGB")), transform(reference.convert("RGB"))]
        )
        images = images.unsqueeze(0).to(next(self.parameters()).device)
        output = self(images, prompt=[prompt])[0].float().cpu() * 0.5 + 0.5
        return transforms.ToPILImage()(output.clamp(0, 1))


def save_training_checkpoint(
    model: SelectiveDifix,
    optimizer,
    scheduler,
    path: str | Path,
    global_step: int,
    experiment_metadata: dict | None = None,
) -> None:
    checkpoint = {
        "global_step": global_step,
        "timestep": int(model.timesteps.item()),
        "vae_lora_target_modules": model.target_modules_vae,
        "rank_vae": model.lora_rank_vae,
        "state_dict_unet": model.unet.state_dict(),
        "state_dict_vae": {
            key: value
            for key, value in model.vae.state_dict().items()
            if "lora" in key or "skip_conv" in key
        },
        "state_dict_detail": model.detail_state_dict(),
        "detail_config": model.detail_config(),
        "state_dict_text_film": model.text_film_state_dict(),
        "text_film_config": model.text_film_config(),
        "text_mode": model.detail_text_mode,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
    }
    if experiment_metadata is not None:
        checkpoint["experiment_metadata"] = dict(experiment_metadata)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    torch.save(checkpoint, temporary)
    for attempt in range(3):
        try:
            os.replace(temporary, destination)
            break
        except OSError as error:
            if error.errno != errno.EIO or attempt == 2:
                raise
            time.sleep(1)


def _apply_model_checkpoint(
    model: SelectiveDifix,
    checkpoint: dict,
    *,
    initialize_detail_from_disabled: bool = False,
) -> int:
    checkpoint_rank = int(checkpoint.get("rank_vae", model.lora_rank_vae))
    if checkpoint_rank != model.lora_rank_vae:
        raise ValueError(
            f"Checkpoint VAE LoRA rank is {checkpoint_rank}, but the model was built "
            f"with rank {model.lora_rank_vae}"
        )
    if "timestep" in checkpoint and int(checkpoint["timestep"]) != int(
        model.timesteps.item()
    ):
        raise ValueError(
            f"Checkpoint timestep is {checkpoint['timestep']}, but the model was built "
            f"with timestep {int(model.timesteps.item())}"
        )
    model.unet.load_state_dict(checkpoint["state_dict_unet"], strict=True)
    vae_state = model.vae.state_dict()
    vae_state.update(checkpoint["state_dict_vae"])
    model.vae.load_state_dict(vae_state, strict=True)
    saved_detail_config = checkpoint.get("detail_config")
    saved_detail_state = checkpoint.get("state_dict_detail")
    if saved_detail_config is None and saved_detail_state is None:
        if model.detail_enabled:
            warnings.warn(
                "Loading a legacy SelectiveDifix checkpoint; DAEM-lite modules "
                "remain freshly initialized.",
                stacklevel=2,
            )
    elif not isinstance(saved_detail_config, dict) or not isinstance(
        saved_detail_state, dict
    ):
        raise ValueError(
            "Checkpoint must contain both detail_config and state_dict_detail"
        )
    else:
        current_detail_config = model.detail_config()
        mismatches = {
            key: (saved_detail_config.get(key), current_detail_config[key])
            for key in current_detail_config
            if saved_detail_config.get(key) != current_detail_config[key]
        }
        fresh_detail = (
            initialize_detail_from_disabled
            and saved_detail_config.get("enabled") is False
            and model.detail_enabled
        )
        if mismatches and not fresh_detail:
            raise ValueError(
                f"Checkpoint DAEM-lite configuration mismatch: {mismatches}"
            )
        if not fresh_detail:
            model.vae.decoder.detail_blocks.load_state_dict(saved_detail_state, strict=True)

    saved_text_config = checkpoint.get("text_film_config")
    saved_text_state = checkpoint.get("state_dict_text_film")
    if saved_text_config is None and saved_text_state is None:
        if model.text_film_enabled:
            warnings.warn(
                "Loading a checkpoint without Text-FiLM state; the new delta gates "
                "remain zero-initialized.",
                stacklevel=2,
            )
    elif not isinstance(saved_text_config, dict) or not isinstance(
        saved_text_state, dict
    ):
        raise ValueError(
            "Checkpoint must contain both text_film_config and state_dict_text_film"
        )
    else:
        current_text_config = model.text_film_config()
        mismatches = {
            key: (saved_text_config.get(key), current_text_config[key])
            for key in current_text_config
            if saved_text_config.get(key) != current_text_config[key]
        }
        if mismatches:
            raise ValueError(
                f"Checkpoint Text-FiLM configuration mismatch: {mismatches}"
            )
        if model.text_film_enabled:
            if set(saved_text_state) != {"projection", "gates"}:
                raise ValueError("Checkpoint Text-FiLM state is incomplete")
            model.vae.decoder.detail_text_projection.load_state_dict(
                saved_text_state["projection"], strict=True
            )
            model.vae.decoder.detail_text_gates.load_state_dict(
                saved_text_state["gates"], strict=True
            )
        elif saved_text_state:
            raise ValueError("Checkpoint contains Text-FiLM weights for text mode none")
    return int(checkpoint["global_step"])


def load_model_checkpoint(
    model: SelectiveDifix,
    path: str | Path,
    *,
    expected_dataset: str | None = None,
    expected_seed: int | None = None,
    initialize_detail_from_disabled: bool = False,
) -> int:
    """Load model weights without optimizer state."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.loaded_checkpoint_metadata = dict(checkpoint.get("experiment_metadata") or {})
    if expected_dataset is not None or expected_seed is not None:
        metadata = checkpoint.get("experiment_metadata")
        if not isinstance(metadata, dict):
            raise ValueError(
                "Checkpoint lacks experiment_metadata required by the evaluation protocol"
            )
        if expected_dataset is not None and metadata.get("dataset") != expected_dataset:
            raise ValueError(
                f"Checkpoint dataset is {metadata.get('dataset')!r}, expected "
                f"{expected_dataset!r}"
            )
        if expected_seed is not None and int(metadata.get("seed", -1)) != expected_seed:
            raise ValueError(
                f"Checkpoint seed is {metadata.get('seed')!r}, expected {expected_seed}"
            )
    return _apply_model_checkpoint(
        model,
        checkpoint,
        initialize_detail_from_disabled=initialize_detail_from_disabled,
    )


def load_model_from_checkpoint(
    path: str | Path,
    *,
    expected_dataset: str | None = None,
    expected_seed: int | None = None,
) -> tuple[SelectiveDifix, int, dict]:
    """Construct and load a model using architecture metadata in its checkpoint."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("experiment_metadata")
    if expected_dataset is not None or expected_seed is not None:
        if not isinstance(metadata, dict):
            raise ValueError(
                "Checkpoint lacks experiment_metadata required by the evaluation protocol"
            )
        if expected_dataset is not None and metadata.get("dataset") != expected_dataset:
            raise ValueError(
                f"Checkpoint dataset is {metadata.get('dataset')!r}, expected "
                f"{expected_dataset!r}"
            )
        if expected_seed is not None and int(metadata.get("seed", -1)) != expected_seed:
            raise ValueError(
                f"Checkpoint seed is {metadata.get('seed')!r}, expected {expected_seed}"
            )

    saved_detail_config = checkpoint.get("detail_config")
    if saved_detail_config is None:
        detail_config = {
            "enabled": False,
            "num_naf_blocks": 1,
            "gate_reduction": 4,
            "prompt_condition": False,
            "prompt_proj_dim": 32,
        }
    elif not isinstance(saved_detail_config, dict):
        raise ValueError("Checkpoint detail_config must be a dictionary")
    else:
        detail_config = saved_detail_config

    saved_text_config = checkpoint.get("text_film_config")
    if saved_text_config is None:
        text_config = {
            "mode": "none",
            "text_proj_dim": 128,
            "film_hidden_ratio": 4,
            "condition_scale": 1.0,
        }
    elif not isinstance(saved_text_config, dict):
        raise ValueError("Checkpoint text_film_config must be a dictionary")
    else:
        text_config = saved_text_config

    model = SelectiveDifix(
        lora_rank_vae=int(checkpoint.get("rank_vae", 4)),
        timestep=int(checkpoint.get("timestep", 199)),
        detail_enabled=bool(detail_config.get("enabled", False)),
        detail_num_blocks=int(detail_config.get("num_naf_blocks", 1)),
        detail_gate_reduction=int(detail_config.get("gate_reduction", 4)),
        detail_gate_use_prompt=bool(detail_config.get("prompt_condition", False)),
        detail_prompt_proj_dim=int(detail_config.get("prompt_proj_dim", 32)),
        detail_text_mode=str(text_config.get("mode", "none")),
        detail_text_proj_dim=int(text_config.get("text_proj_dim", 128)),
        detail_film_hidden_ratio=int(text_config.get("film_hidden_ratio", 4)),
        detail_text_condition_scale=float(text_config.get("condition_scale", 1.0)),
    )
    global_step = _apply_model_checkpoint(model, checkpoint)
    return model, global_step, dict(metadata or {})


def load_training_checkpoint(
    model: SelectiveDifix,
    optimizer,
    scheduler,
    path: str | Path,
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("experiment_metadata") or {}
    saved_scope = metadata.get("train_scope")
    if saved_scope is not None and saved_scope != model.train_scope:
        raise ValueError(
            f"Cannot resume train_scope={saved_scope!r} as {model.train_scope!r}; "
            "use --init-checkpoint between training stages"
        )
    saved_optimizer = checkpoint.get("optimizer")
    current_names = None
    if hasattr(optimizer, "param_groups") and isinstance(saved_optimizer, dict):
        saved_names = [
            group.get("name") for group in saved_optimizer.get("param_groups", [])
        ]
        current_names = [group.get("name") for group in optimizer.param_groups]
        if (
            any(name is not None for name in saved_names)
            and saved_names != current_names
        ):
            raise ValueError(
                "Cannot resume with different optimizer parameter groups: "
                f"checkpoint={saved_names}, current={current_names}"
            )
    global_step = _apply_model_checkpoint(model, checkpoint)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if current_names is not None:
        for group, name in zip(optimizer.param_groups, current_names):
            group["name"] = name
    scheduler.load_state_dict(checkpoint["lr_scheduler"])
    return global_step
