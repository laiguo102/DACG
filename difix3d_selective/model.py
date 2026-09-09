"""Difix3D model adapted to decode only the coarse primary view."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDPMScheduler
from einops import rearrange, repeat
from peft import LoraConfig
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel

from .daem_lite import GatedDetailSkip
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
    for index, up_block in enumerate(self.up_blocks):
        skip = skip_convs[index](self.incoming_skip_acts[::-1][index] * self.gamma)
        if self.detail_enabled:
            skip = self.detail_blocks[index](
                decoder_feature=sample,
                projected_skip=skip,
                prompt_condition=self.incoming_detail_condition,
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
    ):
        super().__init__()
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
        vae.decoder.incoming_detail_condition = None
        self.detail_enabled = bool(detail_enabled)
        self.detail_num_blocks = int(detail_num_blocks)
        self.detail_gate_reduction = int(detail_gate_reduction)
        self.detail_alpha_init = float(detail_alpha_init)
        self.detail_gate_use_prompt = bool(detail_gate_use_prompt)
        self.detail_prompt_proj_dim = int(detail_prompt_proj_dim)
        self._last_detail_condition = None

        from .mv_unet import UNet2DConditionModel

        self.unet = UNet2DConditionModel.from_pretrained(SD_TURBO, subfolder="unet")
        self.vae = vae
        self.register_buffer("timesteps", torch.tensor([timestep], dtype=torch.long))
        self.text_encoder.requires_grad_(False)
        self.train_scope = "all"
        self.set_train("all")

    def set_train(self, train_scope: str | None = None) -> None:
        train_scope = self.train_scope if train_scope is None else train_scope
        if train_scope not in ("all", "detail", "detail+vae"):
            raise ValueError(f"Unknown train scope: {train_scope}")
        if train_scope != "all" and not self.detail_enabled:
            raise ValueError(f"train scope {train_scope!r} requires detail_enabled")
        self.train_scope = train_scope

        self.unet.train(train_scope == "all").requires_grad_(train_scope == "all")
        self.vae.train().requires_grad_(False)
        if train_scope in ("all", "detail+vae"):
            for parameter in self.vae_adaptation_parameters():
                parameter.requires_grad = True
        if self.detail_enabled:
            self.vae.decoder.detail_blocks.requires_grad_(True)
        self.text_encoder.eval().requires_grad_(False)

    def set_eval(self) -> None:
        self.unet.eval().requires_grad_(False)
        self.vae.eval().requires_grad_(False)
        self.text_encoder.eval().requires_grad_(False)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def detail_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.vae.decoder.detail_blocks.parameters())

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
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "total": total,
            "baseline_total": total - detail,
            "detail": detail,
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
                    f"{layer_prefix}/alpha": float(block.alpha.detach()),
                    f"{layer_prefix}/residual_ratio": float(
                        block.last_residual_ratio
                    ),
                }
            )
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

    def _prompt_embeddings(
        self,
        images: torch.Tensor,
        *,
        prompt: list[str] | str | None,
        prompt_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        if (prompt is None) == (prompt_tokens is None):
            raise ValueError("Provide exactly one of prompt or prompt_tokens")
        if prompt is not None:
            tokens = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(images.device)
        else:
            tokens = prompt_tokens.to(images.device)
        if not self.detail_gate_use_prompt:
            self._last_detail_condition = None
            return self.text_encoder(tokens)[0]
        text_output = self.text_encoder(tokens, return_dict=True)
        self._last_detail_condition = text_output.pooler_output
        return text_output.last_hidden_state

    def denoised_main_latent(
        self,
        images: torch.Tensor,
        *,
        prompt: list[str] | str | None = None,
        prompt_tokens: torch.Tensor | None = None,
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
            images, prompt=prompt, prompt_tokens=prompt_tokens
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
    ) -> torch.Tensor:
        """Decode one primary-view latent using explicitly captured VAE skips."""

        self.vae.decoder.incoming_skip_acts = encoder_skips
        self.vae.decoder.incoming_detail_condition = prompt_condition
        output = self.vae.decode(latent / self.vae.config.scaling_factor).sample
        return output.clamp(-1, 1)

    def cfg_latents(
        self,
        positive_images: torch.Tensor,
        negative_images: torch.Tensor,
        *,
        positive_prompt_tokens: torch.Tensor,
        negative_prompt_tokens: torch.Tensor,
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
            timesteps=timesteps,
        )
        z_negative, negative_skips = self.denoised_main_latent(
            negative_images,
            prompt_tokens=negative_prompt_tokens,
            timesteps=timesteps,
        )
        return z_positive, z_negative, positive_skips, negative_skips

    def forward(
        self,
        images: torch.Tensor,
        *,
        prompt: list[str] | str | None = None,
        prompt_tokens: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        main_latent, main_skips = self.denoised_main_latent(
            images,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            timesteps=timesteps,
        )
        return self.decode_main_latent(
            main_latent,
            main_skips,
            prompt_condition=self._last_detail_condition,
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
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
    }
    if experiment_metadata is not None:
        checkpoint["experiment_metadata"] = dict(experiment_metadata)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, destination)


def _apply_model_checkpoint(model: SelectiveDifix, checkpoint: dict) -> int:
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
        if mismatches:
            raise ValueError(
                f"Checkpoint DAEM-lite configuration mismatch: {mismatches}"
            )
        model.vae.decoder.detail_blocks.load_state_dict(
            saved_detail_state, strict=True
        )
    return int(checkpoint["global_step"])


def load_model_checkpoint(
    model: SelectiveDifix,
    path: str | Path,
    *,
    expected_dataset: str | None = None,
    expected_seed: int | None = None,
) -> int:
    """Load model weights only, for validation/backfill without an optimizer."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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
    return _apply_model_checkpoint(model, checkpoint)


def load_training_checkpoint(
    model: SelectiveDifix,
    optimizer,
    scheduler,
    path: str | Path,
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    global_step = _apply_model_checkpoint(model, checkpoint)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["lr_scheduler"])
    return global_step
