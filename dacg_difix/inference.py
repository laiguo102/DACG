"""Single-image DACG -> degradation-guided Difix3D cascade inference."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .train import load_adapter_checkpoint


def image_to_tensor(path: str | Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div(255.0).unsqueeze(0).to(device)


def save_tensor_image(image: torch.Tensor, path: str | Path) -> None:
    array = (
        image.detach().float().squeeze(0).clamp(0, 1).mul(255).round()
        .to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(output)


def pad_to_multiple(
    image: torch.Tensor,
    multiple: int = 8,
) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = image.shape[-2:]
    pad_h, pad_w = (-height) % multiple, (-width) % multiple
    if pad_h or pad_w:
        mode = "reflect" if height > pad_h and width > pad_w else "replicate"
        image = F.pad(image, (0, pad_w, 0, pad_h), mode=mode)
    return image, (height, width)


def tokenize_prompt(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    return tokenizer(
        prompt,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)


@torch.inference_mode()
def cascade_restore(
    dacg: torch.nn.Module,
    difix: torch.nn.Module,
    degraded_01: torch.Tensor,
    prompt_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore one native-resolution image and return Difix/DACG RGB tensors."""

    coarse_01, p_global = dacg.forward_with_degradation(degraded_01)
    coarse_01 = coarse_01.clamp(0, 1)
    main = coarse_01.mul(2).sub(1)
    reference = degraded_01.mul(2).sub(1)
    main, original_size = pad_to_multiple(main)
    reference, _ = pad_to_multiple(reference)
    prediction = difix(main, reference, prompt_tokens, p_global=p_global)
    height, width = original_size
    prediction_01 = prediction[..., :height, :width].float().add(1).mul(0.5).clamp(0, 1)
    return prediction_01, coarse_01


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def run(args: argparse.Namespace) -> None:
    from cdd11_full.model import load_network

    from .model import DACGDifix

    device = _device(args.device)
    dacg, _ = load_network(str(args.dacg_checkpoint), device)
    dacg.requires_grad_(False).eval()
    difix = DACGDifix(
        pretrained_model_name_or_path=args.pretrained_model,
        local_files_only=args.local_files_only,
        lora_mode=args.lora_mode,
        lora_rank_unet=args.lora_rank_unet,
        lora_rank_vae=args.lora_rank_vae,
        timestep=args.timestep,
    ).to(device)
    load_adapter_checkpoint(args.difix_checkpoint, difix)
    difix.requires_grad_(False).eval()

    degraded = image_to_tensor(args.input, device)
    tokens = tokenize_prompt(difix.tokenizer, args.prompt, device)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.precision == "bf16" and device.type == "cuda"
        else nullcontext()
    )
    with autocast:
        prediction, coarse = cascade_restore(dacg, difix, degraded, tokens)
    save_tensor_image(prediction, args.output)
    if args.coarse_output is not None:
        save_tensor_image(coarse, args.coarse_output)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--input", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--coarse-output", type=Path)
    value.add_argument("--dacg-checkpoint", type=Path, required=True)
    value.add_argument("--difix-checkpoint", type=Path, required=True)
    value.add_argument("--pretrained-model", default="stabilityai/sd-turbo")
    value.add_argument("--local-files-only", action="store_true")
    value.add_argument(
        "--lora-mode",
        choices=("static", "p-global", "p-global-layer-id"),
        default="p-global-layer-id",
    )
    value.add_argument("--lora-rank-unet", type=int, default=32)
    value.add_argument("--lora-rank-vae", type=int, default=16)
    value.add_argument("--timestep", type=int, default=199)
    value.add_argument("--prompt", default="remove degradation")
    value.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    value.add_argument("--device", default="auto")
    return value


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
