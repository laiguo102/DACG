"""JSONL dataset for coarse/main, degraded/reference and selective targets."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

from .protocol import preserve_pair_prompt


def load_records(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _image_tensor(path: str, resolution: int) -> torch.Tensor:
    with Image.open(path) as image:
        tensor = TF.to_tensor(image.convert("RGB"))
    tensor = TF.resize(
        tensor,
        [resolution, resolution],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    return TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def negative_conditioning(
    coarse: torch.Tensor, degraded: torch.Tensor
) -> torch.Tensor:
    """Build the preserve-both condition used by negative CCDD-11 samples."""

    if coarse.shape != degraded.shape:
        raise ValueError(
            "coarse/degraded shape mismatch: "
            f"coarse={tuple(coarse.shape)}, degraded={tuple(degraded.shape)}"
        )
    # Inputs are normalized to [-1, 1]. Multiplying their difference by 0.5
    # converts it to the signed [0, 1]-space texture used during training.
    degradation_texture = ((degraded - coarse) * 0.5).clamp(-1, 1)
    stack_dimension = 1 if coarse.ndim == 4 else 0
    return torch.stack([degraded, degradation_texture], dim=stack_dimension)


class SelectiveDifixDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest: str | Path,
        tokenizer,
        resolution: int = 512,
        *,
        negative_probability: float = 0.0,
        training_mode: str = "auto",
        deduplicate_negative: bool = False,
    ):
        self.records = load_records(manifest)
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.negative_probability = float(negative_probability)
        self.training_mode = training_mode
        if not 0.0 <= self.negative_probability <= 1.0:
            raise ValueError("negative_probability must be in [0, 1]")
        if training_mode not in ("auto", "positive", "negative"):
            raise ValueError(f"Unknown training mode: {training_mode}")
        if deduplicate_negative and training_mode != "negative":
            raise ValueError(
                "deduplicate_negative is only valid for forced negative mode"
            )
        if deduplicate_negative:
            unique_records = []
            seen: set[tuple[int, str]] = set()
            for record in self.records:
                key = (int(record.get("pair_id", -1)), str(record.get("scene_id", "")))
                if key in seen:
                    continue
                seen.add(key)
                unique_records.append(record)
            self.records = unique_records

    def __len__(self) -> int:
        return len(self.records)

    def _is_negative(self) -> bool:
        if self.training_mode != "auto":
            return self.training_mode == "negative"
        if self.negative_probability == 0.0:
            return False
        if self.negative_probability == 1.0:
            return True
        return bool(torch.rand(()).item() < self.negative_probability)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        coarse = _image_tensor(record["image"], self.resolution)
        degraded = _image_tensor(record["ref_image"], self.resolution)
        is_negative = self._is_negative()
        if is_negative:
            conditioning = negative_conditioning(coarse, degraded)
            target = degraded
            prompt = preserve_pair_prompt(int(record["pair_id"]))
            mode = "negative"
            sample_id = (
                f"{record.get('split', '')}/{record.get('pair', '')}/"
                f"{record.get('scene_id', '')}/preserve-both"
            ).lstrip("/")
            remove = ""
            preserve = record.get("pair", "")
        else:
            conditioning = torch.stack([coarse, degraded])
            target = _image_tensor(record["target_image"], self.resolution)
            prompt = record["prompt"]
            mode = "positive"
            sample_id = record["id"]
            remove = record.get("remove", "")
            preserve = record.get("preserve", "")
        ground_truth = _image_tensor(record["clear_image"], self.resolution)
        input_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        return {
            "conditioning_pixel_values": conditioning,
            "output_pixel_values": target,
            "ground_truth_pixel_values": ground_truth,
            "dacg_coarse_pixel_values": coarse,
            "input_ids": input_ids,
            "prompt": prompt,
            "training_mode": mode,
            "is_negative": is_negative,
            "sample_id": sample_id,
            "split": record.get("split", ""),
            "scene_id": str(record.get("scene_id", "")),
            "pair_id": int(record.get("pair_id", -1)),
            "pair": record.get("pair", ""),
            "remove": remove,
            "preserve": preserve,
        }
