"""JSONL dataset for coarse/main, degraded/reference and selective targets."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode


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


class SelectiveDifixDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: str | Path, tokenizer, resolution: int = 512):
        self.records = load_records(manifest)
        self.tokenizer = tokenizer
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        main = _image_tensor(record["image"], self.resolution)
        reference = _image_tensor(record["ref_image"], self.resolution)
        target = _image_tensor(record["target_image"], self.resolution)
        ground_truth = _image_tensor(record["clear_image"], self.resolution)
        input_ids = self.tokenizer(
            record["prompt"],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        return {
            "conditioning_pixel_values": torch.stack([main, reference]),
            "output_pixel_values": target,
            "ground_truth_pixel_values": ground_truth,
            "input_ids": input_ids,
            "prompt": record["prompt"],
            "sample_id": record["id"],
            "split": record.get("split", ""),
            "scene_id": str(record.get("scene_id", "")),
            "pair_id": int(record.get("pair_id", -1)),
            "pair": record.get("pair", ""),
            "remove": record.get("remove", ""),
            "preserve": record.get("preserve", ""),
        }
