"""Prepared CDD11 data loading for DACG-conditioned Difix3D training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


PROMPT_LENGTH = 77


def tokenize_prompt(tokenizer: Any, prompt: str, max_length: int = PROMPT_LENGTH) -> torch.Tensor:
    """Tokenize without importing a particular tokenizer implementation."""

    encoded = tokenizer(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    return torch.as_tensor(input_ids, dtype=torch.long).reshape(-1)[:max_length]


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def _resolve_image_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def _image_tensor(path: Path, resolution: int | None) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if resolution is not None:
            image = image.resize((resolution, resolution), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div_(255.0).contiguous()


class PreparedCDD11Dataset(Dataset):
    """Aligned coarse/reference/target triples from a prepared JSONL manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        resolution: int | None,
        tokenizer: Any | None = None,
        prompt: str = "",
        prompt_tokens: Sequence[int] | torch.Tensor | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.resolution = None if resolution is None else int(resolution)
        self.records = _load_records(self.manifest_path)
        if prompt_tokens is not None:
            tokens = torch.as_tensor(prompt_tokens, dtype=torch.long).reshape(-1)
        elif tokenizer is not None:
            tokens = tokenize_prompt(tokenizer, prompt)
        else:
            tokens = torch.zeros(PROMPT_LENGTH, dtype=torch.long)
        if tokens.numel() != PROMPT_LENGTH:
            raise ValueError(f"prompt_tokens must contain {PROMPT_LENGTH} ids")
        self.prompt_tokens = tokens.contiguous()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        main_01 = _image_tensor(
            _resolve_image_path(record["coarse"], self.manifest_path), self.resolution
        )
        ref_01 = _image_tensor(
            _resolve_image_path(record["degraded"], self.manifest_path), self.resolution
        )
        target_01 = _image_tensor(
            _resolve_image_path(record["target"], self.manifest_path), self.resolution
        )
        return {
            "main": main_01.mul(2.0).sub(1.0),
            "ref": ref_01.mul(2.0).sub(1.0),
            "target": target_01.mul(2.0).sub(1.0),
            "ref_01": ref_01,
            "prompt_tokens": self.prompt_tokens.clone(),
            "id": str(record["id"]),
            "degradation": str(record["degradation"]),
            "split": str(record["split"]),
            "scene_id": str(record["scene_id"]),
            "arity": int(record["arity"]),
        }


def build_difix_loader(
    manifest_path: str | Path,
    resolution: int | None,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int = 0,
    tokenizer: Any | None = None,
    prompt: str = "",
    prompt_tokens: Sequence[int] | torch.Tensor | None = None,
    drop_last: bool = False,
) -> DataLoader:
    dataset = PreparedCDD11Dataset(
        manifest_path=manifest_path,
        resolution=resolution,
        tokenizer=tokenizer,
        prompt=prompt,
        prompt_tokens=prompt_tokens,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
    )
