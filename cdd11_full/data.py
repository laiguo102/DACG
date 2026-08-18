"""CDD-11-v1 manifest dataset and deterministic one-per-degradation sampling."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from .protocol import DEGRADATION_ARITY, DEGRADATIONS, PROTOCOL_NAME, deterministic_seed

MAX_TORCH_SEED = (1 << 63) - 1


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    degradation: str
    split: str
    input_path: Path
    target_path: Path
    scene_id: str
    arity: int


def load_manifest(path: Path, expected_split: str) -> list[ManifestRecord]:
    records, identifiers = [], set()
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"id", "degradation", "split", "input", "target", "scene_id", "metadata"}
            missing = required - set(row)
            if missing:
                raise ValueError(f"{path}:{line_number} missing keys: {sorted(missing)}")
            sample_id, degradation, split = str(row["id"]), str(row["degradation"]), str(row["split"])
            if sample_id in identifiers:
                raise ValueError(f"Duplicate sample ID: {sample_id}")
            if degradation not in DEGRADATIONS or split != expected_split:
                raise ValueError(f"Invalid degradation/split at {path}:{line_number}")
            input_path, target_path = Path(str(row["input"])), Path(str(row["target"]))
            if not input_path.is_absolute() or not target_path.is_absolute():
                raise ValueError("CDD-11-v1 manifests require absolute paths")
            arity = int(row["metadata"].get("arity", -1))
            if arity != DEGRADATION_ARITY[degradation]:
                raise ValueError(f"Invalid arity for {sample_id}")
            identifiers.add(sample_id)
            records.append(ManifestRecord(sample_id, degradation, split, input_path, target_path,
                                          str(row["scene_id"]), arity))
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    return records


def _tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div_(255.0).contiguous()


def _train_transform(degraded, target, patch_size: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    height, width = degraded.shape[-2:]
    if target.shape != degraded.shape:
        raise ValueError(f"Input/target shape mismatch: {degraded.shape} != {target.shape}")
    pad_h, pad_w = max(0, patch_size - height), max(0, patch_size - width)
    if pad_h or pad_w:
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        mode = "reflect" if padding[0] < width and padding[1] < width and padding[2] < height and padding[3] < height else "replicate"
        degraded, target = F.pad(degraded, padding, mode=mode), F.pad(target, padding, mode=mode)
    height, width = degraded.shape[-2:]
    top = int(torch.randint(height - patch_size + 1, (1,), generator=generator).item())
    left = int(torch.randint(width - patch_size + 1, (1,), generator=generator).item())
    degraded = degraded[:, top:top + patch_size, left:left + patch_size]
    target = target[:, top:top + patch_size, left:left + patch_size]
    if int(torch.randint(2, (1,), generator=generator).item()):
        degraded, target = degraded.flip(-1), target.flip(-1)
    if int(torch.randint(2, (1,), generator=generator).item()):
        degraded, target = degraded.flip(-2), target.flip(-2)
    turns = int(torch.randint(4, (1,), generator=generator).item())
    if turns:
        degraded, target = degraded.rot90(turns, (-2, -1)), target.rot90(turns, (-2, -1))
    return degraded, target


class CDD11ManifestDataset(Dataset):
    def __init__(self, manifest: Path, split: str, patch_size: int | None = None):
        if split == "train" and patch_size is None:
            raise ValueError("Training requires patch_size")
        if split != "train" and patch_size is not None:
            raise ValueError("Validation/test must retain native resolution")
        self.split, self.patch_size = split, patch_size
        self.records = load_manifest(manifest, split)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, request):
        if isinstance(request, (tuple, list)):
            index, sample_seed = int(request[0]), int(request[1])
        else:
            index, sample_seed = int(request), 0
            if self.split == "train":
                raise ValueError("Training requires deterministic (index, seed) requests")
        record = self.records[index]
        degraded, target = _tensor(record.input_path), _tensor(record.target_path)
        if self.split == "train":
            degraded, target = _train_transform(degraded, target, int(self.patch_size), sample_seed)
        return {"degraded": degraded, "target": target, "degradation": record.degradation,
                "arity": record.arity, "sample_id": record.sample_id,
                "scene_id": record.scene_id, "sample_seed": sample_seed}


class BalancedDegradationBatchSampler(Sampler):
    def __init__(self, dataset: CDD11ManifestDataset, start_step: int, num_batches: int, seed: int):
        if dataset.split != "train" or min(start_step, num_batches) < 0:
            raise ValueError("Invalid balanced sampler arguments")
        self.dataset, self.start_step, self.num_batches, self.seed = dataset, start_step, num_batches, seed
        grouped = {value: defaultdict(list) for value in DEGRADATIONS}
        for index, record in enumerate(dataset.records):
            grouped[record.degradation][record.scene_id].append(index)
        self.by_scene = {degradation: {scene: tuple(indices) for scene, indices in sorted(values.items())}
                         for degradation, values in grouped.items()}
        self.scenes = {degradation: tuple(values) for degradation, values in self.by_scene.items()}
        if any(not value for value in self.scenes.values()):
            raise ValueError("Training manifest must contain all 11 degradations")

    @property
    def batch_size(self):
        return len(DEGRADATIONS)

    def __len__(self):
        return self.num_batches

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        for global_step in range(self.start_step, self.start_step + self.num_batches):
            generator = torch.Generator().manual_seed(deterministic_seed(
                f"{PROTOCOL_NAME}:balanced-batch:{self.seed}:{global_step}"))
            requests = []
            for degradation in DEGRADATIONS:
                scenes = self.scenes[degradation]
                scene = scenes[int(torch.randint(len(scenes), (1,), generator=generator).item())]
                indices = self.by_scene[degradation][scene]
                index = indices[int(torch.randint(len(indices), (1,), generator=generator).item())]
                sample_seed = int(torch.randint(MAX_TORCH_SEED, (1,), generator=generator).item())
                requests.append((index, sample_seed))
            order = torch.randperm(len(requests), generator=generator).tolist()
            yield [requests[index] for index in order]


def build_train_loader(manifest: Path, patch_size: int, start_step: int, num_batches: int,
                       seed: int, num_workers: int):
    dataset = CDD11ManifestDataset(manifest, "train", patch_size)
    sampler = BalancedDegradationBatchSampler(dataset, start_step, num_batches, seed)
    options = dict(dataset=dataset, batch_sampler=sampler, num_workers=num_workers, pin_memory=True,
                   persistent_workers=num_workers > 0,
                   generator=torch.Generator().manual_seed(
                       deterministic_seed(f"{PROTOCOL_NAME}:train-loader:{seed}")))
    if num_workers:
        options.update(prefetch_factor=2, multiprocessing_context="spawn")
    return DataLoader(**options), dataset, sampler


def build_eval_loader(manifest: Path, split: str, num_workers: int):
    dataset = CDD11ManifestDataset(manifest, split)
    options = dict(dataset=dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                   pin_memory=True, persistent_workers=num_workers > 0)
    if num_workers:
        options.update(prefetch_factor=2, multiprocessing_context="spawn")
    return DataLoader(**options), dataset
