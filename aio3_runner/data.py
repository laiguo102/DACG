"""Frozen-manifest data input and deterministic balanced sampling for AIO3-v1."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

PROTOCOL = "aio3-v1"
TASKS = ("denoise", "derain", "dehaze")
SIGMAS = (15, 25, 50)
EXPECTED_MANIFEST_SHA256 = {
    "train.jsonl": "bd153a3b211957184de7b6171d6bc06a48f321b1c571906604d869b1aa19ca7e",
    "val.jsonl": "9c66c4c74a0279858ecab33df998b8eb55d6df021d2e59bbd1c253830ab3f50b",
    "test.jsonl": "7d80fd0af7aeaac2b6e901e20e71a744d7d705f98641f54913aa278e12c2b63a",
    "data_audit.json": "2959e402ecdb76172b9fe9bba3fae13c090348379dcd33898992abdc198e06b8",
    "visual_samples.json": "62e9f6e761e3db2c23895958f3707414a59baac30840919044fe6bf848ff628b",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(text: str) -> int:
    """Return a process-independent 63-bit seed derived from SHA256."""

    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little") & ((1 << 63) - 1)


def read_manifest(path: str | Path, expected_split: str | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = {"id", "task", "split", "target", "scene_id", "metadata"} - record.keys()
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}")
            if record["id"] in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id {record['id']!r}")
            if record["task"] not in TASKS:
                raise ValueError(f"{path}:{line_number}: invalid task {record['task']!r}")
            if expected_split and record["split"] != expected_split:
                raise ValueError(f"{path}:{line_number}: expected split {expected_split!r}")
            if record["task"] != "denoise" and not record.get("input"):
                raise ValueError(f"{path}:{line_number}: paired task requires input")
            seen.add(record["id"])
            records.append(record)
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records


def verify_frozen_manifests(manifest_dir: str | Path) -> dict[str, str]:
    """Verify the three published AIO3-v1 manifest hashes."""

    root = Path(manifest_dir)
    actual = {name: sha256_file(root / name) for name in EXPECTED_MANIFEST_SHA256}
    mismatches = {
        name: (EXPECTED_MANIFEST_SHA256[name], value)
        for name, value in actual.items()
        if value != EXPECTED_MANIFEST_SHA256[name]
    }
    if mismatches:
        details = ", ".join(f"{n}: expected {e}, got {a}" for n, (e, a) in mismatches.items())
        raise ValueError(f"frozen manifest hash mismatch: {details}")
    return actual


def manifest_statistics(manifest_dir: str | Path) -> dict[str, Any]:
    root = Path(manifest_dir)
    output: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        records = read_manifest(root / f"{split}.jsonl", split)
        output[split] = {
            "total": len(records),
            "samples": {task: sum(record["task"] == task for record in records) for task in TASKS},
            "scenes": {
                task: len({str(record["scene_id"]) for record in records if record["task"] == task})
                for task in TASKS
            },
        }
    return output


def _resolve_path(value: str | None, data_root: Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute() or data_root is None:
        return path
    return data_root / path


def _load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


def _pad_to_patch(tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    pad_h, pad_w = max(0, patch_size - height), max(0, patch_size - width)
    if not (pad_h or pad_w):
        return tensor
    # Reflection needs each padding amount to be smaller than its dimension.
    mode = "reflect" if height > pad_h and width > pad_w and height > 1 and width > 1 else "replicate"
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)


class AIO3ManifestDataset(Dataset):
    """Read AIO3 JSONL records and apply deterministic protocol transforms.

    Training indices may be ``(record_index, sample_seed)`` requests emitted by
    :class:`BalancedTaskBatchSampler`. Validation/test samples use the seed in
    metadata, or the protocol SHA256 formula when it is absent.
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        split: str,
        data_root: str | Path | None = None,
        patch_size: int = 128,
    ):
        self.manifest_path = Path(manifest)
        self.split = split
        self.data_root = Path(data_root) if data_root else None
        self.patch_size = patch_size
        self.records = read_manifest(self.manifest_path, split)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, request: int | tuple[int, int]) -> dict[str, Any]:
        if isinstance(request, tuple):
            index, sample_seed = int(request[0]), int(request[1])
        else:
            index = int(request)
            record_for_seed = self.records[index]
            metadata = record_for_seed.get("metadata", {})
            sigma = metadata.get("sigma", record_for_seed.get("sigma", "none"))
            sample_seed = int(metadata.get("seed", record_for_seed.get("seed", stable_seed(
                f"{PROTOCOL}:{self.split}:{record_for_seed['id']}:sigma{sigma}"
            ))))

        record = self.records[index]
        target_path = _resolve_path(record["target"], self.data_root)
        input_path = _resolve_path(record.get("input"), self.data_root)
        if target_path is None:
            raise ValueError(f"record {record['id']} has no target")
        target = _load_rgb(target_path)
        degraded = target.clone() if record["task"] == "denoise" else _load_rgb(input_path)  # type: ignore[arg-type]
        if degraded.shape != target.shape:
            raise ValueError(f"record {record['id']} input/target shape mismatch")

        sigma: int | None = None
        generator = torch.Generator().manual_seed(sample_seed)
        if self.split == "train":
            degraded = _pad_to_patch(degraded, self.patch_size)
            target = _pad_to_patch(target, self.patch_size)
            height, width = target.shape[-2:]
            top = int(torch.randint(height - self.patch_size + 1, (1,), generator=generator).item())
            left = int(torch.randint(width - self.patch_size + 1, (1,), generator=generator).item())
            degraded = degraded[:, top : top + self.patch_size, left : left + self.patch_size]
            target = target[:, top : top + self.patch_size, left : left + self.patch_size]
            if torch.rand((), generator=generator).item() < 0.5:
                degraded, target = degraded.flip(-1), target.flip(-1)
            if torch.rand((), generator=generator).item() < 0.5:
                degraded, target = degraded.flip(-2), target.flip(-2)
            rotations = int(torch.randint(4, (1,), generator=generator).item())
            degraded, target = torch.rot90(degraded, rotations, (-2, -1)), torch.rot90(target, rotations, (-2, -1))

        if record["task"] == "denoise":
            metadata = record.get("metadata", {})
            sigma_value = metadata.get("sigma", record.get("sigma"))
            sigma = int(sigma_value) if sigma_value is not None else SIGMAS[
                int(torch.randint(len(SIGMAS), (1,), generator=generator).item())
            ]
            noise = torch.randn(target.shape, generator=generator, dtype=target.dtype)
            degraded = target + noise * (sigma / 255.0)

        return {
            "id": record["id"],
            "task": record["task"],
            "scene_id": str(record["scene_id"]),
            # -1 keeps default PyTorch collation valid for non-denoising rows.
            # Callers should interpret it as N/A outside the denoise task.
            "sigma": sigma if sigma is not None else -1,
            "input": degraded,
            "target": target,
            "sample_seed": sample_seed,
        }


class BalancedTaskBatchSampler(Sampler[list[tuple[int, int]]]):
    """Yield deterministic 12-sample batches with exactly four samples per task."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        max_steps: int,
        seed: int = 3407,
        start_step: int = 0,
        samples_per_task: int = 4,
    ):
        self.records = records
        self.max_steps = max_steps
        self.seed = seed
        self.start_step = start_step
        self.samples_per_task = samples_per_task
        self.by_task_scene: dict[str, dict[str, list[int]]] = {
            task: defaultdict(list) for task in TASKS
        }
        for index, record in enumerate(records):
            self.by_task_scene[str(record["task"])][str(record["scene_id"])].append(index)
        missing = [task for task in TASKS if not self.by_task_scene[task]]
        if missing:
            raise ValueError(f"training manifest has no records for tasks: {missing}")

    def __len__(self) -> int:
        return max(0, self.max_steps - self.start_step)

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        for global_step in range(self.start_step, self.max_steps):
            rng = random.Random(stable_seed(f"{PROTOCOL}:{self.seed}:step:{global_step}"))
            batch: list[tuple[int, int]] = []
            for task in TASKS:
                scenes = sorted(self.by_task_scene[task])
                for slot in range(self.samples_per_task):
                    scene = scenes[rng.randrange(len(scenes))]
                    candidates = self.by_task_scene[task][scene]
                    index = candidates[rng.randrange(len(candidates))]
                    sample_seed = stable_seed(
                        f"{PROTOCOL}:{self.seed}:step:{global_step}:{task}:{slot}"
                    )
                    batch.append((index, sample_seed))
            rng.shuffle(batch)
            yield batch


def make_training_loader(
    manifest: str | Path,
    *,
    max_steps: int,
    seed: int = 3407,
    start_step: int = 0,
    data_root: str | Path | None = None,
    num_workers: int = 8,
) -> DataLoader:
    """Construct the canonical AIO3-v1 training loader.

    The returned loader has one deterministic, balanced effective batch per
    optimizer step. Micro-batching/gradient accumulation must preserve all 12
    samples together at the training-loop level.
    """

    dataset = AIO3ManifestDataset(manifest, split="train", data_root=data_root, patch_size=128)
    sampler = BalancedTaskBatchSampler(
        dataset.records, max_steps=max_steps, seed=seed, start_step=start_step, samples_per_task=4
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )
