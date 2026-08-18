"""Exact CDD-11 pairing and deterministic full-dataset sampling."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image

from .protocol import DEGRADATIONS

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def index_images(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing CDD-11 directory: {directory}")
    images: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in images:
                raise ValueError(f"Duplicate scene ID {path.stem!r} in {directory}")
            images[path.stem] = path
    return images


def validate_partition(data_root: str | Path, partition: str, expected_scenes: int | None = None) -> list[str]:
    base = Path(data_root) / partition
    clear = index_images(base / "clear")
    if expected_scenes is not None and len(clear) != expected_scenes:
        raise ValueError(f"{partition}/clear: expected {expected_scenes} scenes, got {len(clear)}")
    clear_ids = set(clear)
    problems: list[str] = []
    for degradation in DEGRADATIONS:
        observed = set(index_images(base / degradation))
        missing, extra = clear_ids - observed, observed - clear_ids
        if missing:
            problems.append(f"{degradation}: missing {len(missing)} ({sorted(missing)[:3]})")
        if extra:
            problems.append(f"{degradation}: extra {len(extra)} ({sorted(extra)[:3]})")
    if problems:
        raise ValueError(f"Invalid CDD-11 {partition} pairing:\n" + "\n".join(problems))
    return sorted(clear)


def dataset_fingerprint(data_root: str | Path, partition: str, scene_ids: Iterable[str]) -> str:
    """Hash relative paths and file sizes without reading 15k images into memory."""
    root = Path(data_root).resolve()
    rows: list[str] = []
    for scene_id in sorted(scene_ids):
        for folder in ("clear", *DEGRADATIONS):
            candidates = [path for path in (root / partition / folder).glob(f"{scene_id}.*") if path.suffix.lower() in IMAGE_SUFFIXES]
            if len(candidates) != 1:
                raise ValueError(f"Expected one file for {partition}/{folder}/{scene_id}, got {len(candidates)}")
            path = candidates[0]
            rows.append(f"{path.relative_to(root).as_posix()}\t{path.stat().st_size}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _to_tensor(image: Image.Image):
    import numpy as np
    import torch

    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class CDD11FullDataset:
    """All unique `(scene, degradation)` pairs; request tuples seed transforms."""

    def __init__(
        self,
        data_root: str | Path,
        partition: str,
        scene_ids: Iterable[str],
        *,
        patch_size: int | None = None,
        augment: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.partition = partition
        self.scene_ids = list(scene_ids)
        self.patch_size = patch_size
        self.augment = augment
        base = self.data_root / partition
        self.clear = index_images(base / "clear")
        self.degraded = {name: index_images(base / name) for name in DEGRADATIONS}
        unknown = set(self.scene_ids) - set(self.clear)
        if unknown:
            raise ValueError(f"Unknown {partition} scene IDs: {sorted(unknown)[:5]}")
        for name, paths in self.degraded.items():
            missing = set(self.scene_ids) - set(paths)
            if missing:
                raise ValueError(f"{name} misses selected scenes: {sorted(missing)[:5]}")
        self.samples = [(scene_id, degradation) for scene_id in self.scene_ids for degradation in DEGRADATIONS]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, request):
        import torch
        import torch.nn.functional as F

        if isinstance(request, (tuple, list)):
            index, sample_seed = int(request[0]), int(request[1])
        else:
            index, sample_seed = int(request), 0
        scene_id, degradation = self.samples[index]
        lq_path, gt_path = self.degraded[degradation][scene_id], self.clear[scene_id]
        with Image.open(lq_path) as image:
            lq = _to_tensor(image)
        with Image.open(gt_path) as image:
            gt = _to_tensor(image)
        if lq.shape != gt.shape:
            raise ValueError(f"Shape mismatch: {lq_path} {tuple(lq.shape)} vs {gt_path} {tuple(gt.shape)}")

        generator = torch.Generator().manual_seed(sample_seed)
        if self.patch_size is not None:
            size = self.patch_size
            pad_h, pad_w = max(0, size - lq.shape[-2]), max(0, size - lq.shape[-1])
            if pad_h or pad_w:
                lq = F.pad(lq, (0, pad_w, 0, pad_h), mode="replicate")
                gt = F.pad(gt, (0, pad_w, 0, pad_h), mode="replicate")
            top = int(torch.randint(lq.shape[-2] - size + 1, (1,), generator=generator).item())
            left = int(torch.randint(lq.shape[-1] - size + 1, (1,), generator=generator).item())
            lq, gt = lq[:, top:top + size, left:left + size], gt[:, top:top + size, left:left + size]
        if self.augment:
            # Match the repository's random_augmentation modes 1..7 exactly:
            # vertical flip, rotations by 90/180/270 degrees, and each
            # rotation followed by a vertical flip. Identity is not sampled.
            mode = int(torch.randint(1, 8, (1,), generator=generator).item())
            turns = mode // 2 if mode >= 2 else 0
            if turns:
                lq, gt = lq.rot90(turns, (-2, -1)), gt.rot90(turns, (-2, -1))
            if mode % 2 == 1:
                lq, gt = lq.flip(-2), gt.flip(-2)
        return {
            "lq": lq, "gt": gt, "scene_id": scene_id, "degradation": degradation,
            "lq_path": str(lq_path), "gt_path": str(gt_path), "sample_seed": sample_seed,
        }


class DeterministicStepBatchSampler:
    """Emit deterministic microbatches grouped by optimizer step.

    Each epoch uses one seeded permutation and consumes each selected pair at
    most once. The small tail that cannot fill an effective batch is dropped.
    Restarting at `start_step` yields exactly the same next request.
    """

    def __init__(
        self,
        dataset_size: int,
        *,
        batch_size: int,
        accumulate: int,
        epochs: int,
        seed: int,
        start_step: int = 0,
    ) -> None:
        if min(dataset_size, batch_size, accumulate, epochs) < 1:
            raise ValueError("dataset size, batch size, accumulation and epochs must be positive")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.accumulate = accumulate
        self.epochs = epochs
        self.seed = seed
        self.steps_per_epoch = dataset_size // (batch_size * accumulate)
        if self.steps_per_epoch < 1:
            raise ValueError("effective batch is larger than the dataset")
        self.total_steps = self.steps_per_epoch * epochs
        if not 0 <= start_step <= self.total_steps:
            raise ValueError(f"start_step must be within [0,{self.total_steps}]")
        self.start_step = start_step

    def __len__(self) -> int:
        return (self.total_steps - self.start_step) * self.accumulate

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        import torch

        cached_epoch, permutation = -1, None
        for global_step in range(self.start_step, self.total_steps):
            epoch, step_in_epoch = divmod(global_step, self.steps_per_epoch)
            if epoch != cached_epoch:
                generator = torch.Generator().manual_seed(self.seed + epoch)
                permutation = torch.randperm(self.dataset_size, generator=generator).tolist()
                cached_epoch = epoch
            assert permutation is not None
            effective = self.batch_size * self.accumulate
            offset = step_in_epoch * effective
            for microbatch in range(self.accumulate):
                start = offset + microbatch * self.batch_size
                indices = permutation[start:start + self.batch_size]
                yield [(index, self.seed * 1_000_003 + epoch * self.dataset_size + index) for index in indices]
