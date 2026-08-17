"""CDD-11 indexing and PyTorch datasets driven by scene-ID manifests."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

from PIL import Image

from .protocol import DEGRADATIONS

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def index_images(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing CDD-11 directory: {directory}")
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in result:
                raise ValueError(f"Two files have scene ID {path.stem!r} in {directory}")
            result[path.stem] = path
    return result


def validate_partition(data_root: Path, partition: str, expected_scenes: int | None = None) -> list[str]:
    base = data_root / partition
    clear = index_images(base / "clear")
    if expected_scenes is not None and len(clear) != expected_scenes:
        raise ValueError(f"{partition}/clear: expected {expected_scenes} scenes, got {len(clear)}")
    clear_ids = set(clear)
    problems: list[str] = []
    for degradation in DEGRADATIONS:
        ids = set(index_images(base / degradation))
        missing, extra = clear_ids - ids, ids - clear_ids
        if missing:
            problems.append(f"{degradation}: missing {len(missing)} ({sorted(missing)[:3]})")
        if extra:
            problems.append(f"{degradation}: extra {len(extra)} ({sorted(extra)[:3]})")
    if problems:
        raise ValueError(f"Invalid CDD-11 {partition} pairing:\n" + "\n".join(problems))
    return sorted(clear)


def _to_tensor(image: Image.Image):
    import numpy as np
    import torch

    array = np.array(image.convert("RGB"), dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class CDD11SceneDataset:
    """All 11 degraded/clear pairs for an explicit set of scene IDs."""

    def __init__(self, data_root: str | Path, partition: str, scene_ids: Iterable[str], *, patch_size: int | None = None, augment: bool = False):
        from torch.utils.data import Dataset

        # Registering as a virtual subclass is unnecessary; DataLoader only needs
        # __len__ and __getitem__. Avoid importing torch at split-generation time.
        del Dataset
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
        for degradation, paths in self.degraded.items():
            missing = set(self.scene_ids) - set(paths)
            if missing:
                raise ValueError(f"{degradation} misses selected scenes: {sorted(missing)[:5]}")
        self.samples = [(scene_id, degradation) for scene_id in self.scene_ids for degradation in DEGRADATIONS]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import torch.nn.functional as F

        scene_id, degradation = self.samples[index]
        lq_path = self.degraded[degradation][scene_id]
        gt_path = self.clear[scene_id]
        with Image.open(lq_path) as image:
            lq = _to_tensor(image)
        with Image.open(gt_path) as image:
            gt = _to_tensor(image)
        if lq.shape != gt.shape:
            raise ValueError(f"Shape mismatch: {lq_path} {tuple(lq.shape)} vs {gt_path} {tuple(gt.shape)}")
        if self.patch_size is not None:
            size = self.patch_size
            pad_h, pad_w = max(0, size - lq.shape[-2]), max(0, size - lq.shape[-1])
            if pad_h or pad_w:
                lq = F.pad(lq, (0, pad_w, 0, pad_h), mode="replicate")
                gt = F.pad(gt, (0, pad_w, 0, pad_h), mode="replicate")
            top = random.randint(0, lq.shape[-2] - size)
            left = random.randint(0, lq.shape[-1] - size)
            lq, gt = lq[:, top:top + size, left:left + size], gt[:, top:top + size, left:left + size]
        if self.augment:
            if random.random() < 0.5:
                lq, gt = lq.flip(-1), gt.flip(-1)
            if random.random() < 0.5:
                lq, gt = lq.flip(-2), gt.flip(-2)
            turns = random.randrange(4)
            lq, gt = lq.rot90(turns, (-2, -1)), gt.rot90(turns, (-2, -1))
        return {"lq": lq, "gt": gt, "scene_id": scene_id, "degradation": degradation, "lq_path": str(lq_path)}
