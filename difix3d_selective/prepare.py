"""Precompute DACG coarse images and build directed Difix manifests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from .protocol import PROMPT_NAMES, directed_tasks, make_scene_split, selected_pairs

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def index_images(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def save_tensor(tensor: torch.Tensor, path: Path) -> None:
    array = (
        tensor.detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, "RGB").save(path)


def tiled_forward(
    model: torch.nn.Module, image: torch.Tensor, tile_size: int, overlap: int
) -> torch.Tensor:
    height, width = image.shape[-2:]
    if tile_size <= 0 or (height <= tile_size and width <= tile_size):
        return model(image)[..., :height, :width]
    if overlap >= tile_size:
        raise ValueError("DACG tile overlap must be smaller than tile size")
    stride = tile_size - overlap
    tops = list(range(0, max(height - tile_size, 0), stride)) + [max(height - tile_size, 0)]
    lefts = list(range(0, max(width - tile_size, 0), stride)) + [max(width - tile_size, 0)]
    output = torch.zeros_like(image)
    weight = torch.zeros_like(image)
    for top in dict.fromkeys(tops):
        for left in dict.fromkeys(lefts):
            patch = image[..., top : top + tile_size, left : left + tile_size]
            prediction = model(patch)[..., : patch.shape[-2], : patch.shape[-1]]
            output[..., top : top + prediction.shape[-2], left : left + prediction.shape[-1]] += prediction
            weight[..., top : top + prediction.shape[-2], left : left + prediction.shape[-1]] += 1
    return output / weight.clamp_min(1)


def load_dacg_checkpoint(checkpoint: str | Path, device: torch.device):
    from cdd11_full.model import load_network

    return load_network(str(Path(checkpoint).resolve()), device)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def generate_cdd11_coarse(
    *,
    data_root: str | Path,
    dacg_checkpoint: str | Path,
    coarse_root: str | Path,
    pair_ids: list[int],
    device: torch.device,
    tile_size: int = 0,
    tile_overlap: int = 64,
) -> dict[str, Path | int]:
    data_root = Path(data_root).resolve()
    coarse_root = Path(coarse_root).resolve()
    train_root = data_root / "train"
    pairs = selected_pairs(pair_ids)
    clear = index_images(train_root / "clear")
    all_scene_ids = set(clear)

    source_images: dict[str, dict[str, Path]] = {}
    for _, pair in pairs:
        source_images[pair] = index_images(train_root / pair)
        if set(source_images[pair]) != all_scene_ids:
            raise ValueError(f"CDD11 scene mismatch in train/{pair}")

    jobs: list[tuple[Path, Path]] = []
    for _, pair in pairs:
        for scene_id, source in source_images[pair].items():
            output = coarse_root / pair / f"{scene_id}.png"
            if not output.is_file():
                jobs.append((source, output))

    if jobs:
        model, checkpoint = load_dacg_checkpoint(dacg_checkpoint, device)
        del checkpoint
        with torch.inference_mode():
            for source, output in tqdm(jobs, desc="Generating DACG coarse"):
                prediction = tiled_forward(
                    model,
                    image_tensor(source).to(device, non_blocking=True),
                    tile_size,
                    tile_overlap,
                )[0]
                save_tensor(prediction, output)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    info = {
        "data_root": str(data_root),
        "dacg_checkpoint": str(Path(dacg_checkpoint).resolve()),
        "coarse_root": str(coarse_root),
        "degradation_pairs": [{"id": pair_id, "name": pair} for pair_id, pair in pairs],
        "coarse_images": len(clear) * len(pairs),
    }
    coarse_root.mkdir(parents=True, exist_ok=True)
    (coarse_root / "coarse_preparation.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"coarse_root": coarse_root, "coarse_images": len(clear) * len(pairs)}


def prepare_selective_manifests(
    *,
    data_root: str | Path,
    coarse_root: str | Path,
    output_dir: str | Path,
    pair_ids: list[int],
) -> dict[str, Path | int]:
    data_root = Path(data_root).resolve()
    coarse_root = Path(coarse_root).resolve()
    prepared_root = Path(output_dir).resolve() / "prepared"
    train_root = data_root / "train"
    pairs = selected_pairs(pair_ids)
    clear = index_images(train_root / "clear")
    splits = make_scene_split(list(clear))
    all_scene_ids = set(clear)

    source_images: dict[str, dict[str, Path]] = {}
    coarse_images: dict[str, dict[str, Path]] = {}
    target_images: dict[str, dict[str, Path]] = {}
    for _, pair in pairs:
        source_images[pair] = index_images(train_root / pair)
        coarse_images[pair] = index_images(coarse_root / pair)
        if set(source_images[pair]) != all_scene_ids:
            raise ValueError(f"CDD11 scene mismatch in train/{pair}")
        if set(coarse_images[pair]) != all_scene_ids:
            raise ValueError(f"CDD11 coarse scene mismatch in {coarse_root / pair}")
    for degradation in PROMPT_NAMES:
        target_images[degradation] = index_images(train_root / degradation)
        if set(target_images[degradation]) != all_scene_ids:
            raise ValueError(f"CDD11 scene mismatch in train/{degradation}")

    manifests: dict[str, Path] = {}
    counts: dict[str, int] = {}
    for split, scene_ids in splits.items():
        records: list[dict] = []
        for pair_id, pair in pairs:
            for scene_id in scene_ids:
                coarse = coarse_images[pair][scene_id]
                reference = source_images[pair][scene_id]
                for task in directed_tasks(pair_id):
                    target = target_images[task.preserve][scene_id]
                    records.append(
                        {
                            "id": f"{split}/{pair}/{scene_id}/remove-{task.remove}-preserve-{task.preserve}",
                            "split": split,
                            "scene_id": scene_id,
                            "pair_id": pair_id,
                            "pair": pair,
                            "remove": task.remove,
                            "preserve": task.preserve,
                            "prompt": task.prompt,
                            "image": str(coarse.resolve()),
                            "ref_image": str(reference.resolve()),
                            "target_image": str(target.resolve()),
                        }
                    )
        manifest = prepared_root / "manifests" / f"{split}.jsonl"
        _write_jsonl(manifest, records)
        manifests[split] = manifest
        counts[split] = len(records)

    info = {
        "seed": 42,
        "data_root": str(data_root),
        "coarse_root": str(coarse_root),
        "degradation_pairs": [{"id": pair_id, "name": pair} for pair_id, pair in pairs],
        "train_scene_ids": splits["train"],
        "validation_scene_ids": splits["validation"],
        "coarse_images": len(clear) * len(pairs),
        "train_samples": counts["train"],
        "validation_samples": counts["validation"],
    }
    prepared_root.mkdir(parents=True, exist_ok=True)
    (prepared_root / "split_and_preparation.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "train_manifest": manifests["train"],
        "validation_manifest": manifests["validation"],
        "coarse_images": len(clear) * len(pairs),
        "train_samples": counts["train"],
        "validation_samples": counts["validation"],
    }
