"""Precompute DACG coarse images and build directed Difix manifests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from .protocol import (
    NUM_SCENES,
    NUM_TEST_SCENES,
    PROMPT_NAMES,
    directed_tasks,
    make_scene_split,
    selected_pairs,
)

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
    tops = list(range(0, max(height - tile_size, 0), stride)) + [
        max(height - tile_size, 0)
    ]
    lefts = list(range(0, max(width - tile_size, 0), stride)) + [
        max(width - tile_size, 0)
    ]
    output = torch.zeros_like(image)
    weight = torch.zeros_like(image)
    for top in dict.fromkeys(tops):
        for left in dict.fromkeys(lefts):
            patch = image[..., top : top + tile_size, left : left + tile_size]
            prediction = model(patch)[..., : patch.shape[-2], : patch.shape[-1]]
            output[
                ...,
                top : top + prediction.shape[-2],
                left : left + prediction.shape[-1],
            ] += prediction
            weight[
                ...,
                top : top + prediction.shape[-2],
                left : left + prediction.shape[-1],
            ] += 1
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
    split: str = "train",
) -> dict[str, Path | int]:
    data_root = Path(data_root).resolve()
    coarse_root = Path(coarse_root).resolve()
    if split not in ("train", "test"):
        raise ValueError(f"Unsupported CDD11 split: {split}")
    split_root = data_root / split
    pairs = selected_pairs(pair_ids)
    clear = index_images(split_root / "clear")
    expected_scenes = NUM_SCENES if split == "train" else NUM_TEST_SCENES
    if len(clear) != expected_scenes:
        raise ValueError(
            f"CDD11 {split} must contain {expected_scenes} clear scenes, got {len(clear)}"
        )
    all_scene_ids = set(clear)

    source_images: dict[str, dict[str, Path]] = {}
    for _, pair in pairs:
        source_images[pair] = index_images(split_root / pair)
        if set(source_images[pair]) != all_scene_ids:
            raise ValueError(f"CDD11 scene mismatch in {split}/{pair}")

    metadata_path = coarse_root / "coarse_preparation.json"
    metadata_pairs = {pair_id: pair for pair_id, pair in pairs}
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        existing_split = existing.get("split", "train")
        if existing_split != split:
            raise ValueError(
                f"{coarse_root} contains {existing_split} coarse images; use a separate "
                f"directory for CDD11 {split}"
            )
        if (
            existing.get("data_root")
            and Path(existing["data_root"]).resolve() != data_root
        ):
            raise ValueError(
                f"{coarse_root} was prepared from {existing['data_root']}, not {data_root}"
            )
        requested_checkpoint = Path(dacg_checkpoint).resolve()
        if (
            existing.get("dacg_checkpoint")
            and Path(existing["dacg_checkpoint"]).resolve() != requested_checkpoint
        ):
            raise ValueError(
                f"{coarse_root} was prepared with a different DACG checkpoint; "
                "use a separate coarse directory to prevent mixed predictions"
            )
        for value in existing.get("degradation_pairs", []):
            metadata_pairs[int(value["id"])] = str(value["name"])
    elif split == "test" and any(
        (coarse_root / pair).is_dir() and any((coarse_root / pair).iterdir())
        for _, pair in pairs
    ):
        raise ValueError(
            "Refusing to label existing unverified files as official test coarse images; "
            "use a new --coarse-root, or pass an existing test coarse root directly to "
            "the evaluator with --allow-unverified-test-coarse"
        )

    jobs: list[tuple[Path, Path]] = []
    for _, pair in pairs:
        for scene_id, source in source_images[pair].items():
            output = coarse_root / pair / f"{scene_id}.png"
            if not output.is_file():
                jobs.append((source, output))

    info = {
        "status": "preparing" if jobs else "completed",
        "split": split,
        "data_root": str(data_root),
        "dacg_checkpoint": str(Path(dacg_checkpoint).resolve()),
        "coarse_root": str(coarse_root),
        "degradation_pairs": [
            {"id": pair_id, "name": metadata_pairs[pair_id]}
            for pair_id in sorted(metadata_pairs)
        ],
        "coarse_images": len(clear) * len(metadata_pairs),
    }
    coarse_root.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

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

    incomplete_pairs = []
    for pair in metadata_pairs.values():
        directory = coarse_root / pair
        if not directory.is_dir() or set(index_images(directory)) != all_scene_ids:
            incomplete_pairs.append(pair)
    if incomplete_pairs:
        info["status"] = "preparing"
        metadata_path.write_text(
            json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        raise RuntimeError(
            "Coarse preparation remains incomplete for: "
            + ", ".join(sorted(incomplete_pairs))
            + ". Rerun with the corresponding --degradation-pairs."
        )

    info["status"] = "completed"
    metadata_path.write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "coarse_root": coarse_root,
        "coarse_images": len(clear) * len(metadata_pairs),
    }


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
                    ground_truth = clear[scene_id]
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
                            "clear_image": str(ground_truth.resolve()),
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


def prepare_selective_test_manifest(
    *,
    data_root: str | Path,
    coarse_root: str | Path,
    output_dir: str | Path,
    pair_ids: list[int],
    require_coarse_metadata: bool = True,
) -> dict[str, Path | int]:
    """Build directed selective-restoration records from official CDD11/test."""

    data_root = Path(data_root).resolve()
    coarse_root = Path(coarse_root).resolve()
    output_dir = Path(output_dir).resolve()
    test_root = data_root / "test"
    pairs = selected_pairs(pair_ids)

    metadata_path = coarse_root / "coarse_preparation.json"
    if require_coarse_metadata:
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing {metadata_path}. Generate test coarse images with "
                "prepare_cdd11_coarse.py --split test, or explicitly pass "
                "--allow-unverified-test-coarse."
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("split") != "test" or metadata.get("status") != "completed":
            raise ValueError(
                f"Invalid official-test coarse metadata in {metadata_path}: "
                f"split={metadata.get('split')!r}, status={metadata.get('status')!r}"
            )
        if Path(metadata.get("data_root", "")).resolve() != data_root:
            raise ValueError(
                f"Test coarse metadata was prepared from {metadata.get('data_root')}, "
                f"not {data_root}"
            )
        prepared_pairs = {int(value["id"]) for value in metadata["degradation_pairs"]}
        missing_pairs = sorted(set(pair_ids) - prepared_pairs)
        if missing_pairs:
            raise ValueError(
                f"Test coarse metadata is missing pair IDs: {missing_pairs}"
            )

    clear = index_images(test_root / "clear")
    if len(clear) != NUM_TEST_SCENES:
        raise ValueError(
            f"CDD11 test must contain {NUM_TEST_SCENES} clear scenes, got {len(clear)}"
        )
    all_scene_ids = set(clear)
    source_images: dict[str, dict[str, Path]] = {}
    coarse_images: dict[str, dict[str, Path]] = {}
    target_images: dict[str, dict[str, Path]] = {}
    for _, pair in pairs:
        source_images[pair] = index_images(test_root / pair)
        coarse_images[pair] = index_images(coarse_root / pair)
        if set(source_images[pair]) != all_scene_ids:
            raise ValueError(f"CDD11 scene mismatch in test/{pair}")
        if set(coarse_images[pair]) != all_scene_ids:
            raise ValueError(
                f"CDD11 test coarse scene mismatch in {coarse_root / pair}"
            )
    for degradation in PROMPT_NAMES:
        target_images[degradation] = index_images(test_root / degradation)
        if set(target_images[degradation]) != all_scene_ids:
            raise ValueError(f"CDD11 scene mismatch in test/{degradation}")

    records: list[dict] = []
    for pair_id, pair in pairs:
        for scene_id in sorted(all_scene_ids):
            for task in directed_tasks(pair_id):
                records.append(
                    {
                        "id": f"test/{pair}/{scene_id}/remove-{task.remove}-preserve-{task.preserve}",
                        "split": "test",
                        "scene_id": scene_id,
                        "pair_id": pair_id,
                        "pair": pair,
                        "remove": task.remove,
                        "preserve": task.preserve,
                        "prompt": task.prompt,
                        "image": str(coarse_images[pair][scene_id].resolve()),
                        "ref_image": str(source_images[pair][scene_id].resolve()),
                        "target_image": str(
                            target_images[task.preserve][scene_id].resolve()
                        ),
                        "clear_image": str(clear[scene_id].resolve()),
                    }
                )

    manifest = output_dir / "manifest.jsonl"
    _write_jsonl(manifest, records)
    return {
        "manifest": manifest,
        "test_scenes": len(clear),
        "test_samples": len(records),
        "coarse_metadata_verified": require_coarse_metadata,
    }
