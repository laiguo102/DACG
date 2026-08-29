"""CCDD-11 adapters for coarse preparation and selective Difix manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch

from .prepare import (
    IMAGE_SUFFIXES,
    image_tensor,
    load_dacg_checkpoint,
    save_tensor,
    tiled_forward,
)
from .protocol import (
    NUM_SCENES,
    NUM_TEST_SCENES,
    directed_tasks,
    make_scene_split,
    selected_pairs,
)


DATASET_NAME = "CCDD-11"
TARGET_KIND = "native_selective_sub_data"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_index_images(directory: str | Path) -> dict[str, Path]:
    """Index images by stem while rejecting ambiguous duplicate scene IDs."""

    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in result:
            raise ValueError(
                f"Duplicate scene ID {path.stem!r} in {directory}: "
                f"{result[path.stem].name}, {path.name}"
            )
        result[path.stem] = path
    return result


def strict_index_scene_directories(directory: str | Path) -> dict[str, Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_dir():
            continue
        if path.name in result:
            raise ValueError(f"Duplicate scene directory {path.name!r} in {directory}")
        result[path.name] = path
    return result


def native_target_path(sub_root: str | Path, pair: str, scene_id: str, preserve: str) -> Path:
    """Return the CCDD native target; the filename suffix is the preserved component."""

    return Path(sub_root) / pair / scene_id / f"{scene_id}_{preserve}_.png"


def _assert_same_scenes(label: str, actual: Iterable[str], expected: set[str]) -> None:
    actual_set = set(actual)
    if actual_set == expected:
        return
    missing = sorted(expected - actual_set)[:10]
    extra = sorted(actual_set - expected)[:10]
    raise ValueError(f"CCDD-11 scene mismatch in {label}; missing={missing}, extra={extra}")


def _contains_component(path: str | Path, component: str) -> bool:
    return component.casefold() in {part.casefold() for part in Path(path).parts}


def validate_manifest_records(records: list[dict], pair_ids: list[int]) -> None:
    allowed_pairs = {pair for _, pair in selected_pairs(pair_ids)}
    for record in records:
        pair = record.get("pair")
        preserve = record.get("preserve")
        remove = record.get("remove")
        if pair not in allowed_pairs or remove == preserve:
            raise ValueError(f"Invalid directed CCDD-11 record: {record.get('id')}")
        if record.get("dataset") != DATASET_NAME or record.get("target_kind") != TARGET_KIND:
            raise ValueError(f"Missing CCDD-11 identity in record: {record.get('id')}")
        target = Path(record["target_image"])
        if target.name != f"{record['scene_id']}_{preserve}_.png":
            raise ValueError(f"Reversed or malformed CCDD-11 target: {target}")
        if "_half_" in target.name or _contains_component(target, "half_test"):
            raise ValueError(f"Forbidden CCDD-11 training target: {target}")
        for key in ("image", "ref_image", "target_image", "clear_image"):
            if _contains_component(record[key], "half_test"):
                raise ValueError(f"half_test leakage in {key}: {record[key]}")
        expected_prompt = next(
            task.prompt
            for task in directed_tasks(int(record["pair_id"]))
            if task.remove == remove and task.preserve == preserve
        )
        if record.get("prompt") != expected_prompt:
            raise ValueError(f"Prompt mismatch in record: {record.get('id')}")


def build_ccdd11_records(
    *,
    split: str,
    scene_ids: list[str],
    pairs: list[tuple[int, str]],
    clear: dict[str, Path],
    source_images: dict[str, dict[str, Path]],
    coarse_images: dict[str, dict[str, Path]],
    sub_root: Path,
) -> list[dict]:
    """Pure record builder kept separate so miniature fixtures can test counts."""

    records: list[dict] = []
    for pair_id, pair in pairs:
        for scene_id in scene_ids:
            for task in directed_tasks(pair_id):
                target = native_target_path(sub_root, pair, scene_id, task.preserve)
                if not target.is_file():
                    raise FileNotFoundError(target)
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
                        "image": str(coarse_images[pair][scene_id].resolve()),
                        "ref_image": str(source_images[pair][scene_id].resolve()),
                        "target_image": str(target.resolve()),
                        "clear_image": str(clear[scene_id].resolve()),
                        "dataset": DATASET_NAME,
                        "target_kind": TARGET_KIND,
                    }
                )
    return records


def _load_and_validate_coarse_metadata(
    coarse_root: Path,
    data_root: Path,
    source_root: Path,
    pairs: list[tuple[int, str]],
) -> dict:
    path = coarse_root / "coarse_preparation.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing CCDD-11 coarse metadata: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("dataset") != DATASET_NAME or metadata.get("split") != "half_train":
        raise ValueError(f"Not completed CCDD-11 half_train coarse metadata: {path}")
    if metadata.get("status") != "completed":
        raise ValueError(f"CCDD-11 coarse preparation is not completed: {path}")
    for key, expected in (("data_root", data_root), ("source_root", source_root)):
        if not metadata.get(key) or Path(metadata[key]).resolve() != expected:
            raise ValueError(f"CCDD-11 coarse metadata {key} mismatch in {path}")
    available_ids = {int(value) for value in metadata.get("pair_ids", [])}
    requested_ids = {pair_id for pair_id, _ in pairs}
    if not requested_ids <= available_ids:
        raise ValueError(f"CCDD-11 coarse metadata lacks pair IDs {sorted(requested_ids - available_ids)}")
    if int(metadata.get("expected_scenes", -1)) != NUM_SCENES:
        raise ValueError(f"CCDD-11 coarse metadata expected_scenes mismatch in {path}")
    return metadata


def prepare_ccdd11_selective_manifests(
    *, data_root: str | Path, coarse_root: str | Path, output_dir: str | Path, pair_ids: list[int]
) -> dict[str, Path | int]:
    data_root = Path(data_root).resolve()
    coarse_root = Path(coarse_root).resolve()
    output_dir = Path(output_dir).resolve()
    main_root = data_root / "half_train" / "main_data"
    sub_root = data_root / "half_train" / "sub_data"
    pairs = selected_pairs(pair_ids)

    if _contains_component(main_root, "half_test") or _contains_component(sub_root, "half_test"):
        raise ValueError("CCDD-11 training preparation may not use half_test")
    clear = strict_index_images(main_root / "clear")
    if len(clear) != NUM_SCENES:
        raise ValueError(f"CCDD-11 half_train must contain {NUM_SCENES} clear scenes, got {len(clear)}")
    all_scene_ids = set(clear)
    splits = make_scene_split(list(clear))
    if set(splits["train"]) & set(splits["validation"]):
        raise RuntimeError("CCDD-11 train/validation scene leakage")

    coarse_metadata = _load_and_validate_coarse_metadata(
        coarse_root, data_root, main_root.resolve(), pairs
    )
    source_images: dict[str, dict[str, Path]] = {}
    coarse_images: dict[str, dict[str, Path]] = {}
    for _, pair in pairs:
        source_images[pair] = strict_index_images(main_root / pair)
        coarse_images[pair] = strict_index_images(coarse_root / pair)
        _assert_same_scenes(f"half_train/main_data/{pair}", source_images[pair], all_scene_ids)
        _assert_same_scenes(str(coarse_root / pair), coarse_images[pair], all_scene_ids)
        sub_scenes = strict_index_scene_directories(sub_root / pair)
        _assert_same_scenes(f"half_train/sub_data/{pair}", sub_scenes, all_scene_ids)

    manifests: dict[str, Path] = {}
    counts: dict[str, int] = {}
    for split, scene_ids in splits.items():
        records = build_ccdd11_records(
            split=split,
            scene_ids=scene_ids,
            pairs=pairs,
            clear=clear,
            source_images=source_images,
            coarse_images=coarse_images,
            sub_root=sub_root,
        )
        validate_manifest_records(records, pair_ids)
        manifest = output_dir / "prepared" / "manifests" / f"{split}.jsonl"
        _write_jsonl(manifest, records)
        manifests[split] = manifest
        counts[split] = len(records)

    expected_train = len(splits["train"]) * len(pairs) * 2
    expected_validation = len(splits["validation"]) * len(pairs) * 2
    if counts != {"train": expected_train, "validation": expected_validation}:
        raise RuntimeError(f"Unexpected CCDD-11 manifest counts: {counts}")
    info = {
        "dataset": DATASET_NAME,
        "target_kind": TARGET_KIND,
        "target_rule": "target filename suffix equals preserve",
        "seed": 42,
        "data_root": str(data_root),
        "coarse_root": str(coarse_root),
        "coarse_metadata": coarse_metadata,
        "degradation_pairs": [{"id": pair_id, "name": pair} for pair_id, pair in pairs],
        "train_scene_ids": splits["train"],
        "validation_scene_ids": splits["validation"],
        "coarse_images": len(clear) * len(pairs),
        "train_samples": counts["train"],
        "validation_samples": counts["validation"],
    }
    _write_json(output_dir / "prepared" / "split_and_preparation.json", info)
    return {
        "train_manifest": manifests["train"],
        "validation_manifest": manifests["validation"],
        "coarse_images": len(clear) * len(pairs),
        "train_samples": counts["train"],
        "validation_samples": counts["validation"],
    }


def generate_ccdd11_coarse(
    *,
    data_root: str | Path,
    dacg_checkpoint: str | Path,
    coarse_root: str | Path,
    pair_ids: list[int],
    device: torch.device,
    tile_size: int = 0,
    tile_overlap: int = 64,
    split: str = "half_train",
) -> dict[str, Path | int]:
    """Generate CCDD coarse images while reusing the shared DACG inference helpers."""

    if split not in ("half_train", "half_test"):
        raise ValueError(f"Unsupported CCDD-11 split: {split}")
    data_root = Path(data_root).resolve()
    parent_coarse_root = Path(coarse_root).resolve()
    destination = parent_coarse_root / split
    source_root = data_root / split / "main_data"
    pairs = selected_pairs(pair_ids)
    clear = strict_index_images(source_root / "clear")
    expected_scenes = NUM_SCENES if split == "half_train" else NUM_TEST_SCENES
    if len(clear) != expected_scenes:
        raise ValueError(
            f"CCDD-11 {split} must contain {expected_scenes} clear scenes, got {len(clear)}"
        )
    all_scene_ids = set(clear)
    sources: dict[str, dict[str, Path]] = {}
    for _, pair in pairs:
        sources[pair] = strict_index_images(source_root / pair)
        _assert_same_scenes(f"{split}/main_data/{pair}", sources[pair], all_scene_ids)

    metadata_path = destination / "coarse_preparation.json"
    checkpoint = Path(dacg_checkpoint).resolve()
    checkpoint_sha256 = file_sha256(checkpoint)
    existing_pair_ids: set[int] = set()
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        checks = {
            "dataset": DATASET_NAME,
            "split": split,
            "data_root": str(data_root),
            "source_root": str(source_root.resolve()),
            "dacg_checkpoint": str(checkpoint),
            "dacg_checkpoint_sha256": checkpoint_sha256,
        }
        for key, expected in checks.items():
            if existing.get(key) != expected:
                raise ValueError(f"Existing CCDD-11 coarse metadata {key} mismatch in {metadata_path}")
        existing_pair_ids = {int(value) for value in existing.get("pair_ids", [])}
    elif destination.is_dir() and any(destination.iterdir()):
        raise ValueError(f"Refusing to mix unverified files into {destination}")

    all_pair_ids = existing_pair_ids | {pair_id for pair_id, _ in pairs}
    jobs: list[tuple[Path, Path]] = []
    for _, pair in pairs:
        for scene_id, source in sources[pair].items():
            output = destination / pair / f"{scene_id}.png"
            if not output.is_file():
                jobs.append((source, output))
    info = {
        "dataset": DATASET_NAME,
        "status": "preparing" if jobs else "completed",
        "split": split,
        "data_root": str(data_root),
        "source_root": str(source_root.resolve()),
        "coarse_root": str(destination),
        "dacg_checkpoint": str(checkpoint),
        "dacg_checkpoint_sha256": checkpoint_sha256,
        "pair_ids": sorted(all_pair_ids),
        "degradation_pairs": [
            {"id": pair_id, "name": pair}
            for pair_id, pair in selected_pairs(sorted(all_pair_ids))
        ],
        "expected_scenes": expected_scenes,
        "expected_images": expected_scenes * len(all_pair_ids),
    }
    _write_json(metadata_path, info)

    if jobs:
        from tqdm.auto import tqdm

        model, checkpoint_payload = load_dacg_checkpoint(checkpoint, device)
        del checkpoint_payload
        with torch.inference_mode():
            for source, output in tqdm(jobs, desc=f"Generating CCDD-11 {split} DACG coarse"):
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

    incomplete = []
    for pair_id, pair in selected_pairs(sorted(all_pair_ids)):
        del pair_id
        directory = destination / pair
        if not directory.is_dir():
            incomplete.append(pair)
            continue
        try:
            _assert_same_scenes(str(directory), strict_index_images(directory), all_scene_ids)
        except ValueError:
            incomplete.append(pair)
    if incomplete:
        info["status"] = "preparing"
        _write_json(metadata_path, info)
        raise RuntimeError("CCDD-11 coarse preparation remains incomplete for: " + ", ".join(incomplete))
    info["status"] = "completed"
    _write_json(metadata_path, info)
    return {"coarse_root": destination, "coarse_images": expected_scenes * len(all_pair_ids)}
