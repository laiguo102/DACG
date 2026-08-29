"""Audit CCDD-11 native selective targets and render target-semantics montages."""

from __future__ import annotations

import argparse
import json
import math
import random
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from difix3d_selective.ccdd_prepare import (
    DATASET_NAME,
    native_target_path,
    strict_index_images,
    strict_index_scene_directories,
)
from difix3d_selective.protocol import NUM_SCENES, directed_tasks, make_scene_split, selected_pairs


def _decode_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            rgb.load()
            return rgb.size
    except Exception as error:
        raise ValueError(f"PIL could not decode {path}: {error}") from error


def _load_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _psnr(first: np.ndarray, second: np.ndarray) -> float:
    error = np.mean((first.astype(np.float64) - second.astype(np.float64)) ** 2)
    return math.inf if error == 0 else 20 * math.log10(255.0 / math.sqrt(error))


def _montage(paths: list[Path], labels: list[str], header: str, output: Path) -> None:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    width = sum(image.width for image in images)
    label_height = 24
    header_lines = textwrap.wrap(header, width=150) or [header]
    header_height = 18 * len(header_lines) + 12
    canvas = Image.new("RGB", (width, max(image.height for image in images) + label_height + header_height), "white")
    draw = ImageDraw.Draw(canvas)
    y = 6
    for line in header_lines:
        draw.text((6, y), line, fill="black")
        y += 18
    left = 0
    for image, label in zip(images, labels):
        draw.text((left + 5, header_height + 4), label, fill="black")
        canvas.paste(image, (left, header_height + label_height))
        left += image.width
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def audit(data_root: str | Path, pair_ids: list[int], audit_output: str | Path) -> dict:
    data_root = Path(data_root).resolve()
    audit_output = Path(audit_output).resolve()
    main_root = data_root / "half_train" / "main_data"
    sub_root = data_root / "half_train" / "sub_data"
    pairs = selected_pairs(pair_ids)
    clear = strict_index_images(main_root / "clear")
    if len(clear) != NUM_SCENES:
        raise ValueError(f"CCDD-11 half_train clear scenes: expected {NUM_SCENES}, got {len(clear)}")
    scene_ids = set(clear)
    splits = make_scene_split(list(clear))
    if set(splits["train"]) & set(splits["validation"]):
        raise RuntimeError("CCDD-11 train/validation scene leakage")
    if set(splits["train"]) | set(splits["validation"]) != scene_ids:
        raise RuntimeError("CCDD-11 split does not cover exactly the clear scenes")

    main_components = {
        component: strict_index_images(main_root / component)
        for component in {value for _, pair in pairs for value in pair.split("_")}
    }
    for component, images in main_components.items():
        if set(images) != scene_ids:
            raise ValueError(f"CCDD-11 main_data/{component} scene mismatch")

    decoded_sizes: dict[Path, tuple[int, int]] = {}

    def decoded_size(path: Path):
        resolved = path.resolve()
        if resolved not in decoded_sizes:
            decoded_sizes[resolved] = _decode_size(resolved)
        return decoded_sizes[resolved]

    pair_reports: dict[str, dict] = {}
    targets: dict[tuple[str, str, str], Path] = {}
    sources: dict[str, dict[str, Path]] = {}
    for pair_id, pair in pairs:
        print(f"Auditing {pair}: {len(scene_ids)} scenes", flush=True)
        sources[pair] = strict_index_images(main_root / pair)
        if set(sources[pair]) != scene_ids:
            raise ValueError(f"CCDD-11 main_data/{pair} scene mismatch")
        sub_scenes = strict_index_scene_directories(sub_root / pair)
        if set(sub_scenes) != scene_ids:
            raise ValueError(f"CCDD-11 sub_data/{pair} scene mismatch")
        target_counts = {component: 0 for component in pair.split("_")}
        half_count = 0
        for scene_id in sorted(scene_ids):
            source_size = decoded_size(sources[pair][scene_id])
            clear_size = decoded_size(clear[scene_id])
            if clear_size != source_size:
                raise ValueError(f"Image size mismatch for {pair}/{scene_id}: source={source_size}, clear={clear_size}")
            for task in directed_tasks(pair_id):
                target = native_target_path(sub_root, pair, scene_id, task.preserve)
                if not target.is_file():
                    raise FileNotFoundError(target)
                targets[(pair, scene_id, task.preserve)] = target
                target_size = decoded_size(target)
                preserve_size = decoded_size(main_components[task.preserve][scene_id])
                if target_size != source_size or preserve_size != source_size:
                    raise ValueError(
                        f"Image size mismatch for {pair}/{scene_id}/{task.preserve}: "
                        f"source={source_size}, target={target_size}, main_preserve={preserve_size}"
                    )
                target_counts[task.preserve] += 1
            if (sub_scenes[scene_id] / f"{scene_id}_half_.png").is_file():
                half_count += 1
        pair_reports[pair] = {
            "source": len(sources[pair]),
            "sub_scenes": len(sub_scenes),
            "targets": target_counts,
            "half": half_count,
        }
        print(
            f"Completed {pair}: source={len(sources[pair])}, "
            f"targets={sum(target_counts.values())}, half={half_count}",
            flush=True,
        )

    montage_scene = random.Random(42).choice(sorted(scene_ids))
    semantic_rows = []
    semantics_root = audit_output / "target_semantics"
    for pair_id, pair in pairs:
        for task in directed_tasks(pair_id):
            target = targets[(pair, montage_scene, task.preserve)]
            preserve = main_components[task.preserve][montage_scene]
            target_array = _load_array(target)
            preserve_array = _load_array(preserve)
            diagnostic_psnr = _psnr(target_array, preserve_array)
            output = semantics_root / f"{pair}_remove-{task.remove}_preserve-{task.preserve}.png"
            header = (
                f"pair={pair} | remove={task.remove} | preserve={task.preserve} | "
                f"target={target} | diagnostic PSNR(target, main preserve)={diagnostic_psnr:.4f}"
            )
            _montage(
                [sources[pair][montage_scene], target, preserve, clear[montage_scene]],
                ["original composite", "native selective target", "main_data preserve", "clean"],
                header,
                output,
            )
            semantic_rows.append(
                {
                    "pair": pair,
                    "remove": task.remove,
                    "preserve": task.preserve,
                    "target": str(target.resolve()),
                    "diagnostic_psnr_vs_main_preserve": diagnostic_psnr,
                    "montage": str(output),
                }
            )

    report = {
        "dataset": DATASET_NAME,
        "status": "PASS",
        "data_root": str(data_root),
        "scenes": len(clear),
        "train_scenes": len(splits["train"]),
        "validation_scenes": len(splits["validation"]),
        "pairs": pair_reports,
        "directed_tasks": len(pairs) * 2,
        "expected_coarse": len(clear) * len(pairs),
        "expected_train_records": len(splits["train"]) * len(pairs) * 2,
        "expected_validation_records": len(splits["validation"]) * len(pairs) * 2,
        "decoded_required_images": len(decoded_sizes),
        "semantic_audit_scene": montage_scene,
        "target_semantics": semantic_rows,
    }
    audit_output.mkdir(parents=True, exist_ok=True)
    (audit_output / "ccdd11_selective_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument(
        "--degradation-pairs", nargs="+", type=int, choices=range(1, 6),
        default=[1, 2, 3, 4, 5],
    )
    value.add_argument("--audit-output", type=Path, default=Path("audit"))
    return value


def main() -> None:
    args = parser().parse_args()
    report = audit(args.data_root, args.degradation_pairs, args.audit_output)
    print("CCDD-11 selective audit\n")
    print(f"scenes: {report['scenes']}\n")
    for pair, values in report["pairs"].items():
        print(f"{pair}:")
        print(f"  source: {values['source']}")
        for component, count in values["targets"].items():
            print(f"  target {component}: {count}")
        print(f"  half: {values['half']}\n")
    print(f"directed tasks: {report['directed_tasks']}")
    print(f"expected coarse: {report['expected_coarse']}")
    print(f"expected train records: {report['expected_train_records']}")
    print(f"expected val records: {report['expected_validation_records']}\n")
    print("status: PASS")


if __name__ == "__main__":
    main()
