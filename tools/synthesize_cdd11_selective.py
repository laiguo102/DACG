#!/usr/bin/env python3
"""Generate family-wise deterministic CDD-11 selective states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from cdd11_selective_synthesis.config import load_config, validate_config
from cdd11_selective_synthesis.io import (
    index_images,
    load_clean,
    load_depth_map,
    load_light_map,
    load_mask,
    save_image,
)
from cdd11_selective_synthesis.manifest import family_records, write_jsonl
from cdd11_selective_synthesis.params import sample_family_params
from cdd11_selective_synthesis.renderer import render_family
from cdd11_selective_synthesis.split import (
    canonical_split,
    deterministic_scene_split,
    load_split_file,
    mask_pool_directory,
    validate_disjoint_mask_pools,
)
from cdd11_selective_synthesis.states import STATE_NAMES, VALID_STATE_TUPLES
from cdd11_selective_synthesis.validation import (
    build_dataset_stats,
    build_distribution_report,
    is_complete_family,
    load_family_meta,
    validate_family,
    validate_manifest,
    write_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML synthesis configuration")
    parser.add_argument("--split", choices=("train", "val", "validation", "test"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _source_indices(config):
    clean = index_images(config.clean_root)
    light = index_images(config.light_root)
    depth = index_images(config.depth_root)
    if set(clean) != set(light) or set(clean) != set(depth):
        raise ValueError(
            "Clean, light-map, and depth-map stems must match exactly: "
            f"clean={len(clean)}, light={len(light)}, depth={len(depth)}"
        )
    return clean, light, depth


def _split_map(config, scene_ids):
    if config.split_file is not None:
        split_map = load_split_file(config.split_file)
        all_split_scenes = set().union(*(set(values) for values in split_map.values()))
        if all_split_scenes != set(scene_ids):
            raise ValueError("Split file scene IDs do not match clean_root")
        return split_map
    return deterministic_scene_split(
        scene_ids,
        seed=config.seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )


def _validate_mask_pools(config):
    if not config.separate_mask_pools:
        return
    rain_roots = {}
    snow_roots = {}
    for split in ("train", "val", "test"):
        rain_root = mask_pool_directory(config.rain_mask_root, split, True)
        snow_root = mask_pool_directory(config.snow_mask_root, split, True)
        rain_roots[split] = list(index_images(rain_root).values())
        snow_roots[split] = list(index_images(snow_root).values())
    validate_disjoint_mask_pools(rain_roots)
    validate_disjoint_mask_pools(snow_roots)


def _family_metadata(params, source_paths, output_root):
    metadata = params.to_dict()
    metadata.update(
        {
            "schema_version": 1,
            "render_order": ["low", "rain_or_snow", "haze"],
            "states": [STATE_NAMES[state] for state in VALID_STATE_TUPLES],
            "sources": {
                name: str(Path(path).expanduser().resolve())
                for name, path in source_paths.items()
            },
            "output_root": str(output_root.resolve()),
        }
    )
    return metadata


def _family_dir(output_root: Path, split: str, scene_id: str, family_id: int) -> Path:
    return output_root / split / str(scene_id) / f"family_{family_id:03d}"


def _render_one_family(config, split, scene_id, family_id, clean_path, light_path, depth_path, rain_pool, snow_pool, overwrite):
    directory = _family_dir(config.output_root, split, scene_id, family_id)
    if is_complete_family(directory) and not overwrite:
        metadata = validate_family(directory)
        print(f"[SKIP] {split} {scene_id} family_{family_id:03d}", flush=True)
        return directory, metadata

    params = sample_family_params(
        scene_id=scene_id,
        family_id=family_id,
        rain_mask_pool=rain_pool,
        snow_mask_pool=snow_pool,
        global_seed=config.seed,
        gamma_min=config.gamma_min,
        gamma_max=config.gamma_max,
        noise_sigma_min=config.noise_sigma_min,
        noise_sigma_max=config.noise_sigma_max,
        beta_min=config.beta_min,
        beta_max=config.beta_max,
        atmospheric_light_min=config.atmospheric_light_min,
        atmospheric_light_max=config.atmospheric_light_max,
    )
    print(f"[BUILD] {split} {scene_id} family_{family_id:03d}", flush=True)
    clean = load_clean(clean_path)
    light_map = load_light_map(light_path)
    depth_map = load_depth_map(depth_path)
    rain_mask = load_mask(params.rain.mask_path)
    snow_mask = load_mask(params.snow.mask_path)
    rendered = render_family(
        clean=clean,
        light_map=light_map,
        depth_map=depth_map,
        rain_mask=rain_mask,
        snow_mask=snow_mask,
        params=params,
    )
    directory.mkdir(parents=True, exist_ok=True)
    for state, image in rendered.items():
        save_image(image, directory / f"{STATE_NAMES[state]}.png")
    metadata = _family_metadata(
        params,
        {
            "clean": clean_path,
            "light_map": light_path,
            "depth_map": depth_path,
            "rain_mask": params.rain.mask_path,
            "snow_mask": params.snow.mask_path,
        },
        config.output_root,
    )
    if config.save_meta:
        write_json(directory / "meta.json", metadata)
    validate_family(directory)
    return directory, metadata


def _run_split(config, split, source_indices, split_map, args):
    clean, light, depth = source_indices
    scene_ids = list(split_map[split])
    if args.max_scenes is not None:
        if args.max_scenes < 1:
            raise ValueError("--max-scenes must be positive")
        scene_ids = scene_ids[: args.max_scenes]
    families = config.families_per_scene[split]
    if families == 0 or not scene_ids:
        return None

    rain_root = mask_pool_directory(
        config.rain_mask_root, split, config.separate_mask_pools
    )
    snow_root = mask_pool_directory(
        config.snow_mask_root, split, config.separate_mask_pools
    )
    rain_pool = list(index_images(rain_root).values())
    snow_pool = list(index_images(snow_root).values())
    if args.dry_run:
        expected_families = len(scene_ids) * families
        expected_images = expected_families * len(VALID_STATE_TUPLES)
        expected_pairs = expected_families * 20
        print(
            f"split={split} scenes={len(scene_ids)} families={expected_families} "
            f"images={expected_images} selective_pairs={expected_pairs} "
            f"output={config.output_root / split}",
            flush=True,
        )
        return None

    artifacts = []
    for scene_id in scene_ids:
        for family_id in range(families):
            artifact = _render_one_family(
                config,
                split,
                scene_id,
                family_id,
                clean[scene_id],
                light[scene_id],
                depth[scene_id],
                rain_pool,
                snow_pool,
                args.overwrite or config.overwrite,
            )
            artifacts.append(artifact)

    records = []
    metadata = []
    for directory, meta in artifacts:
        records.extend(
            family_records(
                split=split,
                scene_id=meta["scene_id"],
                family_id=int(meta["family_id"]),
                family_directory=directory,
            )
        )
        metadata.append(meta)
    validate_manifest(records, expected_families=len(artifacts))
    config.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = write_jsonl(config.output_root / f"{split}_manifest.jsonl", records)
    stats = build_dataset_stats(metadata, records)
    stats_path = config.output_root / "dataset_stats.json"
    all_stats = {}
    if stats_path.is_file():
        all_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    all_stats[split] = stats
    write_json(stats_path, all_stats)
    report_path = config.output_root / "distribution_report.json"
    all_reports = {}
    if report_path.is_file():
        all_reports = json.loads(report_path.read_text(encoding="utf-8"))
    all_reports[split] = build_distribution_report(metadata)
    write_json(report_path, all_reports)
    print(
        f"[DONE] {split}: families={len(artifacts)} images={stats['images']} "
        f"pairs={stats['selective_pairs']} manifest={manifest_path}",
        flush=True,
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    validate_config(config)
    requested = [canonical_split(args.split)] if args.split else ["train", "val", "test"]
    source_indices = _source_indices(config)
    _validate_mask_pools(config)
    split_map = _split_map(config, source_indices[0].keys())
    for split in requested:
        _run_split(config, split, source_indices, split_map, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
