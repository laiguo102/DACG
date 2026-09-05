"""Build two-scene pixel-beta galleries beside saved latent/skip CFG results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .cfg import blend_cfg_pixels
from .cfg_evaluate import beta_label, normalize_betas, save_contact_sheet
from .data import _image_tensor
from .protocol import PAIR_FOLDERS, directed_tasks


DEFAULT_BETAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.05, 1.1, 1.2)


def _safe_id(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace("\\", "__")


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _save_rgb(value: np.ndarray, path: Path) -> None:
    from PIL import Image

    array = np.clip(value, 0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(array * 255).astype(np.uint8), "RGB").save(path)


def _to_display_tensor(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(value).permute(2, 0, 1)


def _reference_tensor(path: str, resolution: int) -> torch.Tensor:
    """Use the exact CCDD dataset resize and normalization path."""

    return _image_tensor(path, resolution).add(1).mul(0.5).clamp(0, 1)


def _expected_task_keys() -> set[tuple[int, str, str, str]]:
    return {
        (pair_id, pair, task.remove, task.preserve)
        for pair_id, pair in PAIR_FOLDERS.items()
        for task in directed_tasks(pair_id)
    }


def _load_cfg_gallery_records(cfg_dir: Path) -> list[dict]:
    records_dir = cfg_dir / "records"
    gallery_dir = cfg_dir / "gallery"
    if not records_dir.is_dir():
        raise FileNotFoundError(records_dir)
    if not gallery_dir.is_dir():
        raise FileNotFoundError(gallery_dir)

    by_gallery_name: dict[str, dict] = {}
    for path in sorted(records_dir.rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        sample_id = str(record.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"CFG record has no sample_id: {path}")
        name = _safe_id(sample_id)
        if name in by_gallery_name:
            raise ValueError(f"Duplicate CFG record for gallery ID {name}")
        by_gallery_name[name] = record

    result = []
    for gallery in sorted(path for path in gallery_dir.iterdir() if path.is_dir()):
        if gallery.name not in by_gallery_name:
            raise ValueError(f"No CFG record matches gallery directory {gallery}")
        record = dict(by_gallery_name[gallery.name])
        record["cfg_gallery_dir"] = str(gallery.resolve())
        result.append(record)
    if not result:
        raise ValueError("No CFG gallery samples were found")
    return result


def _task_key(record: dict) -> tuple[int, str, str, str]:
    return (
        int(record["pair_id"]),
        str(record["pair"]),
        str(record["remove"]),
        str(record["preserve"]),
    )


def select_two_complete_scenes(
    records: list[dict], explicit_scene_ids: list[str] | None
) -> tuple[list[str], list[dict]]:
    """Select exactly two scenes containing each of the ten directed tasks."""

    expected = _expected_task_keys()
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_scene[str(record["scene_id"])].append(record)

    complete: dict[str, list[dict]] = {}
    incomplete: dict[str, tuple[set[tuple], set[tuple]]] = {}
    for scene_id, scene_records in by_scene.items():
        keys = [_task_key(record) for record in scene_records]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Scene {scene_id} contains duplicate directed tasks")
        missing = expected - set(keys)
        extra = set(keys) - expected
        if not missing and not extra:
            complete[scene_id] = scene_records
        else:
            incomplete[scene_id] = (missing, extra)

    if explicit_scene_ids is None:
        scene_ids = sorted(complete)
        if len(scene_ids) != 2:
            raise ValueError(
                "Expected exactly two complete gallery scenes, found "
                f"{len(scene_ids)}: {scene_ids}. Pass --scene-ids SCENE_A SCENE_B."
            )
    else:
        scene_ids = [str(value) for value in explicit_scene_ids]
        if len(scene_ids) != 2 or len(set(scene_ids)) != 2:
            raise ValueError("--scene-ids requires exactly two distinct scene IDs")
        unknown = [scene_id for scene_id in scene_ids if scene_id not in by_scene]
        if unknown:
            raise ValueError(f"Scene IDs are absent from the CFG gallery: {unknown}")
        not_complete = [scene_id for scene_id in scene_ids if scene_id not in complete]
        if not_complete:
            details = {
                scene_id: {
                    "missing": sorted(incomplete[scene_id][0]),
                    "extra": sorted(incomplete[scene_id][1]),
                }
                for scene_id in not_complete
            }
            raise ValueError(f"Selected gallery scenes are incomplete: {details}")

    selected = []
    for scene_id in scene_ids:
        indexed = {_task_key(record): record for record in complete[scene_id]}
        for pair_id, pair in PAIR_FOLDERS.items():
            for task in directed_tasks(pair_id):
                selected.append(indexed[(pair_id, pair, task.remove, task.preserve)])
    if len(selected) != 20:
        raise RuntimeError(f"Expected 20 selected tasks, got {len(selected)}")
    return scene_ids, selected


def _load_references(record: dict, resolution: int) -> dict[str, torch.Tensor]:
    return {
        "negative_identity": _reference_tensor(record["degraded_path"], resolution),
        "coarse": _reference_tensor(record["coarse_path"], resolution),
        "positive_target": _reference_tensor(
            record["positive_target_path"], resolution
        ),
        "clean_gt": _reference_tensor(record["clean_gt_path"], resolution),
    }


def _log_wandb(
    args: argparse.Namespace,
    config: dict,
    output_records: list[dict],
) -> None:
    if args.report_to == "none":
        return
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("wandb is required when --report-to wandb") from error

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        dir=str(args.output_dir),
        config=config,
        job_type="pixel-cfg-comparison",
    )
    rows = [
        [
            record["scene_id"],
            record["pair"],
            record["remove"],
            record["preserve"],
            wandb.Image(record["cfg_contact_sheet_path"]),
            wandb.Image(record["pixel_contact_sheet_path"]),
        ]
        for record in output_records
    ]
    run.log(
        {
            "pixel_cfg/gallery": wandb.Table(
                columns=[
                    "scene_id",
                    "pair",
                    "remove",
                    "preserve",
                    "cfg_gallery",
                    "pixel_gallery",
                ],
                data=rows,
            )
        }
    )
    run.summary["scene_ids"] = config["scene_ids"]
    run.summary["gallery_rows"] = len(rows)
    run.finish()


def run(args: argparse.Namespace) -> Path:
    cfg_dir = args.cfg_dir.resolve()
    state_path = cfg_dir / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "completed":
        raise ValueError("CFG evaluation must be completed before pixel comparison")
    cfg_config = state.get("config", {})
    protocol = cfg_config.get("protocol", "")
    if protocol != "ccdd11-endpoint-correct-state-cfg-validation-v2":
        raise ValueError(f"Unsupported CFG protocol: {protocol!r}")
    resolution = int(cfg_config.get("resolution", 0))
    if resolution != 512:
        raise ValueError(f"Expected CFG resolution 512, got {resolution}")

    betas = normalize_betas(
        list(args.betas) if args.betas is not None else cfg_config["betas"]
    )
    if len(betas) != 8:
        raise ValueError(
            "The 4x3 gallery requires exactly eight beta values; "
            f"got {len(betas)}"
        )
    if 0.0 not in betas or 1.0 not in betas:
        raise ValueError("Pixel blending requires beta=0 and beta=1 endpoints")

    gallery_records = _load_cfg_gallery_records(cfg_dir)
    scene_ids, selected_records = select_two_complete_scenes(
        gallery_records, args.scene_ids
    )
    output_dir = (args.output_dir or cfg_dir / "pixel_beta_comparison").resolve()
    args.output_dir = output_dir
    report_path = output_dir / "comparison.json"
    if report_path.exists():
        raise RuntimeError(f"Completed pixel comparison already exists: {report_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "protocol": "ccdd11-two-scene-pixel-beta-gallery-v2",
        "source_cfg_dir": str(cfg_dir),
        "source_cfg_protocol": protocol,
        "scene_ids": scene_ids,
        "task_count": len(selected_records),
        "resolution": resolution,
        "betas": betas,
        "formula": (
            "I_pixel = clip(I_negative + beta * "
            "(I_positive - I_negative), 0, 1)"
        ),
        "endpoint_images": {"negative": "beta_0.png", "positive": "beta_1.png"},
    }
    state_output = output_dir / "state.json"
    state_output.write_text(
        json.dumps({"status": "generating", "config": config}, indent=2) + "\n",
        encoding="utf-8",
    )

    output_records = []
    for record in selected_records:
        cfg_gallery = Path(record["cfg_gallery_dir"])
        cfg_contact_sheet = cfg_gallery / "contact_sheet.png"
        if not cfg_contact_sheet.is_file():
            raise FileNotFoundError(cfg_contact_sheet)
        negative = _load_rgb(cfg_gallery / "beta_0.png")
        positive = _load_rgb(cfg_gallery / "beta_1.png")
        if positive.shape != negative.shape:
            raise ValueError(f"Endpoint image shape mismatch in {cfg_gallery}")
        if positive.shape[:2] != (resolution, resolution):
            raise ValueError(
                f"Expected {resolution}x{resolution} CFG images in {cfg_gallery}, "
                f"got {positive.shape[:2]}"
            )

        sample_output = output_dir / _safe_id(record["sample_id"])
        predictions: dict[float, torch.Tensor] = {}
        prediction_paths = {}
        for beta in betas:
            label = beta_label(beta)
            cfg_beta = cfg_gallery / f"beta_{label}.png"
            if not cfg_beta.is_file():
                raise FileNotFoundError(cfg_beta)
            pixel = blend_cfg_pixels(positive, negative, beta)
            destination = sample_output / f"pixel_beta_{label}.png"
            _save_rgb(pixel, destination)
            predictions[beta] = _to_display_tensor(pixel)
            prediction_paths[label] = str(destination)

        pixel_contact_sheet = sample_output / "pixel_contact_sheet.png"
        save_contact_sheet(
            _load_references(record, resolution),
            predictions,
            pixel_contact_sheet,
        )
        output_records.append(
            {
                "sample_id": record["sample_id"],
                "scene_id": str(record["scene_id"]),
                "pair_id": int(record["pair_id"]),
                "pair": record["pair"],
                "remove": record["remove"],
                "preserve": record["preserve"],
                "cfg_contact_sheet_path": str(cfg_contact_sheet.resolve()),
                "pixel_contact_sheet_path": str(pixel_contact_sheet.resolve()),
                "pixel_prediction_paths": prediction_paths,
            }
        )

    if len(output_records) != 20:
        raise RuntimeError(f"Expected 20 output galleries, got {len(output_records)}")
    state_output.write_text(
        json.dumps(
            {"status": "local_completed_reporting_pending", "config": config},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _log_wandb(args, config, output_records)
    report_path.write_text(
        json.dumps(
            {"config": config, "records": output_records},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    state_output.write_text(
        json.dumps({"status": "completed", "config": config}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Created {len(output_records)} pixel-beta galleries for scenes "
        f"{', '.join(scene_ids)}: {output_dir}",
        flush=True,
    )
    return output_dir


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--cfg-dir",
        type=Path,
        required=True,
        help="Completed evaluate_ccdd11_cfg.py output directory",
    )
    value.add_argument("--output-dir", type=Path)
    value.add_argument(
        "--scene-ids",
        nargs=2,
        metavar=("SCENE_A", "SCENE_B"),
        help="Two complete CFG gallery scenes; defaults to auto-detection",
    )
    value.add_argument(
        "--betas", nargs="+", type=float, help="Defaults to the CFG run's beta grid"
    )
    value.add_argument("--report-to", choices=("wandb", "none"), default="wandb")
    value.add_argument("--wandb-entity", default="c14150591-sjtu")
    value.add_argument("--wandb-project", default="difix-ccdd11-selective")
    value.add_argument(
        "--wandb-run-name",
        default="ccdd-all5-100k-best-state-pixel-beta-comparison-v2",
    )
    return value


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
