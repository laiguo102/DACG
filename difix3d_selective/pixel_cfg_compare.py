"""Compare latent/skip CFG outputs with RGB pixel-space beta blending."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .cfg import blend_cfg_pixels
from .cfg_evaluate import beta_label, normalize_betas


def _load_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _save_rgb(value: np.ndarray, path: Path) -> None:
    array = np.clip(value, 0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(array * 255).astype(np.uint8), "RGB").save(path)


def _comparison_metrics(
    pixel: np.ndarray, latent: np.ndarray
) -> dict[str, float | bool | None]:
    difference = pixel.astype(np.float64) - latent.astype(np.float64)
    mse = float(np.mean(np.square(difference)))
    return {
        "exact_match": mse == 0,
        "pixel_vs_cfg_mae": float(np.mean(np.abs(difference))),
        "pixel_vs_cfg_mse": mse,
        "pixel_vs_cfg_psnr": None if mse == 0 else -10.0 * math.log10(mse),
        "pixel_vs_cfg_max_abs": float(np.max(np.abs(difference))),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_sheet(
    rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]], path: Path
) -> None:
    height, width = rows[0][1].shape[:2]
    header = 24
    sheet = Image.new("RGB", (width * 3, (height + header) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for row_index, (beta, cfg, pixel, difference) in enumerate(rows):
        top = row_index * (height + header)
        panels = (
            (f"CFG beta={beta}", cfg),
            (f"pixel beta={beta}", pixel),
            ("abs diff x4", difference),
        )
        for column, (label, value) in enumerate(panels):
            left = column * width
            draw.text((left + 5, top + 5), label, fill="black", font=font)
            panel_array = np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8)
            panel = Image.fromarray(panel_array, "RGB")
            sheet.paste(panel, (left, top + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _discover_samples(gallery: Path, sample_ids: list[str] | None) -> list[Path]:
    if not gallery.is_dir():
        raise FileNotFoundError(gallery)
    available = sorted(path for path in gallery.iterdir() if path.is_dir())
    if not sample_ids:
        return available
    by_name = {path.name: path for path in available}
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_name]
    if missing:
        raise ValueError(f"Unknown gallery sample IDs: {missing}")
    return [by_name[sample_id] for sample_id in sample_ids]


def run(args: argparse.Namespace) -> Path:
    cfg_dir = args.cfg_dir.resolve()
    gallery = cfg_dir / "gallery"
    state_path = cfg_dir / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "completed":
        raise ValueError("CFG evaluation must be completed before pixel comparison")
    protocol = state.get("config", {}).get("protocol", "")
    if protocol != "ccdd11-endpoint-correct-state-cfg-validation-v2":
        raise ValueError(f"Unsupported CFG protocol: {protocol!r}")

    betas = normalize_betas(
        list(args.betas) if args.betas is not None else state["config"]["betas"]
    )
    output_dir = (args.output_dir or cfg_dir / "pixel_beta_comparison").resolve()
    report_path = output_dir / "comparison.json"
    if report_path.exists():
        raise RuntimeError(f"Completed pixel comparison already exists: {report_path}")
    samples = _discover_samples(gallery, args.sample_ids)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("No CFG gallery samples were found")

    rows: list[dict] = []
    for sample_dir in samples:
        negative = _load_rgb(sample_dir / "beta_0.png")
        positive = _load_rgb(sample_dir / "beta_1.png")
        if positive.shape != negative.shape:
            raise ValueError(f"Endpoint image shape mismatch in {sample_dir}")
        sheet_rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        sample_output = output_dir / sample_dir.name
        for beta in betas:
            label = beta_label(beta)
            cfg = _load_rgb(sample_dir / f"beta_{label}.png")
            if cfg.shape != positive.shape:
                cfg_path = sample_dir / f"beta_{label}.png"
                raise ValueError(f"CFG image shape mismatch: {cfg_path}")
            pixel = blend_cfg_pixels(positive, negative, beta)
            difference = np.abs(pixel - cfg)
            pixel_path = sample_output / f"pixel_beta_{label}.png"
            difference_path = sample_output / f"abs_diff_beta_{label}.png"
            _save_rgb(pixel, pixel_path)
            _save_rgb(np.clip(difference * 4.0, 0, 1), difference_path)
            rows.append(
                {
                    "sample_id": sample_dir.name,
                    "beta": beta,
                    **_comparison_metrics(pixel, cfg),
                    "cfg_path": str((sample_dir / f"beta_{label}.png").resolve()),
                    "pixel_path": str(pixel_path),
                    "abs_diff_x4_path": str(difference_path),
                }
            )
            sheet_rows.append((label, cfg, pixel, np.clip(difference * 4, 0, 1)))
        _save_sheet(sheet_rows, sample_output / "comparison_sheet.png")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "comparison.csv", rows)
    finite_psnr = [
        row["pixel_vs_cfg_psnr"]
        for row in rows
        if row["pixel_vs_cfg_psnr"] is not None
    ]
    report = {
        "protocol": "ccdd11-cfg-vs-pixel-beta-comparison-v1",
        "source_cfg_dir": str(cfg_dir),
        "source_cfg_protocol": protocol,
        "formula": (
            "I_pixel = clip(I_negative + beta * "
            "(I_positive - I_negative), 0, 1)"
        ),
        "endpoint_images": {"negative": "beta_0.png", "positive": "beta_1.png"},
        "betas": betas,
        "sample_count": len(samples),
        "row_count": len(rows),
        "aggregate": {
            "mean_mae": float(np.mean([row["pixel_vs_cfg_mae"] for row in rows])),
            "mean_mse": float(np.mean([row["pixel_vs_cfg_mse"] for row in rows])),
            "mean_finite_psnr": (
                float(np.mean(finite_psnr)) if finite_psnr else None
            ),
            "max_abs": max(row["pixel_vs_cfg_max_abs"] for row in rows),
        },
        "rows": rows,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Compared {len(samples)} samples x {len(betas)} betas: {output_dir}",
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
        "--betas", nargs="+", type=float, help="Defaults to the CFG run's beta grid"
    )
    value.add_argument(
        "--sample-id",
        dest="sample_ids",
        action="append",
        help="Gallery directory name; repeat to select exact examples",
    )
    value.add_argument(
        "--max-samples",
        type=int,
        help="Compare only the first N selected gallery examples",
    )
    return value


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
