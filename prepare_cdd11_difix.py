"""Prepare native-resolution DACG coarse images for Difix3D training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cdd11_full.model import load_network
from dacg_difix.prepare import (
    SPLITS,
    prepare_cdd11_manifests,
    prepare_existing_cdd11_manifests,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest-dir", type=Path, required=True)
    source = value.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument(
        "--coarse-root",
        type=Path,
        help="reuse existing DACG background images instead of running DACG",
    )
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--device", default="cuda")
    value.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    value.add_argument(
        "--tile-size", type=int, default=512,
        help="overlapping inference tile size; use 0 for a native full-image forward",
    )
    value.add_argument("--tile-overlap", type=int, default=128)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.coarse_root is not None:
        metadata = prepare_existing_cdd11_manifests(
            source_manifest_dir=args.manifest_dir,
            coarse_root=args.coarse_root,
            output_dir=args.output_dir,
            splits=args.splits,
        )
    else:
        device = torch.device(args.device)
        model, _ = load_network(str(args.checkpoint), device)
        model.requires_grad_(False)
        metadata = prepare_cdd11_manifests(
            source_manifest_dir=args.manifest_dir,
            output_dir=args.output_dir,
            model=model,
            device=device,
            checkpoint_path=args.checkpoint,
            splits=args.splits,
            tile_size=args.tile_size or None,
            tile_overlap=args.tile_overlap,
        )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
