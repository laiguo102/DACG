"""Precompute DACG outputs for all CDD11 double-degradation training images."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from difix3d_selective.prepare import generate_cdd11_coarse


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--dacg-checkpoint", type=Path, required=True)
    value.add_argument("--coarse-root", type=Path, required=True)
    value.add_argument(
        "--degradation-pairs",
        nargs="+",
        type=int,
        choices=range(1, 6),
        default=[1, 2, 3, 4, 5],
    )
    value.add_argument("--device", default="cuda")
    value.add_argument("--tile-size", type=int, default=0)
    value.add_argument("--tile-overlap", type=int, default=64)
    return value


def main() -> None:
    args = parser().parse_args()
    result = generate_cdd11_coarse(
        data_root=args.data_root,
        dacg_checkpoint=args.dacg_checkpoint,
        coarse_root=args.coarse_root,
        pair_ids=args.degradation_pairs,
        device=torch.device(args.device),
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
    )
    print(f"Prepared {result['coarse_images']} coarse images in {result['coarse_root']}")


if __name__ == "__main__":
    main()
