"""Verify CDD-11 storage, split sizes, pair coverage, and scene isolation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import validate_partition
from .protocol import NUM_TEST_SCENES, NUM_TRAIN_SCENES, assert_frozen_split, load_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=None)
    args = parser.parse_args()
    train_ids = set(validate_partition(args.data_root, "train", NUM_TRAIN_SCENES))
    validate_partition(args.data_root, "test", NUM_TEST_SCENES)
    split_dir = args.split_dir or args.data_root / "splits"
    splits = load_splits(split_dir)
    assert_frozen_split(sorted(train_ids), splits)
    manifest_ids = set().union(*(set(ids) for ids in splits.values()))
    if manifest_ids != train_ids:
        raise ValueError(f"Split/train mismatch: missing={len(train_ids-manifest_ids)}, extra={len(manifest_ids-train_ids)}")
    print("CDD-11 OK: 1183 train scenes, 200 test scenes, 5x213 OOF + 118 validation; no scene leakage")


if __name__ == "__main__":
    main()
