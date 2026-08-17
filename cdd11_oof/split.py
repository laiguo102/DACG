"""Generate the one-time frozen scene split from standard CDD-11 storage."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import validate_partition
from .protocol import NUM_TEST_SCENES, NUM_TRAIN_SCENES, SEED, make_split, write_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True, help="CDD11 root containing train/ and test/")
    parser.add_argument("--split-dir", type=Path, default=None, help="Default: <data-root>/splits")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    split_dir = args.split_dir or args.data_root / "splits"
    if split_dir.exists() and any(split_dir.iterdir()):
        raise FileExistsError(f"Refusing to replace non-empty split directory: {split_dir}")
    train_ids = validate_partition(args.data_root, "train", NUM_TRAIN_SCENES)
    validate_partition(args.data_root, "test", NUM_TEST_SCENES)
    write_splits(split_dir, make_split(train_ids, args.seed))
    print(f"Wrote frozen seed-{SEED} split to {split_dir}")


if __name__ == "__main__":
    main()
