"""Audit the standard CDD-11 layout for full training and official testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import dataset_fingerprint, validate_partition
from .protocol import DEGRADATIONS, NUM_TEST_SCENES, NUM_TRAIN_SCENES, protocol_metadata
from .runtime import atomic_json


def audit(data_root: Path) -> dict:
    train_ids = validate_partition(data_root, "train", NUM_TRAIN_SCENES)
    test_ids = validate_partition(data_root, "test", NUM_TEST_SCENES)
    overlap = sorted(set(train_ids) & set(test_ids))
    # Scene filenames need not be globally unique across official partitions;
    # report overlap rather than treating it as image-content leakage.
    return {
        "status": "pass", "protocol": protocol_metadata(),
        "train": {"scenes": len(train_ids), "pairs": len(train_ids) * len(DEGRADATIONS),
                  "fingerprint": dataset_fingerprint(data_root, "train", train_ids)},
        "test": {"scenes": len(test_ids), "pairs": len(test_ids) * len(DEGRADATIONS),
                 "fingerprint": dataset_fingerprint(data_root, "test", test_ids)},
        "filename_overlap_count": len(overlap), "filename_overlap_examples": overlap[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit full CDD-11 layout without generating any split")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.data_root)
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
