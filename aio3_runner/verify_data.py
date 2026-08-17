"""Validate frozen manifest identity and record-level schema."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .data import read_manifest, verify_frozen_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify AIO3-v1 frozen manifests")
    parser.add_argument("--manifest-dir", required=True)
    args = parser.parse_args()
    root = Path(args.manifest_dir)
    hashes = verify_frozen_manifests(root)
    counts = {}
    for split in ("train", "val", "test"):
        records = read_manifest(root / f"{split}.jsonl", split)
        counts[split] = dict(Counter(record["task"] for record in records))
    print(json.dumps({"sha256": hashes, "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
