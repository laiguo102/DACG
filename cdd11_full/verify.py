"""Verify reuse of the exact CDD-11-v1 manifest bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_manifest
from .protocol import DEGRADATIONS, OBJECTIVE_VARIANT, PROTOCOL_NAME
from .train import _verify_manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    args = parser.parse_args()
    directory, hashes = _verify_manifests(args.manifest_dir)
    counts = {}
    expected = {"train": 11913, "val": 1100, "test": 2200}
    for split, total in expected.items():
        records = load_manifest(directory / f"{split}.jsonl", split)
        by_degradation = {value: sum(record.degradation == value for record in records) for value in DEGRADATIONS}
        if len(records) != total or len(set(by_degradation.values())) != 1:
            raise RuntimeError(f"Invalid frozen {split} manifest: rows={len(records)}, counts={by_degradation}")
        counts[split] = {"rows": len(records), "rows_by_degradation": by_degradation}
    print(json.dumps({"status": "pass", "protocol": PROTOCOL_NAME,
                      "objective_variant": OBJECTIVE_VARIANT, "directory": str(directory),
                      "hashes": hashes, "splits": counts}, indent=2), flush=True)


if __name__ == "__main__":
    main()
