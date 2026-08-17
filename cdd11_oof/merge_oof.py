"""Verify and combine five fold manifests into the Difix training manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import DEGRADATIONS, NUM_OOF_SCENES, load_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-root", type=Path, required=True, help="Contains fold1/... through fold5/...")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    splits = load_splits(args.split_dir)
    records = []
    for fold in range(1, 6):
        path = args.oof_root / f"fold{fold}" / "manifest.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record["scene_id"] not in splits[f"fold{fold}"]:
                    raise ValueError(f"Wrong-fold scene {record['scene_id']} in {path}")
                if not Path(record["coarse"]).is_file():
                    raise FileNotFoundError(record["coarse"])
                records.append(record)
    keys = {(r["scene_id"], r["degradation"]) for r in records}
    expected = NUM_OOF_SCENES * len(DEGRADATIONS)
    if len(records) != expected or len(keys) != expected:
        raise ValueError(f"Expected {expected} unique OOF triples, got {len(records)} records / {len(keys)} unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    print(f"Wrote {expected} verified OOF triples for Difix to {args.output}")


if __name__ == "__main__":
    main()
