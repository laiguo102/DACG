import json
from pathlib import Path
import tempfile
import unittest

from cdd11_oof.data import validate_partition
from cdd11_oof.protocol import (
    DEGRADATIONS,
    NUM_OOF_SCENES,
    NUM_VAL_SCENES,
    SCENES_PER_FOLD,
    make_split,
    write_splits,
    load_splits,
)


class TestCDD11OOF(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_complete(self):
        scene_ids = [f"{i:06d}" for i in range(1183)]
        first = make_split(scene_ids)
        second = make_split(list(reversed(scene_ids)))
        self.assertEqual(first, second)
        self.assertTrue(all(len(first[f"fold{i}"]) == SCENES_PER_FOLD for i in range(1, 6)))
        self.assertEqual(len(first["val"]), NUM_VAL_SCENES)
        flattened = [x for values in first.values() for x in values]
        self.assertEqual(len(flattened), 1183)
        self.assertEqual(len(set(flattened)), 1183)
        self.assertEqual(sum(len(first[f"fold{i}"]) for i in range(1, 6)), NUM_OOF_SCENES)
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            write_splits(root, first)
            self.assertEqual(load_splits(root), first)
            self.assertEqual(json.loads((root / "split_info.json").read_text())["seed"], 42)

    def test_split_rejects_nonstandard_count_and_seed(self):
        with self.assertRaisesRegex(ValueError, "1183"):
            make_split(["only-one"])
        with self.assertRaisesRegex(ValueError, "seed=42"):
            make_split([f"{i:06d}" for i in range(1183)], seed=7)

    def test_partition_validation_checks_all_eleven_pairings(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            base = root / "train"
            for folder in ("clear", *DEGRADATIONS):
                (base / folder).mkdir(parents=True)
                (base / folder / "000001.png").touch()
            self.assertEqual(validate_partition(root, "train"), ["000001"])
            (base / "rain" / "000001.png").unlink()
            with self.assertRaisesRegex(ValueError, "rain: missing"):
                validate_partition(root, "train")

    def test_load_splits_detects_scene_leakage(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            split = make_split([f"{i:06d}" for i in range(1183)])
            split["fold2"][0] = split["fold1"][0]
            write_splits(root, split)
            with self.assertRaisesRegex(ValueError, "leakage"):
                load_splits(root)


if __name__ == "__main__":
    unittest.main()
