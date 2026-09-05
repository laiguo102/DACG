import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from difix3d_selective.pixel_cfg_compare import (
    DEFAULT_BETAS,
    run,
    select_two_complete_scenes,
)
from difix3d_selective.protocol import PAIR_FOLDERS, directed_tasks


def scene_records(scene_id):
    records = []
    for pair_id, pair in PAIR_FOLDERS.items():
        for task in directed_tasks(pair_id):
            records.append(
                {
                    "sample_id": (
                        f"validation/{pair}/{scene_id}/remove-{task.remove}-"
                        f"preserve-{task.preserve}"
                    ),
                    "scene_id": scene_id,
                    "pair_id": pair_id,
                    "pair": pair,
                    "remove": task.remove,
                    "preserve": task.preserve,
                }
            )
    return records


class FakeTable:
    def __init__(self, *, columns, data):
        self.columns = columns
        self.data = data


class FakeRun:
    def __init__(self):
        self.summary = {}
        self.logged = []
        self.finished = False

    def log(self, payload):
        self.logged.append(payload)

    def finish(self):
        self.finished = True


class TestSceneSelection(unittest.TestCase):
    def test_auto_selects_two_complete_scenes_in_task_order(self):
        records = list(reversed(scene_records("00002") + scene_records("00001")))
        scene_ids, selected = select_two_complete_scenes(records, None)
        self.assertEqual(scene_ids, ["00001", "00002"])
        self.assertEqual(len(selected), 20)
        self.assertEqual([row["scene_id"] for row in selected[:10]], ["00001"] * 10)
        self.assertEqual(
            [row["pair_id"] for row in selected[:10]],
            [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        )

    def test_rejects_incomplete_or_ambiguous_scene_sets(self):
        incomplete = scene_records("00001")[:-1] + scene_records("00002")
        with self.assertRaisesRegex(ValueError, "exactly two complete"):
            select_two_complete_scenes(incomplete, None)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            select_two_complete_scenes(incomplete, ["00001", "00002"])

        three_scenes = (
            scene_records("00001")
            + scene_records("00002")
            + scene_records("00003")
        )
        with self.assertRaisesRegex(ValueError, "found 3"):
            select_two_complete_scenes(three_scenes, None)
        scene_ids, selected = select_two_complete_scenes(
            three_scenes, ["00003", "00001"]
        )
        self.assertEqual(scene_ids, ["00003", "00001"])
        self.assertEqual(len(selected), 20)


class TestPixelCfgComparison(unittest.TestCase):
    def _build_cfg_output(self, root):
        cfg_dir = root / "cfg"
        records_dir = cfg_dir / "records"
        gallery_dir = cfg_dir / "gallery"
        records_dir.mkdir(parents=True)
        gallery_dir.mkdir(parents=True)
        (cfg_dir / "state.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "config": {
                        "protocol": (
                            "ccdd11-endpoint-correct-state-cfg-validation-v2"
                        ),
                        "resolution": 512,
                        "betas": list(DEFAULT_BETAS),
                    },
                }
            ),
            encoding="utf-8",
        )
        references = {}
        for name, value in (
            ("degraded", 32),
            ("coarse", 64),
            ("target", 96),
            ("clean", 128),
        ):
            path = root / f"{name}.png"
            Image.new("RGB", (640, 480), (value, value, value)).save(path)
            references[name] = str(path)

        records = scene_records("00001") + scene_records("00002")
        for index, record in enumerate(records):
            record.update(
                {
                    "coarse_path": references["coarse"],
                    "degraded_path": references["degraded"],
                    "positive_target_path": references["target"],
                    "clean_gt_path": references["clean"],
                }
            )
            (records_dir / f"{index:02d}.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
            sample_dir = gallery_dir / record["sample_id"].replace("/", "__")
            sample_dir.mkdir()
            Image.new("RGB", (2048, 1608), "white").save(
                sample_dir / "contact_sheet.png"
            )
            for beta in DEFAULT_BETAS:
                label = f"{beta:g}"
                value = int(round(51 + beta * (204 - 51)))
                value = min(255, max(0, value))
                Image.new("RGB", (512, 512), (value, value, value)).save(
                    sample_dir / f"beta_{label}.png"
                )
        return cfg_dir

    def test_writes_twenty_matching_galleries_and_logs_dual_wandb_table(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory).resolve()
            cfg_dir = self._build_cfg_output(root)
            output = root / "comparison"
            fake_run = FakeRun()
            fake_wandb = ModuleType("wandb")
            fake_wandb.Image = lambda path: ("image", path)
            fake_wandb.Table = FakeTable
            fake_wandb.init = Mock(return_value=fake_run)
            args = argparse.Namespace(
                cfg_dir=cfg_dir,
                output_dir=output,
                scene_ids=None,
                betas=None,
                report_to="wandb",
                wandb_entity="entity",
                wandb_project="project",
                wandb_run_name="pixel-run",
            )
            with patch.dict(sys.modules, {"wandb": fake_wandb}):
                result = run(args)

            self.assertEqual(result, output)
            report = json.loads(
                (output / "comparison.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["config"]["scene_ids"], ["00001", "00002"])
            self.assertEqual(len(report["records"]), 20)
            self.assertNotIn("metrics", report)
            for record in report["records"]:
                sheet_path = Path(record["pixel_contact_sheet_path"])
                with Image.open(sheet_path) as sheet:
                    self.assertEqual(sheet.size, (2048, 1608))
                self.assertEqual(len(record["pixel_prediction_paths"]), 8)

            first = Path(report["records"][0]["pixel_contact_sheet_path"])
            with Image.open(first) as sheet:
                image = np.asarray(sheet)
            self.assertTrue(np.all(image[24, 0] == 32))
            self.assertTrue(np.all(image[24, 512] == 64))
            self.assertTrue(np.all(image[24, 1024] == 96))
            self.assertTrue(np.all(image[24, 1536] == 128))
            midpoint_path = Path(
                report["records"][0]["pixel_prediction_paths"]["0.5"]
            )
            self.assertTrue(np.all(np.asarray(Image.open(midpoint_path)) == 128))

            fake_wandb.init.assert_called_once()
            init_kwargs = fake_wandb.init.call_args.kwargs
            self.assertEqual(init_kwargs["job_type"], "pixel-cfg-comparison")
            table = fake_run.logged[0]["pixel_cfg/gallery"]
            self.assertEqual(len(table.data), 20)
            self.assertEqual(table.columns[-2:], ["cfg_gallery", "pixel_gallery"])
            self.assertTrue(all(row[-2][0] == "image" for row in table.data))
            self.assertTrue(all(row[-1][0] == "image" for row in table.data))
            self.assertTrue(fake_run.finished)

    def test_missing_endpoint_or_beta_image_fails(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory).resolve()
            cfg_dir = self._build_cfg_output(root)
            first_gallery = next((cfg_dir / "gallery").iterdir())
            args = argparse.Namespace(
                cfg_dir=cfg_dir,
                output_dir=root / "comparison",
                scene_ids=None,
                betas=None,
                report_to="none",
                wandb_entity="unused",
                wandb_project="unused",
                wandb_run_name="unused",
            )
            endpoint = first_gallery / "beta_1.png"
            endpoint.unlink()
            with self.assertRaises(FileNotFoundError):
                run(args)
            Image.new("RGB", (512, 512), (204, 204, 204)).save(endpoint)

            (first_gallery / "beta_1.2.png").unlink()
            with self.assertRaises(FileNotFoundError):
                run(args)


if __name__ == "__main__":
    unittest.main()
