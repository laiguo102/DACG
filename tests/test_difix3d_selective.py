import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from difix3d_selective.main_view import select_main_view, select_main_view_skips
from difix3d_selective.prepare import prepare_selective_manifests, tiled_forward
from difix3d_selective.protocol import (
    NUM_TRAIN_SCENES,
    NUM_VALIDATION_SCENES,
    directed_tasks,
    expected_counts,
    make_scene_split,
    selected_pairs,
)


class FakeTokenizer:
    model_max_length = 4

    def __call__(self, *args, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3, 4]]))


class TestSelectiveProtocol(unittest.TestCase):
    def test_scene_split_is_fixed_disjoint_and_complete(self):
        scene_ids = [f"{index:06d}" for index in range(1183)]
        first = make_scene_split(scene_ids)
        second = make_scene_split(list(reversed(scene_ids)))
        self.assertEqual(first, second)
        self.assertEqual(len(first["train"]), NUM_TRAIN_SCENES)
        self.assertEqual(len(first["validation"]), NUM_VALIDATION_SCENES)
        self.assertFalse(set(first["train"]) & set(first["validation"]))
        self.assertEqual(set(first["train"]) | set(first["validation"]), set(scene_ids))

    def test_pair_selection_and_bidirectional_prompts(self):
        self.assertEqual(
            selected_pairs([1, 3, 5]),
            [(1, "low_haze"), (3, "low_snow"), (5, "haze_snow")],
        )
        forward, reverse = directed_tasks(1)
        self.assertEqual(forward.prompt, "remove low light, preserve haze")
        self.assertEqual(forward.preserve, "haze")
        self.assertEqual(reverse.prompt, "remove haze, preserve low light")
        self.assertEqual(reverse.preserve, "low")
        self.assertEqual(
            expected_counts([1, 3, 5]),
            {"coarse": 3549, "train": 6390, "validation": 708},
        )


class TestMainViewSelection(unittest.TestCase):
    def test_latent_and_all_skips_keep_view_zero(self):
        value = torch.arange(4 * 3).reshape(4, 3)
        selected = select_main_view(value, batch_size=2)
        self.assertTrue(torch.equal(selected, value[[0, 2]]))
        skips = [value, value + 100, value + 200, value + 300]
        selected_skips = select_main_view_skips(skips, batch_size=2)
        for source, result in zip(skips, selected_skips):
            self.assertTrue(torch.equal(result, source[[0, 2]]))


class TestDatasetAndPreparation(unittest.TestCase):
    @staticmethod
    def _save(path: Path, rgb):
        array = np.zeros((5, 7, 3), dtype=np.uint8)
        array[:] = rgb
        Image.fromarray(array, "RGB").save(path)

    @unittest.skipIf(importlib.util.find_spec("torchvision") is None, "torchvision is not installed")
    def test_dataset_loads_two_conditioning_views_and_one_target(self):
        from difix3d_selective.data import SelectiveDifixDataset

        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            main, reference, target = root / "main.png", root / "ref.png", root / "target.png"
            self._save(main, (0, 0, 0))
            self._save(reference, (255, 0, 0))
            self._save(target, (0, 255, 0))
            manifest = root / "train.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "train/low_haze/000001/remove-low-preserve-haze",
                        "prompt": "remove low light, preserve haze",
                        "image": str(main),
                        "ref_image": str(reference),
                        "target_image": str(target),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            sample = SelectiveDifixDataset(manifest, FakeTokenizer(), resolution=8)[0]
            self.assertEqual(tuple(sample["conditioning_pixel_values"].shape), (2, 3, 8, 8))
            self.assertEqual(tuple(sample["output_pixel_values"].shape), (3, 8, 8))
            self.assertTrue(torch.equal(sample["input_ids"], torch.tensor([1, 2, 3, 4])))
            self.assertLess(float(sample["conditioning_pixel_values"][0].mean()), -0.99)
            self.assertGreater(float(sample["output_pixel_values"][1].mean()), 0.99)

    def test_tiled_dacg_forward_reconstructs_full_shape(self):
        class Double(torch.nn.Module):
            def forward(self, value):
                return value * 2

        image = torch.rand(1, 3, 9, 11)
        result = tiled_forward(Double(), image, tile_size=6, overlap=2)
        self.assertEqual(result.shape, image.shape)
        self.assertTrue(torch.allclose(result, image * 2))

    def test_manifests_read_main_images_from_independent_coarse_root(self):
        root = Path.cwd().resolve()
        data_root = root / "fake-cdd11"
        coarse_root = root / "fake-coarse"
        output_dir = root / "fake-run"
        scene_ids = ("000001", "000002")
        written = {}

        def fake_index_images(path):
            path = Path(path)
            if path == coarse_root / "low_haze":
                return {scene: path / f"{scene}.png" for scene in scene_ids}
            return {scene: path / f"{scene}.jpg" for scene in scene_ids}

        def capture_jsonl(path, records):
            written[Path(path).name] = records

        with (
            patch("difix3d_selective.prepare.index_images", side_effect=fake_index_images),
            patch("difix3d_selective.prepare._write_jsonl", side_effect=capture_jsonl),
            patch(
                "difix3d_selective.prepare.make_scene_split",
                return_value={"train": [scene_ids[0]], "validation": [scene_ids[1]]},
            ),
            patch("pathlib.Path.mkdir"),
            patch("pathlib.Path.write_text"),
        ):
            prepare_selective_manifests(
                data_root=data_root,
                coarse_root=coarse_root,
                output_dir=output_dir,
                pair_ids=[1],
            )

        train_records = written["train.jsonl"]
        self.assertEqual(len(train_records), 2)
        self.assertTrue(
            all(
                Path(record["image"]).parent == coarse_root / "low_haze"
                for record in train_records
            )
        )
        by_prompt = {
            record["prompt"]: Path(record["target_image"]).parent.name for record in train_records
        }
        self.assertEqual(by_prompt["remove low light, preserve haze"], "haze")
        self.assertEqual(by_prompt["remove haze, preserve low light"], "low")

    def test_cdd11_full_checkpoint_loader_uses_model_and_config_keys(self):
        from cdd11_full import model as dacg_model

        tiny = torch.nn.Conv2d(3, 3, 1)
        checkpoint = {"config": {"model": "DACG_IR"}, "model": tiny.state_dict()}
        with (
            patch.object(dacg_model.torch, "load", return_value=checkpoint),
            patch.object(dacg_model, "build_model", return_value=torch.nn.Conv2d(3, 3, 1)),
        ):
            loaded, loaded_checkpoint = dacg_model.load_network(
                "final.pth", torch.device("cpu")
            )
        self.assertEqual(loaded_checkpoint["config"]["model"], "DACG_IR")
        for key, value in tiny.state_dict().items():
            self.assertTrue(torch.equal(loaded.state_dict()[key], value))


if __name__ == "__main__":
    unittest.main()
