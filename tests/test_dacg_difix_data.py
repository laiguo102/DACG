from __future__ import annotations

import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image
from torch import nn

import cdd11_full.model as cdd11_model
from dacg_difix.data import PreparedCDD11Dataset, build_difix_loader
from dacg_difix.prepare import (
    make_folder_scene_split,
    prepare_cdd11_folder_manifests,
    prepare_cdd11_manifests,
    prepare_existing_cdd11_manifests,
)
from src.net.model import DACG_IR


SMALL_CONFIG = {
    "dim": 8,
    "num_blocks": [1, 1, 1, 1],
    "heads": [1, 1, 1, 1],
    "num_refinement_blocks": 0,
    "num_scales": 1,
}


def _save_image(path: Path, value: int, size: tuple[int, int] = (7, 5)) -> None:
    array = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    Image.fromarray(array, "RGB").save(path)


@contextmanager
def _temporary_workspace():
    path = Path.cwd() / f".dacg-difix-test-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


class HalfModel(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * 0.5


class FakeTokenizer:
    def __call__(self, _prompt, *, max_length, padding, truncation, return_tensors):
        assert padding == "max_length" and truncation and return_tensors == "pt"
        return {"input_ids": torch.arange(max_length).unsqueeze(0)}


class DACGDifixDataTests(unittest.TestCase):
    def test_folder_split_matches_selective_branch(self):
        scene_ids = [f"{index:06d}" for index in range(1183)]
        first = make_folder_scene_split(scene_ids)
        second = make_folder_scene_split(reversed(scene_ids))
        self.assertEqual(first, second)
        self.assertEqual(len(first["train"]), 1065)
        self.assertEqual(len(first["validation"]), 118)
        self.assertFalse(set(first["train"]) & set(first["validation"]))

    def test_folder_preparation_writes_train_validation_and_untouched_test(self):
        with _temporary_workspace() as root:
            data_root = root / "CDD11"
            coarse_root = root / "background"
            output_dir = root / "run"

            def fake_index(path):
                path = Path(path)
                count = 200 if "test" in path.parts else 1183
                return {
                    f"{index:06d}": path / f"{index:06d}.png"
                    for index in range(count)
                }

            def fake_coarse(root_path, split, degradation):
                return Path(root_path) / split / degradation

            with (
                mock.patch("dacg_difix.prepare.index_images", side_effect=fake_index),
                mock.patch("dacg_difix.prepare._coarse_directory", side_effect=fake_coarse),
            ):
                metadata = prepare_cdd11_folder_manifests(
                    data_root, coarse_root, output_dir
                )

            self.assertEqual(
                metadata["counts"],
                {"train": 1065 * 11, "validation": 118 * 11, "test": 200 * 11},
            )
            test_rows = (output_dir / "prepared" / "manifests" / "test.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(test_rows), 2200)

    def test_prepare_reuses_existing_background_without_dacg(self):
        with _temporary_workspace() as root:
            manifests = root / "manifests"
            background = root / "background"
            output = root / "prepared"
            manifests.mkdir()
            degraded = root / "degraded.png"
            target = root / "target.png"
            _save_image(degraded, 32, size=(12, 12))
            _save_image(target, 224, size=(12, 12))
            record = {
                "id": "scene/train",
                "degradation": "low_haze",
                "input": str(degraded.resolve()),
                "target": str(target.resolve()),
                "scene_id": "scene",
                "metadata": {"arity": 2},
            }
            for split in ("train", "val", "test"):
                (manifests / f"{split}.jsonl").write_text(
                    json.dumps(record) + "\n", encoding="utf-8"
                )
                coarse_dir = background / split / "low_haze"
                coarse_dir.mkdir(parents=True)
                _save_image(coarse_dir / degraded.name, 128, size=(12, 12))

            metadata = prepare_existing_cdd11_manifests(
                manifests, background, output
            )
            self.assertEqual(metadata["coarse_source"], "existing-background")
            self.assertEqual(metadata["counts"], {"train": 1, "val": 1, "test": 1})
            row = json.loads((output / "train.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(Path(row["coarse"]), (background / "train" / "low_haze" / degraded.name).resolve())
            self.assertEqual(Path(row["degraded"]), degraded.resolve())
            self.assertEqual(Path(row["target"]), target.resolve())

    def test_forward_with_degradation_preserves_forward_and_state_dict(self):
        model = DACG_IR(**SMALL_CONFIG).eval()
        keys = tuple(model.state_dict())
        value = torch.rand(2, 3, 17, 19)
        with torch.inference_mode():
            restored = model(value)
            restored_with_context, p_global = model.forward_with_degradation(value)
        self.assertTrue(torch.equal(restored, restored_with_context))
        self.assertEqual(restored.shape, value.shape)
        self.assertEqual(p_global.shape, (2, 16))
        self.assertEqual(tuple(model.state_dict()), keys)

    def test_checkpoint_loaders_accept_string_and_mapping_model_config(self):
        original_config = cdd11_model.MODEL_CONFIGS["DACG_IR"]
        cdd11_model.MODEL_CONFIGS["DACG_IR"] = dict(SMALL_CONFIG)
        try:
            with _temporary_workspace() as root:
                source = DACG_IR(**SMALL_CONFIG)
                for model_field in ("string", "mapping"):
                    with self.subTest(model_field=model_field):
                        config_value = "DACG_IR" if model_field == "string" else {
                            "model_name": "DACG_IR", **SMALL_CONFIG
                        }
                        checkpoint_path = root / f"{model_field}.pth"
                        torch.save(
                            {"model": source.state_dict(), "config": {"model": config_value}},
                            checkpoint_path,
                        )
                        loaded, _ = cdd11_model.load_network(
                            str(checkpoint_path), torch.device("cpu")
                        )
                        self.assertEqual(tuple(loaded.state_dict()), tuple(source.state_dict()))
                        dam, _ = cdd11_model.load_dam_encoder(
                            str(checkpoint_path), torch.device("cpu")
                        )
                        with torch.inference_mode():
                            p_global = dam(torch.rand(3, 3, 17, 19))
                        self.assertEqual(p_global.shape, (3, 16))
                        self.assertFalse(any(parameter.requires_grad for parameter in dam.parameters()))
        finally:
            cdd11_model.MODEL_CONFIGS["DACG_IR"] = original_config

    def test_prepare_and_load_aligned_difix_data(self):
        with _temporary_workspace() as root:
            source_dir = root / "source"
            source_dir.mkdir()
            degraded = source_dir / "degraded.png"
            target = source_dir / "target.png"
            _save_image(degraded, 128)
            _save_image(target, 224)
            for split in ("train", "val", "test"):
                record = {
                    "id": f"scene/{split}",
                    "degradation": "low_haze",
                    "split": split,
                    "input": str(degraded.resolve()),
                    "target": str(target.resolve()),
                    "scene_id": "scene",
                    "metadata": {"arity": 2},
                }
                (source_dir / f"{split}.jsonl").write_text(
                    json.dumps(record) + "\n", encoding="utf-8"
                )

            checkpoint = root / "dacg.pth"
            checkpoint.write_bytes(b"checkpoint identity")
            output_dir = root / "prepared"
            metadata = prepare_cdd11_manifests(
                source_manifest_dir=source_dir,
                output_dir=output_dir,
                model=HalfModel(),
                device=torch.device("cpu"),
                checkpoint_path=checkpoint,
                tile_size=4,
                tile_overlap=2,
            )
            self.assertEqual(metadata["counts"], {"train": 1, "val": 1, "test": 1})

            row = json.loads((output_dir / "train.jsonl").read_text(encoding="utf-8"))
            expected_fields = {
                "coarse", "degraded", "target", "degradation", "split", "id", "scene_id", "arity"
            }
            self.assertTrue(expected_fields <= row.keys())
            self.assertEqual(row["dacg_checkpoint_sha256"], metadata["dacg_checkpoint_sha256"])
            with Image.open(row["coarse"]) as coarse:
                self.assertEqual(coarse.size, (7, 5))

            sample = PreparedCDD11Dataset(
                output_dir / "train.jsonl",
                resolution=8,
                tokenizer=FakeTokenizer(),
                prompt="remove degradation",
            )[0]
            self.assertEqual(sample["main"].shape, (3, 8, 8))
            self.assertEqual(sample["ref"].shape, (3, 8, 8))
            self.assertEqual(sample["target"].shape, (3, 8, 8))
            self.assertEqual(sample["ref_01"].shape, (3, 8, 8))
            self.assertEqual(sample["prompt_tokens"].shape, (77,))
            self.assertTrue(torch.equal(sample["prompt_tokens"], torch.arange(77)))
            self.assertTrue(torch.allclose(sample["ref"], sample["ref_01"] * 2.0 - 1.0))
            self.assertEqual(sample["arity"], 2)

            native = PreparedCDD11Dataset(
                output_dir / "val.jsonl",
                resolution=None,
                prompt_tokens=torch.arange(77),
            )[0]
            self.assertEqual(native["main"].shape, (3, 5, 7))
            self.assertEqual(native["ref"].shape, (3, 5, 7))
            self.assertEqual(native["target"].shape, (3, 5, 7))

            batch = next(iter(build_difix_loader(
                output_dir / "train.jsonl",
                resolution=8,
                batch_size=1,
                shuffle=False,
                prompt_tokens=torch.arange(77),
            )))
            self.assertEqual(batch["main"].shape, (1, 3, 8, 8))
            self.assertEqual(batch["prompt_tokens"].shape, (1, 77))


if __name__ == "__main__":
    unittest.main()
