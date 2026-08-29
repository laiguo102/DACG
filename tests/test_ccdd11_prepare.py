import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from difix3d_selective.ccdd_prepare import (
    DATASET_NAME,
    TARGET_KIND,
    prepare_ccdd11_selective_manifests,
    strict_index_images,
    validate_manifest_records,
)


def _image(path: Path, value: int = 64):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 10, 3), value, dtype=np.uint8), "RGB").save(path)


def _fixture(tmp_path: Path, *, omit_target: tuple[str, str, str] | None = None):
    data_root = tmp_path / "CCDD-11"
    coarse_root = tmp_path / "coarse" / "half_train"
    output_dir = tmp_path / "run"
    main_root = data_root / "half_train" / "main_data"
    sub_root = data_root / "half_train" / "sub_data"
    scenes = ("00001", "00002")
    for scene in scenes:
        _image(main_root / "clear" / f"{scene}.png", 128)
        _image(main_root / "low_haze" / f"{scene}.png", 32)
        _image(main_root / "low" / f"{scene}.png", 70)
        _image(main_root / "haze" / f"{scene}.png", 90)
        _image(coarse_root / "low_haze" / f"{scene}.png", 48)
        for component, value in (("low", 70), ("haze", 90)):
            target = sub_root / "low_haze" / scene / f"{scene}_{component}_.png"
            if omit_target != (scene, "low_haze", component):
                _image(target, value)
        _image(sub_root / "low_haze" / scene / f"{scene}_half_.png", 80)
    metadata = {
        "dataset": DATASET_NAME,
        "status": "completed",
        "split": "half_train",
        "data_root": str(data_root.resolve()),
        "source_root": str(main_root.resolve()),
        "pair_ids": [1],
        "expected_scenes": 2,
    }
    coarse_root.mkdir(parents=True, exist_ok=True)
    (coarse_root / "coarse_preparation.json").write_text(json.dumps(metadata), encoding="utf-8")
    return data_root, coarse_root, output_dir, scenes


def _prepare_miniature(tmp_path: Path, **fixture_kwargs):
    data_root, coarse_root, output_dir, scenes = _fixture(tmp_path, **fixture_kwargs)
    with (
        patch("difix3d_selective.ccdd_prepare.NUM_SCENES", 2),
        patch(
            "difix3d_selective.ccdd_prepare.make_scene_split",
            return_value={"train": [scenes[0]], "validation": [scenes[1]]},
        ),
    ):
        result = prepare_ccdd11_selective_manifests(
            data_root=data_root,
            coarse_root=coarse_root,
            output_dir=output_dir,
            pair_ids=[1],
        )
    return result, data_root, coarse_root, output_dir


def test_miniature_records_have_counts_paths_and_no_leakage(tmp_path):
    result, data_root, coarse_root, output_dir = _prepare_miniature(tmp_path)
    assert result["coarse_images"] == 2
    assert result["train_samples"] == 2
    assert result["validation_samples"] == 2
    manifests = output_dir / "prepared" / "manifests"
    train = [json.loads(line) for line in (manifests / "train.jsonl").read_text().splitlines()]
    validation = [
        json.loads(line) for line in (manifests / "validation.jsonl").read_text().splitlines()
    ]
    assert {row["scene_id"] for row in train}.isdisjoint(
        {row["scene_id"] for row in validation}
    )
    assert all(row["dataset"] == DATASET_NAME for row in train + validation)
    assert all(row["target_kind"] == TARGET_KIND for row in train + validation)
    assert all(Path(row["image"]).parent == coarse_root / "low_haze" for row in train)
    assert all("half_train/main_data/low_haze" in row["ref_image"].replace("\\", "/") for row in train)
    assert all("half_train/sub_data/low_haze" in row["target_image"].replace("\\", "/") for row in train)
    assert all(Path(row["target_image"]).name == f"{row['scene_id']}_{row['preserve']}_.png" for row in train)
    assert all("_half_" not in row["target_image"] for row in train + validation)
    assert all("half_test" not in json.dumps(row) for row in train + validation)
    info = json.loads((output_dir / "prepared" / "split_and_preparation.json").read_text())
    assert info["target_rule"] == "target filename suffix equals preserve"
    assert info["train_scene_ids"] == ["00001"]
    assert info["validation_scene_ids"] == ["00002"]


def test_missing_native_target_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="00001_haze_"):
        _prepare_miniature(tmp_path, omit_target=("00001", "low_haze", "haze"))


def test_coarse_metadata_identity_mismatch_fails(tmp_path):
    data_root, coarse_root, output_dir, scenes = _fixture(tmp_path)
    metadata_path = coarse_root / "coarse_preparation.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["dataset"] = "CDD-11"
    metadata_path.write_text(json.dumps(metadata))
    with (
        patch("difix3d_selective.ccdd_prepare.NUM_SCENES", 2),
        patch(
            "difix3d_selective.ccdd_prepare.make_scene_split",
            return_value={"train": [scenes[0]], "validation": [scenes[1]]},
        ),
        pytest.raises(ValueError, match="Not completed CCDD-11"),
    ):
        prepare_ccdd11_selective_manifests(
            data_root=data_root, coarse_root=coarse_root, output_dir=output_dir, pair_ids=[1]
        )


def test_duplicate_image_stem_is_rejected(tmp_path):
    directory = tmp_path / "images"
    _image(directory / "00001.png")
    _image(directory / "00001.jpg")
    with pytest.raises(ValueError, match="Duplicate scene ID"):
        strict_index_images(directory)


def test_record_validator_rejects_half_and_half_test():
    base = {
        "id": "train/low_haze/00001/remove-low-preserve-haze",
        "scene_id": "00001",
        "pair_id": 1,
        "pair": "low_haze",
        "remove": "low",
        "preserve": "haze",
        "prompt": "remove low light, preserve haze",
        "image": "/coarse/low_haze/00001.png",
        "ref_image": "/data/half_train/main_data/low_haze/00001.png",
        "target_image": "/data/half_train/sub_data/low_haze/00001/00001_haze_.png",
        "clear_image": "/data/half_train/main_data/clear/00001.png",
        "dataset": DATASET_NAME,
        "target_kind": TARGET_KIND,
    }
    validate_manifest_records([base], [1])
    half = dict(base, target_image="/data/half_train/sub_data/low_haze/00001/00001_half_.png")
    with pytest.raises(ValueError):
        validate_manifest_records([half], [1])
    leaked = dict(base, ref_image="/data/half_test/main_data/low_haze/00001.png")
    with pytest.raises(ValueError, match="half_test leakage"):
        validate_manifest_records([leaked], [1])


@pytest.mark.skipif(importlib.util.find_spec("torchvision") is None, reason="torchvision missing")
def test_real_ccdd_manifest_dataset_tensor_contract(tmp_path):
    from difix3d_selective.data import SelectiveDifixDataset

    result, _, _, _ = _prepare_miniature(tmp_path)

    class Tokenizer:
        model_max_length = 4

        def __call__(self, *args, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3, 4]]))

    sample = SelectiveDifixDataset(result["train_manifest"], Tokenizer(), resolution=512)[0]
    assert tuple(sample["conditioning_pixel_values"].shape) == (2, 3, 512, 512)
    assert tuple(sample["output_pixel_values"].shape) == (3, 512, 512)
    assert tuple(sample["ground_truth_pixel_values"].shape) == (3, 512, 512)
    assert torch.equal(sample["input_ids"], torch.tensor([1, 2, 3, 4]))


def test_training_parser_defaults_keep_cdd11_and_ccdd_wrapper_can_override():
    from difix3d_selective.train import parser

    cdd_defaults = {action.dest: action.default for action in parser()._actions}
    ccdd_defaults = {action.dest: action.default for action in parser("ccdd11")._actions}
    assert cdd_defaults["dataset_format"] == "cdd11"
    assert ccdd_defaults["dataset_format"] == "ccdd11"
    assert cdd_defaults["prepare_only"] is False


def test_validation_reports_directed_task_metrics():
    from difix3d_selective.validation import validate

    class Model(torch.nn.Module):
        def forward(self, source, prompt_tokens):
            return torch.zeros_like(source[:, 0])

        def set_train(self):
            self.train()

    class Lpips(torch.nn.Module):
        def forward(self, prediction, target):
            return (prediction - target).abs().mean((1, 2, 3), keepdim=True)

    class Accelerator:
        is_main_process = True

        @staticmethod
        def unwrap_model(model):
            return model

        @staticmethod
        def gather_for_metrics(value):
            return value

    batch = {
        "conditioning_pixel_values": torch.zeros(1, 2, 3, 12, 12),
        "output_pixel_values": torch.zeros(1, 3, 12, 12),
        "ground_truth_pixel_values": torch.zeros(1, 3, 12, 12),
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "sample_id": ["validation/low_haze/00001/remove-low-preserve-haze"],
        "prompt": ["remove low light, preserve haze"],
        "pair_id": torch.tensor([1]),
        "remove": ["low"],
    }
    metrics, _ = validate(Model(), [batch], Lpips(), Accelerator())
    assert "psnr" in metrics
    assert "tasks/low_haze/remove_low/psnr" in metrics
    assert "tasks/low_haze/remove_low/lpips" in metrics


def test_audit_decodes_data_counts_half_and_renders_semantic_montages(tmp_path):
    from verify_ccdd11_selective import audit

    data_root, _, _, scenes = _fixture(tmp_path)
    audit_output = tmp_path / "audit"
    with (
        patch("verify_ccdd11_selective.NUM_SCENES", 2),
        patch(
            "verify_ccdd11_selective.make_scene_split",
            return_value={"train": [scenes[0]], "validation": [scenes[1]]},
        ),
    ):
        report = audit(data_root, [1], audit_output)
    assert report["status"] == "PASS"
    assert report["pairs"]["low_haze"]["half"] == 2
    assert report["pairs"]["low_haze"]["targets"] == {"low": 2, "haze": 2}
    assert len(report["target_semantics"]) == 2
    assert len(list((audit_output / "target_semantics").glob("*.png"))) == 2
    assert (audit_output / "ccdd11_selective_audit.json").is_file()
