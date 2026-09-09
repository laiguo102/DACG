import argparse
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
    TEST_TARGET_KIND,
    prepare_ccdd11_selective_manifests,
    prepare_ccdd11_selective_test_manifest,
    strict_index_images,
    validate_manifest_records,
)
from difix3d_selective.protocol import expected_test_count


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
        "dacg_checkpoint": "/checkpoints/best_macro_psnr.pth",
        "dacg_checkpoint_sha256": "a" * 64,
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


def _test_fixture(tmp_path: Path, *, metadata_split: str = "half_test"):
    data_root = tmp_path / "CCDD-11"
    coarse_root = tmp_path / "coarse" / "half_test"
    output_dir = tmp_path / "evaluation" / "prepared"
    main_root = data_root / "half_test" / "main_data"
    scenes = ("10001", "10002")
    for scene in scenes:
        _image(main_root / "clear" / f"{scene}.png", 128)
        _image(main_root / "low_haze" / f"{scene}.png", 32)
        _image(main_root / "low" / f"{scene}.png", 70)
        _image(main_root / "haze" / f"{scene}.png", 90)
        _image(coarse_root / "low_haze" / f"{scene}.png", 48)
    metadata = {
        "dataset": DATASET_NAME,
        "status": "completed",
        "split": metadata_split,
        "data_root": str(data_root.resolve()),
        "source_root": str(main_root.resolve()),
        "pair_ids": [1],
        "expected_scenes": 2,
        "dacg_checkpoint": "/checkpoints/best_macro_psnr.pth",
        "dacg_checkpoint_sha256": "a" * 64,
    }
    coarse_root.mkdir(parents=True, exist_ok=True)
    (coarse_root / "coarse_preparation.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    half_train_root = coarse_root.parent / "half_train"
    half_train_root.mkdir(parents=True, exist_ok=True)
    half_train_metadata = {
        **metadata,
        "split": "half_train",
        "source_root": str((data_root / "half_train" / "main_data").resolve()),
        "expected_scenes": 1183,
    }
    (half_train_root / "coarse_preparation.json").write_text(
        json.dumps(half_train_metadata), encoding="utf-8"
    )
    return data_root, coarse_root, output_dir


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
    assert all("_half_" not in Path(row["target_image"]).name for row in train + validation)
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


def test_half_test_old_style_manifest_uses_main_data_preserve_targets(tmp_path):
    data_root, coarse_root, output_dir = _test_fixture(tmp_path)
    with patch("difix3d_selective.ccdd_prepare.NUM_TEST_SCENES", 2):
        result = prepare_ccdd11_selective_test_manifest(
            data_root=data_root,
            coarse_root=coarse_root,
            output_dir=output_dir,
            pair_ids=[1],
        )
    records = [
        json.loads(line)
        for line in Path(result["manifest"]).read_text(encoding="utf-8").splitlines()
    ]
    assert result["test_samples"] == 4
    assert {row["split"] for row in records} == {"half_test"}
    assert {row["target_kind"] for row in records} == {TEST_TARGET_KIND}
    assert {row["dataset"] for row in records} == {DATASET_NAME}
    assert all(Path(row["ref_image"]).parent.name == "low_haze" for row in records)
    assert all(Path(row["target_image"]).parent.name == row["preserve"] for row in records)
    assert all(Path(row["target_image"]).name == f"{row['scene_id']}.png" for row in records)
    assert all("half_train" not in json.dumps(row) for row in records)
    assert all("sub_data" not in json.dumps(row) for row in records)
    assert all("_half_" not in Path(row["target_image"]).name for row in records)
    info = json.loads((output_dir / "test_preparation.json").read_text(encoding="utf-8"))
    assert info["split"] == "half_test"
    assert info["target_kind"] == TEST_TARGET_KIND


def test_half_test_rejects_half_train_coarse_metadata(tmp_path):
    data_root, coarse_root, output_dir = _test_fixture(
        tmp_path, metadata_split="half_train"
    )
    with (
        patch("difix3d_selective.ccdd_prepare.NUM_TEST_SCENES", 2),
        pytest.raises(ValueError, match="half_test coarse metadata"),
    ):
        prepare_ccdd11_selective_test_manifest(
            data_root=data_root,
            coarse_root=coarse_root,
            output_dir=output_dir,
            pair_ids=[1],
        )


def test_half_test_requires_same_dacg_checkpoint_as_half_train(tmp_path):
    data_root, coarse_root, output_dir = _test_fixture(tmp_path)
    metadata_path = coarse_root.parent / "half_train" / "coarse_preparation.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dacg_checkpoint_sha256"] = "b" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with (
        patch("difix3d_selective.ccdd_prepare.NUM_TEST_SCENES", 2),
        pytest.raises(ValueError, match="different dacg_checkpoint_sha256"),
    ):
        prepare_ccdd11_selective_test_manifest(
            data_root=data_root,
            coarse_root=coarse_root,
            output_dir=output_dir,
            pair_ids=[1],
        )


def test_ccdd_full_half_test_protocol_has_2000_records():
    assert expected_test_count([1, 2, 3, 4, 5]) == 2000


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

    negative_validation = SelectiveDifixDataset(
        result["validation_manifest"],
        Tokenizer(),
        resolution=512,
        training_mode="negative",
        deduplicate_negative=True,
    )
    assert len(negative_validation) == 1
    negative = negative_validation[0]
    assert negative["training_mode"] == "negative"
    assert negative["prompt"] == "preserve low light, preserve haze"
    assert torch.equal(
        negative["conditioning_pixel_values"][0], negative["output_pixel_values"]
    )


def test_training_parser_defaults_keep_cdd11_and_ccdd_wrapper_can_override():
    from difix3d_selective.train import parser, probability

    cdd_defaults = {action.dest: action.default for action in parser()._actions}
    ccdd_defaults = {action.dest: action.default for action in parser("ccdd11")._actions}
    assert cdd_defaults["dataset_format"] == "cdd11"
    assert ccdd_defaults["dataset_format"] == "ccdd11"
    assert cdd_defaults["prepare_only"] is False
    assert "negative_train_probability" not in cdd_defaults
    assert ccdd_defaults["negative_train_probability"] == 0.2
    assert cdd_defaults["detail_enabled"] is False
    assert cdd_defaults["detail_num_blocks"] == 1
    assert cdd_defaults["detail_gate_reduction"] == 4
    assert cdd_defaults["detail_alpha_init"] == 0.1
    assert cdd_defaults["detail_gate_use_prompt"] is False
    assert cdd_defaults["train_scope"] == "all"
    assert cdd_defaults["detail_learning_rate"] == 1e-4
    assert probability("0") == 0.0
    assert probability("1") == 1.0
    with pytest.raises(argparse.ArgumentTypeError, match=r"\[0, 1\]"):
        probability("1.01")


def test_ccdd_checkpoint_metadata_describes_negative_training_contract():
    from difix3d_selective.train import _experiment_metadata

    metadata = _experiment_metadata(
        SimpleNamespace(
            dataset_format="ccdd11",
            seed=42,
            degradation_pairs=[1, 2, 3, 4, 5],
            negative_train_probability=0.2,
            train_scope="all",
            detail_enabled=False,
            detail_num_blocks=1,
            detail_gate_reduction=4,
            detail_alpha_init=0.1,
            detail_gate_use_prompt=False,
            detail_prompt_proj_dim=32,
            detail_learning_rate=1e-4,
        )
    )
    assert metadata["negative_train_probability"] == 0.2
    assert metadata["training_mode_sampling"] == "dynamic_per_record_read"
    assert metadata["negative_prompt"] == "preserve A, preserve B"
    assert metadata["negative_condition"] == [
        "original double-degradation image",
        "signed original-minus-DACG texture",
    ]
    assert metadata["detail"]["enabled"] is False


def test_negative_validation_deduplicates_directions_to_590_records(tmp_path):
    from difix3d_selective.data import SelectiveDifixDataset

    records = []
    for scene_index in range(118):
        for pair_id in range(1, 6):
            for direction in range(2):
                records.append(
                    {
                        "scene_id": f"{scene_index:05d}",
                        "pair_id": pair_id,
                        "direction": direction,
                    }
                )
    manifest = tmp_path / "validation.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    dataset = SelectiveDifixDataset(
        manifest,
        tokenizer=None,
        training_mode="negative",
        deduplicate_negative=True,
    )
    assert len(dataset) == 590


def test_evaluation_parser_defaults_keep_cdd11_and_ccdd_wrapper_can_override():
    from difix3d_selective.evaluate import parser

    cdd_defaults = {action.dest: action.default for action in parser()._actions}
    ccdd_defaults = {action.dest: action.default for action in parser("ccdd11")._actions}
    assert cdd_defaults["dataset_format"] == "cdd11"
    assert ccdd_defaults["dataset_format"] == "ccdd11"
    assert ccdd_defaults["resolution"] == 512
    assert ccdd_defaults["lora_rank_vae"] == 4
    assert ccdd_defaults["timestep"] == 199
    assert ccdd_defaults["seed"] == 42


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


def test_negative_validation_reports_identity_metrics_and_visualization():
    from difix3d_selective.validation import validate_negative

    class Model(torch.nn.Module):
        def forward(self, source, prompt_tokens):
            return source[:, 0]

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

    source = torch.zeros(1, 2, 3, 12, 12)
    source[:, 0].fill_(-0.5)
    source[:, 1].fill_(0.25)
    batch = {
        "conditioning_pixel_values": source,
        "output_pixel_values": source[:, 0].clone(),
        "dacg_coarse_pixel_values": torch.full((1, 3, 12, 12), -0.75),
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "sample_id": ["validation/low_haze/00001/negative"],
        "prompt": ["preserve low light, preserve haze"],
        "pair_id": torch.tensor([1]),
    }
    metrics, visualizations = validate_negative(
        Model(), [batch], Lpips(), Accelerator(), visualization_limit=1
    )
    assert metrics["mean_absolute_change"] == 0.0
    assert metrics["lpips"] == 0.0
    assert "pairs/low_haze/psnr" in metrics
    assert len(visualizations) == 1
    assert tuple(visualizations[0][0].shape) == (3, 12, 60)


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
