import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from difix3d_selective.data import load_records
from difix3d_selective.paired_validation import (
    COARSE_METRICS,
    DETAIL_VAE_COMPARISON_PROFILE,
    NEGATIVE_METRICS,
    POSITIVE_METRICS,
    PROTOCOL,
    _load_reused_reference,
    _load_role_records,
    _negative_ids,
    _record_path,
    _run_prediction,
    _write_record_json,
    paired_rows,
    run,
    sample_seed,
    summarize_effects,
)
from difix3d_selective.protocol import PAIR_FOLDERS, directed_tasks, preserve_pair_prompt
from difix3d_selective.validation import stratified_indices
from difix3d_selective.evaluate import _file_sha256


def _full_manifest_records(root: Path) -> list[dict]:
    images = {
        name: str(root / f"{name}.png")
        for name in ("coarse", "degraded", "target", "clean")
    }
    records = []
    for scene in range(118):
        for pair_id, pair in PAIR_FOLDERS.items():
            for task in directed_tasks(pair_id):
                records.append(
                    {
                        "id": (
                            f"validation/{pair}/{scene:06d}/"
                            f"remove-{task.remove}-preserve-{task.preserve}"
                        ),
                        "split": "validation",
                        "scene_id": f"{scene:06d}",
                        "pair_id": pair_id,
                        "pair": pair,
                        "remove": task.remove,
                        "preserve": task.preserve,
                        "prompt": task.prompt,
                        "image": images["coarse"],
                        "ref_image": images["degraded"],
                        "target_image": images["target"],
                        "clear_image": images["clean"],
                    }
                )
    return records


def _write_run(root: Path, name: str, manifest_text: str) -> Path:
    run_dir = root / name
    prepared = run_dir / "prepared"
    manifests = prepared / "manifests"
    checkpoints = run_dir / "checkpoints"
    manifests.mkdir(parents=True)
    checkpoints.mkdir(parents=True)
    (manifests / "validation.jsonl").write_text(manifest_text, encoding="utf-8")
    (manifests / "train.jsonl").write_text(manifest_text, encoding="utf-8")
    (prepared / "split_and_preparation.json").write_text(
        json.dumps(
            {
                "dataset": "CCDD-11",
                "target_kind": "native_selective_sub_data",
                "degradation_pairs": [
                    {"id": pair_id, "name": pair}
                    for pair_id, pair in PAIR_FOLDERS.items()
                ],
            }
        ),
        encoding="utf-8",
    )
    checkpoint = checkpoints / "best_psnr.pkl"
    checkpoint.write_bytes(name.encode())
    return checkpoint


class _Tokenizer:
    model_max_length = 4

    def __call__(self, prompt, **_):
        return SimpleNamespace(input_ids=torch.ones(1, 4, dtype=torch.long))


class _FakeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest,
        tokenizer,
        *,
        resolution,
        training_mode,
        deduplicate_negative=False,
        **_,
    ):
        del tokenizer, resolution
        records = load_records(manifest)
        if deduplicate_negative:
            unique = {}
            for record in records:
                unique.setdefault((record["pair_id"], record["scene_id"]), record)
            records = list(unique.values())
        self.records = records
        self.training_mode = training_mode

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        negative = self.training_mode == "negative"
        sample_id = (
            f"validation/{record['pair']}/{record['scene_id']}/preserve-both"
            if negative
            else record["id"]
        )
        value = int(record["scene_id"]) / 1000
        coarse = torch.full((3, 4, 4), value)
        degraded = torch.full((3, 4, 4), value + 0.1)
        conditioning = torch.stack([degraded, coarse]) if negative else torch.stack([coarse, degraded])
        return {
            "conditioning_pixel_values": conditioning,
            "output_pixel_values": degraded if negative else torch.full((3, 4, 4), value + 0.2),
            "ground_truth_pixel_values": torch.full((3, 4, 4), value + 0.3),
            "input_ids": torch.ones(4, dtype=torch.long),
            "prompt": preserve_pair_prompt(record["pair_id"]) if negative else record["prompt"],
            "sample_id": sample_id,
            "scene_id": record["scene_id"],
            "pair_id": record["pair_id"],
            "pair": record["pair"],
            "remove": "" if negative else record["remove"],
            "preserve": record["pair"] if negative else record["preserve"],
        }


class _FakeModel(nn.Module):
    successful_calls = 0
    fail_after = None

    def __init__(self, detail_enabled: bool, offset: float):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.detail_enabled = detail_enabled
        self.offset = offset
        self.tokenizer = _Tokenizer()
        self.unet = SimpleNamespace(
            enable_xformers_memory_efficient_attention=lambda: None
        )

    def set_eval(self):
        self.eval().requires_grad_(False)

    def forward(self, images, *, prompt_tokens):
        del prompt_tokens
        if self.fail_after is not None and self.successful_calls >= self.fail_after:
            raise RuntimeError("intentional interruption")
        type(self).successful_calls += 1
        return torch.rand_like(images[:, 0]) * 0.1 + self.offset


class _FakeMetricSuite:
    def __init__(self, _device):
        pass

    def __call__(self, prediction, references):
        value = float(prediction.mean())
        result = {}
        for name in references:
            result[f"{name}_mse_l2"] = 1 - value
            result[f"{name}_psnr"] = value
            result[f"{name}_ssim"] = value
            result[f"{name}_lpips_vgg"] = 1 - value
            result[f"{name}_dists"] = 1 - value
        return result


def _fake_factory(path, **_):
    path_text = str(path)
    detail_vae = "detail_vae" in path_text
    detail_only = "detail_only" in path_text
    candidate = "candidate" in path_text
    detail_enabled = candidate or detail_only or detail_vae
    offset = 0.1 if detail_vae else (0.05 if detail_enabled else 0.0)
    model = _FakeModel(detail_enabled, offset)
    metadata = {
        "dataset": "CCDD-11",
        "seed": 42,
        "pairs": [1, 2, 3, 4, 5],
        "negative_train_probability": 0.2,
        "train_scope": "detail+vae" if detail_vae else ("detail" if detail_enabled else "all"),
    }
    return model, 10000 if detail_enabled else 90000, metadata


def _fake_checkpoint_summary(path: Path) -> dict:
    detail_vae = "detail_vae" in str(path)
    return {
        "global_step": 10000,
        "rank_vae": 4,
        "timestep": 199,
        "detail_config": {
            "enabled": True,
            "num_naf_blocks": 1,
            "gate_reduction": 4,
            "prompt_condition": False,
            "prompt_proj_dim": 32,
        },
        "metadata": {
            "dataset": "CCDD-11",
            "seed": 42,
            "pairs": [1, 2, 3, 4, 5],
            "negative_train_probability": 0.2,
            "train_scope": "detail+vae" if detail_vae else "detail",
            "detail": {"alpha_init": 0.1},
        },
    }


def _args(baseline: Path, candidate: Path, output: Path):
    return SimpleNamespace(
        baseline_checkpoint=baseline,
        candidate_checkpoint=candidate,
        output_dir=output,
        workers=0,
        device="cpu",
        mixed_precision="no",
        seed=42,
        bootstrap_resamples=100,
        bootstrap_seed=42,
        num_gallery_samples=0,
        max_samples=20,
        enable_xformers_memory_efficient_attention=False,
        comparison_profile="baseline-vs-daem",
        reuse_baseline_results=None,
        reuse_baseline_role="candidate",
        baseline_label="baseline",
        candidate_label="candidate",
    )


def _write_reusable_results(
    root: Path,
    checkpoint: Path,
    manifest: Path,
    records: list[dict],
) -> Path:
    results = root / "ab_full"
    records_root = results / "records"
    for source in records:
        sample_id = str(source["id"])
        positive = {
            "sample_id": sample_id,
            "scene_id": str(source["scene_id"]),
            "pair_id": int(source["pair_id"]),
            "pair": str(source["pair"]),
            "remove": str(source["remove"]),
            "preserve": str(source["preserve"]),
            "prompt": str(source["prompt"]),
            "coarse_path": source["image"],
            "degraded_path": source["ref_image"],
            "target_path": source["target_image"],
            "clean_gt_path": source["clear_image"],
            "role": "candidate",
            "mode": "positive",
            "global_step": 10000,
            "inference_seed": sample_seed(42, sample_id),
            "inference_time_seconds": 0.0,
            "prediction_path": "",
        }
        for metric in POSITIVE_METRICS:
            positive[metric] = (
                0.95 if metric.endswith(("mse_l2", "lpips_vgg", "dists")) else 0.05
            )
        positive.update({metric: 0.5 for metric in COARSE_METRICS})
        _write_record_json(
            _record_path(records_root, "candidate", "positive", positive), positive
        )

    by_negative = {}
    for source in records:
        key = (str(source["pair"]), str(source["scene_id"]))
        by_negative.setdefault(key, source)
    for source in by_negative.values():
        sample_id = (
            f"validation/{source['pair']}/{source['scene_id']}/preserve-both"
        )
        negative = {
            "sample_id": sample_id,
            "scene_id": str(source["scene_id"]),
            "pair_id": int(source["pair_id"]),
            "pair": str(source["pair"]),
            "prompt": preserve_pair_prompt(int(source["pair_id"])),
            "role": "candidate",
            "mode": "negative",
            "global_step": 10000,
            "inference_seed": sample_seed(42, sample_id),
            "inference_time_seconds": 0.0,
            "prediction_path": "",
        }
        for metric in NEGATIVE_METRICS:
            negative[metric] = (
                0.95 if metric.endswith(("mae", "mse_l2", "lpips_vgg", "dists")) else 0.05
            )
        _write_record_json(
            _record_path(records_root, "candidate", "negative", negative), negative
        )

    config = {
        "protocol": PROTOCOL,
        "candidate_checkpoint": {"sha256": _file_sha256(checkpoint)},
        "validation_manifest_sha256": _file_sha256(manifest),
        "seed": 42,
        "mixed_precision": "no",
        "xformers_memory_efficient_attention": False,
        "max_samples": None,
    }
    results.mkdir(parents=True, exist_ok=True)
    (results / "state.json").write_text(
        json.dumps({"status": "completed", "config": config}), encoding="utf-8"
    )
    (results / "metrics.json").write_text(
        json.dumps(
            {
                "metadata": {"config": config},
                "counts": {
                    "positive_per_model": 1180,
                    "negative_per_model": 590,
                },
            }
        ),
        encoding="utf-8",
    )
    return results


class TestPairedValidation(unittest.TestCase):
    def test_empty_partial_record_is_treated_as_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            record = (
                Path(directory)
                / "baseline"
                / "positive"
                / "low_haze"
                / "empty.json"
            )
            record.parent.mkdir(parents=True)
            record.write_bytes(b"")
            with self.assertWarnsRegex(UserWarning, "can be recomputed"):
                loaded = _load_role_records(
                    Path(directory), "baseline", "positive"
                )
            self.assertEqual(loaded, {})

    def test_fixed_seed_replays_random_path(self):
        self.assertEqual(sample_seed(42, "a"), sample_seed(42, "a"))
        self.assertNotEqual(sample_seed(42, "a"), sample_seed(42, "b"))
        model = _FakeModel(False, 0.0)
        batch = {
            "conditioning_pixel_values": torch.zeros(1, 2, 3, 4, 4),
            "input_ids": torch.ones(1, 4, dtype=torch.long),
        }
        _FakeModel.successful_calls = 0
        left, _ = _run_prediction(model, batch, torch.device("cpu"), "no", 123)
        torch.rand(100)
        right, _ = _run_prediction(model, batch, torch.device("cpu"), "no", 123)
        torch.testing.assert_close(left, right)

    def test_candidate_advantage_direction_and_bootstrap_are_deterministic(self):
        baseline_positive = {}
        candidate_positive = {}
        baseline_negative = {}
        candidate_negative = {}
        for pair_id, pair in PAIR_FOLDERS.items():
            for task in directed_tasks(pair_id):
                sample_id = f"{pair}/{task.remove}"
                common = {
                    "sample_id": sample_id,
                    "scene_id": "000000",
                    "pair_id": pair_id,
                    "pair": pair,
                    "prompt": task.prompt,
                    "inference_seed": 7,
                    "remove": task.remove,
                    "preserve": task.preserve,
                    "coarse_path": "coarse",
                    "degraded_path": "degraded",
                    "target_path": "target",
                    "clean_gt_path": "clean",
                }
                left = dict(common)
                right = dict(common)
                for metric in POSITIVE_METRICS:
                    lower = metric.endswith(("mse_l2", "lpips_vgg", "dists"))
                    left[metric] = 1.0
                    right[metric] = 0.9 if lower else 1.1
                left.update({metric: 0.5 for metric in COARSE_METRICS})
                right.update({metric: 0.5 for metric in COARSE_METRICS})
                baseline_positive[sample_id] = left
                candidate_positive[sample_id] = right

            negative_id = f"{pair}/negative"
            common_negative = {
                "sample_id": negative_id,
                "scene_id": "000000",
                "pair_id": pair_id,
                "pair": pair,
                "prompt": preserve_pair_prompt(pair_id),
                "inference_seed": 8,
            }
            left = dict(common_negative)
            right = dict(common_negative)
            for metric in NEGATIVE_METRICS:
                lower = metric.endswith(("mae", "mse_l2", "lpips_vgg", "dists"))
                left[metric] = 1.0
                right[metric] = 0.9 if lower else 1.1
            baseline_negative[negative_id] = left
            candidate_negative[negative_id] = right

        positive = paired_rows(
            baseline_positive, candidate_positive, metrics=POSITIVE_METRICS, mode="positive"
        )
        negative = paired_rows(
            baseline_negative, candidate_negative, metrics=NEGATIVE_METRICS, mode="negative"
        )
        self.assertTrue(
            all(row["candidate_advantage_target_psnr"] > 0 for row in positive)
        )
        self.assertTrue(
            all(row["candidate_advantage_target_mse_l2"] > 0 for row in positive)
        )
        effects_a, conditions = summarize_effects(
            positive, negative, resamples=100, seed=42
        )
        effects_b, _ = summarize_effects(positive, negative, resamples=100, seed=42)
        self.assertEqual(effects_a, effects_b)
        self.assertTrue(all(row["candidate_advantage_mean"] > 0 for row in effects_a))
        self.assertTrue(any(row["model"] == "coarse" for row in conditions))

    def test_full_counts_negative_dedup_and_gallery_stratification(self):
        records = _full_manifest_records(Path("/unused"))
        self.assertEqual(len(records), 1180)
        self.assertEqual(len(_negative_ids(records)), 590)
        self.assertEqual(len(records) + len(_negative_ids(records)), 1770)
        gallery = [records[index] for index in stratified_indices(records, 20)]
        tasks = {(row["pair_id"], row["remove"], row["preserve"]) for row in gallery}
        self.assertEqual(len(tasks), 10)
        self.assertEqual({sum(1 for row in gallery if (row["pair_id"], row["remove"], row["preserve"]) == task) for task in tasks}, {2})

    def test_interrupted_smoke_resumes_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = _full_manifest_records(root)
            manifest_text = "\n".join(json.dumps(row) for row in records) + "\n"
            baseline = _write_run(root, "baseline", manifest_text)
            candidate = _write_run(root, "candidate", manifest_text)
            output = root / "output"
            args = _args(baseline, candidate, output)
            _FakeModel.successful_calls = 0
            _FakeModel.fail_after = 5
            fake_model_module = ModuleType("difix3d_selective.model")
            fake_model_module.load_model_from_checkpoint = _fake_factory
            with (
                patch.dict(sys.modules, {"difix3d_selective.model": fake_model_module}),
                patch(
                    "difix3d_selective.paired_validation.SelectiveDifixDataset",
                    _FakeDataset,
                ),
                patch(
                    "difix3d_selective.paired_validation.PairedMetricSuite",
                    _FakeMetricSuite,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "intentional interruption"):
                    run(args)
            self.assertEqual(
                len(list((output / "records" / "baseline" / "positive").rglob("*.json"))),
                5,
            )

            _FakeModel.fail_after = None
            with (
                patch.dict(sys.modules, {"difix3d_selective.model": fake_model_module}),
                patch(
                    "difix3d_selective.paired_validation.SelectiveDifixDataset",
                    _FakeDataset,
                ),
                patch(
                    "difix3d_selective.paired_validation.PairedMetricSuite",
                    _FakeMetricSuite,
                ),
            ):
                run(args)
            self.assertEqual(_FakeModel.successful_calls, 60)
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["counts"]["positive_per_model"], 20)
            self.assertEqual(metrics["counts"]["negative_per_model"], 10)
            self.assertEqual(metrics["counts"]["total_model_forwards"], 60)
            state = json.loads((output / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "completed")
            with (output / "positive_per_image.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 20)
            self.assertAlmostEqual(float(rows[0]["raw_delta_target_psnr"]), 0.05, places=6)

            (output / "metrics.json").unlink()
            args.bootstrap_seed = 43
            with self.assertRaisesRegex(RuntimeError, "different configuration"):
                run(args)

    def test_manifest_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = _full_manifest_records(root)
            manifest_text = "\n".join(json.dumps(row) for row in records) + "\n"
            baseline = _write_run(root, "baseline", manifest_text)
            candidate = _write_run(root, "candidate", manifest_text + " ")
            with self.assertRaisesRegex(ValueError, "manifest SHA256 differ"):
                run(_args(baseline, candidate, root / "output"))

    def test_detail_vae_profile_reuses_detail_only_without_loading_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = _full_manifest_records(root)
            manifest_text = "\n".join(json.dumps(row) for row in records) + "\n"
            detail_only = _write_run(root, "detail_only", manifest_text)
            detail_vae = _write_run(root, "detail_vae", manifest_text)
            reference = _write_reusable_results(
                root,
                detail_only,
                detail_only.parent.parent / "prepared" / "manifests" / "validation.jsonl",
                records,
            )
            args = _args(detail_only, detail_vae, root / "output")
            args.comparison_profile = DETAIL_VAE_COMPARISON_PROFILE
            args.reuse_baseline_results = reference
            args.baseline_label = "detail_only"
            args.candidate_label = "detail_vae"
            factory_calls = []

            def tracked_factory(path, **kwargs):
                factory_calls.append(Path(path))
                return _fake_factory(path, **kwargs)

            _FakeModel.successful_calls = 0
            _FakeModel.fail_after = 5
            fake_model_module = ModuleType("difix3d_selective.model")
            fake_model_module.load_model_from_checkpoint = tracked_factory
            with (
                patch.dict(sys.modules, {"difix3d_selective.model": fake_model_module}),
                patch(
                    "difix3d_selective.paired_validation.SelectiveDifixDataset",
                    _FakeDataset,
                ),
                patch(
                    "difix3d_selective.paired_validation.PairedMetricSuite",
                    _FakeMetricSuite,
                ),
                patch(
                    "difix3d_selective.paired_validation._checkpoint_summary",
                    side_effect=_fake_checkpoint_summary,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "intentional interruption"):
                    run(args)

            self.assertEqual(
                len(
                    list(
                        (root / "output" / "records" / "candidate" / "positive").rglob(
                            "*.json"
                        )
                    )
                ),
                5,
            )
            _FakeModel.fail_after = None
            with (
                patch.dict(sys.modules, {"difix3d_selective.model": fake_model_module}),
                patch(
                    "difix3d_selective.paired_validation.SelectiveDifixDataset",
                    _FakeDataset,
                ),
                patch(
                    "difix3d_selective.paired_validation.PairedMetricSuite",
                    _FakeMetricSuite,
                ),
                patch(
                    "difix3d_selective.paired_validation._checkpoint_summary",
                    side_effect=_fake_checkpoint_summary,
                ),
            ):
                run(args)

            self.assertEqual(factory_calls, [detail_vae.resolve(), detail_vae.resolve()])
            self.assertEqual(_FakeModel.successful_calls, 30)
            metrics = json.loads(
                (root / "output" / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["counts"]["new_model_forwards"], 30)
            self.assertEqual(metrics["counts"]["reused_reference_records"], 30)
            self.assertEqual(
                metrics["metadata"]["comparison_profile"],
                DETAIL_VAE_COMPARISON_PROFILE,
            )
            self.assertEqual(
                metrics["metadata"]["config"]["reused_reference"][
                    "positive_records"
                ],
                1180,
            )
            with (root / "output" / "positive_per_image.csv").open(
                encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["baseline_label"], "detail_only")
            self.assertEqual(rows[0]["candidate_label"], "detail_vae")
            self.assertGreater(float(rows[0]["raw_delta_target_psnr"]), 0)

            state_path = reference / "state.json"
            metrics_path = reference / "metrics.json"
            source_state = json.loads(state_path.read_text(encoding="utf-8"))
            source_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            source_state["config"]["seed"] = 41
            source_metrics["metadata"]["config"]["seed"] = 41
            state_path.write_text(json.dumps(source_state), encoding="utf-8")
            metrics_path.write_text(json.dumps(source_metrics), encoding="utf-8")
            args.output_dir = root / "seed_mismatch"
            with patch(
                "difix3d_selective.paired_validation._checkpoint_summary",
                side_effect=_fake_checkpoint_summary,
            ):
                with self.assertRaisesRegex(ValueError, "different seed"):
                    run(args)

    def test_detail_vae_profile_rejects_scope_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = _full_manifest_records(root)
            manifest_text = "\n".join(json.dumps(row) for row in records) + "\n"
            detail_only = _write_run(root, "detail_only", manifest_text)
            detail_vae = _write_run(root, "detail_vae", manifest_text)
            args = _args(detail_only, detail_vae, root / "output")
            args.comparison_profile = DETAIL_VAE_COMPARISON_PROFILE
            args.reuse_baseline_results = root / "unused"

            def bad_summary(path):
                summary = _fake_checkpoint_summary(path)
                summary["metadata"]["train_scope"] = "detail"
                return summary

            with patch(
                "difix3d_selective.paired_validation._checkpoint_summary",
                side_effect=bad_summary,
            ):
                with self.assertRaisesRegex(ValueError, "expected 'detail\\+vae'"):
                    run(args)

    def test_reused_reference_rejects_protocol_identity_and_sample_mismatches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "reference"
            records_root = results / "records"
            positive_id = "validation/low_haze/000000/remove-low-preserve-haze"
            negative_id = "validation/low_haze/000000/preserve-both"
            positive = {
                "sample_id": positive_id,
                "pair": "low_haze",
                "scene_id": "000000",
                "remove": "low",
                "preserve": "haze",
                "global_step": 10000,
                "inference_seed": sample_seed(42, positive_id),
            }
            negative = {
                "sample_id": negative_id,
                "pair": "low_haze",
                "scene_id": "000000",
                "global_step": 10000,
                "inference_seed": sample_seed(42, negative_id),
            }
            _write_record_json(
                _record_path(records_root, "candidate", "positive", positive),
                positive,
            )
            _write_record_json(
                _record_path(records_root, "candidate", "negative", negative),
                negative,
            )
            base_config = {
                "protocol": PROTOCOL,
                "candidate_checkpoint": {"sha256": "checkpoint-sha"},
                "validation_manifest_sha256": "manifest-sha",
                "seed": 42,
                "mixed_precision": "bf16",
                "xformers_memory_efficient_attention": True,
                "max_samples": None,
            }

            def write_source(config):
                results.mkdir(parents=True, exist_ok=True)
                (results / "state.json").write_text(
                    json.dumps({"status": "completed", "config": config}),
                    encoding="utf-8",
                )
                (results / "metrics.json").write_text(
                    json.dumps(
                        {
                            "metadata": {"config": config},
                            "counts": {
                                "positive_per_model": 1,
                                "negative_per_model": 1,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            common = {
                "role": "candidate",
                "baseline_checkpoint_sha256": "checkpoint-sha",
                "validation_manifest_sha256": "manifest-sha",
                "seed": 42,
                "mixed_precision": "bf16",
                "xformers": True,
                "full_positive_ids": {positive_id},
                "full_negative_ids": {negative_id},
                "selected_positive_ids": {positive_id},
                "selected_negative_ids": {negative_id},
                "gallery_ids": set(),
            }
            write_source(base_config)
            reused_positive, reused_negative, _, step = _load_reused_reference(
                results, **common
            )
            self.assertEqual(
                (set(reused_positive), set(reused_negative), step),
                ({positive_id}, {negative_id}, 10000),
            )

            cases = (
                ("protocol", "wrong", "different protocol"),
                ("seed", 41, "different seed"),
                ("validation_manifest_sha256", "wrong", "different validation manifest"),
            )
            for field, value, message in cases:
                with self.subTest(field=field):
                    config = json.loads(json.dumps(base_config))
                    config[field] = value
                    write_source(config)
                    with self.assertRaisesRegex(ValueError, message):
                        _load_reused_reference(results, **common)

            config = json.loads(json.dumps(base_config))
            config["candidate_checkpoint"]["sha256"] = "wrong"
            write_source(config)
            with self.assertRaisesRegex(ValueError, "checkpoint SHA256 mismatch"):
                _load_reused_reference(results, **common)

            write_source(base_config)
            mismatched = dict(common)
            mismatched["full_positive_ids"] = {positive_id, "missing"}
            with self.assertRaisesRegex(ValueError, "sample set"):
                _load_reused_reference(results, **mismatched)

            bad_positive = dict(positive)
            bad_positive["inference_seed"] = 0
            _write_record_json(
                _record_path(records_root, "candidate", "positive", bad_positive),
                bad_positive,
            )
            with self.assertRaisesRegex(ValueError, "reference seed mismatch"):
                _load_reused_reference(results, **common)


if __name__ == "__main__":
    unittest.main()
