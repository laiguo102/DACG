import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from difix3d_selective.compare import paired_rows, summarize_comparison
from difix3d_selective.evaluate import _resolve_device
from difix3d_selective.loss import gram_matrix
from difix3d_selective.main_view import select_main_view, select_main_view_skips
from difix3d_selective.prepare import (
    prepare_selective_triple_test_manifest,
    prepare_selective_manifests,
    prepare_selective_test_manifest,
    tiled_forward,
)
from difix3d_selective.protocol import (
    NUM_TRAIN_SCENES,
    NUM_VALIDATION_SCENES,
    directed_tasks,
    expected_counts,
    expected_test_count,
    expected_triple_test_count,
    make_scene_split,
    selected_pairs,
    selected_triples,
    triple_tasks,
)


class FakeTokenizer:
    model_max_length = 4

    def __call__(self, *args, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3, 4]]))


class TestSelectiveProtocol(unittest.TestCase):
    def test_evaluation_resolves_unindexed_cuda_for_older_torch(self):
        self.assertEqual(_resolve_device("cuda"), torch.device("cuda:0"))
        self.assertEqual(_resolve_device("cuda:2"), torch.device("cuda:2"))
        self.assertEqual(_resolve_device("cpu"), torch.device("cpu"))

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
        self.assertEqual(expected_test_count([1, 3, 5]), 1200)

    def test_triple_ood_tasks_and_prompt_templates(self):
        self.assertEqual(
            selected_triples([1, 2]),
            [(1, "low_haze_rain"), (2, "low_haze_snow")],
        )
        preserve_first = triple_tasks(1, "preserve-first")
        self.assertEqual(len(preserve_first), 3)
        self.assertEqual(
            preserve_first[0].prompt,
            "preserve low light, remove haze and rain",
        )
        self.assertEqual(
            preserve_first[1].prompt,
            "preserve haze, remove low light and rain",
        )
        self.assertEqual(
            triple_tasks(2, "remove-first")[2].prompt,
            "remove low light and haze, preserve snow",
        )
        self.assertEqual(expected_triple_test_count([1, 2]), 1200)


class TestTrainedInitializationComparison(unittest.TestCase):
    @staticmethod
    def _row(sample_id, final_psnr, final_ssim, final_lpips, final_dists):
        row = {
            "sample_id": sample_id,
            "scene_id": sample_id.split("-")[0],
            "pair_id": "1",
            "pair": "low_haze",
            "remove": "low",
            "preserve": "haze",
            "prompt": "remove low light, preserve haze",
            "final_psnr": str(final_psnr),
            "final_ssim": str(final_ssim),
            "final_lpips_vgg": str(final_lpips),
            "final_dists": str(final_dists),
        }
        for stage in ("degraded", "coarse"):
            for metric, value in (
                ("psnr", 10),
                ("ssim", 0.5),
                ("lpips_vgg", 0.4),
                ("dists", 0.3),
            ):
                row[f"{stage}_{metric}"] = str(value)
        return row

    def test_paired_advantage_is_positive_when_trained_is_better(self):
        trained = [self._row("001-a", 25, 0.8, 0.1, 0.05)]
        initialization = [self._row("001-a", 20, 0.7, 0.2, 0.1)]
        row = paired_rows(trained, initialization)[0]
        self.assertEqual(row["trained_advantage_psnr"], 5)
        self.assertAlmostEqual(row["trained_advantage_ssim"], 0.1)
        self.assertAlmostEqual(row["trained_advantage_lpips_vgg"], 0.1)
        self.assertAlmostEqual(row["trained_advantage_dists"], 0.05)

    def test_scene_cluster_bootstrap_reports_trained_better(self):
        trained = [
            self._row(f"{scene}-a", 25, 0.8, 0.1, 0.05)
            for scene in ("001", "002", "003")
        ]
        initialization = [
            self._row(f"{scene}-a", 20, 0.7, 0.2, 0.1)
            for scene in ("001", "002", "003")
        ]
        summary = summarize_comparison(
            paired_rows(trained, initialization), resamples=100, seed=42
        )
        overall = {row["metric"]: row for row in summary if row["group"] == "overall"}
        self.assertEqual(overall["psnr"]["conclusion"], "trained_better")
        self.assertGreater(overall["lpips_vgg"]["ci95_low"], 0)

        triple_summary = summarize_comparison(
            paired_rows(trained, initialization),
            resamples=100,
            seed=42,
            combination_group="triple",
        )
        self.assertTrue(any(row["group"] == "triple" for row in triple_summary))
        self.assertFalse(any(row["group"] == "pair" for row in triple_summary))


class TestMainViewSelection(unittest.TestCase):
    def test_latent_and_all_skips_keep_view_zero(self):
        value = torch.arange(4 * 3).reshape(4, 3)
        selected = select_main_view(value, batch_size=2)
        self.assertTrue(torch.equal(selected, value[[0, 2]]))
        skips = [value, value + 100, value + 200, value + 300]
        selected_skips = select_main_view_skips(skips, batch_size=2)
        for source, result in zip(skips, selected_skips):
            self.assertTrue(torch.equal(result, source[[0, 2]]))


class TestValidationVisualization(unittest.TestCase):
    def test_comparison_order_is_degraded_coarse_target_final_gt(self):
        from difix3d_selective.validation import comparison_image

        source = torch.zeros(1, 2, 3, 12, 8)
        source[:, 0].fill_(-0.5)  # coarse -> 0.25
        source[:, 1].fill_(-1.0)  # degraded -> 0.0
        prediction = torch.zeros(1, 3, 12, 8)  # final -> 0.5
        target = torch.full((1, 3, 12, 8), 0.5)  # target -> 0.75
        ground_truth = torch.ones(1, 3, 12, 8)  # clean GT -> 1.0
        comparison = comparison_image(source, target, prediction, ground_truth)

        self.assertEqual(tuple(comparison.shape), (3, 12, 40))
        panel_means = [float(panel.mean()) for panel in comparison.split(8, dim=-1)]
        self.assertEqual(panel_means, [0.0, 0.25, 0.75, 0.5, 1.0])

    def test_stratified_indices_round_robin_all_directed_tasks(self):
        from difix3d_selective.validation import stratified_indices

        records = []
        for scene in range(3):
            for pair_id in range(1, 6):
                for direction in range(2):
                    records.append(
                        {
                            "pair_id": pair_id,
                            "remove": f"r{direction}",
                            "preserve": f"p{direction}",
                            "scene": scene,
                        }
                    )
        selected = stratified_indices(records, 10)
        keys = {
            (records[index]["pair_id"], records[index]["remove"]) for index in selected
        }
        self.assertEqual(len(selected), 10)
        self.assertEqual(len(keys), 10)

    def test_validation_exposes_primary_and_diagnostic_metrics(self):
        from difix3d_selective.validation import METRIC_NAMES, validate

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

        source = torch.full((1, 2, 3, 12, 12), -0.5)
        batch = {
            "conditioning_pixel_values": source,
            "output_pixel_values": torch.full((1, 3, 12, 12), 0.2),
            "ground_truth_pixel_values": torch.full((1, 3, 12, 12), 0.4),
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "sample_id": ["validation/low_haze/1/remove-low-preserve-haze"],
            "prompt": ["remove low light, preserve haze"],
        }
        metrics, images = validate(
            Model(), [batch], Lpips(), Accelerator(), visualization_limit=1
        )
        self.assertEqual(set(metrics), set(METRIC_NAMES))
        self.assertEqual(len(images), 1)
        self.assertEqual(tuple(images[0][0].shape), (3, 12, 60))


class TestGramLoss(unittest.TestCase):
    def test_gram_matrix_is_per_image_and_batch_one_compatible(self):
        features = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)
        grams = gram_matrix(features)
        expected = torch.stack(
            [image.flatten(1) @ image.flatten(1).T for image in features]
        )
        self.assertEqual(tuple(grams.shape), (2, 3, 3))
        self.assertTrue(torch.equal(grams, expected))
        self.assertTrue(torch.equal(gram_matrix(features[:1])[0], expected[0]))

    def test_second_sample_does_not_change_first_gram(self):
        features = torch.randn(2, 4, 3, 3)
        original = gram_matrix(features)[0]
        features[1].mul_(1000)
        self.assertTrue(torch.equal(gram_matrix(features)[0], original))


class TestDatasetAndPreparation(unittest.TestCase):
    @staticmethod
    def _save(path: Path, rgb):
        array = np.zeros((5, 7, 3), dtype=np.uint8)
        array[:] = rgb
        Image.fromarray(array, "RGB").save(path)

    @unittest.skipIf(
        importlib.util.find_spec("torchvision") is None, "torchvision is not installed"
    )
    def test_dataset_loads_two_conditioning_views_and_one_target(self):
        from difix3d_selective.data import SelectiveDifixDataset

        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            main, reference, target, clear = (
                root / "main.png",
                root / "ref.png",
                root / "target.png",
                root / "clear.png",
            )
            self._save(main, (0, 0, 0))
            self._save(reference, (255, 0, 0))
            self._save(target, (0, 255, 0))
            self._save(clear, (0, 0, 255))
            manifest = root / "train.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "train/low_haze/000001/remove-low-preserve-haze",
                        "prompt": "remove low light, preserve haze",
                        "image": str(main),
                        "ref_image": str(reference),
                        "target_image": str(target),
                        "clear_image": str(clear),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            sample = SelectiveDifixDataset(manifest, FakeTokenizer(), resolution=8)[0]
            self.assertEqual(
                tuple(sample["conditioning_pixel_values"].shape), (2, 3, 8, 8)
            )
            self.assertEqual(tuple(sample["output_pixel_values"].shape), (3, 8, 8))
            self.assertEqual(
                tuple(sample["ground_truth_pixel_values"].shape), (3, 8, 8)
            )
            self.assertTrue(
                torch.equal(sample["input_ids"], torch.tensor([1, 2, 3, 4]))
            )
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
            patch(
                "difix3d_selective.prepare.index_images", side_effect=fake_index_images
            ),
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
            record["prompt"]: Path(record["target_image"]).parent.name
            for record in train_records
        }
        self.assertEqual(by_prompt["remove low light, preserve haze"], "haze")
        self.assertEqual(by_prompt["remove haze, preserve low light"], "low")
        self.assertTrue(
            all(
                Path(record["clear_image"]).parent.name == "clear"
                for record in train_records
            )
        )

    def test_official_test_manifest_builds_both_directions_from_test_only(self):
        root = Path.cwd().resolve()
        data_root = root / "fake-cdd11"
        coarse_root = root / "fake-test-coarse"
        output_dir = root / "fake-test-output"
        scene_ids = ("000001", "000002")
        written = {}

        def fake_index_images(path):
            path = Path(path)
            suffix = ".png" if path.parent == coarse_root else ".jpg"
            return {scene: path / f"{scene}{suffix}" for scene in scene_ids}

        def capture_jsonl(path, records):
            written[Path(path).name] = records

        with (
            patch(
                "difix3d_selective.prepare.index_images", side_effect=fake_index_images
            ),
            patch("difix3d_selective.prepare._write_jsonl", side_effect=capture_jsonl),
            patch("difix3d_selective.prepare.NUM_TEST_SCENES", 2),
        ):
            result = prepare_selective_test_manifest(
                data_root=data_root,
                coarse_root=coarse_root,
                output_dir=output_dir,
                pair_ids=[1],
                require_coarse_metadata=False,
            )

        records = written["manifest.jsonl"]
        self.assertEqual(result["test_samples"], 4)
        self.assertEqual({record["split"] for record in records}, {"test"})
        self.assertEqual(
            {record["prompt"] for record in records},
            {
                "remove low light, preserve haze",
                "remove haze, preserve low light",
            },
        )
        self.assertTrue(
            all(
                Path(record["ref_image"]).parts[-3:-1] == ("test", "low_haze")
                for record in records
            )
        )
        self.assertTrue(
            all(
                Path(record["image"]).parent == coarse_root / "low_haze"
                for record in records
            )
        )

    def test_official_test_rejects_training_coarse_metadata(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory).resolve()
            coarse_root = root / "coarse"
            coarse_root.mkdir()
            (coarse_root / "coarse_preparation.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "split": "train",
                        "data_root": str(root / "CDD11"),
                        "degradation_pairs": [{"id": 1, "name": "low_haze"}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "official-test coarse metadata"):
                prepare_selective_test_manifest(
                    data_root=root / "CDD11",
                    coarse_root=coarse_root,
                    output_dir=root / "output",
                    pair_ids=[1],
                )

    def test_triple_test_manifest_builds_three_preservation_tasks(self):
        root = Path.cwd().resolve()
        data_root = root / "fake-cdd11"
        coarse_root = root / "fake-triple-coarse"
        output_dir = root / "fake-triple-output"
        scene_ids = ("000001", "000002")
        written = {}

        def fake_index_images(path):
            path = Path(path)
            suffix = ".png" if path.parent == coarse_root else ".jpg"
            return {scene: path / f"{scene}{suffix}" for scene in scene_ids}

        def capture_jsonl(path, records):
            written[Path(path).name] = records

        with (
            patch(
                "difix3d_selective.prepare.index_images", side_effect=fake_index_images
            ),
            patch("difix3d_selective.prepare._write_jsonl", side_effect=capture_jsonl),
            patch("difix3d_selective.prepare.NUM_TEST_SCENES", 2),
        ):
            result = prepare_selective_triple_test_manifest(
                data_root=data_root,
                coarse_root=coarse_root,
                output_dir=output_dir,
                triple_ids=[1],
                prompt_template="preserve-first",
                require_coarse_metadata=False,
            )

        records = written["manifest.jsonl"]
        self.assertEqual(result["test_samples"], 6)
        self.assertEqual({record["task_family"] for record in records}, {"triple"})
        self.assertEqual(
            {record["prompt"] for record in records},
            {
                "preserve low light, remove haze and rain",
                "preserve haze, remove low light and rain",
                "preserve rain, remove low light and haze",
            },
        )
        self.assertEqual(
            {Path(record["target_image"]).parent.name for record in records},
            {"low", "haze", "rain"},
        )


class TestSelectiveTestMetrics(unittest.TestCase):
    def test_perceptual_metrics_compare_final_to_coarse_with_positive_gain(self):
        from difix3d_selective.metrics import (
            FullReferenceMetricSuite,
            flatten_sample_metrics,
            summarize_test_rows,
        )

        class MeanDistance(torch.nn.Module):
            def forward(self, prediction, target):
                return (prediction - target).abs().mean((1, 2, 3), keepdim=True)

        target = torch.zeros(1, 3, 12, 12)
        candidates = {
            "degraded": torch.full_like(target, 0.4),
            "coarse": torch.full_like(target, 0.2),
            "final": torch.full_like(target, 0.1),
        }
        suite = FullReferenceMetricSuite(
            torch.device("cpu"),
            lpips_model=MeanDistance(),
            dists_model=MeanDistance(),
        )
        flattened = flatten_sample_metrics(suite(candidates, target))
        for metric in ("psnr", "ssim", "lpips_vgg", "dists"):
            self.assertGreater(flattened[f"improvement_{metric}"], 0)
            self.assertEqual(flattened[f"final_wins_{metric}"], 1)

        rows = []
        for remove, preserve in (("low", "haze"), ("haze", "low")):
            rows.append(
                {
                    "pair": "low_haze",
                    "remove": remove,
                    "preserve": preserve,
                    "inference_time_seconds": 1.0,
                    **flattened,
                }
            )
        summary = summarize_test_rows(rows)
        self.assertEqual(summary["overall"]["macro"]["tasks"], 2)
        self.assertEqual(summary["overall"]["macro"]["win_rate_dists"], 1.0)
        self.assertAlmostEqual(
            summary["overall"]["macro"]["improvement_lpips_vgg"],
            flattened["improvement_lpips_vgg"],
        )
        triple_summary = summarize_test_rows(rows, combination_group="triple")
        self.assertIn("triples", triple_summary)
        self.assertNotIn("pairs", triple_summary)
        with self.assertRaises(ValueError):
            summarize_test_rows(rows, combination_group="quadruple")

    @unittest.skipIf(
        importlib.util.find_spec("torchvision") is None, "torchvision is not installed"
    )
    def test_evaluator_writes_resumable_outputs_and_completed_summary(self):
        from difix3d_selective.evaluate import run

        class FakeModel(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.tokenizer = FakeTokenizer()

            def forward(self, source, prompt_tokens):
                return source[:, 0]

            def set_eval(self):
                self.eval().requires_grad_(False)

        class FakeMetricSuite:
            def __call__(self, candidates, target):
                return {
                    "degraded": {
                        "psnr": 10,
                        "ssim": 0.5,
                        "lpips_vgg": 0.4,
                        "dists": 0.3,
                    },
                    "coarse": {
                        "psnr": 20,
                        "ssim": 0.7,
                        "lpips_vgg": 0.2,
                        "dists": 0.15,
                    },
                    "final": {"psnr": 21, "ssim": 0.75, "lpips_vgg": 0.1, "dists": 0.1},
                }

        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory).resolve()
            images = {}
            for name, color in (
                ("coarse", (64, 64, 64)),
                ("degraded", (32, 32, 32)),
                ("target", (96, 96, 96)),
                ("clear", (128, 128, 128)),
            ):
                images[name] = root / f"{name}.png"
                array = np.zeros((12, 12, 3), dtype=np.uint8)
                array[:] = color
                Image.fromarray(array, "RGB").save(images[name])
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "test/low_haze/000001/remove-low-preserve-haze",
                        "split": "test",
                        "scene_id": "000001",
                        "pair_id": 1,
                        "pair": "low_haze",
                        "remove": "low",
                        "preserve": "haze",
                        "prompt": "remove low light, preserve haze",
                        "image": str(images["coarse"]),
                        "ref_image": str(images["degraded"]),
                        "target_image": str(images["target"]),
                        "clear_image": str(images["clear"]),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            checkpoint = root / "run" / "checkpoints" / "best_psnr.pkl"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint-identity")
            output_dir = root / "test-output"
            args = SimpleNamespace(
                checkpoint=checkpoint,
                degradation_pairs=[1],
                output_dir=output_dir,
                data_root=root / "CDD11",
                test_coarse_root=root / "test-coarse",
                allow_unverified_test_coarse=False,
                resolution=12,
                lora_rank_vae=4,
                timestep=199,
                seed=42,
                mixed_precision="no",
                save_predictions=False,
                device="cpu",
                enable_xformers_memory_efficient_attention=False,
                workers=0,
                num_gallery_samples=0,
            )
            prepared = {
                "manifest": manifest,
                "test_samples": 1,
                "coarse_metadata_verified": True,
            }
            fake_model_module = ModuleType("difix3d_selective.model")
            fake_model_module.SelectiveDifix = FakeModel
            fake_model_module.load_model_checkpoint = lambda model, path: 123
            with (
                patch.dict(
                    sys.modules,
                    {"difix3d_selective.model": fake_model_module},
                ),
                patch(
                    "difix3d_selective.prepare.prepare_selective_test_manifest",
                    return_value=prepared,
                ),
                patch(
                    "difix3d_selective.evaluate.FullReferenceMetricSuite",
                    return_value=FakeMetricSuite(),
                ),
                patch("difix3d_selective.evaluate.expected_test_count", return_value=1),
            ):
                run(args)

            state = json.loads((output_dir / "state.json").read_text(encoding="utf-8"))
            metrics = json.loads(
                (output_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["status"], "completed")
            self.assertEqual(state["global_step"], 123)
            self.assertEqual(metrics["per_image_count"], 1)
            self.assertEqual(metrics["summary"]["overall"]["macro"]["final_psnr"], 21)
            self.assertTrue((output_dir / "per_image_metrics.csv").is_file())
            self.assertTrue((output_dir / "summary.csv").is_file())
            self.assertEqual(len(list((output_dir / "records").rglob("*.json"))), 1)


class TestTrainingPolicy(unittest.TestCase):
    def test_lucid_defaults_and_independent_triggers(self):
        from difix3d_selective.train import gram_is_enabled, parser, trigger_schedule

        action = parser()
        defaults = {item.dest: item.default for item in action._actions}
        self.assertEqual(defaults["max_train_steps"], 100_000)
        self.assertEqual(defaults["train_batch_size"], 4)
        self.assertEqual(defaults["learning_rate"], 5e-6)
        self.assertEqual(defaults["lr_scheduler"], "linear")
        self.assertEqual(defaults["lr_warmup_steps"], 500)
        self.assertEqual(defaults["lambda_gram"], 0.0)
        self.assertTrue(trigger_schedule(500, 100_000, 500))
        self.assertFalse(trigger_schedule(500, 100_000, 1000))
        self.assertTrue(trigger_schedule(100_000, 100_000, 0))
        self.assertFalse(gram_is_enabled(0.0, 10_000, 2000))
        self.assertFalse(gram_is_enabled(1.0, 1999, 2000))
        self.assertTrue(gram_is_enabled(1.0, 2000, 2000))

    @unittest.skipIf(
        importlib.util.find_spec("diffusers") is None, "diffusers is not installed"
    )
    def test_linear_scheduler_warmup_and_decay(self):
        from diffusers.optimization import get_scheduler

        parameter = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.AdamW([parameter], lr=5e-6)
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=100_000,
        )
        values = [optimizer.param_groups[0]["lr"]]
        for _ in range(100_000):
            optimizer.step()
            scheduler.step()
            if scheduler.last_epoch in (500, 100_000):
                values.append(optimizer.param_groups[0]["lr"])
        self.assertAlmostEqual(values[0], 0.0)
        self.assertAlmostEqual(values[1], 5e-6)
        self.assertAlmostEqual(values[2], 0.0)

    def test_backfill_checkpoint_discovery_deduplicates_final(self):
        from difix3d_selective.backfill import discover_checkpoints

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model_1000.pkl").touch()
            (root / "model_002000.pkl").touch()
            torch.save({"global_step": 2000}, root / "final.pkl")
            found = discover_checkpoints(root)
        self.assertEqual([step for step, _ in found], [1000, 2000])

    def test_cdd11_full_checkpoint_loader_uses_model_and_config_keys(self):
        from cdd11_full import model as dacg_model

        tiny = torch.nn.Conv2d(3, 3, 1)
        checkpoint = {
            "config": {"model": {"model_name": "DACG_IR", "id": "dacg"}},
            "architecture": {"model_name": "DACG_IR", "dim": 48},
            "model": tiny.state_dict(),
        }
        with (
            patch.object(dacg_model.torch, "load", return_value=checkpoint),
            patch.object(
                dacg_model, "build_model", return_value=torch.nn.Conv2d(3, 3, 1)
            ),
        ):
            loaded, loaded_checkpoint = dacg_model.load_network(
                "final.pth", torch.device("cpu")
            )
        self.assertEqual(loaded_checkpoint["config"]["model"]["model_name"], "DACG_IR")
        for key, value in tiny.state_dict().items():
            self.assertTrue(torch.equal(loaded.state_dict()[key], value))


if __name__ == "__main__":
    unittest.main()
