import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from difix3d_selective.cfg import (
    blend_cfg_latents,
    blend_cfg_pixels,
    blend_cfg_skips,
)
from difix3d_selective.cfg_evaluate import (
    CFG_METRIC_FIELDS,
    CfgMetricSuite,
    _log_wandb,
    normalize_betas,
    save_contact_sheet,
    summarize_cfg_rows,
    upload_wandb_results,
    validate_full_manifest,
    wandb_task_curves,
)
from difix3d_selective.data import negative_conditioning
from difix3d_selective.evaluate import _atomic_json, _read_json


class MeanDistance(torch.nn.Module):
    def forward(self, prediction, target):
        return (prediction - target).abs().mean((1, 2, 3), keepdim=True)


class TestCfgInputs(unittest.TestCase):
    def test_read_json_retries_transient_empty_file(self):
        destination = Path("transient-record.json")
        payload = json.dumps({"sample_id": "validation/example"}).encode("utf-8")
        with (
            patch.object(Path, "read_bytes", side_effect=[b"", payload]),
            patch("difix3d_selective.evaluate.time.sleep"),
        ):
            self.assertEqual(
                _read_json(destination), {"sample_id": "validation/example"}
            )

    def test_atomic_json_repairs_empty_destination_after_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "record.json"
            original_replace = Path.replace

            def replace_then_expose_empty(source, target):
                result = original_replace(source, target)
                Path(target).write_bytes(b"")
                return result

            with patch.object(Path, "replace", new=replace_then_expose_empty):
                _atomic_json(destination, {"sample_id": "validation/example"})

            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                {"sample_id": "validation/example"},
            )

    def test_negative_conditioning_matches_training_definition(self):
        coarse = torch.full((2, 3, 4, 5), -0.5)
        degraded = torch.full_like(coarse, 0.5)
        result = negative_conditioning(coarse, degraded)
        self.assertEqual(tuple(result.shape), (2, 2, 3, 4, 5))
        self.assertTrue(torch.equal(result[:, 0], degraded))
        self.assertTrue(torch.equal(result[:, 1], torch.full_like(coarse, 0.5)))

    def test_beta_grid_is_sorted_unique_and_non_negative(self):
        self.assertEqual(normalize_betas([1.2, 0, 1, 0.25]), [0, 0.25, 1, 1.2])
        with self.assertRaisesRegex(ValueError, "unique"):
            normalize_betas([0, 0])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            normalize_betas([-0.1, 1])

    def test_latent_cfg_endpoints_and_extrapolation(self):
        positive = torch.tensor([[[[3.0]]]])
        negative = torch.tensor([[[[1.0]]]])
        self.assertIs(blend_cfg_latents(positive, negative, 0), negative)
        self.assertIs(blend_cfg_latents(positive, negative, 1), positive)
        self.assertAlmostEqual(
            float(blend_cfg_latents(positive, negative, 1.2)), 3.4, places=6
        )
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            blend_cfg_latents(positive, negative.expand(2, -1, -1, -1), 1)

        positive_skips = [positive, positive + 2]
        negative_skips = [negative, negative + 1]
        blended_zero = blend_cfg_skips(positive_skips, negative_skips, 0)
        blended_one = blend_cfg_skips(positive_skips, negative_skips, 1)
        self.assertTrue(
            all(a is b for a, b in zip(blended_zero, negative_skips))
        )
        self.assertTrue(
            all(a is b for a, b in zip(blended_one, positive_skips))
        )
        with self.assertRaisesRegex(ValueError, "skip count mismatch"):
            blend_cfg_skips(positive_skips, negative_skips[:1], 0.5)

    def test_pixel_cfg_endpoints_interpolation_and_clipped_extrapolation(self):
        import numpy as np

        negative = np.full((2, 3, 3), 0.2, dtype=np.float32)
        positive = np.full((2, 3, 3), 0.8, dtype=np.float32)
        self.assertIs(blend_cfg_pixels(positive, negative, 0), negative)
        self.assertIs(blend_cfg_pixels(positive, negative, 1), positive)
        np.testing.assert_allclose(blend_cfg_pixels(positive, negative, 0.5), 0.5)
        np.testing.assert_allclose(blend_cfg_pixels(positive, negative, 2), 1.0)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            blend_cfg_pixels(positive, negative[:1], 0.5)


class TestCfgMetrics(unittest.TestCase):
    def test_metric_suite_reports_all_three_references_and_training_objective(self):
        prediction = torch.zeros(1, 3, 12, 12)
        references = {
            "positive_target": torch.full_like(prediction, 0.2),
            "negative_identity": torch.full_like(prediction, -0.2),
            "clean_gt": torch.full_like(prediction, 0.4),
        }
        suite = CfgMetricSuite(
            torch.device("cpu"),
            lpips_model=MeanDistance(),
            dists_model=MeanDistance(),
        )
        result = suite(prediction, references)
        self.assertEqual(set(result), set(CFG_METRIC_FIELDS))
        self.assertAlmostEqual(result["positive_target_mse_l2"], 0.04, places=6)
        self.assertAlmostEqual(result["positive_target_lpips_vgg"], 0.2, places=6)
        self.assertAlmostEqual(
            result["positive_target_objective_l2_lpips"], 0.24, places=6
        )

    @staticmethod
    def _row(beta, remove, positive_objective, negative_objective, psnr):
        row = {
            "beta": beta,
            "pair": "low_haze",
            "remove": remove,
            "preserve": "haze" if remove == "low" else "low",
            "branch_inference_time_seconds": 1.0,
            "decode_time_seconds": 0.5,
        }
        for field in CFG_METRIC_FIELDS:
            row[field] = 0.1
        row["positive_target_objective_l2_lpips"] = positive_objective
        row["negative_identity_objective_l2_lpips"] = negative_objective
        row["positive_target_psnr"] = psnr
        return row

    def test_summary_selects_positive_optima_and_reports_pareto_betas(self):
        rows = []
        for remove in ("low", "haze"):
            rows.append(self._row(0.0, remove, 2.0, 0.0, 10.0))
            rows.append(self._row(1.0, remove, 0.0, 2.0, 30.0))
            rows.append(self._row(1.2, remove, 1.0, 3.0, 20.0))
        summary = summarize_cfg_rows(rows)
        self.assertEqual(
            summary["selection"]["best_positive_target_psnr_beta"], 1.0
        )
        self.assertEqual(
            summary["selection"]["best_positive_target_objective_beta"], 1.0
        )
        self.assertEqual(
            summary["selection"]["positive_identity_objective_pareto_betas"],
            [0.0, 1.0],
        )
        self.assertEqual(summary["betas"]["1"]["overall"]["macro"]["tasks"], 2)

    def test_wandb_curves_use_beta_and_compare_every_directed_task(self):
        rows = []
        for pair_index in range(1, 6):
            for remove in ("low", "haze"):
                for beta, values in (
                    (0.0, (2.0, 0.0, 10.0)),
                    (1.0, (0.0, 2.0, 30.0)),
                ):
                    row = self._row(beta, remove, *values)
                    row["pair"] = f"pair_{pair_index}"
                    rows.append(row)
        summary = summarize_cfg_rows(rows)
        curves = wandb_task_curves(summary)
        self.assertEqual(set(curves), set(CFG_METRIC_FIELDS))
        self.assertEqual(curves["positive_target_psnr"]["betas"], [0.0, 1.0])
        self.assertEqual(len(curves["positive_target_psnr"]["tasks"]), 10)
        self.assertEqual(len(curves["positive_target_psnr"]["values"]), 10)

        class FakeRun:
            def __init__(self):
                self.defined = []
                self.logged = []
                self.summary = {}
                self.finished = False

            def define_metric(self, name, **kwargs):
                self.defined.append((name, kwargs))

            def log(self, payload):
                self.logged.append(payload)

            def finish(self):
                self.finished = True

        fake_run = FakeRun()
        line_series_calls = []
        fake_wandb = ModuleType("wandb")
        fake_wandb.init = lambda **kwargs: fake_run
        fake_wandb.Table = lambda **kwargs: ("table", kwargs)
        fake_wandb.Image = lambda path: ("image", path)

        def line_series(**kwargs):
            line_series_calls.append(kwargs)
            return ("line_series", kwargs)

        fake_wandb.plot = SimpleNamespace(line_series=line_series)
        args = SimpleNamespace(
            report_to="wandb",
            wandb_entity="entity",
            wandb_project="project",
            wandb_run_name="run",
            output_dir=Path("output"),
        )
        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            _log_wandb(args, {"protocol": "test"}, summary, [])

        scalar_definitions = {
            name: kwargs for name, kwargs in fake_run.defined if name != "cfg/beta"
        }
        self.assertTrue(scalar_definitions)
        self.assertTrue(
            all(
                kwargs == {"step_metric": "cfg/beta"}
                for kwargs in scalar_definitions.values()
            )
        )
        self.assertEqual(len(line_series_calls), len(CFG_METRIC_FIELDS))
        self.assertTrue(all(call["xname"] == "beta" for call in line_series_calls))
        self.assertTrue(all(call["xs"] == [0.0, 1.0] for call in line_series_calls))
        self.assertTrue(all(len(call["keys"]) == 10 for call in line_series_calls))
        self.assertTrue(fake_run.finished)

    def test_contact_sheet_contains_references_and_all_beta_outputs(self):
        references = {
            "negative_identity": torch.zeros(1, 3, 8, 10),
            "coarse": torch.full((1, 3, 8, 10), 0.1),
            "positive_target": torch.full((1, 3, 8, 10), 0.2),
            "clean_gt": torch.full((1, 3, 8, 10), 0.3),
        }
        predictions = {
            beta: torch.full((1, 3, 8, 10), beta / 2)
            for beta in (0, 0.25, 0.5, 0.75, 1, 1.05, 1.1, 1.2)
        }
        with patch.object(Image.Image, "save", autospec=True) as save:
            save_contact_sheet(references, predictions, Path("sheet.png"))
        self.assertEqual(save.call_args.args[0].size, (40, 96))


class TestCfgManifest(unittest.TestCase):
    def test_full_manifest_requires_ten_balanced_directed_tasks(self):
        records = []
        pairs = {
            1: ("low", "haze"),
            2: ("low", "rain"),
            3: ("low", "snow"),
            4: ("haze", "rain"),
            5: ("haze", "snow"),
        }
        for scene in range(118):
            for pair_id, (first, second) in pairs.items():
                pair = f"{first}_{second}"
                for remove, preserve in ((first, second), (second, first)):
                    records.append(
                        {
                            "id": f"validation/{pair}/{scene}/remove-{remove}",
                            "split": "validation",
                            "pair_id": pair_id,
                            "remove": remove,
                            "preserve": preserve,
                        }
                    )
        validate_full_manifest(records)
        with self.assertRaisesRegex(ValueError, "1180"):
            validate_full_manifest(records[:-1])

    def test_cfg_evaluator_writes_resumable_validation_outputs(self):
        from difix3d_selective.cfg_evaluate import run
        from difix3d_selective.protocol import PAIR_FOLDERS, directed_tasks

        class Tokenizer:
            model_max_length = 4

            def __call__(self, prompts, **kwargs):
                return SimpleNamespace(input_ids=torch.ones(len(prompts), 4, dtype=torch.long))

        class FakeModel(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.tokenizer = Tokenizer()

            def set_eval(self):
                self.eval().requires_grad_(False)

            def cfg_latents(
                self,
                positive_source,
                negative_source,
                positive_prompt_tokens,
                negative_prompt_tokens,
            ):
                positive = positive_source[:, 0]
                negative = negative_source[:, 0]
                return positive, negative, [positive], [negative]

            def decode_main_latent(self, latent, skips):
                return (latent + skips[0]) * 0.5

        class FakeMetricSuite:
            def __call__(self, prediction, references):
                value = float(prediction.float().mean())
                return {field: value for field in CFG_METRIC_FIELDS}

        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory).resolve()
            run_dir = root / "run"
            prepared = run_dir / "prepared"
            manifests = prepared / "manifests"
            checkpoints = run_dir / "checkpoints"
            manifests.mkdir(parents=True)
            checkpoints.mkdir(parents=True)
            images = {}
            for name, value in (
                ("coarse", 64),
                ("degraded", 32),
                ("target", 96),
                ("clean", 128),
            ):
                path = root / f"{name}.png"
                Image.new("RGB", (12, 12), (value, value, value)).save(path)
                images[name] = path
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
                                "image": str(images["coarse"]),
                                "ref_image": str(images["degraded"]),
                                "target_image": str(images["target"]),
                                "clear_image": str(images["clean"]),
                            }
                        )
            manifest = manifests / "validation.jsonl"
            manifest.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            preparation = {
                "dataset": "CCDD-11",
                "target_kind": "native_selective_sub_data",
                "degradation_pairs": [
                    {"id": pair_id, "name": pair}
                    for pair_id, pair in PAIR_FOLDERS.items()
                ],
            }
            (prepared / "split_and_preparation.json").write_text(
                json.dumps(preparation), encoding="utf-8"
            )
            checkpoint = checkpoints / "best_psnr.pkl"
            torch.save(
                {
                    "experiment_metadata": {
                        "dataset": "CCDD-11",
                        "seed": 42,
                        "negative_train_probability": 0.2,
                    }
                },
                checkpoint,
            )
            output_dir = root / "cfg-output"
            args = SimpleNamespace(
                checkpoint=checkpoint,
                output_dir=output_dir,
                betas=[0, 1, 1.2],
                num_gallery_samples=2,
                max_samples=2,
                resolution=512,
                lora_rank_vae=4,
                timestep=199,
                workers=0,
                device="cpu",
                mixed_precision="no",
                seed=42,
                enable_xformers_memory_efficient_attention=False,
                report_to="none",
                wandb_entity="unused",
                wandb_project="unused",
                wandb_run_name="unused",
            )
            fake_model_module = ModuleType("difix3d_selective.model")
            fake_model_module.SelectiveDifix = FakeModel
            fake_model_module.load_model_checkpoint = lambda *args, **kwargs: 100000
            with (
                patch.dict(sys.modules, {"difix3d_selective.model": fake_model_module}),
                patch(
                    "difix3d_selective.cfg_evaluate.CfgMetricSuite",
                    return_value=FakeMetricSuite(),
                ),
            ):
                run(args)

            state = json.loads((output_dir / "state.json").read_text(encoding="utf-8"))
            metrics = json.loads(
                (output_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["status"], "completed")
            self.assertEqual(metrics["validation_sample_count"], 2)
            self.assertEqual(metrics["per_image_count"], 6)
            self.assertEqual(
                state["config"]["protocol"],
                "ccdd11-endpoint-correct-state-cfg-validation-v2",
            )
            self.assertEqual(len(list((output_dir / "records").rglob("*.json"))), 2)
            self.assertEqual(len(list((output_dir / "gallery").rglob("contact_sheet.png"))), 2)
            first_record_path = next((output_dir / "records").rglob("*.json"))
            first_record = json.loads(first_record_path.read_text(encoding="utf-8"))
            by_beta = {float(row["beta"]): row for row in first_record["results"]}
            self.assertAlmostEqual(
                by_beta[0.0]["positive_target_mse_l2"], 2 * 32 / 255 - 1,
                places=5,
            )
            self.assertAlmostEqual(
                by_beta[1.0]["positive_target_mse_l2"], 2 * 64 / 255 - 1,
                places=5,
            )

    def test_completed_results_can_be_uploaded_without_inference(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            results_dir = Path(directory).resolve()
            record = {"sample_id": "validation/task/scene"}
            records_dir = results_dir / "records"
            records_dir.mkdir()
            _atomic_json(records_dir / "sample.json", record)
            metrics = {
                "metadata": {"config": {"protocol": "test"}},
                "summary": {"betas": {}, "selection": {}},
                "validation_sample_count": 1,
            }
            _atomic_json(results_dir / "metrics.json", metrics)
            args = SimpleNamespace(
                results_dir=results_dir,
                wandb_entity="entity",
                wandb_project="project",
                wandb_run_name="run",
            )
            with patch(
                "difix3d_selective.cfg_evaluate._log_wandb"
            ) as log_wandb:
                upload_wandb_results(args)
            self.assertEqual(args.output_dir, results_dir)
            self.assertEqual(args.report_to, "wandb")
            self.assertEqual(log_wandb.call_args.args[1], {"protocol": "test"})
            self.assertEqual(log_wandb.call_args.args[3], [record])


if __name__ == "__main__":
    unittest.main()
