import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from difix3d_selective.cfg_evaluate import CFG_METRIC_FIELDS
from difix3d_selective.factorial_cfg_evaluate import (
    LATENT_GRID,
    PROTOCOL,
    SKIP_GRID,
    SKIP_NAMES_BY_ENCODER_ORDER,
    STAGE_SAMPLE_COUNTS,
    _validate_partial_record,
    _validate_screening_payload,
    condition_filename,
    condition_skips,
    confirm_conditions,
    core_conditions,
    grid_condition,
    interaction_summary,
    paired_effects,
    screen_conditions,
    select_best_screen_cell,
)
from difix3d_selective.protocol import PAIR_FOLDERS, directed_tasks


def screening_payload(*, best=(1.0, 1.0), checkpoint="checkpoint", manifest="manifest"):
    summaries = {}
    for condition in screen_conditions():
        score = 10.0
        if (condition.latent_beta, condition.skip_beta) == best:
            score = 20.0
        summaries[condition.condition_id] = {
            "condition": condition.to_dict(),
            "overall": {"macro": {"positive_target_psnr": score}},
        }
    return {
        "metadata": {
            "protocol": PROTOCOL,
            "config": {
                "stage": "screen",
                "checkpoint": {"sha256": checkpoint},
                "validation_manifest_sha256": manifest,
                "seed": 42,
            },
        },
        "summary": {"conditions": summaries},
    }


class TestConditions(unittest.TestCase):
    def test_condition_sets_have_expected_sizes_and_stable_ids(self):
        self.assertEqual(len(core_conditions()), 13)
        self.assertEqual(len(screen_conditions()), 25)
        self.assertEqual(STAGE_SAMPLE_COUNTS["smoke"] * len(core_conditions()), 260)
        self.assertEqual(STAGE_SAMPLE_COUNTS["screen"] * len(screen_conditions()), 5000)
        self.assertIsNone(STAGE_SAMPLE_COUNTS["confirm"])
        self.assertEqual(grid_condition(1, 0.25).condition_id, "grid:z1:s0.25")
        self.assertEqual(condition_filename("grid:z1:s0.25"), "grid__z1__s0.25")

    def test_skip_masks_follow_decoder_reverse_order(self):
        positive = [torch.full((1,), index + 1.0) for index in range(4)]
        negative = [torch.full((1,), -(index + 1.0)) for index in range(4)]
        only_deep = next(
            condition
            for condition in core_conditions()
            if condition.condition_id == "z1:only-skip_conv_1"
        )
        selected = condition_skips(only_deep, positive, negative)
        self.assertEqual([float(value) for value in selected], [0, 0, 0, 4])
        zero = condition_skips(core_conditions()[4], positive, negative)
        self.assertEqual([float(value) for value in zero], [0, 0, 0, 0])
        blended = condition_skips(grid_condition(1, 0), positive, negative)
        self.assertTrue(all(a is b for a, b in zip(blended, negative)))

    def test_best_screen_tie_break_and_confirm_neighbourhood(self):
        payload = screening_payload(best=(1.1, 0.75))
        self.assertEqual(select_best_screen_cell(payload), (1.1, 0.75))
        conditions = confirm_conditions(payload)
        ids = {condition.condition_id for condition in conditions}
        for expected in (
            "grid:z1.1:s0.75",
            "grid:z1:s0.75",
            "grid:z1.2:s0.75",
            "grid:z1.1:s0.5",
            "grid:z1.1:s1",
            "z1:skip-zero",
            "z1:drop-skip_conv_4",
        ):
            self.assertIn(expected, ids)
        self.assertEqual(len(ids), len(conditions))
        payload["summary"]["conditions"]["grid:z1:s1"]["overall"]["macro"][
            "positive_target_psnr"
        ] = 20.0
        self.assertEqual(select_best_screen_cell(payload), (1.0, 1.0))

    def test_screening_gate_checks_protocol_identity(self):
        payload = screening_payload()
        _validate_screening_payload(
            payload,
            checkpoint_sha256="checkpoint",
            manifest_sha256="manifest",
            seed=42,
        )
        payload["metadata"]["config"]["checkpoint"]["sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
            _validate_screening_payload(
                payload,
                checkpoint_sha256="checkpoint",
                manifest_sha256="manifest",
                seed=42,
            )

    def test_partial_record_requires_the_exact_ordered_condition_set(self):
        record = {
            "sample_id": "sample",
            "results": [{"condition_id": "a"}, {"condition_id": "b"}],
        }
        _validate_partial_record(record, ["a", "b"])
        with self.assertRaisesRegex(ValueError, "different condition set"):
            _validate_partial_record(record, ["b", "a"])


class TestStatistics(unittest.TestCase):
    @staticmethod
    def rows():
        values = {
            "grid:z0:s0": 1.0,
            "grid:z1:s0": 2.0,
            "grid:z0:s1": 4.0,
            "grid:z1:s1": 8.0,
            "z1:skip-zero": 3.0,
        }
        rows = []
        for scene in ("a", "b"):
            for condition_id, value in values.items():
                row = {
                    "sample_id": scene,
                    "scene_id": scene,
                    "condition_id": condition_id,
                }
                row.update({field: value for field in CFG_METRIC_FIELDS})
                rows.append(row)
        return rows

    def test_effects_are_paired_and_direction_aware(self):
        effects, differences = paired_effects(self.rows(), resamples=100, seed=42)
        psnr = next(
            row
            for row in effects
            if row["effect"] == "skip_positive_vs_negative_at_z1"
            and row["metric"] == "positive_target_psnr"
        )
        mse = next(
            row
            for row in effects
            if row["effect"] == "skip_positive_vs_negative_at_z1"
            and row["metric"] == "positive_target_mse_l2"
        )
        self.assertEqual(psnr["advantage_mean"], 6.0)
        self.assertEqual(psnr["conclusion"], "improved")
        self.assertEqual(mse["advantage_mean"], -6.0)
        self.assertEqual(mse["conclusion"], "degraded")
        self.assertTrue(differences)

    def test_interaction_uses_four_corner_contrast(self):
        result = interaction_summary(self.rows(), resamples=100, seed=42)
        mse = next(row for row in result if row["metric"] == "positive_target_mse_l2")
        self.assertEqual(mse["interaction_mean"], 3.0)
        self.assertEqual(mse["conclusion"], "positive")


class TestSmokeIntegration(unittest.TestCase):
    def test_smoke_writes_twenty_by_thirteen_results(self):
        from difix3d_selective.factorial_cfg_evaluate import run

        class Tokenizer:
            model_max_length = 4

            def __call__(self, prompts, **kwargs):
                return SimpleNamespace(
                    input_ids=torch.ones(len(prompts), 4, dtype=torch.long)
                )

        class FakeModel(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.tokenizer = Tokenizer()

            def set_eval(self):
                self.eval().requires_grad_(False)

            def cfg_latents(self, positive_source, negative_source, **kwargs):
                positive = positive_source[:, 0]
                negative = negative_source[:, 0]
                return (
                    positive,
                    negative,
                    [positive * scale for scale in (0.1, 0.2, 0.3, 0.4)],
                    [negative * scale for scale in (0.1, 0.2, 0.3, 0.4)],
                )

            def decode_main_latent(self, latent, skips):
                return (latent + sum(skips)) / 2

        class FakeMetricSuite:
            def __call__(self, prediction, references):
                value = float(prediction.mean())
                return {field: value for field in CFG_METRIC_FIELDS}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run_dir = root / "run"
            manifests = run_dir / "prepared" / "manifests"
            checkpoints = run_dir / "checkpoints"
            manifests.mkdir(parents=True)
            checkpoints.mkdir(parents=True)
            images = {}
            for name, value in (("coarse", 64), ("degraded", 32), ("target", 96), ("clean", 128)):
                path = root / f"{name}.png"
                Image.new("RGB", (8, 8), (value, value, value)).save(path)
                images[name] = path
            records = []
            for scene in range(118):
                for pair_id, pair in PAIR_FOLDERS.items():
                    for task in directed_tasks(pair_id):
                        records.append(
                            {
                                "id": f"validation/{pair}/{scene:06d}/remove-{task.remove}-preserve-{task.preserve}",
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
            (run_dir / "prepared" / "split_and_preparation.json").write_text(
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
            output = root / "output"
            args = SimpleNamespace(
                stage="smoke",
                checkpoint=checkpoint,
                screening_results=None,
                output_dir=output,
                num_gallery_samples=0,
                bootstrap_resamples=100,
                bootstrap_seed=42,
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
            fake_module = ModuleType("difix3d_selective.model")
            fake_module.SelectiveDifix = FakeModel
            fake_module.load_model_checkpoint = lambda *args, **kwargs: 100000
            with (
                patch.dict(sys.modules, {"difix3d_selective.model": fake_module}),
                patch(
                    "difix3d_selective.factorial_cfg_evaluate.CfgMetricSuite",
                    return_value=FakeMetricSuite(),
                ),
            ):
                run(args)
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["validation_sample_count"], 20)
            self.assertEqual(metrics["condition_count"], 13)
            self.assertEqual(metrics["per_image_count"], 260)
            self.assertEqual(len(list((output / "records").rglob("*.json"))), 20)


if __name__ == "__main__":
    unittest.main()
