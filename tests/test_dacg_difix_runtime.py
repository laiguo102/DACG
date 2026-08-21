"""Offline unittest coverage for DACG-conditioned Difix runtime code."""

from __future__ import annotations

import unittest
from argparse import Namespace
from unittest import mock

import torch

from cdd11_full.protocol import DEGRADATIONS
from dacg_difix.evaluate import summarize_metrics
from dacg_difix.inference import cascade_restore, pad_to_multiple
from dacg_difix.loss import DifixRestorationLoss, gram_matrix
from dacg_difix.train import (
    CHECKPOINT_FORMAT,
    adapter_checkpoint,
    load_adapter_checkpoint,
    parser as train_parser,
    save_training_checkpoint,
)


class _DummyLPIPS(torch.nn.Module):
    def forward(self, prediction, target):
        return (prediction - target).square().mean((1, 2, 3), keepdim=True)


class _DummyGram(torch.nn.Module):
    def forward(self, prediction, target):
        return (prediction.mean((-2, -1)) - target.mean((-2, -1))).square().mean()


class _DummyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.tensor([10.0]), requires_grad=False)
        self.adapter = torch.nn.Parameter(torch.tensor([2.0]))

    def adapter_state_dict(self):
        return {
            "format": CHECKPOINT_FORMAT,
            "config": {"lora_mode": "p-global-layer-id"},
            "state_dict": {"adapter": self.adapter.detach().cpu()},
        }

    def load_adapter_state_dict(self, state, strict=True):
        if not strict:
            raise AssertionError("runtime must load adapter checkpoints strictly")
        self.adapter.data.copy_(state["state_dict"]["adapter"])


class _DummyDACG(torch.nn.Module):
    def forward_with_degradation(self, degraded):
        descriptor = degraded.mean((1, 2, 3)).unsqueeze(1).repeat(1, 96)
        return degraded * 0.5, descriptor


class _DummyDifix(torch.nn.Module):
    def forward(self, main, ref, prompt_tokens, p_global=None):
        if main.shape[-2] % 8 or main.shape[-1] % 8:
            raise AssertionError("Difix input was not padded to a multiple of eight")
        if ref.shape != main.shape or prompt_tokens.shape != (main.shape[0], 77):
            raise AssertionError("two-view or prompt shape mismatch")
        if p_global.shape != (main.shape[0], 96):
            raise AssertionError("DAM descriptor shape mismatch")
        return main


class TestDifixRuntime(unittest.TestCase):
    def test_gram_matrix_is_batched_not_cross_sample(self):
        features = torch.stack((torch.zeros(2, 3, 3), torch.ones(2, 3, 3)))
        grams = gram_matrix(features)
        self.assertEqual(grams.shape, (2, 2, 2))
        self.assertEqual(torch.count_nonzero(grams[0]).item(), 0)
        self.assertTrue(torch.allclose(grams[1], torch.full((2, 2), 9.0)))

    def test_restoration_loss_delays_gram_and_keeps_prediction_gradient(self):
        loss_model = DifixRestorationLoss(
            _DummyLPIPS(),
            _DummyGram(),
            lambda_mse=1,
            lambda_lpips=1,
            lambda_gram=1,
            gram_warmup_steps=2,
        )
        prediction = torch.ones(1, 3, 8, 8, requires_grad=True)
        target = torch.zeros_like(prediction)
        before, before_parts = loss_model(prediction, target, global_step=1)
        after, after_parts = loss_model(prediction, target, global_step=2)
        self.assertEqual(before_parts["gram"].item(), 0)
        self.assertGreater(after_parts["gram"].item(), 0)
        self.assertGreater(after.item(), before.item())
        after.backward()
        self.assertIsNotNone(prediction.grad)

    def test_adapter_checkpoint_excludes_frozen_base_and_round_trips(self):
        model = _DummyAdapter()
        payload = adapter_checkpoint(model, 17, Namespace(note="offline"))
        self.assertEqual(payload["format"], CHECKPOINT_FORMAT)
        self.assertNotIn("base", payload["adapter"]["state_dict"])
        captured = {}
        with mock.patch(
            "dacg_difix.train.atomic_torch_save",
            side_effect=lambda _path, value: captured.update(payload=value),
        ):
            save_training_checkpoint("adapter.pt", model, 17)
        model.adapter.data.zero_()
        with mock.patch("dacg_difix.train.torch.load", return_value=captured["payload"]):
            self.assertEqual(load_adapter_checkpoint("adapter.pt", model), 17)
        self.assertEqual(model.adapter.item(), 2.0)

    def test_cascade_padding_and_crop_preserve_native_size(self):
        degraded = torch.full((1, 3, 13, 17), 0.5)
        padded, size = pad_to_multiple(degraded)
        self.assertEqual(padded.shape[-2:], (16, 24))
        self.assertEqual(size, (13, 17))
        prediction, coarse = cascade_restore(
            _DummyDACG(),
            _DummyDifix(),
            degraded,
            torch.zeros(1, 77, dtype=torch.long),
        )
        self.assertEqual(prediction.shape, degraded.shape)
        self.assertEqual(coarse.shape, degraded.shape)
        self.assertTrue(torch.allclose(prediction, torch.full_like(prediction, 0.25)))

    def test_summary_has_all_11_classes_arity_and_category_macro(self):
        rows = [
            {
                "degradation": name,
                "coarse_psnr": float(index),
                "coarse_ssim": float(index) / 100,
                "coarse_lpips": 1.0 - float(index) / 100,
                "difix_psnr": float(index) + 1,
                "difix_ssim": float(index) / 100 + 0.01,
                "difix_lpips": 0.9 - float(index) / 100,
            }
            for index, name in enumerate(DEGRADATIONS, 1)
        ]
        summary = summarize_metrics(rows)
        self.assertEqual(set(summary["by_degradation"]), set(DEGRADATIONS))
        self.assertEqual(set(summary["by_arity"]), {"single", "double", "triple"})
        self.assertAlmostEqual(summary["macro"]["coarse_psnr"], 6.0)
        self.assertAlmostEqual(summary["macro"]["difix_psnr"], 7.0)
        self.assertEqual(summary["by_arity"]["single"]["categories"], 4)
        self.assertEqual(summary["by_arity"]["triple"]["categories"], 2)

    def test_training_defaults_match_experiment_contract(self):
        parsed = train_parser().parse_args(
            [
                "--train-manifest", "train.jsonl",
                "--validation-manifest", "validation.jsonl",
                "--dam-checkpoint", "dacg.pt",
                "--output-dir", "run",
            ]
        )
        self.assertEqual(parsed.max_train_steps, 10_000)
        self.assertEqual(parsed.mixed_precision, "bf16")
        self.assertEqual(parsed.timestep, 199)
        self.assertEqual(parsed.seed, 42)
        self.assertEqual(
            (parsed.lambda_mse, parsed.lambda_lpips, parsed.lambda_gram),
            (1.0, 1.0, 1.0),
        )
        self.assertEqual(parsed.gram_loss_warmup_steps, 2_000)


if __name__ == "__main__":
    unittest.main()
