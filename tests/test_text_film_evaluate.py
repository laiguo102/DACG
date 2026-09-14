from types import SimpleNamespace

import torch

from difix3d_selective.text_film_evaluate import (
    swapped_prompts,
    text_variant_predictions,
)
from difix3d_selective.validation import validate_text_film


class _FakeTextModel:
    def __init__(self):
        self.unet_calls = 0
        self.detail_prompts = None
        self._last_detail_condition = None
        self.text_film_enabled = True
        self.train_scope = "film"

    def eval(self):
        return self

    def train(self):
        return self

    def set_train(self):
        return None

    def denoised_main_latent(self, source, *, prompt_tokens, prompt_attention_mask):
        self.unet_calls += 1
        self._last_detail_condition = (
            prompt_tokens.float() * prompt_attention_mask.float()
        ).mean(dim=1, keepdim=True)
        return source[:, 0], [source[:, 0]]

    def detail_condition(self, images, *, prompt):
        del images
        self.detail_prompts = prompt
        return torch.full((len(prompt), 1), 0.25)

    def decode_main_latent(
        self,
        latent,
        skips,
        *,
        prompt_condition,
        text_condition_scale=None,
    ):
        del skips
        scale = 1.0 if text_condition_scale is None else text_condition_scale
        return latent + scale * prompt_condition[:, :, None, None]

    def analyze_detail_gates(self, latent, skips, correct, swapped):
        del latent, skips, correct, swapped
        return [{"gate_difference_mean": torch.tensor(0.25)}]


def _batch():
    return {
        "conditioning_pixel_values": torch.zeros(2, 2, 3, 4, 4),
        "input_ids": torch.tensor([[1, 2, 0], [3, 4, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 0]]),
        "remove": ["low", "rain"],
        "preserve": ["haze", "snow"],
    }


def test_swapped_prompts_reverse_only_remove_and_preserve_roles():
    assert swapped_prompts(_batch()) == [
        "remove haze, preserve low light",
        "remove snow, preserve rain",
    ]


def test_text_variants_share_one_correct_prompt_unet_latent():
    model = _FakeTextModel()
    predictions, analysis = text_variant_predictions(
        model,
        _batch(),
        torch.device("cpu"),
        "no",
        42,
        capture_gates=True,
    )

    assert model.unet_calls == 1
    assert model.detail_prompts == swapped_prompts(_batch())
    torch.testing.assert_close(
        predictions["no_text"], torch.zeros_like(predictions["no_text"])
    )
    assert not torch.equal(predictions["correct"], predictions["swapped"])
    assert float(analysis[0]["gate_difference_mean"]) == 0.25


def test_fast_text_validation_reuses_latent_and_names_film_baseline():
    class Accelerator:
        @staticmethod
        def unwrap_model(model):
            return model

        @staticmethod
        def gather_for_metrics(value):
            return value

    model = _FakeTextModel()
    batch = {
        "conditioning_pixel_values": torch.full((1, 2, 3, 4, 4), -0.5),
        "output_pixel_values": torch.full((1, 3, 4, 4), 0.2),
        "input_ids": torch.tensor([[1, 1, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0]]),
        "remove": ["low"],
        "preserve": ["haze"],
    }

    metrics = validate_text_film(model, [batch], Accelerator())

    assert model.unet_calls == 1
    assert set(metrics) == {
        "val/text_film/prompt_swap_output_delta_l1",
        "val/text_film/correct_vs_swapped_target_psnr_gap",
        "val/text_film/output_delta_l1_vs_b20",
    }
