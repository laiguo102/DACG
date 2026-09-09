import importlib
import sys
from types import ModuleType
from unittest.mock import patch

import pytest
import torch
from torch import nn

from difix3d_selective.daem_lite import (
    GatedDetailSkip,
    LayerNorm2d,
    NAFBlockLite,
    SimpleGate,
)


class _PassthroughBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, value, latent_embeds=None):
        return value + self.dummy * 0


class _Decoder:
    def __init__(self, channels: int, *, detail_enabled: bool):
        self.conv_in = nn.Identity()
        self.mid_block = _PassthroughBlock()
        self.up_blocks = nn.ModuleList([_PassthroughBlock()])
        self.conv_norm_out = nn.Identity()
        self.conv_act = nn.Identity()
        self.conv_out = nn.Identity()
        self.skip_conv_1 = nn.Identity()
        self.skip_conv_2 = nn.Identity()
        self.skip_conv_3 = nn.Identity()
        self.skip_conv_4 = nn.Identity()
        self.incoming_skip_acts = [torch.randn(1, channels, 5, 6)]
        self.incoming_detail_condition = None
        self.detail_enabled = detail_enabled
        self.detail_blocks = nn.ModuleList([GatedDetailSkip(channels)])
        self.gamma = 1


def test_layer_norm_and_simple_gate_shapes():
    value = torch.randn(2, 8, 7, 5)
    assert LayerNorm2d(8)(value).shape == value.shape
    assert SimpleGate()(value).shape == (2, 4, 7, 5)


@pytest.mark.parametrize("channels", [32, 64])
def test_gated_detail_skip_shape_gate_range_and_baseline_equivalence(channels):
    block = GatedDetailSkip(channels, alpha_init=0.1)
    decoder = torch.randn(2, channels, 8, 8)
    projected_skip = torch.randn_like(decoder)

    output = block(decoder, projected_skip)

    assert output.shape == projected_skip.shape
    assert block.last_gate_shape == tuple(projected_skip.shape)
    assert 0.0 <= float(block.last_gate_min) <= float(block.last_gate_max) <= 1.0
    torch.testing.assert_close(output, projected_skip, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(block.last_gate_mean, torch.tensor(0.5))
    assert float(block.last_gate_std) == 0.0
    assert float(block.last_residual_ratio) == 0.0
    assert torch.count_nonzero(block.gate[-1].weight) == 0
    assert torch.count_nonzero(block.gate[-1].bias) == 0


def test_naf_block_is_identity_at_initialization():
    block = NAFBlockLite(16)
    value = torch.randn(2, 16, 6, 7)
    torch.testing.assert_close(block(value), value, atol=0, rtol=0)


def test_prompt_condition_is_spatially_expanded():
    block = GatedDetailSkip(32, prompt_dim=12, prompt_proj_dim=8)
    decoder = torch.randn(2, 32, 6, 5)
    projected_skip = torch.randn_like(decoder)
    prompt = torch.randn(2, 12)

    output = block(decoder, projected_skip, prompt)

    assert output.shape == projected_skip.shape
    assert block.gate[0].in_channels == 72
    with pytest.raises(ValueError, match="prompt_condition"):
        block(decoder, projected_skip)


def test_initial_gate_has_gradient_and_refiner_gradients_are_finite():
    block = GatedDetailSkip(32, alpha_init=0.1)
    decoder = torch.randn(2, 32, 6, 5)
    projected_skip = torch.randn_like(decoder)

    block(decoder, projected_skip).square().mean().backward()

    gate_final_gradient = block.gate[-1].weight.grad
    assert gate_final_gradient is not None
    assert torch.isfinite(gate_final_gradient).all()
    assert torch.count_nonzero(gate_final_gradient) > 0
    refiner_gradients = [
        parameter.grad
        for parameter in block.refiner.parameters()
        if parameter.grad is not None
    ]
    assert refiner_gradients
    assert all(torch.isfinite(gradient).all() for gradient in refiner_gradients)


def test_shape_mismatch_is_rejected():
    block = GatedDetailSkip(32)
    with pytest.raises(ValueError, match="same shape"):
        block(torch.randn(1, 32, 8, 8), torch.randn(1, 32, 4, 4))


def test_decoder_detail_disabled_and_zero_initialized_enabled_match_baseline():
    try:
        model_module = importlib.import_module("difix3d_selective.model")
    except ModuleNotFoundError:
        diffusers = ModuleType("diffusers")
        diffusers.AutoencoderKL = object
        diffusers.DDPMScheduler = object
        peft = ModuleType("peft")
        peft.LoraConfig = object
        transformers = ModuleType("transformers")
        transformers.AutoTokenizer = object
        transformers.CLIPTextModel = object
        torchvision = ModuleType("torchvision")
        torchvision.transforms = ModuleType("torchvision.transforms")
        with patch.dict(
            sys.modules,
            {
                "diffusers": diffusers,
                "peft": peft,
                "transformers": transformers,
                "torchvision": torchvision,
                "torchvision.transforms": torchvision.transforms,
            },
        ):
            model_module = importlib.import_module("difix3d_selective.model")
    vae_decoder_forward = model_module.vae_decoder_forward
    sample = torch.randn(1, 8, 5, 6)
    baseline_decoder = _Decoder(8, detail_enabled=False)
    detail_decoder = _Decoder(8, detail_enabled=True)
    detail_decoder.incoming_skip_acts = baseline_decoder.incoming_skip_acts

    baseline = vae_decoder_forward(baseline_decoder, sample)
    with_detail = vae_decoder_forward(detail_decoder, sample)

    expected = sample + baseline_decoder.incoming_skip_acts[0]
    torch.testing.assert_close(baseline, expected, atol=0, rtol=0)
    torch.testing.assert_close(with_detail, baseline, atol=1e-7, rtol=1e-6)
