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
    TextConditionProjection,
    TextFiLMDeltaGate,
    masked_mean_pool,
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
        self.incoming_detail_analysis_conditions = None
        self.last_detail_analysis = []
        self.incoming_text_condition_scale = 1.0
        self.detail_enabled = detail_enabled
        self.detail_text_enabled = False
        self.detail_gate_use_prompt = False
        self.detail_text_projection = None
        self.detail_text_gates = nn.ModuleList()
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


def test_masked_mean_pool_excludes_padding():
    hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    torch.testing.assert_close(
        masked_mean_pool(hidden, mask), torch.tensor([[2.0, 4.0]])
    )


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("channels", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_text_film_delta_is_zero_initialized(batch_size, channels, dtype):
    projection = TextConditionProjection(12, 8).to(dtype=dtype)
    gate = TextFiLMDeltaGate(channels, 8, hidden_ratio=4).to(dtype=dtype)
    decoder = torch.randn(batch_size, channels, 6, 5, dtype=dtype)
    projected_skip = torch.randn_like(decoder)
    condition = projection(torch.randn(batch_size, 12, dtype=dtype))

    delta = gate(decoder, projected_skip, condition)

    assert delta.shape == projected_skip.shape
    torch.testing.assert_close(delta, torch.zeros_like(delta), atol=0, rtol=0)
    assert torch.count_nonzero(gate.delta_out.weight) == 0
    assert torch.count_nonzero(gate.delta_out.bias) == 0
    assert torch.count_nonzero(gate.to_gamma_beta.weight) > 0


@pytest.mark.parametrize("token_count", [2, 7])
def test_masked_mean_pool_handles_variable_token_lengths(token_count):
    hidden = torch.randn(2, token_count, 6)
    mask = torch.ones(2, token_count, dtype=torch.long)
    mask[1, -1] = 0

    pooled = masked_mean_pool(hidden, mask)

    torch.testing.assert_close(pooled[0], hidden[0].mean(0))
    torch.testing.assert_close(pooled[1], hidden[1, :-1].mean(0))


def test_text_film_four_decoder_levels_keep_b20_channels():
    channels = (512, 512, 512, 256)
    gates = nn.ModuleList(TextFiLMDeltaGate(value, 128) for value in channels)
    condition = torch.randn(1, 128)

    for value, gate in zip(channels, gates):
        decoder = torch.randn(1, value, 3, 3)
        delta = gate(decoder, torch.randn_like(decoder), condition)
        assert delta.shape == decoder.shape

    assert [gate.hidden_channels for gate in gates] == [128, 128, 128, 64]


def test_zero_delta_preserves_base_gate_and_scale_zero_is_fallback():
    block = GatedDetailSkip(32)
    text_gate = TextFiLMDeltaGate(32, 8)
    decoder = torch.randn(2, 32, 6, 5)
    projected_skip = torch.randn_like(decoder)
    condition = torch.randn(2, 8)
    baseline = block(decoder, projected_skip)
    delta = text_gate(decoder, projected_skip, condition)

    with_text = block(decoder, projected_skip, delta_logits=delta)
    disabled = block(
        decoder,
        projected_skip,
        delta_logits=torch.randn_like(delta),
        text_condition_scale=0.0,
    )

    torch.testing.assert_close(with_text, baseline, atol=0, rtol=0)
    torch.testing.assert_close(disabled, baseline, atol=0, rtol=0)


def test_text_film_upstream_gradients_begin_after_zero_output_updates():
    projection = TextConditionProjection(12, 8)
    gate = TextFiLMDeltaGate(32, 8)
    optimizer = torch.optim.SGD(
        list(projection.parameters()) + list(gate.parameters()), lr=0.1
    )
    decoder = torch.randn(2, 32, 6, 5)
    projected_skip = torch.randn_like(decoder)
    text = torch.randn(2, 12)

    gate(decoder, projected_skip, projection(text)).sum().backward()
    assert torch.count_nonzero(gate.delta_out.weight.grad) > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    gate(decoder, projected_skip, projection(text)).sum().backward()

    assert torch.count_nonzero(gate.to_gamma_beta.weight.grad) > 0
    assert torch.count_nonzero(projection.projection.weight.grad) > 0


def test_nonzero_delta_branch_is_prompt_sensitive():
    gate = TextFiLMDeltaGate(32, 8)
    with torch.no_grad():
        gate.delta_out.weight.normal_(std=0.01)
    decoder = torch.randn(2, 32, 6, 5)
    projected_skip = torch.randn_like(decoder)
    first = gate(decoder, projected_skip, torch.randn(2, 8))
    second = gate(decoder, projected_skip, torch.randn(2, 8))
    assert not torch.equal(first, second)


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
