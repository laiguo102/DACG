import unittest
from types import SimpleNamespace

import torch
from torch import nn

from dacg_difix.conditional_lora import (
    ConditionalLoRAConditioner,
    assign_condition_matrices,
    install_conditional_lora_forward,
    unet_stage_id,
    vae_decoder_stage_id,
)
from dacg_difix.model import DACGDifix


class FakeLoraLayer(nn.Module):
    def __init__(self, base_layer, lora_a, lora_b, scaling=1.0):
        super().__init__()
        self.base_layer = base_layer
        self.lora_A = nn.ModuleDict({"default": lora_a})
        self.lora_B = nn.ModuleDict({"default": lora_b})
        self.lora_dropout = nn.ModuleDict({"default": nn.Identity()})
        self.scaling = {"default": scaling}
        self.active_adapters = ["default"]
        self.disable_adapters = False
        self.merged = False
        self.use_dora = {"default": False}


class TestConditionalLoraMath(unittest.TestCase):
    def test_linear_c_equals_identity_plus_delta_and_gradients(self):
        layer = FakeLoraLayer(
            nn.Linear(2, 2, bias=False),
            nn.Linear(2, 2, bias=False),
            nn.Linear(2, 2, bias=False),
            scaling=0.5,
        )
        with torch.no_grad():
            layer.base_layer.weight.copy_(torch.eye(2))
            layer.lora_A["default"].weight.copy_(torch.eye(2))
            layer.lora_B["default"].weight.copy_(torch.eye(2))
        install_conditional_lora_forward(layer)

        value = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        delta = torch.tensor(
            [[[0.1, 0.2], [0.3, 0.4]], [[-0.1, 0.0], [0.2, 0.1]]],
            requires_grad=True,
        )
        layer.condition_matrix = torch.eye(2).unsqueeze(0) + delta
        result = layer(value)
        mixed = torch.einsum("br,brs->bs", value, layer.condition_matrix)
        expected = value + mixed * 0.5
        self.assertTrue(torch.allclose(result, expected))

        result.square().sum().backward()
        self.assertIsNotNone(value.grad)
        self.assertIsNotNone(delta.grad)
        self.assertIsNotNone(layer.lora_A["default"].weight.grad)
        self.assertIsNotNone(layer.lora_B["default"].weight.grad)

    def test_conv2d_rank_mixing_and_gradients(self):
        layer = FakeLoraLayer(
            nn.Conv2d(2, 2, 1, bias=False),
            nn.Conv2d(2, 2, 1, bias=False),
            nn.Conv2d(2, 2, 1, bias=False),
        )
        with torch.no_grad():
            identity = torch.eye(2).reshape(2, 2, 1, 1)
            layer.base_layer.weight.zero_()
            layer.lora_A["default"].weight.copy_(identity)
            layer.lora_B["default"].weight.copy_(identity)
        install_conditional_lora_forward(layer)

        value = torch.arange(16.0).reshape(2, 2, 2, 2).requires_grad_()
        matrix = torch.tensor(
            [[[1.0, 0.5], [0.0, 1.0]], [[1.0, 0.0], [0.25, 1.0]]],
            requires_grad=True,
        )
        layer.condition_matrix = matrix
        result = layer(value)
        expected = torch.einsum("brhw,brs->bshw", value, matrix)
        self.assertTrue(torch.allclose(result, expected))

        result.sum().backward()
        self.assertIsNotNone(value.grad)
        self.assertIsNotNone(matrix.grad)
        self.assertIsNotNone(layer.lora_A["default"].weight.grad)


class TestConditioningModesAndStages(unittest.TestCase):
    def test_static_and_p_global_modes(self):
        p_global = torch.randn(3, 96)
        static = ConditionalLoRAConditioner(2, 10, "static")
        static_matrix = static(p_global)
        self.assertEqual(tuple(static_matrix.shape), (3, 1, 2, 2))
        self.assertTrue(torch.equal(static_matrix[0, 0], torch.eye(2)))

        conditional = ConditionalLoRAConditioner(2, 10, "p-global")
        identity_start = conditional(p_global)
        self.assertEqual(tuple(identity_start.shape), (3, 1, 2, 2))
        self.assertTrue(torch.allclose(identity_start[:, 0], torch.eye(2).expand(3, -1, -1)))
        with torch.no_grad():
            conditional.to_delta.weight.normal_(std=0.1)
        p_global.requires_grad_()
        conditional(p_global).sum().backward()
        self.assertIsNotNone(p_global.grad)

    def test_p_global_layer_id_uses_ten_distinct_stage_slots(self):
        conditioner = ConditionalLoRAConditioner(2, 10, "p-global-layer-id")
        with torch.no_grad():
            conditioner.p_global_proj.weight.zero_()
            conditioner.p_global_proj.bias.zero_()
            conditioner.stage_embedding.weight.zero_()
            conditioner.stage_embedding.weight[:, 0] = torch.arange(10.0)
            conditioner.to_delta.weight.zero_()
            conditioner.to_delta.bias.zero_()
            conditioner.to_delta.weight[0, 256] = 1.0
        matrices = conditioner(torch.zeros(1, 96))
        self.assertEqual(tuple(matrices.shape), (1, 10, 2, 2))
        self.assertTrue(torch.equal(matrices[0, :, 0, 0], torch.arange(10.0) + 1.0))

    def test_stage_maps_and_unet_two_view_repeat(self):
        self.assertEqual(unet_stage_id("down_blocks.3.attentions.0.to_q"), 3)
        self.assertEqual(unet_stage_id("mid_block.attentions.0.to_q"), 4)
        self.assertEqual(unet_stage_id("up_blocks.0.attentions.0.to_q"), 5)
        self.assertEqual(unet_stage_id("up_blocks.3.attentions.0.to_q"), 8)
        self.assertEqual(unet_stage_id("conv_out"), 9)
        self.assertEqual(vae_decoder_stage_id("decoder.conv_in"), 0)
        self.assertEqual(vae_decoder_stage_id("decoder.mid_block.attentions.0.to_q"), 0)
        self.assertEqual(vae_decoder_stage_id("decoder.up_blocks.0.resnets.0.conv1"), 1)
        self.assertEqual(vae_decoder_stage_id("decoder.up_blocks.2.resnets.0.conv1"), 3)
        self.assertEqual(vae_decoder_stage_id("decoder.up_blocks.3.resnets.0.conv1"), 4)
        self.assertEqual(vae_decoder_stage_id("decoder.skip_conv_1"), 1)
        self.assertEqual(vae_decoder_stage_id("decoder.skip_conv_4.base_layer"), 4)
        self.assertEqual(vae_decoder_stage_id("decoder.conv_out"), 5)

        class StageRoot(nn.Module):
            def __init__(self):
                super().__init__()
                self.down_blocks = nn.ModuleList(
                    [nn.ModuleDict({"proj": FakeLoraLayer(nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 2))})]
                )
                self.mid_block = nn.ModuleDict(
                    {"proj": FakeLoraLayer(nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 2))}
                )

        root = StageRoot()
        matrices = torch.zeros(2, 10, 2, 2)
        for stage in range(10):
            matrices[:, stage].fill_(stage)
        assign_condition_matrices(root, matrices, unet_stage_id, repeat_interleave=2)
        down = root.down_blocks[0]["proj"].condition_matrix
        middle = root.mid_block["proj"].condition_matrix
        self.assertEqual(tuple(down.shape), (4, 2, 2))
        self.assertTrue(torch.equal(down[:, 0, 0], torch.zeros(4)))
        self.assertTrue(torch.equal(middle[:, 0, 0], torch.full((4,), 4.0)))


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.current_down_blocks = []


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.incoming_skip_acts = []
        self.last_latent = None
        self.last_skips = None


class FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FakeEncoder()
        self.decoder = FakeDecoder()
        self.config = SimpleNamespace(scaling_factor=1.0)

    def encode(self, value):
        self.encoder.current_down_blocks = [value + offset for offset in range(4)]
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: value))

    def decode(self, value):
        self.decoder.last_latent = value
        self.decoder.last_skips = list(self.decoder.incoming_skip_acts)
        return SimpleNamespace(sample=value)


class FakeTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))

    def forward(self, tokens):
        return (tokens.float().unsqueeze(-1),)


class FakeUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.last_sample = None
        self.last_text = None

    def forward(self, sample, timestep, encoder_hidden_states):
        self.last_sample = sample
        self.last_text = encoder_hidden_states
        return SimpleNamespace(sample=torch.zeros_like(sample))


class FakeScheduler:
    def __init__(self):
        self.alphas_cumprod = torch.ones(1000)

    def step(self, prediction, timestep, sample, return_dict=True):
        return SimpleNamespace(prev_sample=sample)


def make_fake_model(mode="static"):
    return DACGDifix(
        pretrained_model_name_or_path="fake",
        lora_mode=mode,
        tokenizer=object(),
        text_encoder=FakeTextEncoder(),
        vae=FakeVAE(),
        unet=FakeUNet(),
        scheduler=FakeScheduler(),
    )


class FakeDAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.last_input = None

    def forward(self, value):
        self.last_input = value
        p_global = value.mean(dim=(1, 2, 3), keepdim=False)[:, None].expand(-1, 96)
        return [], p_global


class TestDACGDifixModel(unittest.TestCase):
    def test_both_views_reach_unet_but_only_view_zero_latent_and_skips_decode(self):
        model = make_fake_model("static")
        main = torch.stack((torch.full((3, 2, 2), 0.2), torch.full((3, 2, 2), 0.4)))
        ref = torch.stack((torch.full((3, 2, 2), 0.8), torch.full((3, 2, 2), 0.9)))
        tokens = torch.tensor([[10], [20]])

        output = model(main, ref, tokens)
        self.assertEqual(tuple(output.shape), (2, 3, 2, 2))
        self.assertTrue(torch.equal(output, main))
        self.assertTrue(torch.equal(model.unet.last_sample[0], main[0]))
        self.assertTrue(torch.equal(model.unet.last_sample[1], ref[0]))
        self.assertTrue(torch.equal(model.unet.last_sample[2], main[1]))
        self.assertTrue(torch.equal(model.vae.decoder.last_latent, main))
        for offset, selected_skip in enumerate(model.vae.decoder.last_skips):
            self.assertTrue(torch.equal(selected_skip, main + offset))
        self.assertTrue(
            torch.equal(model.unet.last_text[:, 0, 0], torch.tensor([10.0, 10.0, 20.0, 20.0]))
        )

    def test_adapter_state_round_trip_and_base_freeze(self):
        source = make_fake_model("p-global")
        self.assertFalse(source.unet.anchor.requires_grad)
        self.assertFalse(source.vae.encoder.anchor.requires_grad)
        self.assertTrue(source.trainable_parameters())
        with torch.no_grad():
            for parameter in source.trainable_parameters():
                parameter.fill_(0.125)
        state = source.adapter_state_dict()
        self.assertEqual(state["format"], "dacg-difix-adapter-v1")

        target = make_fake_model("p-global")
        target.load_adapter_state_dict(state)
        target_state = target.adapter_state_dict()["state_dict"]
        for key, value in state["state_dict"].items():
            self.assertTrue(torch.equal(target_state[key], value), key)

    def test_attached_frozen_dam_supplies_optional_p_global(self):
        dam = FakeDAM()
        model = DACGDifix(
            pretrained_model_name_or_path="fake",
            lora_mode="p-global",
            dam_encoder=dam,
            tokenizer=object(),
            text_encoder=FakeTextEncoder(),
            vae=FakeVAE(),
            unet=FakeUNet(),
            scheduler=FakeScheduler(),
        )
        main = torch.zeros(1, 3, 2, 2)
        ref = torch.full_like(main, -0.5)
        output = model(main, ref, torch.tensor([[1]]))
        self.assertEqual(tuple(output.shape), (1, 3, 2, 2))
        self.assertTrue(torch.allclose(dam.last_input, torch.full_like(ref, 0.25)))
        self.assertFalse(dam.anchor.requires_grad)


if __name__ == "__main__":
    unittest.main()
