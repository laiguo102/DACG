import importlib
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch
from torch import nn

from difix3d_selective.daem_lite import GatedDetailSkip


def _load_model_module():
    try:
        return importlib.import_module("difix3d_selective.model")
    except ModuleNotFoundError:
        diffusers = ModuleType("diffusers")
        diffusers.AutoencoderKL = object
        diffusers.DDPMScheduler = object
        peft = ModuleType("peft")
        peft.LoraConfig = object
        transformers = ModuleType("transformers")
        transformers.AutoTokenizer = object
        transformers.CLIPTextModel = object
        with patch.dict(
            sys.modules,
            {
                "diffusers": diffusers,
                "peft": peft,
                "transformers": transformers,
            },
        ):
            return importlib.import_module("difix3d_selective.model")


model_module = _load_model_module()


class _TinyVae(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.skip_conv_1 = nn.Conv2d(4, 4, 1, bias=False)
        self.decoder.detail_blocks = nn.ModuleList([GatedDetailSkip(4)])
        self.lora_adapter = nn.Linear(4, 4, bias=False)


class _TinySelectiveDifix(nn.Module):
    detail_config = model_module.SelectiveDifix.detail_config
    detail_state_dict = model_module.SelectiveDifix.detail_state_dict

    def __init__(self, *, detail_enabled=True):
        super().__init__()
        self.unet = nn.Linear(4, 4)
        self.vae = _TinyVae()
        self.lora_rank_vae = 4
        self.target_modules_vae = ["decoder.skip_conv_1"]
        self.register_buffer("timesteps", torch.tensor([199], dtype=torch.long))
        self.detail_enabled = detail_enabled
        self.detail_num_blocks = 1
        self.detail_gate_reduction = 4
        self.detail_gate_use_prompt = False
        self.detail_prompt_proj_dim = 32


class _Stateful:
    def __init__(self, value):
        self.value = value

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = state["value"]


def _optimizer_and_scheduler(model):
    del model
    return _Stateful("optimizer"), _Stateful("scheduler")


def _checkpoint_payload(model, optimizer, scheduler, *, step):
    captured = {}
    with (
        patch.object(model_module.torch, "save", side_effect=lambda value, _: captured.update(value)),
        patch.object(model_module.os, "replace"),
    ):
        model_module.save_training_checkpoint(
            model,
            optimizer,
            scheduler,
            Path("checkpoint.pkl"),
            step,
            {"dataset": "CCDD-11"},
        )
    return captured


def test_new_checkpoint_round_trip_preserves_detail_state_and_prediction():
    source = _TinySelectiveDifix()
    with torch.no_grad():
        for parameter in source.vae.decoder.detail_blocks.parameters():
            parameter.uniform_(-0.2, 0.2)
    optimizer, scheduler = _optimizer_and_scheduler(source)
    checkpoint = _checkpoint_payload(source, optimizer, scheduler, step=17)

    decoder = torch.randn(1, 4, 5, 5)
    projected_skip = torch.randn_like(decoder)
    expected = source.vae.decoder.detail_blocks[0](decoder, projected_skip)

    restored = _TinySelectiveDifix()
    restored_optimizer, restored_scheduler = _optimizer_and_scheduler(restored)
    with patch.object(model_module.torch, "load", return_value=checkpoint):
        loaded_step = model_module.load_training_checkpoint(
            restored, restored_optimizer, restored_scheduler, Path("checkpoint.pkl")
        )
    actual = restored.vae.decoder.detail_blocks[0](decoder, projected_skip)

    assert loaded_step == 17
    torch.testing.assert_close(actual, expected)
    for key, value in source.detail_state_dict().items():
        torch.testing.assert_close(restored.detail_state_dict()[key], value)


def test_legacy_checkpoint_keeps_fresh_detail_initialization():
    source = _TinySelectiveDifix(detail_enabled=False)
    optimizer, scheduler = _optimizer_and_scheduler(source)
    legacy = deepcopy(_checkpoint_payload(source, optimizer, scheduler, step=9))
    legacy.pop("detail_config")
    legacy.pop("state_dict_detail")

    target = _TinySelectiveDifix(detail_enabled=True)
    before = {
        key: value.clone() for key, value in target.detail_state_dict().items()
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with patch.object(model_module.torch, "load", return_value=legacy):
            step = model_module.load_model_checkpoint(target, Path("legacy.pkl"))

    assert step == 9
    assert any("legacy SelectiveDifix checkpoint" in str(item.message) for item in caught)
    for key, value in before.items():
        torch.testing.assert_close(target.detail_state_dict()[key], value)


def test_checkpoint_rejects_incompatible_detail_configuration():
    source = _TinySelectiveDifix()
    optimizer, scheduler = _optimizer_and_scheduler(source)
    checkpoint = _checkpoint_payload(source, optimizer, scheduler, step=3)

    target = _TinySelectiveDifix()
    target.detail_gate_reduction = 8
    try:
        with patch.object(model_module.torch, "load", return_value=checkpoint):
            model_module.load_model_checkpoint(target, Path("checkpoint.pkl"))
    except ValueError as error:
        assert "gate_reduction" in str(error)
    else:
        raise AssertionError("Expected an incompatible detail configuration error")
