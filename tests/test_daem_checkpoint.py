import importlib
import importlib.util
import errno
import os
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from difix3d_selective.daem_lite import GatedDetailSkip


def _load_model_module():
    optional_modules = ("diffusers", "peft", "transformers")
    if any(importlib.util.find_spec(name) is None for name in optional_modules):
        diffusers = ModuleType("diffusers")
        diffusers.AutoencoderKL = object
        diffusers.DDPMScheduler = object
        peft = ModuleType("peft")
        peft.LoraConfig = object
        transformers = ModuleType("transformers")
        transformers.AutoTokenizer = object
        transformers.CLIPTextModel = object
        torchvision = ModuleType("torchvision")
        torchvision_transforms = ModuleType("torchvision.transforms")
        torchvision.transforms = torchvision_transforms
        with patch.dict(
            sys.modules,
            {
                "diffusers": diffusers,
                "peft": peft,
                "transformers": transformers,
                "torchvision": torchvision,
                "torchvision.transforms": torchvision_transforms,
            },
        ):
            return importlib.import_module("difix3d_selective.model")
    return importlib.import_module("difix3d_selective.model")


model_module = _load_model_module()


class _TinyVae(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.skip_conv_1 = nn.Conv2d(4, 4, 1, bias=False)
        self.decoder.detail_blocks = nn.ModuleList([GatedDetailSkip(4)])
        self.lora_vae_skip = nn.Linear(4, 4, bias=False)
        self.encoder = nn.Linear(4, 4, bias=False)


class _TinySelectiveDifix(nn.Module):
    detail_config = model_module.SelectiveDifix.detail_config
    detail_state_dict = model_module.SelectiveDifix.detail_state_dict
    detail_parameters = model_module.SelectiveDifix.detail_parameters
    detail_diagnostics = model_module.SelectiveDifix.detail_diagnostics
    parameter_counts = model_module.SelectiveDifix.parameter_counts
    set_train = model_module.SelectiveDifix.set_train
    trainable_parameters = model_module.SelectiveDifix.trainable_parameters
    vae_adaptation_parameters = model_module.SelectiveDifix.vae_adaptation_parameters

    def __init__(
        self,
        lora_rank_vae=4,
        timestep=199,
        *,
        detail_enabled=True,
        detail_num_blocks=1,
        detail_gate_reduction=4,
        detail_alpha_init=0.1,
        detail_gate_use_prompt=False,
        detail_prompt_proj_dim=32,
    ):
        del detail_alpha_init
        super().__init__()
        self.unet = nn.Linear(4, 4)
        self.vae = _TinyVae()
        self.text_encoder = nn.Linear(4, 4)
        self.lora_rank_vae = lora_rank_vae
        self.target_modules_vae = ["decoder.skip_conv_1"]
        self.register_buffer("timesteps", torch.tensor([timestep], dtype=torch.long))
        self.detail_enabled = detail_enabled
        self.detail_num_blocks = detail_num_blocks
        self.detail_gate_reduction = detail_gate_reduction
        self.detail_gate_use_prompt = detail_gate_use_prompt
        self.detail_prompt_proj_dim = detail_prompt_proj_dim
        self.train_scope = "all"


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


def test_checkpoint_save_ignores_stale_temp_and_retries_eio(tmp_path):
    model = _TinySelectiveDifix()
    optimizer, scheduler = _optimizer_and_scheduler(model)
    destination = tmp_path / "checkpoint.pkl"
    stale_temporary = tmp_path / "checkpoint.pkl.tmp"
    stale_temporary.write_bytes(b"stale")
    replace = os.replace
    replace_sources = []

    def flaky_replace(source, target):
        replace_sources.append(Path(source))
        if len(replace_sources) == 1:
            raise OSError(errno.EIO, "transient mounted-storage error")
        replace(source, target)

    with (
        patch.object(model_module.os, "replace", side_effect=flaky_replace),
        patch.object(model_module.time, "sleep") as sleep,
    ):
        model_module.save_training_checkpoint(
            model,
            optimizer,
            scheduler,
            destination,
            14_000,
            {"dataset": "CCDD-11"},
        )

    checkpoint = torch.load(destination, map_location="cpu", weights_only=False)
    assert checkpoint["global_step"] == 14_000
    assert stale_temporary.read_bytes() == b"stale"
    assert len(replace_sources) == 2
    assert replace_sources[0] != stale_temporary
    assert replace_sources[0].parent == destination.parent
    sleep.assert_called_once_with(1)


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


def test_checkpoint_factory_constructs_saved_detail_architecture_once():
    source = _TinySelectiveDifix(detail_enabled=True)
    optimizer, scheduler = _optimizer_and_scheduler(source)
    checkpoint = _checkpoint_payload(source, optimizer, scheduler, step=23)
    checkpoint["experiment_metadata"] = {
        "dataset": "CCDD-11",
        "seed": 42,
        "pairs": [1, 2, 3, 4, 5],
    }
    constructed = []

    class FactoryTiny(_TinySelectiveDifix):
        def __init__(self, *args, **kwargs):
            constructed.append(dict(kwargs))
            super().__init__(*args, **kwargs)

    with (
        patch.object(model_module, "SelectiveDifix", FactoryTiny),
        patch.object(model_module.torch, "load", return_value=checkpoint) as load,
    ):
        restored, step, metadata = model_module.load_model_from_checkpoint(
            Path("daem.pkl"), expected_dataset="CCDD-11", expected_seed=42
        )

    assert load.call_count == 1
    assert step == 23
    assert restored.detail_enabled
    assert constructed[0]["detail_num_blocks"] == 1
    assert constructed[0]["detail_gate_reduction"] == 4
    assert metadata["pairs"] == [1, 2, 3, 4, 5]


def test_checkpoint_factory_builds_legacy_checkpoint_without_detail():
    source = _TinySelectiveDifix(detail_enabled=False)
    optimizer, scheduler = _optimizer_and_scheduler(source)
    checkpoint = _checkpoint_payload(source, optimizer, scheduler, step=90_000)
    checkpoint.pop("detail_config")
    checkpoint.pop("state_dict_detail")
    checkpoint["experiment_metadata"] = {
        "dataset": "CCDD-11",
        "seed": 42,
        "pairs": [1, 2, 3, 4, 5],
    }

    with (
        patch.object(model_module, "SelectiveDifix", _TinySelectiveDifix),
        patch.object(model_module.torch, "load", return_value=checkpoint),
    ):
        restored, step, _ = model_module.load_model_from_checkpoint(
            Path("baseline.pkl"), expected_dataset="CCDD-11", expected_seed=42
        )

    assert step == 90_000
    assert not restored.detail_enabled


def _parameter_ids(parameters):
    return {id(parameter) for parameter in parameters}


def test_train_scopes_select_exact_parameter_sets():
    model = _TinySelectiveDifix()
    detail = _parameter_ids(model.detail_parameters())
    vae = _parameter_ids(model.vae_adaptation_parameters())
    unet = _parameter_ids(model.unet.parameters())

    model.set_train("detail")
    assert _parameter_ids(model.trainable_parameters()) == detail
    model.set_train("detail+vae")
    assert _parameter_ids(model.trainable_parameters()) == detail | vae
    model.set_train("all")
    assert _parameter_ids(model.trainable_parameters()) == detail | vae | unet
    assert not any(parameter.requires_grad for parameter in model.text_encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.vae.encoder.parameters())


def test_optimizer_groups_keep_baseline_single_group_and_split_detail_learning_rate():
    from difix3d_selective.train import _optimizer_parameter_groups

    baseline = _TinySelectiveDifix(detail_enabled=False)
    baseline.set_train("all")
    baseline_args = SimpleNamespace(
        detail_enabled=False,
        train_scope="all",
        learning_rate=5e-6,
        detail_learning_rate=1e-4,
    )
    baseline_groups = _optimizer_parameter_groups(baseline, baseline_args)
    assert len(baseline_groups) == 1
    assert baseline_groups[0]["lr"] == 5e-6

    detail = _TinySelectiveDifix()
    detail.set_train("detail+vae")
    detail_args = SimpleNamespace(
        detail_enabled=True,
        train_scope="detail+vae",
        learning_rate=5e-6,
        detail_learning_rate=1e-4,
    )
    detail_groups = _optimizer_parameter_groups(detail, detail_args)
    assert [group["lr"] for group in detail_groups] == [5e-6, 1e-4]
    assert _parameter_ids(detail_groups[0]["params"]) == _parameter_ids(
        detail.vae_adaptation_parameters()
    )
    assert _parameter_ids(detail_groups[1]["params"]) == _parameter_ids(
        detail.detail_parameters()
    )

    detail.set_train("detail")
    detail_args.train_scope = "detail"
    detail_only_groups = _optimizer_parameter_groups(detail, detail_args)
    assert [group["lr"] for group in detail_only_groups] == [1e-4]
    assert _parameter_ids(detail_only_groups[0]["params"]) == _parameter_ids(
        detail.detail_parameters()
    )

    detail.set_train("all")
    detail_args.train_scope = "all"
    all_groups = _optimizer_parameter_groups(detail, detail_args)
    assert [group["lr"] for group in all_groups] == [5e-6, 5e-6, 1e-4]
    assert _parameter_ids(all_groups[0]["params"]) == _parameter_ids(
        detail.unet.parameters()
    )
    assert _parameter_ids(all_groups[1]["params"]) == _parameter_ids(
        detail.vae_adaptation_parameters()
    )
    assert _parameter_ids(all_groups[2]["params"]) == _parameter_ids(
        detail.detail_parameters()
    )

def test_parameter_counts_and_detail_diagnostics():
    model = _TinySelectiveDifix()
    model.set_train("detail")
    block = model.vae.decoder.detail_blocks[0]
    decoder = torch.randn(1, 4, 5, 5)
    block(decoder, torch.randn_like(decoder))

    counts = model.parameter_counts()
    assert counts["total"] == counts["baseline_total"] + counts["detail"]
    assert counts["trainable"] == counts["detail"]
    diagnostics = model.detail_diagnostics()
    assert set(diagnostics) == {
        "train/detail/l0/gate_mean",
        "train/detail/l0/gate_std",
        "train/detail/l0/gate_min",
        "train/detail/l0/gate_max",
        "train/detail/l0/alpha",
        "train/detail/l0/residual_ratio",
    }
    assert diagnostics["train/detail/l0/gate_mean"] == 0.5
    assert diagnostics["train/detail/l0/residual_ratio"] == 0.0
