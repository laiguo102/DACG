"""Audit B-20k and verify zero-init Text-FiLM equivalence before training."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .data import SelectiveDifixDataset
from .evaluate import _atomic_json, _autocast, _file_sha256, _save_image
from .paired_validation import _selected_records, _training_run, sample_seed


def _group_summary(groups: list[dict]) -> list[dict]:
    return [
        {
            "name": group["name"],
            "parameters": sum(parameter.numel() for parameter in group["params"]),
            "learning_rate": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
        }
        for group in groups
    ]


def _build_text_model(checkpoint: dict):
    from .model import SelectiveDifix

    detail = checkpoint["detail_config"]
    return SelectiveDifix(
        lora_rank_vae=int(checkpoint.get("rank_vae", 4)),
        timestep=int(checkpoint.get("timestep", 199)),
        detail_enabled=True,
        detail_num_blocks=int(detail.get("num_naf_blocks", 1)),
        detail_gate_reduction=int(detail.get("gate_reduction", 4)),
        detail_alpha_init=float(
            checkpoint.get("experiment_metadata", {})
            .get("detail", {})
            .get("alpha_init", 0.1)
        ),
        detail_gate_use_prompt=False,
        detail_text_mode="full",
        detail_text_proj_dim=128,
        detail_film_hidden_ratio=4,
        detail_text_condition_scale=1.0,
    )


def _selected_batches(dataset, sample_count: int):
    selected = _selected_records(dataset.records, sample_count)
    by_id = {str(record["id"]): index for index, record in enumerate(dataset.records)}
    for record in selected:
        sample = dataset[by_id[str(record["id"])]]
        yield {
            key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
            for key, value in sample.items()
        }


@torch.inference_mode()
def _predictions(model, batches, device, precision: str, output_dir: Path | None):
    values = {}
    model.set_eval()
    for batch in batches:
        sample_id = str(batch["sample_id"][0])
        seed = sample_seed(42, sample_id)
        devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            with _autocast(device, precision):
                prediction = model(
                    batch["conditioning_pixel_values"].to(device),
                    prompt_tokens=batch["input_ids"].to(device),
                    prompt_attention_mask=batch["attention_mask"].to(device),
                )
        values[sample_id] = prediction.detach().float().cpu()
        if output_dir is not None:
            _save_image(
                prediction[0].float().add(1).mul(0.5).clamp(0, 1),
                output_dir / f"{sample_id.replace('/', '__')}.png",
            )
    return values


def run(args: argparse.Namespace) -> None:
    from .model import (
        load_model_checkpoint,
        load_model_from_checkpoint,
        save_training_checkpoint,
    )
    from .train import _optimizer_parameter_groups

    checkpoint_path = args.b20_checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("text_film_config") is not None:
        raise ValueError("E0 requires the legacy B-20k checkpoint without Text-FiLM")
    _, manifest, preparation = _training_run(checkpoint_path)
    baseline_model, baseline_step, baseline_metadata = load_model_from_checkpoint(
        checkpoint_path, expected_dataset="CCDD-11", expected_seed=42
    )
    if baseline_step != 20_000 or baseline_metadata.get("train_scope") != "detail":
        raise ValueError("E0 requires the detail-only B-20k checkpoint")
    dataset = SelectiveDifixDataset(
        manifest, baseline_model.tokenizer, resolution=512, training_mode="positive"
    )
    batches = list(_selected_batches(dataset, args.samples))
    reference = {}
    baseline_model = baseline_model.to(device)
    for precision in args.precisions:
        reference[precision] = _predictions(
            baseline_model,
            batches,
            device,
            precision,
            output_dir / "b20_outputs" if precision == "no" else None,
        )
    del baseline_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    text_model = _build_text_model(checkpoint)
    load_model_checkpoint(
        text_model,
        checkpoint_path,
        expected_dataset="CCDD-11",
        expected_seed=42,
    )
    text_model = text_model.to(device)
    equivalence = {}
    for precision in args.precisions:
        predictions = _predictions(text_model, batches, device, precision, None)
        differences = [
            float((prediction - reference[precision][sample_id]).abs().max())
            for sample_id, prediction in predictions.items()
        ]
        equivalence[precision] = max(differences)

    delta_abs_max = max(
        float(gate.last_delta_logits_abs_mean)
        for gate in text_model.vae.decoder.detail_text_gates
    )
    group_args = SimpleNamespace(
        detail_enabled=True,
        train_scope="film",
        learning_rate=5e-6,
        detail_learning_rate=1e-5,
        text_learning_rate=1e-4,
        detail_alpha_learning_rate=5e-6,
        adam_weight_decay=1e-2,
    )
    text_model.set_train("film")
    film_groups = _optimizer_parameter_groups(text_model, group_args)
    text_model.set_train("film+detail")
    group_args.train_scope = "film+detail"
    group_args.text_learning_rate = 5e-5
    hybrid_groups = _optimizer_parameter_groups(text_model, group_args)
    parameter_group_audit = {
        "film": _group_summary(film_groups),
        "film+detail": _group_summary(hybrid_groups),
    }
    text_model.set_train("film")
    optimizer = torch.optim.AdamW(film_groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    batch = batches[0]
    source = batch["conditioning_pixel_values"].to(device)
    target = batch["output_pixel_values"].to(device)
    seed = sample_seed(42, str(batch["sample_id"][0]))

    def backward_once():
        devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            with _autocast(device, args.gradient_precision):
                prediction = text_model(
                    source,
                    prompt_tokens=batch["input_ids"].to(device),
                    prompt_attention_mask=batch["attention_mask"].to(device),
                )
                loss = F.mse_loss(prediction.float(), target.float())
            loss.backward()

    backward_once()
    delta_out_grad = sum(
        int(torch.count_nonzero(gate.delta_out.weight.grad))
        for gate in text_model.vae.decoder.detail_text_gates
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    backward_once()
    upstream_grad = sum(
        int(torch.count_nonzero(parameter.grad))
        for parameter in text_model.vae.decoder.detail_text_projection.parameters()
        if parameter.grad is not None
    ) + sum(
        int(torch.count_nonzero(gate.to_gamma_beta.weight.grad))
        for gate in text_model.vae.decoder.detail_text_gates
    )
    text_parameter_ids = {
        id(parameter) for parameter in text_model.text_film_parameters()
    }
    frozen_gradients = sum(
        parameter.grad is not None
        for parameter in text_model.parameters()
        if id(parameter) not in text_parameter_ids
    )

    parameter_counts = text_model.parameter_counts()
    film_trainable_parameters = sum(
        parameter.numel() for parameter in text_model.text_film_parameters()
    )
    film_detail_trainable_parameters = sum(
        item["parameters"] for item in parameter_group_audit["film+detail"]
    )
    roundtrip_reference = _predictions(text_model, [batch], device, "no", None)
    captured_text_config = text_model.text_film_config()
    captured_text_state = {
        section: {key: value.detach().cpu().clone() for key, value in state.items()}
        for section, state in text_model.text_film_state_dict().items()
    }
    roundtrip_path = output_dir / "text_film_roundtrip.pkl"
    save_training_checkpoint(
        text_model,
        optimizer,
        scheduler,
        roundtrip_path,
        0,
        {
            "dataset": "CCDD-11",
            "seed": 42,
            "pairs": [1, 2, 3, 4, 5],
            "train_scope": "film",
            "parent_checkpoint": str(checkpoint_path),
            "parent_checkpoint_sha256": _file_sha256(checkpoint_path),
        },
    )
    del optimizer, scheduler, text_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    restored = _build_text_model(checkpoint)
    load_model_checkpoint(
        restored, roundtrip_path, expected_dataset="CCDD-11", expected_seed=42
    )
    roundtrip_equal = captured_text_config == restored.text_film_config() and all(
        torch.equal(value, restored.text_film_state_dict()[section][key])
        for section, state in captured_text_state.items()
        for key, value in state.items()
    )
    restored = restored.to(device)
    roundtrip_prediction = _predictions(restored, [batch], device, "no", None)
    roundtrip_output_max_abs_diff = max(
        float((prediction - roundtrip_reference[sample_id]).abs().max())
        for sample_id, prediction in roundtrip_prediction.items()
    )
    report = {
        "b20_checkpoint": str(checkpoint_path),
        "b20_sha256": _file_sha256(checkpoint_path),
        "b20_global_step": baseline_step,
        "b20_detail_config": checkpoint["detail_config"],
        "b20_metadata": baseline_metadata,
        "preparation": preparation,
        "samples": [str(batch["sample_id"][0]) for batch in batches],
        "step0_max_abs_diff": equivalence,
        "delta_logits_abs_max": delta_abs_max,
        "delta_out_nonzero_gradient_elements": delta_out_grad,
        "upstream_nonzero_gradient_elements_after_update": upstream_grad,
        "frozen_parameters_with_gradients": frozen_gradients,
        "parameter_counts": parameter_counts,
        "parameter_group_audit": parameter_group_audit,
        "film_trainable_parameters": film_trainable_parameters,
        "film_detail_trainable_parameters": film_detail_trainable_parameters,
        "checkpoint_roundtrip_equal": roundtrip_equal,
        "checkpoint_roundtrip_output_max_abs_diff": roundtrip_output_max_abs_diff,
    }
    _atomic_json(output_dir / "report.json", report)

    thresholds = {"no": 1e-6, "bf16": 2e-3, "fp16": 2e-3}
    failed_precision = [
        precision
        for precision, difference in equivalence.items()
        if difference >= thresholds[precision]
    ]
    if (
        failed_precision
        or delta_abs_max != 0
        or delta_out_grad == 0
        or upstream_grad == 0
        or frozen_gradients != 0
        or not roundtrip_equal
        or roundtrip_output_max_abs_diff >= 1e-6
    ):
        raise RuntimeError(f"Text-FiLM E0 failed; see {output_dir / 'report.json'}")
    print(f"Text-FiLM E0 passed: {output_dir / 'report.json'}", flush=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--b20-checkpoint", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--samples", type=int, default=4)
    value.add_argument("--device", default="cuda:0")
    value.add_argument(
        "--precisions",
        nargs="+",
        choices=("no", "fp16", "bf16"),
        default=("no", "bf16"),
    )
    value.add_argument(
        "--gradient-precision", choices=("no", "fp16", "bf16"), default="bf16"
    )
    return value


def main() -> None:
    run(parser().parse_args())
