"""Frozen AIO3-v1 configuration and exact optimizer-step scheduling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

PROTOCOL = "aio3-v1"
SEED = 3407
PATCH_SIZE = 128
EFFECTIVE_BATCH_SIZE = 12
TASK_SAMPLES = {"denoise": 4, "derain": 4, "dehaze": 4}
BASE_LR = 2.0e-4
MIN_LR = 1.0e-6
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 1.0e-4
GRAD_CLIP_NORM = 1.0


@dataclass(frozen=True)
class RunKind:
    max_steps: int
    warmup_steps: int
    scalar_interval_steps: int
    validation_interval_steps: int
    media_interval_steps: int


RUN_KINDS = {
    "smoke": RunKind(100, 100, 10, 100, 100),
    "pilot": RunKind(5000, 2000, 50, 5000, 5000),
    "formal": RunKind(200000, 2000, 50, 5000, 10000),
}


def learning_rate(update_index: int, *, max_steps: int, warmup_steps: int) -> float:
    """LR used by zero-based optimizer update ``update_index``.

    The first update is 1e-7 for formal/pilot, update 2000 is 2e-4, and
    update 200000 is 1e-6 (one-based descriptions in the protocol).
    """

    if not 0 <= update_index < max_steps:
        raise ValueError(f"update_index must be in [0, {max_steps}), got {update_index}")
    if update_index < warmup_steps:
        return BASE_LR * (update_index + 1) / warmup_steps
    cosine_updates = max_steps - warmup_steps
    completed_after_warmup = update_index - warmup_steps + 1
    progress = completed_after_warmup / cosine_updates
    return MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1.0 + math.cos(math.pi * progress))


class AIO3LRScheduler:
    """Small stateful scheduler whose state is the completed optimizer steps."""

    def __init__(self, optimizer: Any, *, max_steps: int, warmup_steps: int, completed_steps: int = 0):
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.completed_steps = completed_steps
        self.set_next_lr()

    def set_next_lr(self) -> float:
        if self.completed_steps >= self.max_steps:
            value = MIN_LR
        else:
            value = learning_rate(
                self.completed_steps, max_steps=self.max_steps, warmup_steps=self.warmup_steps
            )
        for group in self.optimizer.param_groups:
            group["lr"] = value
        return value

    def step(self) -> None:
        self.completed_steps += 1
        self.set_next_lr()

    def state_dict(self) -> dict[str, int]:
        return {"completed_steps": self.completed_steps}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.completed_steps = int(state["completed_steps"])
        self.set_next_lr()


def resolved_config(
    *,
    model_id: str,
    run_kind: str,
    seed: int,
    num_workers: int,
    micro_batch_size: int,
    model_config: dict[str, Any],
    wandb_mode: str,
    wandb_entity: str | None,
) -> dict[str, Any]:
    kind = RUN_KINDS[run_kind]
    return {
        "protocol": PROTOCOL,
        "model_id": model_id,
        "model_config": model_config,
        "seed": seed,
        "run_kind": run_kind,
        "data": {
            "patch_size": PATCH_SIZE,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": EFFECTIVE_BATCH_SIZE // micro_batch_size,
            "task_samples_per_optimizer_step": TASK_SAMPLES,
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": num_workers > 0,
        },
        "training": {
            "max_steps": kind.max_steps,
            "precision": "bf16",
            "grad_clip_norm": GRAD_CLIP_NORM,
            "loss": "mean_pixel_l1",
        },
        "optimizer": {
            "name": "adamw", "learning_rate": BASE_LR,
            "betas": list(BETAS), "weight_decay": WEIGHT_DECAY,
        },
        "scheduler": {
            "name": "warmup_cosine", "warmup_steps": kind.warmup_steps,
            "min_learning_rate": MIN_LR,
        },
        "validation": {
            "interval_steps": kind.validation_interval_steps,
            "batch_size": 1, "native_resolution": True,
        },
        "checkpoint": {"interval_steps": kind.validation_interval_steps, "milestone_interval_steps": 50000},
        "monitoring": {
            "provider": "wandb", "mode": wandb_mode, "entity": wandb_entity,
            "project": "aio3-restoration", "group": PROTOCOL,
            "scalar_interval_steps": kind.scalar_interval_steps,
            "media_interval_steps": kind.media_interval_steps,
        },
        "forbidden_features": {"pretrained": False, "ema": False, "tta": False, "tiled_inference": False},
    }
