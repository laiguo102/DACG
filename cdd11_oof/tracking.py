"""Protocol-style W&B logging for the CDD-11 OOF workflow."""

from __future__ import annotations

import json
import os
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback

from .protocol import PROTOCOL_NAME

WANDB_VERSION = "0.25.1"
WANDB_PROJECT = "cdd11-restoration"
WANDB_GROUP = PROTOCOL_NAME


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _json_safe(value: Any) -> Any:
    if hasattr(value, "numel") and value.numel() == 1:
        return value.detach().cpu().item()
    return value


class CDD11WandbLogger(WandbLogger):
    """W&B logger with fixed axes, local fallback records, and exact resume ID."""

    def __init__(self, *, run_dir: Path, role: str, model_name: str, entity: str | None, mode: str, config: dict[str, Any], resume: bool):
        import wandb

        if wandb.__version__ != WANDB_VERSION:
            raise RuntimeError(f"CDD-11 monitoring requires wandb=={WANDB_VERSION}, found {wandb.__version__}")
        if mode == "online" and not entity:
            raise ValueError("Online W&B monitoring requires --wandb-entity")
        self.run_dir = run_dir
        self.metrics_path = run_dir / "train_metrics.jsonl"
        self.validation_metrics_path = run_dir / "validation_metrics.jsonl"
        self.error_path = run_dir / "logs" / "wandb_errors.jsonl"
        state_path = run_dir / "wandb_state.json"
        id_path = run_dir / "wandb_run_id.txt"
        if resume:
            if not state_path.is_file() or not id_path.is_file():
                raise RuntimeError("Resume checkpoint exists but W&B state/run ID is missing")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            run_id = id_path.read_text(encoding="utf-8").strip()
            run_name = state["run_name"]
        else:
            run_id = uuid.uuid4().hex
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            run_name = f"{model_name.lower()}-{role}-seed42-{timestamp}"

        wandb_root = run_dir / "wandb"
        for directory in (wandb_root, wandb_root / "cache", wandb_root / "staging", wandb_root / "artifacts"):
            directory.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_DIR"] = str(wandb_root)
        os.environ["WANDB_CACHE_DIR"] = str(wandb_root / "cache")
        os.environ["WANDB_DATA_DIR"] = str(wandb_root / "staging")
        os.environ["WANDB_ARTIFACT_DIR"] = str(wandb_root / "artifacts")
        self._state_path = state_path
        self._run_id = run_id
        self._run_name = run_name
        self._mode = mode
        super().__init__(
            entity=entity,
            project=WANDB_PROJECT,
            group=WANDB_GROUP,
            name=run_name,
            id=run_id,
            resume="must" if resume else "never",
            save_dir=str(run_dir),
            offline=mode == "offline",
            tags=[PROTOCOL_NAME, role, model_name.lower(), "signed-residual"],
            log_model=False,
            config=config,
            force=mode == "online",
        )
        experiment = self.experiment  # Force login/connectivity before training starts.
        experiment.define_metric("global_step")
        for pattern in ("train/*", "diagnostics/*", "system/*", "val/*", "best/*"):
            experiment.define_metric(pattern, step_metric="global_step")
        for degradation in config["degradations"]:
            for metric in ("psnr", "ssim"):
                experiment.define_metric(f"val/{degradation}/{metric}", step_metric="global_step")
        experiment.define_metric("val/macro/psnr", step_metric="global_step", summary="max")
        experiment.define_metric("val/macro/ssim", step_metric="global_step", summary="max")
        experiment.define_metric("val/fixed_samples", step_metric="global_step")
        id_path.write_text(run_id + "\n", encoding="utf-8")
        self._write_state("running", getattr(experiment, "url", None))

    def _write_state(self, status: str, url: str | None = None) -> None:
        self._state_path.write_text(json.dumps({
            "status": status,
            "run_id": self._run_id,
            "run_name": self._run_name,
            "project": WANDB_PROJECT,
            "group": WANDB_GROUP,
            "mode": self._mode,
            "url": url,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        payload = dict(metrics)
        payload["global_step"] = int(step or 0)
        local_path = self.validation_metrics_path if any(key.startswith("val/") for key in payload) else self.metrics_path
        _append_jsonl(local_path, {key: _json_safe(value) for key, value in payload.items()})
        try:
            super().log_metrics(payload, step=step)
        except Exception as error:
            _append_jsonl(self.error_path, {
                "operation": "log_metrics", "global_step": int(step or 0),
                "error_type": type(error).__name__, "error": str(error),
            })
            print(f"W&B warning: {error}; local JSONL logging continues", flush=True)

    def update_best(self, *, step: int, psnr: float, ssim: float) -> None:
        try:
            self.experiment.summary["best/val_macro_psnr"] = psnr
            self.experiment.summary["best/val_macro_ssim"] = ssim
            self.experiment.summary["best/global_step"] = step
        except Exception as error:
            _append_jsonl(self.error_path, {"operation": "update_best", "error": str(error)})

    def log_table(self, *, key: str, table: Any, step: int) -> None:
        try:
            self.experiment.log({"global_step": step, key: table})
        except Exception as error:
            _append_jsonl(self.error_path, {"operation": "log_table", "global_step": step, "error": str(error)})
            print(f"W&B media warning: {error}; training continues", flush=True)

    def finalize(self, status: str) -> None:
        self._write_state("completed" if status == "success" else "failed", getattr(self.experiment, "url", None))
        try:
            super().finalize(status)
        except Exception as error:
            _append_jsonl(self.error_path, {"operation": "finish", "error": str(error)})


class CDD11ArtifactCallback(Callback):
    def __init__(self, *, split_dir: Path, checkpoint_callback: Any, role: str, model_name: str, split_fingerprint: str):
        self.split_dir = split_dir
        self.checkpoint_callback = checkpoint_callback
        self.role = role
        self.model_name = model_name
        self.split_fingerprint = split_fingerprint

    def on_fit_start(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return
        try:
            import wandb
            artifact = wandb.Artifact(
                f"{PROTOCOL_NAME}-splits", type="dataset",
                metadata={"protocol": PROTOCOL_NAME, "split_fingerprint": self.split_fingerprint},
            )
            for name in [*(f"fold{i}.txt" for i in range(1, 6)), "val.txt", "split_info.json"]:
                artifact.add_file(str(self.split_dir / name), name=name)
            trainer.logger.experiment.log_artifact(artifact)
        except Exception as error:
            _append_jsonl(trainer.logger.error_path, {"operation": "dataset_artifact", "error": str(error)})

    def on_train_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return
        try:
            import wandb
            best_path = self.checkpoint_callback.best_model_path
            artifact = wandb.Artifact(
                f"{PROTOCOL_NAME}-{self.model_name.lower()}-{self.role}", type="model",
                metadata={"protocol": PROTOCOL_NAME, "role": self.role, "split_fingerprint": self.split_fingerprint},
            )
            if best_path:
                artifact.add_file(best_path, name="best_macro_psnr.ckpt")
            artifact.add_file(str(self.split_dir / "split_info.json"), name="split_info.json")
            trainer.logger.experiment.log_artifact(artifact, aliases=["best", self.role])
        except Exception as error:
            _append_jsonl(trainer.logger.error_path, {"operation": "model_artifact", "error": str(error)})


class PerformanceCallback(Callback):
    def __init__(self, *, interval_steps: int, effective_batch_size: int):
        self.interval_steps = interval_steps
        self.effective_batch_size = effective_batch_size
        self._start_time = 0.0
        self._start_step = 0

    def on_train_start(self, trainer, pl_module) -> None:
        self._start_time = time.perf_counter()
        self._start_step = trainer.global_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = trainer.global_step
        if step <= self._start_step or step % self.interval_steps:
            return
        elapsed = max(time.perf_counter() - self._start_time, 1e-9)
        completed = step - self._start_step
        trainer.logger.log_metrics({
            "train/step_time_seconds": elapsed / completed,
            "train/images_per_second": self.effective_batch_size * completed / elapsed,
            "train/samples_per_optimizer_step": self.effective_batch_size,
        }, step=step)
        self._start_time = time.perf_counter()
        self._start_step = step

    def on_validation_end(self, trainer, pl_module) -> None:
        self._start_time = time.perf_counter()
        self._start_step = trainer.global_step
