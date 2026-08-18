"""CDD-11-v1 W&B monitoring with durable local JSONL."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from .runtime import append_jsonl, atomic_json, preserve_rng_state

WANDB_VERSION = "0.25.1"


class Tracker:
    def __init__(self, run_dir: Path, config: dict, resume: bool):
        monitoring = config["monitoring"]
        self.run_dir, self.errors, self.resume = run_dir, 0, bool(resume)
        self.run, self.wandb = None, None
        self.run_name = str(config["run_name"])
        self.resolved_entity, self.url = None, None
        self.monitoring = dict(monitoring)
        self.error_path = run_dir / "logs" / "wandb_errors.jsonl"
        self.state_path = run_dir / "wandb_state.json"
        if monitoring["mode"] == "disabled":
            self._state(False)
            return
        try:
            import wandb
        except ImportError as error:
            self._state(False, f"ImportError: {error}")
            raise RuntimeError("W&B monitoring is enabled but wandb is not installed") from error
        if wandb.__version__ != WANDB_VERSION:
            self._state(False, f"version mismatch: {wandb.__version__}")
            raise RuntimeError(f"CDD-11-v1 requires wandb=={WANDB_VERSION}, got {wandb.__version__}")
        self.wandb = wandb
        root = run_dir / "wandb"
        for path in (root, root / "cache", root / "artifacts", root / "staging"):
            path.mkdir(parents=True, exist_ok=True)
        os.environ.update({"WANDB_DIR": str(root), "WANDB_CACHE_DIR": str(root / "cache"),
                           "WANDB_ARTIFACT_DIR": str(root / "artifacts"),
                           "WANDB_DATA_DIR": str(root / "staging")})
        try:
            with preserve_rng_state():
                self.run = wandb.init(
                    entity=monitoring["entity"], project=monitoring["project"], group=monitoring["group"],
                    job_type="train", name=config["run_name"], id=monitoring["wandb_run_id"],
                    resume="must" if resume else "never", mode=monitoring["mode"], dir=str(root),
                    config=config, tags=monitoring["tags"], force=monitoring["mode"] == "online",
                )
        except Exception as error:
            self._state(False, f"{type(error).__name__}: {error}")
            raise RuntimeError("Could not initialize W&B; verify login/network or use offline mode") from error
        self.resolved_entity = getattr(self.run, "entity", None)
        self.url = getattr(self.run, "url", None)
        self._state(True)
        self._safe("define_metrics", self._define_metrics)
        if not resume:
            self._safe("manifest_artifact", lambda: self._log_artifact(config))

    def _define_metrics(self):
        self.run.define_metric("global_step")
        for pattern in ("train/*", "val/*", "system/*", "diagnostics/*", "test/*"):
            self.run.define_metric(pattern, step_metric="global_step")

    def _log_artifact(self, config):
        artifact = self.wandb.Artifact(
            name="cdd11-v1-frozen-manifests", type="dataset",
            metadata={"protocol": config["protocol"],
                      "manifest_sha256": config["data"]["manifest_sha256"]})
        manifest_dir = Path(config["paths"]["manifest_dir"])
        for filename in ("train.jsonl", "val.jsonl", "test.jsonl", "data_audit.json", "visual_samples.json"):
            artifact.add_file(str(manifest_dir / filename), name=filename)
        self.run.log_artifact(artifact)

    def _state(self, active: bool, initialization_error: str | None = None):
        monitoring = self.monitoring
        value = {
            "active": active, "errors": self.errors, "provider": "wandb",
            "project": monitoring.get("project"), "group": monitoring.get("group"),
            "mode": monitoring.get("mode"), "requested_entity": monitoring.get("entity"),
            "resolved_entity": self.resolved_entity, "resume": self.resume,
            "run_id": monitoring.get("wandb_run_id"),
            "run_name": self.run_name, "url": self.url,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if initialization_error:
            value["initialization_error"] = initialization_error
        atomic_json(self.state_path, value)

    def _safe(self, operation, function):
        if self.run is None:
            return
        try:
            with preserve_rng_state():
                function()
        except Exception as error:
            self.errors += 1
            append_jsonl(self.error_path, {"operation": operation, "error": str(error)})
            self._state(True)
            print(f"W&B warning during {operation}: {error}; local artifacts remain authoritative", flush=True)

    def log(self, payload: dict, step: int):
        self._safe("log_scalars", lambda: self.run.log({"global_step": step, **payload}, step=step))

    def update_best(self, best: dict):
        def operation():
            self.run.summary["best/macro_psnr"] = best["macro_psnr"]
            self.run.summary["best/macro_ssim"] = best["macro_ssim"]
            self.run.summary["best/global_step"] = best["global_step"]
        self._safe("update_best", operation)

    def log_validation(self, summary: dict, visuals: list[dict], step: int):
        def operation():
            payload = {"global_step": step, **{f"val/{key}": value for key, value in summary.items()}}
            if visuals:
                columns = ["global_step", "degradation", "arity", "sample_id", "input",
                           "prediction", "target", "absolute_error", "signed_residual", "psnr", "ssim"]
                table = self.wandb.Table(columns=columns)
                for visual in visuals:
                    table.add_data(step, visual["degradation"], visual["arity"], visual["sample_id"],
                                   self.wandb.Image(visual["input_path"]),
                                   self.wandb.Image(visual["prediction_path"]),
                                   self.wandb.Image(visual["target_path"]),
                                   self.wandb.Image(visual["absolute_error_path"]),
                                   self.wandb.Image(visual["signed_residual_path"]),
                                   visual["psnr"], visual["ssim"])
                payload["val/fixed_samples"] = table
            self.run.log(payload, step=step)
        self._safe("log_validation", operation)

    def finish(self):
        if self.run is not None:
            self._safe("finish", self.run.finish)
            self.run = None
        self._state(False)
