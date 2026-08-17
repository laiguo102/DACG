"""Protocol-constrained W&B logging that cannot perturb training RNG."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Callable

from .runtime import append_jsonl, rng_isolated

TRAIN_METRICS = [
    "train/loss", "train/denoise_l1", "train/derain_l1", "train/dehaze_l1",
    "train/learning_rate", "train/grad_norm", "train/step_time_seconds",
    "train/images_per_second", "train/samples_denoise", "train/samples_derain",
    "train/samples_dehaze", "system/gpu_memory_allocated_gib", "system/gpu_memory_reserved_gib",
    "diagnostics/residual_mean", "diagnostics/residual_std", "diagnostics/residual_min",
    "diagnostics/residual_max", "diagnostics/residual_negative_fraction",
    "diagnostics/residual_positive_fraction", "diagnostics/residual_near_zero_fraction",
    "diagnostics/prediction_below_zero_fraction", "diagnostics/prediction_above_one_fraction",
]
VAL_METRICS = [
    f"val/denoise/sigma{sigma}/{metric}" for sigma in (15, 25, 50) for metric in ("psnr", "ssim")
] + [
    f"val/{group}/{metric}" for group in ("denoise/mean", "derain", "dehaze", "macro")
    for metric in ("psnr", "ssim")
] + [
    f"diagnostics/{task}/residual_negative_fraction" for task in ("denoise", "derain", "dehaze")
]
TEST_METRICS = [
    f"test/{group}/{metric}" for group in (
        "bsd68/sigma15", "bsd68/sigma25", "bsd68/sigma50", "bsd68/mean",
        "rain100l", "sots_outdoor", "macro",
    ) for metric in ("psnr", "ssim")
] + ["test/total_runtime_seconds"]


class WandbTracker:
    def __init__(
        self,
        *,
        run_dir: Path,
        config: dict[str, Any],
        run_id: str | None,
        resume: bool,
    ):
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("AIO3-v1 requires wandb==0.25.1") from error
        if wandb.__version__ != "0.25.1":
            raise RuntimeError(f"AIO3-v1 requires wandb==0.25.1, found {wandb.__version__}")
        self.wandb = wandb
        self.config = config
        self.run_dir = run_dir
        self.error_log = run_dir / "logs" / "wandb_errors.jsonl"
        self.run_id = run_id or uuid.uuid4().hex
        wandb_root = run_dir / "wandb"
        for child in (wandb_root, wandb_root / "cache", wandb_root / "artifacts", wandb_root / "staging"):
            child.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_DIR"] = str(wandb_root)
        os.environ["WANDB_CACHE_DIR"] = str(wandb_root / "cache")
        os.environ["WANDB_ARTIFACT_DIR"] = str(wandb_root / "artifacts")
        os.environ["WANDB_DATA_DIR"] = str(wandb_root / "staging")
        monitoring = config["monitoring"]
        tags = ["aio3-v1", config["model_id"], config["run_kind"], "signed-residual"]
        self.run = rng_isolated(lambda: wandb.init(
            entity=monitoring.get("entity"), project="aio3-restoration", group="aio3-v1",
            job_type="train", name=run_dir.name, id=self.run_id,
            resume="must" if resume else "never", dir=str(run_dir), config=config,
            tags=tags, mode=monitoring["mode"],
        ))
        self._define_metrics()

    def _call(self, operation: str, call: Callable[[], Any], *, fatal: bool = False) -> Any:
        try:
            return rng_isolated(call)
        except Exception as error:
            append_jsonl(self.error_log, {"operation": operation, "error_type": type(error).__name__, "error": str(error)})
            if fatal:
                raise
            print(f"W&B warning during {operation}: {error}", flush=True)
            return None

    def _define_metrics(self) -> None:
        self._call("define_metrics", lambda: self._define_metrics_inner(), fatal=True)

    def _define_metrics_inner(self) -> None:
        self.run.define_metric("global_step")
        for pattern in ("train/*", "diagnostics/*", "system/*", "val/*", "test/*", "best/*"):
            self.run.define_metric(pattern, step_metric="global_step")
        for name in TRAIN_METRICS + VAL_METRICS + TEST_METRICS:
            kwargs = {"step_metric": "global_step"}
            if name in ("val/macro/psnr", "val/macro/ssim"):
                kwargs["summary"] = "max"
            self.run.define_metric(name, **kwargs)
        self.run.define_metric("diagnostics/fixed_samples/residual_histogram", step_metric="global_step")

    def log(self, payload: dict[str, Any]) -> None:
        if "global_step" not in payload:
            raise ValueError("every W&B payload must contain global_step")
        self._call("log", lambda: self.run.log(payload))

    def histogram(self, values: Any) -> Any:
        return self._call("histogram", lambda: self.wandb.Histogram(values))

    def update_best(self, *, step: int, psnr: float, ssim: float) -> None:
        def update() -> None:
            self.run.summary["best/val_macro_psnr"] = psnr
            self.run.summary["best/val_macro_ssim"] = ssim
            self.run.summary["best/global_step"] = step
        self._call("update_summary", update)

    def log_dataset_artifact(self, files: list[Path], metadata: dict[str, Any]) -> None:
        def upload() -> None:
            artifact = self.wandb.Artifact("aio3-v1-manifests", type="dataset", metadata=metadata)
            for path in files:
                artifact.add_file(str(path), name=path.name)
            self.run.log_artifact(artifact)
        self._call("dataset_artifact", upload)

    def log_model_artifact(self, files: list[Path], seed: int) -> None:
        def upload() -> None:
            artifact = self.wandb.Artifact(
                f"aio3-v1-{self.config['model_id']}-seed{seed}", type="model",
                metadata={"protocol": "aio3-v1", "seed": seed, "model_id": self.config["model_id"]}
            )
            for path in files:
                if path.exists():
                    artifact.add_file(str(path), name=path.name)
            self.run.log_artifact(artifact, aliases=["best", f"seed{seed}"])
        self._call("model_artifact", upload)

    def log_evaluation_artifact(self, files: list[Path], gallery_dir: Path) -> None:
        def upload() -> None:
            artifact = self.wandb.Artifact(
                f"aio3-v1-{self.config['model_id']}-evaluation", type="evaluation",
                metadata={"protocol": "aio3-v1", "model_id": self.config["model_id"]}
            )
            for path in files:
                if path.exists():
                    artifact.add_file(str(path), name=path.name)
            if gallery_dir.exists():
                artifact.add_dir(str(gallery_dir), name="gallery")
            self.run.log_artifact(artifact)
        self._call("evaluation_artifact", upload)

    def finish(self, exit_code: int = 0) -> None:
        self._call("finish", lambda: self.run.finish(exit_code=exit_code))
