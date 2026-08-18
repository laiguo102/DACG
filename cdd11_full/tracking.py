"""W&B monitoring with local JSONL as the durable source of truth."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .protocol import PROTOCOL_NAME
from .runtime import append_jsonl, atomic_json

WANDB_VERSION = "0.25.1"
WANDB_PROJECT = "cdd11-restoration"
WANDB_GROUP = PROTOCOL_NAME


class Tracker:
    def __init__(
        self,
        *,
        run_dir: Path,
        config: dict[str, Any],
        entity: str | None,
        mode: str,
        resume: bool,
    ) -> None:
        import wandb

        if wandb.__version__ != WANDB_VERSION:
            raise RuntimeError(f"CDD-11 requires wandb=={WANDB_VERSION}, found {wandb.__version__}")
        if mode == "online" and not entity:
            raise ValueError("Online W&B requires --wandb-entity")
        self.wandb = wandb
        self.run_dir = run_dir
        self.error_path = run_dir / "logs" / "wandb_errors.jsonl"
        self.metrics_path = run_dir / "train_metrics.jsonl"
        self.state_path = run_dir / "wandb_state.json"
        self.id_path = run_dir / "wandb_run_id.txt"
        if resume:
            if not self.id_path.is_file() or not self.state_path.is_file():
                raise RuntimeError("Resume requires wandb_run_id.txt and wandb_state.json")
            run_id = self.id_path.read_text(encoding="utf-8").strip()
            run_name = json.loads(self.state_path.read_text(encoding="utf-8"))["run_name"]
        else:
            run_id = uuid.uuid4().hex
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            run_name = f"dacg-full-seed{config['seed']}-{timestamp}"
        wandb_root = run_dir / "wandb"
        for path in (wandb_root, wandb_root / "cache", wandb_root / "artifacts", wandb_root / "staging"):
            path.mkdir(parents=True, exist_ok=True)
        os.environ.update({
            "WANDB_DIR": str(wandb_root), "WANDB_CACHE_DIR": str(wandb_root / "cache"),
            "WANDB_ARTIFACT_DIR": str(wandb_root / "artifacts"),
            "WANDB_DATA_DIR": str(wandb_root / "staging"),
        })
        self.run_id, self.run_name, self.mode = run_id, run_name, mode
        self.run = wandb.init(
            entity=entity, project=WANDB_PROJECT, group=WANDB_GROUP,
            name=run_name, id=run_id, resume="must" if resume else "never",
            mode=mode, dir=str(wandb_root), config=config,
            tags=[PROTOCOL_NAME, "full-official-train", "no-oof", config["model"].lower()],
        )
        self.run.define_metric("global_step")
        for pattern in ("train/*", "diagnostics/*", "system/*", "test/*"):
            self.run.define_metric(pattern, step_metric="global_step")
        self.id_path.write_text(run_id + "\n", encoding="utf-8")
        self._state("running")

    def _state(self, status: str) -> None:
        atomic_json(self.state_path, {
            "status": status, "run_id": self.run_id, "run_name": self.run_name,
            "project": WANDB_PROJECT, "group": WANDB_GROUP, "mode": self.mode,
            "url": getattr(self.run, "url", None),
        })

    def log(self, payload: dict[str, Any], step: int) -> None:
        record = {"global_step": step, **payload}
        append_jsonl(self.metrics_path, record)
        try:
            self.run.log(record, step=step)
        except Exception as error:
            append_jsonl(self.error_path, {
                "operation": "log", "global_step": step,
                "error_type": type(error).__name__, "error": str(error),
            })
            print(f"W&B warning: {error}; local JSONL continues", flush=True)

    def finish(self, status: str) -> None:
        self._state(status)
        try:
            self.run.finish(exit_code=0 if status == "completed" else 1)
        except Exception as error:
            append_jsonl(self.error_path, {"operation": "finish", "error": str(error)})
