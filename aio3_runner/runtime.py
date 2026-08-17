"""Run metadata, RNG and atomic persistence helpers."""

from __future__ import annotations

import json
import os
import platform
import random
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


def atomic_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def append_jsonl(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_torch_save(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    # Mandatory checkpoint round-trip before validation.
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if int(loaded.get("global_step", -1)) != int(payload.get("global_step", -2)):
        raise RuntimeError(f"checkpoint round-trip failed: {path}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def rng_isolated(call: Callable[[], Any]) -> Any:
    state = capture_rng_state()
    try:
        return call()
    finally:
        restore_rng_state(state)


def git_state(repo: str | Path | None = None) -> dict[str, Any]:
    cwd = str(repo) if repo else None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=cwd, check=True, capture_output=True, text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip()), "status": status.splitlines()}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"commit": None, "dirty": True, "error": str(error)}


def environment_info(device: torch.device) -> dict[str, Any]:
    info = {
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "python": sys.version, "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "numpy": np.__version__,
    }
    if device.type == "cuda":
        info.update({
            "gpu": torch.cuda.get_device_name(device),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        })
    try:
        import PIL
        info["pillow"] = PIL.__version__
    except Exception:
        info["pillow"] = None
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except Exception:
        info["torchvision"] = None
    try:
        import wandb
        info["wandb"] = wandb.__version__
    except Exception:
        info["wandb"] = None
    return info


def path_is_within(path: str | Path, parent: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False
