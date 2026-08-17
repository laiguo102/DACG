"""One-command CDD-11 5-fold OOF DACG training pipeline.

The script deliberately stops after generating validation coarse images. The
official test must only be run after the downstream Difix model is frozen.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from cdd11_oof.protocol import DEGRADATIONS, NUM_OOF_SCENES, NUM_VAL_SCENES


def run(module: str, *arguments: object) -> None:
    command = [sys.executable, "-m", module, *(str(item) for item in arguments)]
    print("\n>>> " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def manifest_complete(path: Path, expected: int) -> bool:
    if not path.is_file():
        return False
    try:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    keys = {(record.get("scene_id"), record.get("degradation")) for record in records}
    return (
        len(records) == expected
        and len(keys) == expected
        and all(Path(record.get("coarse", "")).is_file() for record in records)
    )


def train_role(args: argparse.Namespace, role: str) -> Path:
    run_dir = args.output_root / f"dacg_{role}"
    best = run_dir / "checkpoints" / "best_macro_psnr.ckpt"
    last = run_dir / "checkpoints" / "last.ckpt"
    if best.is_file():
        print(f"[skip] {role} training already complete: {best}", flush=True)
        return best
    command: list[object] = [
        "--data-root", args.data_root,
        "--split-dir", args.split_dir,
        "--output-dir", run_dir,
        "--role", role,
        "--model", args.model,
        "--epochs", args.epochs,
        "--batch-size", args.batch_size,
        "--accumulate-grad-batches", args.accumulate_grad_batches,
        "--patch-size", args.patch_size,
        "--lr", args.lr,
        "--num-workers", args.num_workers,
        "--num-gpus", args.num_gpus,
        "--val-every", args.val_every,
        "--precision", args.precision,
    ]
    if args.wandb_project:
        command += ["--wandb-project", args.wandb_project]
    if last.is_file():
        print(f"[resume] {role} from {last}", flush=True)
        command += ["--resume", last]
    run("cdd11_oof.train", *command)
    if not best.is_file():
        raise RuntimeError(f"Training finished without the expected best checkpoint: {best}")
    return best


def infer_target(args: argparse.Namespace, checkpoint: Path, target: str, output_dir: Path, expected: int) -> None:
    manifest = output_dir / "manifest.jsonl"
    if manifest_complete(manifest, expected):
        print(f"[skip] {target} inference already complete: {manifest}", flush=True)
        return
    command: list[object] = [
        "--checkpoint", checkpoint,
        "--data-root", args.data_root,
        "--split-dir", args.split_dir,
        "--target", target,
        "--output-dir", output_dir,
        "--num-workers", args.infer_workers,
        "--tile-size", args.tile_size,
        "--tile-overlap", args.tile_overlap,
    ]
    if output_dir.exists() and any(output_dir.iterdir()):
        command.append("--overwrite")
    run("cdd11_oof.infer", *command)
    if not manifest_complete(manifest, expected):
        raise RuntimeError(f"Incomplete {target} inference output: {manifest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train all five OOF DACG models, generate Difix OOF triples, and train DACG-final."
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Standard CDD11 root containing train/ and test/")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=None, help="Default: <output-root>/splits")
    parser.add_argument("--model", choices=["DACG_IR", "DACG_IR_S"], default="DACG_IR")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8, help="Per-GPU batch size")
    parser.add_argument(
        "--accumulate-grad-batches", type=int, default=1,
        help="Global effective batch = batch-size * num-gpus * this value",
    )
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--infer-workers", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--precision", choices=["32-true", "16-mixed", "bf16-mixed"], default="bf16-mixed")
    parser.add_argument("--tile-size", type=int, default=0, help="Use 512 if whole-image inference runs out of memory")
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--wandb-project")
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    args.split_dir = (args.split_dir or args.output_root / "splits").resolve()
    if args.batch_size < 1 or args.accumulate_grad_batches < 1 or args.num_gpus < 1:
        parser.error("batch size, gradient accumulation, and GPU count must all be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    frozen_config = {
        "data_root": str(args.data_root),
        "split_dir": str(args.split_dir),
        "model": args.model,
        "epochs": args.epochs,
        "batch_size_per_gpu": args.batch_size,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "global_effective_batch_size": args.batch_size * args.num_gpus * args.accumulate_grad_batches,
        "patch_size": args.patch_size,
        "learning_rate": args.lr,
        "num_gpus": args.num_gpus,
        "validation_interval_epochs": args.val_every,
        "precision": args.precision,
        "seed": 42,
    }
    config_path = args.output_root / "pipeline_config.json"
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != frozen_config:
            raise RuntimeError(
                f"Arguments differ from the frozen run config in {config_path}. "
                "Resume with the original arguments or choose a new --output-root."
            )
    else:
        config_path.write_text(
            json.dumps(frozen_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    required_split_files = [
        *(args.split_dir / f"fold{i}.txt" for i in range(1, 6)),
        args.split_dir / "val.txt",
        args.split_dir / "split_info.json",
    ]
    if not all(path.is_file() for path in required_split_files):
        if args.split_dir.exists() and any(args.split_dir.iterdir()):
            raise RuntimeError(
                f"Incomplete non-empty split directory {args.split_dir}; repair it instead of silently regenerating"
            )
        run("cdd11_oof.split", "--data-root", args.data_root, "--split-dir", args.split_dir)
    run("cdd11_oof.verify", "--data-root", args.data_root, "--split-dir", args.split_dir)

    per_fold = 213 * len(DEGRADATIONS)
    for fold in range(1, 6):
        role = f"fold{fold}"
        checkpoint = train_role(args, role)
        infer_target(args, checkpoint, role, args.output_root / "oof_outputs" / role, per_fold)

    difix_manifest = args.output_root / "difix_train_oof.jsonl"
    if manifest_complete(difix_manifest, NUM_OOF_SCENES * len(DEGRADATIONS)):
        print(f"[skip] merged Difix OOF manifest already complete: {difix_manifest}", flush=True)
    else:
        run(
            "cdd11_oof.merge_oof",
            "--oof-root", args.output_root / "oof_outputs",
            "--split-dir", args.split_dir,
            "--output", difix_manifest,
        )

    final_checkpoint = train_role(args, "final")
    infer_target(
        args,
        final_checkpoint,
        "val",
        args.output_root / "val_coarse",
        NUM_VAL_SCENES * len(DEGRADATIONS),
    )
    print("\nCDD-11 DACG OOF pipeline complete.", flush=True)
    print(f"Difix training manifest: {difix_manifest}", flush=True)
    print(f"Validation coarse manifest: {args.output_root / 'val_coarse' / 'manifest.jsonl'}", flush=True)
    print("Official Test was intentionally not run; freeze Difix first.", flush=True)


if __name__ == "__main__":
    main()
