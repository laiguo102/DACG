"""Backfill PSNR/SSIM and fixed five-panel media for saved checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


MODEL_PATTERN = re.compile(r"model_(\d+)\.pkl$")
MEDIA_KEY = "validation_full/degraded_coarse_target_final_gt"


def _checkpoint_step(path: Path) -> int:
    match = MODEL_PATTERN.fullmatch(path.name)
    if match:
        return int(match.group(1))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint["global_step"])


def discover_checkpoints(directory: str | Path) -> list[tuple[int, Path]]:
    """Return ascending unique training steps, preferring named milestones."""

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(root)
    by_step: dict[int, Path] = {}
    for path in sorted(root.glob("model_*.pkl")):
        match = MODEL_PATTERN.fullmatch(path.name)
        if match:
            by_step[int(match.group(1))] = path
    for name in ("final.pkl", "latest.pkl"):
        path = root / name
        if path.is_file():
            step = _checkpoint_step(path)
            by_step.setdefault(step, path)
    if not by_step:
        raise FileNotFoundError(f"no selective Difix checkpoints found in {root}")
    return sorted(by_step.items())


def _write_results(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    import lpips
    from accelerate import Accelerator
    from torch.utils.data import DataLoader, Subset
    from torchvision.utils import save_image

    from .data import SelectiveDifixDataset
    from .model import SelectiveDifix, load_model_checkpoint
    from .prepare import prepare_selective_manifests
    from .validation import stratified_indices, validate, wandb_images

    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "backfill")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    prepare_selective_manifests(
        data_root=args.data_root,
        coarse_root=args.coarse_root,
        output_dir=run_dir,
        pair_ids=args.degradation_pairs,
    )
    validation_manifest = run_dir / "prepared" / "manifests" / "validation.jsonl"

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    model = SelectiveDifix(lora_rank_vae=args.lora_rank_vae, timestep=args.timestep)
    perceptual_model = lpips.LPIPS(net="vgg").eval().requires_grad_(False)
    dataset = SelectiveDifixDataset(
        validation_manifest, model.tokenizer, resolution=args.resolution
    )
    visual_indices = stratified_indices(
        dataset.records, args.num_validation_visualizations
    )
    full_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    visual_loader = DataLoader(
        Subset(dataset, visual_indices), batch_size=1, shuffle=False,
        num_workers=args.workers,
    )
    model, full_loader, visual_loader = accelerator.prepare(
        model, full_loader, visual_loader
    )
    perceptual_model = perceptual_model.to(accelerator.device)

    wandb_run = None
    if args.report_to == "wandb" and accelerator.is_main_process:
        import wandb

        wandb_run = wandb.init(
            project=args.tracker_project_name,
            entity=args.wandb_entity,
            name=args.tracker_run_name,
            dir=str(output_dir),
            job_type="checkpoint-evaluation",
            config={
                "source_run": args.source_wandb_run,
                "run_dir": str(run_dir),
                "degradation_pairs": args.degradation_pairs,
                "primary_reference": "single-degradation target",
            },
        )
        wandb_run.define_metric("checkpoint_step")
        wandb_run.define_metric("validation_full/*", step_metric="checkpoint_step")

    rows: list[dict] = []
    for expected_step, checkpoint_path in discover_checkpoints(
        args.checkpoint_dir or (run_dir / "checkpoints")
    ):
        loaded_step = load_model_checkpoint(
            accelerator.unwrap_model(model), checkpoint_path
        )
        if loaded_step != expected_step:
            raise ValueError(
                f"checkpoint filename step {expected_step} != payload step {loaded_step}: "
                f"{checkpoint_path}"
            )
        metrics, _ = validate(
            model, full_loader, perceptual_model, accelerator
        )
        _, visualizations = validate(
            model,
            visual_loader,
            perceptual_model,
            accelerator,
            visualization_limit=args.num_validation_visualizations,
        )
        row = {
            "checkpoint_step": loaded_step,
            "checkpoint": str(checkpoint_path.resolve()),
            **metrics,
        }
        rows.append(row)
        if accelerator.is_main_process:
            for index, (image, _) in enumerate(visualizations):
                save_image(image, media_dir / f"step_{loaded_step:06d}_{index:02d}.png")
            if wandb_run is not None:
                payload = {
                    "checkpoint_step": loaded_step,
                    **{f"validation_full/{key}": value for key, value in metrics.items()},
                    MEDIA_KEY: wandb_images(visualizations),
                }
                wandb_run.log(payload)
            print(f"backfill step={loaded_step}: {metrics}", flush=True)

    if accelerator.is_main_process:
        _write_results(output_dir / "metrics.jsonl", rows)
        best = max(rows, key=lambda row: row["psnr"])
        (output_dir / "summary.json").write_text(
            json.dumps({"best": best, "checkpoints": len(rows)}, indent=2) + "\n",
            encoding="utf-8",
        )
        if wandb_run is not None:
            wandb_run.summary["best_checkpoint_step"] = best["checkpoint_step"]
            wandb_run.summary["best_validation_full_psnr"] = best["psnr"]
            wandb_run.summary["best_validation_full_ssim"] = best["ssim"]
            wandb_run.finish()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--coarse-root", type=Path, required=True)
    value.add_argument("--run-dir", type=Path, required=True)
    value.add_argument("--checkpoint-dir", type=Path)
    value.add_argument("--output-dir", type=Path)
    value.add_argument(
        "--degradation-pairs", nargs="+", type=int, choices=range(1, 6), required=True
    )
    value.add_argument("--resolution", type=int, default=512)
    value.add_argument("--lora-rank-vae", type=int, default=4)
    value.add_argument("--timestep", type=int, default=199)
    value.add_argument("--num-validation-visualizations", type=int, default=10)
    value.add_argument("--workers", type=int, default=4)
    value.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    value.add_argument("--report-to", choices=("wandb", "none"), default="wandb")
    value.add_argument("--tracker-project-name", default="difix-cdd11-selective")
    value.add_argument("--tracker-run-name", default="difix-selective-10k-backfill")
    value.add_argument("--wandb-entity")
    value.add_argument("--source-wandb-run")
    return value


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
