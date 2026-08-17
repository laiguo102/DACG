"""Train one scene-disjoint DACG fold model or the final DACG model."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from .data import CDD11SceneDataset, validate_partition
from .model import DACGLitModel, MODEL_CONFIGS
from .protocol import DEGRADATIONS, NUM_TRAIN_SCENES, PROTOCOL_NAME, SEED, assert_frozen_split, load_splits, split_fingerprint
from .tracking import CDD11ArtifactCallback, CDD11WandbLogger, PerformanceCallback, WANDB_GROUP, WANDB_PROJECT


def training_ids(splits: dict[str, list[str]], role: str) -> list[str]:
    held_out = int(role.removeprefix("fold")) if role.startswith("fold") else None
    return [scene for i in range(1, 6) if i != held_out for scene in splits[f"fold{i}"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", choices=[*(f"fold{i}" for i in range(1, 6)), "final"], required=True)
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default="DACG_IR")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--precision", default="bf16-mixed", choices=["32-true", "16-mixed", "bf16-mixed"])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-media-every", type=int, default=40, help="Validation media interval in epochs")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.seed != SEED:
        parser.error(f"The frozen protocol requires --seed {SEED}")
    if args.batch_size < 1 or args.accumulate_grad_batches < 1 or args.wandb_media_every < 1:
        parser.error("batch size, accumulation, and W&B media interval must all be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume is None:
        raise FileExistsError(f"Refusing to mix a new run with existing files: {args.output_dir}")
    split_dir = args.split_dir or args.data_root / "splits"
    splits = load_splits(split_dir)
    train_scene_ids = validate_partition(args.data_root, "train", NUM_TRAIN_SCENES)
    assert_frozen_split(train_scene_ids, splits)
    train_ids = training_ids(splits, args.role)
    expected = 852 if args.role.startswith("fold") else 1065
    if len(train_ids) != expected:
        raise RuntimeError(f"{args.role} must train on {expected} scenes, got {len(train_ids)}")

    pl.seed_everything(SEED, workers=True)
    train_set = CDD11SceneDataset(args.data_root, "train", train_ids, patch_size=args.patch_size, augment=True)
    val_set = CDD11SceneDataset(args.data_root, "train", splits["val"])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=args.output_dir / "checkpoints", monitor="val_macro_psnr", mode="max",
        filename="best_macro_psnr", auto_insert_metric_name=False, save_last=True, save_top_k=1,
    )
    monitoring_config = {
        "protocol": PROTOCOL_NAME,
        "model": args.model,
        "role": args.role,
        "seed": SEED,
        "degradations": list(DEGRADATIONS),
        "split_fingerprint": split_fingerprint(splits),
        "train_scenes": len(train_ids),
        "validation_scenes": len(splits["val"]),
        "epochs": args.epochs,
        "batch_size_per_gpu": args.batch_size,
        "num_gpus": args.num_gpus,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "global_effective_batch_size": args.batch_size * args.num_gpus * args.accumulate_grad_batches,
        "patch_size": args.patch_size,
        "learning_rate": args.lr,
        "precision": args.precision,
        "num_workers": args.num_workers,
        "validation_interval_epochs": args.val_every,
        "wandb_media_interval_epochs": args.wandb_media_every,
        "monitoring": {"provider": "wandb", "project": WANDB_PROJECT, "group": WANDB_GROUP, "mode": args.wandb_mode, "entity": args.wandb_entity},
    }
    logger = CDD11WandbLogger(
        run_dir=args.output_dir, role=args.role, model_name=args.model,
        entity=args.wandb_entity, mode=args.wandb_mode,
        config=monitoring_config, resume=args.resume is not None,
    )
    model = DACGLitModel(
        model_name=args.model, lr=args.lr, epochs=args.epochs, role=args.role,
        split_dir=str(split_dir.resolve()), split_fingerprint=split_fingerprint(splits),
        wandb_media_every=args.wandb_media_every,
    )
    artifact_callback = CDD11ArtifactCallback(
        split_dir=split_dir, checkpoint_callback=checkpoint, role=args.role,
        model_name=args.model, split_fingerprint=split_fingerprint(splits),
    )
    performance_callback = PerformanceCallback(
        interval_steps=50,
        effective_batch_size=args.batch_size * args.num_gpus * args.accumulate_grad_batches,
    )
    trainer = pl.Trainer(
        accelerator="gpu", devices=args.num_gpus, strategy="auto" if args.num_gpus == 1 else "ddp",
        max_epochs=args.epochs, precision=args.precision, deterministic=True, logger=logger,
        callbacks=[checkpoint, artifact_callback, performance_callback], check_val_every_n_epoch=args.val_every,
        log_every_n_steps=50 * args.accumulate_grad_batches,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
    )
    effective_batch = args.batch_size * args.num_gpus * args.accumulate_grad_batches
    print(
        f"{args.role}: training {len(train_ids)} scenes / {len(train_set)} pairs; "
        f"validation is isolated ({len(splits['val'])} scenes); global effective batch={effective_batch}"
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=str(args.resume) if args.resume else None)
    print(f"Best checkpoint: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()
