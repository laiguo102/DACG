"""Generate fold OOF coarse images or final-model validation/test coarse images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import CDD11SceneDataset, validate_partition
from .model import load_network
from .protocol import DEGRADATIONS, NUM_TEST_SCENES, NUM_TRAIN_SCENES, assert_frozen_split, load_splits, split_fingerprint


def tiled_forward(net, image: torch.Tensor, tile_size: int, overlap: int) -> torch.Tensor:
    if tile_size <= 0 or (image.shape[-2] <= tile_size and image.shape[-1] <= tile_size):
        prediction = net(image)
        return prediction[..., :image.shape[-2], :image.shape[-1]]
    if overlap >= tile_size:
        raise ValueError("--tile-overlap must be smaller than --tile-size")
    stride = tile_size - overlap
    _, _, height, width = image.shape
    tops = list(range(0, max(height - tile_size, 0), stride)) + [max(height - tile_size, 0)]
    lefts = list(range(0, max(width - tile_size, 0), stride)) + [max(width - tile_size, 0)]
    output, weight = torch.zeros_like(image), torch.zeros_like(image)
    for top in dict.fromkeys(tops):
        for left in dict.fromkeys(lefts):
            patch = image[..., top:min(top + tile_size, height), left:min(left + tile_size, width)]
            prediction = net(patch)[..., :patch.shape[-2], :patch.shape[-1]]
            output[..., top:top + prediction.shape[-2], left:left + prediction.shape[-1]] += prediction
            weight[..., top:top + prediction.shape[-2], left:left + prediction.shape[-1]] += 1
    return output / weight.clamp_min(1)


def save_tensor(tensor: torch.Tensor, path: Path) -> None:
    import numpy as np
    array = tensor.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--target", choices=[*(f"fold{i}" for i in range(1, 6)), "val", "test"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=0, help="0 uses whole-image inference")
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    split_dir = args.split_dir or args.data_root / "splits"
    splits = load_splits(split_dir)
    assert_frozen_split(validate_partition(args.data_root, "train", NUM_TRAIN_SCENES), splits)
    if args.target.startswith("fold"):
        partition, scene_ids, expected_role = "train", splits[args.target], args.target
    elif args.target == "val":
        partition, scene_ids, expected_role = "train", splits["val"], "final"
    else:
        partition = "test"
        scene_ids = validate_partition(args.data_root, "test", NUM_TEST_SCENES)
        expected_role = "final"
    if args.output_dir.exists() and any(args.output_dir.rglob("*")) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")

    device = torch.device("cuda")
    net, hparams = load_network(str(args.checkpoint), device)
    checkpoint_role = hparams.get("role")
    if checkpoint_role != expected_role:
        raise ValueError(f"Target {args.target} requires a {expected_role!r} checkpoint, got {checkpoint_role!r}")
    if hparams.get("split_fingerprint") != split_fingerprint(splits):
        raise ValueError("Checkpoint was trained with a different CDD-11 split manifest")
    dataset = CDD11SceneDataset(args.data_root, partition, scene_ids)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest, torch.inference_mode():
        for batch in loader:
            prediction = tiled_forward(net, batch["lq"].to(device, non_blocking=True), args.tile_size, args.tile_overlap)[0]
            scene_id, degradation = batch["scene_id"][0], batch["degradation"][0]
            output = args.output_dir / degradation / f"{scene_id}.png"
            save_tensor(prediction, output)
            gt = args.data_root / partition / "clear" / Path(dataset.clear[scene_id]).name
            record = {"scene_id": scene_id, "degradation": degradation, "degraded": batch["lq_path"][0], "coarse": str(output.resolve()), "gt": str(gt.resolve()), "source_checkpoint": str(args.checkpoint.resolve())}
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    expected = len(scene_ids) * len(DEGRADATIONS)
    print(f"Generated {expected} leakage-safe coarse images in {args.output_dir}")


if __name__ == "__main__":
    main()
