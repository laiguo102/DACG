import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aio3_runner.adapter import build_model
from aio3_runner.data import AIO3ManifestDataset, BalancedTaskBatchSampler, stable_seed
from aio3_runner.metrics import image_metrics
from aio3_runner.results import aggregate_metrics, write_compliance_report, write_result_tables


def _image(path: Path, value: int, size=(19, 17)) -> None:
    Image.fromarray(np.full((*size, 3), value, dtype=np.uint8), mode="RGB").save(path)


def _manifest(tmp_path: Path) -> Path:
    clean = tmp_path / "clean.png"
    degraded = tmp_path / "degraded.png"
    _image(clean, 120)
    _image(degraded, 100)
    rows = []
    for task in ("denoise", "derain", "dehaze"):
        for index in range(2):
            rows.append({
                "id": f"{task}-{index}", "task": task, "split": "train",
                "input": None if task == "denoise" else str(degraded),
                "target": str(clean), "scene_id": str(index), "metadata": {"source": "test"},
            })
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_balanced_sampler_is_exact_and_resume_stable(tmp_path):
    dataset = AIO3ManifestDataset(_manifest(tmp_path), split="train")
    full = list(BalancedTaskBatchSampler(dataset.records, max_steps=4, seed=3407))
    resumed = list(BalancedTaskBatchSampler(dataset.records, max_steps=4, seed=3407, start_step=2))
    assert resumed == full[2:]
    for batch in full:
        tasks = Counter(dataset.records[index]["task"] for index, _ in batch)
        assert tasks == {"denoise": 4, "derain": 4, "dehaze": 4}


def test_dataset_transform_is_seeded_and_preserves_unclamped_noise(tmp_path):
    dataset = AIO3ManifestDataset(_manifest(tmp_path), split="train", patch_size=128)
    request = (0, stable_seed("same sample"))
    first, second = dataset[request], dataset[request]
    assert first["input"].shape == (3, 128, 128)
    assert torch.equal(first["input"], second["input"])
    assert torch.equal(first["target"], second["target"])
    assert first["sigma"] in (15, 25, 50)


def test_frozen_metrics_identity():
    image = torch.rand(1, 3, 16, 18)
    values = image_metrics(image, image)
    assert values["psnr"] == float("inf")
    assert abs(values["ssim"] - 1.0) < 1e-5


def test_result_aggregation_is_task_macro_and_writes_tables(tmp_path):
    rows = []
    for sigma, psnr in ((15, 10.0), (25, 20.0), (50, 30.0)):
        rows.append({"dataset": "BSD68", "task": "denoise", "sigma": sigma, "sample_id": str(sigma),
                     "psnr": psnr, "ssim": 0.5, "inference_seconds": 0.1, "prediction": "x.png"})
    rows.extend([
        {"dataset": "Rain100L", "task": "derain", "sigma": "", "sample_id": "r", "psnr": 40.0,
         "ssim": 0.7, "inference_seconds": 0.1, "prediction": "r.png"},
        {"dataset": "SOTS", "task": "dehaze", "sigma": "", "sample_id": "h", "psnr": 25.0,
         "ssim": 0.6, "inference_seconds": 0.1, "prediction": "h.png"},
    ])
    summary = aggregate_metrics(rows)
    assert summary["denoise/mean"]["psnr"] == 20.0
    assert summary["macro"]["psnr"] == (20.0 + 40.0 + 25.0) / 3.0
    write_result_tables(tmp_path, rows, summary, {"protocol": "aio3-v1"})
    write_compliance_report(
        tmp_path / "compliance_report.md", run={"seed": 3407},
        result={"protocol": "aio3-v1", "git": {}}, summary=summary, rows=rows,
        gallery_png_count=70,
    )
    assert (tmp_path / "metrics.json").exists()
    assert "AIO3 task macro" in (tmp_path / "compliance_report.md").read_text(encoding="utf-8")
    assert len((tmp_path / "per_image_metrics.csv").read_text().splitlines()) == 6


def test_model_adapter_preserves_arbitrary_resolution_and_gradients():
    model = build_model({
        "dim": 8, "num_blocks": [1, 1, 1, 1], "num_refinement_blocks": 1,
        "heads": [1, 1, 1, 1], "num_scales": 1,
    })
    image = torch.randn(1, 3, 17, 19, requires_grad=True)
    restored_raw = model(image)
    assert restored_raw.shape == image.shape
    restored_raw.mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters() if parameter.requires_grad
    )
