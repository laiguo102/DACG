import math
from pathlib import Path

import torch

from aio3_runner.protocol import AIO3LRScheduler, BASE_LR, MIN_LR, learning_rate
from aio3_runner.runtime import atomic_torch_save, capture_rng_state, restore_rng_state, rng_isolated
from aio3_runner.validation import visual_sample_ids


def test_learning_rate_matches_frozen_boundaries():
    assert math.isclose(learning_rate(0, max_steps=200000, warmup_steps=2000), 1e-7, rel_tol=0, abs_tol=1e-20)
    assert math.isclose(
        learning_rate(1999, max_steps=200000, warmup_steps=2000), BASE_LR,
        rel_tol=0, abs_tol=1e-20,
    )
    assert math.isclose(
        learning_rate(199999, max_steps=200000, warmup_steps=2000), MIN_LR,
        rel_tol=0, abs_tol=1e-15,
    )


def test_scheduler_resume_selects_the_same_next_lr():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=BASE_LR)
    scheduler = AIO3LRScheduler(optimizer, max_steps=200000, warmup_steps=2000)
    for _ in range(2371):
        scheduler.step()
    saved = scheduler.state_dict()
    expected = optimizer.param_groups[0]["lr"]
    other_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))], lr=BASE_LR)
    restored = AIO3LRScheduler(other_optimizer, max_steps=200000, warmup_steps=2000)
    restored.load_state_dict(saved)
    assert other_optimizer.param_groups[0]["lr"] == expected


def test_rng_isolation_restores_torch_state():
    torch.manual_seed(123)
    before = capture_rng_state()
    rng_isolated(lambda: torch.rand(100))
    after = capture_rng_state()
    assert torch.equal(before["torch_cpu"], after["torch_cpu"])
    restore_rng_state(before)


def test_visual_samples_parser_ignores_protocol_metadata(tmp_path: Path):
    import json

    path = tmp_path / "visual_samples.json"
    path.write_text(json.dumps({
        "protocol": "aio3-v1",
        "samples": [{"id": f"sample-{index}"} for index in range(14)],
    }), encoding="utf-8")
    assert visual_sample_ids(path) == [f"sample-{index}" for index in range(14)]


def test_atomic_checkpoint_roundtrip(tmp_path: Path):
    path = tmp_path / "latest.pth"
    atomic_torch_save(path, {"global_step": 50, "value": torch.tensor([1.0, 2.0])})
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded["global_step"] == 50
    assert torch.equal(loaded["value"], torch.tensor([1.0, 2.0]))
