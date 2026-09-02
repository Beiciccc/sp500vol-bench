"""Checkpoint manager correctness tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sp500vol.training.checkpoint import Checkpointer, CheckpointState


def _state(epoch: int) -> CheckpointState:
    return CheckpointState(
        epoch=epoch,
        model_state={"layer.weight": np.array([epoch, epoch + 1], dtype=float)},
        optimizer_state={"lr": 0.001, "step": epoch * 100},
        scheduler_state=None,
        rng_state={"numpy": np.random.default_rng(epoch).bit_generator.state},
        best_val_metric=float(epoch),
        config={"model": "C1_bert_s1"},
    )


def test_save_creates_epoch_and_latest(tmp_path: Path) -> None:
    ck = Checkpointer(tmp_path / "run", keep_last_k=3)
    saved_path = ck.save(_state(1))

    assert saved_path.exists()
    assert saved_path.name == "epoch_0001.pt"
    latest = ck.ckpt_dir / "latest.pt"
    assert latest.exists() or latest.is_symlink()


def test_maybe_load_returns_latest(tmp_path: Path) -> None:
    ck = Checkpointer(tmp_path / "run")
    assert ck.maybe_load() is None
    ck.save(_state(1))
    ck.save(_state(2))
    ck.save(_state(3))

    loaded = ck.maybe_load()
    assert loaded is not None
    assert loaded.epoch == 3
    assert loaded.best_val_metric == 3.0
    np.testing.assert_array_equal(loaded.model_state["layer.weight"], np.array([3.0, 4.0]))


def test_prune_keeps_last_k(tmp_path: Path) -> None:
    ck = Checkpointer(tmp_path / "run", keep_last_k=2)
    for epoch in range(1, 6):
        ck.save(_state(epoch))

    epochs = ck.list_epochs()
    assert epochs == [4, 5]


def test_save_load_roundtrip_preserves_state(tmp_path: Path) -> None:
    ck = Checkpointer(tmp_path / "run")
    original = _state(7)
    ck.save(original)

    loaded = ck.maybe_load()
    assert loaded is not None
    assert loaded.epoch == original.epoch
    assert loaded.optimizer_state == original.optimizer_state
    assert loaded.config == original.config
    np.testing.assert_array_equal(
        loaded.model_state["layer.weight"], original.model_state["layer.weight"]
    )


def test_maybe_load_falls_back_to_highest_epoch_without_latest(tmp_path: Path) -> None:
    ck = Checkpointer(tmp_path / "run")
    ck.save(_state(1))
    ck.save(_state(2))

    # Simulate a corrupted latest.pt pointer being removed.
    latest = ck.ckpt_dir / "latest.pt"
    if latest.exists() or latest.is_symlink():
        latest.unlink()

    loaded = ck.maybe_load()
    assert loaded is not None
    assert loaded.epoch == 2


def test_keep_last_k_rejects_zero(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="keep_last_k"):
        Checkpointer(tmp_path / "run", keep_last_k=0)


def test_atomic_save_does_not_leave_tmp_files(tmp_path: Path) -> None:
    ck = Checkpointer(tmp_path / "run")
    ck.save(_state(1))
    ck.save(_state(2))

    tmp_files = list(ck.ckpt_dir.glob("*.tmp"))
    assert tmp_files == []
