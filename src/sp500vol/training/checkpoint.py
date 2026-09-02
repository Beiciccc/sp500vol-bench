"""Checkpoint save/load for resume-on-preemption.

Spot instances on Vast.ai / RunPod can be interrupted. Every training loop must
call `Checkpointer.save(...)` at end of epoch and `Checkpointer.maybe_load(...)`
at start so a re-launched job picks up where it stopped.

Layout under ``{run_dir}/checkpoints/``:

    epoch_0001.pt        # individual epoch snapshots (zero-padded for sort)
    epoch_0002.pt
    ...
    latest.pt            # symlink (or copy fallback) to the most recent epoch

Atomicity: every write goes to a sibling ``*.tmp`` file first and is then
``os.replace``'d into place, so a crash mid-save never produces a partial file.
"""

from __future__ import annotations

import contextlib
import os
import pickle
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_EPOCH_PREFIX = "epoch_"
_EPOCH_WIDTH = 4
_LATEST_NAME = "latest.pt"


@dataclass
class CheckpointState:
    """Everything a training run needs to deterministically resume."""

    epoch: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any] | None = None
    rng_state: dict[str, Any] = field(default_factory=dict)  # numpy + torch + python random
    best_val_metric: float = float("inf")
    config: dict[str, Any] = field(default_factory=dict)


class Checkpointer:
    """Atomic, resumable checkpoint manager with bounded retention."""

    def __init__(self, run_dir: Path, keep_last_k: int = 3) -> None:
        if keep_last_k < 1:
            raise ValueError("keep_last_k must be >= 1")
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_k = keep_last_k

    # === Public API ===

    def save(self, state: CheckpointState) -> Path:
        """Atomic save of `state` as epoch_{N:04d}.pt and update latest.pt.

        Returns:
            Path to the canonical epoch file (not the latest pointer).
        """
        epoch_path = self._epoch_path(state.epoch)
        self._atomic_dump(state, epoch_path)
        self._update_latest_pointer(epoch_path)
        self._prune_old_epochs()
        return epoch_path

    def maybe_load(self) -> CheckpointState | None:
        """Return the latest checkpoint if one exists, else None."""
        latest = self.ckpt_dir / _LATEST_NAME
        if latest.exists() or latest.is_symlink():
            target = latest.resolve() if latest.is_symlink() else latest
            return self._load(target)

        # Fallback: scan for the highest epoch file (e.g. after manual cleanup).
        epochs = self.list_epochs()
        if not epochs:
            return None
        return self._load(self._epoch_path(epochs[-1]))

    def list_epochs(self) -> list[int]:
        """All epoch numbers present on disk, ascending."""
        epochs = []
        for path in self.ckpt_dir.glob(f"{_EPOCH_PREFIX}*.pt"):
            try:
                epochs.append(int(path.stem[len(_EPOCH_PREFIX) :]))
            except ValueError:
                continue
        return sorted(epochs)

    # === Implementation ===

    def _epoch_path(self, epoch: int) -> Path:
        return self.ckpt_dir / f"{_EPOCH_PREFIX}{epoch:0{_EPOCH_WIDTH}d}.pt"

    def _atomic_dump(self, state: CheckpointState, dest: Path) -> None:
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        # Use pickle directly so the module works without a torch import.
        # When torch tensors are present in state.model_state, torch's pickling
        # plug-ins fire automatically.
        with tmp.open("wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)

    def _update_latest_pointer(self, target: Path) -> None:
        latest = self.ckpt_dir / _LATEST_NAME
        tmp_link = self.ckpt_dir / (_LATEST_NAME + ".tmp")
        if tmp_link.exists() or tmp_link.is_symlink():
            tmp_link.unlink()

        try:
            tmp_link.symlink_to(target.name)  # relative symlink within ckpt_dir
            os.replace(tmp_link, latest)
        except (OSError, NotImplementedError):
            # Windows or filesystem without symlink support — copy instead.
            if tmp_link.exists():
                tmp_link.unlink()
            shutil.copy2(target, tmp_link)
            os.replace(tmp_link, latest)

    def _prune_old_epochs(self) -> None:
        epochs = self.list_epochs()
        to_drop = epochs[: -self.keep_last_k] if len(epochs) > self.keep_last_k else []
        for epoch in to_drop:
            with contextlib.suppress(FileNotFoundError):
                self._epoch_path(epoch).unlink()

    def _load(self, path: Path) -> CheckpointState:
        with path.open("rb") as fh:
            state = pickle.load(fh)
        if not isinstance(state, CheckpointState):
            raise ValueError(f"checkpoint at {path} is not a CheckpointState")
        return state
