"""Shared training logging and per-horizon checkpoint helpers for neural models."""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import torch

from sp500vol.utils import get_logger


def log_fit_horizon_start(model: Any, *, horizon: Any, n_train: int) -> None:
    get_logger("train").info(
        "fit horizon start",
        model=getattr(model, "name", type(model).__name__),
        horizon=_normalise(horizon),
        n_train=int(n_train),
    )


def log_fit_horizon_skip(model: Any, *, horizon: Any, n_train: int, checkpoint_path: Path) -> None:
    get_logger("train").info(
        "fit horizon checkpoint skip",
        model=getattr(model, "name", type(model).__name__),
        horizon=_normalise(horizon),
        n_train=int(n_train),
        checkpoint=str(checkpoint_path),
    )


def log_epoch_done(
    model: Any,
    *,
    horizon: Any,
    epoch: int,
    max_epochs: int,
    running_loss: float,
    n_batches: int,
    start_time: float,
) -> None:
    mean_loss = running_loss / max(1, n_batches)
    get_logger("train").info(
        "epoch done",
        model=getattr(model, "name", type(model).__name__),
        horizon=_normalise(horizon),
        epoch=int(epoch),
        max=int(max_epochs),
        mean_loss=float(mean_loss),
        secs=round(monotonic() - start_time, 3),
    )


def log_epoch_val(
    model: Any,
    *,
    horizon: Any,
    epoch: int,
    val_loss: float,
    val_r2: float,
    best_epoch: int,
    num_bad: int,
) -> None:
    get_logger("train").info(
        "epoch val",
        model=getattr(model, "name", type(model).__name__),
        horizon=_normalise(horizon),
        epoch=int(epoch),
        val_loss=_finite_or_none(val_loss),
        val_r2=_finite_or_none(val_r2),
        best_epoch=int(best_epoch),
        num_bad=int(num_bad),
    )


def log_fit_horizon_done(model: Any, *, horizon: Any, n_train: int, secs: float) -> None:
    get_logger("train").info(
        "fit horizon done",
        model=getattr(model, "name", type(model).__name__),
        horizon=_normalise(horizon),
        n_train=int(n_train),
        secs=round(float(secs), 3),
    )


def log_predict_horizon_start(model: Any, *, horizon: Any, n_rows: int) -> None:
    get_logger("train").info(
        "predict horizon start",
        model=getattr(model, "name", type(model).__name__),
        horizon=_normalise(horizon),
        n_rows=int(n_rows),
    )


def log_predict_horizon_done(model: Any, *, horizon: Any, n_rows: int, secs: float) -> None:
    get_logger("train").info(
        "predict horizon done",
        model=getattr(model, "name", type(model).__name__),
        horizon=_normalise(horizon),
        n_rows=int(n_rows),
        secs=round(float(secs), 3),
    )


def maybe_load_horizon_checkpoint(
    model: Any,
    *,
    horizon: Any,
    n_train: int,
) -> dict[str, Any] | None:
    path = _checkpoint_path(model, horizon)
    if path is None or not path.exists():
        return None

    expected_meta = checkpoint_meta(model, horizon=horizon, n_train=n_train)
    try:
        payload = _torch_load(path)
    except Exception as exc:  # pragma: no cover - defensive against partial/manual files
        get_logger("train").warning(
            "checkpoint load failed; retraining horizon",
            model=getattr(model, "name", type(model).__name__),
            horizon=_normalise(horizon),
            checkpoint=str(path),
            error=str(exc),
        )
        return None
    if not isinstance(payload, dict) or "meta" not in payload or "state" not in payload:
        get_logger("train").warning(
            "checkpoint payload invalid; retraining horizon",
            model=getattr(model, "name", type(model).__name__),
            horizon=_normalise(horizon),
            checkpoint=str(path),
        )
        return None

    actual_meta = payload["meta"]
    if actual_meta != expected_meta:
        get_logger("train").info(
            "checkpoint fingerprint mismatch; retraining horizon",
            model=getattr(model, "name", type(model).__name__),
            horizon=_normalise(horizon),
            checkpoint=str(path),
        )
        return None
    state = payload["state"]
    if not isinstance(state, dict):
        get_logger("train").warning(
            "checkpoint state invalid; retraining horizon",
            model=getattr(model, "name", type(model).__name__),
            horizon=_normalise(horizon),
            checkpoint=str(path),
        )
        return None
    return state


def save_horizon_checkpoint(
    model: Any,
    *,
    horizon: Any,
    n_train: int,
    state: dict[str, Any],
) -> Path | None:
    path = _checkpoint_path(model, horizon)
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": checkpoint_meta(model, horizon=horizon, n_train=n_train),
        "state": state,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    get_logger("train").info(
        "checkpoint saved",
        model=getattr(model, "name", type(model).__name__),
        horizon=_normalise(horizon),
        checkpoint=str(path),
    )
    return path


def horizon_checkpoint_path(model: Any, horizon: Any) -> Path | None:
    return _checkpoint_path(model, horizon)


def checkpoint_meta(model: Any, *, horizon: Any, n_train: int) -> dict[str, Any]:
    encoder_cfg = getattr(model, "encoder_cfg", None)
    return {
        "model_id": getattr(model, "name", type(model).__name__),
        "pretrained": getattr(encoder_cfg, "pretrained", None),
        "max_length": getattr(encoder_cfg, "max_length", None),
        "seed": _normalise(getattr(model, "seed", None)),
        "horizon": _normalise(horizon),
        "n_train": int(n_train),
        "strategy": getattr(model, "strategy", None),
        "config": _checkpoint_config(model),
    }


def epoch_start() -> float:
    return monotonic()


def now() -> float:
    return monotonic()


def _resolve_amp(model: Any) -> tuple[str | None, Any]:
    """Return (device_type, dtype) for autocast, or (None, None) when disabled.

    Mixed precision is enabled only when the model requests bf16 AND runs on CUDA
    (bf16 autocast on CPU is slow and not the target), so local CPU/MPS runs and
    fp32 models transparently fall back to full precision."""
    mp = getattr(model, "mixed_precision", "no")
    device = getattr(model, "device", None)
    device_type = device.type if device is not None else None
    if mp == "bf16" and device_type == "cuda":
        return device_type, torch.bfloat16
    return None, None


def make_scheduler(
    optimiser,
    *,
    steps_per_epoch: int,
    max_epochs: int,
    grad_accumulation_steps: int,
    warmup_ratio: float,
):
    """Linear warmup -> linear decay LR scheduler; None when warmup_ratio <= 0.

    Sized to the total optimiser-step budget over ``max_epochs`` (early stopping
    may end sooner; warmup still completes in the first ``warmup_ratio`` fraction).
    Required for the large-batch / high-lr regime to train stably."""
    if warmup_ratio <= 0:
        return None
    opt_steps_per_epoch = max(1, math.ceil(steps_per_epoch / max(1, grad_accumulation_steps)))
    total = max(1, opt_steps_per_epoch * max(1, max_epochs))
    warmup = max(1, int(warmup_ratio * total))
    from transformers import get_linear_schedule_with_warmup

    return get_linear_schedule_with_warmup(
        optimiser, num_warmup_steps=warmup, num_training_steps=total
    )


def _train_one_epoch(
    *,
    loader,
    optimiser,
    loss_fn,
    forward_batch,
    modules,
    grad_accumulation_steps,
    scheduler=None,
    amp_device_type=None,
    amp_dtype=None,
) -> tuple[float, int]:
    """Run one train-mode epoch; returns (running_loss, n_batches).

    When ``amp_dtype`` is set, the forward + loss run under ``torch.autocast``
    (mixed precision); weights/grads stay fp32 and backward runs outside autocast
    (bf16 needs no GradScaler)."""
    use_amp = amp_dtype is not None
    for module in modules:
        module.train()
    optimiser.zero_grad(set_to_none=True)
    # Accumulate the loss as a detached ON-DEVICE scalar and read it once at epoch end, so the
    # hot loop has no per-step device->host sync (the old per-step isfinite + loss.item() forced
    # two syncs every micro-step). Weights are bit-identical: the loss VALUE never feeds the
    # update (the gradient comes from loss.backward()); only when its scalar is read to the host
    # changes. Non-finiteness still propagates through the sum (inf+(-inf)=nan, nan+x=nan), so a
    # diverging run is still caught and aborted at epoch end — and a diverged run is discarded by
    # the matrix either way, so no surviving run's result changes.
    running_loss_t: torch.Tensor | None = None
    n_batches = 0
    n_steps = len(loader)
    for step, batch in enumerate(loader, start=1):
        if use_amp:
            with torch.autocast(device_type=amp_device_type, dtype=amp_dtype):
                pred, target = forward_batch(batch)
                loss = loss_fn(pred, target)
        else:
            pred, target = forward_batch(batch)
            loss = loss_fn(pred, target)
        detached = loss.detach()
        running_loss_t = detached if running_loss_t is None else running_loss_t + detached
        n_batches += 1
        (loss / grad_accumulation_steps).backward()
        if step % grad_accumulation_steps == 0 or step == n_steps:
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
    running_loss = float(running_loss_t.item()) if running_loss_t is not None else 0.0
    if n_batches and not math.isfinite(running_loss):
        raise RuntimeError("non-finite training loss")
    return running_loss, n_batches


def _evaluate_validation(
    val_eval, modules, *, amp_device_type=None, amp_dtype=None
) -> tuple[float, float]:
    """Eval-mode validation pass (or NaNs when no validation closure)."""
    if val_eval is None:
        return math.nan, math.nan
    for module in modules:
        module.eval()
    if amp_dtype is not None:
        with torch.autocast(device_type=amp_device_type, dtype=amp_dtype):
            return val_eval()
    return val_eval()


def _select_state(
    *, select_best, snapshot_state, val_loss, best, epoch, min_delta
) -> tuple[dict[str, Any] | None, float, int, int]:
    """Update best-state tracking; returns (state, best_loss, best_epoch, num_bad)."""
    best_state, best_loss, best_epoch, num_bad = best
    have_val = math.isfinite(val_loss)
    if not select_best:
        # Fixed-epoch behaviour: the most recent epoch is the kept state.
        return snapshot_state(), best_loss, epoch, 0
    if best_state is None or (have_val and val_loss < best_loss - min_delta):
        return snapshot_state(), (val_loss if have_val else best_loss), epoch, 0
    if have_val:
        return best_state, best_loss, best_epoch, num_bad + 1
    return best_state, best_loss, best_epoch, num_bad


def run_training_loop(
    model: Any,
    *,
    horizon: Any,
    loader,
    optimiser,
    loss_fn,
    forward_batch: Callable[[Any], tuple[torch.Tensor, torch.Tensor]],
    snapshot_state: Callable[[], dict[str, Any]],
    modules,
    val_eval: Callable[[], tuple[float, float]] | None = None,
    scheduler: Any = None,
    grad_accumulation_steps: int = 1,
    max_epochs: int = 3,
    early_stopping: bool = True,
    patience: int = 1,
    min_delta: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Shared neural training loop with optional validation-based early stopping.

    Trains for up to ``max_epochs``; when a ``val_eval`` closure is supplied it
    evaluates the validation split each epoch and records a per-epoch curve. When
    ``early_stopping`` is also true it keeps the best-validation state and stops
    after ``patience`` consecutive non-improving epochs (restoring the best
    state); otherwise it preserves the original fixed-epoch behaviour (final
    epoch's state) while still recording the curve for analysis.

    Closures keep this loop model-agnostic:
      * ``forward_batch(batch) -> (pred, target)`` — one train-mode forward;
      * ``val_eval() -> (val_loss, val_r2)`` — eval-mode pass over validation
        (loss on the training scale, R^2 on the RV scale; NaN ⇒ no signal);
      * ``snapshot_state() -> dict`` — a CPU-cloned checkpoint state dict.
    ``modules`` is the list of ``nn.Module`` toggled train()/eval() per phase.
    Returns ``(selected_state, val_curve)``.
    """
    select_best = bool(early_stopping) and (val_eval is not None)
    amp_device_type, amp_dtype = _resolve_amp(model)
    best_state: dict[str, Any] | None = None
    best_loss = math.inf
    best_epoch = 0
    num_bad = 0
    curve: list[dict[str, Any]] = []

    for epoch in range(max_epochs):
        epoch_start = monotonic()
        running_loss, n_batches = _train_one_epoch(
            loader=loader,
            optimiser=optimiser,
            loss_fn=loss_fn,
            forward_batch=forward_batch,
            modules=modules,
            grad_accumulation_steps=grad_accumulation_steps,
            scheduler=scheduler,
            amp_device_type=amp_device_type,
            amp_dtype=amp_dtype,
        )
        log_epoch_done(
            model,
            horizon=horizon,
            epoch=epoch + 1,
            max_epochs=max_epochs,
            running_loss=running_loss,
            n_batches=n_batches,
            start_time=epoch_start,
        )
        train_loss = running_loss / max(1, n_batches)

        val_loss, val_r2 = _evaluate_validation(
            val_eval, modules, amp_device_type=amp_device_type, amp_dtype=amp_dtype
        )
        best_state, best_loss, best_epoch, num_bad = _select_state(
            select_best=select_best,
            snapshot_state=snapshot_state,
            val_loss=val_loss,
            best=(best_state, best_loss, best_epoch, num_bad),
            epoch=epoch + 1,
            min_delta=min_delta,
        )
        curve.append(
            {
                "epoch": epoch + 1,
                "train_loss": _finite_or_none(train_loss),
                "val_loss": _finite_or_none(val_loss),
                "val_r2": _finite_or_none(val_r2),
                "is_best": best_epoch == epoch + 1,
            }
        )
        log_epoch_val(
            model,
            horizon=horizon,
            epoch=epoch + 1,
            val_loss=val_loss,
            val_r2=val_r2,
            best_epoch=best_epoch,
            num_bad=num_bad,
        )
        if select_best and math.isfinite(val_loss) and num_bad > 0 and num_bad >= patience:
            get_logger("train").info(
                "early stop",
                model=getattr(model, "name", type(model).__name__),
                horizon=_normalise(horizon),
                stopped_epoch=epoch + 1,
                best_epoch=best_epoch,
                patience=int(patience),
            )
            break

    if best_state is None:
        best_state = snapshot_state()
    return best_state, curve


def _finite_or_none(value: float) -> float | None:
    out = float(value)
    return out if math.isfinite(out) else None


def _checkpoint_path(model: Any, horizon: Any) -> Path | None:
    if not bool(getattr(model, "checkpoint", True)):
        return None
    checkpoint_dir = getattr(model, "checkpoint_dir", None)
    if checkpoint_dir is None:
        return None
    return Path(checkpoint_dir) / f"horizon_{_horizon_slug(horizon)}.pt"


def _checkpoint_config(model: Any) -> dict[str, Any]:
    attrs = [
        "hidden_dim",
        "dropout",
        "lr",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "grad_accumulation_steps",
        "early_stopping",
        "es_patience",
        "es_min_delta",
        "mixed_precision",
        "warmup_ratio",
        "log_target",
        "pretokenize",
        "tokenization_batch_size",
        "tokenizer_threads",
        "chunk_stride",
        "max_chunks",
        "attn_dim",
        "chunk_num_heads",
        "chunk_encoder_layers",
        "chunk_ff_dim",
        "proj_dim",
        "instruction",
        "normalize_emb",
    ]
    return {name: _normalise(getattr(model, name)) for name in attrs if hasattr(model, name)}


def _normalise(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _horizon_slug(horizon: Any) -> str:
    value = _normalise(horizon)
    return str(value).replace("/", "_").replace(" ", "_")


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch<2.0
        return torch.load(path, map_location="cpu")
