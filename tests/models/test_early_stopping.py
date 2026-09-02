"""Unit tests for the shared early-stopping training loop (run_training_loop).

These drive ``run_training_loop`` with a trivial real module (so autograd works)
and a *scripted* validation closure, so the best-epoch selection, patience-based
early stop, and per-epoch validation curve are tested deterministically without
any model download or GPU. The same loop backs every neural/fusion model, so
this pins the behaviour all of them inherit.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import torch
from torch import nn

from sp500vol.models.neural_text._train_utils import make_scheduler, run_training_loop

_MODEL = SimpleNamespace(name="test-model")


def _components():
    """A tiny trainable module + no-op-ish data so autograd runs cheaply."""
    module = nn.Linear(1, 1)
    optimiser = torch.optim.SGD(module.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    loader = [torch.ones(2, 1), torch.ones(2, 1)]

    def forward_batch(batch):
        return module(batch).squeeze(-1), torch.zeros(2)

    counter = itertools.count(1)

    def snapshot_state():
        return {"call": next(counter)}

    return module, optimiser, loss_fn, loader, forward_batch, snapshot_state


def test_early_stops_before_max_epochs():
    module, optimiser, loss_fn, loader, forward_batch, snapshot_state = _components()
    # val worsens immediately: epoch 1 best, epoch 2 worse → patience 1 → stop.
    scripted = iter([(1.0, 0.5), (2.0, 0.4), (3.0, 0.3), (4.0, 0.2)])
    state, curve = run_training_loop(
        _MODEL,
        horizon=5,
        loader=loader,
        optimiser=optimiser,
        loss_fn=loss_fn,
        forward_batch=forward_batch,
        snapshot_state=snapshot_state,
        modules=[module],
        val_eval=lambda: next(scripted),
        max_epochs=4,
        early_stopping=True,
        patience=1,
    )
    assert [c["epoch"] for c in curve] == [1, 2]  # stopped early (2 of 4)
    assert curve[0]["is_best"] and not curve[1]["is_best"]
    assert state["call"] == 1  # best state is epoch 1's snapshot


def test_selects_best_midstream():
    module, optimiser, loss_fn, loader, forward_batch, snapshot_state = _components()
    # best validation is epoch 2; epoch 3 worse → stop after epoch 3 (== max).
    scripted = iter([(1.0, 0.1), (0.5, 0.3), (0.6, 0.2)])
    state, curve = run_training_loop(
        _MODEL,
        horizon=5,
        loader=loader,
        optimiser=optimiser,
        loss_fn=loss_fn,
        forward_batch=forward_batch,
        snapshot_state=snapshot_state,
        modules=[module],
        val_eval=lambda: next(scripted),
        max_epochs=3,
        early_stopping=True,
        patience=1,
    )
    assert [c["epoch"] for c in curve] == [1, 2, 3]
    assert curve[1]["is_best"] and not curve[2]["is_best"]
    assert state["call"] == 2  # best state is epoch 2's snapshot
    assert all(c["val_loss"] is not None for c in curve)


def test_no_validation_keeps_final_epoch():
    module, optimiser, loss_fn, loader, forward_batch, snapshot_state = _components()
    state, curve = run_training_loop(
        _MODEL,
        horizon=5,
        loader=loader,
        optimiser=optimiser,
        loss_fn=loss_fn,
        forward_batch=forward_batch,
        snapshot_state=snapshot_state,
        modules=[module],
        val_eval=None,
        max_epochs=3,
        early_stopping=True,
        patience=1,
    )
    assert [c["epoch"] for c in curve] == [1, 2, 3]  # never stops without val
    assert all(c["val_loss"] is None for c in curve)
    assert state["call"] == 3  # final epoch kept


def test_grad_accumulation_controls_optimiser_step():
    """With grad_accumulation_steps=2 over 4 batches, optimiser.step fires twice
    (at batch 2 and the final batch) — the effective-batch path used by the
    chunked models (physical batch 4 x accum 4 = effective 16)."""
    module, _opt, loss_fn, _loader, forward_batch, snapshot_state = _components()
    loader4 = [torch.ones(2, 1) for _ in range(4)]

    class CountingOpt:
        def __init__(self):
            self.steps = 0

        def zero_grad(self, set_to_none=True):
            pass

        def step(self):
            self.steps += 1

    opt = CountingOpt()
    run_training_loop(
        _MODEL,
        horizon=5,
        loader=loader4,
        optimiser=opt,
        loss_fn=loss_fn,
        forward_batch=forward_batch,
        snapshot_state=snapshot_state,
        modules=[module],
        val_eval=None,
        grad_accumulation_steps=2,
        max_epochs=1,
        early_stopping=False,
    )
    assert opt.steps == 2  # batches 2 and 4


def test_early_stopping_disabled_records_curve_without_stopping():
    module, optimiser, loss_fn, loader, forward_batch, snapshot_state = _components()
    scripted = iter([(1.0, 0.1), (2.0, 0.0), (3.0, -0.1)])  # worsening, would stop if enabled
    state, curve = run_training_loop(
        _MODEL,
        horizon=5,
        loader=loader,
        optimiser=optimiser,
        loss_fn=loss_fn,
        forward_batch=forward_batch,
        snapshot_state=snapshot_state,
        modules=[module],
        val_eval=lambda: next(scripted),
        max_epochs=3,
        early_stopping=False,
        patience=1,
    )
    assert [c["epoch"] for c in curve] == [1, 2, 3]  # runs all epochs
    assert all(c["val_loss"] is not None for c in curve)  # curve still recorded
    assert state["call"] == 3  # fixed-epoch behaviour: final epoch kept


def test_make_scheduler_warmup_then_decay():
    """Linear warmup to peak at warmup boundary, then linear decay toward 0."""
    module = nn.Linear(1, 1)
    optimiser = torch.optim.AdamW(module.parameters(), lr=1.0)
    # total opt steps = ceil(10/1) * 2 = 20; warmup = int(0.1 * 20) = 2
    sched = make_scheduler(
        optimiser,
        steps_per_epoch=10,
        max_epochs=2,
        grad_accumulation_steps=1,
        warmup_ratio=0.1,
    )
    lrs = []
    for _ in range(20):
        optimiser.step()
        sched.step()
        lrs.append(optimiser.param_groups[0]["lr"])
    assert lrs[0] < lrs[1]  # warming up
    assert max(lrs) == lrs[1]  # peak at end of warmup
    assert lrs[-1] < lrs[1]  # decaying after peak
    assert lrs[-1] < 0.05  # near zero at the end


def test_make_scheduler_none_when_warmup_zero():
    optimiser = torch.optim.AdamW(nn.Linear(1, 1).parameters(), lr=1.0)
    sched = make_scheduler(
        optimiser,
        steps_per_epoch=10,
        max_epochs=2,
        grad_accumulation_steps=1,
        warmup_ratio=0.0,
    )
    assert sched is None
