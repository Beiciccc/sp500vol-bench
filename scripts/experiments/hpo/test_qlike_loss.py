"""Unit tests for the QLIKE-aligned objective (spec section 3.4: mandatory before
any trial runs). Run: .venv/bin/python -m pytest scripts/experiments/hpo/test_qlike_loss.py -q
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qlike_loss import EPS, U_CLAMP, make_objective, qlike_np  # noqa: E402


def test_zero_at_perfect_prediction():
    log_y = np.log(np.array([0.1, 0.25, 1.7]))
    assert np.allclose(qlike_np(log_y, log_y), 0.0, atol=1e-12)


def test_matches_direct_formula():
    rng = np.random.default_rng(0)
    y = np.exp(rng.normal(-1.5, 0.8, 500))
    f = np.exp(rng.normal(-1.5, 0.8, 500))
    direct = y / f - np.log(y / f) - 1.0
    ours = qlike_np(np.log(y), np.log(f))
    assert np.allclose(ours, direct, rtol=1e-10)


def test_nonnegative_and_convex_shape():
    log_y = np.zeros(1)
    zs = np.linspace(-5, 5, 201)
    vals = np.array([qlike_np(log_y, np.array([z]))[0] for z in zs])
    assert (vals >= -1e-12).all()
    i = int(np.argmin(vals))
    assert abs(zs[i]) < 0.06  # minimum at z = log_y = 0
    assert (np.diff(vals[: i + 1]) <= 1e-12).all() and (np.diff(vals[i:]) >= -1e-12).all()


def test_torch_matches_numpy_and_gradients_flow():
    rng = np.random.default_rng(1)
    log_y = torch.tensor(rng.normal(-1.0, 0.5, 64))
    z = torch.tensor(rng.normal(-1.0, 0.5, 64), requires_grad=True)
    loss = make_objective("qlike")(z, log_y)
    ref = float(qlike_np(log_y.numpy(), z.detach().numpy()).mean())
    assert np.isclose(float(loss), ref, rtol=1e-10)
    loss.backward()
    g = z.grad.numpy()
    assert np.isfinite(g).all()
    # under-prediction (z < log_y, u > 0): gradient d/dz = -(exp(u) - 1) < 0 -> push z UP
    under = z.detach().numpy() < log_y.numpy()
    assert (g[under] < 0).all() and (g[~under] > 0).all()


def test_extreme_inputs_stay_finite():
    log_y = torch.tensor([np.log(EPS), 5.0, -40.0])
    z = torch.tensor([50.0, -50.0, 40.0], requires_grad=True)
    loss = make_objective("qlike")(z, log_y)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(z.grad).all()
    assert float(torch.exp(torch.tensor(U_CLAMP))) < 1e14  # clamp keeps exp bounded


def test_mse_objective_unchanged():
    log_y = torch.tensor([0.0, 1.0])
    z = torch.tensor([1.0, 1.0])
    assert np.isclose(float(make_objective("mse")(z, log_y)), 0.5)


def test_batch_mean_equals_pointwise_mean():
    rng = np.random.default_rng(2)
    log_y = torch.tensor(rng.normal(size=100))
    z = torch.tensor(rng.normal(size=100))
    whole = float(make_objective("qlike")(z, log_y))
    parts = [float(make_objective("qlike")(z[i:i + 10], log_y[i:i + 10])) for i in range(0, 100, 10)]
    assert np.isclose(whole, np.mean(parts), rtol=1e-10)
