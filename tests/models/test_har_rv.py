"""HAR-RV model tests."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from sp500vol.models.price.har_rv import HARRV

_SAVE_EPSILON = 1e-10


def _feature_frame(
    n_rows: int = 48,
    *,
    include_rv_1d: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(7)
    rv_1d = rng.uniform(0.08, 0.35, size=n_rows)
    rv_5d = rng.uniform(0.10, 0.32, size=n_rows)
    rv_22d = rng.uniform(0.12, 0.28, size=n_rows)
    frame = pd.DataFrame(
        {
            "feature_rv_5d": rv_5d,
            "feature_rv_22d": rv_22d,
        }
    )
    if include_rv_1d:
        frame["feature_rv_1d"] = rv_1d
    else:
        frame["feature_return_1d"] = rv_1d / np.sqrt(252.0)
    return frame, rv_1d


def _target(
    rv_1d: np.ndarray,
    rv_5d: np.ndarray,
    rv_22d: np.ndarray,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    log_target = (
        np.log(0.03 * scale) + 0.20 * np.log(rv_1d) + 0.45 * np.log(rv_5d) + 0.35 * np.log(rv_22d)
    )
    return np.exp(log_target)


def test_fit_predict_dataframe_returns_finite_positive_values() -> None:
    features, rv_1d = _feature_frame()
    y = _target(
        rv_1d,
        features["feature_rv_5d"].to_numpy(),
        features["feature_rv_22d"].to_numpy(),
    )

    model = HARRV()
    model.fit(features, y)
    prediction = model.predict(features)

    assert prediction.shape == y.shape
    assert np.isfinite(prediction).all()
    assert (prediction > 0).all()


def test_fit_predict_uses_return_fallback_for_daily_rv() -> None:
    features, rv_1d = _feature_frame(include_rv_1d=False)
    y = _target(
        rv_1d,
        features["feature_rv_5d"].to_numpy(),
        features["feature_rv_22d"].to_numpy(),
    )

    model = HARRV()
    model.fit(features, y)
    prediction = model.predict(features)

    assert np.isfinite(prediction).all()
    assert (prediction > 0).all()


def test_dataframe_prefers_rv_1d_over_return_fallback() -> None:
    rv_1d = np.linspace(0.08, 0.35, num=24)
    rv_5d = np.full_like(rv_1d, 0.20)
    rv_22d = np.full_like(rv_1d, 0.24)
    features = pd.DataFrame(
        {
            "feature_rv_1d": rv_1d,
            "feature_return_1d": np.full_like(rv_1d, 0.99),
            "feature_rv_5d": rv_5d,
            "feature_rv_22d": rv_22d,
        }
    )
    y = _target(rv_1d, rv_5d, rv_22d)

    model = HARRV()
    model.fit(features, y)
    prediction = model.predict(features)

    assert np.std(prediction) > 0.0
    np.testing.assert_allclose(prediction, y, rtol=1e-8, atol=1e-12)


def test_fit_predict_numpy_array() -> None:
    features, rv_1d = _feature_frame()
    y = _target(
        rv_1d,
        features["feature_rv_5d"].to_numpy(),
        features["feature_rv_22d"].to_numpy(),
    )
    array = features[["feature_rv_1d", "feature_rv_5d", "feature_rv_22d"]].to_numpy()

    model = HARRV()
    model.fit(array, y)
    prediction = model.predict(array[:5])

    assert prediction.shape == (5,)
    assert np.isfinite(prediction).all()
    assert (prediction > 0).all()


def test_horizon_specific_models_fit_independently() -> None:
    base, rv_1d = _feature_frame()
    y_5d = _target(rv_1d, base["feature_rv_5d"].to_numpy(), base["feature_rv_22d"].to_numpy())
    y_20d = _target(
        rv_1d,
        base["feature_rv_5d"].to_numpy(),
        base["feature_rv_22d"].to_numpy(),
        scale=2.0,
    )
    features = pd.concat(
        [
            base.assign(horizon_days=5),
            base.assign(horizon_days=20),
        ],
        ignore_index=True,
    )
    y = np.concatenate([y_5d, y_20d])

    model = HARRV()
    model.fit(features, y)
    sample = pd.concat(
        [
            base.iloc[[0]].assign(horizon_days=5),
            base.iloc[[0]].assign(horizon_days=20),
        ],
        ignore_index=True,
    )

    prediction = model.predict(sample)

    assert prediction[1] > prediction[0]
    np.testing.assert_allclose(prediction, [y_5d[0], y_20d[0]], rtol=1e-8, atol=1e-12)


def test_save_load_round_trip(tmp_path) -> None:
    features, rv_1d = _feature_frame()
    y = _target(
        rv_1d,
        features["feature_rv_5d"].to_numpy(),
        features["feature_rv_22d"].to_numpy(),
    )
    model = HARRV(epsilon=_SAVE_EPSILON)
    model.fit(features, y)

    path = tmp_path / "har_rv.pkl"
    model.save(path)
    loaded = HARRV.load(path)

    np.testing.assert_allclose(loaded.predict(features), model.predict(features))
    with path.open("rb") as file:
        state = pickle.load(file)
    assert {"feature_names", "coefs", "intercepts", "epsilon"}.issubset(state)
    assert state["feature_names"] == ["feature_rv_1d", "feature_rv_5d", "feature_rv_22d"]
    assert state["epsilon"] == _SAVE_EPSILON
