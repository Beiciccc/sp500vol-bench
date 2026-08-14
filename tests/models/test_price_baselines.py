"""Smoke tests for A1 HV, A3 GARCH, A4 EGARCH, A5 ARIMA.

Uses tiny synthetic data (single ticker, ~200 days). Verifies fit/predict
return finite positive values; predictive quality is out of scope.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sp500vol.models.price import EGARCH, GARCH, ARIMAVol, NaiveHV


@pytest.fixture()
def synthetic_returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-03", periods=250, freq="B")
    return pd.DataFrame(
        {
            "ticker": "AAPL",
            "date": dates,
            "log_return": rng.normal(loc=0.0, scale=0.015, size=len(dates)),
        }
    )


@pytest.fixture()
def filings(synthetic_returns: pd.DataFrame) -> pd.DataFrame:
    # Pick three filing dates spread through the synthetic period
    sample_dates = synthetic_returns["date"].iloc[[120, 180, 240]].tolist()
    rows = []
    for d in sample_dates:
        for horizon in (5, 10, 20):
            rows.append(
                {
                    "ticker": "AAPL",
                    "form": "8-K",
                    "filing_time_utc": pd.Timestamp(d).tz_localize("UTC"),
                    "feature_window_end": pd.Timestamp(d) - pd.Timedelta(days=1),
                    "feature_rv_22d": 0.25,
                    "horizon_days": horizon,
                    "label_realised_vol": 0.20 + 0.01 * horizon,
                }
            )
    return pd.DataFrame(rows)


def test_naive_hv_uses_rv_22d(filings: pd.DataFrame) -> None:
    model = NaiveHV()
    model.fit(filings, filings["label_realised_vol"].to_numpy())
    pred = model.predict(filings)

    np.testing.assert_allclose(pred, filings["feature_rv_22d"].to_numpy())


def test_naive_hv_uses_horizon_aware_features(filings: pd.DataFrame) -> None:
    rows = filings.iloc[:3].copy()
    rows["horizon_days"] = [5, 10, 20]
    rows["feature_rv_5d"] = 0.10
    rows["feature_rv_22d"] = 0.30

    model = NaiveHV()
    model.fit(rows, rows["label_realised_vol"].to_numpy())
    pred = model.predict(rows)

    expected_10d = np.sqrt((2.0 / 3.0) * 0.10**2 + (1.0 / 3.0) * 0.30**2)
    np.testing.assert_allclose(pred, [0.10, expected_10d, 0.30])


def test_garch_fit_predict_returns_positive(
    synthetic_returns: pd.DataFrame, filings: pd.DataFrame
) -> None:
    model = GARCH(market_returns_df=synthetic_returns)
    model.fit(filings, filings["label_realised_vol"].to_numpy())
    pred = model.predict(filings)

    assert pred.shape == (len(filings),)
    assert np.isfinite(pred).all()
    assert (pred > 0).all()


def test_egarch_fit_predict_returns_positive(
    synthetic_returns: pd.DataFrame, filings: pd.DataFrame
) -> None:
    model = EGARCH(market_returns_df=synthetic_returns)
    model.fit(filings, filings["label_realised_vol"].to_numpy())
    pred = model.predict(filings)

    assert pred.shape == (len(filings),)
    assert np.isfinite(pred).all()
    assert (pred > 0).all()


def test_arima_fit_predict_returns_positive(
    synthetic_returns: pd.DataFrame, filings: pd.DataFrame
) -> None:
    model = ARIMAVol(market_returns_df=synthetic_returns)
    model.fit(filings, filings["label_realised_vol"].to_numpy())
    pred = model.predict(filings)

    assert pred.shape == (len(filings),)
    assert np.isfinite(pred).all()
    assert (pred > 0).all()


def test_garch_falls_back_to_rv22_when_insufficient_history(filings: pd.DataFrame) -> None:
    # Provide market data that doesn't cover the filing tickers
    empty_market = pd.DataFrame(
        {"ticker": ["MSFT"], "date": [pd.Timestamp("2022-01-03")], "log_return": [0.0]}
    )
    model = GARCH(market_returns_df=empty_market)
    model.fit(filings, filings["label_realised_vol"].to_numpy())
    pred = model.predict(filings)

    # Falls back to feature_rv_22d = 0.25 for all rows
    assert np.allclose(pred, 0.25)


def test_garch_falls_back_to_rv22_when_forecast_is_tiny(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_returns: pd.DataFrame,
    filings: pd.DataFrame,
) -> None:
    # The refit+forecast runs in joblib worker processes, so monkeypatch the
    # parallel driver (as imported into the garch module) to emulate implausibly
    # tiny forecasts for every row; predict must then fall back to feature_rv_22d.
    import sp500vol.models.price.garch as garch_mod

    monkeypatch.setattr(
        garch_mod,
        "parallel_refit_predict",
        lambda df, returns_by_ticker, refit_one, params, **kwargs: np.full(len(df), 1e-12),
    )

    model = GARCH(market_returns_df=synthetic_returns)
    model.fit(filings, filings["label_realised_vol"].to_numpy())
    pred = model.predict(filings)

    assert np.allclose(pred, 0.25)


def test_garch_falls_back_to_horizon_hv_when_forecast_explodes(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_returns: pd.DataFrame,
    filings: pd.DataFrame,
) -> None:
    import sp500vol.models.price.garch as garch_mod

    monkeypatch.setattr(
        garch_mod,
        "parallel_refit_predict",
        lambda df, returns_by_ticker, refit_one, params, **kwargs: np.full(len(df), 1e30),
    )

    rows = filings.iloc[:3].copy()
    rows["horizon_days"] = [5, 10, 20]
    rows["feature_rv_5d"] = 0.10
    rows["feature_rv_22d"] = 0.30
    model = GARCH(market_returns_df=synthetic_returns)
    model.fit(rows, rows["label_realised_vol"].to_numpy())
    pred = model.predict(rows)

    expected_10d = np.sqrt((2.0 / 3.0) * 0.10**2 + (1.0 / 3.0) * 0.30**2)
    np.testing.assert_allclose(pred, [0.10, expected_10d, 0.30])


def test_garch_parallel_maps_predictions_to_correct_ticker_rows() -> None:
    # Two tickers with very different vol; the by-ticker parallel path must map
    # each prediction back to its own ticker's rows without cross-contaminating.
    rng = np.random.default_rng(1)
    dates = pd.date_range("2022-01-03", periods=250, freq="B")
    low = pd.DataFrame(
        {"ticker": "LOW", "date": dates, "log_return": rng.normal(0.0, 0.004, len(dates))}
    )
    high = pd.DataFrame(
        {"ticker": "HIGH", "date": dates, "log_return": rng.normal(0.0, 0.045, len(dates))}
    )
    market = pd.concat([low, high], ignore_index=True)

    rows = [
        {
            "ticker": tk,
            "form": "8-K",
            "filing_time_utc": pd.Timestamp(dates[200]).tz_localize("UTC"),
            "feature_window_end": pd.Timestamp(dates[199]),
            "feature_rv_22d": 0.2,
            "horizon_days": h,
            "label_realised_vol": 0.2,
        }
        for tk in ("LOW", "HIGH")
        for h in (5, 10, 20)
    ]
    filings = pd.DataFrame(rows)

    model = GARCH(market_returns_df=market)
    model.fit(filings, filings["label_realised_vol"].to_numpy())
    pred = model.predict(filings)

    assert np.isfinite(pred).all()
    assert (pred > 0).all()
    tickers = filings["ticker"].to_numpy()
    assert pred[tickers == "HIGH"].mean() > pred[tickers == "LOW"].mean()
