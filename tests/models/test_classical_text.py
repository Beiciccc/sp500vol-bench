"""Smoke tests for Block B classical text baselines.

These tests create a tiny on-disk corpus and verify each B-model can fit,
predict, and round-trip through save/load. They do not test predictive quality
— that is what the dry_run_medium + full ingestion + DM tests are for.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sp500vol.features.lm_dictionary import LoughranMcDonaldDictionary
from sp500vol.models.classical_text import BoWRidge, LMFeatures, LMLinear, TfidfRidge, _fit_utils
from sp500vol.models.classical_text.bow_ridge import _maybe_exp as bow_maybe_exp
from sp500vol.models.classical_text.lm_linear import _maybe_exp as lm_maybe_exp

_TEXTS_PER_FILING = [
    (
        "Apple reported strong revenue growth and innovation in services. "
        "Risks include uncertainty in supply chains and litigation. "
        "Management discusses growth strategy and gross margin expansion."
    ),
    (
        "Microsoft cloud revenue continues to grow. "
        "Operating loss in legacy segments narrowed. "
        "We may face adverse regulatory developments."
    ),
    (
        "Nvidia GPU sales decelerated. Inventory growth was significant. "
        "Court settlement may impose constraints on advertising claims."
    ),
    (
        "Amazon retail margins improved. "
        "Could face adverse currency impacts. "
        "Required disclosures relating to litigation are detailed below."
    ),
    (
        "Tesla deliveries grew despite supply chain disruption. "
        "Innovation in battery technology continues to expand."
    ),
    ("Meta advertising revenue stabilised. Adverse impact from European regulation may persist."),
]


@pytest.fixture()
def corpus(tmp_path: Path) -> pd.DataFrame:
    rows = []
    horizons = [5, 10, 20]
    for i, text in enumerate(_TEXTS_PER_FILING):
        text_path = tmp_path / f"filing_{i}.txt"
        text_path.write_text(text, encoding="utf-8")
        for h in horizons:
            rows.append(
                {
                    "accession": f"acc-{i:04d}",
                    "horizon_days": h,
                    "text_path": str(text_path),
                    "label_realised_vol": 0.15 + 0.01 * i + 0.001 * h,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "model_cls",
    [BoWRidge, TfidfRidge],
)
def test_bow_family_fit_predict_round_trip(corpus: pd.DataFrame, tmp_path: Path, model_cls) -> None:
    model = model_cls(max_features=200, ridge_alpha=0.5)
    model.fit(corpus, corpus["label_realised_vol"].to_numpy())
    pred = model.predict(corpus)
    assert pred.shape == (len(corpus),)
    assert np.isfinite(pred).all()
    assert (pred > 0).all()

    save_path = tmp_path / f"{model.name}.pkl"
    model.save(save_path)
    loaded = model_cls.load(save_path)
    np.testing.assert_allclose(loaded.predict(corpus), pred)


@pytest.mark.parametrize(
    "model_cls",
    [BoWRidge, TfidfRidge],
)
def test_bow_family_ridgecv_round_trip(corpus: pd.DataFrame, tmp_path: Path, model_cls) -> None:
    model = model_cls(max_features=200, ridge_alpha=None)
    model.fit(corpus, corpus["label_realised_vol"].to_numpy())
    pred = model.predict(corpus)

    save_path = tmp_path / f"{model.name}_ridgecv.pkl"
    model.save(save_path)
    loaded = model_cls.load(save_path)

    assert loaded.ridge_alpha is None
    np.testing.assert_allclose(loaded.predict(corpus), pred)


def test_bow_family_predictions_are_floored() -> None:
    pred = bow_maybe_exp(np.array([-100.0, 100.0]), log_target=True)

    np.testing.assert_allclose(pred, np.array([0.02, 5.0]))


def test_lm_linear_fits_and_predicts(corpus: pd.DataFrame, tmp_path: Path) -> None:
    model = LMLinear(dictionary=LoughranMcDonaldDictionary.mock(), ridge_alpha=0.5)
    model.fit(corpus, corpus["label_realised_vol"].to_numpy())
    pred = model.predict(corpus)
    assert pred.shape == (len(corpus),)
    assert (pred > 0).all()

    save_path = tmp_path / "lm_linear.pkl"
    model.save(save_path)
    loaded = LMLinear.load(save_path)
    np.testing.assert_allclose(loaded.predict(corpus), pred)


def test_lm_linear_ridgecv_round_trip(corpus: pd.DataFrame, tmp_path: Path) -> None:
    model = LMLinear(dictionary=LoughranMcDonaldDictionary.mock(), ridge_alpha=None)
    model.fit(corpus, corpus["label_realised_vol"].to_numpy())
    pred = model.predict(corpus)

    save_path = tmp_path / "lm_linear_ridgecv.pkl"
    model.save(save_path)
    loaded = LMLinear.load(save_path)

    assert loaded.ridge_alpha is None
    np.testing.assert_allclose(loaded.predict(corpus), pred)


def test_lm_linear_predictions_are_floored() -> None:
    pred = lm_maybe_exp(np.array([-100.0, 100.0]), log_target=True)

    np.testing.assert_allclose(pred, np.array([0.02, 5.0]))


def test_lm_features_extends_feature_names(corpus: pd.DataFrame) -> None:
    model = LMFeatures(dictionary=LoughranMcDonaldDictionary.mock(), ridge_alpha=0.5)
    model.fit(corpus, corpus["label_realised_vol"].to_numpy())
    pred = model.predict(corpus)
    assert pred.shape == (len(corpus),)
    assert (pred > 0).all()
    # B4 must produce strictly more features than B3
    assert set(model.feature_names).issuperset(LMLinear().feature_names)
    assert len(model.feature_names) == len(LMLinear().feature_names) + 4


def test_text_n_jobs_defaults_to_safe_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SP500VOL_TEXT_N_JOBS", raising=False)

    assert _fit_utils.text_n_jobs() == 4


def test_text_n_jobs_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SP500VOL_TEXT_N_JOBS", "2")
    assert _fit_utils.text_n_jobs() == 2

    monkeypatch.setenv("SP500VOL_TEXT_N_JOBS", "-1")
    with pytest.raises(ValueError, match="positive integer"):
        _fit_utils.text_n_jobs()
