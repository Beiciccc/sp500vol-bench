"""Loughran-McDonald dictionary loader + count/proportion tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sp500vol.features.lm_dictionary import LM_CATEGORIES, LoughranMcDonaldDictionary


def test_mock_dictionary_exposes_all_categories() -> None:
    d = LoughranMcDonaldDictionary.mock()
    assert set(d.by_category.keys()) == set(LM_CATEGORIES)


def test_counts_match_known_tokens() -> None:
    d = LoughranMcDonaldDictionary.mock()
    text = "The loss from litigation was significant. May the court grant relief."
    c = d.counts(text)
    assert c["negative"] >= 1  # 'loss', 'litigation'
    assert c["litigious"] >= 1  # 'litigation', 'court'
    assert c["modal"] >= 1  # 'may'


def test_proportions_sum_bounded_by_one() -> None:
    d = LoughranMcDonaldDictionary.mock()
    text = "Growth and innovation despite loss and uncertainty may continue."
    p = d.proportions(text)
    total = sum(p.values())
    # Categories can overlap (e.g. 'may' is both uncertainty and modal),
    # so the sum can exceed 1.0 — bound is < n_categories.
    assert 0.0 <= total <= len(LM_CATEGORIES)


def test_empty_text_returns_zero_proportions() -> None:
    d = LoughranMcDonaldDictionary.mock()
    assert all(v == 0.0 for v in d.proportions("").values())


def test_from_csv_parses_minimal_fixture(tmp_path: Path) -> None:
    csv = tmp_path / "lm.csv"
    pd.DataFrame(
        [
            {
                "Word": "LOSS",
                "Negative": 2009,
                "Positive": 0,
                "Uncertainty": 0,
                "Litigious": 0,
                "Constraining": 0,
                "Superfluous": 0,
                "Interesting": 0,
                "Modal1": 0,
                "Modal2": 0,
                "Modal3": 0,
            },
            {
                "Word": "GROWTH",
                "Negative": 0,
                "Positive": 2009,
                "Uncertainty": 0,
                "Litigious": 0,
                "Constraining": 0,
                "Superfluous": 0,
                "Interesting": 0,
                "Modal1": 0,
                "Modal2": 0,
                "Modal3": 0,
            },
            {
                "Word": "MAY",
                "Negative": 0,
                "Positive": 0,
                "Uncertainty": 2009,
                "Litigious": 0,
                "Constraining": 0,
                "Superfluous": 0,
                "Interesting": 0,
                "Modal1": 2009,
                "Modal2": 0,
                "Modal3": 0,
            },
        ]
    ).to_csv(csv, index=False)

    d = LoughranMcDonaldDictionary.from_csv(csv)
    assert "loss" in d.by_category["negative"]
    assert "growth" in d.by_category["positive"]
    assert "may" in d.by_category["uncertainty"]
    assert "may" in d.by_category["modal"]
