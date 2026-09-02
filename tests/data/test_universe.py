"""Time-varying universe membership validation tests."""

from __future__ import annotations

import pandas as pd
import pytest

from sp500vol.data.universe import is_member_on, members_on, validate_membership_table


def test_membership_schema_validation_normalizes_required_columns() -> None:
    permno = 14593
    table = validate_membership_table(
        pd.DataFrame(
            [
                {
                    "ticker": " aapl ",
                    "cik": 320193,
                    "member_from": "2020-01-01",
                    "member_to": "",
                    "permno": permno,
                    "source": "fixture",
                }
            ]
        )
    )

    row = table.iloc[0]
    assert row["ticker"] == "AAPL"
    assert row["cik"] == "0000320193"
    assert row["member_from"] == pd.Timestamp("2020-01-01")
    assert pd.isna(row["member_to"])
    # CRSP extras (permno, source) ride along through validation.
    assert row["source"] == "fixture"
    assert row["permno"] == permno


def test_membership_schema_validation_rejects_missing_columns_and_bad_cik() -> None:
    missing = pd.DataFrame([{"ticker": "AAPL", "cik": "0000320193", "member_from": "2020-01-01"}])
    with pytest.raises(ValueError, match="missing columns"):
        validate_membership_table(missing)

    bad_cik = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "cik": "not-a-cik",
                "member_from": "2020-01-01",
                "member_to": None,
            }
        ]
    )
    with pytest.raises(ValueError, match="invalid CIK"):
        validate_membership_table(bad_cik)


def test_membership_schema_validation_rejects_invalid_date_range() -> None:
    invalid_range = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "cik": "0000320193",
                "member_from": "2020-02-01",
                "member_to": "2020-01-31",
            }
        ]
    )

    with pytest.raises(ValueError, match="invalid date ranges"):
        validate_membership_table(invalid_range)


def test_membership_queries_include_boundary_dates() -> None:
    table = _membership_table(member_to="2020-01-31")

    assert is_member_on("aapl", pd.Timestamp("2020-01-01"), table)
    assert is_member_on("AAPL", pd.Timestamp("2020-01-31"), table)
    assert not is_member_on("AAPL", pd.Timestamp("2019-12-31"), table)
    assert members_on(pd.Timestamp("2020-01-31"), table) == ["AAPL"]


def test_removed_company_is_not_member_after_member_to() -> None:
    table = _membership_table(member_to="2020-01-31")

    assert not is_member_on("AAPL", pd.Timestamp("2020-02-01"), table)
    assert members_on(pd.Timestamp("2020-02-01"), table) == []


def test_membership_validation_rejects_overlapping_intervals() -> None:
    overlapping = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "cik": "0000320193",
                "member_from": "2020-01-01",
                "member_to": "2020-01-31",
            },
            {
                "ticker": "aapl",
                "cik": 320193,
                "member_from": "2020-01-31",
                "member_to": None,
            },
        ]
    )

    with pytest.raises(ValueError, match="overlapping intervals"):
        validate_membership_table(overlapping)


def _membership_table(*, member_to: str | None) -> pd.DataFrame:
    return validate_membership_table(
        pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "cik": "0000320193",
                    "member_from": "2020-01-01",
                    "member_to": member_to,
                }
            ]
        )
    )
