"""Regression tests for WRDS CIK override ingestion."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd


def _load_ingest_wrds_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "ingest_wrds.py"
    spec = importlib.util.spec_from_file_location("ingest_wrds_for_tests", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ingest_wrds = _load_ingest_wrds_module()


class _Log:
    def info(self, *args: object, **kwargs: object) -> None:
        return None

    def warning(self, *args: object, **kwargs: object) -> None:
        return None


def test_build_membership_splits_spells_across_cik_windows() -> None:
    con = pd.DataFrame(
        [
            _constituent_day(69550, "MYL", "2010-01-04", "2020-11-16", "2010-01-04"),
            _constituent_day(69550, "MYL", "2010-01-04", "2020-11-16", "2020-11-16"),
            _constituent_day(27983, "XRX", "2010-01-04", "2021-03-19", "2010-01-04"),
            _constituent_day(27983, "XRX", "2010-01-04", "2021-03-19", "2021-03-19"),
            _constituent_day(70500, "CCE", "2010-01-04", "2016-05-27", "2010-01-04"),
            _constituent_day(70500, "CCE", "2010-01-04", "2016-05-27", "2016-05-27"),
        ]
    )
    links = pd.DataFrame(
        [
            _link(69550, "0000069499", "1973-02-23", "2015-02-26"),
            _link(69550, "0001623613", "2015-02-27", "2020-11-16"),
            _link(27983, "0000108772", "1962-01-31", "2019-07-31"),
            _link(27983, "0001770450", "2019-08-01", None),
            _link(70500, "0000804055", "1986-11-21", "2010-10-01"),
            _link(70500, "0001491675", "2010-10-02", "2016-05-27"),
        ]
    )

    membership = ingest_wrds.build_membership(con, links, _Log())

    assert _cik_sequence(membership, "MYL") == ["0000069499", "0001623613"]
    assert _cik_sequence(membership, "XRX") == ["0000108772", "0001770450"]
    assert _cik_sequence(membership, "CCE") == ["0000804055", "0001491675"]
    assert membership.loc[
        (membership["ticker"] == "MYL") & (membership["cik"] == "0001623613"),
        "member_from",
    ].iloc[0] == pd.Timestamp("2015-02-27")


# permno, correct old CIK, wrong successor CIK in raw CCM, in-scope spell.
STAGE2_BACKFILL = {
    "CVC": (68857, "0001053112", "0001702780", "2010-12-20", "2016-06-21"),
    "DNB": (88590, "0001115222", "0001799208", "2010-01-04", "2017-04-04"),
    "HFC": (32803, "0000048039", "0001915657", "2018-06-18", "2021-06-03"),
    "JNS": (88313, "0001065865", "0002043380", "2010-01-04", "2011-11-22"),
    "NE": (90537, "0001458891", "0001895262", "2011-01-18", "2015-07-17"),
    "RX": (84020, "0001058083", "0001595262", "2010-01-04", "2010-02-25"),
}


def test_stage2_backfill_overrides_substitute_old_ciks() -> None:
    """M&A-successor backfills relabel each PERMNO's full spell to its old CIK.

    Regression for the silent zero-recovery bug: CCM's 2026 header pointed each
    of these PERMNOs at a successor CIK that files nothing in 2010-2025, so the
    old registrant was never queued. The override must restore the old CIK across
    the whole in-scope membership window and never emit the wrong successor CIK.
    """
    overrides = ingest_wrds._override_frame().set_index("permno")
    wrong_ciks = {wrong for _p, _old, wrong, _s, _e in STAGE2_BACKFILL.values()}

    con_rows = []
    for ticker, (permno, _old, _wrong, start, end) in STAGE2_BACKFILL.items():
        assert overrides.loc[permno, "cik"] == _old
        con_rows.append(_constituent_day(permno, ticker, start, end, start))
        con_rows.append(_constituent_day(permno, ticker, start, end, end))

    membership = ingest_wrds.build_membership(
        pd.DataFrame(con_rows), ingest_wrds._override_frame(), _Log()
    )

    for ticker, (_permno, old_cik, _wrong, start, end) in STAGE2_BACKFILL.items():
        rows = membership[membership["ticker"] == ticker]
        assert _cik_sequence(membership, ticker) == [old_cik]
        assert rows["member_from"].iloc[0] == pd.Timestamp(start)
        assert rows["member_to"].iloc[0] == pd.Timestamp(end)
    assert not set(membership["cik"]).intersection(wrong_ciks)


def test_stage1_detector_flags_only_too_early_successor_ciks() -> None:
    membership = pd.DataFrame(
        [
            {
                "ticker": "XRX",
                "permno": 27983,
                "cik": "0001770450",
                "member_from": pd.Timestamp("2010-01-04"),
                "member_to": pd.Timestamp("2021-03-19"),
            },
            {
                "ticker": "XRX",
                "permno": 27983,
                "cik": "0001770450",
                "member_from": pd.Timestamp("2019-08-01"),
                "member_to": pd.Timestamp("2021-03-19"),
            },
        ]
    )

    anomalies = ingest_wrds.detect_stage1_exited_cik_anomalies(membership)

    assert anomalies["member_from"].tolist() == [pd.Timestamp("2010-01-04")]


def _constituent_day(
    permno: int,
    ticker: str,
    start: str,
    end: str,
    day: str,
) -> dict[str, object]:
    return {
        "permno": permno,
        "ticker": ticker,
        "MbrStartDt": pd.Timestamp(start),
        "MbrEndDt": pd.Timestamp(end),
        "DlyCalDt": pd.Timestamp(day),
    }


def _link(permno: int, cik: str, start: str, end: str | None) -> dict[str, object]:
    return {
        "permno": permno,
        "cik": cik,
        "link_start": pd.Timestamp(start),
        "link_end": pd.NaT if end is None else pd.Timestamp(end),
    }


def _cik_sequence(membership: pd.DataFrame, ticker: str) -> list[str]:
    rows = membership[membership["ticker"] == ticker].sort_values("member_from")
    return rows["cik"].tolist()
