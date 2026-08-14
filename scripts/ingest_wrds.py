"""Build canonical SP500Vol artifacts from the 5 WRDS CRSP/Compustat zips.

One-time, idempotent. Each zip holds ONE csv with a random inner name; we read
it via zipfile (do not rely on the name). Emits three artifacts:

  A. data/universe/sp500_membership.parquet   (in-repo, small)
       schema: ticker, permno, cik, member_from, member_to, source
  B. $SP500VOL_DATA_ROOT/market/crsp/{market_returns,daily_returns}.parquet
       market_returns: ticker, date, log_return = log1p(DlyRet)
       daily_returns:  ticker, date, open, high, low, close, adj_close, volume
  C. data/universe/crsp_cik_links.parquet      (in-repo, small)
       point-in-time PERMNO->CIK link table (CCM, wide link filters)

CRSP DlyRet is the split/dividend-adjusted total return -> log_return = log1p.
CIK is resolved point-in-time (a PERMNO can map to different CIKs across M&A).

Usage:
    python scripts/ingest_wrds.py --wrds-dir "/Volumes/Z/sp500vol-data/raw/wrds"
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sp500vol.data.universe import validate_membership_table
from sp500vol.utils import configure_logging, get_logger
from sp500vol.utils.paths import data_path

EXTRACT_END = pd.Timestamp("2025-12-31")

# Manual CIK windows for WRDS/CCM rows where the 2026 company header points a
# historical PERMNO at a successor registrant. Each PERMNO listed here replaces
# all CCM-derived rows for that PERMNO, so contiguous successor windows are
# included when the same PERMNO remains in the 2010-2025 universe.
#
# FRC (PERMNO 12448) and SBNY (PERMNO 90090) are intentionally not backfilled:
# both are FDIC document-only failures in this window, with no SEC periodic
# registrant history to synthesize into the equity filing universe.
CIK_LINK_OVERRIDES = [
    {
        "permno": 76226,  # CBS/VIAC/PARA before the Paramount Skydance merger.
        "cik": "0000813828",
        "link_start": pd.Timestamp("1994-10-01"),
        "link_end": pd.Timestamp("2025-08-06"),
    },
    {
        "permno": 11081,  # Dell Inc before the 2013 go-private transaction.
        "cik": "0000826083",
        "link_start": pd.Timestamp("1988-06-22"),
        "link_end": pd.Timestamp("2013-10-31"),
    },
    {
        "permno": 69550,  # Mylan Inc.
        "cik": "0000069499",
        "link_start": pd.Timestamp("1973-02-23"),
        "link_end": pd.Timestamp("2015-02-26"),
    },
    {
        "permno": 69550,  # Mylan N.V.
        "cik": "0001623613",
        "link_start": pd.Timestamp("2015-02-27"),
        "link_end": pd.Timestamp("2020-11-16"),
    },
    {
        "permno": 27983,  # Xerox Corp before the holding-company reorg.
        "cik": "0000108772",
        "link_start": pd.Timestamp("1962-01-31"),
        "link_end": pd.Timestamp("2019-07-31"),
    },
    {
        "permno": 27983,  # Xerox Holdings Corp.
        "cik": "0001770450",
        "link_start": pd.Timestamp("2019-08-01"),
        "link_end": pd.NaT,
    },
    {
        "permno": 40125,  # Computer Sciences Corp before DXC.
        "cik": "0000023082",
        "link_start": pd.Timestamp("1964-04-30"),
        "link_end": pd.Timestamp("2015-11-30"),
    },
    {
        "permno": 20626,  # Dow Chemical before DowDuPont.
        "cik": "0000029915",
        "link_start": pd.Timestamp("1950-06-01"),
        "link_end": pd.Timestamp("2017-08-31"),
    },
    {
        "permno": 77768,  # Praxair before Linde PLC.
        "cik": "0000884905",
        "link_start": pd.Timestamp("1992-07-06"),
        "link_end": pd.Timestamp("2018-10-30"),
    },
    {
        "permno": 23077,  # H.J. Heinz before Kraft Heinz.
        "cik": "0000046640",
        "link_start": pd.Timestamp("1950-05-01"),
        "link_end": pd.Timestamp("2013-06-10"),
    },
    {
        "permno": 66325,  # SLM Corp before the Navient split.
        "cik": "0001032033",
        "link_start": pd.Timestamp("1983-09-23"),
        "link_end": pd.NaT,
    },
    {
        "permno": 45356,  # Tyco/JCI share line; CIK 833444 covers Tyco and JCI filings.
        "cik": "0000833444",
        "link_start": pd.Timestamp("1967-11-14"),
        "link_end": pd.NaT,
    },
    {
        "permno": 19502,  # Walgreen Co.
        "cik": "0000104207",
        "link_start": pd.Timestamp("1949-10-01"),
        "link_end": pd.Timestamp("2014-12-30"),
    },
    {
        "permno": 19502,  # Walgreens Boots Alliance.
        "cik": "0001618921",
        "link_start": pd.Timestamp("2014-12-31"),
        "link_end": pd.Timestamp("2025-08-29"),
    },
    {
        "permno": 78916,  # Watson Pharmaceuticals before Actavis/Allergan.
        "cik": "0000884629",
        "link_start": pd.Timestamp("1993-02-17"),
        "link_end": pd.Timestamp("2013-01-23"),
    },
    {
        "permno": 21186,  # MeadWestvaco before WestRock.
        "cik": "0001159297",
        "link_start": pd.Timestamp("1949-11-01"),
        "link_end": pd.Timestamp("2015-07-01"),
    },
    {
        "permno": 21186,  # WRKCo Inc, initial WestRock registrant.
        "cik": "0001636023",
        "link_start": pd.Timestamp("2015-07-02"),
        "link_end": pd.Timestamp("2018-11-04"),
    },
    {
        "permno": 21186,  # WestRock Co.
        "cik": "0001732845",
        "link_start": pd.Timestamp("2018-11-05"),
        "link_end": pd.Timestamp("2024-07-31"),
    },
    {
        "permno": 77668,  # Express Scripts Inc.
        "cik": "0000885721",
        "link_start": pd.Timestamp("1992-06-09"),
        "link_end": pd.Timestamp("2012-04-05"),
    },
    {
        "permno": 77668,  # Express Scripts Holding Co.
        "cik": "0001532063",
        "link_start": pd.Timestamp("2012-04-06"),
        "link_end": pd.Timestamp("2018-12-31"),
    },
    {
        "permno": 77182,  # Perrigo Co before Irish plc.
        "cik": "0000820096",
        "link_start": pd.Timestamp("1991-12-17"),
        "link_end": pd.Timestamp("2013-08-27"),
    },
    {
        "permno": 77182,  # Perrigo Co plc.
        "cik": "0001585364",
        "link_start": pd.Timestamp("2013-08-28"),
        "link_end": pd.NaT,
    },
    {
        "permno": 70500,  # Coca-Cola Enterprises before the 2010 NA transaction.
        "cik": "0000804055",
        "link_start": pd.Timestamp("1986-11-21"),
        "link_end": pd.Timestamp("2010-10-01"),
    },
    {
        "permno": 70500,  # Coca-Cola European Partners US.
        "cik": "0001491675",
        "link_start": pd.Timestamp("2010-10-02"),
        "link_end": pd.Timestamp("2016-05-27"),
    },
    {
        "permno": 92618,  # Dr Pepper Snapple: blank CIK in CCM.
        "cik": "0001418135",
        "link_start": pd.Timestamp("2008-05-07"),
        "link_end": pd.NaT,
    },
    # M&A-successor backfills: CCM's 2026 header collapses each of these PERMNOs
    # onto a single SUCCESSOR registrant CIK that only begins filing AFTER the
    # 2010-2025 membership window (verified: successor CIK has 0 in-window
    # 10-K/10-Q/8-K), so without an override the old registrant — which holds all
    # in-window filings — is never queued. Each PERMNO exits the index at the
    # reorg, so a single old-CIK window (no successor window) is full coverage.
    {
        "permno": 68857,  # Cablevision Systems Corp before the Altice acquisition.
        "cik": "0001053112",  # CCM points 68857 at successor 0001702780 (Optimum/Altice).
        "link_start": pd.Timestamp("1986-01-17"),
        "link_end": pd.Timestamp("2016-06-21"),
    },
    {
        "permno": 88590,  # Dun & Bradstreet Corp before the 2019 take-private.
        "cik": "0001115222",  # CCM points 88590 at successor 0001799208 (D&B Holdings).
        "link_start": pd.Timestamp("2000-10-03"),
        "link_end": pd.Timestamp("2019-02-08"),
    },
    {
        "permno": 32803,  # HollyFrontier Corp (was Holly Corp) before the HF Sinclair holdco.
        "cik": "0000048039",  # CCM points 32803 at successor 0001915657 (HF Sinclair).
        "link_start": pd.Timestamp("1965-11-30"),
        "link_end": pd.Timestamp("2022-03-13"),
    },
    {
        "permno": 88313,  # Janus Capital Group before the Janus Henderson merger.
        "cik": "0001065865",  # CCM points 88313 at successor 0002043380 (Janus Henderson US).
        "link_start": pd.Timestamp("2000-06-26"),
        "link_end": pd.Timestamp("2017-05-30"),
    },
    {
        "permno": 90537,  # Noble Corp (Switzerland/plc) before the 2022 Finco reorg.
        "cik": "0001458891",  # CCM points 90537 at successor 0001895262 (Noble Finco).
        "link_start": pd.Timestamp("1985-09-23"),
        "link_end": pd.Timestamp("2022-09-30"),
    },
    {
        "permno": 84020,  # IMS Health Inc before the 2010 take-private.
        "cik": "0001058083",  # CCM points 84020 at successor 0001595262 (IMS Health Holdings).
        "link_start": pd.Timestamp("1996-11-04"),
        "link_end": pd.Timestamp("2010-03-02"),
    },
]

STAGE1_EXITED_TOO_YOUNG_MIN_START = {
    ("CBS", "0002041610"): pd.Timestamp("2025-08-07"),
    ("CCE", "0001650107"): pd.Timestamp("2016-05-28"),
    ("CSC", "0001688568"): pd.Timestamp("2017-04-01"),
    ("DELL", "0001571996"): pd.Timestamp("2024-09-23"),
    ("DOW", "0001666700"): pd.Timestamp("2017-09-01"),
    ("ESRX", "0001532063"): pd.Timestamp("2012-04-06"),
    ("HNZ", "0001637459"): pd.Timestamp("2015-07-02"),
    ("MYL", "0001792044"): pd.Timestamp("2020-11-17"),
    ("PARA", "0002041610"): pd.Timestamp("2025-08-07"),
    ("PRGO", "0001585364"): pd.Timestamp("2013-08-28"),
    ("PX", "0001707925"): pd.Timestamp("2018-10-31"),
    ("SLM", "0001593538"): pd.Timestamp("2014-05-01"),
    ("TYC", "0001608109"): pd.Timestamp("2014-06-04"),
    ("VIAC", "0002041610"): pd.Timestamp("2025-08-07"),
    ("WAG", "0001618921"): pd.Timestamp("2014-12-31"),
    ("WPI", "0001578845"): pd.Timestamp("2013-10-01"),
    ("WRK", "0001732845"): pd.Timestamp("2018-11-05"),
    ("XRX", "0001770450"): pd.Timestamp("2019-08-01"),
}

CCM_ZIP = "ccm_csv.zip"
CONSTITUENTS_ZIP = "sp500_constituents_2010_2025_csv.zip"


def _read_zip_csv(zip_path: Path, **read_kwargs) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        inner = zf.namelist()[0]
        with zf.open(inner) as fh:
            return pd.read_csv(fh, **read_kwargs)


def _zpad_cik(value: object) -> str | None:
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    s = s.split(".")[0]  # tolerate float-like "12345.0"
    return s.zfill(10) if s.isdigit() else None


def _parse_link_end(value: object) -> pd.Timestamp:
    s = str(value).strip()
    if s in {"E", "", "nan", "None", "NaT"}:
        return pd.NaT
    ts = pd.to_datetime(s, errors="coerce")
    return pd.NaT if (pd.isna(ts) or ts >= EXTRACT_END) else ts


def _override_frame() -> pd.DataFrame:
    overrides = pd.DataFrame(CIK_LINK_OVERRIDES).copy()
    overrides["permno"] = overrides["permno"].astype(int)
    overrides["cik"] = overrides["cik"].map(_zpad_cik)
    overrides["link_start"] = pd.to_datetime(overrides["link_start"], errors="coerce")
    overrides["link_end"] = pd.to_datetime(overrides["link_end"], errors="coerce")
    if overrides[["permno", "cik", "link_start"]].isna().any().any():
        raise ValueError("CIK_LINK_OVERRIDES contains invalid permno/cik/link_start")
    return overrides


def build_cik_links(ccm_zip: Path, log) -> pd.DataFrame:
    """CCM -> point-in-time PERMNO->CIK link table (primary links only)."""
    raw = _read_zip_csv(
        ccm_zip,
        usecols=["GVKEY", "LPERMNO", "cik", "LINKDT", "LINKENDDT", "LINKTYPE", "LINKPRIM"],
        dtype=str,
    )
    # CCM link filter: confirmed/unconfirmed/subsidiary links + joint (J) primary
    # so dual-class (GOOG vs GOOGL, LINKPRIM=J), reorganised entities (Eaton LS,
    # Ceridian/Dayforce LX) all resolve. LD/LN duplicate/none links excluded.
    raw = raw[
        raw["LINKTYPE"].isin(["LC", "LU", "LS", "LX"]) & raw["LINKPRIM"].isin(["P", "C", "J"])
    ]
    raw["permno"] = pd.to_numeric(raw["LPERMNO"], errors="coerce")
    raw = raw.dropna(subset=["permno"])
    raw["permno"] = raw["permno"].astype(int)
    raw["cik"] = raw["cik"].map(_zpad_cik)
    # CIK is company-level (per GVKEY); secondary share-class rows (GOOG vs GOOGL,
    # FOX vs FOXA, NWS vs NWSA, DISCK vs DISCA) often leave cik blank. Backfill
    # each row's cik from the GVKEY's known cik so all share classes resolve.
    gvkey_cik = raw.dropna(subset=["cik"]).groupby("GVKEY")["cik"].first()
    raw["cik"] = raw["cik"].fillna(raw["GVKEY"].map(gvkey_cik))
    # PERMNO-level backfill for pure renames / reverse mergers: the old GVKEY's
    # cik is blank but the SAME PERMNO continues under a new GVKEY with one
    # unchanged cik (Alcoa->Arconic->Howmet 4281; MetroPCS->T-Mobile 1283699).
    # Skip PERMNOs carrying >1 distinct cik — those are genuine M&A successions
    # (Baker Hughes Inc->Co) whose point-in-time windows must stay separate.
    known = raw.dropna(subset=["cik"])
    single_cik = known.groupby("permno")["cik"].nunique()
    single_permnos = single_cik[single_cik == 1].index
    permno_cik = known.groupby("permno")["cik"].first()
    fill = raw["cik"].isna() & raw["permno"].isin(single_permnos)
    raw.loc[fill, "cik"] = raw.loc[fill, "permno"].map(permno_cik)
    raw["link_start"] = pd.to_datetime(raw["LINKDT"], errors="coerce")
    raw["link_end"] = raw["LINKENDDT"].map(_parse_link_end)

    links = raw.loc[raw["cik"].notna(), ["permno", "cik", "link_start", "link_end"]].copy()
    overrides = _override_frame()
    links = links[~links["permno"].isin(overrides["permno"])]
    links = pd.concat([links, overrides], ignore_index=True)
    links = links.sort_values(["permno", "link_start"]).reset_index(drop=True)
    log.info(
        "CIK links built",
        rows=len(links),
        distinct_permno=int(links["permno"].nunique()),
        open_links=int(links["link_end"].isna().sum()),
        override_permnos=int(overrides["permno"].nunique()),
        override_rows=len(overrides),
    )
    return links


def _read_constituents(con_zip: Path) -> pd.DataFrame:
    cols = [
        "PERMNO",
        "Ticker",
        "MbrStartDt",
        "MbrEndDt",
        "DlyCalDt",
        "DlyOpen",
        "DlyHigh",
        "DlyLow",
        "DlyClose",
        "DlyVol",
        "DlyRet",
    ]
    con = _read_zip_csv(con_zip, usecols=cols, dtype=str)
    con["permno"] = pd.to_numeric(con["PERMNO"], errors="coerce").astype("Int64")
    con = con.dropna(subset=["permno"])
    con["permno"] = con["permno"].astype(int)
    con["ticker"] = con["Ticker"].astype(str).str.strip().str.upper()
    for c in ["MbrStartDt", "MbrEndDt", "DlyCalDt"]:
        con[c] = pd.to_datetime(con[c], errors="coerce")
    for c in ["DlyOpen", "DlyHigh", "DlyLow", "DlyClose", "DlyVol", "DlyRet"]:
        con[c] = pd.to_numeric(con[c], errors="coerce")
    return con


def _coalesce_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping/adjacent intervals within each (ticker, cik) group.

    Dual-class share classes reuse one ticker under a single company CIK (e.g.
    Under Armour UA class A then class C), producing overlapping membership rows
    that are really one continuous span. Genuine re-entries (long gaps) stay split.
    """
    merged: list[dict] = []
    for _key, grp in intervals.groupby(["ticker", "cik"], sort=False):
        run: list[dict] = []
        for rec in grp.sort_values("member_from").to_dict("records"):
            if run:
                prev_end = run[-1]["member_to"]
                if pd.isna(prev_end) or rec["member_from"] <= prev_end + pd.Timedelta(days=1):
                    if pd.isna(prev_end) or pd.isna(rec["member_to"]):
                        run[-1]["member_to"] = pd.NaT
                    else:
                        run[-1]["member_to"] = max(prev_end, rec["member_to"])
                    continue
            run.append(dict(rec))
        merged.extend(run)
    return pd.DataFrame(merged, columns=list(intervals.columns))


def _split_membership_by_cik_windows(spells: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    link_groups = {int(permno): grp.copy() for permno, grp in links.groupby("permno", sort=False)}
    for rec in spells.to_dict("records"):
        permno = int(rec["permno"])
        spell_start = pd.Timestamp(rec["member_from"]).normalize()
        spell_end = (
            EXTRACT_END if pd.isna(rec["member_to"]) else pd.Timestamp(rec["member_to"]).normalize()
        )
        sub = link_groups.get(permno)
        if sub is None or sub.empty:
            rows.append({**rec, "cik": None})
            continue

        for link in sub.sort_values("link_start").itertuples(index=False):
            link_start = pd.Timestamp(link.link_start).normalize()
            link_end = (
                EXTRACT_END if pd.isna(link.link_end) else pd.Timestamp(link.link_end).normalize()
            )
            seg_start = max(spell_start, link_start)
            seg_end = min(spell_end, link_end)
            if seg_start > seg_end:
                continue
            member_to = pd.NaT if pd.isna(rec["member_to"]) and pd.isna(link.link_end) else seg_end
            rows.append(
                {
                    "ticker": rec["ticker"],
                    "permno": permno,
                    "cik": str(link.cik).zfill(10),
                    "member_from": seg_start,
                    "member_to": member_to,
                    "source": "crsp_wrds_2010_2025",
                }
            )
    return pd.DataFrame(
        rows,
        columns=["ticker", "permno", "cik", "member_from", "member_to", "source"],
    )


def detect_stage1_exited_cik_anomalies(membership: pd.DataFrame) -> pd.DataFrame:
    """Return known exited-single-window CIK regressions from the stage-1 audit."""
    if membership.empty:
        return membership.copy()
    required = {"ticker", "cik", "member_from", "member_to"}
    if missing := required.difference(membership.columns):
        raise ValueError(f"membership missing required columns: {sorted(missing)}")

    ticker = membership["ticker"].astype("string").str.strip().str.upper()
    cik = membership["cik"].astype("string").str.strip().str.zfill(10)
    member_from = pd.to_datetime(membership["member_from"], errors="coerce")
    ended = pd.to_datetime(membership["member_to"], errors="coerce").notna()
    too_young = []
    for ticker_value, cik_value, start in zip(ticker, cik, member_from, strict=True):
        min_start = STAGE1_EXITED_TOO_YOUNG_MIN_START.get((ticker_value, cik_value))
        too_young.append(min_start is not None and pd.notna(start) and start < min_start)
    bad = pd.Series(too_young, index=membership.index)
    return membership.loc[ended & bad].copy()


def build_membership(con: pd.DataFrame, links: pd.DataFrame, log) -> pd.DataFrame:
    """One interval per PIT (PERMNO, ticker, CIK, membership-spell) segment."""
    g = con.groupby(["permno", "ticker", "MbrStartDt", "MbrEndDt"], as_index=False).agg(
        first_day=("DlyCalDt", "min"), last_day=("DlyCalDt", "max")
    )
    g["member_from"] = g[["MbrStartDt", "first_day"]].max(axis=1)
    g["member_to"] = g[["MbrEndDt", "last_day"]].min(axis=1)
    g["member_to"] = g["member_to"].where(g["member_to"] < EXTRACT_END, pd.NaT)

    out = _split_membership_by_cik_windows(g, links)
    n_nocik = int(out["cik"].isna().sum())
    if n_nocik:
        sample = out.loc[out["cik"].isna(), ["permno", "ticker"]].drop_duplicates().head(15)
        log.warning(
            "intervals with unresolved CIK (dropped)",
            count=n_nocik,
            sample=sample.to_dict("records"),
        )
    out = out.loc[
        out["cik"].notna(), ["ticker", "permno", "cik", "member_from", "member_to", "source"]
    ]
    # Coalesce dual-class ticker reuse (Under Armour UA class A then class C) and
    # adjacent spells into continuous (ticker, cik) intervals so validation passes
    # and is_member_on stays exact.
    out = _coalesce_intervals(out)
    out = out.sort_values(["ticker", "member_from"]).reset_index(drop=True)
    anomalies = detect_stage1_exited_cik_anomalies(out)
    if not anomalies.empty:
        sample = anomalies[["ticker", "permno", "cik", "member_from", "member_to"]].head(20)
        raise ValueError(
            "stage-1 exited CIK regressions remain after overrides: "
            + sample.to_dict("records").__repr__()
        )
    log.info(
        "Membership intervals built",
        rows=len(out),
        distinct_permno=int(out["permno"].nunique()),
        distinct_ticker=int(out["ticker"].nunique()),
        current_members=int(out["member_to"].isna().sum()),
    )
    return out


def build_returns(con: pd.DataFrame, log) -> tuple[pd.DataFrame, pd.DataFrame]:
    """CRSP member-day rows -> (market_returns, ohlcv) with log1p(DlyRet)."""
    df = con.rename(columns={"DlyCalDt": "date"}).copy()
    df["date"] = df["date"].dt.normalize()
    df["log_return"] = np.log1p(df["DlyRet"])

    market_returns = (
        df.loc[df["log_return"].notna(), ["ticker", "date", "log_return"]]
        .drop_duplicates(["ticker", "date"])
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    ohlcv = (
        df.assign(
            open=df["DlyOpen"],
            high=df["DlyHigh"],
            low=df["DlyLow"],
            close=df["DlyClose"],
            adj_close=df["DlyClose"],
            volume=df["DlyVol"],
        )[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]]
        .drop_duplicates(["ticker", "date"])
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    log.info(
        "Returns store built",
        market_returns_rows=len(market_returns),
        ohlcv_rows=len(ohlcv),
        nan_log_return=int(df["log_return"].isna().sum()),
    )
    return market_returns, ohlcv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrds-dir", type=Path, required=True)
    parser.add_argument("--out-universe", type=Path, default=REPO_ROOT / "data" / "universe")
    args = parser.parse_args()

    configure_logging("INFO")
    log = get_logger("ingest_wrds")
    args.out_universe.mkdir(parents=True, exist_ok=True)

    # C. CIK links (in-repo)
    links = build_cik_links(args.wrds_dir / CCM_ZIP, log)
    links_path = args.out_universe / "crsp_cik_links.parquet"
    links.to_parquet(links_path, index=False)
    log.info("Wrote CIK links", path=str(links_path))

    # Read constituents once (members + daily prices/returns)
    con = _read_constituents(args.wrds_dir / CONSTITUENTS_ZIP)
    log.info(
        "Read constituents",
        rows=len(con),
        distinct_permno=int(con["permno"].nunique()),
        date_range=f"{con['DlyCalDt'].min().date()}..{con['DlyCalDt'].max().date()}",
    )

    # A. Membership (in-repo), validated
    membership = build_membership(con, links, log)
    membership = validate_membership_table(membership)
    mem_path = args.out_universe / "sp500_membership.parquet"
    membership.to_parquet(mem_path, index=False)
    log.info("Wrote membership", path=str(mem_path), rows=len(membership))

    # B. Returns store (on /Volumes/Z)
    market_returns, ohlcv = build_returns(con, log)
    crsp_dir = data_path("market", "crsp")
    crsp_dir.mkdir(parents=True, exist_ok=True)
    market_returns.to_parquet(crsp_dir / "market_returns.parquet", index=False)
    ohlcv.to_parquet(crsp_dir / "daily_returns.parquet", index=False)
    log.info("Wrote returns store", dir=str(crsp_dir))

    log.info("WRDS ingest complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
