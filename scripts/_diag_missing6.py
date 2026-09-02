"""Diagnostic: trace the 6 missing M&A-successor tickers through CCM + membership."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

REPO = Path(".")
WRDS = Path("/path/to/data-root/sp500vol-data/raw/wrds")
DATA = Path("/path/to/data-root/sp500vol-data/processed/full")

TARGETS = {
    "CVC": "0001053112",
    "DNB": "0001115222",
    "HFC": "0000048039",
    "JNS": "0001065865",
    "NE": "0001458891",
    "RX": "0001058083",
}


def read_zip_csv(zip_path: Path, **kw) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        inner = zf.namelist()[0]
        with zf.open(inner) as fh:
            return pd.read_csv(fh, **kw)


# 1. Membership table: what do these tickers map to now?
mem = pd.read_parquet(REPO / "data/universe/sp500_membership.parquet")
print("=== MEMBERSHIP rows for 6 tickers ===")
print(mem[mem["ticker"].isin(TARGETS)].sort_values(["ticker", "member_from"]).to_string())

# 2. Find PERMNOs from constituents file for these tickers
con = read_zip_csv(
    WRDS / "sp500_constituents_2010_2025_csv.zip",
    usecols=["PERMNO", "Ticker", "MbrStartDt", "MbrEndDt", "DlyCalDt"],
    dtype=str,
)
con["ticker"] = con["Ticker"].str.strip().str.upper()
con = con[con["ticker"].isin(TARGETS)]
con["DlyCalDt"] = pd.to_datetime(con["DlyCalDt"], errors="coerce")
perm = (
    con.groupby(["ticker", "PERMNO"])
    .agg(first=("DlyCalDt", "min"), last=("DlyCalDt", "max"), days=("DlyCalDt", "count"))
    .reset_index()
)
print("\n=== CONSTITUENTS permno/window per ticker ===")
print(perm.to_string())

permnos = sorted({int(p) for p in perm["PERMNO"].dropna()})
print("\nPERMNOs:", permnos)

# 3. CCM rows for these PERMNOs (what CIK does CCM assign?)
ccm = read_zip_csv(
    WRDS / "ccm_csv.zip",
    usecols=["GVKEY", "LPERMNO", "cik", "LINKDT", "LINKENDDT", "LINKTYPE", "LINKPRIM"],
    dtype=str,
)
ccm["LPERMNO_i"] = pd.to_numeric(ccm["LPERMNO"], errors="coerce")
sub = ccm[ccm["LPERMNO_i"].isin(permnos)]
print("\n=== CCM rows for these PERMNOs (ALL link types) ===")
print(sub.sort_values(["LPERMNO_i", "LINKDT"]).to_string())

# 4. Does the target old CIK appear anywhere in CCM at all?
print("\n=== Does target OLD CIK appear in CCM (any permno)? ===")
for tk, cik in TARGETS.items():
    raw_cik = cik.lstrip("0")
    hits = ccm[ccm["cik"].astype(str).str.lstrip("0") == raw_cik]
    permnos_hit = sorted(set(hits["LPERMNO_i"].dropna().astype(int)))
    print(f"{tk} old_cik={cik}: {len(hits)} CCM rows; permnos={permnos_hit}")
