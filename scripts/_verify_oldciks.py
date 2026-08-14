"""Verify the 6 old CIKs against live EDGAR: company name + form counts in window."""
# ruff: noqa: E402, I001

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(".")
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("EDGAR_USER_AGENT", "SP500Vol-Bench research <set EDGAR_USER_AGENT>")

import aiohttp

from sp500vol.data.edgar_fetcher import EdgarClient, SUBMISSIONS_BASE

FORMS = ["10-K", "10-Q", "8-K"]

# ticker -> (old_cik, successor_cik, member_from, member_to)
CASES = {
    "CVC": ("0001053112", "0001702780", "2010-12-20", "2016-06-21"),
    "DNB": ("0001115222", "0001799208", "2010-01-04", "2017-04-04"),
    "HFC": ("0000048039", "0001915657", "2018-06-18", "2021-06-03"),
    "JNS": ("0001065865", "0002043380", "2010-01-04", "2011-11-22"),
    "NE": ("0001458891", "0001895262", "2011-01-18", "2015-07-17"),
    "RX": ("0001058083", "0001595262", "2010-01-04", "2010-02-25"),
}


async def name_for(session, cik):
    url = SUBMISSIONS_BASE.format(cik=int(cik))
    async with session.get(url) as r:
        if r.status != 200:
            return f"<HTTP {r.status}>", []
        j = await r.json()
    former = [f.get("name") for f in j.get("formerNames", [])]
    return j.get("name", "?"), former


async def main():
    headers = {"User-Agent": os.environ["EDGAR_USER_AGENT"], "Accept-Encoding": "gzip, deflate"}
    async with (
        EdgarClient(cache_root=REPO / "data/raw", rate_limit_per_sec=8) as client,
        aiohttp.ClientSession(headers=headers) as session,
    ):
        for tk, (old, succ, mf, mt) in CASES.items():
            mf_ts, mt_ts = pd.Timestamp(mf), pd.Timestamp(mt)
            print(f"\n########## {tk}  member {mf}..{mt} ##########")
            for label, cik in [("OLD", old), ("SUCCESSOR", succ)]:
                name, former = await name_for(session, cik)
                try:
                    metas = await client.list_filings(cik, FORMS)
                except Exception as e:
                    print(f"  {label} {cik} {name!r}: list error {e}")
                    continue
                df = pd.DataFrame(
                    [{"form": m.form, "date": pd.Timestamp(m.filing_date)} for m in metas]
                )
                if df.empty:
                    print(f"  {label} {cik} {name!r}: 0 filings total")
                    continue
                inwin = df[(df["date"] >= mf_ts) & (df["date"] <= mt_ts)]
                by_form = inwin["form"].value_counts().to_dict()
                print(f"  {label} {cik} name={name!r}")
                if former:
                    print(f"       formerNames={former}")
                date_min = df["date"].min().date()
                date_max = df["date"].max().date()
                print(f"       total_filings={len(df)} range={date_min}..{date_max}")
                print(f"       IN-WINDOW {mf}..{mt}: total={len(inwin)} by_form={by_form}")


asyncio.run(main())
