"""Post-rerun verification for the 6 M&A-successor CIK backfills.

Read-only. Confirms: (1) each backfilled ticker recovers >0 filings under its
OLD CIK (and 0 under the wrong successor CIK); (2) the 6 firms appear in the
work_items ledger; (3) the net filings count rose by the expected band; (4) the
5 retryable SEC 503s cleared or stayed transient; (5) no PIT look-ahead leak
(effective_trading_day >= ET-date of filing_time_utc); (6) M&A CIK segmentation
for the 6 firms stays clean (one OLD CIK each, no successor leak).
"""

from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

DATA = Path("/path/to/data-root/sp500vol-data/processed/full")
STATE = DATA / "_state"
OLD_TOTAL = 144056
PREV_RUN_ID = "build_dataset_full_20260607T183847Z"
FILINGS_BAND = (144400, 145000)
ET = ZoneInfo("America/New_York")

# ticker -> (old_cik, wrong_successor_cik, member_from, member_to)
CASES = {
    "CVC": ("0001053112", "0001702780", "2010-12-20", "2016-06-21"),
    "DNB": ("0001115222", "0001799208", "2010-01-04", "2017-04-04"),
    "HFC": ("0000048039", "0001915657", "2018-06-18", "2021-06-03"),
    "JNS": ("0001065865", "0002043380", "2010-01-04", "2011-11-22"),
    "NE": ("0001458891", "0001895262", "2011-01-18", "2015-07-17"),
    "RX": ("0001058083", "0001595262", "2010-01-04", "2010-02-25"),
}
OLD_CIKS = {c[0] for c in CASES.values()}
RETRY_ACC = {
    "0001104659-11-036443": "CF",
    "0001171843-15-006581": "WFM",
    "0000072333-15-000182": "JWN",
    "0001041061-15-000039": "YUM",
    "0000029915-11-000044": "DOW",
}

failures: list[str] = []


def hr(title: str) -> None:
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


# 0. RUN META ---------------------------------------------------------------
meta = json.loads((DATA / "_meta.json").read_text())
hr("0) RUN META")
print("run_id      :", meta["run_id"], "(prev:", PREV_RUN_ID + ")")
print("git_sha     :", meta.get("git_sha"))
print("built_at_utc:", meta.get("built_at_utc"))
print("counts      :", meta["counts"])
new_total = int(meta["counts"]["filings"])
if meta["run_id"] == PREV_RUN_ID:
    failures.append("run_id unchanged -> build did not re-run / did not finish")

# 1 + 3. FILINGS PER TICKER UNDER OLD CIK; TOTAL COUNT ----------------------
hr("1) FILINGS PER TICKER UNDER OLD CIK  +  3) TOTAL FILINGS BAND")
fil = pd.read_parquet(DATA / "filings.parquet", columns=["ticker", "cik", "form", "accession"])
fil["cik"] = fil["cik"].astype(str).str.zfill(10)
fil["ticker"] = fil["ticker"].astype(str).str.upper()
contrib = 0
for tk, (old, wrong, _mf, _mt) in CASES.items():
    sub = fil[fil["ticker"] == tk]
    n_old = int((sub["cik"] == old).sum())
    n_wrong = int((sub["cik"] == wrong).sum())
    contrib += n_old
    byform = sub.loc[sub["cik"] == old, "form"].value_counts().to_dict()
    ok = n_old > 0 and n_wrong == 0
    if not ok:
        failures.append(f"{tk}: old_cik filings={n_old}, wrong_cik filings={n_wrong}")
    verdict = "OK" if ok else "FAIL"
    print(f"  {tk}: old {old} -> {n_old:>4} {byform};  wrong {wrong} -> {n_wrong}  [{verdict}]")
delta = new_total - OLD_TOTAL
in_band = FILINGS_BAND[0] <= new_total <= FILINGS_BAND[1]
print(f"\n  6-firm contribution under old CIKs = {contrib}")
print(f"  total filings: old={OLD_TOTAL}  new={new_total}  delta=+{delta}")
print(
    f"  expected total band {FILINGS_BAND[0]}-{FILINGS_BAND[1]}: "
    f"new total is {'IN BAND' if in_band else 'OUTSIDE BAND'}"
)
if not in_band:
    failures.append(f"filings total {new_total} outside expected band {FILINGS_BAND}")

# 2. LEDGER PRESENCE --------------------------------------------------------
hr("2) WORK_ITEMS LEDGER — 6 FIRMS ENQUEUED")
seen = {tk: 0 for tk in CASES}
retry_state: dict[str, dict] = {}
with (STATE / "work_items.jsonl").open() as fh:
    for line in fh:
        if not line.strip():
            continue
        r = json.loads(line)
        cik = str(r.get("cik") or "").zfill(10)
        tk = str(r.get("ticker") or "").upper()
        if tk in seen and cik in OLD_CIKS:
            seen[tk] += 1
        acc = r.get("accession")
        if acc in RETRY_ACC:
            retry_state[acc] = r  # append-only -> last line wins
for tk in CASES:
    ok = seen[tk] > 0
    if not ok:
        failures.append(f"{tk} absent from work_items.jsonl")
    print(f"  {tk}: {seen[tk]:>4} ledger lines under old CIK  [{'OK' if ok else 'FAIL'}]")

# 4. RETRYABLE SEC 503s -----------------------------------------------------
hr("4) RETRYABLE SEC 503s — CLEARED OR STILL TRANSIENT")
filed_accessions = set(fil["accession"])
for acc, tk in RETRY_ACC.items():
    cleared = acc in filed_accessions
    st = retry_state.get(acc, {}).get("status")
    run = retry_state.get(acc, {}).get("status") and retry_state.get(acc, {}).get("run_id")
    verdict = (
        "CLEARED (in filings.parquet)"
        if cleared
        else f"still failed/transient (ledger status={st})"
    )
    print(f"  {acc} ({tk}): ledger status={st} run={run} -> {verdict}")
# transient-only is acceptable; a hard non-retryable failure is not
fl = STATE / "failure_log.jsonl"
hard = []
if fl.exists():
    for line in fl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("accession") in RETRY_ACC and r.get("retryable") is False:
            hard.append(r.get("accession"))
if hard:
    failures.append(f"non-retryable failures among the 5 503s: {hard}")

# 5. PIT NO-LOOK-AHEAD ------------------------------------------------------
hr("5) PIT NO-LOOK-AHEAD  (effective_trading_day < ET-date(filing_time_utc) must be 0)")
al = pd.read_parquet(
    DATA / "aligned_filings.parquet",
    columns=["ticker", "cik", "filing_time_utc", "effective_trading_day"],
)
ft = pd.to_datetime(al["filing_time_utc"], utc=True)
et_date = ft.dt.tz_convert(ET).dt.normalize().dt.tz_localize(None)
eff = pd.to_datetime(al["effective_trading_day"]).dt.tz_localize(None).dt.normalize()
viol = eff < et_date
n_viol = int(viol.sum())
print(f"  total aligned rows = {len(al)};  look-ahead violations = {n_viol}")
if n_viol:
    failures.append(f"PIT look-ahead violations: {n_viol}")
m6 = al["ticker"].astype(str).str.upper().isin(CASES)
print(f"  6-firm aligned rows = {int(m6.sum())};  violations among them = {int((viol & m6).sum())}")

# 6. M&A CIK SEGMENTATION ---------------------------------------------------
hr("6) M&A CIK SEGMENTATION CLEAN FOR 6 FIRMS")
uni = pd.read_parquet(DATA / "universe.parquet")
uni["ticker"] = uni["ticker"].astype(str).str.upper()
uni["cik"] = uni["cik"].astype(str).str.zfill(10)
for tk, (old, _wrong, _mf, _mt) in CASES.items():
    rows = uni[uni["ticker"] == tk].sort_values("member_from")
    ciks = list(rows["cik"].unique())
    ok = ciks == [old]
    if not ok:
        failures.append(f"{tk} universe CIKs={ciks} (expected [{old}])")
    win = [
        f"{pd.Timestamp(a).date()}..{'open' if pd.isna(b) else pd.Timestamp(b).date()}"
        for a, b in zip(rows["member_from"], rows["member_to"], strict=True)
    ]
    print(f"  {tk}: CIKs={ciks} rows={len(rows)} windows={win}  [{'OK' if ok else 'FAIL'}]")

# SUMMARY -------------------------------------------------------------------
hr("SUMMARY")
if failures:
    print("RESULT: FAIL")
    for f in failures:
        print("  -", f)
    raise SystemExit(1)
print("RESULT: PASS — all post-rerun checks green")
