"""T6-16 — How much of the test panel comes from a filer the model has already read.

Section 1.1 asserts that "the training years' filers are largely the test years'
filers too" and Section 2.1.5 that "the same firms sit on both sides of any
calendar split". Both are qualitative. The condition that makes the whole
identity thesis possible is asserted throughout and measured nowhere.

This script measures it, from the RELEASED PUBLIC INDEX only
(release/accession_index.csv). No licensed CRSP data is touched, so the number
is reproducible by anyone holding the public release.

UNIT DISCIPLINE. The index is keyed by CIK, while the report elsewhere counts
PERMNOs (832), tickers (884) and membership intervals (914). The counts below
are DISTINCT FILER CIKs and must be described as such, or they read as a fourth
inconsistent census of the same universe.

Run from repo root:  .venv/bin/python scripts/analysis/filer_overlap.py
Outputs (NEW files): results/tables/filer_overlap.{csv,md}
"""
import sys
from pathlib import Path

import pandas as pd

IX = "release/accession_index.csv"
OUT = Path("results/tables")

# Gate: the corpus census the report already prints.
EXPECT_TOTAL = 144129
EXPECT_TEST = 33144


def gate(cond, msg):
    if not cond:
        sys.exit(f"GATE FAILED: {msg}")


def main():
    ix = pd.read_csv(IX)
    gate(len(ix) == EXPECT_TOTAL,
         f"index has {len(ix)} rows, expected the committed {EXPECT_TOTAL}")
    train = set(ix.loc[ix.split == "train", "cik"].unique())
    te = ix[ix.split == "test"]
    gate(len(te) == EXPECT_TEST,
         f"test split has {len(te)} rows, expected {EXPECT_TEST}")

    te_ciks = set(te.cik.unique())
    seen = te_ciks & train
    rows = []
    rows.append(dict(stratum="all test filings",
                     n_filings=len(te), n_seen=int(te.cik.isin(train).sum()),
                     n_ciks=len(te_ciks), n_ciks_seen=len(seen)))
    for label, mask in [("8-K", te.form == "8-K"),
                        ("long-form (10-K and 10-Q)", te.form.isin(["10-K", "10-Q"]))]:
        s = te[mask]
        rows.append(dict(stratum=label, n_filings=len(s),
                         n_seen=int(s.cik.isin(train).sum()),
                         n_ciks=int(s.cik.nunique()),
                         n_ciks_seen=int(len(set(s.cik.unique()) & train))))
    df = pd.DataFrame(rows)
    df["pct_filings"] = 100 * df.n_seen / df.n_filings
    df["pct_ciks"] = 100 * df.n_ciks_seen / df.n_ciks

    gate(set(te.form.unique()) <= {"8-K", "10-K", "10-Q"},
         f"unexpected forms in the test split: {sorted(set(te.form.unique()))}")

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "filer_overlap.csv", index=False)
    with open(OUT / "filer_overlap.md", "w") as f:
        f.write("# T6-16 — Train/test filer overlap (released public index only)\n\n")
        f.write("Counts are DISTINCT FILER CIKs, not PERMNOs, tickers or membership "
                "intervals. Source: `release/accession_index.csv`.\n\n")
        f.write("| stratum | test filings | from an already-read filer | % | distinct CIKs | also in train | % |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for _, r in df.iterrows():
            f.write(f"| {r.stratum} | {r.n_filings:,} | {r.n_seen:,} | {r.pct_filings:.1f} | "
                    f"{r.n_ciks:,} | {r.n_ciks_seen:,} | {r.pct_ciks:.1f} |\n")
    a = df.iloc[0]
    print(f"{a.n_ciks_seen} of {a.n_ciks} test-era filer CIKs also file in train ({a.pct_ciks:.1f}%); "
          f"{a.n_seen:,} of {a.n_filings:,} test filings ({a.pct_filings:.1f}%)")
    print(f"wrote {OUT/'filer_overlap.md'} and .csv")


if __name__ == "__main__":
    main()
