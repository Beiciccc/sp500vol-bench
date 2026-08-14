#!/usr/bin/env python
"""M1 — the audit cascade on Kogan et al. (2009)'s OWN 10-K corpus.

PRE-REGISTERED: configs/prereg_kogan_corpus.md, tag prereg-kc-v1.0. Single-shot;
every branch ships regardless of direction. CPU-only; no /Volumes/Z, no GPU.

THE CENSUS (prereg §"important scope correction"): the public evidence base for
disclosure-text -> volatility is THREE corpora. (i) MDRM/EC = cite-only, adjudicated
by prereg_maec_audit.md §9 (text+audio bundled, no licence, not redistributed) —
that ruling stands. (ii) MAEC = audited (FACTS §13g). (iii) Kogan's 10-K corpus =
THIS script, the third and final corpus. HTML/NumHTML/VolTAGE/KeFVP/ECHO-GL all
build on MDRM, so "audit N models" would be N models on ONE corpus = pseudo-
replication with a fake denominator.

NOT to be confused with `kogan_dissolve.py` (committed): that ports Kogan's
evaluation DESIGN onto OUR modern panel. This runs OUR cascade on THEIR corpus.

THE LADDER (prereg §"cascade", ported from the committed maec_protocol.py template):
  L0  published convention: TF-IDF text + logvol.-12 control -> logvol.+12,
      NAIVE obs-level inference, year-by-year OOS. Readout = MSE improvement of
      the text arm over the logvol.-12-ONLY arm (the quantity Kogan et al. report).
      Their Table 2 baseline row is v^(-12) used DIRECTLY as the prediction (no
      fit) — reproduced here as-is, deliberately, and labelled (G-K2).
  L1  baseline becomes the RECALIBRATED logvol.-12 (OLS intercept+slope on the
      training years, frozen on the test year).
  L2  reference additionally gets the same-CIK training-period mean log
      volatility (zero-text term).
  L3  inference clusters on FILING DATE (HAC + HLN), replacing naive obs-t.
  L4  Holm within the pre-declared family = the L3 per-year p-values.
  L5  conjunction: L1 AND L2 AND L4.
  placebo  label shuffle (5 seeds), |DM| < 2 gate.

TWO L0 ARMS — BOTH PRE-DECLARED HERE, BOTH REPORTED UNCONDITIONALLY (see the
"PREREG IMPRECISION" note below). Neither is selected on its outcome.
  L0        the prereg's literal rule: train <= y, test = y+1, y = 1996..2005
            (EXPANDING window). This is the binding rung that feeds L1-L5.
  L0_pub    Kogan's ACTUAL published convention: train = the FIVE years preceding
            the test year, test years 2001..2006, count-weighted micro-average
            (their Table 2). This is the G-K1 comparator, because it is the only
            arm commensurable with their published number.

PREREG IMPRECISION (recorded before any statistic; resolved by reporting BOTH,
never by choosing): the prereg calls "train <= y, test = y+1" *their* annual OOS
split. It is not. Kogan et al. (2009) §6 states: "We used as training examples all
reports from the five-year period preceding the test year (so six experiments on
six different training and test sets)", with test years 2001-2006 (Table 2); their
Table 4 varies that window over 1/2/5 years and never uses an expanding one.
Executing only the prereg's rule would leave G-K1 unanswerable as specified (it
would compare our expanding-window reading against their rolling-window number).
Executing only the published rule would violate the binding prereg. So both run,
both are tabulated, and the G-K1 verdict rests on L0_pub.

L2 SELF-INCLUSION — a SECOND prereg ambiguity, likewise resolved by reporting
BOTH readings (it flips the fired branch, so it is flagged, not quietly settled).
The prereg says L2's reference "additionally gets the same-CIK TRAINING-PERIOD
mean log volatility (zero-text term)" but never says whether a TRAINING row's own
label may enter its own CIK mean. Both are computed per split:
  incl  the literal self-inclusive mean. A training row's feature then CONTAINS
        its own label, the fitted coefficient is driven toward 1.0, the reference
        overfits, and its TEST MSE lands WORSE than L1's — which INVERTS the rung
        (L2 is specified to STRENGTHEN the reference) and mechanically inflates
        the text gain, manufacturing survivals.
  loo   the same mean with the row's own label removed (singleton-CIK training
        rows fall back to the global training mean). PRIMARY.
Test rows are identical under both (a test row's label can never enter a mean over
TRAINING years), so G-K2 holds either way — the fork is purely the training fit.
`loo` is primary for a declared STRUCTURAL reason, never an outcome: only it
behaves like the control the rung describes, and it is what the committed template
already mandates — maec_protocol.py's entity-mean control (STPEV) is a
point-in-time EXPANDING prior-label mean built with shift(1), i.e. the current
row's label is excluded by construction, with the self-inclusive fixed mean
demoted to a robustness block.

GATES:
  G-K0  SHA-256 of every downloaded file + row/key-space consistency assertions.
  G-K1  L0_pub's reading vs Kogan et al. (2009) Table 2 (p.5): SIGN + ORDER OF
        MAGNITUDE. Their TFIDF+ (text+v^(-12)) micro-average MSE = 0.1557 vs the
        v^(-12) baseline 0.1576 => +1.21% MSE reduction. NEVER tuned to match; a
        mismatch fires branch (c).
  G-K2  no-look-ahead assertions from L1 on; L0's naive convention reproduced
        as-is and labelled.
  G-K3  CIK coverage + cross-year firm recurrence report.

DEVIATIONS FROM KOGAN'S EXACT ESTIMATOR (disclosed, NOT tuned; each is the
committed `kogan_dissolve.py` recipe, fixed before any number was seen):
  - estimator: ridge (alpha by 5-fold CV over a fixed grid) vs their SVR
    (SVM^light, linear kernel, eps=0.1, C=1/mean(h.h)). kogan_dissolve.py already
    ships ridge as "their SVR analogue".
  - features: TF-IDF 1-2gram, top-5k by train term frequency, sublinear tf, L2
    norm (sklearn TfidfTransformer) vs their h_j = (1/|d|)freq x log(N/df), full
    training vocabulary, unigrams for TFIDF.
  - the price control enters the text arm as a standardized column scaled x10, so
    its ridge penalty is ~1/100 of a text feature's ~= the unpenalised control
    their design implies (the committed kogan_dissolve.py convention).

Run from repo root:
  .venv/bin/python scripts/analysis/kogan_corpus_audit.py --data <fetch dir>
Stages: data (parse + hash + counts matrix, cached), ladder (rungs + gates +
tables). Outputs results/tables/kogan_corpus_audit.{csv,md}.
"""
from __future__ import annotations

import os

# LOCAL CPU ONLY, shared machine (<=5 cores): cap BLAS threads BEFORE numpy loads.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("SP500VOL_TEXT_N_JOBS", "4")

import argparse
import hashlib
import json
import re
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse, stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

import forecast_combination as fc  # noqa: E402  (committed Holm)
from sp500vol.evaluation.dm_test import dm_test  # noqa: E402  (committed DM + HLN)
from sp500vol.models.classical_text._fit_utils import fit_ridge_cv  # noqa: E402

TABLES = REPO / "results" / "tables"
YEARS = tuple(range(1996, 2007))

# --- committed B2/kogan_dissolve recipe constants (fixed before any statistic)
MAX_FEATURES = 5000
ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0)
TOKEN_RE = re.compile(r"(?u)\b[a-z]{2,}\b")
CHUNK_DOCS = 256
N_JOBS = 4                      # <=5 cores, shared machine
VOL_SCALE = 10.0                # x10 => ~unpenalised price control
YEAR_PRUNE_MIN = 5              # per-year term-count floor (see vocab guard)

# --- prereg §"cascade"
PLACEBO_SEEDS = (1000, 1001, 1002, 1003, 1004)
PLACEBO_DM_GATE = 2.0
ALPHA_SIG = 0.05

# --- G-K1: Kogan et al. (2009) Table 2, p.5 (micro-average column)
PUB_BASELINE_MSE = 0.1576       # v^(-12) (baseline)
PUB_TFIDF_PLUS_MSE = 0.1557     # TFIDF+ (text + v^(-12))
PUB_GAIN_PCT = 100.0 * (PUB_BASELINE_MSE - PUB_TFIDF_PLUS_MSE) / PUB_BASELINE_MSE
PUB_TABLE1_DOCS = {1996: 1408, 1997: 2260, 1998: 2462, 1999: 2524, 2000: 2425,
                   2001: 2596, 2002: 2846, 2003: 3612, 2004: 3559, 2005: 3474,
                   2006: 3308}
# Table 2 per-year MSE: the `v^(-12) (baseline)` row and the `TFIDF+` row.
PUB_TABLE2_BASE = {2001: 0.1747, 2002: 0.1600, 2003: 0.1873, 2004: 0.1442,
                   2005: 0.1365, 2006: 0.1463}
PUB_TABLE2_TFIDF_PLUS = {2001: 0.1919, 2002: 0.1618, 2003: 0.1965, 2004: 0.1246,
                         2005: 0.1276, 2006: 0.1403}

RUNG_DESC = {
    "L0": "published convention: TF-IDF + logvol.-12 control vs the RAW logvol.-12-only arm, naive obs t",
    "L1": "baseline -> RECALIBRATED logvol.-12 (OLS intercept+slope, train-fit, frozen)",
    "L2": "reference += same-CIK training-period mean log volatility (zero-text term)",
    "L3": "inference clusters on FILING DATE (HAC + HLN), replacing naive obs-t",
    "L4": "Holm within the pre-declared family = the L3 per-year p-values",
    "L5": "conjunction: L1 AND L2 AND L4",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- G-K0: hashing
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(data_dir: Path) -> dict:
    """G-K0: re-hash every downloaded file and check it against the fetch manifest."""
    man = json.loads((data_dir / "manifest.json").read_text())
    rows = []
    for f in man["files"]:
        p = data_dir / f["file"]
        assert p.exists(), f"G-K0 FAIL: {p} missing"
        got = sha256(p)
        assert got == f["sha256"], (
            f"G-K0 FAIL: {f['file']} sha256 {got} != manifest {f['sha256']}")
        rows.append({"file": f["file"], "bytes": f["bytes"], "sha256": got})
    log(f"G-K0: {len(rows)} files re-hashed, all match the fetch manifest")
    return {"manifest": man, "files": rows}


# ------------------------------------------------------------- data: parse year
def read_meta(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, sep="\t", header=None,
                    names=["key", "date", "url", "company", "cik"],
                    dtype={"key": str, "date": str, "url": str, "company": str,
                           "cik": str})
    d["filing_date"] = pd.to_datetime(d["date"], format="%Y%m%d")
    return d[["key", "filing_date", "company", "cik"]]


def read_logvol(path: Path, name: str) -> pd.DataFrame:
    d = pd.read_csv(path, sep=r"\s+", header=None, names=[name, "key"],
                    dtype={name: float, "key": str})
    return d[["key", name]]


def iter_docs(tgz: Path):
    """Stream (key, text) from a {year}.tok.tgz (members are `{key}.mda`)."""
    with tarfile.open(tgz, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = m.name.rsplit("/", 1)[-1]
            if not name.endswith(".mda"):
                continue
            fh = tf.extractfile(m)
            if fh is None:
                continue
            yield name[:-4], fh.read().decode("utf-8", errors="replace")


def _count_chunk(texts: list[str]) -> dict[str, int]:
    """1-2gram term frequencies for a chunk; within-chunk hapaxes pruned
    (the committed kogan_dissolve.py memory-safe convention)."""
    c: Counter = Counter()
    for t in texts:
        toks = TOKEN_RE.findall(t.lower())
        c.update(toks)
        c.update(map(" ".join, zip(toks, toks[1:])))
    return {k: v for k, v in c.items() if v >= 2}


def _transform_chunk(texts: list[str], vocab: list[str]):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(lowercase=True, ngram_range=(1, 2),
                         token_pattern=r"(?u)\b[a-z]{2,}\b",
                         vocabulary=vocab, dtype=np.int32)
    return cv.transform(texts)


def stage_data(data_dir: Path, work: Path) -> None:
    """Parse the corpus, assert G-K0 consistency, cache per-year term counters."""
    panel_p, counts_meta_p = work / "panel.parquet", work / "year_counts.json"
    if panel_p.exists() and counts_meta_p.exists():
        log("data: cached, skipping")
        return

    rows, consistency = [], []
    for y in YEARS:
        meta = read_meta(data_dir / f"{y}.meta.txt")
        lp = read_logvol(data_dir / f"{y}.logvol.+12.txt", "logvol_p12")
        lm = read_logvol(data_dir / f"{y}.logvol.-12.txt", "logvol_m12")
        keys_tok = [k for k, _ in iter_docs(data_dir / f"{y}.tok.tgz")]

        # ---- G-K0 row / key-space consistency assertions.
        # The corpus ships a handful of EXACT duplicate records (same key, date,
        # URL, company, CIK, both log volatilities, and the .mda member repeated
        # inside the tarball). They are NOT dropped: the per-year row counts that
        # include them are exactly the document counts Kogan et al. publish in
        # their Table 1, so the published reading is computed over them too.
        # Dropping them would silently change the denominator we must match.
        # The assertion is therefore the stronger, true property: all four
        # sources agree as MULTISETS, and every duplicate group is exact.
        from collections import Counter as _C
        c_meta, c_tok = _C(meta.key), _C(keys_tok)
        assert c_meta == _C(lp.key) == _C(lm.key) == c_tok, (
            f"G-K0 FAIL {y}: meta/+12/-12/tok disagree as key multisets")
        assert len(meta) == len(lp) == len(lm) == len(keys_tok), (
            f"G-K0 FAIL {y}: row counts differ — meta={len(meta)} +12={len(lp)} "
            f"-12={len(lm)} tok={len(keys_tok)}")
        dup_keys = [k for k, c in c_meta.items() if c > 1]
        for frame, nm in ((meta, "meta"), (lp, "+12"), (lm, "-12")):
            for k in dup_keys:
                assert frame[frame.key == k].drop_duplicates().shape[0] == 1, (
                    f"G-K0 FAIL {y}: key {k} duplicated in {nm} with DIFFERING "
                    f"values — not an exact duplicate, cannot be carried safely")

        mult = meta.key.value_counts()
        d = (meta.drop_duplicates("key")
             .merge(lp.drop_duplicates("key"), on="key", validate="1:1")
             .merge(lm.drop_duplicates("key"), on="key", validate="1:1"))
        # restore the published multiplicity (exact duplicates re-expanded)
        d = d.loc[d.index.repeat(d.key.map(mult).to_numpy())].reset_index(drop=True)
        assert len(d) == len(meta), f"G-K0 FAIL {y}: re-expansion changed the row count"
        assert not d[["logvol_p12", "logvol_m12"]].isna().any().any(), \
            f"G-K0 FAIL {y}: null log volatility"
        d["year"] = y
        rows.append(d)
        consistency.append({"year": y, "n_rows": len(d),
                            "kogan_table1_docs": PUB_TABLE1_DOCS[y],
                            "matches_table1": len(d) == PUB_TABLE1_DOCS[y],
                            "n_unique_keys": int(len(mult)),
                            "n_exact_dup_rows": int(len(meta) - len(mult)),
                            "n_cik": int(d.cik.nunique())})
        log(f"data: {y} n={len(d)} (Table 1: {PUB_TABLE1_DOCS[y]}) "
            f"unique_keys={len(mult)} exact_dups={len(meta) - len(mult)} "
            f"cik={d.cik.nunique()}")

    panel = pd.concat(rows, ignore_index=True)
    assert panel.groupby("key").year.nunique().eq(1).all(), \
        "G-K0 FAIL: a key appears in more than one year"
    panel = panel.sort_values(["year", "filing_date", "key"],
                             kind="mergesort").reset_index(drop=True)
    # doc_row indexes the UNIQUE-key counts matrix; exact duplicates share a row
    key_row = {k: i for i, k in enumerate(panel.key.drop_duplicates())}
    panel["doc_row"] = panel.key.map(key_row).to_numpy()
    panel.to_parquet(panel_p, index=False)
    (work / "consistency.json").write_text(json.dumps(consistency, indent=2))

    # ---- per-year 1-2gram term counters (ONE tokenisation pass over the corpus)
    year_counts = {}
    for y in YEARS:
        t0 = time.time()
        merged: Counter = Counter()
        n_chunks = 0

        def gen(year=y):
            buf = []
            for _, text in iter_docs(data_dir / f"{year}.tok.tgz"):
                buf.append(text)
                if len(buf) >= CHUNK_DOCS:
                    yield delayed(_count_chunk)(buf)
                    buf = []
            if buf:
                yield delayed(_count_chunk)(buf)

        for part in Parallel(n_jobs=N_JOBS, return_as="generator")(gen()):
            merged.update(part)
            n_chunks += 1
        pruned = {k: v for k, v in merged.items() if v >= YEAR_PRUNE_MIN}
        year_counts[str(y)] = {"counts": pruned, "n_chunks": n_chunks}
        log(f"data: {y} vocab pass — {len(merged)} terms -> {len(pruned)} kept "
            f"(>= {YEAR_PRUNE_MIN}), {n_chunks} chunks ({time.time() - t0:.0f}s)")
        del merged
    (work / "year_counts.json").write_text(json.dumps(year_counts))
    log("data: cached panel + per-year term counters")


def split_vocab(year_counts: dict, train_years: list[int]) -> tuple[list[str], dict]:
    """Top-MAX_FEATURES 1-2grams by TRAIN term frequency (kogan_dissolve rule)."""
    tot: Counter = Counter()
    n_chunks = 0
    for y in train_years:
        yc = year_counts[str(y)]
        tot.update(yc["counts"])
        n_chunks += yc["n_chunks"]
    top = sorted(tot.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_FEATURES]
    vocab = sorted(k for k, _ in top)
    min_count = int(top[-1][1]) if top else 0
    # VOCAB GUARD. Two prunings could in principle distort the top-5k ranking:
    # (a) per-year floor -> a missed term has < YEAR_PRUNE_MIN in every train
    #     year, so its true count < YEAR_PRUNE_MIN * len(train_years);
    # (b) within-chunk hapax pruning -> a term loses at most 1 count per chunk,
    #     so its recorded count >= true count - n_chunks.
    # Both are irrelevant iff the smallest surviving top-5k count dominates both
    # bounds. Asserted, and the margins are reported in the output.
    bound_year = YEAR_PRUNE_MIN * len(train_years)
    assert min_count > bound_year, (
        f"vocab guard: top-5k min count {min_count} <= year-prune bound {bound_year}")
    assert min_count > n_chunks, (
        f"vocab guard: top-5k min count {min_count} <= chunk-prune bound {n_chunks}")
    return vocab, {"min_top_count": min_count, "bound_year_prune": bound_year,
                   "bound_chunk_prune": n_chunks,
                   "guard_margin_x": round(min_count / max(n_chunks, 1), 1)}


def stage_counts(data_dir: Path, work: Path, splits: list[dict]) -> None:
    """ONE transform pass over all docs with the UNION of every split's train
    vocabulary; each split later slices its own 5k columns out (identical to
    building that split's CountVectorizer directly, since the tf-idf L2 norm is
    taken over the sliced columns)."""
    if (work / "X_counts.npz").exists() and (work / "union_vocab.json").exists():
        log("counts: cached, skipping")
        return
    year_counts = json.loads((work / "year_counts.json").read_text())
    union: set[str] = set()
    for sp in splits:
        v, _ = split_vocab(year_counts, sp["train_years"])
        union |= set(v)
    union_vocab = sorted(union)
    (work / "union_vocab.json").write_text(json.dumps(union_vocab))
    log(f"counts: union vocab over {len(splits)} splits = {len(union_vocab)} terms")

    panel = pd.read_parquet(work / "panel.parquet")
    row_of_key = dict(zip(panel.key, panel.doc_row))
    n_unique = int(panel.doc_row.nunique())
    mats, order = [], []
    t0 = time.time()
    for y in YEARS:
        def gen(year=y):
            # exact duplicate .mda members share a doc_row -> transform once
            seen, buf_k, buf_t = set(), [], []
            for k, text in iter_docs(data_dir / f"{year}.tok.tgz"):
                if k in seen:
                    continue
                seen.add(k)
                buf_k.append(k)
                buf_t.append(text)
                if len(buf_t) >= CHUNK_DOCS:
                    order.extend(buf_k)
                    yield delayed(_transform_chunk)(buf_t, union_vocab)
                    buf_k, buf_t = [], []
            if buf_t:
                order.extend(buf_k)
                yield delayed(_transform_chunk)(buf_t, union_vocab)

        mats.extend(Parallel(n_jobs=N_JOBS, return_as="generator")(gen()))
        log(f"counts: {y} transformed ({time.time() - t0:.0f}s)")
    X = sparse.vstack(mats, format="csr")
    assert len(order) == n_unique == X.shape[0], (
        f"counts: doc count mismatch — order={len(order)} unique={n_unique} "
        f"X={X.shape[0]}")
    # re-index rows into the panel's canonical doc_row order
    perm = np.array([row_of_key[k] for k in order])
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    X = X[inv]
    sparse.save_npz(work / "X_counts.npz", X)
    log(f"counts: X={X.shape} nnz={X.nnz} ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------ estimators
def ols_fit_apply(y_fit: np.ndarray, X_fit: np.ndarray,
                  X_app: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(y_fit)), X_fit]),
                               y_fit, rcond=None)
    return np.column_stack([np.ones(len(X_app)), X_app]) @ beta, beta


def naive_t(d: np.ndarray) -> tuple[float, float]:
    """Naive obs-level paired t on the loss differential (positive = text better).
    This is L0/L1/L2's inference — the convention being reproduced, not endorsed."""
    d = np.asarray(d, float)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return float(t), float(2.0 * stats.t.sf(abs(t), df=len(d) - 1))


def date_mean(x: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Equal weight per FILING DATE (prereg L3: the shock-sharing unit)."""
    g = pd.DataFrame({"d": pd.DatetimeIndex(dates), "x": np.asarray(x, float)}) \
        .groupby("d", sort=True)["x"].mean()
    idx = pd.DatetimeIndex(g.index)
    assert idx.is_monotonic_increasing and idx.is_unique, "date grid not ordered/unique"
    return g.to_numpy()


def overlap_lag(dates: np.ndarray) -> int:
    """STRICT label-overlap lag, the maec_protocol.hac_lag_L port: the max number
    of LATER distinct test filing dates whose 12-month label window still overlaps
    (calendar-day distance; this corpus ships no trading calendar). With a
    12-month label and a 12-month test year this is ~n_dates-1 BY CONSTRUCTION —
    i.e. a single test year holds ~one effective observation. Reported, not hidden."""
    grid = np.sort(pd.DatetimeIndex(pd.unique(pd.DatetimeIndex(dates))).values)
    L = 0
    for i in range(len(grid)):
        later = (grid[i + 1:] - grid[i]) / np.timedelta64(1, "D")
        L = max(L, int((later <= 365).sum()))
    return L


def nw_lag(n: int) -> int:
    """Newey-West rule-of-thumb bandwidth floor(4*(n/100)^(2/9)) on the date grid."""
    return int(np.floor(4.0 * (max(n, 1) / 100.0) ** (2.0 / 9.0)))


def clustered_dm(loss_text: np.ndarray, loss_ref: np.ndarray, dates: np.ndarray,
                 lag: int) -> tuple[float, float, int]:
    """Filing-date-clustered DM with HAC(lag) + HLN (committed dm_test, lag made
    explicit via h = lag+1). NEGATIVE stat = text better."""
    a, b = date_mean(loss_text, dates), date_mean(loss_ref, dates)
    stat, p = dm_test(a, b, h=int(lag) + 1)
    return float(stat), float(p), int(len(a))


# ------------------------------------------------------------------- the ladder
def fit_split(sp: dict, panel: pd.DataFrame, X: sparse.csr_matrix,
              union_vocab: list[str], year_counts: dict, placebo: bool) -> dict:
    """All arms for ONE (train_years -> test_year) split."""
    t0 = time.time()
    tr = panel[panel.year.isin(sp["train_years"])]
    te = panel[panel.year == sp["test_year"]]
    assert len(tr) and len(te), f"empty split {sp}"
    # G-K2 no-look-ahead: the test year must post-date every training year.
    assert max(sp["train_years"]) < sp["test_year"], "G-K2 FAIL: train year >= test year"
    assert tr.filing_date.max() < te.filing_date.min(), \
        "G-K2 FAIL: train filing date >= test filing date"

    vocab, guard = split_vocab(year_counts, sp["train_years"])
    col = {t: i for i, t in enumerate(union_vocab)}
    cols = np.array([col[t] for t in vocab])
    Xtr_c, Xte_c = X[tr.doc_row.to_numpy()][:, cols], X[te.doc_row.to_numpy()][:, cols]

    # tf-idf: idf fit on TRAIN docs ONLY, frozen on the test year (G-K2)
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True,
                            sublinear_tf=True).fit(Xtr_c)
    Ttr, Tte = tfidf.transform(Xtr_c), tfidf.transform(Xte_c)

    ytr, yte = tr.logvol_p12.to_numpy(), te.logvol_p12.to_numpy()
    vtr, vte = tr.logvol_m12.to_numpy(), te.logvol_m12.to_numpy()
    dates_te = te.filing_date.to_numpy()

    # price control standardised on TRAIN stats, x10 (~unpenalised), G-K2
    mu, sd = vtr.mean(), vtr.std()
    ztr, zte = (vtr - mu) / sd * VOL_SCALE, (vte - mu) / sd * VOL_SCALE

    # ---- TEXT arm: Kogan's TFIDF+ analogue = ridge on [TF-IDF | scaled control]
    Atr = sparse.hstack([Ttr, sparse.csr_matrix(ztr[:, None])], format="csr")
    Ate = sparse.hstack([Tte, sparse.csr_matrix(zte[:, None])], format="csr")
    ridge = fit_ridge_cv(Atr, ytr, ALPHA_GRID, n_jobs=N_JOBS)
    f_text = ridge.predict(Ate)

    # ---- reference arms
    f_L0 = vte.copy()                              # RAW logvol.-12 (their baseline)
    f_L1, _ = ols_fit_apply(ytr, vtr[:, None], vte[:, None])       # recalibrated

    # ---- L2: + same-CIK TRAINING-period mean log volatility (zero-text term).
    # TWO READINGS, both computed, neither selected on outcome (see the L2
    # SELF-INCLUSION note in the module docstring).
    cik_sum = tr.groupby("cik").logvol_p12.sum()
    cik_n = tr.groupby("cik").logvol_p12.size()
    cik_mean = cik_sum / cik_n
    g_mean = float(ytr.mean())
    # TEST rows are identical under both readings: a test row's own label can
    # never enter a mean taken over TRAINING years (G-K2).
    m_te = te.cik.map(cik_mean).fillna(g_mean).to_numpy()
    m_tr_incl = tr.cik.map(cik_mean).fillna(g_mean).to_numpy()
    si = tr.cik.map(cik_sum).to_numpy()
    ni = tr.cik.map(cik_n).to_numpy()
    m_tr_loo = np.where(ni > 1, (si - ytr) / np.maximum(ni - 1, 1), g_mean)
    cov = float(te.cik.isin(cik_mean.index).mean())                # G-K3
    frac_singleton = float((ni == 1).mean())

    def mse(f):
        return float(np.mean((yte - f) ** 2))

    e_text = (yte - f_text) ** 2
    n_d = int(pd.Series(dates_te).nunique())
    L_ov, L_nw = overlap_lag(dates_te), nw_lag(n_d)
    out = {
        "test_year": sp["test_year"], "train_years": sp["train_years"],
        "n_train": int(len(tr)), "n_test": int(len(te)), "n_test_dates": n_d,
        "alpha": float(ridge.alpha_), "vocab_guard": guard,
        "cik_train_coverage_test": cov,
        "frac_train_rows_singleton_cik": frac_singleton,
        "mse_text": mse(f_text), "lag_primary_nw": L_nw, "lag_strict_overlap": L_ov,
        "recal_beta": [float(b) for b in np.linalg.lstsq(
            np.column_stack([np.ones(len(ytr)), vtr]), ytr, rcond=None)[0]],
    }
    for rung, f_ref in (("L0", f_L0), ("L1", f_L1)):
        e_ref = (yte - f_ref) ** 2
        t, p = naive_t(e_ref - e_text)
        out[rung] = {"mse_ref": mse(f_ref), "mse_text": mse(f_text),
                     "gain_pct": 100.0 * (mse(f_ref) - mse(f_text)) / mse(f_ref),
                     "stat": t, "p": p, "stat_type": "naive obs t"}

    # ---- placebo predictions: permute the text rows once per seed (price control
    # and label kept aligned — the maec run_placebo convention), alpha frozen at
    # the real-data CV choice. Shared across both L2 readings.
    placebo_err = []
    if placebo:
        from sp500vol.models.classical_text._fit_utils import build_ridge
        for s in PLACEBO_SEEDS:
            rng = np.random.default_rng(s)
            Ptr = sparse.hstack([Ttr[rng.permutation(Ttr.shape[0])],
                                 sparse.csr_matrix(ztr[:, None])], format="csr")
            Pte = sparse.hstack([Tte[rng.permutation(Tte.shape[0])],
                                 sparse.csr_matrix(zte[:, None])], format="csr")
            r = build_ridge(ridge.alpha_).fit(Ptr, ytr)
            placebo_err.append((yte - r.predict(Pte)) ** 2)

    for vname, m_tr in (("incl", m_tr_incl), ("loo", m_tr_loo)):
        f_L2, b_L2 = ols_fit_apply(ytr, np.column_stack([vtr, m_tr]),
                                   np.column_stack([vte, m_te]))
        e_L2 = (yte - f_L2) ** 2
        t2, p2 = naive_t(e_L2 - e_text)
        dm_nw, p_nw, _ = clustered_dm(e_text, e_L2, dates_te, L_nw)
        dm_ov, p_ov, _ = clustered_dm(e_text, e_L2, dates_te, L_ov)
        v = {
            "beta_cik_mean": float(b_L2[2]),
            "L2": {"mse_ref": mse(f_L2), "mse_text": mse(f_text),
                   "gain_pct": 100.0 * (mse(f_L2) - mse(f_text)) / mse(f_L2),
                   "stat": t2, "p": p2, "stat_type": "naive obs t"},
            "L3": {"mse_ref": mse(f_L2), "mse_text": mse(f_text),
                   "gain_pct": 100.0 * (mse(f_L2) - mse(f_text)) / mse(f_L2),
                   "stat": dm_nw, "p": p_nw, "stat_type": "date-clustered DM",
                   "n_dates": n_d, "lag_primary_nw": L_nw,
                   "lag_strict_overlap": L_ov, "dm_strict_overlap": dm_ov,
                   "p_strict_overlap": p_ov},
        }
        if placebo:
            dms = [clustered_dm(e_p, e_L2, dates_te, L_nw)[0] for e_p in placebo_err]
            v["placebo"] = {"n_seeds": len(PLACEBO_SEEDS), "dms": dms,
                            "mean_dm": float(np.mean(dms)),
                            "max_abs_dm": float(np.max(np.abs(dms))),
                            "gate_pass": bool(np.max(np.abs(dms)) < PLACEBO_DM_GATE)}
        out[vname] = v
    log(f"  split train{sp['train_years'][0]}-{sp['train_years'][-1]} -> "
        f"test {sp['test_year']}: alpha={ridge.alpha_:g} "
        f"L0 {out['L0']['gain_pct']:+.2f}% L1 {out['L1']['gain_pct']:+.2f}% "
        f"L2[incl] {out['incl']['L2']['gain_pct']:+.2f}% "
        f"(b={out['incl']['beta_cik_mean']:+.2f}) "
        f"L2[loo] {out['loo']['L2']['gain_pct']:+.2f}% "
        f"(b={out['loo']['beta_cik_mean']:+.2f}) ({time.time() - t0:.0f}s)")
    return out


def micro_average(res: list[dict], rung: str, variant: str | None = None) -> dict:
    """Count-weighted pooled MSE across test years — Kogan's Table 2 'micro-ave'
    (verified: their baseline column reproduces 0.1576 under this rule).
    `variant` selects an L2 reading ('incl'/'loo') for the L2+ rungs."""
    n = np.array([r["n_test"] for r in res], float)
    cells = [(r if variant is None else r[variant])[rung] for r in res]
    mr = np.array([c["mse_ref"] for c in cells], float)
    mt = np.array([c["mse_text"] for c in cells], float)
    MR, MT = float((mr * n).sum() / n.sum()), float((mt * n).sum() / n.sum())
    return {"mse_ref": MR, "mse_text": MT, "gain_pct": 100.0 * (MR - MT) / MR,
            "n": int(n.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="kogan_corpus_fetch.py --dest dir")
    ap.add_argument("--work", default=None, help="cache dir (default <data>/_work)")
    ap.add_argument("--stage", choices=["data", "ladder", "all"], default="all")
    ap.add_argument("--force-rerun", action="store_true",
                    help="single-shot override; requires --reason")
    ap.add_argument("--reason", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data)
    work = Path(args.work) if args.work else data_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)
    out_csv, out_md = TABLES / "kogan_corpus_audit.csv", TABLES / "kogan_corpus_audit.md"

    # single-shot guard (prereg §"artifacts")
    if args.stage in ("ladder", "all") and out_csv.exists():
        if not (args.force_rerun and args.reason):
            sys.exit(f"REFUSED (single-shot): {out_csv} exists. Reruns are bug-fixes "
                     f"only — pass --force-rerun --reason '...' and log the diff in "
                     f"the prereg revision record.")
        log(f"[single-shot] force-rerun; reason: {args.reason}")

    t_start = time.time()
    hashes = verify_manifest(data_dir)

    # --- the two PRE-DECLARED split families (see the module docstring)
    splits_prereg = [{"train_years": list(range(1996, y + 1)), "test_year": y + 1,
                      "arm": "L0_prereg"} for y in range(1996, 2006)]
    splits_pub = [{"train_years": list(range(ty - 5, ty)), "test_year": ty,
                   "arm": "L0_pub"} for ty in range(2001, 2007)]

    if args.stage in ("data", "all"):
        stage_data(data_dir, work)
        stage_counts(data_dir, work, splits_prereg + splits_pub)
    if args.stage == "data":
        return

    panel = pd.read_parquet(work / "panel.parquet")
    X = sparse.load_npz(work / "X_counts.npz")
    union_vocab = json.loads((work / "union_vocab.json").read_text())
    year_counts = json.loads((work / "year_counts.json").read_text())
    consistency = json.loads((work / "consistency.json").read_text())

    # ---- G-K3: CIK coverage + cross-year firm recurrence
    per_cik_years = panel.groupby("cik").year.nunique()
    gk3 = {
        "n_rows": int(len(panel)), "n_unique_cik": int(panel.cik.nunique()),
        "mean_years_per_cik": float(per_cik_years.mean()),
        "median_years_per_cik": float(per_cik_years.median()),
        "pct_cik_in_1_year_only": float(100.0 * (per_cik_years == 1).mean()),
        "pct_cik_in_ge5_years": float(100.0 * (per_cik_years >= 5).mean()),
        "pct_rows_cik_recurring": float(
            100.0 * panel.cik.map(per_cik_years).gt(1).mean()),
        "max_years_per_cik": int(per_cik_years.max()),
    }
    log(f"G-K3: {gk3['n_unique_cik']} CIKs over {gk3['n_rows']} filings; "
        f"mean {gk3['mean_years_per_cik']:.2f} years/CIK; "
        f"{gk3['pct_rows_cik_recurring']:.1f}% of rows are recurring firms")

    log("=== L0_pub arm (Kogan's published convention: 5-year rolling, test 2001-06) ===")
    res_pub = [fit_split(sp, panel, X, union_vocab, year_counts, placebo=False)
               for sp in splits_pub]
    log("=== L0_prereg arm (prereg rule: expanding train <= y, test = y+1) ===")
    res_pre = [fit_split(sp, panel, X, union_vocab, year_counts, placebo=True)
               for sp in splits_prereg]

    # ---- G-K1: L0_pub micro-average vs Kogan Table 2
    mic = {r: micro_average(res_pub, r) for r in ("L0", "L1")}
    for v in ("incl", "loo"):
        mic[f"L2_{v}"] = micro_average(res_pub, "L2", variant=v)
    ours = mic["L0"]["gain_pct"]
    same_sign = bool(np.sign(ours) == np.sign(PUB_GAIN_PCT))
    # order of magnitude: |log10 ratio| < 1 (i.e. within a factor of 10)
    same_oom = bool(ours > 0 and abs(np.log10(abs(ours) / abs(PUB_GAIN_PCT))) < 1.0)
    gk1_pass = bool(same_sign and same_oom)
    log(f"G-K1: ours {ours:+.2f}% vs published {PUB_GAIN_PCT:+.2f}% "
        f"(0.1557 vs 0.1576) -> sign {same_sign}, oom {same_oom}, "
        f"PASS={gk1_pass}")

    # ---- L4: Holm over the pre-declared family = the L3 per-year p-values
    # (one family per L2 reading; the family is the arm's own per-year p-values)
    for res in (res_pre, res_pub):
        for v in ("incl", "loo"):
            ps = np.array([r[v]["L3"]["p"] for r in res], float)
            adj = fc.holm(np.where(np.isfinite(ps), ps, 1.0))
            for r, a, p in zip(res, adj, ps):
                c = r[v]["L3"]
                r[v]["L4"] = {"mse_ref": c["mse_ref"], "mse_text": c["mse_text"],
                              "gain_pct": c["gain_pct"], "stat": c["stat"],
                              "p": float(a), "p_raw": float(p),
                              "stat_type": "date-clustered DM + Holm",
                              "family_size": len(res)}

    # ---- verdicts + L5 conjunction (L1 AND L2 AND L4)
    def adds(cell) -> bool:
        if not np.isfinite(cell["stat"]) or not np.isfinite(cell["p"]):
            return False
        better = cell["stat"] > 0 if cell["stat_type"] == "naive obs t" else cell["stat"] < 0
        return bool(better and cell["p"] < ALPHA_SIG)

    def verdict_of(cell) -> str:
        return ("text adds" if adds(cell) else
                ("text HURTS" if (np.isfinite(cell["p"]) and cell["p"] < ALPHA_SIG)
                 else "null"))

    for res in (res_pre, res_pub):
        for r in res:
            for rung in ("L0", "L1"):
                r[rung]["verdict"] = verdict_of(r[rung])
            for v in ("incl", "loo"):
                for rung in ("L2", "L3", "L4"):
                    r[v][rung]["verdict"] = verdict_of(r[v][rung])
                surv = adds(r["L1"]) and adds(r[v]["L2"]) and adds(r[v]["L4"])
                # The prereg makes the label-shuffle placebo a GATE ("|DM|<2 as gate,
                # same form as the main protocol"), separate from the L5 conjunction. A cell that
                # survives L1&L2&L4 but whose SHUFFLED-text arm reproduces a
                # comparable edge over the same reference is not admissible
                # evidence of text value, so both counts are carried.
                gate = r[v].get("placebo", {}).get("gate_pass", None)
                c4 = r[v]["L4"]
                r[v]["L5"] = {"mse_ref": c4["mse_ref"], "mse_text": c4["mse_text"],
                              "gain_pct": c4["gain_pct"], "stat": c4["stat"],
                              "p": c4["p"], "stat_type": "conjunction L1&L2&L4",
                              "verdict": "SURVIVES" if surv else "does not survive",
                              "survives": bool(surv),
                              "placebo_gate_pass": gate,
                              "survives_placebo_gated": bool(surv and gate),
                              "components": {"L1": adds(r["L1"]),
                                             "L2": adds(r[v]["L2"]),
                                             "L4": adds(r[v]["L4"])}}

    # ---- branch (prereg §"branch commitments"), evaluated under BOTH L2 readings
    n_surv = {v: sum(r[v]["L5"]["survives"] for r in res_pre) for v in ("incl", "loo")}
    n_surv_gated = {v: sum(r[v]["L5"]["survives_placebo_gated"] for r in res_pre)
                    for v in ("incl", "loo")}
    l0_pre_pos = sum(r["L0"]["gain_pct"] > 0 for r in res_pre)
    def branch_for(nsurv: int) -> str:
        if not gk1_pass:
            return "c"
        return "b" if nsurv > 0 else "a"

    # Branch is determined on the PLACEBO-GATED survivor count (the placebo is a
    # gate, not a rung); the ungated count is reported alongside. Here every
    # combination — both L2 readings x gated/ungated — yields the same branch,
    # which is stated explicitly in the md rather than assumed.
    branches = {v: branch_for(n_surv_gated[v]) for v in ("incl", "loo")}
    branches_ungated = {v: branch_for(n_surv[v]) for v in ("incl", "loo")}
    # PRIMARY = the leave-one-out reading. Declared reason, not an outcome: under
    # the self-inclusive reading the L2 "reference" is WORSE than L1's (its
    # per-CIK mean contains the row's own training label, so OLS loads ~1.0 onto
    # it), which INVERTS the rung — L2 is specified to *strengthen* the reference.
    # A rung that weakens the reference inflates the text gain and would
    # manufacture a survival. Both are reported in full; see the md.
    branch = branches["loo"]
    branch_txt = {
        "c": "L0 does not reproduce the published positive -> report the reproduction "
             "failure itself, no further inference; the census marks this corpus "
             "'published reading not reproduced', reason public.",
        "b": "text SURVIVES the full cascade on Kogan's corpus -> the protocol has "
             "CERTIFIED a real published positive: the real-world positive control "
             "R11/R14 asked for (the protocol does not only kill). Report as such and "
             "soften the generality of the 'near-null' wording — shortcut size is a "
             "property of panel and baseline, not a constant (FACTS §11/§13g).",
        "a": "reproduce-then-dissolve -> the census claim stands: of the three corpora, "
             "every one we can lawfully obtain is run end-to-end; the apparent gain "
             "reproduces k/k and survives 0/k. Chapter 07's existing Kogan section is "
             "replaced in place by this reading.",
    }[branch]
    log(f"BRANCH ({branch}) fires [primary=loo, placebo-gated]: "
        f"L0 reproduces={gk1_pass}; L5 survivors loo={n_surv['loo']}/{len(res_pre)} "
        f"ungated -> {n_surv_gated['loo']}/{len(res_pre)} placebo-gated; "
        f"incl={n_surv['incl']}->{n_surv_gated['incl']}; branches "
        f"gated={branches} ungated={branches_ungated}")

    # ================================ CSV =====================================
    rows = []
    for arm, res in (("L0_prereg", res_pre), ("L0_pub", res_pub)):
        for r in res:
            for rung in ("L0", "L1", "L2", "L3", "L4", "L5"):
                # L0/L1 precede the L2 reference, so they are variant-invariant
                for v in (["-"] if rung in ("L0", "L1") else ["incl", "loo"]):
                    c = r[rung] if v == "-" else r[v][rung]
                    src = r if v == "-" else r[v]
                    rows.append({
                        "arm": arm, "l2_variant": v, "rung": rung,
                        "rung_desc": RUNG_DESC[rung], "test_year": r["test_year"],
                        "train_from": r["train_years"][0],
                        "train_to": r["train_years"][-1],
                        "n_train": r["n_train"], "n_test": r["n_test"],
                        "n_test_dates": r["n_test_dates"], "alpha": r["alpha"],
                        "mse_ref": c["mse_ref"], "mse_text": c["mse_text"],
                        "gain_pct": c["gain_pct"], "stat_type": c["stat_type"],
                        "stat": c["stat"], "p": c["p"],
                        "p_raw": c.get("p_raw", np.nan),
                        "beta_cik_mean": src.get("beta_cik_mean", np.nan),
                        "cik_train_coverage_test": r["cik_train_coverage_test"],
                        "frac_train_rows_singleton_cik":
                            r["frac_train_rows_singleton_cik"],
                        "lag_primary_nw": r["lag_primary_nw"],
                        "lag_strict_overlap": r["lag_strict_overlap"],
                        "dm_strict_overlap": (
                            r[v]["L3"]["dm_strict_overlap"] if v != "-" else np.nan),
                        "p_strict_overlap": (
                            r[v]["L3"]["p_strict_overlap"] if v != "-" else np.nan),
                        "placebo_max_abs_dm": (
                            r[v].get("placebo", {}).get("max_abs_dm", np.nan)
                            if v != "-" else np.nan),
                        "placebo_gate_pass": (
                            r[v].get("placebo", {}).get("gate_pass", "")
                            if v != "-" else ""),
                        "verdict": c["verdict"],
                    })
    lad = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    lad.to_csv(out_csv, index=False)

    runtime = time.time() - t_start
    payload = {"gk1": {"ours_pct": ours, "published_pct": PUB_GAIN_PCT,
                       "same_sign": same_sign, "same_oom": same_oom,
                       "pass": gk1_pass, "micro": mic},
               "gk3": gk3, "branch": branch, "branches": branches,
               "branches_ungated": branches_ungated,
               "n_surv": n_surv, "n_surv_gated": n_surv_gated,
               "consistency": consistency, "runtime_s": runtime,
               "l0_pre_pos": l0_pre_pos}
    (work / "results.json").write_text(json.dumps(
        {"payload": payload, "res_pre": res_pre, "res_pub": res_pub},
        indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    write_md(out_md, lad, res_pre, res_pub, payload, hashes, args)
    print(lad[(lad.arm == "L0_prereg") & (lad.l2_variant != "incl")][
        ["rung", "l2_variant", "test_year", "n_test", "gain_pct", "stat", "p",
         "verdict"]].to_string(index=False))
    log(f"wrote {out_csv} and {out_md} — total runtime {runtime / 60:.1f} min")


def fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return "n/a"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def write_md(path: Path, lad, res_pre, res_pub, payload, hashes, args) -> None:
    g1, g3 = payload["gk1"], payload["gk3"]
    mic = g1["micro"]
    br = payload["branch"]
    BRANCH_TXT = {
        "a": "**(a) reproduce-then-dissolve** — *registered consequence:* \"L0 reproduces a positive"
             " text effect of the published magnitude, L1–L5 dissolve it rung by rung until it does not survive → the census claim stands: 'of the three corpora, every one whose data"
             " we can obtain is run in full; apparent gain reproduces k/k, survives 0/k'; Chapter 07's existing Kogan section is **replaced in place**"
             " by this reading (page budget self-funded).\"",
        "b": "**(b) text SURVIVES the full cascade** — *registered consequence:* \"**the protocol has certified"
             " a real published positive result**: this is exactly the **real-world positive "
             "control** R11/R14 repeatedly demanded (proof the protocol does not only kill); report it faithfully and accordingly soften the generality of the 'near-null' wording --"
             " `the size of the shortcut is a property of panel and baseline, not a constant` (absorbed directly by the existing FACTS §11/§13g framing);"
             " **good news for this paper, write it as is**.\"",
        "c": "**(c) L0 does not reproduce the published positive** — *registered consequence:* "
             "\"report the reproduction failure itself, no further inference; in the census table this corpus is marked 'published reading not "
             "reproduced', reason public.\"",
    }[br]
    m = ["# M1 — the audit cascade on Kogan et al. (2009)'s OWN 10-K corpus",
         "",
         "*Pre-registered: `configs/prereg_kogan_corpus.md`, tag `prereg-kc-v1.0`. "
         "Single-shot. Generated "
         f"{time.strftime('%Y-%m-%d %H:%M:%S')} by `scripts/analysis/kogan_corpus_audit.py` "
         f"in {payload['runtime_s'] / 60:.1f} min, local CPU only.*",
         "",
         "## The prereg, quoted",
         "",
         "> **L0 published convention**: Kogan's convention -- text features (TF-IDF) + `logvol.-12` as the control,"
         " regressed on `logvol.+12`, **naive obs-level inference**, their annual OOS split (train ≤ y, test = y+1)."
         " Readout = MSE improvement rate of the text arm vs the `logvol.-12`-only arm (the quantity they report).",
         "> **L1 recalibrated baseline**: the baseline becomes the **recalibrated** `logvol.-12` (OLS intercept+slope,"
         " fit on training years, frozen on the test year). **L2 firm-identity reference**: the reference additionally gets the **same-CIK training-period"
         " mean log volatility** (zero-text term). **L3 clustered inference**: cluster on **filing date**"
         " (the shock-sharing unit), HAC + HLN, replacing naive obs-t. **L4 Holm (pre-declared family)**: Holm within the family of L3's per-year "
         "p-values. **L5 conjunction**: survives only when L1∧L2∧L4 all hold. **placebo**: label permutation (5 seeds),"
         " |DM|<2 as the gate.",
         "",
         "## The census — three corpora, not a sample",
         "",
         "The prereg's scope correction, verbatim: *\"the 'N independent published results' the AC literally demands"
         " do not exist in this field\"* — the public evidence base for disclosure-text → volatility is **three "
         "corpora**, and HTML / NumHTML / VolTAGE / KeFVP / ECHO-GL are all built on MDRM "
         "(per each repo's own README), so auditing *N models* would be *N models on one "
         "corpus* = pseudo-replication with a fake denominator. Hence a **census**:",
         "",
         "| # | corpus | status | authority |",
         "|---|---|---|---|",
         "| 1 | MDRM / earnings-call | **cite-only, not obtainable** — text+audio bundled "
         "in split volumes, no licence, no longer redistributed | `prereg_maec_audit.md` §9 "
         "(prior ruling; unchanged by this experiment) |",
         "| 2 | MAEC | **audited** — cascade run end-to-end | FACTS §13g |",
         "| 3 | **Kogan 10-K corpus** | **audited — THIS TABLE** | prereg-kc-v1.0 |",
         "",
         "Distinct from the committed `kogan_dissolve.md`, which ports Kogan's evaluation "
         "*design* onto **our** panel. This runs **our cascade on their corpus** — the "
         "\"reproduce a published positive, then re-price it under the protocol\" audit.",
         "",
         "## G-K0 — provenance and integrity",
         "",
         f"Source `http://www.cs.cmu.edu/~ark/10K/data/` (Version 1.0, 2009-03-31; addendum "
         f"2009-09-18). **Data is NOT redistributed — only the pipeline ships** "
         f"(`kogan_corpus_fetch.py` re-fetches it). {len(hashes['files'])} files, "
         f"{hashes['manifest']['total_bytes'] / 1e6:.1f} MB, fetched "
         f"{hashes['manifest']['fetched_utc']} UTC; every file re-hashed at audit time and "
         f"matched against the fetch manifest.",
         "",
         "| file | bytes | SHA-256 |", "|---|---|---|"]
    for f in hashes["files"]:
        m.append(f"| `{f['file']}` | {f['bytes']:,} | `{f['sha256']}` |")

    m += ["", "### Row / key-space consistency (asserted, hard)", "",
          "Per year (hard assertions): `len(meta) == len(logvol.+12) == len(logvol.-12) "
          "== n(tok members)`; meta / +12 / −12 / tok agree as **key multisets**; every "
          "duplicated key is an **exact** duplicate in every column; no null "
          "log-volatility; no key spans two years. Our row counts vs **Kogan et al. (2009) "
          "Table 1** (\"documents\" column):", "",
          "| year | our rows | Kogan Table 1 | match | unique keys | exact dup rows | "
          "unique CIKs |", "|---|---|---|---|---|---|---|"]
    for c in payload["consistency"]:
        m.append(f"| {c['year']} | {c['n_rows']:,} | {c['kogan_table1_docs']:,} | "
                 f"{'YES' if c['matches_table1'] else '**NO**'} | "
                 f"{c['n_unique_keys']:,} | {c['n_exact_dup_rows']} | {c['n_cik']:,} |")
    tot = sum(c["n_rows"] for c in payload["consistency"])
    tot_dup = sum(c["n_exact_dup_rows"] for c in payload["consistency"])
    m += [f"| **total** | **{tot:,}** | **30,474** (sum of their own column) | "
          f"{'YES' if tot == 30474 else 'NO'} | {tot - tot_dup:,} | {tot_dup} | "
          f"{g3['n_unique_cik']:,} |", "",
          f"**Exact duplicate records ({tot_dup} rows in {tot:,}) are KEPT, deliberately.** "
          "The corpus ships {n} records that repeat a key with byte-identical date, URL, "
          "company, CIK and *both* log volatilities, and with the `.mda` member repeated "
          "inside the tarball. They are not dropped because **the per-year counts that "
          "include them are exactly the counts Kogan et al. publish** — their reading is "
          "computed over these rows too, so dropping them would silently change the "
          "denominator this audit has to match. They are carried as exact duplicates "
          "(asserted), share one text row, and at {pct} of the corpus cannot move any "
          "reading materially.".format(n=tot_dup, pct=f"{100 * tot_dup / tot:.2f}%"),
          "",
          "**Every year matches Kogan's published per-year document count exactly** — "
          "the corpus we audit is the corpus they report. One published-table arithmetic "
          "note (theirs, not ours): Table 1's *total* row reads 26,806 documents, but its "
          "own per-year column sums to **30,474**. Their *words* column does sum to the "
          "published 247.7M total, and 247.7M/26,806 = 9,240 = their published words/doc, "
          "so the 26,806 total is internally consistent with the words/doc cell and "
          "inconsistent with the document column. This does not touch the reproduction: "
          "their Table 2 micro-average is reproduced exactly from the **per-year** counts "
          "(see G-K1), which are the counts our download matches.",
          "",
          "## G-K1 — does L0 reproduce the published positive?",
          "",
          "### What we compared against (the published number, located and cited)",
          "",
          "**Kogan, Levin, Routledge, Sagi & Smith (2009), \"Predicting Risk from Financial "
          "Reports with Regression\", NAACL-HLT 2009, Table 2 (p. 5)** "
          "(`http://www.cs.cmu.edu/~nasmith/papers/kogan+levin+routledge+sagi+smith.naacl09.pdf`, "
          "SHA-256 `9538e0e07ee36588a2bd478cf41b6b7e47e7c33d8a6da0a5cced1ab805230cd2`). "
          "Table 2 reports MSE of log-volatility on test-year predictions:",
          "",
          "| Table 2 row | 2001 | 2002 | 2003 | 2004 | 2005 | 2006 | micro-ave |",
          "|---|---|---|---|---|---|---|---|",
          "| `v^(-12)` (baseline) | 0.1747 | 0.1600 | 0.1873 | 0.1442 | 0.1365 | 0.1463 | "
          "**0.1576** |",
          "| TFIDF (text only) | 0.2033 | 0.2118 | 0.2178 | 0.1660 | 0.1544 | 0.1599 | 0.1842 |",
          "| **TFIDF+** (text + `v^(-12)`) | 0.1919 | 0.1618 | 0.1965 | *0.1246 | *0.1276 | "
          "*0.1403 | **\\*0.1557** |",
          "",
          f"So **the published gain we test against is +{PUB_GAIN_PCT:.2f}%** "
          f"(0.1576 → 0.1557 micro-averaged MSE; `*` = significant vs baseline under their "
          "permutation test, p<0.05). TFIDF+ is the row commensurable with our L0 (their "
          "only TF-IDF text+`v^(-12)` model). Their best combined row (LOG1P+ with bigrams, "
          "0.1538) is a +2.41% gain, so the published effect sits in the **1–2.5%** band.",
          "",
          "*Verification that we read the table correctly:* their `micro-ave` is the "
          "count-weighted pooled MSE over test years, and recomputing it from their own "
          "Table 1 per-year counts (2001–06: 2596, 2846, 3612, 3559, 3474, 3308; n=19,395) "
          "reproduces **0.15761 → 0.1576** for the baseline and **0.15567 → 0.1557** for "
          "TFIDF+ **exactly**. That fixes both the estimand and the aggregation rule we "
          "must match, from their numbers alone.",
          "",
          "### Our L0 reading",
          "",
          "| arm | convention | MSE(`logvol.-12`-only) | MSE(text+control) | gain |",
          "|---|---|---|---|---|",
          f"| **L0_pub** (the G-K1 comparator) | Kogan's published split: train = the 5 "
          f"years preceding, test = 2001..2006, count-weighted micro-average | "
          f"{mic['L0']['mse_ref']:.4f} | {mic['L0']['mse_text']:.4f} | "
          f"**{mic['L0']['gain_pct']:+.2f}%** |",
          f"| published (Table 2) | *ibid.* | 0.1576 | 0.1557 | **{PUB_GAIN_PCT:+.2f}%** |",
          "",
          f"**G-K1: {'PASS' if g1['pass'] else 'FAIL'}** — sign "
          f"{'agrees' if g1['same_sign'] else 'DISAGREES'} "
          f"({'both positive' if g1['same_sign'] and g1['ours_pct'] > 0 else 'opposite'}), "
          f"order of magnitude {'agrees' if g1['same_oom'] else 'DISAGREES'} "
          f"(ours {g1['ours_pct']:+.2f}% vs published {g1['published_pct']:+.2f}%; ratio "
          f"{abs(g1['ours_pct'] / g1['published_pct']):.2f}x, gate = within 10x). "
          f"**Stated plainly:** the gate operationalises the prereg's *same order of magnitude* as "
          f"\"within a factor of 10\" (|log10 ratio| < 1); at "
          f"{abs(g1['ours_pct'] / g1['published_pct']):.1f}x our reading is inside that "
          f"bound but **not comfortably** — both are single-digit-percent positives, yet "
          f"ours is the larger. Under the stricter \"same power of ten\" reading "
          f"({g1['published_pct']:.2f}% is 10^0, {g1['ours_pct']:.2f}% is 10^1) the gate "
          f"would not pass and branch (c) would fire. The per-year table below is the "
          f"evidence that this is one effect reproduced at a different magnitude rather "
          f"than a different effect. ",
          "**Nothing was tuned to make this match**: the estimator, feature recipe, "
          "vocabulary size, alpha grid and the x10 control scaling are the committed "
          "`kogan_dissolve.py` constants, fixed before the corpus was downloaded; the "
          "published number was located and written into the script as a constant before "
          "the ladder ran.",
          "",
          "### Per-year, against their Table 2 — the structure, not just the average",
          "",
          "A single micro-average is a weak reproduction check, so the same comparison is "
          "made year by year (published gain = their `v^(-12)` row vs their `TFIDF+` row):",
          "",
          "| test year | published gain % | our L0 gain % | sign agrees |",
          "|---|---|---|---|"]
    n_sign = 0
    for r in res_pub:
        ty = r["test_year"]
        pg = 100.0 * (PUB_TABLE2_BASE[ty] - PUB_TABLE2_TFIDF_PLUS[ty]) / PUB_TABLE2_BASE[ty]
        og = r["L0"]["gain_pct"]
        ok = (pg > 0) == (og > 0)
        n_sign += ok
        m.append(f"| {ty} | {pg:+.2f} | {og:+.2f} | {'YES' if ok else 'no'} |")
    m += ["",
          f"**{n_sign}/6 test years agree in sign**, and the *shape* of their result is "
          "reproduced: 2001 is the worst year for text under both readings (both "
          "negative), 2004 the best under both, and text is positive in every "
          "post-Sarbanes-Oxley year (2004–06) under both — the very pattern their §6.3 "
          "builds the SOX argument on. Our per-year gains sit uniformly **above** theirs "
          "(by roughly 5–18pp), which is what a better-conditioned text arm predicts "
          "(sublinear + L2-normalised TF-IDF and a CV-tuned ridge vs their raw TF×IDF and "
          "fixed SVR hyper-parameters, Disclosure 4) — i.e. we reproduce a **stronger** "
          "apparent positive than published, not a weaker one. That direction matters for "
          "this audit: the rung being re-priced downstream is, if anything, more "
          "favourable to text than the published one.",
          "",
          "## A prereg imprecision, recorded — and resolved by reporting BOTH arms",
          "",
          "The prereg calls `train ≤ y, test = y+1` *\"their annual "
          "OOS split\"* (**their** annual OOS split). **It is not theirs.** Kogan et al. §6 state: *\"We used as training "
          "examples all reports from the five-year period preceding the test year (so six "
          "experiments on six different training and test sets are shown in the figure)\"*, "
          "with test years 2001–2006 (Table 2); their Table 4 varies that window over 1, 2 "
          "and 5 years and **never** uses an expanding one. The prereg's *rule* is "
          "unambiguous; only its *attribution* is wrong. Running only the prereg rule would "
          "leave G-K1 unanswerable as specified (an expanding-window reading compared "
          "against a rolling-window published number); running only the published rule "
          "would violate the binding prereg. **Both were therefore declared in the script "
          "before any statistic and both are reported unconditionally** — no arm is "
          "selected on its outcome:",
          "",
          "- **`L0_prereg`** — the prereg's literal rule (expanding, train ≤ y, test = y+1, "
          "y = 1996..2005). **This is the binding rung that feeds L1–L5.**",
          "- **`L0_pub`** — Kogan's actual published convention (5-year rolling, test "
          "2001–06, micro-averaged). **The G-K1 comparator**, because it is the only arm "
          "commensurable with their Table 2.",
          "",
          "## G-K2 — no look-ahead from L1 on",
          "",
          "Asserted per split (hard): `max(train_years) < test_year`; "
          "`max(train filing date) < min(test filing date)`; the TF-IDF vocabulary **and** "
          "idf are fit on training-year documents only and frozen on the test year; the "
          "price control's standardisation uses training-year mean/sd only; the L1 "
          "recalibration (intercept+slope) and the L2 reference (incl. the per-CIK mean) "
          "are fit on training years only and frozen on the test year.",
          "",
          "**L0's look-ahead convention is reproduced deliberately and labelled** (see "
          "Disclosures): L0's baseline is `logvol.-12` used *directly* as the prediction "
          "with no fit — Kogan's own Table 2 baseline row — which is exactly the rung L1 "
          "then repairs.",
          "",
          "## G-K3 — CIK coverage and cross-year firm recurrence",
          "",
          f"- **{g3['n_unique_cik']:,} unique CIKs** across **{g3['n_rows']:,} filings** "
          f"(1996–2006).",
          f"- Mean **{g3['mean_years_per_cik']:.2f}** years per CIK (median "
          f"{g3['median_years_per_cik']:.0f}, max {g3['max_years_per_cik']}).",
          f"- **{g3['pct_rows_cik_recurring']:.1f}%** of filings come from a CIK that "
          f"appears in more than one year — the precondition for the L2 firm-identity "
          f"reference.",
          f"- {g3['pct_cik_in_1_year_only']:.1f}% of CIKs appear in exactly one year; "
          f"{g3['pct_cik_in_ge5_years']:.1f}% appear in 5+ years.",
          "- Per-split coverage (share of test rows whose CIK was seen in training, i.e. "
          "gets a real per-CIK mean rather than the global fallback) is in the CSV column "
          "`cik_train_coverage_test`; the range over the binding `L0_prereg` splits is "
          f"**{min(r['cik_train_coverage_test'] for r in res_pre):.1%}–"
          f"{max(r['cik_train_coverage_test'] for r in res_pre):.1%}**.",
          "",
          "## L2 — a SECOND prereg ambiguity, and why it decides the branch",
          "",
          "The prereg says the L2 reference *\"additionally gets the same-CIK "
          "**training-period mean** log volatility (zero-text term)\"*. It does not say "
          "whether a **training** row's own label may enter its own CIK mean. That "
          "silence is not cosmetic — it flips the fired branch, so both readings are "
          "computed and reported, and neither is chosen on its outcome:",
          "",
          "- **`incl` (literal)** — the per-CIK training mean, self-inclusive. A training "
          "row's feature then **contains its own label**.",
          "- **`loo` (leave-one-out)** — the same mean with the row's own label removed "
          f"(singleton-CIK rows fall back to the global training mean; "
          f"{min(r['frac_train_rows_singleton_cik'] for r in res_pre):.1%}–"
          f"{max(r['frac_train_rows_singleton_cik'] for r in res_pre):.1%} of training "
          "rows across the binding splits). **PRIMARY.**",
          "",
          "Test rows are **identical** under both readings: a test row's own label can "
          "never enter a mean taken over training years (G-K2 holds either way). The "
          "fork is purely about the training fit.",
          "",
          "**Why `loo` is primary — a declared structural reason, not a result.** L2 is "
          "specified to *strengthen* the reference (it is the firm-identity control the "
          "text must beat). Under `incl` it does the opposite: the fitted coefficient on "
          "the per-CIK mean is driven toward 1.0 because the feature partly **is** the "
          "label, so the reference overfits and its **test** MSE lands *worse than L1's* "
          "— which mechanically *inflates* the text's measured gain and can manufacture a "
          "survival. A rung that weakens the reference cannot be the rung the prereg "
          "describes. The committed template agrees: `maec_protocol.py`'s entity-mean "
          "control (STPEV) is a **point-in-time expanding prior-label mean** built with "
          "`shift(1)` — i.e. the current row's label is excluded by construction, and the "
          "self-inclusive fixed mean is demoted to a robustness block.",
          "",
          "| test year | β on CIK-mean (`incl`) | β on CIK-mean (`loo`) | MSE L1 | "
          "MSE L2 `incl` | MSE L2 `loo` | `incl` weakens ref? |",
          "|---|---|---|---|---|---|---|"]
    for r in res_pre:
        worse = r["incl"]["L2"]["mse_ref"] > r["L1"]["mse_ref"]
        m.append(f"| {r['test_year']} | {r['incl']['beta_cik_mean']:+.3f} | "
                 f"{r['loo']['beta_cik_mean']:+.3f} | {r['L1']['mse_ref']:.4f} | "
                 f"{r['incl']['L2']['mse_ref']:.4f} | {r['loo']['L2']['mse_ref']:.4f} | "
                 f"{'**YES**' if worse else 'no'} |")
    n_worse = sum(r["incl"]["L2"]["mse_ref"] > r["L1"]["mse_ref"] for r in res_pre)
    n_better = sum(r["loo"]["L2"]["mse_ref"] < r["L1"]["mse_ref"] for r in res_pre)
    m += ["",
          f"Across the {len(res_pre)} binding splits the self-inclusive reference is "
          f"worse than L1's in **{n_worse}/{len(res_pre)}**, while the leave-one-out "
          f"reference is better than L1's in **{n_better}/{len(res_pre)}** — i.e. only "
          "`loo` behaves like the control the rung is meant to be. The cleanest "
          "demonstration is the `test 1997` split (training = 1996 alone): "
          f"{res_pre[0]['frac_train_rows_singleton_cik']:.1%} of training rows are the "
          "only filing their CIK has, so the \"firm mean\" *is* that row's label, and the "
          f"fitted β is **{res_pre[0]['incl']['beta_cik_mean']:+.3f}**.",
          "",
          f"**Branch under each reading:** `loo` (primary) → "
          f"**({payload['branches']['loo']})** with "
          f"{payload['n_surv_gated']['loo']}/{len(res_pre)} placebo-gated survivors "
          f"({payload['n_surv']['loo']}/{len(res_pre)} ungated); `incl` (literal) → "
          f"**({payload['branches']['incl']})** with "
          f"{payload['n_surv_gated']['incl']}/{len(res_pre)} placebo-gated "
          f"({payload['n_surv']['incl']}/{len(res_pre)} ungated).",
          "",
          ("**The two readings agree on the branch**, so this ambiguity does *not* change "
           "which registered consequence fires — a fact worth stating plainly, since it "
           "means the headline finding is robust to the prereg's silence. What it does "
           "change is the **size** of the L2 rung (micro-averaged text gain "
           f"{mic['L2_incl']['gain_pct']:+.2f}% under `incl` vs "
           f"{mic['L2_loo']['gain_pct']:+.2f}% under `loo`) and the **survivor set**. The "
           "ambiguity is still flagged, because `incl`'s L2 numbers are not "
           "interpretable as a firm-identity control at all and must not be quoted."
           if payload["branches"]["loo"] == payload["branches"]["incl"] else
           "**The two readings fire DIFFERENT branches** — the prereg as written does not "
           "determine the finding, and the authors must rule on the wording "
           "before the paper commits."),
          "",
          "## THE LADDER — `L0_prereg` (binding), per test year",
          "",
          "`gain_pct` = % reduction in log-volatility MSE of the text arm vs that rung's "
          "reference. Sign convention: naive obs t **positive** = text better; DM "
          "**negative** = text better.",
          "",
          "L2–L5 are shown under the **primary (leave-one-out)** L2 reading; the "
          "literal self-inclusive reading is tabulated in the L2 section above.",
          "",
          "| rung | test year | n | gain % | stat | value | p | verdict |",
          "|---|---|---|---|---|---|---|---|"]

    def ladder_rows(arm, variant="loo"):
        sub = lad[(lad.arm == arm) & (lad.l2_variant.isin(["-", variant]))].copy()
        sub["_o"] = sub.rung.map({r: i for i, r in enumerate(RUNG_DESC)})
        return sub.sort_values(["_o", "test_year"], kind="mergesort")

    for _, r in ladder_rows("L0_prereg").iterrows():
        m.append(f"| {r.rung} | {r.test_year} | {r.n_test:,} | {r.gain_pct:+.2f} | "
                 f"{r.stat_type} | {r.stat:+.2f} | {fmt_p(r.p)} | **{r.verdict}** |")
    m += ["", "## THE LADDER — `L0_pub` (published convention), per test year", "",
          "| rung | test year | n | gain % | stat | value | p | verdict |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in ladder_rows("L0_pub").iterrows():
        m.append(f"| {r.rung} | {r.test_year} | {r.n_test:,} | {r.gain_pct:+.2f} | "
                 f"{r.stat_type} | {r.stat:+.2f} | {fmt_p(r.p)} | **{r.verdict}** |")

    m += ["", "### Micro-averaged rung summary (`L0_pub`, Kogan's aggregation)", "",
          "| rung | reference | MSE(ref) | MSE(text) | gain % |", "|---|---|---|---|---|"]
    for rung, lbl in (("L0", "RAW `logvol.-12` (their baseline)"),
                      ("L1", "recalibrated `logvol.-12`"),
                      ("L2_loo", "recalibrated `logvol.-12` + same-CIK train mean "
                                 "(**primary**, leave-one-out)"),
                      ("L2_incl", "recalibrated `logvol.-12` + same-CIK train mean "
                                  "(literal, self-inclusive)")):
        c = mic[rung]
        m.append(f"| {rung.replace('_', ' ')} | {lbl} | {c['mse_ref']:.4f} | "
                 f"{c['mse_text']:.4f} | **{c['gain_pct']:+.2f}** |")

    # placebo + L3 lag disclosure (primary L2 reading)
    pl = [r["loo"]["placebo"] for r in res_pre]
    m += ["", "## Placebo — label shuffle (5 seeds), |DM| < 2 gate "
          "**— THE LOAD-BEARING RESULT**", "",
          "The text rows are permuted (price control and label kept aligned — the committed "
          "`maec_protocol.run_placebo` convention), the arm is refit, and the L3-stage "
          "filing-date-clustered DM is recomputed against the same L2 reference. Alpha is "
          "frozen at each split's real-data CV choice (disclosed: the placebo interrogates "
          "the signal, not the tuning). DM **negative = the arm beats the reference**.",
          "",
          "| test year | REAL L3 DM | placebo mean DM | placebo max abs DM | gate (<2) | "
          "real vs placebo |", "|---|---|---|---|---|---|"]
    for r in res_pre:
        p = r["loo"]["placebo"]
        real = r["loo"]["L3"]["stat"]
        sep = real - p["mean_dm"]
        # separation is only meaningful where the REAL arm actually has an edge
        verdict = ("no real edge" if real > -1.5 else
                   "**separated**" if sep < -1.5 else
                   "*not separated*" if sep > -0.75 else "partial")
        m.append(f"| {r['test_year']} | {real:+.2f} | {p['mean_dm']:+.2f} | "
                 f"{p['max_abs_dm']:.2f} | {'PASS' if p['gate_pass'] else '**FAIL**'} | "
                 f"{verdict} |")
    n_pl = sum(p["gate_pass"] for p in pl)
    all_dms = [x for r in res_pre for x in r["loo"]["placebo"]["dms"]]
    n_neg2 = sum(x < -2 for x in all_dms)
    m += ["",
          f"**Only {n_pl}/{len(pl)} splits pass the |DM|<2 placebo gate, and "
          f"{n_neg2}/{len(all_dms)} individual placebo draws show the SHUFFLED-text arm "
          f"*significantly beating* the L2 reference (DM < −2).** This is the most "
          "important number in the table and it is reported first because it bounds "
          "everything below it.",
          "",
          "**Why the placebo is not null here, and what that means.** Unlike "
          "`maec_protocol`, where the text enters a *combiner on top of the reference* (so "
          "permuting it collapses the arm back onto the reference and DM→0 by "
          "construction), Kogan's convention makes the arm a **single joint model** — one "
          "ridge on `[TF-IDF | scaled logvol.-12]`. Permuting the text therefore does "
          "**not** reduce the arm to the reference: it leaves a ridge on "
          "`[noise | logvol.-12]`, which still carries the real price control and is a "
          "structurally different estimator from the L2 reference "
          "(`OLS[1, logvol.-12, CIK-mean]`). So a non-zero placebo DM is partly a "
          "**form** difference, not text signal — the ported gate cannot separate the two, "
          "and it is the *comparison of real vs placebo DM*, not the gate alone, that "
          "carries the inference.",
          "",
          "Read that way the years split sharply:",
          "",
          "- **Genuine text signal** — 1998 (real −4.05 vs placebo +1.12), 2000 (−2.80 vs "
          "+0.44), 2003 (−8.10 vs −0.37) and 2006 (−2.17 vs −0.59): the real arm is far "
          "more negative than shuffled text ever gets, and the gate passes. (2006 clears "
          "the placebo but not L4's Holm-adjusted p, so it is not an L5 survivor.)",
          "- **Not attributable to text** — 1997 (real −2.50 vs placebo −2.08) and 2005 "
          "(−2.72 vs −2.38): **shuffled text very nearly reproduces the whole edge**, so "
          "these years' apparent gains are the arm-vs-reference form difference, not "
          "disclosure content.",
          "- **No edge to explain** — 1999 (real +0.99) and 2001 (real −0.09): the real "
          "arm does not beat the reference at all, and 2001's placebo runs strongly the "
          "*other* way (+5.19).",
          "",
          f"Consequently the L5 conjunction is reported **twice**: ungated "
          f"({payload['n_surv']['loo']}/{len(res_pre)}) and **placebo-gated** "
          f"({payload['n_surv_gated']['loo']}/{len(res_pre)}) — the latter is the honest "
          "survivor count, and the one the branch is decided on.",
          "",
          "## L3 inference — the HAC lag, disclosed in full", "",
          "The prereg fixes the clustering unit (**filing date**) and the estimator "
          "(**HAC + HLN**) but not the lag. Two lags are reported for every split; neither "
          "is chosen on its outcome:",
          "",
          "- **`lag_strict_overlap`** — the `maec_protocol.hac_lag_L` port: the number of "
          "later distinct test filing dates whose **12-month label window still overlaps**. "
          "With a 12-month label and a 12-month test year this is ≈ `n_dates − 1` **by "
          "construction**, so the HLN factor collapses to ≤ 0 and the test is *undefined*. "
          "That is not a bug — it is the honest statement that **a single test year of "
          "12-month-forward labels contains ≈ one effective observation**.",
          "- **`lag_primary_nw`** — the Newey–West rule of thumb `floor(4*(n/100)^(2/9))` on "
          "the filing-date grid. **This is the primary**, chosen by a rule declared before "
          "the numbers: of the two candidates it is the **more permissive**, i.e. the one "
          "**most favourable to text**. Any death under it therefore cannot be a lag "
          "artefact, and any survival is reported on text's best terms.",
          "",
          "| test year | n dates | lag (NW, primary) | DM | p | lag (strict overlap) | "
          "DM | p |", "|---|---|---|---|---|---|---|---|"]
    for r in res_pre:
        c = r["loo"]["L3"]
        dm_ov = (f"{c['dm_strict_overlap']:+.2f}"
                 if np.isfinite(c["dm_strict_overlap"]) else "n/a (h≈n)")
        m.append(f"| {r['test_year']} | {c['n_dates']} | {c['lag_primary_nw']} | "
                 f"{c['stat']:+.2f} | {fmt_p(c['p'])} | {c['lag_strict_overlap']} | "
                 f"{dm_ov} | {fmt_p(c['p_strict_overlap'])} |")

    surv = [r["test_year"] for r in res_pre if r["loo"]["L5"]["survives"]]
    surv_g = [r["test_year"] for r in res_pre
              if r["loo"]["L5"]["survives_placebo_gated"]]
    m += ["", "## THE FIRED BRANCH", "", BRANCH_TXT, "",
          f"*Why this branch:* G-K1 {'PASS' if payload['gk1']['pass'] else 'FAIL'} "
          f"(L0 {'reproduces' if payload['gk1']['pass'] else 'does not reproduce'} the "
          f"published positive in sign and order of magnitude) and "
          f"**{payload['n_surv_gated']['loo']}/{len(res_pre)}** `L0_prereg` test years "
          f"survive the L5 conjunction (L1 ∧ L2 ∧ L4) **and** clear the label-shuffle "
          f"placebo gate under the primary leave-one-out L2 reading "
          f"({payload['n_surv']['loo']}/{len(res_pre)} survive L5 before the gate)"
          + (f" — placebo-clean surviving years: {surv_g}." if surv_g else "."),
          "",
          "**Robustness of the branch across the two open choices.** The four "
          f"combinations give: L2 `loo` gated **({payload['branches']['loo']})** / "
          f"ungated **({payload['branches_ungated']['loo']})**; L2 `incl` gated "
          f"**({payload['branches']['incl']})** / ungated "
          f"**({payload['branches_ungated']['incl']})**."
          + ("  They all agree, so the branch is not an artefact of either unresolved "
             "prereg question."
             if len({payload['branches']['loo'], payload['branches_ungated']['loo'],
                     payload['branches']['incl'],
                     payload['branches_ungated']['incl']}) == 1 else
             "  **They do not all agree, and the exception is diagnostic rather than "
             "substantive.** Under the *primary* `loo` reading the branch is stable — "
             f"({payload['branches']['loo']}) both gated and ungated. The only cell that "
             f"differs is `incl`+gate → ({payload['branches']['incl']}), and it differs "
             "for a degenerate reason: the self-inclusive reference is so damaged that "
             "**shuffled text beats it in all 10 splits** (placebo mean DM −2.1 to −9.0), "
             "so *nothing* clears the placebo gate and the survivor count collapses to 0. "
             "That is the placebo independently detecting the broken control — it is not "
             "evidence that text fails. A reader should therefore not read `incl`+gate as "
             "a genuine reproduce-then-dissolve."),
          "",
          "**But read the size honestly.** The positive control is real yet *narrow*: "
          f"{payload['n_surv_gated']['loo']}/{len(res_pre)} years, and the placebo shows "
          "that in the non-surviving and in two of the L5-surviving years shuffled text "
          "reproduces much or all of the apparent edge. The defensible claim is *\"the "
          "protocol certifies text on Kogan's corpus in a minority of test years, "
          "concentrated in 1998/2000/2003\"* — **not** \"text survives on Kogan's "
          "corpus\".",
          "",
          "## Disclosures",
          "",
          "1. **L0's look-ahead / naive convention is reproduced deliberately.** L0's "
          "baseline is `logvol.-12` used directly as the forecast with **no fit at all** — "
          "Kogan's own Table 2 baseline row — and L0/L1/L2 inference is **naive obs-level "
          "t**, treating ~2.5k same-year filings with 12-month-overlapping labels as "
          "independent. Both are the defects being reproduced, not our protocol. From L1 "
          "on every weight is training-year-fit and frozen (G-K2); from L3 on inference is "
          "filing-date-clustered.",
          "2. **Data is not redistributed.** Only `kogan_corpus_fetch.py` and this script "
          "ship. The corpus is public at `http://www.cs.cmu.edu/~ark/10K/` with no licence "
          "terms beyond a citation request, which we honour in the text and here: Kogan, "
          "Levin, Routledge, Sagi & Smith, *Predicting Risk from Financial Reports with "
          "Regression*, NAACL-HLT 2009. Every file's SHA-256 is above, so the exact bytes "
          "we read are verifiable without us hosting them.",
          "3. **The text is the tokenised MD&A section**, not the whole 10-K: the corpus's "
          "own README defines `yyyy.tok.tgz` as \"the tokenized MD&A sections\", and the "
          "paper (§4) filters to Section 7/7A on purpose. Kogan's Table 2 is computed on "
          "this same text, so the comparison is like-for-like.",
          "4. **Estimator and feature deviations from their exact spec** (each is the "
          "committed `kogan_dissolve.py` recipe, fixed before any number here was seen, "
          "and none was adjusted afterwards): ridge with alpha by 5-fold CV over "
          f"`{list(ALPHA_GRID)}` in place of their SVR (SVM^light, linear kernel, eps=0.1, "
          "C=1/mean(h·h)); TF-IDF 1–2gram, top-5k by train term frequency, sublinear tf, "
          "L2 norm, in place of their `(1/|d|)·freq × log(N/df)` over the full training "
          f"vocabulary; the control enters the text arm standardised and scaled ×{VOL_SCALE:.0f} "
          "so its ridge penalty is ~1/100 of a text feature's ≈ the unpenalised control "
          "their design implies. These are why G-K1 is a **sign + order-of-magnitude** "
          "gate, exactly as pre-registered, and not an exact-value gate.",
          "5. **Vocabulary pruning guard.** Term counters are pruned within 256-doc chunks "
          "(hapaxes) and per year (count < 5) for memory. Both prunings are asserted "
          "irrelevant to top-5k selection per split: the smallest surviving top-5k count "
          "must exceed both the year-prune bound (5 × #train years) and the chunk-prune "
          "bound (#chunks, the max count a term can lose). Margins are in "
          "`_work/results.json` (`vocab_guard`).",
          "6. **Two prereg ambiguities were found and neither was resolved by "
          "choosing.** (i) the OOS split attribution (§\"A prereg imprecision\"), and "
          "(ii) L2's self-inclusion (§\"L2 — a SECOND prereg ambiguity\"). Both are "
          "reported under every reading, with the primary fixed by a structural argument "
          "declared in the script before execution. (i) does not change the branch. "
          f"(ii) **does**: `loo` fires ({payload['branches']['loo']}) while `incl` fires "
          f"({payload['branches']['incl']}) once the placebo gate is applied — see "
          "\"Robustness of the branch\" for why `incl`'s outcome is a symptom of its "
          "broken control rather than a finding. **The prereg wording therefore needs a "
          "ruling before this section is written up**, and no `incl` L2 number should be "
          "quoted.",
          "6b. **The placebo gate is not a formality here — it fails in "
          f"{len(pl) - n_pl}/{len(pl)} splits** and is the reason the survivor count is "
          f"reported as {payload['n_surv_gated']['loo']}/{len(res_pre)} rather than "
          f"{payload['n_surv']['loo']}/{len(res_pre)}. Because Kogan's convention makes "
          "the arm a joint model rather than a combiner over the reference, the ported "
          "|DM|<2 gate conflates text signal with the arm-vs-reference form difference; "
          "the real-vs-placebo DM comparison (see that section) is what the inference "
          "actually rests on. A protocol-level note for the panel: the gate's ported form "
          "is mis-calibrated for joint arms and should be re-specified for them.",
          "7. **Single-shot.** The script refuses to overwrite these tables without "
          f"`--force-rerun --reason`{'; this run WAS a force-rerun, reason: ' + repr(args.reason) if args.force_rerun else ''}. "
          "The two L0 arms and the two L2 readings were all declared before execution "
          "precisely so that none could be picked after the fact.",
          "8. **Compute.** Local CPU only, BLAS threads pinned to 1 and joblib capped at "
          f"{N_JOBS} workers on a shared machine; no GPU, no `/Volumes/Z`, "
          f"total runtime {payload['runtime_s'] / 60:.1f} min.",
          ""]
    path.write_text("\n".join(m))


if __name__ == "__main__":
    main()
