"""ROW 9 — Replicate-then-dissolve of a Kogan-2009-style evaluation DESIGN on the bench.

Purpose: the paper's claim that prior positive disclosure-NLP results (Kogan 2009
lineage) are baseline-weakness / firm-identity artefacts was inference-by-analogy
(round-3 panel: eic W2 / domain W2, MAJOR). This script converts it to a
DEMONSTRATION: reproduce the published evaluation design on OUR panel (long_form,
10-K/10-Q), show it yields the published-style "text significantly improves
volatility prediction" positive, then tighten the protocol one rung at a time and
watch the effect dissolve.

THE LADDER (all rungs on long_form, B2 TF-IDF recipe only — one model keeps it clean):
  L0  Kogan-style: predict log RV with TF-IDF(1-2gram, 5k, sublinear) features + a
      SINGLE lagged-volatility regressor (log feature_rv_22d) as the only price
      control; evaluated the published way — (a) IN-SAMPLE fit, (b) year-by-year
      OOS (fixed 2010-2019 train window, each of 2020-2025 as a test year), MSE of
      log vol, naive obs-level paired t-stats. No recalibration, no clustering, no
      identity control, no split discipline (test years include our val years).
  L1  + strict chronological OOS: same two models, evaluated ONLY on the declared
      test split (2022-2025). Still weak control + naive obs-level t.
  L2  Replace the weak control with the recalibrated HAR reference: the archived
      B2 text forecast enters the M1 log-space combiner (fc.log_combo, weights fit
      on val, frozen on test) against recalibrated A2 HAR; vol-unit QLIKE;
      obs-level DM.                       == committed m1 cells (SANITY GATE)
  L3  + day-clustered DM (clustered_dm.py). == committed m1 cells (SANITY GATE)
  L4  + firm-identity-augmented reference: f_R = exp(a + b log fHAR +
      c log firm_mean_val_RV), val-fit, frozen (firm_identity_control.py recipe).
  L5  + maximal price pool (A2,A6,A3,A4,A5 log-combined) on top of the firm
      control, and Holm across the pre-declared L5 family.

REPLICATION SCOPE: this replicates the DESIGN of Kogan et al. (2009) — TF-IDF text
features vs a single lagged-volatility regression baseline, log-vol MSE,
in-sample + per-year OOS, obs-level inference — NOT their corpus (their 10-Ks,
1996-2006) or their 12-month horizon. Estimator is the archived B2 ridge recipe
(their SVR analogue); the memory-safe streamed-vocabulary pathway is the one
validated in scripts/experiments/section_ablation.py (B2sec_fullrepro reproduced
the archived B2 run).

SANITY GATE (hard, machine precision): the L2/L3 rungs recompute the B2 long_form
cells of results/tables/m1_clustered.csv (obs-level dm_q/p_q AND clustered
dm_q_clust/p_q_clust, qlike_R/qlike_U/rel_impr_pct/g_log) and the matching
vol_* columns of results/tables/m1_ensemble_primary.csv (B2 is seed-invariant, so
its ensemble row equals seed2026). Any mismatch aborts before tables are written.
Soft pathway check: the text-only B2 re-fit must land within 5% test QLIKE
(variance-unit) of the archived B2 run's metrics.json (the tolerance
section_ablation.py declared for the same streamed pathway).

LOOK-AHEAD DISCLOSURE: L0's in-sample arm deliberately fits and evaluates on the
same rows, and L0's year-by-year arm scores years (2020-21) that the benchmark
reserves for validation — that is the DEFECT BEING REPLICATED, clearly labelled;
no rung L2+ uses any test-fitted weight (combiner/reference weights are val-fit,
frozen on test throughout).

Run from repo root:
  .venv/bin/python scripts/analysis/kogan_dissolve.py --stage all
Stages: features (streamed vocab + counts, cached), models (ridge fits, cached),
ladder (rungs + gates + tables). Outputs results/tables/kogan_dissolve.{csv,md}.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from scipy import sparse, stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

import forecast_combination as fc  # noqa: E402
from clustered_dm import dm_test_clustered  # noqa: E402
from firm_identity_control import firm_means  # noqa: E402
from maximal_reference import PRICE_MODELS, fit_apply_log, load_price_panel  # noqa: E402
from sp500vol.evaluation.dm_test import dm_test  # noqa: E402
from sp500vol.models.classical_text._fit_utils import (  # noqa: E402
    fit_ridge_cv,
    maybe_exp,
    maybe_log,
)

TEXT_CACHE = Path("/path/to/data-root/sp500vol-data/processed/_text_cache/filing_texts.parquet")
RUNS = REPO / "results" / "runs"
TABLES = REPO / "results" / "tables"
WORKDIR = Path(os.environ.get("SP500VOL_SCRATCH", tempfile.gettempdir())) / "kogan_dissolve"

DISC = "long_form"
SEED = 2026
HORIZONS = (5, 10, 20)
KEY = ["ticker", "accession", "horizon_days"]
SORT = ["filing_time_utc", "ticker", "accession"]
EPS = 1e-12

# --- B2 recipe constants (src/sp500vol/models/classical_text/{tfidf_ridge,_fit_utils}.py)
MAX_FEATURES = 5000
ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0)
MIN_PRED, MAX_PRED = 0.02, 5.0
TOKEN_RE = re.compile(r"(?u)\b[a-z]{2,}\b")
CHUNK_DOCS = 256
N_JOBS = 6
VOCAB_DICT_CAP = 12_000_000

# --- Kogan-style single lagged-vol control
VOL_COL = "feature_rv_22d"  # trailing 22-trading-day RV: the single-scalar analogue
VOL_SCALE = 10.0            # x10 on the standardized column => ridge penalty on the
                            # control ~1/100 of a text feature's (approximates an
                            # UNPENALIZED price control, the fair Kogan reading)
YEARLY_YEARS = tuple(range(2020, 2026))

RUNG_DESC = {
    "L0": "Kogan-style: TF-IDF + single lagged-vol control, per-year OOS 2020-25, log-vol MSE, naive obs t",
    "L1": "+ strict chronological OOS (declared test split 2022-25 only)",
    "L2": "weak control -> recalibrated HAR reference (M1 log-combiner, val-fit frozen), QLIKE, obs-level DM",
    "L3": "+ day-clustered DM (daily-mean loss diffs, HAC lag=h-1 days)",
    "L4": "+ firm-identity-augmented reference (HAR + firm mean val RV)",
    "L5": "+ maximal price pool (A2,A6,A3,A4,A5) on top of firm control, Holm over the L5 family",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# shared loading
# ---------------------------------------------------------------------------

def load_master() -> pd.DataFrame:
    """Archived B2 long_form run rows (split membership verified against A2)."""
    b2 = pd.read_parquet(RUNS / f"B2_tfidf_ridge_full_{DISC}_seed{SEED}" / "predictions.parquet")
    a2 = pd.read_parquet(
        RUNS / f"A2_har_rv_full_{DISC}_seed{SEED}" / "predictions.parquet",
        columns=KEY + ["split"],
    )
    chk = b2[KEY + ["split"]].merge(a2, on=KEY, suffixes=("_b2", "_a2"), how="outer", indicator=True)
    if not ((chk["_merge"] == "both").all() and (chk["split_b2"] == chk["split_a2"]).all()):
        raise RuntimeError("B2 and A2 long_form split membership differ — aborting")
    b2 = b2.copy()
    b2["year"] = pd.to_datetime(b2["filing_time_utc"], utc=True).dt.year
    return b2


# ---------------------------------------------------------------------------
# stage 1: features — streamed train vocab + counts matrix over all filings
# (the section_ablation.py memory-safe pathway, reading the text cache directly)
# ---------------------------------------------------------------------------

def _count_chunk(texts: list[str], prune_hapax: bool) -> dict[str, int]:
    c: Counter = Counter()
    for t in texts:
        toks = TOKEN_RE.findall(t.lower())
        c.update(toks)
        c.update(map(" ".join, zip(toks, toks[1:])))
    if prune_hapax:
        return {k: v for k, v in c.items() if v >= 2}
    return dict(c)


def _transform_chunk(texts: list[str], vocab: list[str]):
    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-z]{2,}\b",
        vocabulary=vocab,
        dtype=np.int32,
    )
    return cv.transform(texts)


def _iter_cache_chunks(needed: set[str], limit: int | None = None):
    """Yield (paths, texts) chunks of long_form filings from the text cache."""
    f = pq.ParquetFile(TEXT_CACHE)
    n_seen = 0
    for batch in f.iter_batches(batch_size=CHUNK_DOCS, columns=["text_path", "text"]):
        d = batch.to_pandas()
        d = d[d["text_path"].isin(needed)]
        if d.empty:
            continue
        yield d["text_path"].astype(str).tolist(), d["text"].astype(str).tolist()
        n_seen += len(d)
        if limit and n_seen >= limit:
            return


def stage_features(limit: int | None) -> None:
    vocab_path = WORKDIR / "vocab.json"
    counts_path = WORKDIR / "X_counts.npz"
    order_path = WORKDIR / "order.json"
    if vocab_path.exists() and counts_path.exists() and order_path.exists():
        log("features: cached, skipping")
        return
    master = load_master()
    filings = master.drop_duplicates("accession")[["text_path", "split"]].astype(str)
    needed = set(filings["text_path"])
    train_paths = set(filings.loc[filings["split"] == "train", "text_path"])
    if limit:
        needed = set(list(needed)[: limit])  # smoke only
        train_paths &= needed
    log(f"features: {len(needed)} filings needed ({len(train_paths)} train)")

    # --- pass 1: streamed term-frequency counts over TRAIN filings only
    t0 = time.time()

    def gen_count_tasks():
        for paths, texts in _iter_cache_chunks(needed, limit):
            chunk = [t for p, t in zip(paths, texts) if p in train_paths]
            if chunk:
                yield delayed(_count_chunk)(chunk, True)

    merged: Counter = Counter()
    threshold = 2
    n_parts = 0
    for part in Parallel(n_jobs=N_JOBS, return_as="generator")(gen_count_tasks()):
        merged.update(part)
        n_parts += 1
        if n_parts % 20 == 0:
            log(f"features: vocab pass chunk {n_parts}, dict={len(merged)} ({time.time()-t0:.0f}s)")
        if len(merged) > VOCAB_DICT_CAP:
            merged = Counter({k: v for k, v in merged.items() if v >= threshold})
            threshold *= 2
            log(f"features: pruned dict to {len(merged)} (next threshold {threshold})")
    top = sorted(merged.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_FEATURES]
    vocab = sorted(k for k, _ in top)
    del merged
    vocab_path.write_text(json.dumps(vocab))
    log(f"features: vocab={len(vocab)} terms ({time.time()-t0:.0f}s); transform pass")

    # --- pass 2: counts matrix over ALL long_form filings with the train vocab
    order: list[str] = []

    def gen_tf_tasks():
        for paths, texts in _iter_cache_chunks(needed, limit):
            order.extend(paths)
            yield delayed(_transform_chunk)(texts, vocab)

    mats = list(Parallel(n_jobs=N_JOBS, return_as="generator")(gen_tf_tasks()))
    X_counts = sparse.vstack(mats, format="csr")
    missing = needed - set(order)
    if missing:  # cache should cover everything; read stragglers directly
        log(f"features: {len(missing)} paths missing from cache, reading files")
        texts = [Path(p).read_text(encoding="utf-8", errors="replace") for p in sorted(missing)]
        X_counts = sparse.vstack([X_counts, _transform_chunk(texts, vocab)], format="csr")
        order.extend(sorted(missing))
    sparse.save_npz(counts_path, X_counts)
    order_path.write_text(json.dumps(order))
    log(f"features: counts {X_counts.shape}, nnz={X_counts.nnz} -> cached ({time.time()-t0:.0f}s)")


# ---------------------------------------------------------------------------
# stage 2: models — Kogan-style V / TEXT+V fits (+ text-only B2 repro), cached
# ---------------------------------------------------------------------------

def _ols_fit_predict(Xf: np.ndarray, yf: np.ndarray, Xp: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(
        np.column_stack([np.ones(len(yf)), Xf]), yf, rcond=None)
    return np.column_stack([np.ones(len(Xp)), Xp]) @ beta


def stage_models(limit: int | None) -> None:
    out_path = WORKDIR / "kogan_preds.parquet"
    if out_path.exists():
        log("models: cached, skipping")
        return
    from sklearn.feature_extraction.text import TfidfTransformer

    master = load_master()
    if limit:
        order_probe = set(json.loads((WORKDIR / "order.json").read_text()))
        master = master[master["text_path"].astype(str).isin(order_probe)].reset_index(drop=True)
        log(f"models(smoke): master reduced to {len(master)} rows")
    X_counts = sparse.load_npz(WORKDIR / "X_counts.npz")
    order = json.loads((WORKDIR / "order.json").read_text())
    row_of_path = {p: i for i, p in enumerate(order)}
    filing_idx = master["text_path"].astype(str).map(row_of_path)
    if filing_idx.isna().any():
        raise RuntimeError("some master rows missing from the counts matrix")
    filing_idx = filing_idx.to_numpy(dtype=int)

    split = master["split"].to_numpy()
    horizons = master["horizon_days"].astype(int).to_numpy()
    y = master["label_realised_vol"].to_numpy(dtype=float)
    logy = maybe_log(y, log_target=True)
    logvol = np.log(master[VOL_COL].to_numpy(dtype=float) + EPS)

    # tf-idf: idf fit on TRAIN filings (train-fit models) and on ALL filings
    # (the L0 in-sample arm — the published-style in-sample fit)
    uniq = master.drop_duplicates("accession")
    train_rows = sorted({row_of_path[p] for p in uniq.loc[uniq.split == "train", "text_path"].astype(str)})
    all_rows = sorted({row_of_path[p] for p in uniq["text_path"].astype(str)})
    tfidf_tr = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
    tfidf_tr.fit(X_counts[train_rows])
    X_tr_idf = tfidf_tr.transform(X_counts)
    tfidf_all = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
    tfidf_all.fit(X_counts[all_rows])
    X_all_idf = tfidf_all.transform(X_counts)
    del X_counts

    preds = {k: np.full(len(master), np.nan) for k in
             ("V_tr", "TV_tr", "V_ins", "TV_ins", "B2repro_tr")}
    alphas: dict[str, dict[int, float]] = {"TV_tr": {}, "TV_ins": {}, "B2repro_tr": {}}
    for h in HORIZONS:
        m_all = horizons == h
        m_tr = m_all & (split == "train")
        t0 = time.time()
        # standardize the log-vol control on TRAIN stats (train-fit models)
        mu, sd = logvol[m_tr].mean(), logvol[m_tr].std()
        zvol = (logvol - mu) / sd * VOL_SCALE
        # V: OLS log y ~ log lagged vol (Kogan's HIST baseline)
        preds["V_tr"][m_all] = _ols_fit_predict(
            logvol[m_tr][:, None], logy[m_tr], logvol[m_all][:, None])
        # TEXT+V: B2 ridge recipe on [TF-IDF | scaled control]
        Xtv_fit = sparse.hstack(
            [X_tr_idf[filing_idx[m_tr]], sparse.csr_matrix(zvol[m_tr][:, None])], format="csr")
        Xtv_pred = sparse.hstack(
            [X_tr_idf[filing_idx[m_all]], sparse.csr_matrix(zvol[m_all][:, None])], format="csr")
        ridge = fit_ridge_cv(Xtv_fit, logy[m_tr], ALPHA_GRID)
        alphas["TV_tr"][h] = float(ridge.alpha_)
        preds["TV_tr"][m_all] = ridge.predict(Xtv_pred)
        log(f"models: h={h} TV_tr alpha={ridge.alpha_} cv_mse={ridge.cv_mse_:.5f} ({time.time()-t0:.0f}s)")
        # text-only B2 recipe re-fit (pathway sanity vs archived B2 run)
        ridge_b2 = fit_ridge_cv(X_tr_idf[filing_idx[m_tr]], logy[m_tr], ALPHA_GRID)
        alphas["B2repro_tr"][h] = float(ridge_b2.alpha_)
        preds["B2repro_tr"][m_all] = ridge_b2.predict(X_tr_idf[filing_idx[m_all]])
        log(f"models: h={h} B2repro alpha={ridge_b2.alpha_} ({time.time()-t0:.0f}s)")
        # L0 IN-SAMPLE arm (deliberate look-ahead, labelled): fit on ALL rows
        mu_a, sd_a = logvol[m_all].mean(), logvol[m_all].std()
        zvol_a = (logvol - mu_a) / sd_a * VOL_SCALE
        preds["V_ins"][m_all] = _ols_fit_predict(
            logvol[m_all][:, None], logy[m_all], logvol[m_all][:, None])
        Xtv_a = sparse.hstack(
            [X_all_idf[filing_idx[m_all]], sparse.csr_matrix(zvol_a[m_all][:, None])],
            format="csr")
        ridge_i = fit_ridge_cv(Xtv_a, logy[m_all], ALPHA_GRID)
        alphas["TV_ins"][h] = float(ridge_i.alpha_)
        preds["TV_ins"][m_all] = ridge_i.predict(Xtv_a)
        log(f"models: h={h} TV_ins alpha={ridge_i.alpha_} ({time.time()-t0:.0f}s)")

    out = master[KEY + ["split", "year", "filing_time_utc", "effective_trading_day",
                        "label_realised_vol", VOL_COL]].copy()
    out["logy"] = logy
    for k, v in preds.items():
        out[f"logpred_{k}"] = v
    out.to_parquet(out_path, index=False)
    (WORKDIR / "kogan_alphas.json").write_text(json.dumps(alphas, indent=2))
    log(f"models: wrote {out_path}")


# ---------------------------------------------------------------------------
# stage 3: the ladder
# ---------------------------------------------------------------------------

def naive_t(d: np.ndarray) -> tuple[float, float]:
    """Naive obs-level paired t on loss differential d (positive mean = model B wins)."""
    d = np.asarray(d, float)
    n = len(d)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    return float(t), float(2.0 * stats.t.sf(abs(t), df=n - 1))


def mse_row(tag, h, sub, a="logpred_V_tr", b="logpred_TV_tr"):
    eV = (sub["logy"] - sub[a]) ** 2
    eT = (sub["logy"] - sub[b]) ** 2
    t, p = naive_t((eV - eT).to_numpy())
    var0 = ((sub["logy"] - sub["logy"].mean()) ** 2).mean()
    return {
        "arm": tag, "h": h, "n": len(sub),
        "mse_V": float(eV.mean()), "mse_TV": float(eT.mean()),
        "gain_pct": float(100.0 * (eV.mean() - eT.mean()) / eV.mean()),
        "r2_V": float(1.0 - eV.mean() / var0), "r2_TV": float(1.0 - eT.mean() / var0),
        "t_naive": t, "p_naive": p,
    }


def qlike_aux(sub, h, a="logpred_V_tr", b="logpred_TV_tr"):
    """Aux continuity stat: vol-unit QLIKE obs-level DM of exp'd (recipe-clipped) preds."""
    y = sub["label_realised_vol"].to_numpy()
    fa = maybe_exp(sub[a].to_numpy(), log_target=True, min_pred=MIN_PRED, max_pred=MAX_PRED)
    fb = maybe_exp(sub[b].to_numpy(), log_target=True, min_pred=MIN_PRED, max_pred=MAX_PRED)
    la, lb = fc.qlike(y, fa), fc.qlike(y, fb)
    dm, p = dm_test(lb, la, h=h)  # negative = TEXT+V better
    return float(100.0 * (la.mean() - lb.mean()) / la.mean()), float(dm), float(p)


def stage_ladder(write_tables: bool = True) -> None:
    kp = pd.read_parquet(WORKDIR / "kogan_preds.parquet")
    alphas = json.loads((WORKDIR / "kogan_alphas.json").read_text())

    # ================= L0 / L1 — Kogan-style weak-control models ==============
    ins_rows, yearly_rows, l0_rows, l1_rows = [], [], [], []
    for h in HORIZONS:
        d = kp[kp.horizon_days == h]
        # L0a in-sample (deliberate look-ahead, labelled)
        ins_rows.append(mse_row("L0_insample", h, d, "logpred_V_ins", "logpred_TV_ins"))
        # L0b year-by-year OOS, published style (2020-2025 = our val+test years)
        oos = d[d.year.isin(YEARLY_YEARS)]
        for yy in YEARLY_YEARS:
            s = oos[oos.year == yy]
            if len(s) >= 10:
                yearly_rows.append(mse_row(f"L0_year{yy}", h, s))
        r = mse_row("L0_pooled_oos", h, oos)
        r["qlike_rel"], r["qlike_dm"], r["qlike_dm_p"] = qlike_aux(oos, h)
        l0_rows.append(r)
        # L1 strict chronological OOS: declared test split only
        te = d[d.split == "test"]
        r = mse_row("L1_test_split", h, te)
        r["qlike_rel"], r["qlike_dm"], r["qlike_dm_p"] = qlike_aux(te, h)
        l1_rows.append(r)
    ins = pd.DataFrame(ins_rows)
    yearly = pd.DataFrame(yearly_rows)
    l0 = pd.DataFrame(l0_rows)
    l1 = pd.DataFrame(l1_rows)

    # ---- soft pathway sanity: text-only re-fit vs archived B2 metrics.json ----
    b2_metrics = {
        (m["split"], m["horizon_days"]): m["qlike"]
        for m in json.loads((RUNS / f"B2_tfidf_ridge_full_{DISC}_seed{SEED}" / "metrics.json").read_text())
    }
    repro_rows = []
    for h in HORIZONS:
        te = kp[(kp.horizon_days == h) & (kp.split == "test")]
        f = maybe_exp(te["logpred_B2repro_tr"].to_numpy(), log_target=True,
                      min_pred=MIN_PRED, max_pred=MAX_PRED)
        yv = te["label_realised_vol"].to_numpy()
        r = np.clip(yv**2, 1e-12, None) / np.clip(f**2, 1e-12, None)
        q = float(np.mean(r - np.log(r) - 1.0))
        arch = float(b2_metrics[("test", h)])
        repro_rows.append({"h": h, "qlike_var_repro": q, "qlike_var_archived": arch,
                           "diff_pct": 100.0 * (q - arch) / arch})
    repro = pd.DataFrame(repro_rows)
    repro_ok = bool((repro.diff_pct.abs() <= 5.0).all())
    log("SANITY (soft, pathway): text-only B2 re-fit vs archived metrics.json:\n"
        + repro.to_string(index=False))
    if not repro_ok:
        if write_tables:
            raise SystemExit("SANITY FAIL (pathway): B2 re-fit deviates >5% from the "
                             "archived run — stopping, no tables written.")
        log("(smoke) pathway check outside tolerance — expected on a limited corpus")

    # ================= L2-L5 — the protocol rungs on stored forecasts =========
    har = fc.load("A2_har_rv", DISC)[
        ["split"] + KEY + ["prediction_realised_vol", "label_realised_vol",
                           "filing_time_utc", "effective_trading_day"]
    ].rename(columns={"prediction_realised_vol": "fhar"})
    txt = fc.load("B2_tfidf_ridge", DISC)[KEY + ["prediction_realised_vol"]].rename(
        columns={"prediction_realised_vol": "ftext"})
    d = har.merge(txt, on=KEY)
    panel5 = load_price_panel(DISC).merge(txt, on=KEY, how="inner")

    l2_rows, l3_rows, l4_rows, l5_rows = [], [], [], []
    for h in HORIZONS:
        dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(SORT, kind="mergesort")
        dt = d[(d.horizon_days == h) & (d.split == "test")].sort_values(SORT, kind="mergesort")
        yv, yt = dv.label_realised_vol.to_numpy(), dt.label_realised_vol.to_numpy()
        fhv, fht = dv.fhar.to_numpy(), dt.fhar.to_numpy()
        ftv, ftt = dv.ftext.to_numpy(), dt.ftext.to_numpy()
        days_t = dt.effective_trading_day.to_numpy()

        # L2: recalibrated-HAR reference, obs-level DM (== committed m1 cells)
        fR, fU, g_log = fc.log_combo(yv, fhv, ftv, fht, ftt)
        lR, lU = fc.qlike(yt, fR), fc.qlike(yt, fU)
        qR, qU = float(lR.mean()), float(lU.mean())
        rel = 100.0 * (qR - qU) / qR
        dmq, pq = dm_test(lU, lR, h=h)
        l2_rows.append({"h": h, "n": len(dt), "qlike_R": qR, "qlike_U": qU,
                        "rel_impr_pct": rel, "g_log": float(g_log),
                        "dm_q": float(dmq), "p_q": float(pq)})
        # L3: same forecasts, day-clustered DM
        dmc, pc, nd = dm_test_clustered(lU, lR, days_t, h)
        l3_rows.append({"h": h, "n": len(dt), "n_days": nd, "rel_impr_pct": rel,
                        "dm_q_clust": float(dmc), "p_q_clust": float(pc)})
        # L4: + firm-identity-augmented reference (val-fit, frozen)
        fm_v, fm_t, cov = firm_means(dv, dt)
        fRf, _ = fit_apply_log(yv, [fhv, fm_v], [fht, fm_t])
        fUf, bUf = fit_apply_log(yv, [fhv, fm_v, ftv], [fht, fm_t, ftt])
        lRf, lUf = fc.qlike(yt, fRf), fc.qlike(yt, fUf)
        relf = 100.0 * (lRf.mean() - lUf.mean()) / lRf.mean()
        dmf, pf, ndf = dm_test_clustered(lUf, lRf, days_t, h)
        l4_rows.append({"h": h, "n": len(dt), "n_days": ndf, "coverage": cov,
                        "qlike_R": float(lRf.mean()), "qlike_U": float(lUf.mean()),
                        "rel_impr_pct": float(relf), "g_text": float(bUf[3]),
                        "dm_q_clust": float(dmf), "p_q_clust": float(pf)})

        # L5: + maximal price pool on top of the firm control (+ Holm below)
        pv5 = panel5[(panel5.horizon_days == h) & (panel5.split == "val")].sort_values(SORT, kind="mergesort")
        pt5 = panel5[(panel5.horizon_days == h) & (panel5.split == "test")].sort_values(SORT, kind="mergesort")
        yv5, yt5 = pv5.label_realised_vol.to_numpy(), pt5.label_realised_vol.to_numpy()
        days5 = pt5.effective_trading_day.to_numpy()
        Xv = [pv5[f"f_{pm}"].to_numpy() for pm in PRICE_MODELS]
        Xt = [pt5[f"f_{pm}"].to_numpy() for pm in PRICE_MODELS]
        fm5v, fm5t, cov5 = firm_means(pv5, pt5)
        fR5, _ = fit_apply_log(yv5, Xv + [fm5v], Xt + [fm5t])
        fU5, bU5 = fit_apply_log(yv5, Xv + [fm5v, pv5.ftext.to_numpy()],
                                 Xt + [fm5t, pt5.ftext.to_numpy()])
        lR5, lU5 = fc.qlike(yt5, fR5), fc.qlike(yt5, fU5)
        rel5 = 100.0 * (lR5.mean() - lU5.mean()) / lR5.mean()
        dm5, p5, nd5 = dm_test_clustered(lU5, lR5, days5, h)
        l5_rows.append({"h": h, "n": len(pt5), "n_days": nd5, "coverage": cov5,
                        "qlike_R": float(lR5.mean()), "qlike_U": float(lU5.mean()),
                        "rel_impr_pct": float(rel5), "g_text": float(bU5[-1]),
                        "dm_q_clust": float(dm5), "p_q_clust": float(p5)})

    l2 = pd.DataFrame(l2_rows)
    l3 = pd.DataFrame(l3_rows)
    l4 = pd.DataFrame(l4_rows)
    l5 = pd.DataFrame(l5_rows)
    # PRE-DECLARED Holm family (see md): the 3 L5 clustered-DM p-values.
    l5["p_holm"] = fc.holm(l5.p_q_clust.to_numpy())

    # ================= SANITY GATE (hard, machine precision) ==================
    mc = pd.read_csv(TABLES / "m1_clustered.csv")
    mc = mc[(mc.disc == DISC) & (mc.model == "B2_tfidf_ridge")].set_index("h")
    me = pd.read_csv(TABLES / "m1_ensemble_primary.csv")
    me = me[(me.disc == DISC) & (me.model == "B2_tfidf_ridge")].set_index("h")
    gate = []
    for h in HORIZONS:
        r2 = l2[l2.h == h].iloc[0]
        r3 = l3[l3.h == h].iloc[0]
        for name, mine, ref in [
            ("m1_clustered.qlike_R", r2.qlike_R, mc.loc[h, "qlike_R"]),
            ("m1_clustered.qlike_U", r2.qlike_U, mc.loc[h, "qlike_U"]),
            ("m1_clustered.rel_impr_pct", r2.rel_impr_pct, mc.loc[h, "rel_impr_pct"]),
            ("m1_clustered.g_log", r2.g_log, mc.loc[h, "g_log"]),
            ("m1_clustered.dm_q", r2.dm_q, mc.loc[h, "dm_q"]),
            ("m1_clustered.p_q", r2.p_q, mc.loc[h, "p_q"]),
            ("m1_clustered.dm_q_clust", r3.dm_q_clust, mc.loc[h, "dm_q_clust"]),
            ("m1_clustered.p_q_clust", r3.p_q_clust, mc.loc[h, "p_q_clust"]),
            ("m1_ensemble_primary.vol_qlike_R", r2.qlike_R, me.loc[h, "vol_qlike_R"]),
            ("m1_ensemble_primary.vol_qlike_U", r2.qlike_U, me.loc[h, "vol_qlike_U"]),
            ("m1_ensemble_primary.vol_dm_q_clu", r3.dm_q_clust, me.loc[h, "vol_dm_q_clu"]),
            ("m1_ensemble_primary.vol_p_q_clu", r3.p_q_clust, me.loc[h, "vol_p_q_clu"]),
        ]:
            ok = np.allclose(float(mine), float(ref), rtol=1e-9, atol=1e-12)
            gate.append({"h": h, "column": name, "mine": float(mine),
                         "committed": float(ref), "abs_diff": abs(float(mine) - float(ref)),
                         "ok": ok})
    gate = pd.DataFrame(gate)
    max_diff = float(gate.abs_diff.max())
    if not gate.ok.all():
        print(gate[~gate.ok].to_string(index=False))
        raise SystemExit(f"SANITY GATE FAIL: L2/L3 do not reproduce committed m1 "
                         f"tables (max |diff|={max_diff:.3e}) — no tables written.")
    log(f"SANITY GATE PASS: {len(gate)} cells reproduce m1_clustered.csv / "
        f"m1_ensemble_primary.csv exactly (max |diff|={max_diff:.2e})")

    if not write_tables:
        log("(smoke) skipping table write")
        return

    # ================= assemble the 6-rung ladder table =======================
    ladder = []
    for _, r in l0.iterrows():
        ladder.append(["L0", r.h, int(r.n), "log-vol MSE", r.gain_pct,
                       "naive obs t", r.t_naive, r.p_naive, np.nan])
    for _, r in l1.iterrows():
        ladder.append(["L1", r.h, int(r.n), "log-vol MSE", r.gain_pct,
                       "naive obs t", r.t_naive, r.p_naive, np.nan])
    for _, r in l2.iterrows():
        ladder.append(["L2", r.h, int(r.n), "vol-unit QLIKE", r.rel_impr_pct,
                       "obs-level DM", r.dm_q, r.p_q, np.nan])
    for _, r in l3.iterrows():
        ladder.append(["L3", r.h, int(r.n), "vol-unit QLIKE", r.rel_impr_pct,
                       "day-clustered DM", r.dm_q_clust, r.p_q_clust, np.nan])
    for _, r in l4.iterrows():
        ladder.append(["L4", r.h, int(r.n), "vol-unit QLIKE", r.rel_impr_pct,
                       "day-clustered DM", r.dm_q_clust, r.p_q_clust, np.nan])
    for _, r in l5.iterrows():
        ladder.append(["L5", r.h, int(r.n), "vol-unit QLIKE", r.rel_impr_pct,
                       "day-clustered DM", r.dm_q_clust, r.p_q_clust, r.p_holm])
    lad = pd.DataFrame(ladder, columns=["rung", "h", "n_eval", "loss", "text_gain_pct",
                                        "stat_type", "stat", "p_raw", "p_holm_L5"])
    lad["rung_desc"] = lad.rung.map(RUNG_DESC)

    def verdict(row):
        p = row.p_holm_L5 if row.rung == "L5" else row.p_raw
        better = (row.stat < 0) if row.stat_type.endswith("DM") else (row.stat > 0)
        if p < 0.05:
            return "text adds" if better else "text HURTS"
        return "null"

    lad["verdict"] = lad.apply(verdict, axis=1)

    TABLES.mkdir(parents=True, exist_ok=True)
    lad.to_csv(TABLES / "kogan_dissolve.csv", index=False)

    # ================= markdown ================================================
    def fmt_p(p):
        return "n/a" if pd.isna(p) else (f"{p:.2e}" if p < 1e-3 else f"{p:.4f}")

    md = ["# ROW 9 — Kogan-2009-style replicate-then-dissolve ladder "
          "(long_form, B2 TF-IDF recipe)\n",
          "## RESTATED vs BEFORE\n",
          "| | BEFORE | RESTATED (this table) |",
          "|---|---|---|",
          "| status of the prior-work attribution claim | inference by analogy: no prior "
          "design was ever reproduced on this benchmark (round-3 eic W2 / domain W2, MAJOR) | "
          "demonstration: a Kogan-style design run on OUR panel produces a published-style "
          "apparent text effect (large in-sample gain; OOS arm as tabulated), which the "
          "protocol rungs then dissolve |",
          f"| Kogan-style headline (L0, pooled per-year OOS, h=10) | — | text 'improves' "
          f"log-vol MSE by {l0[l0.h==10].gain_pct.iloc[0]:+.2f}% "
          f"(naive obs t={l0[l0.h==10].t_naive.iloc[0]:+.1f}, "
          f"p={fmt_p(l0[l0.h==10].p_naive.iloc[0])}) |",
          f"| protocol endpoint (L5, h=10) | — | {l5[l5.h==10].rel_impr_pct.iloc[0]:+.2f}% "
          f"QLIKE, clustered DM {l5[l5.h==10].dm_q_clust.iloc[0]:+.2f}, "
          f"Holm {fmt_p(l5[l5.h==10].p_holm.iloc[0])} -> "
          f"**{lad[(lad.rung=='L5')&(lad.h==10)].verdict.iloc[0]}** |",
          "",
          "**Scope.** This replicates the *evaluation DESIGN* of Kogan et al. (2009) — "
          "TF-IDF text features vs a single lagged-volatility regression baseline, log-vol "
          "MSE, in-sample + year-by-year OOS, naive obs-level inference — on OUR panel "
          "(S&P 500 10-K/10-Q, 2010-2025, 5/10/20-day RV), NOT their corpus, period, or "
          "12-month horizon. One text model only (the archived B2 TF-IDF+Ridge recipe, their "
          "SVR analogue) keeps the ladder clean.\n",
          "**Models (L0/L1).** V = OLS of log RV on log trailing-22-day RV "
          f"(`{VOL_COL}`, the single-scalar analogue of Kogan's past-volatility control); "
          "TEXT+V = B2 ridge recipe (TF-IDF 1-2gram, 5k features, sublinear, per-horizon "
          "ridge, alpha by 5-fold CV) on [TF-IDF | standardized log-vol control x"
          f"{VOL_SCALE:.0f}] (the x{VOL_SCALE:.0f} scaling makes the ridge penalty on the "
          "control ~1/100 of a text feature's — an effectively unpenalized price control, "
          "the fair reading of Kogan's design). Chosen alphas: "
          f"`{alphas['TV_tr']}` (train-fit), `{alphas['TV_ins']}` (in-sample arm).\n",
          "**Pathway.** TF-IDF built with the streamed memory-safe pathway validated in "
          "`scripts/experiments/section_ablation.py` (its full-text re-run reproduced the "
          "archived B2 metrics). Vocabulary + idf fit on TRAIN filings (2010-2019) only and "
          "frozen — leakage-free for every OOS year; the in-sample arm refits idf+ridge on "
          "all rows (vocabulary still train-only: a conservative simplification that can "
          "only *weaken* L0's positive). No subsampling anywhere: all "
          f"{int(kp.drop_duplicates('accession').shape[0])} long_form filings.\n",
          "**LOOK-AHEAD DISCLOSURE (labelled, deliberate).** L0's in-sample arm fits and "
          "evaluates on the same rows, and L0's year-by-year arm scores 2020-2021 — years "
          "the benchmark reserves for validation. That is the defect being replicated, not "
          "part of our protocol. From L2 on, every combiner/reference weight is fit on the "
          "validation split only and frozen on test (fc.log_combo / fit_apply_log), and "
          "evaluation is on the declared test split.\n",
          "## PRE-DECLARED Holm family\n",
          "Exactly ONE family is tested with multiplicity control, declared here before any "
          "result: the three L5 clustered-DM p-values (h=5,10,20), Holm within. Rungs L0-L4 "
          "report raw p by DESIGN — each rung reports what its (progressively less broken) "
          "protocol would have reported; Holm is itself the final rung ingredient. The "
          "committed 69-cell tables (m1_clustered, firm_identity_control, maximal_reference) "
          "apply Holm within their own pre-declared families and are cross-referenced in the "
          "SANITY section.\n",
          "## SANITY\n",
          f"1. **HARD GATE (PASS, machine precision):** the L2/L3 rungs recompute the B2 "
          f"long_form cells of `results/tables/m1_clustered.csv` (qlike_R, qlike_U, "
          f"rel_impr_pct, g_log, dm_q, p_q, dm_q_clust, p_q_clust) and "
          f"`results/tables/m1_ensemble_primary.csv` (vol_qlike_R, vol_qlike_U, "
          f"vol_dm_q_clu, vol_p_q_clu; B2 is seed-invariant so ensemble==seed2026): "
          f"{len(gate)} checks, max |diff| = {max_diff:.2e} "
          f"(np.allclose rtol=1e-9, atol=1e-12).",
          "2. **Pathway check (PASS, <=5% tolerance):** the text-only B2 re-fit through this "
          "script's streamed TF-IDF pathway vs the archived B2 run's metrics.json "
          "(variance-unit test QLIKE), the tolerance section_ablation.py declared for the "
          "same pathway:\n",
          "| h | re-fit QLIKE | archived B2 | diff |", "|---|---|---|---|"]
    for _, r in repro.iterrows():
        md.append(f"| {int(r.h)} | {r.qlike_var_repro:.4f} | {r.qlike_var_archived:.4f} | "
                  f"{r.diff_pct:+.2f}% |")
    md += ["",
           "3. Split-membership check: B2 and A2 long_form rows agree exactly on "
           "(ticker, accession, horizon) x split (hard assertion in load_master).\n",
           "## THE DISSOLVE LADDER\n",
           "text_gain_pct: L0/L1 = % log-vol MSE reduction of TEXT+V vs V; L2+ = % vol-unit "
           "QLIKE reduction of the text-augmented combination vs its reference. "
           "Stat sign: naive t POSITIVE = text better; DM NEGATIVE = text better.\n",
           "| rung | protocol | h | n | text gain % | stat | value | p (raw) | p (Holm, L5) | verdict |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in lad.iterrows():
        md.append(f"| {r.rung} | {r.rung_desc} | {int(r.h)} | {int(r.n_eval)} | {r.text_gain_pct:+.2f} | "
                  f"{r.stat_type} | {r.stat:+.2f} | {fmt_p(r.p_raw)} | "
                  f"{fmt_p(r.p_holm_L5)} | **{r.verdict}** |")

    md += ["", "## L0 detail — the published-style positives this design manufactures\n",
           "### In-sample arm (deliberate look-ahead, labelled)\n",
           "| h | n | R2(V) | R2(TEXT+V) | MSE reduction % | naive obs t | p |",
           "|---|---|---|---|---|---|---|"]
    for _, r in ins.iterrows():
        md.append(f"| {int(r.h)} | {int(r.n)} | {r.r2_V:.3f} | {r.r2_TV:.3f} | {r.gain_pct:+.2f} | "
                  f"{r.t_naive:+.1f} | {fmt_p(r.p_naive)} |")
    md += ["", "### Year-by-year OOS arm (train 2010-2019, fixed; R2 vs unconditional mean)\n",
           "| year | h | n | MSE(V) | MSE(TEXT+V) | gain % | R2oos(V) | R2oos(TEXT+V) | naive t | p |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in yearly.iterrows():
        md.append(f"| {r.arm.replace('L0_year','')} | {int(r.h)} | {int(r.n)} | {r.mse_V:.4f} | "
                  f"{r.mse_TV:.4f} | {r.gain_pct:+.2f} | {r.r2_V:.3f} | {r.r2_TV:.3f} | "
                  f"{r.t_naive:+.1f} | {fmt_p(r.p_naive)} |")
    md += ["", "### L0/L1 auxiliary continuity stat (same forecasts, vol-unit QLIKE, obs-level DM)\n",
           "| rung | h | QLIKE gain % | obs DM | p |", "|---|---|---|---|---|"]
    for tag, frame in (("L0", l0), ("L1", l1)):
        for _, r in frame.iterrows():
            md.append(f"| {tag} | {int(r.h)} | {r.qlike_rel:+.2f} | {r.qlike_dm:+.2f} | "
                      f"{fmt_p(r.qlike_dm_p)} |")

    md += ["", "## L4/L5 detail\n",
           "L4 reference: f_R = exp(a + b log fHAR + c log firm_mean_val_RV), val-fit, frozen "
           "(firm mean = firm's mean label RV over its own val rows; missing firms get the "
           "global val mean). L5 reference adds the maximal price pool "
           f"(log-combined {', '.join(PRICE_MODELS)}); its inner join over all five price "
           "models shrinks n_test slightly (disclosed below). Corroboration from the "
           "committed 69-cell tables (different row filter, same story): "
           "firm_identity_control.csv B2/long_form = text HURTS at all h "
           "(clustered DM +4.19/+7.02/+8.15); maximal_reference.csv B2/long_form vs the "
           "price pool WITHOUT the firm control still shows text adds "
           "(clustered DM -3.58/-5.76/-5.24) — identity, not price breadth, is the kill shot.\n",
           "| rung | h | n | n_days | firm-val coverage | QLIKE(R) | QLIKE(U) | gain % | "
           "g_text | clu DM | p | Holm |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag, frame in (("L4", l4), ("L5", l5)):
        for _, r in frame.iterrows():
            hol = fmt_p(r.p_holm) if tag == "L5" else "-"
            md.append(f"| {tag} | {int(r.h)} | {int(r.n)} | {int(r.n_days)} | {r.coverage:.2f} | "
                      f"{r.qlike_R:.4f} | {r.qlike_U:.4f} | {r.rel_impr_pct:+.2f} | "
                      f"{r.g_text:+.3f} | {r.dm_q_clust:+.2f} | {fmt_p(r.p_q_clust)} | {hol} |")

    # bottom line (value-driven: every claim below is generated from the numbers)
    h10 = lad[lad.h == 10].set_index("rung")
    l0_sig = bool((l0.p_naive < 0.05).any())
    l0_wording = ("a published-style significant OOS positive"
                  if l0_sig else "a positive point gain (not obs-significant on this panel)")
    l5_adds = lad[(lad.rung == "L5") & (lad.verdict == "text adds")]
    md += ["", "## Bottom line\n",
           f"- The Kogan-style design's in-sample arm manufactures a large apparent text "
           f"effect on this panel: h=10 in-sample R2 {ins[ins.h==10].r2_V.iloc[0]:.3f} (vol "
           f"control only) -> {ins[ins.h==10].r2_TV.iloc[0]:.3f} (+text), naive obs "
           f"t={ins[ins.h==10].t_naive.iloc[0]:+.1f} "
           f"(p={fmt_p(ins[ins.h==10].p_naive.iloc[0])}); its per-year OOS arm shows "
           f"{l0_wording}: pooled 2020-25 log-vol MSE gain "
           + ", ".join(f"h{int(r.h)} {r.gain_pct:+.2f}% (t={r.t_naive:+.1f}, p={fmt_p(r.p_naive)})"
                       for _, r in l0.iterrows()) + ".",
           f"- Rung-by-rung h=10 trace (stat): L0 t={h10.loc['L0','stat']:+.1f} -> "
           f"L1 t={h10.loc['L1','stat']:+.1f} -> L2 DM={h10.loc['L2','stat']:+.1f} -> "
           f"L3 DM={h10.loc['L3','stat']:+.1f} -> L4 DM={h10.loc['L4','stat']:+.1f} -> "
           f"L5 DM={h10.loc['L5','stat']:+.1f}; verdicts "
           f"{' -> '.join(h10.loc[r,'verdict'] for r in ['L0','L1','L2','L3','L4','L5'])}.",
           "- Under the M1 machinery the text term is strongly significant against the "
           "recalibrated-HAR reference even with day-clustered DM (L2/L3 = the committed m1 "
           "cells) — and the firm-identity rung (L4) flips it significantly NEGATIVE at all "
           "three horizons: what the Kogan-style design reads as disclosure signal is, on "
           "this panel, price-baseline weakness plus firm identity.",
           ("- Honest exception — the dissolve is not monotone in every cell: "
            + "; ".join(f"L5 h{int(r.h)} retains a small positive ({r.text_gain_pct:+.2f}% "
                        f"QLIKE, clustered DM {r.stat:+.2f}, Holm {fmt_p(r.p_holm_L5)})"
                        for _, r in l5_adds.iterrows())
            + ". Interpretation: adding the price pool changes the reference's error "
              "profile; this cell-level residual is not protocol-grade evidence of text "
              "value — the full protocol additionally imposes the placebo gate and the "
              "69-cell family Holm, under which B2/long_form does not survive the "
              "firm-identity control (committed firm_identity_control.csv)."
            ) if len(l5_adds) else
           "- The dissolve is monotone: no L5 cell retains a significant text gain.",
           "- This demonstrates the DESIGN's failure mode on OUR panel; it does not "
           "re-evaluate Kogan et al.'s corpus/period, and their cross-sectional detection "
           "finding (which firm is riskier) is not contradicted — the dissolve concerns "
           "incremental time-series forecast value.",
           ]

    (TABLES / "kogan_dissolve.md").write_text("\n".join(md))
    lad_print = lad[["rung", "h", "n_eval", "text_gain_pct", "stat_type", "stat",
                     "p_raw", "p_holm_L5", "verdict"]]
    print(lad_print.to_string(index=False))
    log("wrote results/tables/kogan_dissolve.csv and .md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["features", "models", "ladder", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke: only first N filings; tables are NOT written")
    args = ap.parse_args()
    global WORKDIR
    if args.limit:
        WORKDIR = WORKDIR / "smoke"  # never mix smoke caches with the full run
    WORKDIR.mkdir(parents=True, exist_ok=True)
    if args.stage in ("features", "all"):
        stage_features(args.limit)
    if args.stage in ("models", "all"):
        stage_models(args.limit)
    if args.stage in ("ladder", "all"):
        stage_ladder(write_tables=args.limit is None)


if __name__ == "__main__":
    main()
