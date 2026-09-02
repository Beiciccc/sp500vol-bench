"""ROW 2A — DELTA-TF-IDF (B2d): change-text representation baseline (CPU).

Round-3 remediation (results/REVIEW_ROUND3_FRESH_PANEL.md, MUST-RUN row 2,
Lazy-Prices / levels-objective blind spot): every archived text model encodes
filing LEVELS; this run represents each long-form filing as the DIFFERENCE of
consecutive TF-IDF vectors within the firm's same-form sequence, so the firm-
stable level component is removed by construction.

Recipe (identical to archived B2_tfidf_ridge except the delta representation):
  - TF-IDF: lowercase, 1-2 grams, 5000 features, sublinear tf,
    token_pattern (?u)\\b[a-z]{2,}\\b — vectoriser FIT ON TRAIN-SPLIT DOCS ONLY
    (B2 fits its vectoriser on X_train), applied to all docs.
  - Sequences: per firm (cik), filings ordered by effective_trading_day WITHIN
    form type (a 10-K sequence and a 10-Q sequence per firm; stable tie-break
    filing_time_utc, accession). delta_i = tfidf(doc_i) - tfidf(doc_{i-1}).
  - The FIRST filing of each (cik, form) sequence has no predecessor and is
    EXCLUDED from training and from predictions.parquet (disclosed in
    config.json with per-split counts).
  - One ridge per horizon on log RV (alpha by the shared 5-fold CV, B2 grid),
    retransform exp(raw) * Duan smear (smear = mean(exp(train residuals)) per
    horizon), clipped to the B2 range [0.02, 5.0].
  - Splits: the pinned chronological convention from scripts/train.py
    (_assign_splits on effective_trading_day), identical to Blocks A-D.

Event-driven variant: SKIPPED BY DESIGN — 8-Ks are event-driven, not periodic;
there is no well-defined "previous filing of the same kind" sequence whose
diff isolates disclosure change (consecutive 8-Ks report unrelated events).
Requesting --disclosure event_driven exits with this note.

Run dir (standard conventions): results/runs/B2d_tfidf_delta_full_long_form_seed2026
with predictions.parquet / metrics.json / config.json.

Usage (box, CPU only):
    SP500VOL_DATA_ROOT=... python scripts/experiments/row2_demeaned/delta_tfidf.py
    # dry run on ~200 filings:
    ... delta_tfidf.py --sample-docs 200 --out-root /root/gpu-data/_row2_dryrun
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sp500vol.models.classical_text._fit_utils import (  # noqa: E402
    fit_ridge_cv,
    maybe_log,
)
from sp500vol.models.classical_text._text_dataset import load_texts  # noqa: E402
from sp500vol.utils import configure_logging, get_logger, seed_everything  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _textcache import ensure_texts_available  # noqa: E402

# reuse the canonical data pipeline from scripts/train.py (single source of truth
# for dataset loading, disclosure filtering, split assignment, row validation,
# prediction schema and metrics grouping)
_spec = importlib.util.spec_from_file_location("train_base", REPO_ROOT / "scripts" / "train.py")
train_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_base)

MODEL_ID = "B2d_tfidf_delta"
ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0)  # B2 default grid
MIN_PRED, MAX_PRED = 0.02, 5.0  # B2 retransform clip
MAX_FEATURES = 5000

EVENT_DRIVEN_NOTE = (
    "event_driven (8-K) delta variant SKIPPED by design: 8-Ks are event-driven, "
    "not periodic — consecutive 8-Ks of the same firm report unrelated events, so "
    "a consecutive-filing TF-IDF difference does not isolate disclosure change the "
    "way a 10-K/10-Q same-form diff does (Lazy Prices is defined on periodic "
    "filings). Disclosed in the paper as a scope note."
)


def _build_vectorizer():
    """Byte-identical to sp500vol.models.classical_text.tfidf_ridge.TfidfRidge."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(
        lowercase=True,
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-z]{2,}\b",
    )


def _doc_table(data: pd.DataFrame) -> pd.DataFrame:
    """Unique filings with prev-accession within (cik, form), ordered by
    effective_trading_day (stable tie-break: filing_time_utc, accession)."""
    docs = (
        data.drop_duplicates("accession")[
            ["cik", "ticker", "form", "accession", "effective_trading_day",
             "filing_time_utc", "text_path", "split"]
        ]
        .sort_values(
            ["cik", "form", "effective_trading_day", "filing_time_utc", "accession"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    grp = docs.groupby(["cik", "form"], sort=False)
    docs["prev_accession"] = grp["accession"].shift(1)
    docs["prev_effective_trading_day"] = grp["effective_trading_day"].shift(1)
    docs["gap_days"] = (
        docs["effective_trading_day"] - docs["prev_effective_trading_day"]
    ).dt.days.astype(float)
    docs["has_prev"] = docs["prev_accession"].notna()
    return docs


def _sample_firms(data: pd.DataFrame, n_docs: int) -> pd.DataFrame:
    """Keep whole firms (full sequences) until >= n_docs unique filings."""
    per_firm = data.drop_duplicates("accession").groupby("cik").size().sort_index()
    keep, total = [], 0
    for cik, n in per_firm.items():
        keep.append(cik)
        total += int(n)
        if total >= n_docs:
            break
    return data[data["cik"].isin(keep)].copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="full")
    parser.add_argument(
        "--disclosure", default="long_form", choices=["long_form", "event_driven"]
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--sample-docs", type=int, default=None,
        help="Dry run: keep whole firms until ~N unique filings (full pipeline).",
    )
    parser.add_argument(
        "--out-root", default=None,
        help="Run-dir root (default results/runs). Point dry runs elsewhere.",
    )
    args = parser.parse_args()

    configure_logging("INFO")
    log = get_logger("row2.delta_tfidf")

    if args.disclosure == "event_driven":
        print(f"SKIP: {EVENT_DRIVEN_NOTE}")
        return 0

    seed_everything(args.seed)
    t0 = time.time()

    run_id = f"{MODEL_ID}_{args.dataset}_{args.disclosure}_seed{args.seed}"
    if args.sample_docs:
        run_id += f"_sample{args.sample_docs}"
    out_root = Path(args.out_root) if args.out_root else REPO_ROOT / "results" / "runs"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- canonical data pipeline (identical to scripts/train.py) ------------
    data = train_base._load_dataset(args.dataset)
    data = train_base._filter_disclosure(data, args.disclosure)
    data = train_base._assign_splits(data, args.dataset)
    if args.sample_docs:
        data = _sample_firms(data, args.sample_docs)
    # doc table BEFORE row-validity drop: a filing whose label rows are invalid
    # can still serve as the PREVIOUS document of a delta (its text exists).
    docs = _doc_table(data)
    data = train_base._drop_invalid_rows(data)
    train_base._validate_trainable(data)

    kept = docs[docs["has_prev"]]
    excluded = docs[~docs["has_prev"]]
    log.info(
        "delta sequences built",
        docs=len(docs), kept=len(kept), excluded_first_of_sequence=len(excluded),
        firms=int(docs["cik"].nunique()),
        median_gap_days=float(kept["gap_days"].median()),
    )

    # --- texts + TF-IDF (train-fit vectoriser, B2 recipe) --------------------
    text_source = ensure_texts_available(str(docs["text_path"].iloc[0]))
    log.info("text source", source=text_source)
    texts = load_texts(docs.reset_index(drop=True), persist_new=False)
    train_doc_mask = (docs["split"] == "train").to_numpy()
    vectorizer = _build_vectorizer()
    vectorizer.fit([t for t, keep in zip(texts, train_doc_mask, strict=True) if keep])
    matrix = vectorizer.transform(texts)  # all docs, train-fitted vocab/IDF
    log.info("tfidf built", shape=str(matrix.shape), n_train_fit_docs=int(train_doc_mask.sum()))

    # --- consecutive-filing delta vectors ------------------------------------
    doc_pos = {acc: i for i, acc in enumerate(docs["accession"].astype(str))}
    cur_idx = np.array([doc_pos[a] for a in kept["accession"].astype(str)])
    prev_idx = np.array([doc_pos[a] for a in kept["prev_accession"].astype(str)])
    delta = matrix[cur_idx] - matrix[prev_idx]  # sparse; first-of-sequence excluded
    delta_pos = {acc: j for j, acc in enumerate(kept["accession"].astype(str))}

    # rows (filing x horizon) restricted to delta-covered filings
    rows = data[data["accession"].astype(str).isin(delta_pos)].reset_index(drop=True)
    row_j = np.array([delta_pos[a] for a in rows["accession"].astype(str)])
    y = rows["label_realised_vol"].to_numpy(float)
    horizons = rows["horizon_days"].astype(int).to_numpy()
    split = rows["split"].to_numpy()

    # --- one ridge per horizon on log RV; Duan smear + B2 clip ---------------
    preds = np.empty(len(rows), dtype=float)
    alphas: dict[int, float] = {}
    smears: dict[int, float] = {}
    for h in sorted(set(horizons.tolist())):
        mh = horizons == h
        tr = mh & (split == "train")
        X_tr = delta[row_j[tr]]
        y_tr_log = maybe_log(y[tr], log_target=True)
        ridge = fit_ridge_cv(X_tr, y_tr_log, ALPHA_GRID)
        raw_tr = ridge.predict(X_tr)
        smear = float(np.mean(np.exp(y_tr_log - raw_tr)))  # Duan (1983)
        raw_all = ridge.predict(delta[row_j[mh]])
        preds[mh] = np.clip(np.exp(raw_all) * smear, MIN_PRED, MAX_PRED)
        alphas[int(h)] = float(ridge.alpha_)
        smears[int(h)] = smear
        log.info("horizon fitted", horizon=int(h), n_train=int(tr.sum()),
                 alpha=alphas[int(h)], smear=round(smear, 4))

    # --- standard run-dir outputs (train.py conventions) ----------------------
    predictions = rows.copy()
    predictions["prediction_realised_vol"] = preds
    predictions["run_id"] = run_id
    predictions["model_id"] = MODEL_ID
    predictions["dataset"] = args.dataset
    predictions["seed"] = args.seed
    predictions["disclosure_subset"] = args.disclosure
    predictions["feature_rv_1d"] = train_base._feature_rv_1d(predictions)
    cols = train_base._prediction_columns(predictions)
    predictions[cols].to_parquet(run_dir / "predictions.parquet", index=False)

    metrics = train_base._metrics_by_group(predictions)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    excl_by_split = excluded["split"].value_counts().to_dict()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model": MODEL_ID,
                "dataset": args.dataset,
                "disclosure": args.disclosure,
                "seed": args.seed,
                "sample_docs": args.sample_docs,
                "model_config": {
                    "model_id": MODEL_ID,
                    "note": (
                        "ROW-2A delta-TF-IDF: each long-form filing represented as the "
                        "difference of consecutive TF-IDF vectors within the firm's "
                        "same-form (10-K / 10-Q) sequence ordered by effective_trading_day; "
                        "vectoriser (B2 recipe: 1-2gram, 5k features, sublinear tf) fit on "
                        "train-split docs only; one ridge per horizon on log RV (B2 alpha "
                        "CV grid); retransform exp(raw)*Duan-smear clipped to [0.02, 5.0]. "
                        "First filing of each (cik, form) sequence excluded (no "
                        "predecessor) from training AND predictions. Splits: pinned "
                        "chronological convention from scripts/train.py."
                    ),
                    "vectoriser": {
                        "kind": "tfidf", "ngram_range": [1, 2],
                        "max_features": MAX_FEATURES, "sublinear_tf": True,
                        "fit_on": "train-split unique docs",
                    },
                    "training": {
                        "log_target": True, "ridge_alpha": None,
                        "alpha_grid": list(ALPHA_GRID),
                        "selected_alphas": alphas,
                        "duan_smear": smears,
                        "clip": [MIN_PRED, MAX_PRED],
                    },
                },
                "sequencing": {
                    "order_by": "effective_trading_day within (cik, form)",
                    "n_unique_docs": int(len(docs)),
                    "n_kept_docs": int(len(kept)),
                    "n_excluded_first_of_sequence": int(len(excluded)),
                    "excluded_by_split": {str(k): int(v) for k, v in excl_by_split.items()},
                    "gap_days_median": float(kept["gap_days"].median()),
                    "gap_days_p95": float(kept["gap_days"].quantile(0.95)),
                    "no_gap_cap": "consecutive same-form filings regardless of gap (disclosed)",
                },
                "event_driven_note": EVENT_DRIVEN_NOTE,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    log.info(
        "row2a done", run_dir=str(run_dir), rows=len(predictions),
        secs=round(time.time() - t0, 1),
    )
    print(f"WROTE {run_dir} rows={len(predictions)} alphas={alphas} smears="
          f"{ {k: round(v, 4) for k, v in smears.items()} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
