"""E2 — Section ablation for the long_form disclosure subset.

Which 10-K/10-Q sections carry the incremental text signal?

Background finding (established in this script's development):
  ``sections_json`` in aligned_filings.parquet stores section TEXT (not char
  offsets) produced by ``sp500vol.data.parser._extract_section``, which takes
  the FIRST regex match of the item label. In filings with a table of contents
  the first match is the TOC line, so the stored value is usually just a
  heading + page number (median 24-96 chars; only ~10-17% of filings captured
  substantive text). We therefore RE-EXTRACT sections from the cached full
  text with a TOC-robust rule: scan label matches in order and accept the
  first span >= MIN_SECTION_CHARS; if none qualifies, fall back to the last
  match's span (the body heading usually follows the TOC). Because we extract
  spans (offsets) ourselves, variant (d) "rest = full text minus sections" is
  possible.

Section mapping per form (mirrors the dataset parser's label grammar):
  10-K: item1a = "item 1a" -> ["item 1b","item 2"]     (Risk Factors)
        item7  = "item 7"  -> ["item 7a","item 8"]     (MD&A)
        item7a = "item 7a" -> ["item 8"]               (market risk)
  10-Q: item1a = "item 1a" -> ["item 2","item 3"]      (Part II Item 1A)
        item7  = "item 2"  -> ["item 3","item 4"]      (Part I Item 2, MD&A)
        item7a = (not defined by the dataset parser)   -> empty string

Model recipe replicates B2_tfidf_ridge exactly (read from
src/sp500vol/models/classical_text/{tfidf_ridge,bow_ridge,_fit_utils}.py):
  TfidfVectorizer(lowercase=True, max_features=5000, ngram_range=(1,2),
                  sublinear_tf=True, token_pattern=r"(?u)\b[a-z]{2,}\b")
  fitted on unique TRAIN filings (dedup by accession); one Ridge per horizon
  on log(y + 1e-12), alpha by 5-fold KFold(shuffle=True, random_state=0) CV
  over (0.1,1,10,100,1000,10000), Ridge(solver="lsqr", tol=1e-4,
  max_iter=1000); predictions clipped in log space to [log(0.02), log(5)]
  then exponentiated (imported directly from the repo:
  fit_ridge_cv / maybe_log / maybe_exp).

Memory note: the original B2 run used sklearn's in-RAM vocabulary build on a
large-RAM server. On a 16GB Mac the full 1-2gram vocabulary (tens of millions
of bigrams over an ~8.6GB corpus) does not fit, so we replicate max_features
selection with a streamed, chunk-parallel term-frequency count (exact counts;
safe pruning of within-chunk hapax terms only — the top-5000 cutoff is orders
of magnitude above the prune threshold), then use sklearn
CountVectorizer(vocabulary=top5000) + TfidfTransformer(sublinear_tf=True),
which is mathematically identical to TfidfVectorizer given that vocabulary.
The B2 full-text re-run ("B2sec_fullrepro") sanity-checks this approximation
against the archived B2 run (required to agree within ~5% test QLIKE).

Run from repo root:
  .venv/bin/python scripts/experiments/section_ablation.py --stage all
Stages: extract (section store), fit (models + run dirs), evaluate (tables).
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
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import Parallel, delayed

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from sp500vol.evaluation.dm_test import dm_test  # noqa: E402
from sp500vol.models.classical_text._fit_utils import (  # noqa: E402
    fit_ridge_cv,
    maybe_exp,
    maybe_log,
)

ALIGNED = Path("/Volumes/Z/sp500vol-data/processed/full/aligned_filings.parquet")
TEXT_CACHE = Path("/Volumes/Z/sp500vol-data/processed/_text_cache/filing_texts.parquet")
RUNS = REPO / "results" / "runs"
TABLES = REPO / "results" / "tables"
DEFAULT_WORKDIR = Path(os.environ.get("SP500VOL_SCRATCH", tempfile.gettempdir())) / "section_ablation"

DISCLOSURE = "long_form"
SEED = 2026
HORIZONS = (5, 10, 20)
MIN_SECTION_CHARS = 500  # spans shorter than this are treated as TOC lines
MAX_FEATURES = 5000
ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0)
MIN_PRED, MAX_PRED = 0.02, 5.0
TOKEN_RE = re.compile(r"(?u)\b[a-z]{2,}\b")
CHUNK_DOCS = 256
N_JOBS = 6
VOCAB_DICT_CAP = 12_000_000  # global prune trigger for streamed TF counting
KEY = ["ticker", "accession", "horizon_days"]
SORT = ["filing_time_utc", "ticker", "accession"]
EPS = 1e-8

SECTION_SPECS = {
    "10-K": {
        "item1a": ("item 1a", ["item 1b", "item 2"]),
        "item7": ("item 7", ["item 7a", "item 8"]),
        "item7a": ("item 7a", ["item 8"]),
    },
    "10-Q": {
        "item1a": ("item 1a", ["item 2", "item 3"]),
        "item7": ("item 2", ["item 3", "item 4"]),
        # item7a (market risk) is not defined by the dataset parser for 10-Q
    },
}
SECTIONS_JSON_KEYMAP = {  # aligned sections_json key -> our section name
    "10-K": {"item_1a": "item1a", "item_7": "item7", "item_7a": "item7a"},
    "10-Q": {"part_ii_item_1a": "item1a", "part_i_item_2": "item7"},
}
VARIANTS = {  # model_id -> store column
    "B2sec_item1a": "sec_item1a",
    "B2sec_item7": "sec_item7",
    "B2sec_item7a": "sec_item7a",
    "B2sec_rest": "sec_rest",
    "B2sec_fullrepro": "full",  # sanity re-run of B2 recipe; NOT written to results/runs
}
PUBLISH = ("B2sec_item1a", "B2sec_item7", "B2sec_item7a", "B2sec_rest")

VARIANT_NOTES = {
    "B2sec_item1a": "Risk Factors only (10-K Item 1A / 10-Q Part II Item 1A)",
    "B2sec_item7": "MD&A only (10-K Item 7 / 10-Q Part I Item 2)",
    "B2sec_item7a": "Market risk only (10-K Item 7A; empty for 10-Q)",
    "B2sec_rest": "Full text MINUS Item 1A + MD&A + Item 7A spans",
    "B2sec_fullrepro": "B2 recipe re-run on full text (sanity, memory-safe vocab path)",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# stage 1: extraction
# --------------------------------------------------------------------------

def _label_pattern(label: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(label).replace(r"\ ", r"\s+") + r"\b")


def _find_spans(low: str, start_label: str, end_labels: list[str]) -> list[tuple[int, int]]:
    spans = []
    for m in _label_pattern(start_label).finditer(low):
        end = len(low)
        for el in end_labels:
            em = _label_pattern(el).search(low, m.end())
            if em is not None:
                end = min(end, em.start())
        spans.append((m.start(), end))
    return spans


def _pick_span(spans: list[tuple[int, int]]) -> tuple[int, int] | None:
    """First span >= MIN_SECTION_CHARS (skips TOC lines); else last match."""
    for s, e in spans:
        if e - s >= MIN_SECTION_CHARS:
            return (s, e)
    return spans[-1] if spans else None


def extract_one(text: str, form: str):
    low = text.lower()
    secs: dict[str, str] = {"item1a": "", "item7": "", "item7a": ""}
    spans: dict[str, tuple[int, int]] = {}
    for name, (start_label, end_labels) in SECTION_SPECS[form].items():
        span = _pick_span(_find_spans(low, start_label, end_labels))
        if span is not None:
            spans[name] = span
            secs[name] = text[span[0] : span[1]].strip()
    # rest = complement of the union of accepted spans
    ivals = sorted(spans.values())
    merged: list[list[int]] = []
    for s, e in ivals:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    parts, pos = [], 0
    for s, e in merged:
        if s > pos:
            parts.append(text[pos:s])
        pos = max(pos, e)
    if pos < len(text):
        parts.append(text[pos:])
    secs_rest = " ".join(p.strip() for p in parts if p.strip())
    return secs, secs_rest, spans


def _load_reference_sections(needed: set[str]) -> dict[str, dict[str, tuple[int, str]]]:
    """text_path -> {section: (len, first 300 chars)} from aligned sections_json."""
    ref: dict[str, dict[str, tuple[int, str]]] = {}
    f = pq.ParquetFile(ALIGNED)
    for batch in f.iter_batches(batch_size=2048, columns=["form", "text_path", "sections_json"]):
        d = batch.to_pandas()
        d = d[d["form"].isin(["10-K", "10-Q"])]
        for tp, form, sj in zip(d["text_path"], d["form"], d["sections_json"]):
            if tp in ref or tp not in needed:
                continue
            try:
                parsed = json.loads(sj)
            except (TypeError, ValueError):
                continue
            entry = {}
            for k, name in SECTIONS_JSON_KEYMAP[form].items():
                v = parsed.get(k, "")
                entry[name] = (len(v), v[:300])
            ref[tp] = entry
    return ref


def stage_extract(workdir: Path, limit: int | None) -> None:
    store_path = workdir / "sections_store.parquet"
    diag_path = workdir / "extract_diag.json"
    if store_path.exists() and diag_path.exists():
        log(f"extract: {store_path} exists, skipping")
        return
    master = load_master()
    filings = master.drop_duplicates("accession")[["accession", "text_path", "form"]]
    needed = dict(zip(filings["text_path"].astype(str), filings["form"].astype(str)))
    if limit:
        needed = dict(list(needed.items())[:limit])
    log(f"extract: {len(needed)} unique text_paths needed")
    ref = _load_reference_sections(set(needed))
    log(f"extract: loaded sections_json reference for {len(ref)} filings")

    schema = pa.schema(
        [
            ("text_path", pa.string()),
            ("form", pa.string()),
            ("sec_item1a", pa.large_string()),
            ("sec_item7", pa.large_string()),
            ("sec_item7a", pa.large_string()),
            ("sec_rest", pa.large_string()),
            ("full", pa.large_string()),
        ]
    )
    writer = pq.ParquetWriter(store_path.with_suffix(".tmp"), schema, compression="zstd")
    agree = Counter()
    lens_rows = []
    seen: set[str] = set()
    f = pq.ParquetFile(TEXT_CACHE)
    n_done = 0
    for batch in f.iter_batches(batch_size=CHUNK_DOCS, columns=["text_path", "text"]):
        d = batch.to_pandas()
        d = d[d["text_path"].isin(needed.keys())]
        if d.empty:
            continue
        rows = {k: [] for k in schema.names}
        for tp, text in zip(d["text_path"].astype(str), d["text"].astype(str)):
            form = needed[tp]
            secs, rest, spans = extract_one(text, form)
            rows["text_path"].append(tp)
            rows["form"].append(form)
            rows["sec_item1a"].append(secs["item1a"])
            rows["sec_item7"].append(secs["item7"])
            rows["sec_item7a"].append(secs["item7a"])
            rows["sec_rest"].append(rest)
            rows["full"].append(text)
            seen.add(tp)
            lens_rows.append(
                (tp, form, len(secs["item1a"]), len(secs["item7"]), len(secs["item7a"]),
                 len(rest), len(text))
            )
            # agreement vs stored sections_json where it captured substantive text
            r = ref.get(tp, {})
            for name in ("item1a", "item7", "item7a"):
                rl, rhead = r.get(name, (0, ""))
                if rl > 2000:
                    agree[f"{name}_n"] += 1
                    if rhead in secs[name]:
                        agree[f"{name}_ok"] += 1
        writer.write_table(pa.table(rows, schema=schema))
        n_done += len(d)
        if n_done % 5120 < CHUNK_DOCS:
            log(f"extract: {n_done}/{len(needed)}")
        if limit and n_done >= len(needed):
            break
    # any filings missing from the cache: read the .txt directly
    missing = set(needed) - seen
    if missing:
        log(f"extract: {len(missing)} paths missing from cache, reading files")
        rows = {k: [] for k in schema.names}
        for tp in sorted(missing):
            text = Path(tp).read_text(encoding="utf-8", errors="replace")
            form = needed[tp]
            secs, rest, spans = extract_one(text, form)
            rows["text_path"].append(tp)
            rows["form"].append(form)
            rows["sec_item1a"].append(secs["item1a"])
            rows["sec_item7"].append(secs["item7"])
            rows["sec_item7a"].append(secs["item7a"])
            rows["sec_rest"].append(rest)
            rows["full"].append(text)
            lens_rows.append(
                (tp, form, len(secs["item1a"]), len(secs["item7"]), len(secs["item7a"]),
                 len(rest), len(text))
            )
        writer.write_table(pa.table(rows, schema=schema))
    writer.close()
    store_path.with_suffix(".tmp").replace(store_path)

    lens = pd.DataFrame(
        lens_rows,
        columns=["text_path", "form", "len_item1a", "len_item7", "len_item7a", "len_rest", "len_full"],
    )
    lens.to_parquet(workdir / "section_lengths.parquet", index=False)
    diag = {
        "n_filings": len(lens),
        "agreement_vs_sections_json": {
            name: {
                "n_substantive_ref": int(agree[f"{name}_n"]),
                "n_head_recovered": int(agree[f"{name}_ok"]),
                "pct": (100.0 * agree[f"{name}_ok"] / agree[f"{name}_n"]) if agree[f"{name}_n"] else None,
            }
            for name in ("item1a", "item7", "item7a")
        },
    }
    diag_path.write_text(json.dumps(diag, indent=2))
    log(f"extract: done -> {store_path} ({len(lens)} filings); diag: {diag['agreement_vs_sections_json']}")


# --------------------------------------------------------------------------
# stage 2: fit variants
# --------------------------------------------------------------------------

def load_master() -> pd.DataFrame:
    """Row table + split assignment: the archived B2 long_form run (whose
    membership is verified identical to A2_har_rv long_form on KEY+split)."""
    b2 = pd.read_parquet(RUNS / f"B2_tfidf_ridge_full_{DISCLOSURE}_seed{SEED}" / "predictions.parquet")
    a2 = pd.read_parquet(
        RUNS / f"A2_har_rv_full_{DISCLOSURE}_seed{SEED}" / "predictions.parquet",
        columns=KEY + ["split"],
    )
    chk = b2[KEY + ["split"]].merge(a2, on=KEY, suffixes=("_b2", "_a2"), how="outer", indicator=True)
    if not ((chk["_merge"] == "both").all() and (chk["split_b2"] == chk["split_a2"]).all()):
        raise RuntimeError("B2 and A2 long_form split membership differ — aborting")
    return b2


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
    )
    return cv.transform(texts)


def _iter_store_chunks(store_path: Path, col: str):
    f = pq.ParquetFile(store_path)
    for batch in f.iter_batches(batch_size=CHUNK_DOCS, columns=["text_path", col]):
        d = batch.to_pandas()
        yield d["text_path"].astype(str).tolist(), d[col].astype(str).tolist()


def build_vocab(store_path: Path, col: str, train_paths: set[str], prune_hapax: bool) -> list[str]:
    def gen_tasks():
        for paths, texts in _iter_store_chunks(store_path, col):
            chunk = [t for p, t in zip(paths, texts) if p in train_paths]
            if chunk:
                yield delayed(_count_chunk)(chunk, prune_hapax)

    merged: Counter = Counter()
    threshold = 2
    results = Parallel(n_jobs=N_JOBS, return_as="generator")(gen_tasks())
    for part in results:
        merged.update(part)
        if len(merged) > VOCAB_DICT_CAP:
            merged = Counter({k: v for k, v in merged.items() if v >= threshold})
            threshold *= 2
            log(f"  vocab: pruned to {len(merged)} (next threshold {threshold})")
    top = sorted(merged.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_FEATURES]
    return sorted(k for k, _ in top)


def build_matrix(store_path: Path, col: str, vocab: list[str]):
    from scipy import sparse

    order: list[str] = []

    def gen_tasks():
        for paths, texts in _iter_store_chunks(store_path, col):
            order.extend(paths)
            yield delayed(_transform_chunk)(texts, vocab)

    mats = list(Parallel(n_jobs=N_JOBS, return_as="generator")(gen_tasks()))
    return sparse.vstack(mats, format="csr"), order


def fit_variant(model_id: str, store_path: Path, master: pd.DataFrame, workdir: Path,
                publish_root: Path) -> None:
    col = VARIANTS[model_id]
    run_id = f"{model_id}_full_{DISCLOSURE}_seed{SEED}"
    out_dir = (publish_root if model_id in PUBLISH else workdir / "runs") / run_id
    if (out_dir / "predictions.parquet").exists():
        log(f"fit {model_id}: exists, skipping")
        return
    t0 = time.time()
    train_paths = set(master.loc[master["split"] == "train", "text_path"].astype(str))
    log(f"fit {model_id}: vocab pass over {len(train_paths)} train filings (col={col})")
    prune = col in ("full", "sec_rest", "sec_item7", "sec_item1a")
    vocab = build_vocab(store_path, col, train_paths, prune_hapax=prune)
    log(f"fit {model_id}: vocab={len(vocab)} terms ({time.time()-t0:.0f}s); transform pass")
    X_counts, order = build_matrix(store_path, col, vocab)
    row_of_path = {p: i for i, p in enumerate(order)}
    log(f"fit {model_id}: counts {X_counts.shape}, nnz={X_counts.nnz} ({time.time()-t0:.0f}s)")

    from sklearn.feature_extraction.text import TfidfTransformer

    train_rows = sorted({row_of_path[p] for p in train_paths if p in row_of_path})
    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
    tfidf.fit(X_counts[train_rows])
    X = tfidf.transform(X_counts)
    del X_counts

    filing_idx = master["text_path"].astype(str).map(row_of_path).to_numpy()
    if np.isnan(filing_idx.astype(float)).any():
        raise RuntimeError(f"{model_id}: some master rows missing from section store")
    filing_idx = filing_idx.astype(int)
    y = master["label_realised_vol"].to_numpy(dtype=float)
    horizons = master["horizon_days"].astype(int).to_numpy()
    split = master["split"].to_numpy()

    preds = np.empty(len(master), dtype=float)
    alphas = {}
    for h in sorted(set(horizons.tolist())):
        m_tr = (horizons == h) & (split == "train")
        ridge = fit_ridge_cv(X[filing_idx[m_tr]], maybe_log(y[m_tr], log_target=True), ALPHA_GRID)
        alphas[int(h)] = float(ridge.alpha_)
        m_all = horizons == h
        raw = ridge.predict(X[filing_idx[m_all]])
        preds[m_all] = maybe_exp(raw, log_target=True, min_pred=MIN_PRED, max_pred=MAX_PRED)
        log(f"fit {model_id}: h={h} alpha={ridge.alpha_} cv_mse={ridge.cv_mse_:.5f}")

    out = master.copy()
    out["run_id"] = run_id
    out["model_id"] = model_id
    out["prediction_realised_vol"] = preds
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "predictions.parquet", index=False)

    metrics = compute_metrics(out)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    lens = pd.read_parquet(workdir / "section_lengths.parquet")
    lens_col = {"sec_item1a": "len_item1a", "sec_item7": "len_item7",
                "sec_item7a": "len_item7a", "sec_rest": "len_rest", "full": "len_full"}[col]
    fil = master.drop_duplicates("accession")[["text_path", "form"]].astype(str)
    fil = fil.merge(lens[["text_path", lens_col]], on="text_path", how="left")
    frac = {
        "all": float((fil[lens_col] > 0).mean()),
        "10-K": float((fil.loc[fil["form"] == "10-K", lens_col] > 0).mean()),
        "10-Q": float((fil.loc[fil["form"] == "10-Q", lens_col] > 0).mean()),
    }
    config = {
        "model_id": model_id,
        "note": (
            f"E2 section ablation ({VARIANT_NOTES[model_id]}). B2_tfidf_ridge recipe: "
            f"TF-IDF 1-2grams max_features=5000 sublinear_tf, per-horizon Ridge(lsqr) on "
            f"log target, alpha CV grid {ALPHA_GRID} -> chosen {alphas}; sections re-extracted "
            f"from cached full text (sections_json is TOC-contaminated first-match text), "
            f"TOC-skip rule: first label-match span >= {MIN_SECTION_CHARS} chars else last match. "
            f"Missing section => empty string. Non-empty section fraction: {frac}. "
            f"Splits joined from A2_har_rv {DISCLOSURE} seed{SEED}."
        ),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    log(f"fit {model_id}: done -> {out_dir} ({time.time()-t0:.0f}s)")


def compute_metrics(pred_df: pd.DataFrame) -> list[dict]:
    rows = []
    for (sp, h), g in pred_df.groupby(["split", "horizon_days"], sort=True):
        y = g["label_realised_vol"].to_numpy(dtype=float)
        f = g["prediction_realised_vol"].to_numpy(dtype=float)
        r = np.clip(y**2, 1e-12, None) / np.clip(f**2, 1e-12, None)
        rows.append(
            {
                "split": sp,
                "disclosure_subset": DISCLOSURE,
                "horizon_days": int(h),
                "n": int(len(g)),
                "mae": float(np.mean(np.abs(y - f))),
                "rmse": float(np.sqrt(np.mean((y - f) ** 2))),
                "r2": float(1 - np.sum((y - f) ** 2) / np.sum((y - y.mean()) ** 2)),
                "qlike": float(np.mean(r - np.log(r) - 1.0)),
            }
        )
    return rows


def stage_fit(workdir: Path, publish_root: Path, smoke: bool = False) -> None:
    store_path = workdir / "sections_store.parquet"
    master = load_master()
    if smoke:  # keep only rows whose filing made it into the (possibly limited) store
        store_paths = set(
            pq.read_table(store_path, columns=["text_path"])["text_path"].to_pylist()
        )
        master = master[master["text_path"].astype(str).isin(store_paths)].reset_index(drop=True)
        log(f"fit(smoke): master reduced to {len(master)} rows")
    for model_id in VARIANTS:
        fit_variant(model_id, store_path, master, workdir, publish_root)


# --------------------------------------------------------------------------
# stage 3: evaluate
# --------------------------------------------------------------------------

def qlike_vol(y, f):
    y = np.clip(np.asarray(y, float), EPS, None)
    f = np.clip(np.asarray(f, float), EPS, None)
    return y / f - np.log(y / f) - 1.0


def stage_evaluate(workdir: Path, publish_root: Path) -> None:
    import forecast_combination as fc  # scripts/analysis

    lens = pd.read_parquet(workdir / "section_lengths.parquet")
    master = load_master()
    fil = master.drop_duplicates("accession")[["text_path"]].astype(str)
    fil = fil.merge(lens, on="text_path", how="inner")  # lens carries 'form'

    har = pd.read_parquet(
        RUNS / f"A2_har_rv_full_{DISCLOSURE}_seed{SEED}" / "predictions.parquet",
        columns=["split"] + KEY + ["prediction_realised_vol", "label_realised_vol", "filing_time_utc"],
    ).rename(columns={"prediction_realised_vol": "fhar"})

    b2_qlike = {
        (m["split"], m["horizon_days"]): m["qlike"]
        for m in json.loads(
            (RUNS / f"B2_tfidf_ridge_full_{DISCLOSURE}_seed{SEED}" / "metrics.json").read_text()
        )
    }

    def load_preds(model_id: str) -> pd.DataFrame:
        if model_id == "B2_tfidf_ridge":
            p = RUNS / f"B2_tfidf_ridge_full_{DISCLOSURE}_seed{SEED}"
        elif model_id in PUBLISH:
            p = publish_root / f"{model_id}_full_{DISCLOSURE}_seed{SEED}"
        else:
            p = workdir / "runs" / f"{model_id}_full_{DISCLOSURE}_seed{SEED}"
        return pd.read_parquet(p / "predictions.parquet")

    len_col = {"B2sec_item1a": "len_item1a", "B2sec_item7": "len_item7",
               "B2sec_item7a": "len_item7a", "B2sec_rest": "len_rest",
               "B2sec_fullrepro": "len_full", "B2_tfidf_ridge": "len_full"}
    rows = []
    for model_id in ["B2_tfidf_ridge", "B2sec_fullrepro", "B2sec_item1a", "B2sec_item7",
                     "B2sec_item7a", "B2sec_rest"]:
        pred = load_preds(model_id)
        lc = len_col[model_id]
        frac_all = float((fil[lc] > 0).mean())
        frac_k = float((fil.loc[fil["form"] == "10-K", lc] > 0).mean())
        frac_q = float((fil.loc[fil["form"] == "10-Q", lc] > 0).mean())
        txt = pred[KEY + ["prediction_realised_vol"]].rename(columns={"prediction_realised_vol": "ftext"})
        d = har.merge(txt, on=KEY)
        for h in HORIZONS:
            te = pred[(pred["split"] == "test") & (pred["horizon_days"] == h)]
            y = te["label_realised_vol"].to_numpy(float)
            f = te["prediction_realised_vol"].to_numpy(float)
            r = np.clip(y**2, 1e-12, None) / np.clip(f**2, 1e-12, None)
            q_var = float(np.mean(r - np.log(r) - 1.0))
            dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(SORT, kind="mergesort")
            dt = d[(d.horizon_days == h) & (d.split == "test")].sort_values(SORT, kind="mergesort")
            if len(dv) < 100 or len(dt) < 30:  # same guard as forecast_combination.py
                qR = qU = rel = g_text = dm_stat = dm_p = float("nan")
            else:
                yv, fhv, ftv = dv.label_realised_vol.to_numpy(), dv.fhar.to_numpy(), dv.ftext.to_numpy()
                yt, fhr, ftt = dt.label_realised_vol.to_numpy(), dt.fhar.to_numpy(), dt.ftext.to_numpy()
                fR, fU, g_text = fc.log_combo(yv, fhv, ftv, fhr, ftt)
                lR, lU = fc.qlike(yt, fR), fc.qlike(yt, fU)
                qR, qU = float(lR.mean()), float(lU.mean())
                rel = 100.0 * (qR - qU) / qR if qR > 0 else float("nan")
                dm_stat, dm_p = dm_test(lU, lR, h=h)  # negative stat => text adds
            rows.append(
                {
                    "model_id": model_id,
                    "section": VARIANT_NOTES.get(model_id, "full text (archived B2 run)"),
                    "horizon_days": h,
                    "n_test": int(len(te)),
                    "frac_nonempty_all": round(frac_all, 4),
                    "frac_nonempty_10K": round(frac_k, 4),
                    "frac_nonempty_10Q": round(frac_q, 4),
                    "qlike_test_var": round(q_var, 4),
                    "qlike_vs_B2_pct": round(100.0 * (q_var - b2_qlike[("test", h)]) / b2_qlike[("test", h)], 2),
                    "m1_qlike_fR": round(qR, 5),
                    "m1_qlike_fU": round(qU, 5),
                    "m1_rel_improve_pct": round(rel, 3),
                    "m1_g_text": round(g_text, 4),
                    "m1_dm_stat": round(dm_stat, 3),
                    "m1_dm_p": round(dm_p, 5),
                }
            )
    res = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    res.to_csv(TABLES / "section_ablation.csv", index=False)

    diag = json.loads((workdir / "extract_diag.json").read_text())
    repro = res[res.model_id == "B2sec_fullrepro"]
    ref = res[res.model_id == "B2_tfidf_ridge"]
    sanity_lines = []
    for h in HORIZONS:
        a = float(ref[ref.horizon_days == h]["qlike_test_var"].iloc[0])
        b = float(repro[repro.horizon_days == h]["qlike_test_var"].iloc[0])
        sanity_lines.append(
            f"| {h} | {a:.4f} | {b:.4f} | {100.0*(b-a)/a:+.2f}% |"
        )
    md = [
        "# E2 — Section ablation (long_form, B2 TF-IDF+Ridge recipe, seed 2026)",
        "",
        "Which 10-K/10-Q sections carry the incremental text signal over a recalibrated",
        "HAR-RV (A2)? Four TF-IDF(1-2gram, 5k features, sublinear)+per-horizon-Ridge",
        "variants trained on a single section (or the complement) of each filing,",
        "replicating the archived B2_tfidf_ridge recipe exactly; splits joined from A2.",
        "",
        "**sections_json finding:** values are section TEXT, not offsets — but the dataset",
        "parser takes the FIRST label match, which in filings with a table of contents is",
        "the TOC line (median 24–96 chars; only ~10–17% substantive). Sections were",
        "therefore re-extracted from the cached full text with a TOC-skip rule (first",
        f"label-match span ≥ {MIN_SECTION_CHARS} chars, else last match). Where sections_json DID",
        "capture substantive text (>2000 chars), the re-extraction recovers it:",
        "",
    ]
    for name, a in diag["agreement_vs_sections_json"].items():
        if a["n_substantive_ref"]:
            md.append(f"- {name}: {a['n_head_recovered']}/{a['n_substantive_ref']} "
                      f"({a['pct']:.1f}%) of substantive sections_json heads recovered")
    md += [
        "",
        "## Sanity: full-text B2 recipe re-run vs archived B2 (test QLIKE, variance-unit)",
        "",
        "| horizon | archived B2 | re-run | diff |",
        "|---|---|---|---|",
        *sanity_lines,
        "",
        "## Results",
        "",
        "Standalone = test QLIKE (variance-unit, as metrics.json). M1 = leakage-free",
        "log-space combination vs recalibrated HAR (fit on val, frozen on test);",
        "rel improve % = QLIKE(f_R) → QLIKE(f_U) reduction (vol-unit QLIKE, fc.qlike);",
        "DM on fc.qlike losses, h=horizon, NEGATIVE stat = text adds (same convention",
        "as scripts/analysis/forecast_combination.py).",
        "",
        res.to_markdown(index=False),
        "",
    ]
    (TABLES / "section_ablation.md").write_text("\n".join(md))
    log(f"evaluate: wrote {TABLES/'section_ablation.csv'} and .md")
    print(res.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["extract", "fit", "evaluate", "all"], default="all")
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N filings")
    ap.add_argument("--smoke", action="store_true", help="write run dirs under workdir, not results/runs")
    args = ap.parse_args()
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    publish_root = (workdir / "runs") if args.smoke else RUNS
    if args.limit and not args.smoke:
        ap.error("--limit requires --smoke (limited store must not publish to results/runs)")
    if args.stage in ("extract", "all"):
        stage_extract(workdir, args.limit)
    if args.stage in ("fit", "all"):
        stage_fit(workdir, publish_root, smoke=args.smoke)
    if args.stage in ("evaluate", "all"):
        stage_evaluate(workdir, publish_root)


if __name__ == "__main__":
    main()
