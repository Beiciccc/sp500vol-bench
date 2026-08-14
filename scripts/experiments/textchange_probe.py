"""E3 — TEXTUAL-CHANGE PROBE (Lazy Prices, Cohen-Malloy-Nguyen 2020): does the
long-form disclosure signal come from CHANGES in the document rather than levels?

For each long_form filing (10-K/10-Q) find the firm's PREVIOUS filing of the SAME
form (cik + form, ordered by filing_time_utc, gap < 550 days) and compute:
  - TF-IDF cosine similarity (sublinear tf, smooth idf; IDF fitted on TRAIN-split
    docs ONLY, applied to all)              -> change_score = 1 - cosine
  - Jaccard similarity on word 5-shingles (bottom-k minhash sketch, k=256)
                                            -> jaccard_change = 1 - jaccard
  - log doc length ratio  log(n_tokens_cur / n_tokens_prev)

Model B6_textchange: Ridge on [change_score, jaccard_change, log_len_ratio,
form dummy], one ridge per horizon, log-vol target with the SAME retransform
convention as B2 (exp + clip to [0.02, 5.0]; alpha by the shared 5-fold CV).
Filings with no previous filing get the train-split mean features.

Evaluation (test split, weights frozen on val, per M1 conventions):
  (1) standalone test QLIKE (vol-unit) vs B2 full-text + DM
  (2) M1 incremental value vs recalibrated A2 (fc.log_combo + DM/qlike)
  (3) KEY: joint log combiner [1, log fHAR, log fB2, log fB6] on val -> test,
      DM vs the level-only f_U (=[1, log fHAR, log fB2])
  (4) raw corr(change_score, future realised vol) — Lazy Prices sign check.

Outputs:
  results/tables/_textchange_features.parquet   (per-doc change features, cached)
  results/runs/B6_textchange_full_long_form_seed2026/{predictions.parquet,metrics.json,config.json}
  results/tables/textchange_probe.{csv,md}

Run from repo root:  .venv/bin/python scripts/experiments/textchange_probe.py
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/analysis")
import forecast_combination as fc

from sp500vol.evaluation.dm_test import dm_test
from sp500vol.models.classical_text._fit_utils import (
    fit_ridge_cv,
    maybe_exp,
    maybe_log,
)

ALIGNED = "/Volumes/Z/sp500vol-data/processed/full/aligned_filings.parquet"
TEXT_CACHE = "/Volumes/Z/sp500vol-data/processed/_text_cache/filing_texts.parquet"
A2_RUN = "results/runs/A2_har_rv_full_long_form_seed2026/predictions.parquet"
B2_RUN = "results/runs/B2_tfidf_ridge_full_long_form_seed2026/predictions.parquet"
FEATURE_CACHE = Path("results/tables/_textchange_features.parquet")
RUN_DIR = Path("results/runs/B6_textchange_full_long_form_seed2026")
OUT_CSV = Path("results/tables/textchange_probe.csv")
OUT_MD = Path("results/tables/textchange_probe.md")

TOKEN_RE = re.compile(r"\b[a-z]{2,}\b")  # same token pattern as B1/B2
MAX_GAP_DAYS = 550.0
SKETCH_K = 256  # bottom-k minhash sketch size for the 5-shingle Jaccard
SHINGLE = 5
MIN_PRED, MAX_PRED = 0.02, 5.0  # B2 retransform clip
KEY = ["ticker", "accession", "horizon_days"]
SORT = ["filing_time_utc", "ticker", "accession"]
HORIZONS = (5, 10, 20)
EPS = 1e-8
FEATS = ["change_score", "jaccard_change", "log_len_ratio", "is_10k"]

# odd 64-bit mixing constants for the rolling 5-shingle hash
_C = np.array(
    [0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9,
     0x27D4EB2F165667C5, 0x85EBCA77C2B2AE63], dtype=np.uint64)


def _mix64(h: np.ndarray) -> np.ndarray:
    h = h.copy()
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xFF51AFD7ED558CCD)
    h ^= h >> np.uint64(29)
    return h


def shingle_sketch(ids: np.ndarray) -> np.ndarray:
    """Bottom-k sketch (k smallest distinct 64-bit hashes) of word 5-shingles."""
    n = len(ids)
    if n < SHINGLE:
        return np.empty(0, dtype=np.uint64)
    u = ids.astype(np.uint64)
    h = np.zeros(n - SHINGLE + 1, dtype=np.uint64)
    for j in range(SHINGLE):
        h += _C[j] * (u[j:n - SHINGLE + 1 + j] + np.uint64(1))
    h = np.unique(_mix64(h))
    if len(h) > SKETCH_K:
        h = np.sort(np.partition(h, SKETCH_K)[:SKETCH_K])
    return h


def bottomk_jaccard(sa: np.ndarray, sb: np.ndarray) -> float:
    """Jaccard estimate from two bottom-k sketches (exact if either doc small)."""
    if len(sa) == 0 or len(sb) == 0:
        return float("nan")
    union = np.union1d(sa, sb)
    k = min(SKETCH_K, len(union))
    low = union[:k]  # union1d returns sorted -> k smallest of the union
    inter = np.intersect1d(np.intersect1d(low, sa, assume_unique=True), sb,
                           assume_unique=True)
    return float(len(inter) / k)


# --------------------------------------------------------------------------
# STAGE 1 — stream texts once, build per-doc reps, compute pair similarities
# --------------------------------------------------------------------------

def build_features() -> pd.DataFrame:
    aligned = pd.read_parquet(
        ALIGNED, columns=["cik", "form", "filing_time_utc", "text_path",
                          "accession", "token_count"])
    docs = (aligned[aligned.form.isin(["10-K", "10-Q"])]
            .drop_duplicates("accession").reset_index(drop=True))
    print(f"[stage1] {len(docs)} unique long-form docs", flush=True)

    # previous filing of the SAME form for the same cik
    docs = docs.sort_values(["cik", "form", "filing_time_utc", "accession"],
                            kind="mergesort").reset_index(drop=True)
    grp = docs.groupby(["cik", "form"], sort=False)
    docs["prev_accession"] = grp["accession"].shift(1)
    docs["prev_time"] = grp["filing_time_utc"].shift(1)
    gap = (docs.filing_time_utc - docs.prev_time).dt.total_seconds() / 86400.0
    docs["gap_days"] = gap
    ok = docs.prev_accession.notna() & (gap > 0) & (gap < MAX_GAP_DAYS)
    docs.loc[~ok, ["prev_accession", "gap_days"]] = [None, np.nan]
    print(f"[stage1] prev matched: {ok.sum()}/{len(docs)} "
          f"({100*ok.mean():.1f}%)", flush=True)

    # split per accession from the canonical A2 run (constant per accession)
    a2 = pd.read_parquet(A2_RUN, columns=["accession", "split"])
    split_map = a2.drop_duplicates("accession").set_index("accession")["split"]
    docs["split"] = docs.accession.map(split_map)  # NaN = not in run (prev-only)

    # ---- streaming pass over the 3.3GB text cache -------------------------
    path2acc = dict(zip(docs.text_path.astype(str), docs.accession.astype(str), strict=False))
    needed = pa.array(list(path2acc.keys()), type=pa.string())
    vocab: dict[str, int] = {}
    reps: dict[str, tuple] = {}  # acc -> (uniq_ids u32, counts u32, sketch u64, n_tok)
    t0, seen_rows, done = time.time(), 0, 0
    pf = pq.ParquetFile(TEXT_CACHE)
    for batch in pf.iter_batches(batch_size=64, columns=["text_path", "text"]):
        seen_rows += batch.num_rows
        mask = pc.is_in(batch.column("text_path"), value_set=needed)
        if pc.sum(mask).as_py() in (None, 0):
            continue
        sub = batch.filter(mask)
        paths = sub.column("text_path").to_pylist()
        texts = sub.column("text").to_pylist()
        for pth, text in zip(paths, texts, strict=False):
            acc = path2acc[pth]
            toks = TOKEN_RE.findall(text.lower())
            n_tok = len(toks)
            if n_tok == 0:
                reps[acc] = (np.empty(0, np.uint32), np.empty(0, np.uint32),
                             np.empty(0, np.uint64), 0)
                continue
            sd = vocab.setdefault
            ids = np.fromiter((sd(t, len(vocab)) for t in toks),
                              dtype=np.int64, count=n_tok)
            uniq, cnts = np.unique(ids, return_counts=True)
            reps[acc] = (uniq.astype(np.uint32), cnts.astype(np.uint32),
                         shingle_sketch(ids), n_tok)
            done += 1
        if done and done % 2000 < len(paths):
            el = time.time() - t0
            print(f"[stage1] docs={done}/{len(docs)} cache_rows={seen_rows} "
                  f"vocab={len(vocab)} elapsed={el:.0f}s", flush=True)
    print(f"[stage1] text pass done: {done} docs, vocab={len(vocab)}, "
          f"{time.time()-t0:.0f}s", flush=True)

    # ---- IDF from TRAIN-split docs only ------------------------------------
    train_accs = [a for a in docs.loc[docs.split == "train", "accession"].astype(str)
                  if a in reps]
    df_counts = np.zeros(len(vocab), dtype=np.int64)
    for a in train_accs:
        df_counts[reps[a][0]] += 1  # ids unique within doc -> fancy += is exact
    n_train = len(train_accs)
    idf = np.log((1.0 + n_train) / (1.0 + df_counts)) + 1.0  # sklearn smooth_idf
    print(f"[stage1] idf fitted on {n_train} train docs", flush=True)

    def tfidf_vec(acc):
        uniq, cnts, _, _ = reps[acc]
        w = (1.0 + np.log(cnts.astype(float))) * idf[uniq]  # sublinear tf
        nrm = np.linalg.norm(w)
        return uniq, (w / nrm if nrm > 0 else w)

    rows = []
    for r in docs.itertuples(index=False):
        acc = str(r.accession)
        cur = reps.get(acc)
        prev_acc = None if pd.isna(r.prev_accession) else str(r.prev_accession)
        prev = reps.get(prev_acc) if prev_acc else None
        rec = {"accession": acc, "form": r.form, "split": r.split,
               "filing_time_utc": r.filing_time_utc,
               "prev_accession": prev_acc, "gap_days": r.gap_days,
               "cos_sim": np.nan, "jaccard": np.nan, "log_len_ratio": np.nan,
               "n_tok": cur[3] if cur else np.nan,
               "prev_n_tok": prev[3] if prev else np.nan}
        if cur and prev and cur[3] > 0 and prev[3] > 0:
            ia, wa = tfidf_vec(acc)
            ib, wb = tfidf_vec(prev_acc)
            common, ca, cb = np.intersect1d(ia, ib, assume_unique=True,
                                            return_indices=True)
            rec["cos_sim"] = float(np.dot(wa[ca], wb[cb]))
            rec["jaccard"] = bottomk_jaccard(cur[2], prev[2])
            rec["log_len_ratio"] = float(np.log(cur[3] / prev[3]))
        rows.append(rec)
    feat = pd.DataFrame(rows)
    feat["change_score"] = 1.0 - feat.cos_sim
    feat["jaccard_change"] = 1.0 - feat.jaccard
    feat["is_10k"] = (feat.form == "10-K").astype(float)
    feat["has_prev"] = feat.cos_sim.notna()
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(FEATURE_CACHE, index=False)
    print(f"[stage1] wrote {FEATURE_CACHE} ({len(feat)} docs, "
          f"{feat.has_prev.mean()*100:.1f}% with prev similarity)", flush=True)
    return feat


# --------------------------------------------------------------------------
# STAGE 2 — B6_textchange ridge run (B2 conventions), write run dir
# --------------------------------------------------------------------------

def fit_b6(feat: pd.DataFrame) -> pd.DataFrame:
    b2 = pd.read_parquet(B2_RUN)
    p = b2.copy()
    p["run_id"] = "B6_textchange_full_long_form_seed2026"
    p["model_id"] = "B6_textchange"
    f = feat.set_index("accession")
    for c in FEATS + ["has_prev"]:
        p[c] = p.accession.astype(str).map(f[c])
    p["has_prev"] = p["has_prev"].fillna(False).astype(bool)

    # train-split mean imputation for docs with no previous filing
    tr_doc = feat[(feat.split == "train") & feat.has_prev]
    imput = {c: float(tr_doc[c].mean()) for c in
             ["change_score", "jaccard_change", "log_len_ratio"]}
    frac_imputed = 1.0 - p.has_prev.mean()
    for c, v in imput.items():
        p[c] = p[c].fillna(v)

    # standardise on train, per-horizon ridge on log-vol (B2 conventions)
    tr = p.split == "train"
    mu = p.loc[tr, FEATS].mean()
    sd = p.loc[tr, FEATS].std().replace(0.0, 1.0)
    X = ((p[FEATS] - mu) / sd).to_numpy(float)
    y = p.label_realised_vol.to_numpy(float)
    pred = np.empty(len(p))
    alphas = {}
    for h in HORIZONS:
        mh = (p.horizon_days == h).to_numpy()
        fit_mask = mh & tr.to_numpy()
        ridge = fit_ridge_cv(X[fit_mask], maybe_log(y[fit_mask], log_target=True),
                             (0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0))
        pred[mh] = maybe_exp(ridge.predict(X[mh]), log_target=True,
                             min_pred=MIN_PRED, max_pred=MAX_PRED)
        alphas[h] = float(ridge.alpha_)
    p["prediction_realised_vol"] = pred

    # ---- write run dir ------------------------------------------------------
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out = p[b2.columns]  # exactly the B2 schema
    out.to_parquet(RUN_DIR / "predictions.parquet", index=False)

    metrics = []
    for split in ("test", "train", "val"):
        for h in HORIZONS:
            s = p[(p.split == split) & (p.horizon_days == h)]
            yy = s.label_realised_vol.to_numpy(float)
            ff = s.prediction_realised_vol.to_numpy(float)
            a = np.clip(yy**2, EPS, None); b = np.clip(ff**2, EPS, None)
            metrics.append({
                "split": split, "disclosure_subset": "long_form",
                "horizon_days": int(h), "n": len(s),
                "mae": float(np.abs(ff - yy).mean()),
                "rmse": float(np.sqrt(((ff - yy) ** 2).mean())),
                "r2": float(1 - ((ff - yy) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()),
                "qlike": float((a / b - np.log(a / b) - 1).mean()),
            })
    (RUN_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (RUN_DIR / "config.json").write_text(json.dumps({
        "model_id": "B6_textchange",
        "note": ("E3 Lazy-Prices textual-change probe: Ridge on [change_score(=1-TFIDF "
                 "cosine vs previous same-form filing, IDF fit on train docs only), "
                 "jaccard_change(=1-bottom-k minhash Jaccard on word 5-shingles, k=256), "
                 "log_len_ratio, 10-K dummy]; gap<550d; log-vol target, per-horizon "
                 "ridge CV + exp retransform clipped to [0.02,5.0] (B2 conventions); "
                 f"no-prev docs imputed with train means (fraction={frac_imputed:.4f}); "
                 f"ridge alphas={alphas}; splits inherited from A2_har_rv long_form."),
    }, indent=2))
    print(f"[stage2] wrote {RUN_DIR} (alphas={alphas}, "
          f"imputed fraction={frac_imputed:.3f})", flush=True)
    p.attrs["frac_imputed"] = frac_imputed
    p.attrs["imput"] = imput
    return p


# --------------------------------------------------------------------------
# STAGE 3 — evaluation
# --------------------------------------------------------------------------

def evaluate(p6: pd.DataFrame, feat: pd.DataFrame):
    har = pd.read_parquet(A2_RUN)[["split"] + KEY + [
        "prediction_realised_vol", "label_realised_vol", "filing_time_utc"]
        ].rename(columns={"prediction_realised_vol": "fhar"})
    b2 = pd.read_parquet(B2_RUN)[KEY + ["prediction_realised_vol"]].rename(
        columns={"prediction_realised_vol": "fb2"})
    b6 = p6[KEY + ["prediction_realised_vol", "change_score", "has_prev"]].rename(
        columns={"prediction_realised_vol": "fb6"})
    d = har.merge(b2, on=KEY).merge(b6, on=KEY)
    assert len(d) == len(har), "key merge lost rows"

    rows = []
    for h in HORIZONS:
        dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(SORT, kind="mergesort")
        dt = d[(d.horizon_days == h) & (d.split == "test")].sort_values(SORT, kind="mergesort")
        yv, yt = dv.label_realised_vol.to_numpy(), dt.label_realised_vol.to_numpy()
        fhv, fhr = dv.fhar.to_numpy(), dt.fhar.to_numpy()
        f2v, f2t = dv.fb2.to_numpy(), dt.fb2.to_numpy()
        f6v, f6t = dv.fb6.to_numpy(), dt.fb6.to_numpy()

        # (1) standalone: B6 vs B2 (vol-unit QLIKE, as in M1)
        q6, q2 = fc.qlike(yt, f6t), fc.qlike(yt, f2t)
        qhar = fc.qlike(yt, fhr)
        dm12, p12 = dm_test(q6, q2, h=h)  # + => B6 worse than B2

        # (2) M1 incremental value of B6 over recalibrated HAR
        fR, fU6, g6 = fc.log_combo(yv, fhv, f6v, fhr, f6t)
        lR, lU6 = fc.qlike(yt, fR), fc.qlike(yt, fU6)
        dm6, pdm6 = dm_test(lU6, lR, h=h)  # - => B6 adds over recalibrated HAR
        rel6 = 100.0 * (lR.mean() - lU6.mean()) / lR.mean()

        # level-only combiner (HAR + B2 level text)
        _, fU2, g2 = fc.log_combo(yv, fhv, f2v, fhr, f2t)
        lU2 = fc.qlike(yt, fU2)
        dm2, pdm2 = dm_test(lU2, lR, h=h)
        rel2 = 100.0 * (lR.mean() - lU2.mean()) / lR.mean()

        # (3) KEY — joint combiner: change BEYOND level
        cl = lambda a: np.log(np.clip(a, 1e-8, None))
        Xv = np.column_stack([np.ones(len(yv)), cl(fhv), cl(f2v), cl(f6v)])
        bJ = fc.ols(cl(yv), Xv)
        fJ = np.exp(bJ[0] + bJ[1] * cl(fhr) + bJ[2] * cl(f2t) + bJ[3] * cl(f6t))
        lJ = fc.qlike(yt, fJ)
        dmJ, pJ = dm_test(lJ, lU2, h=h)  # - => change adds BEYOND level
        relJ = 100.0 * (lU2.mean() - lJ.mean()) / lU2.mean()

        # (4) Lazy Prices sign check — corr(change_score, future realised vol),
        # real-prev filings only
        real = dt[dt.has_prev]
        pear = float(np.corrcoef(real.change_score, real.label_realised_vol)[0, 1])
        from scipy.stats import spearmanr
        spear, spear_p = spearmanr(real.change_score, real.label_realised_vol)
        alls = d[(d.horizon_days == h) & d.has_prev]
        pear_all = float(np.corrcoef(alls.change_score, alls.label_realised_vol)[0, 1])

        rows.append({
            "h": h, "n_test": len(dt),
            "qlike_har_raw": float(qhar.mean()),
            "qlike_b6": float(q6.mean()), "qlike_b2": float(q2.mean()),
            "dm_b6_vs_b2": float(dm12), "p_b6_vs_b2": float(p12),
            "qlike_R": float(lR.mean()),
            "qlike_U_b6": float(lU6.mean()), "g_b6": g6,
            "dm_b6_vs_R": float(dm6), "p_b6_vs_R": float(pdm6), "rel_b6_pct": rel6,
            "qlike_U_b2": float(lU2.mean()), "g_b2": g2,
            "dm_b2_vs_R": float(dm2), "p_b2_vs_R": float(pdm2), "rel_b2_pct": rel2,
            "qlike_joint": float(lJ.mean()),
            "g_level_joint": float(bJ[2]), "g_change_joint": float(bJ[3]),
            "dm_joint_vs_levelU": float(dmJ), "p_joint_vs_levelU": float(pJ),
            "rel_joint_pct": relJ,
            "corr_change_vol_test": pear, "spearman_change_vol_test": float(spear),
            "spearman_p": float(spear_p), "corr_change_vol_all": pear_all,
        })
    res = pd.DataFrame(rows)

    # sanity block
    matched = feat.has_prev
    yearly = (feat.assign(yr=feat.filing_time_utc.dt.year)
              .groupby("yr")["has_prev"].mean())
    frac_imputed = p6.attrs["frac_imputed"]
    imput = p6.attrs["imput"]
    fr = feat[feat.has_prev]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False)

    md = ["# E3 — Textual-change probe (Lazy Prices): change vs level in long-form disclosure\n",
          "B6_textchange = Ridge on [change_score(=1-TFIDF-cosine vs previous same-form filing), "
          "jaccard_change(1-Jaccard on word 5-shingles, bottom-k minhash k=256), log length ratio, "
          "10-K dummy]; IDF fit on train docs only; gap<550d; B2 log-target/retransform conventions; "
          "splits inherited from A2. DM sign: negative = first argument better.\n",
          f"**Prev-filing match:** {int(matched.sum())}/{len(feat)} unique long-form docs "
          f"({100*matched.mean():.1f}%) matched to a previous same-form filing "
          f"(row-level imputed fraction in the run: {100*frac_imputed:.1f}%; imputed with train means "
          f"{ {k: round(v,4) for k,v in imput.items()} }).",
          "Match rate by year: " + ", ".join(f"{y}:{v*100:.0f}%" for y, v in yearly.items()) + "\n",
          f"**Change-feature distribution (matched docs):** change_score mean={fr.change_score.mean():.3f} "
          f"sd={fr.change_score.std():.3f} p10={fr.change_score.quantile(.1):.3f} "
          f"p90={fr.change_score.quantile(.9):.3f}; jaccard_change mean={fr.jaccard_change.mean():.3f}; "
          f"median gap {fr.gap_days.median():.0f}d.\n",
          "## (1) Standalone test QLIKE (vol-unit) — B6 change vs B2 full-text level\n"
          "| h | n_test | QLIKE HAR raw | QLIKE B2 | QLIKE B6 | DM B6 vs B2 | p |\n|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        md.append(f"| {int(r.h)} | {int(r.n_test)} | {r.qlike_har_raw:.4f} | {r.qlike_b2:.4f} | "
                  f"{r.qlike_b6:.4f} | {r.dm_b6_vs_b2:+.2f} | {r.p_b6_vs_b2:.4f} |")
    md.append("\n## (2) M1 incremental value over recalibrated HAR (log combo, val-frozen)\n"
              "| h | QLIKE f_R | QLIKE f_U(B6) | g_B6 | DM | p | rel% | QLIKE f_U(B2) | g_B2 | DM(B2) | p(B2) | rel%(B2) |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in res.iterrows():
        md.append(f"| {int(r.h)} | {r.qlike_R:.4f} | {r.qlike_U_b6:.4f} | {r.g_b6:+.3f} | "
                  f"{r.dm_b6_vs_R:+.2f} | {r.p_b6_vs_R:.4f} | {r.rel_b6_pct:+.2f} | "
                  f"{r.qlike_U_b2:.4f} | {r.g_b2:+.3f} | {r.dm_b2_vs_R:+.2f} | {r.p_b2_vs_R:.4f} | "
                  f"{r.rel_b2_pct:+.2f} |")
    md.append("\n## (3) KEY — does CHANGE add increment BEYOND the full-text LEVEL?\n"
              "Joint val-frozen log combiner [1, log fHAR, log fB2, log fB6] vs level-only f_U(HAR+B2).\n"
              "| h | QLIKE level-only f_U | QLIKE joint | g_level | g_change | DM joint vs level | p | rel% |\n"
              "|---|---|---|---|---|---|---|---|")
    for _, r in res.iterrows():
        md.append(f"| {int(r.h)} | {r.qlike_U_b2:.4f} | {r.qlike_joint:.4f} | {r.g_level_joint:+.3f} | "
                  f"{r.g_change_joint:+.3f} | {r.dm_joint_vs_levelU:+.2f} | {r.p_joint_vs_levelU:.4f} | "
                  f"{r.rel_joint_pct:+.2f} |")
    md.append("\n## (4) Lazy Prices sign check — corr(change_score, future realised vol), matched docs\n"
              "| h | Pearson (test) | Spearman (test) | Spearman p | Pearson (all splits) |\n|---|---|---|---|---|")
    for _, r in res.iterrows():
        md.append(f"| {int(r.h)} | {r.corr_change_vol_test:+.4f} | {r.spearman_change_vol_test:+.4f} | "
                  f"{r.spearman_p:.2e} | {r.corr_change_vol_all:+.4f} |")
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"[stage3] wrote {OUT_CSV} and {OUT_MD}", flush=True)
    return res


def main():
    recompute = "--recompute" in sys.argv
    if FEATURE_CACHE.exists() and not recompute:
        feat = pd.read_parquet(FEATURE_CACHE)
        print(f"[stage1] loaded cached features {FEATURE_CACHE} ({len(feat)})", flush=True)
    else:
        feat = build_features()
    p6 = fit_b6(feat)
    res = evaluate(p6, feat)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
