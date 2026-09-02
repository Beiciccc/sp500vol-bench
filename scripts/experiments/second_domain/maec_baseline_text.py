#!/usr/bin/env python
"""MAEC audit — CPU arms on the canonical MAEC panel (prereg
configs/prereg_maec_audit.md §5, tag prereg-maec-v1.0 + v1.1/v1.2 revisions).
Yelp yelp_baseline_text.py port. FIT STAGE ONLY.

FIT-STAGE DISCIPLINE (audit hard guard, binding on every executed path of
this script): NOTHING is computed on split=='test' beyond writing frozen
predictions — no test-split MSE/QLIKE/DM/placebo/rel% anywhere in code or
output here. The val-selection MSE for the ridge alpha is the pre-registered
tuning rule (§5-1) and is the only validation statistic recorded. All audited
readouts belong to maec_protocol.py (scoring stage, single-shot §6.5).

Arms (one predictions parquet per arm; schema = what the maec_protocol.py
loader consumes, the consumer of record:
    [permno, call_date, horizon, split, label, prediction, arm]
call_id / alignment ride along as provenance — the loader projects
KEY + [label, f_text] and ignores extras):

  r_ar / r_har     REFERENCE cross-check halves (§5): R-AR = OLS[1, V_past^(n)],
                   R-HAR = OLS[1, V_past^(5), V_past^(22), V_past^(66)], both
                   VAL-fit test-frozen (§5 combiner discipline; val rows are
                   in-sample for the fit and flagged as such). maec_protocol.py
                   rebuilds the same references from the panel at scoring time;
                   these parquets exist for a PREDICTION-LEVEL two-half
                   cross-check (the former fit-stage test-MSE cross-check was
                   removed under the fit-stage discipline above — the stored
                   val-fit betas serve the same purpose). Written per alignment.
  tfidf            TF-IDF ridge (§5-1): word 1-2 gram, min_df=5, max_features
                   50,000, sublinear-tf, vocabulary+idf from TRAIN text only;
                   ridge alpha in {1e-2..1e3} (log grid) fit on train, chosen
                   on val; target = v (the §2.1 label). Fit ONCE under the
                   PRIMARY alignment (SS2.3: "zero GPU: all text-arm predictions unchanged" -- text-arm
                   predictions are frozen across alignments; the shifted branch
                   only re-derives labels and combiner fits, handled by the
                   protocol's intersection merge). TRAIN rows are predicted too
                   (in-sample, never scored): the protocol's primary-alignment
                   merge asserts FULL panel coverage (n_drop == 0).
  tfidf_published  published-convention TF-IDF (§4 v1.1): MAEC (CIKM 2020)
                   Table 5 year panels (2015 / 2016 / 2017-18), chronological
                   7:1:2 inside each panel via the pinned boundary dates, one
                   INDEPENDENT fit per (year panel, horizon) ("different models
                   for different years"); alignment = day-1-start unadjusted
                   (= the primary labels, v1.1). v_past_match rides along so
                   the published raw-V_past^(n) reading has its input pinned to
                   the same year-panel row assignment (no scoring here). Rows
                   dated after a panel's Table-5 test end fall outside the
                   published convention -> dropped + counted (disclosed).
  qwen_emb         frozen-embedding ridge (§5-4, OPEN-5): Qwen3-Emb-8B
                   mean-pooled embeddings + ridge (same grid, val-selected).
                   The EMBEDDING COMPUTATION IS A GPU STEP — see fit_qwen_emb();
                   this script consumes a precomputed embeddings parquet
                   (--embeddings, one row per call_id).

Usage (from repo root; threads capped before numpy import):
    .venv/bin/python scripts/experiments/second_domain/maec_baseline_text.py \
        --panel /path/to/data-root/second-domain/earnings_calls/maec_panel.parquet \
        --out-dir results/second_domain/maec/preds \
        [--alignment primary|shifted|both] \
        [--arms r_ar,r_har,tfidf,tfidf_published,qwen_emb] [--embeddings ...]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
CLIP_LO, CLIP_HI = float(np.log(1e-4)), 0.0     # v-units (§5)
KEY = ["permno", "call_date", "horizon"]
HORIZONS = (3, 7, 15, 30)
ALPHAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)  # §5-1 log grid
AR_COLS = {"r_ar": ["v_past_match"],
           "r_har": ["v_past_5", "v_past_22", "v_past_66"]}

# §4 v1.1 published-convention: MAEC (CIKM 2020) Table 5 pinned boundaries,
# chronological 7:1:2 inside each year panel (train_end, val_end, test_end).
PUB_PANELS = {
    "2015": ("2015-10-22", "2015-10-28", "2015-12-17"),
    "2016": ("2016-08-03", "2016-08-12", "2016-11-15"),
    "2017-18": ("2017-11-07", "2018-02-15", "2018-06-21"),
}
PUB_YEAR_OF = {2015: "2015", 2016: "2016", 2017: "2017-18", 2018: "2017-18"}


def mse(y, f) -> float:
    return float(np.mean((np.asarray(y, float) - np.asarray(f, float)) ** 2))


def clip_v(f):
    return np.clip(np.asarray(f, float), CLIP_LO, CLIP_HI)


def ols_val_frozen(yv, Xv, Xt):
    """§5 reference: OLS with intercept fit on VAL, frozen on test."""
    A = np.column_stack([np.ones(len(yv)), Xv])
    beta, *_ = np.linalg.lstsq(A, np.asarray(yv, float), rcond=None)
    fv = clip_v(A @ beta)
    ft = clip_v(np.column_stack([np.ones(len(Xt)), Xt]) @ beta)
    return fv, ft, beta


def fit_ridge_select(Xtr, ytr, Xv, yv, alphas=ALPHAS):
    """Ridge per alpha on TRAIN, alpha minimising val MSE (Yelp port; the
    pre-registered §5-1 tuning rule — the only val statistic this stage
    records). The winner stays train-fit — val predictions are honestly out
    of sample."""
    from sklearn.linear_model import Ridge
    best = None
    for a in alphas:
        model = Ridge(alpha=a)
        model.fit(Xtr, ytr)
        m = mse(yv, clip_v(model.predict(Xv)))
        if best is None or m < best[0]:
            best = (m, a, model)
    _, alpha, model = best
    return model, alpha


def arm_frame(sub: pd.DataFrame, pred, arm: str) -> pd.DataFrame:
    """Protocol loader schema + provenance extras (ignored by the loader)."""
    out = sub[KEY + ["call_id", "alignment", "split", "label"]].copy()
    out["prediction"] = clip_v(pred)
    out["arm"] = arm
    assert np.isfinite(out["prediction"]).all(), f"non-finite predictions in {arm}"
    return out[KEY + ["call_id", "alignment", "split", "label", "prediction", "arm"]]


def load_texts(d: pd.DataFrame) -> dict:
    """Read transcript text per unique call (sentence-per-line -> spaces)."""
    paths = d.drop_duplicates("call_id").set_index("call_id")["text_path"]
    texts = {}
    for cid, p in paths.items():
        texts[cid] = " ".join(Path(p).read_text(
            encoding="utf-8", errors="replace").split())
    assert all(len(t) > 0 for t in texts.values()), \
        "empty transcript text on a kept call"
    return texts


# ------------------------------------------------------------- reference halves
def run_references(panel: pd.DataFrame, alignment: str, out: Path, metrics: dict):
    """§5 reference cross-check halves, VAL-fit test-frozen. Fit-stage
    discipline: no metric is computed on split=='test' — only the frozen
    predictions are written; the val-fit betas are stored for the scoring-time
    prediction-level cross-check against the protocol's own rebuild."""
    frames = []
    metrics[alignment] = {}
    for h in HORIZONS:
        d = panel[panel["horizon"] == h]
        tr, va, te = (d[d["split"] == s] for s in ("train", "val", "test"))
        yv = va["label"].to_numpy(float)
        met = {"n_train": len(tr), "n_val": len(va), "n_test": len(te)}
        for arm, cols in AR_COLS.items():
            Xv = va[cols].to_numpy(float)
            Xt = te[cols].to_numpy(float)
            fv, ft, beta = ols_val_frozen(yv, Xv, Xt)
            met[f"beta_{arm}"] = [float(b) for b in beta]
            # val rows are IN-SAMPLE for the val-fit reference (flagged; the
            # protocol rebuilds references itself — these files are cross-checks)
            frames += [arm_frame(va, fv, f"{arm}_{alignment}"),
                       arm_frame(te, ft, f"{arm}_{alignment}")]
        metrics[alignment][str(h)] = met
        b_ar = ", ".join(f"{b:+.4f}" for b in met["beta_r_ar"])
        print(f"[{alignment}] h={h:>2}: refs val-fit test-frozen  "
              f"beta_r_ar=[{b_ar}]  (n {len(tr)}/{len(va)}/{len(te)})")
    stacked = pd.concat(frames, ignore_index=True)
    for arm, g in stacked.groupby("arm"):
        g.reset_index(drop=True).to_parquet(out / f"preds_{arm}.parquet", index=False)


# ------------------------------------------------------------------ tfidf arm
def run_tfidf(panel: pd.DataFrame, texts: dict, out: Path, metrics: dict):
    """§5-1 TF-IDF ridge, PRIMARY-alignment fit only (§2.3). Emits TRAIN rows
    too (in-sample, never scored) because the protocol's primary merge asserts
    full panel coverage. Fit-stage discipline: no test-split metric."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    tx_all = panel["call_id"].map(texts)
    frames, metrics["tfidf"] = [], {}
    for h in HORIZONS:
        m = panel["horizon"] == h
        d, tx = panel[m], tx_all[m]
        tr, va, te = (d["split"] == s for s in ("train", "val", "test"))
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50_000,
                              sublinear_tf=True)
        Ttr = vec.fit_transform(tx[tr])         # vocabulary + idf from TRAIN only
        Tv, Tt = vec.transform(tx[va]), vec.transform(tx[te])
        ytr, yv = (d.loc[s, "label"].to_numpy(float) for s in (tr, va))
        model, alpha = fit_ridge_select(Ttr, ytr, Tv, yv)
        ftr = clip_v(model.predict(Ttr))        # in-sample, coverage only
        fv, ft = clip_v(model.predict(Tv)), clip_v(model.predict(Tt))
        metrics["tfidf"][str(h)] = {
            "alpha": alpha, "n_features": len(vec.vocabulary_),
            "mse_val": mse(yv, fv),
            "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum())}
        frames += [arm_frame(d[tr], ftr, "tfidf"), arm_frame(d[va], fv, "tfidf"),
                   arm_frame(d[te], ft, "tfidf")]
        print(f"[tfidf] h={h:>2}: alpha={alpha}  val MSE={mse(yv, fv):.4f}  "
              f"n_features={len(vec.vocabulary_)}  "
              f"(n {int(tr.sum())}/{int(va.sum())}/{int(te.sum())})")
    stacked = pd.concat(frames, ignore_index=True)
    assert len(stacked) == len(panel), \
        "tfidf predictions do not cover the full primary panel"
    stacked.to_parquet(out / "preds_tfidf.parquet", index=False)


# ------------------------------------------- published-convention (§4 v1.1) arm
def assign_published(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Table-5 year-panel row assignment on the primary-alignment panel:
    year panel by calendar year of call_date; split by the pinned chronological
    7:1:2 boundaries; rows after a panel's test end are outside the published
    convention (dropped + counted). The §4 primary split is kept as
    split_primary for provenance."""
    d = panel.copy()
    d["year_panel"] = d["call_date"].dt.year.map(PUB_YEAR_OF)
    assert d["year_panel"].notna().all(), "call year outside Table-5 panels"
    parts, dropped = [], {}
    for yp, (b_tr, b_va, b_te) in PUB_PANELS.items():
        b_tr, b_va, b_te = (pd.Timestamp(b) for b in (b_tr, b_va, b_te))
        g = d[d["year_panel"] == yp].copy()
        out_of = g["call_date"] > b_te
        dropped[yp] = {"rows": int(out_of.sum()),
                       "calls": int(g.loc[out_of, "call_id"].nunique())}
        g = g[~out_of].copy()
        g = g.rename(columns={"split": "split_primary"})
        g["split"] = np.where(g["call_date"] <= b_tr, "train",
                              np.where(g["call_date"] <= b_va, "val", "test"))
        parts.append(g)
    return pd.concat(parts, ignore_index=True), dropped


def run_tfidf_published(panel: pd.DataFrame, texts: dict, out: Path, metrics: dict):
    """§4 v1.1 published-convention TF-IDF: one independent vectorizer+ridge
    fit per (year panel, horizon); same §5-1 grid and val-selection rule.
    NO published-convention readout is computed here (that is scoring-stage G1/G2
    territory); v_past_match rides along as the raw-V_past^(n) input."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    pub, dropped = assign_published(panel)
    tx_all = pub["call_id"].map(texts)
    frames, metrics["tfidf_published"] = [], {"_dropped_after_test_end": dropped}
    for yp in PUB_PANELS:
        metrics["tfidf_published"][yp] = {}
        for h in HORIZONS:
            m = (pub["year_panel"] == yp) & (pub["horizon"] == h)
            d, tx = pub[m], tx_all[m]
            tr, va, te = (d["split"] == s for s in ("train", "val", "test"))
            assert tr.sum() > 0 and va.sum() > 0 and te.sum() > 0, \
                f"empty published split in panel {yp} h={h}"
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5,
                                  max_features=50_000, sublinear_tf=True)
            Ttr = vec.fit_transform(tx[tr])     # per-panel TRAIN-only vocabulary
            Tv, Tt = vec.transform(tx[va]), vec.transform(tx[te])
            ytr, yv = (d.loc[s, "label"].to_numpy(float) for s in (tr, va))
            model, alpha = fit_ridge_select(Ttr, ytr, Tv, yv)
            preds = {"train": clip_v(model.predict(Ttr)),
                     "val": clip_v(model.predict(Tv)),
                     "test": clip_v(model.predict(Tt))}
            metrics["tfidf_published"][yp][str(h)] = {
                "alpha": alpha, "n_features": len(vec.vocabulary_),
                "mse_val": mse(yv, preds["val"]),
                "n_train": int(tr.sum()), "n_val": int(va.sum()),
                "n_test": int(te.sum())}
            for s, msk in (("train", tr), ("val", va), ("test", te)):
                f = d[msk][KEY + ["call_id", "alignment", "year_panel", "split",
                                  "label", "v_past_match"]].copy()
                f["prediction"] = preds[s]
                f["arm"] = "tfidf_published"
                assert np.isfinite(f["prediction"]).all()
                frames.append(f)
            print(f"[tfidf_published] {yp} h={h:>2}: alpha={alpha}  "
                  f"val MSE={metrics['tfidf_published'][yp][str(h)]['mse_val']:.4f}  "
                  f"(n {int(tr.sum())}/{int(va.sum())}/{int(te.sum())})")
    stacked = pd.concat(frames, ignore_index=True)
    assert len(stacked) == len(pub), "published preds do not cover the assignment"
    stacked.to_parquet(out / "preds_tfidf_published.parquet", index=False)
    return pub


# ------------------------------------------------------------ qwen_emb (stub)
BOX_CMDS = """\
[qwen_emb] SKIPPED — no --embeddings parquet given (GPU box step pending).
This stub expects, in order:
  (box, GPU)  compute frozen-encoder embeddings — model Qwen/Qwen3-Embedding-8B,
              vLLM runner='pooling' convert='embed', mean-pooled over the
              head-truncated transcript (maec_prompt budget); ONE row per
              call_id; parquet columns [call_id, emb_0..emb_{d-1}] float32.
              Encoder script to be added mirroring yelp_arm_embed.py encode,
              e.g. per shard i of 4:
                .venv/bin/python scripts/experiments/second_domain/maec_arm_embed.py encode \\
                    --panel /path/to/data-root/second-domain/earnings_calls/maec_panel.parquet \\
                    --out-dir /path/to/data-root/second-domain/earnings_calls/qwen3emb \\
                    --model Qwen/Qwen3-Embedding-8B --shard i --num-shards 4
  (here, CPU) .venv/bin/python scripts/experiments/second_domain/maec_baseline_text.py \\
                  --arms qwen_emb \\
                  --embeddings /path/to/data-root/second-domain/earnings_calls/maec_qwen3emb.parquet"""


def fit_qwen_emb(panel: pd.DataFrame, out: Path, metrics: dict,
                 embeddings_path: str | None):
    """§5-4 frozen-embedding ridge (Qwen3-Emb-8B mean-pool + ridge). The
    embedding computation is a GPU step (see BOX_CMDS); downstream ridge
    (grid {1e-2..1e3} train-fit, val-selected, per horizon, primary alignment)
    mirrors the TF-IDF arm exactly, incl. TRAIN-row coverage and the
    fit-stage no-test-metric discipline."""
    if not embeddings_path:
        print(BOX_CMDS)
        return
    emb = pd.read_parquet(embeddings_path)
    ecols = [c for c in emb.columns if c.startswith("emb_")]
    assert ecols, "embeddings parquet needs emb_* columns"
    frames, metrics["qwen_emb"] = [], {}
    for h in HORIZONS:
        d = panel[panel["horizon"] == h].merge(emb, on="call_id", how="inner",
                                               validate="m:1")
        assert len(d) == int((panel["horizon"] == h).sum()), \
            "embeddings missing for some kept calls"
        tr, va, te = (d[d["split"] == s] for s in ("train", "val", "test"))
        model, alpha = fit_ridge_select(
            tr[ecols].to_numpy(np.float32), tr["label"].to_numpy(float),
            va[ecols].to_numpy(np.float32), va["label"].to_numpy(float))
        ftr = clip_v(model.predict(tr[ecols].to_numpy(np.float32)))
        fv = clip_v(model.predict(va[ecols].to_numpy(np.float32)))
        ft = clip_v(model.predict(te[ecols].to_numpy(np.float32)))
        metrics["qwen_emb"][str(h)] = {
            "alpha": alpha, "dim": len(ecols),
            "mse_val": mse(va["label"], fv),
            "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
        frames += [arm_frame(tr, ftr, "qwen_emb"), arm_frame(va, fv, "qwen_emb"),
                   arm_frame(te, ft, "qwen_emb")]
        print(f"[qwen_emb] h={h:>2}: alpha={alpha}  "
              f"val MSE={metrics['qwen_emb'][str(h)]['mse_val']:.4f}")
    stacked = pd.concat(frames, ignore_index=True)
    assert len(stacked) == len(panel), \
        "qwen_emb predictions do not cover the full primary panel"
    stacked.to_parquet(out / "preds_qwen_emb.parquet", index=False)


# ----------------------------------------------------------------- integrity
def verify_outputs(out: Path, panel: pd.DataFrame, aligns: list,
                   arms: set, pub: pd.DataFrame | None):
    """Row-set integrity: every preds parquet covers EXACTLY the panel row-set
    of its split domain (KEY match both ways), unique keys, finite predictions.
    Prints the per-split x horizon count tables."""
    def check(fname: str, expect: pd.DataFrame, extra_group=None):
        f = out / fname
        g = pd.read_parquet(f)
        assert g["prediction"].notna().all() and \
            np.isfinite(g["prediction"]).all(), f"{fname}: bad predictions"
        assert not g.duplicated(KEY).any(), f"{fname}: duplicate keys"
        m = expect[KEY].merge(g[KEY], on=KEY, how="outer", indicator=True)
        n_miss = int((m["_merge"] == "left_only").sum())
        n_extra = int((m["_merge"] == "right_only").sum())
        assert n_miss == 0 and n_extra == 0, \
            f"{fname}: row-set mismatch (missing={n_miss}, extra={n_extra})"
        grp = ([extra_group] if extra_group else []) + ["split", "horizon"]
        tab = g.groupby(grp).size().unstack("horizon")
        print(f"\n{fname}: {len(g):,} rows — exact row-set match")
        print(tab.to_string())

    print("\n=== integrity: row-set coverage per preds parquet ===")
    prim = panel[panel["alignment"] == "primary"]
    if "tfidf" in arms:
        check("preds_tfidf.parquet", prim)
    if "tfidf_published" in arms and pub is not None:
        check("preds_tfidf_published.parquet", pub, extra_group="year_panel")
    if arms & {"r_ar", "r_har"}:
        for al in aligns:
            dom = panel[(panel["alignment"] == al)
                        & panel["split"].isin(("val", "test"))]
            for arm in ("r_ar", "r_har"):
                if arm in arms:
                    check(f"preds_{arm}_{al}.parquet", dom)
    if "qwen_emb" in arms and (out / "preds_qwen_emb.parquet").exists():
        check("preds_qwen_emb.parquet", prim)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel",
                    default="/path/to/data-root/second-domain/earnings_calls/maec_panel.parquet")
    ap.add_argument("--out-dir", default=str(REPO / "results/second_domain/maec/preds"))
    ap.add_argument("--alignment", choices=["primary", "shifted", "both"],
                    default="both",
                    help="alignment(s) for the REFERENCE cross-check arms; the "
                         "text arms are always fit under primary (§2.3)")
    ap.add_argument("--arms", default="r_ar,r_har,tfidf,tfidf_published,qwen_emb",
                    help="comma list from {r_ar,r_har,tfidf,tfidf_published,qwen_emb}")
    ap.add_argument("--embeddings", default=None,
                    help="precomputed Qwen3-Emb-8B parquet (box GPU step)")
    args = ap.parse_args()
    arms = set(args.arms.split(","))
    aligns = ["primary", "shifted"] if args.alignment == "both" else [args.alignment]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(args.panel)
    need = {"alignment", "call_id", "split", "label", "text_path",
            "v_past_match", "v_past_5", "v_past_22", "v_past_66", *KEY}
    assert need.issubset(panel.columns), f"panel missing: {need - set(panel.columns)}"

    met_path = out / "maec_baseline_metrics.json"
    metrics = json.loads(met_path.read_text()) if met_path.exists() else {}

    if arms & {"r_ar", "r_har"}:
        for al in aligns:
            run_references(panel[panel["alignment"] == al].copy(), al, out, metrics)
    prim = panel[panel["alignment"] == "primary"].reset_index(drop=True)
    texts = (load_texts(prim) if arms & {"tfidf", "tfidf_published"} else None)
    if "tfidf" in arms:
        run_tfidf(prim, texts, out, metrics)
    pub = None
    if "tfidf_published" in arms:
        pub = run_tfidf_published(prim, texts, out, metrics)
    if "qwen_emb" in arms:
        fit_qwen_emb(prim, out, metrics, args.embeddings)

    met_path.write_text(json.dumps(metrics, indent=2))
    verify_outputs(out, panel, aligns, arms, pub)
    print(f"\nwrote {out}/preds_<arm>.parquet + {met_path.name}")


if __name__ == "__main__":
    main()
