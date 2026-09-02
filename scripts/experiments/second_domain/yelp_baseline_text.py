#!/usr/bin/env python
"""Second-domain baselines and text models on the canonical Yelp panel.

Arms (predictions parquet written per arm, per SECOND_DOMAIN_PLAN.md):
  (a) ar_ridge         AR-type ridge on [ar_last_mean, ar_roll3_mean, ar_roll12_mean,
                       ar_log_n_reviews]; fit on train, alpha chosen on val, raw
                       val/test predictions (val predictions are honestly out of
                       sample -- downstream combiners MUST fit their weights on these).
      ar_ridge_recal   val-period LOG-SPACE recalibration of ar_ridge (ported from
                       scripts/analysis/forecast_combination.py: stars lie in [1, 5],
                       strictly positive, so log space is safe):
                           f' = exp(a + b * log f), (a, b) val-fit, TEST-frozen.
                       Its val rows are in-sample for the recalibration and are
                       flagged as such -- use the raw arm's val rows for combining.
      global_mean      expanding mean label (train mean for val rows, train+val mean
                       for test rows; never sees test labels).
      last_value       prediction = ar_last_mean (the event month's own mean stars).
  (b) tfidf_chrono     TF-IDF (1-2 gram, 50k vocabulary, sublinear tf; vectoriser
                       fitted on TRAIN text only) + ridge, chronological split.
  (c) tfidf_naive_pooled  the SAME TF-IDF + ridge design evaluated under a pooled
                       RANDOM 80/20 split over all events (the field's standard
                       rating-prediction design) -- the demonstration arm that must
                       "credit text" (gate G3).

Prediction parquet schema (one file per arm, both horizons stacked):
    [entity_id, event_time, split, horizon_months, label, prediction, arm]
    split in {val, test} for chronological arms; {random_test} for the naive arm.

Gates enforced with HARD assertions:
    G2: recalibrated AR beats global-mean AND last-value on test (both horizons),
        and the recalibration slope b lies in [0.5, 1.5];
    G3: the naive pooled-split arm credits text (text MSE < pooled-mean MSE).

No look-ahead anywhere: models fit on train; alpha and recalibration on val;
test frozen. The pooled random split is INTENTIONALLY leaky -- that is the point.

Usage:
    python yelp_baseline_text.py --panel <panel.parquet> --out-dir <dir> [--tag RUN]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

EPS = 1e-8
CLIP_LO, CLIP_HI = 1.0, 5.0
FEATS = ["ar_last_mean", "ar_roll3_mean", "ar_roll12_mean", "ar_log_n_reviews"]
ALPHAS_AR = (0.01, 0.1, 1.0, 10.0, 100.0)
ALPHAS_TXT = (0.1, 1.0, 10.0, 100.0, 1000.0)
HORIZONS = (1, 3)
RECAL_B_LO, RECAL_B_HI = 0.5, 1.5


def mse(y, f) -> float:
    y = np.asarray(y, float)
    f = np.asarray(f, float)
    return float(np.mean((y - f) ** 2))


def clip(f):
    return np.clip(np.asarray(f, float), CLIP_LO, CLIP_HI)


def fit_ridge_select(Xtr, ytr, Xv, yv, alphas):
    """Fit one ridge per alpha on TRAIN, pick the alpha minimising val MSE.

    Returns (fitted model, chosen alpha, val predictions). The winning model is the
    train-fit one -- it is never refit on validation data.
    """
    best = None
    for a in alphas:
        model = Ridge(alpha=a)
        model.fit(Xtr, ytr)
        m = mse(yv, model.predict(Xv))
        if best is None or m < best[0]:
            best = (m, a, model)
    _, alpha, model = best
    return model, alpha, clip(model.predict(Xv))


def log_recalibrate(yv, fv):
    """Val-fit log-space recalibration (port of forecast_combination.log_combo's
    price-only leg). Stars are strictly positive so the logs are safe."""
    ly = np.log(np.clip(np.asarray(yv, float), EPS, None))
    lf = np.log(np.clip(np.asarray(fv, float), EPS, None))
    A = np.column_stack([np.ones(len(lf)), lf])
    beta, *_ = np.linalg.lstsq(A, ly, rcond=None)
    return float(beta[0]), float(beta[1])


def apply_recal(a, b, f):
    return clip(np.exp(a + b * np.log(np.clip(np.asarray(f, float), EPS, None))))


def arm_frame(sub: pd.DataFrame, pred, split, arm: str) -> pd.DataFrame:
    out = sub[["entity_id", "event_time", "horizon_months", "label"]].copy()
    out["split"] = split
    out["prediction"] = np.asarray(pred, dtype=np.float64)
    out["arm"] = arm
    assert np.isfinite(out.prediction).all(), f"non-finite predictions in {arm}"
    assert (out.prediction > 0).all(), f"non-positive predictions in {arm}"
    return out[["entity_id", "event_time", "split", "horizon_months",
                "label", "prediction", "arm"]]


def run_horizon(d: pd.DataFrame, h: int, rng: np.random.Generator, tag: str):
    """Run every arm for one horizon; returns (list of arm frames, metrics dict)."""
    tr = d[d.split == "train"].reset_index(drop=True)
    va = d[d.split == "val"].reset_index(drop=True)
    te = d[d.split == "test"].reset_index(drop=True)
    assert len(tr) and len(va) >= 100 and len(te) >= 30, "split floors violated"
    assert tr.event_time.max() < va.event_time.min(), "train/val overlap in time"
    assert va.event_time.max() < te.event_time.min(), "val/test overlap in time"

    ytr, yv, yt = (s.label.to_numpy(float) for s in (tr, va, te))
    frames, met = [], {"horizon_months": h, "n_train": len(tr), "n_val": len(va),
                       "n_test": len(te)}

    # ---- (a) AR ridge + val-period log-space recalibration -------------------------
    scaler = StandardScaler().fit(tr[FEATS])
    Xtr, Xv, Xt = (scaler.transform(s[FEATS]) for s in (tr, va, te))
    ar_model, ar_alpha, f_va = fit_ridge_select(Xtr, ytr, Xv, yv, ALPHAS_AR)
    f_te = clip(ar_model.predict(Xt))
    a, b = log_recalibrate(yv, f_va)          # val-fit, test-frozen
    assert RECAL_B_LO <= b <= RECAL_B_HI, \
        f"GATE G2 FAIL (h={h}m): recalibration slope b={b:.3f} outside [0.5, 1.5]"
    f_va_rc, f_te_rc = apply_recal(a, b, f_va), apply_recal(a, b, f_te)

    g_val = float(np.mean(ytr))                # train mean -> val rows
    g_test = float(np.mean(np.r_[ytr, yv]))    # train+val mean -> test rows
    lv_va, lv_te = va.ar_last_mean.to_numpy(float), te.ar_last_mean.to_numpy(float)

    met.update(ar_alpha=ar_alpha, recal_a=round(a, 4), recal_b=round(b, 4),
               mse_test_global_mean=mse(yt, g_test),
               mse_test_last_value=mse(yt, lv_te),
               mse_test_ar_raw=mse(yt, f_te),
               mse_test_ar_recal=mse(yt, f_te_rc))
    assert met["mse_test_ar_recal"] < met["mse_test_global_mean"], \
        f"GATE G2 FAIL (h={h}m): recalibrated AR does not beat global mean"
    assert met["mse_test_ar_recal"] < met["mse_test_last_value"], \
        f"GATE G2 FAIL (h={h}m): recalibrated AR does not beat last value"

    frames += [
        arm_frame(va, f_va, "val", "ar_ridge"), arm_frame(te, f_te, "test", "ar_ridge"),
        arm_frame(va, f_va_rc, "val", "ar_ridge_recal"),   # in-sample for the recal fit
        arm_frame(te, f_te_rc, "test", "ar_ridge_recal"),
        arm_frame(va, np.full(len(va), g_val), "val", "global_mean"),
        arm_frame(te, np.full(len(te), g_test), "test", "global_mean"),
        arm_frame(va, lv_va, "val", "last_value"),
        arm_frame(te, lv_te, "test", "last_value"),
    ]

    # ---- (b) TF-IDF + ridge, chronological split ------------------------------------
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True)
    Ttr = vec.fit_transform(tr.text)           # vocabulary and idf from TRAIN only
    Tv, Tt = vec.transform(va.text), vec.transform(te.text)
    txt_model, txt_alpha, t_va = fit_ridge_select(Ttr, ytr, Tv, yv, ALPHAS_TXT)
    t_te = clip(txt_model.predict(Tt))
    met.update(tfidf_alpha=txt_alpha, mse_test_tfidf_chrono=mse(yt, t_te))
    frames += [arm_frame(va, t_va, "val", "tfidf_chrono"),
               arm_frame(te, t_te, "test", "tfidf_chrono")]

    # ---- (c) NAIVE demonstration arm: pooled RANDOM split (field-standard design) ---
    n = len(d)
    idx = rng.permutation(n)
    n_tr = int(0.8 * n)
    itr, ite = idx[:n_tr], idx[n_tr:]
    d_tr, d_te = d.iloc[itr].reset_index(drop=True), d.iloc[ite].reset_index(drop=True)
    inner = rng.permutation(n_tr)
    n_in = int(0.9 * n_tr)
    ii_tr, ii_va = inner[:n_in], inner[n_in:]

    nvec = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True)
    N_tr = nvec.fit_transform(d_tr.text)
    N_te = nvec.transform(d_te.text)
    y_ntr, y_nte = d_tr.label.to_numpy(float), d_te.label.to_numpy(float)
    _, n_alpha, _ = fit_ridge_select(N_tr[ii_tr], y_ntr[ii_tr],
                                     N_tr[ii_va], y_ntr[ii_va], ALPHAS_TXT)
    n_model = Ridge(alpha=n_alpha).fit(N_tr, y_ntr)
    f_naive = clip(n_model.predict(N_te))

    pool_mean = float(np.mean(y_ntr))
    # context: the AR features under the same leaky split
    nsc = StandardScaler().fit(d_tr[FEATS])
    nar_model, _, _ = fit_ridge_select(nsc.transform(d_tr.iloc[ii_tr][FEATS]),
                                       y_ntr[ii_tr],
                                       nsc.transform(d_tr.iloc[ii_va][FEATS]),
                                       y_ntr[ii_va], ALPHAS_AR)
    f_nar = clip(nar_model.predict(nsc.transform(d_te[FEATS])))

    m_text, m_pool, m_ar = mse(y_nte, f_naive), mse(y_nte, pool_mean), mse(y_nte, f_nar)
    gain_pct = 100.0 * (m_pool - m_text) / m_pool
    met.update(naive_n_test=len(d_te), naive_alpha=n_alpha,
               mse_naive_text=m_text, mse_naive_pooled_mean=m_pool,
               mse_naive_ar=m_ar, naive_text_gain_vs_pooled_mean_pct=gain_pct)
    assert m_text < m_pool, \
        f"GATE G3 FAIL (h={h}m): naive pooled-split arm does not credit text"
    frames.append(arm_frame(d_te, f_naive, "random_test", "tfidf_naive_pooled"))

    print(f"\n[{tag}] h={h}m  (train/val/test = {len(tr):,}/{len(va):,}/{len(te):,})")
    print(f"  test MSE: global-mean={met['mse_test_global_mean']:.4f}  "
          f"last-value={met['mse_test_last_value']:.4f}  "
          f"AR raw={met['mse_test_ar_raw']:.4f}  "
          f"AR recal={met['mse_test_ar_recal']:.4f} (a={a:+.3f}, b={b:.3f})  "
          f"TF-IDF chrono={met['mse_test_tfidf_chrono']:.4f}")
    print(f"  NAIVE pooled random split (n_test={len(d_te):,}): "
          f"text={m_text:.4f} vs pooled-mean={m_pool:.4f} "
          f"(apparent text gain {gain_pct:+.1f}%; AR same split={m_ar:.4f})")
    return frames, met


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="results/second_domain/yelp_panel.parquet")
    ap.add_argument("--out-dir", default="results/second_domain/preds")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--tag", default="RUN", help="banner label, e.g. SYNTHETIC")
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    need = {"entity_id", "event_time", "split", "horizon_months", "label",
            "text", *FEATS}
    assert need.issubset(panel.columns), f"panel missing columns: {need - set(panel.columns)}"
    assert panel.label.between(CLIP_LO, CLIP_HI).all(), "labels outside [1, 5]"
    assert (panel.text.str.len() > 0).all(), "empty text rows"

    rng = np.random.default_rng(args.seed)
    all_frames, metrics = [], []
    for h in HORIZONS:
        d = (panel[panel.horizon_months == h]
             .sort_values(["entity_id", "event_time"], kind="mergesort")
             .reset_index(drop=True))
        frames, met = run_horizon(d, h, rng, args.tag)
        all_frames += frames
        metrics.append(met)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stacked = pd.concat(all_frames, ignore_index=True)
    for arm, g in stacked.groupby("arm"):
        g.reset_index(drop=True).to_parquet(out / f"preds_{arm}.parquet", index=False)
    with open(out / "baseline_metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"tag": args.tag, "seed": args.seed, "panel": str(args.panel),
                   "horizons": metrics}, fh, indent=2)

    print(f"\n[{args.tag}] GATES: G2 PASS (AR recal beats global-mean and last-value; "
          f"b in [0.5, 1.5] both horizons); G3 PASS (naive pooled arm credits text)")
    print(f"wrote {out}/preds_<arm>.parquet for arms: "
          f"{sorted(stacked.arm.unique())} + baseline_metrics.json")


if __name__ == "__main__":
    main()
