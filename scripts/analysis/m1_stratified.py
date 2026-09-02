"""M1 STRATIFIED incremental value — WHERE does disclosure text add value over a
RECALIBRATED HAR-RV forecast, out of sample?

Extends the M1 log-space nested forecast-combination (scripts/analysis/forecast_combination.py)
to a stratified OOS analysis. For each (disclosure, horizon, text_model) cell we:
  1. load A2_har_rv (fHAR) and the text model (fText); join on [ticker,accession,horizon_days];
     split into val / test.
  2. fit the log-space combiner on VAL ONLY via fc.log_combo:
        f_R = exp(a + b*log fHAR)                 recalibrated price-only reference
        f_U = exp(a + b*log fHAR + g*log fText)    + text
     apply FROZEN to TEST -> per-test-obs f_R, f_U.
  3. per test obs: qR = qlike(y,f_R), qU = qlike(y,f_U); increment d = qR - qU
     (positive = text helps).
  4. PARTITION test obs into strata (weights are NOT refit per stratum). Within each
     stratum report: n, rel QLIKE improvement % = 100*mean(d)/mean(qR),
     DM on QLIKE via dm_test(qU,qR,h) (negative stat = text better),
     moving-block 95% CI of mean(d), and a label-shuffle PLACEBO increment (permute fText
     rows on val AND test with a fixed rng seed 2026, rerun the combiner, recompute the
     stratum mean d_placebo).

STRATA AXES:
  (a) vol_regime : terciles of feature_rv_22d on the TEST set -> low / mid / high
  (b) period     : filing-year buckets adapted to the confirmed test range 2022-2025 ->
                    {<=2022, 2023, 2024-2025}
  (c) form       : long_form -> 10-K vs 10-Q ; event_driven -> top-4 item_subtype (+ Other) ;
                   combined -> skipped
Every axis also emits an ALL stratum (the pooled cell) used for the sanity reconciliation
against forecast_combination_summary.json.

Leakage discipline: combiner fit on split=="val" ONLY, applied frozen to test. Strata
partition TEST residuals only; nothing is refit per stratum.

Run from repo root:  .venv/bin/python scripts/analysis/m1_stratified.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/analysis")
import forecast_combination as fc  # qlike, se, log_combo, moving_block_ci, holm, load
sys.path.insert(0, "src")
from sp500vol.evaluation.dm_test import dm_test

KEY = ["ticker", "accession", "horizon_days"]
SORT = ["filing_time_utc", "ticker", "accession"]
HORIZONS = (5, 10, 20)
PLACEBO_SEED = 2026

# Union of (i) models in genuine_cells (B1-B4, C1_bert_s1, C2_finbert_s1, C4_longformer)
# and (ii) best-of-family reps (C2_finbert_s1, C4_longformer, best C5 = C5_qwen3 by test
# QLIKE, D2_gated_fusion).
MODELS = [
    "B1_bow_ridge", "B2_tfidf_ridge", "B3_lm_linear", "B4_lm_features",
    "C1_bert_s1", "C2_finbert_s1", "C4_longformer", "C5_qwen3", "D2_gated_fusion",
]
DISCLOSURES = ["long_form", "event_driven", "combined"]


def period_bucket(years):
    """filing-year -> {<=2022, 2023, 2024-2025}. Test range confirmed 2022-2025."""
    y = np.asarray(years, int)
    out = np.empty(len(y), dtype=object)
    out[y <= 2022] = "<=2022"
    out[y == 2023] = "2023"
    out[y >= 2024] = "2024-2025"
    return out


def vol_tercile(rv22):
    """Terciles of feature_rv_22d on the TEST set -> low/mid/high."""
    rv = np.asarray(rv22, float)
    q1, q2 = np.nanquantile(rv, [1 / 3, 2 / 3])
    out = np.full(len(rv), "mid", dtype=object)
    out[rv <= q1] = "low"
    out[rv > q2] = "high"
    return out


def form_bucket(disc, form, item_subtype):
    """(c) disclosure form axis. long_form: 10-K vs 10-Q. event_driven: top-4 item_subtype
    (+Other). combined: skipped (returns None)."""
    if disc == "long_form":
        return np.asarray(form, dtype=object)
    if disc == "event_driven":
        it = pd.Series(item_subtype).astype(object)
        top4 = it.value_counts().head(4).index.tolist()
        return np.where(it.isin(top4), it, "Other").astype(object)
    return None


def stratum_stats(d, qR, qU, h, d_placebo):
    """Given a boolean-free selection already applied, compute the per-stratum row pieces."""
    n = len(d)
    mqR = float(qR.mean())
    rel = 100.0 * float(d.mean()) / mqR if mqR > 0 else float("nan")
    # DM on QLIKE: dm_test(qU, qR): positive stat => qU worse; negative => text better.
    if n >= max(2 * h, 8):
        dm_stat, dm_p = dm_test(qU, qR, h=h)
        _, lo, hi = fc.moving_block_ci(d, h)
    else:
        dm_stat, dm_p, lo, hi = (float("nan"),) * 4
    rel_placebo = 100.0 * float(d_placebo.mean()) / mqR if mqR > 0 else float("nan")
    return n, rel, float(dm_stat), float(dm_p), lo, hi, rel_placebo


def run_cell(disc, model, h):
    """Return (list-of-rows, pooled_rel) for one (disc, model, h) cell, or (None, None)."""
    har = fc.load("A2_har_rv", disc)[["split"] + KEY + [
        "prediction_realised_vol", "label_realised_vol", "filing_time_utc",
        "form", "item_subtype", "feature_rv_22d"]].rename(
        columns={"prediction_realised_vol": "fhar"})
    txt = fc.load(model, disc)[KEY + ["prediction_realised_vol"]].rename(
        columns={"prediction_realised_vol": "ftext"})
    d = har.merge(txt, on=KEY)

    dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(SORT, kind="mergesort")
    dt = d[(d.horizon_days == h) & (d.split == "test")].sort_values(SORT, kind="mergesort")
    if len(dv) < 100 or len(dt) < 30:
        return None, None

    yv, fhv, ftv = dv.label_realised_vol.to_numpy(), dv.fhar.to_numpy(), dv.ftext.to_numpy()
    yt, fhr, ftt = dt.label_realised_vol.to_numpy(), dt.fhar.to_numpy(), dt.ftext.to_numpy()

    # --- REAL combiner: fit on val, frozen to test ---
    fR, fU, _g = fc.log_combo(yv, fhv, ftv, fhr, ftt)
    qR = fc.qlike(yt, fR)
    qU = fc.qlike(yt, fU)
    dinc = qR - qU  # per-test-obs increment (positive = text helps)

    # --- PLACEBO combiner: permute fText on val AND test (fixed seed), frozen to test ---
    rng = np.random.default_rng(PLACEBO_SEED)
    ftv_p = rng.permutation(ftv)
    ftt_p = rng.permutation(ftt)
    fRp, fUp, _gp = fc.log_combo(yv, fhv, ftv_p, fhr, ftt_p)
    qRp = fc.qlike(yt, fRp)
    qUp = fc.qlike(yt, fUp)
    dinc_p = qRp - qUp

    # test-obs strata labels
    years = pd.to_datetime(dt.filing_time_utc).dt.year.to_numpy()
    axes = {
        "all": np.full(len(dt), "ALL", dtype=object),
        "vol_regime": vol_tercile(dt.feature_rv_22d.to_numpy()),
        "period": period_bucket(years),
    }
    fb = form_bucket(disc, dt.form.to_numpy(), dt.item_subtype.to_numpy())
    if fb is not None:
        axes["form"] = fb

    rows = []
    pooled_rel = None
    for axis, labels in axes.items():
        for stratum in pd.unique(labels):
            mask = labels == stratum
            n, rel, dm_stat, dm_p, lo, hi, rel_pl = stratum_stats(
                dinc[mask], qR[mask], qU[mask], h, dinc_p[mask])
            if axis == "all":
                pooled_rel = rel
                axis_out, stratum_out = "all", "ALL"
            else:
                axis_out, stratum_out = axis, str(stratum)
            rows.append({
                "model": model, "disclosure": disc, "horizon": h,
                "axis": axis_out, "stratum": stratum_out, "n": int(n),
                "rel_impr_pct": rel, "dm_q_stat": dm_stat, "dm_q_p": dm_p,
                "ci_lo": lo, "ci_hi": hi, "placebo_rel_impr_pct": rel_pl,
            })
    return rows, pooled_rel


def main():
    all_rows = []
    pooled = {}  # (disc, model, h) -> pooled rel_impr_pct  (for sanity)
    test_ranges = {}
    for disc in DISCLOSURES:
        for model in MODELS:
            for h in HORIZONS:
                rows, prel = run_cell(disc, model, h)
                if rows is None:
                    continue
                all_rows.extend(rows)
                pooled[(disc, model, h)] = prel

    df = pd.DataFrame(all_rows)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    csv_cols = ["model", "disclosure", "horizon", "axis", "stratum", "n",
                "rel_impr_pct", "dm_q_stat", "dm_q_p", "ci_lo", "ci_hi",
                "placebo_rel_impr_pct"]
    df[csv_cols].to_csv("results/tables/m1_stratified.csv", index=False)

    # ---------- readable markdown, grouped by axis ----------
    md = ["# M1 stratified — WHERE does disclosure text add incremental value over a "
          "recalibrated HAR?\n",
          "Per (disclosure, horizon, text_model) cell the log-space nested combiner "
          "(`f_R=exp(a+b·log fHAR)` vs `f_U=+g·log fText`) is fit on VALIDATION and applied "
          "frozen to TEST; test residuals are then partitioned into strata WITHOUT refitting. "
          "`rel_impr_pct` = 100·mean(qR−qU)/mean(qR) (positive = text helps). "
          "`dm_q_stat` from dm_test(qU,qR) — NEGATIVE = text better. `placebo` permutes the "
          "text forecast (seed 2026): near-zero confirms a real signal.\n",
          "Test filing-date range (both long_form & event_driven): **2022-2025**; period "
          "buckets `<=2022 / 2023 / 2024-2025`.\n"]

    def block(axis, title, note=""):
        sub = df[df.axis == axis]
        if sub.empty:
            return
        md.append(f"\n## Axis ({axis}) — {title}{note}\n"
                  "| model | disclosure | h | stratum | n | rel% | DM-Q | DM-Q p | CI lo | CI hi | placebo% |\n"
                  "|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in sub.sort_values(["disclosure", "model", "horizon", "stratum"]).iterrows():
            md.append(f"| {r.model} | {r.disclosure} | {int(r.horizon)} | {r.stratum} | "
                      f"{int(r.n)} | {r.rel_impr_pct:+.2f} | {r.dm_q_stat:+.2f} | {r.dm_q_p:.4f} | "
                      f"{r.ci_lo:+.5f} | {r.ci_hi:+.5f} | {r.placebo_rel_impr_pct:+.2f} |")

    block("vol_regime", "volatility regime (terciles of feature_rv_22d on TEST)")
    block("period", "temporal period (filing year)")
    block("form", "disclosure form (long_form: 10-K/10-Q; event_driven: top-4 item_subtype +Other)")

    # highlight where text helps most / least (real, significant, placebo-null, non-ALL)
    sig = df[(df.axis != "all") & (df.dm_q_stat < 0) & (df.dm_q_p < 0.05)
             & (df.placebo_rel_impr_pct.abs() < 0.5)].copy()
    md.append("\n## Where text helps MOST (significant, placebo-null strata)\n"
              "| model | disclosure | h | axis | stratum | n | rel% | DM-Q | placebo% |\n"
              "|---|---|---|---|---|---|---|---|---|")
    for _, r in sig.sort_values("rel_impr_pct", ascending=False).head(15).iterrows():
        md.append(f"| {r.model} | {r.disclosure} | {int(r.horizon)} | {r.axis} | {r.stratum} | "
                  f"{int(r.n)} | {r.rel_impr_pct:+.2f} | {r.dm_q_stat:+.2f} | {r.placebo_rel_impr_pct:+.2f} |")
    md.append("\n## Where text helps LEAST / hurts (significant strata)\n"
              "| model | disclosure | h | axis | stratum | n | rel% | DM-Q | placebo% |\n"
              "|---|---|---|---|---|---|---|---|---|")
    least = df[(df.axis != "all") & (df.dm_q_p < 0.05)].copy()
    for _, r in least.sort_values("rel_impr_pct").head(15).iterrows():
        md.append(f"| {r.model} | {r.disclosure} | {int(r.horizon)} | {r.axis} | {r.stratum} | "
                  f"{int(r.n)} | {r.rel_impr_pct:+.2f} | {r.dm_q_stat:+.2f} | {r.placebo_rel_impr_pct:+.2f} |")

    with open("results/tables/m1_stratified.md", "w") as fh:
        fh.write("\n".join(md))

    print("=== M1 stratified — done ===")
    print(f"rows={len(df)}  cells={len(pooled)}")
    print("wrote results/tables/m1_stratified.csv + .md")
    return df, pooled


if __name__ == "__main__":
    main()
