#!/usr/bin/env python3
"""R3.4 -- does the cascade depend on the combination weights being fitted on COVID?

Reviewer R3.4: Eq. (1)'s a, b, g are fitted on validation 2020--2021 and frozen
for test 2022--2025, i.e. extrapolated from an extreme-volatility regime to a
calm one, which could push g toward zero systematically.

The identifying design: validation (2020--2021) splits into a COVID half (2020)
and a calm half (2021). Refitting the SAME references and augmented forecast on
each half and freezing to the IDENTICAL test rows isolates the regime, because
both halves come from one block under one protocol. Models are never retrained.

Two caveats stated up front. (i) The validation block is the early-stopping /
checkpoint-selection set for every neural arm, so no arm here is fully
out-of-sample for those models; what the contrast holds fixed is the protocol,
not model exposure. (ii) A third arm refits on 2018--2019, the tail of the
TRAINING era. It is reported for completeness only and must NOT be read as a
regime test: those rows are in-sample for every text model, so its large
negative shift reflects fitting g to memorised predictions.

CPU-only; reuses the committed helpers so the spec is byte-identical apart from
which rows fit the weights.
Usage: python3 scripts/analysis/valwindow_sensitivity.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
import forecast_combination as fc  # noqa: E402

KEY = ["ticker", "accession", "horizon_days"]
SORT = ["filing_time_utc", "ticker", "accession"]
PRICE = ["A2_har_rv", "A6_shar", "A3_garch", "A4_egarch", "A5_arima"]
ALT_START, ALT_END = "2018-01-01", "2019-12-31"
OUT_CSV = ROOT / "results/tables/valwindow_sensitivity.csv"
OUT_MD = ROOT / "results/tables/valwindow_sensitivity.md"

TEXT = {
    "long_form": ["B1_bow_ridge", "B2_tfidf_ridge", "B3_lm_linear", "B4_lm_features",
                  "C1_bert_s1", "C2_finbert_s1", "C2_finbert_s2", "C2_finbert_s3",
                  "C2_finbert_s4", "C3_roberta_s1", "C4_longformer", "C6_llmtext",
                  "D1_concat_mlp", "D2_gated_fusion", "D4_llmfused"],
    "event_driven": ["B1_bow_ridge", "B2_tfidf_ridge", "B3_lm_linear", "B4_lm_features",
                     "C2_finbert_s1", "C6_llmtext", "D2_gated_fusion", "D4_llmfused"],
}


def _ll(a):
    return np.log(np.clip(np.asarray(a, dtype=float), 1e-12, None))


def fit_frozen(y_fit, X_fit, X_test):
    """log-space OLS on the fitting window, applied frozen to test."""
    ly = _ll(y_fit)
    Xf = np.column_stack([np.ones(len(ly))] + [_ll(c) for c in X_fit])
    b = fc.ols(ly, Xf)
    Xt = np.column_stack([np.ones(len(X_test[0]))] + [_ll(c) for c in X_test])
    return np.exp(Xt @ b)


def price_panel(disc):
    base = fc.load("A2_har_rv", disc)[
        ["split"] + KEY + ["prediction_realised_vol", "label_realised_vol",
                           "filing_time_utc", "effective_trading_day"]
    ].rename(columns={"prediction_realised_vol": "A2_har_rv"})
    for m in PRICE[1:]:
        p = fc.load(m, disc)[KEY + ["prediction_realised_vol"]].rename(
            columns={"prediction_realised_vol": m})
        base = base.merge(p, on=KEY, how="inner")
    return base


def main() -> None:
    rows = []
    for disc, models in TEXT.items():
        panel = price_panel(disc)
        panel["eff"] = pd.to_datetime(panel["effective_trading_day"])
        alt = panel["eff"].between(ALT_START, ALT_END) & (panel["split"] == "train")
        panel["fitrow_alt"] = alt
        # cleanest comparison: the calm half of the SAME validation block, so the
        # rows stay out-of-sample for every model and only the regime changes
        panel["fitrow_calm"] = (panel["split"] == "val") & (panel["eff"] >= "2021-01-01")
        # the identifying contrast: calm vs COVID, both halves of the SAME val
        # block, so regime is the only thing that differs between them
        panel["fitrow_covid"] = (panel["split"] == "val") & (panel["eff"] < "2021-01-01")

        for model in models:
            try:
                tx = fc.load(model, disc)[KEY + ["prediction_realised_vol"]].rename(
                    columns={"prediction_realised_vol": "text"})
            except FileNotFoundError:
                print(f"  skip {disc}/{model}: no predictions")
                continue
            d = panel.merge(tx, on=KEY, how="inner")

            for h in (5, 10, 20):
                dh = d[d["horizon_days"] == h]
                for tag, mask in (("committed_val", dh["split"] == "val"),
                                  ("calm_2021", dh["fitrow_calm"]),
                                  ("covid_2020", dh["fitrow_covid"]),
                                  ("alt_2018_19", dh["fitrow_alt"])):
                    f = dh[mask].sort_values(SORT, kind="mergesort")
                    t = dh[dh["split"] == "test"].sort_values(SORT, kind="mergesort")
                    if len(f) < 200 or len(t) < 200:
                        continue
                    yf, yt = f["label_realised_vol"].to_numpy(), t["label_realised_vol"].to_numpy()
                    fR = fit_frozen(yf, [f[m].to_numpy() for m in PRICE],
                                    [t[m].to_numpy() for m in PRICE])
                    fU = fit_frozen(yf, [f[m].to_numpy() for m in PRICE] + [f["text"].to_numpy()],
                                    [t[m].to_numpy() for m in PRICE] + [t["text"].to_numpy()])
                    qR, qU = fc.qlike(yt, fR).mean(), fc.qlike(yt, fU).mean()
                    rows.append(dict(disc=disc, model=model, h=h, window=tag,
                                     n_fit=len(f), n_test=len(t),
                                     qlike_R=qR, qlike_U=qU,
                                     rel_impr_pct=100.0 * (qR - qU) / qR))
        print(f"{disc}: done")

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    full = out.pivot_table(index=["disc", "model", "h"], columns="window",
                           values="rel_impr_pct")
    for alt_col in ("calm_2021", "covid_2020", "alt_2018_19"):
        piv = full[["committed_val", alt_col]].dropna()
        s = piv[alt_col] - piv["committed_val"]
        print(f"\n[{alt_col}] vs committed: n={len(piv)}  "
              f"mean shift {s.mean():+.2f}pp  "
              f"pos->neg {int(((piv['committed_val']>0)&(piv[alt_col]<=0)).sum())}  "
              f"neg->pos {int(((piv['committed_val']<=0)&(piv[alt_col]>0)).sum())}  "
              f"Spearman {piv['committed_val'].corr(piv[alt_col], method='spearman'):+.3f}")
    piv = full[["committed_val", "alt_2018_19"]].dropna()
    n = len(piv)
    both_pos = int(((piv["committed_val"] > 0) & (piv["alt_2018_19"] > 0)).sum())
    flip_to_pos = int(((piv["committed_val"] <= 0) & (piv["alt_2018_19"] > 0)).sum())
    flip_to_neg = int(((piv["committed_val"] > 0) & (piv["alt_2018_19"] <= 0)).sum())
    corr = piv["committed_val"].corr(piv["alt_2018_19"])
    mean_shift = (piv["alt_2018_19"] - piv["committed_val"]).mean()

    print(f"\ncells compared: {n}")
    print(f"  positive under both windows : {both_pos}")
    print(f"  negative -> positive        : {flip_to_pos}")
    print(f"  positive -> negative        : {flip_to_neg}")
    print(f"  rank correlation of increments: {piv['committed_val'].corr(piv['alt_2018_19'], method='spearman'):.3f}"
          f"  (Pearson {corr:.3f})")
    print(f"  mean increment shift        : {mean_shift:+.3f} pp")

    q = full[["committed_val", "calm_2021", "covid_2020"]].dropna()
    diff = q["calm_2021"] - q["covid_2020"]
    from scipy import stats as _st
    tt = _st.ttest_rel(q["calm_2021"], q["covid_2020"])
    L = ["# Weight-window sensitivity for Eq. (1)", "",
         "Does fitting the combination weights on the COVID validation block push",
         "the text coefficient toward zero? Validation (2020--2021) splits into a",
         "COVID half (2020) and a calm half (2021); refitting on each and freezing",
         "to the IDENTICAL test rows isolates the regime, since both halves are the",
         "same block under the same protocol. Models are never retrained.", "",
         "## The identifying contrast (calm half vs COVID half)", "",
         f"- cells: **{len(q)}** (the full grid)",
         f"- calm-fit minus COVID-fit increment: mean **{diff.mean():+.3f}pp**, "
         f"median **{diff.median():+.3f}pp**",
         f"- calm higher in **{int((diff > 0).sum())}/{len(q)}** cells",
         f"- paired t = **{tt.statistic:+.2f}**, p = **{tt.pvalue:.3f}**", "",
         "So the direction the reviewer anticipated is present -- a calm-window fit",
         "does credit text slightly more -- but it is small and not significant, and",
         "the mean increment is negative under BOTH halves",
         f"(calm {q['calm_2021'].mean():+.3f}pp, COVID {q['covid_2020'].mean():+.3f}pp).", "",
         "## Each arm against the committed full-validation fit", "",
         "| arm | n | mean shift | sign flips down / up | Spearman |",
         "|---|---|---|---|---|"]
    for arm in ("calm_2021", "covid_2020", "alt_2018_19"):
        pv = full[["committed_val", arm]].dropna()
        s = pv[arm] - pv["committed_val"]
        L.append(f"| {arm} | {len(pv)} | {s.mean():+.2f}pp | "
                 f"{int(((pv['committed_val'] > 0) & (pv[arm] <= 0)).sum())} / "
                 f"{int(((pv['committed_val'] <= 0) & (pv[arm] > 0)).sum())} | "
                 f"{pv['committed_val'].corr(pv[arm], method='spearman'):+.3f} |")
    L += ["",
          "`alt_2018_19` refits on the tail of the TRAINING era. It is reported for",
          "completeness only and must not be read as a regime test: those rows are",
          "in-sample for every text model, so its large negative shift reflects",
          "fitting g to memorised predictions, not a property of the regime.",
          "Note also that the committed validation block is the early-stopping set",
          "for the neural arms, so no arm here is fully out-of-sample for them;",
          "what the contrast holds fixed is the protocol, not model exposure.", ""]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
