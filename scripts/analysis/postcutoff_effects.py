"""ROW 8 (REVIEW_ROUND3_FRESH_PANEL) — Post-training-cutoff EFFECT SIZES + 95% CIs
for the 8-K (C6) residual.

llm_contamination.{csv,md} already era-splits the C6 increment at 2024-07-01 and
reports significance COUNTS (5/6 Holm post-cutoff vs HAR; 3/6 beyond identity).
Round-3 perspective reviewer (MAJOR, freeze-table row 8) wants EFFECT SIZES WITH
UNCERTAINTY: for each C6 cell (2 disclosures x 3 horizons), vs BOTH references —

  (a) single recalibrated HAR   f_R  = exp OLS[1, log fHAR]            (val-fit)
  (b) identity-augmented        f_R' = exp OLS[1, log fHAR, log f_datefirm] (val-fit)

— report on the PRE- and POST-cutoff test subsamples separately: n_obs, n_days,
rel%, day-clustered DM, raw p, Holm p, and a day-block moving-bootstrap 95% CI of
the MEAN DAILY loss differential.

NO-LOOK-AHEAD / NO REFIT: combiner weights are the ORIGINAL full-validation fit,
frozen; ONLY the test evaluation is stratified at 2024-07-01. No weight is ever
refit on a subsample.

BOOTSTRAP: reuses the committed clustered_dm.mbb_ci_daily (moving-block bootstrap
over DAYS of the daily-mean loss differential, block length = h days) with its
committed defaults B=2000 draws, seed 2026. (The brief's fallback spec of 1000
draws applies only if no committed bootstrap existed — it does, so it is reused.)
The post-minus-pre difference CI uses the same block scheme with one rng seeded
2026 per cell, drawing pre then post (independent draws).

HOLM FAMILY (PRE-DECLARED, also in the md BEFORE the results table): one family =
the 24 stratified cells (2 disclosures x 3 horizons x 2 references x 2 strata),
Holm within it — identical to the committed llm_contamination cutoff block, so the
Holm column must reproduce it. Full-sample rows are CONTEXT only; their Holm values
are inherited from the committed llm_contamination families (no new inference).

SANITY GATE (blocking — numbers do not ship past a failure):
  A (required by brief): full-sample fulltext-vs-HAR rel%/DM/p/n must reproduce
     results/tables/llm_contamination.csv block=variant arm=fulltext rows exactly
     (machine precision).
  B: full-sample beyond-identity rows must reproduce block=joint rows exactly.
  C: the 24 stratified rel%/DM/p/n/Holm must reproduce block=cutoff rows exactly.

Run from the repo root:  .venv/bin/python scripts/analysis/postcutoff_effects.py
Outputs (NEW files): results/tables/postcutoff_effects.{csv,md}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/analysis")
import forecast_combination as fc  # noqa: E402
import clustered_dm as cdm  # noqa: E402

KEY = fc.KEY
EPS = 1e-8
HORIZONS = (5, 10, 20)
CUTOFF = pd.Timestamp("2024-07-01", tz="UTC")
BOOT_B, BOOT_SEED = 2000, 2026  # = mbb_ci_daily committed defaults, reused
REFS = ("fulltext_vs_har", "beyond_identity")
ANCHOR = "results/tables/llm_contamination.csv"  # sanity-gate anchor table


def ols(y, X):
    b, *_ = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)
    return b


def L(x):
    return np.log(np.clip(np.asarray(x, float), EPS, None))


def mbb_draws(dd, h, B, rng):
    """B moving-block-bootstrap means of the daily series dd (block length h days).
    Same scheme as clustered_dm.mbb_ci_daily. Returns None when n < 2h."""
    dd = np.asarray(dd, float)
    n = len(dd)
    Lb = max(int(h), 1)
    if n < 2 * Lb:
        return None
    nb = int(np.ceil(n / Lb))
    starts = rng.integers(0, n, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(Lb)[None, None, :]) % n
    return dd[idx.reshape(B, -1)[:, :n]].mean(axis=1)


def stratum_stats(lR, lU, days, mask, h):
    """Effect size + inference on one test subsample. Weights already frozen."""
    lR_s, lU_s, days_s = lR[mask], lU[mask], np.asarray(days)[mask]
    d = lU_s - lR_s  # negative = text arm better
    dm, p, nd = cdm.dm_test_clustered(lU_s, lR_s, days_s, h)
    mean_d, lo, hi = cdm.mbb_ci_daily(d, days_s, h, B=BOOT_B, seed=BOOT_SEED)
    qR, qU = float(lR_s.mean()), float(lU_s.mean())
    qR_daily = float(cdm.daily_mean(lR_s, days_s)[0].mean())
    return dict(
        n_test=int(mask.sum()), n_days=nd,
        qlike_R=qR, qlike_U=qU,
        rel_pct=100.0 * (qR - qU) / qR if qR > 0 else float("nan"),
        dm_clu=dm, p_raw=p,
        mean_daily_d=mean_d, ci_lo=lo, ci_hi=hi,
        qlike_R_daily=qR_daily,
        rel_daily_pct=-100.0 * mean_d / qR_daily,
        rel_ci_lo=-100.0 * hi / qR_daily,
        rel_ci_hi=-100.0 * lo / qR_daily,
    )


def main():
    rows, diff_rows = [], []

    for disc in ("long_form", "event_driven"):
        print(f"[{disc}] loading stored forecasts ...", flush=True)
        a2 = fc.load("A2_har_rv", disc)[
            KEY + ["split", "label_realised_vol", "prediction_realised_vol",
                   "effective_trading_day", "filing_time_utc"]
        ].rename(columns={"prediction_realised_vol": "fh"})
        ft = fc.load("C6_llmtext", disc)[KEY + ["prediction_realised_vol"]] \
            .rename(columns={"prediction_realised_vol": "f_fulltext"})
        df_ = fc.load("C6_datefirm", disc)[KEY + ["prediction_realised_vol"]] \
            .rename(columns={"prediction_realised_vol": "f_datefirm"})

        for h in HORIZONS:
            print(f"[{disc} h={h}] combiner (val-fit, frozen) + strata ...", flush=True)
            ah = a2[a2.horizon_days == h]

            # ---- (a) fulltext vs single recalibrated HAR (pairwise merge, as committed) ----
            m = ah.merge(ft[ft.horizon_days == h], on=KEY).dropna()
            v, te = m[m.split == "val"], m[m.split == "test"]
            fR, fU, _g = fc.log_combo(
                v.label_realised_vol.values, v.fh.values, v.f_fulltext.values,
                te.fh.values, te.f_fulltext.values)
            lR, lU = fc.qlike(te.label_realised_vol.values, fR), \
                fc.qlike(te.label_realised_vol.values, fU)
            days = te.effective_trading_day.values
            post = (te.filing_time_utc >= CUTOFF).values

            # ---- (b) identity-augmented reference (joint merge, as committed) ----
            mj = ah.merge(ft[ft.horizon_days == h], on=KEY) \
                   .merge(df_[df_.horizon_days == h], on=KEY, suffixes=("", "_df")) \
                   .dropna()
            vj, tej = mj[mj.split == "val"], mj[mj.split == "test"]
            yvj, ytj = vj.label_realised_vol.values, tej.label_realised_vol.values
            XR_v = np.column_stack([np.ones(len(vj)), L(vj.fh), L(vj.f_datefirm)])
            XU_v = np.column_stack([XR_v, L(vj.f_fulltext)])
            bR, bU = ols(L(yvj), XR_v), ols(L(yvj), XU_v)
            fRj = np.exp(bR[0] + bR[1] * L(tej.fh) + bR[2] * L(tej.f_datefirm))
            fUj = np.exp(bU[0] + bU[1] * L(tej.fh) + bU[2] * L(tej.f_datefirm)
                         + bU[3] * L(tej.f_fulltext))
            lRj, lUj = fc.qlike(ytj, fRj), fc.qlike(ytj, fUj)
            daysj = tej.effective_trading_day.values
            postj = (tej.filing_time_utc >= CUTOFF).values

            for ref, (lR_, lU_, days_, post_) in (
                ("fulltext_vs_har", (lR, lU, days, post)),
                ("beyond_identity", (lRj, lUj, daysj, postj)),
            ):
                per = {}
                for stratum, mask in (("full", np.ones(len(lR_), bool)),
                                      ("pre", ~post_), ("post", post_)):
                    st = stratum_stats(lR_, lU_, days_, mask, h)
                    per[stratum] = st
                    rows.append(dict(disc=disc, h=h, ref=ref, stratum=stratum, **st))

                # post-minus-pre difference of the mean daily differential (block bootstrap)
                d_all = lU_ - lR_
                dd_pre, _ = cdm.daily_mean(d_all[~post_], np.asarray(days_)[~post_])
                dd_post, _ = cdm.daily_mean(d_all[post_], np.asarray(days_)[post_])
                rng = np.random.default_rng(BOOT_SEED)
                dr_pre = mbb_draws(dd_pre, h, BOOT_B, rng)
                dr_post = mbb_draws(dd_post, h, BOOT_B, rng)
                diff_pt = per["post"]["mean_daily_d"] - per["pre"]["mean_daily_d"]
                if dr_pre is not None and dr_post is not None:
                    dlo, dhi = np.quantile(dr_post - dr_pre, [0.025, 0.975])
                else:
                    dlo = dhi = float("nan")
                diff_rows.append(dict(disc=disc, h=h, ref=ref,
                                      diff_daily_d=diff_pt,
                                      diff_ci_lo=float(dlo), diff_ci_hi=float(dhi)))

    out = pd.DataFrame(rows)
    diffs = pd.DataFrame(diff_rows)

    # ---- Holm within the PRE-DECLARED family: the 24 stratified cells ----
    strat_mask = out.stratum != "full"
    out.loc[strat_mask, "p_holm"] = fc.holm(out.loc[strat_mask, "p_raw"]
                                            .fillna(1.0).values)

    # full-sample rows: Holm inherited from the committed llm_contamination families
    lc = pd.read_csv(ANCHOR)
    inh = pd.concat([
        lc[(lc.block == "variant") & (lc.arm == "fulltext")]
        .assign(ref="fulltext_vs_har")[["disc", "h", "ref", "p_holm"]],
        lc[lc.block == "joint"]
        .assign(ref="beyond_identity")[["disc", "h", "ref", "p_holm"]],
    ]).rename(columns={"p_holm": "p_holm_committed"})
    out = out.merge(inh, on=["disc", "h", "ref"], how="left")
    full_mask = out.stratum == "full"
    out.loc[full_mask, "p_holm"] = out.loc[full_mask, "p_holm_committed"]
    out = out.drop(columns=["p_holm_committed"])

    # ================= SANITY GATES (blocking) =================
    def gate(mine, ref_tab, cols, holm_too=False):
        mrg = mine.merge(ref_tab, on=["disc", "h"] + (["arm", "stratum"]
                         if "arm" in ref_tab.columns and "stratum" in mine.columns
                         else []), suffixes=("_mine", "_ref"))
        report = {"n_rows_compared": int(len(mrg))}
        ok = len(mrg) == len(mine) == len(ref_tab)
        for c in cols:
            dmax = float((mrg[f"{c}_mine"] - mrg[f"{c}_ref"]).abs().max())
            report[f"max_abs_diff_{c}"] = dmax
            ok = ok and dmax <= 1e-12
        report["pass"] = bool(ok)
        return report

    gates = {}
    # A — full-sample fulltext vs HAR == block=variant arm=fulltext
    mineA = out[(out.ref == "fulltext_vs_har") & (out.stratum == "full")][
        ["disc", "h", "n_test", "n_days", "rel_pct", "dm_clu", "p_raw"]]
    refA = lc[(lc.block == "variant") & (lc.arm == "fulltext")][
        ["disc", "h", "n_test", "n_days", "rel_pct", "dm_clu", "p_raw"]]
    gates["A_fulltext_vs_har_full"] = gate(
        mineA, refA, ["n_test", "n_days", "rel_pct", "dm_clu", "p_raw"])
    # B — full-sample beyond identity == block=joint
    mineB = out[(out.ref == "beyond_identity") & (out.stratum == "full")][
        ["disc", "h", "n_test", "n_days", "rel_pct", "dm_clu", "p_raw"]]
    refB = lc[lc.block == "joint"][
        ["disc", "h", "n_test", "n_days", "rel_pct", "dm_clu", "p_raw"]]
    gates["B_beyond_identity_full"] = gate(
        mineB, refB, ["n_test", "n_days", "rel_pct", "dm_clu", "p_raw"])
    # C — 24 stratified cells (incl. Holm) == block=cutoff
    mineC = out[out.stratum != "full"][
        ["disc", "h", "ref", "stratum", "n_test", "n_days",
         "rel_pct", "dm_clu", "p_raw", "p_holm"]].rename(columns={"ref": "arm"})
    refC = lc[lc.block == "cutoff"][
        ["disc", "h", "arm", "stratum", "n_test", "n_days",
         "rel_pct", "dm_clu", "p_raw", "p_holm"]]
    gates["C_stratified_incl_holm"] = gate(
        mineC, refC, ["n_test", "n_days", "rel_pct", "dm_clu", "p_raw", "p_holm"])

    all_pass = all(g["pass"] for g in gates.values())
    print("SANITY GATES:", json.dumps(gates, indent=2))
    if not all_pass:
        print("SANITY GATE FAILED — anchor table results/tables/"
              "llm_contamination.csv not reproduced. NOT writing outputs.")
        sys.exit(1)

    # ================= headline aggregates =================
    P = out[out.stratum == "post"]
    PRE = out[out.stratum == "pre"]
    key = ["disc", "h", "ref"]
    sb = PRE.merge(P, on=key, suffixes=("_pre", "_post")).merge(diffs, on=key)

    def n_of(df_, cond):
        return int(cond(df_).sum())

    stats_ = {}
    for ref in REFS:
        p_ = P[P.ref == ref]
        s_ = sb[sb.ref == ref]
        stats_[ref] = dict(
            post_holm=n_of(p_, lambda d: (d.dm_clu < 0) & (d.p_holm < 0.05)),
            post_raw=n_of(p_, lambda d: (d.dm_clu < 0) & (d.p_raw < 0.05)),
            post_ci_excl0=n_of(p_, lambda d: d.ci_hi < 0),
            post_gt_pre=n_of(s_, lambda d: d.rel_pct_post > d.rel_pct_pre),
            post_dailyd_larger=n_of(
                s_, lambda d: d.mean_daily_d_post < d.mean_daily_d_pre),
            diff_ci_excl0=n_of(s_, lambda d: (d.diff_ci_hi < 0) | (d.diff_ci_lo > 0)),
            diff_sig_shrink=n_of(s_, lambda d: d.diff_ci_lo > 0),  # sig SMALLER post
            diff_sig_grow=n_of(s_, lambda d: d.diff_ci_hi < 0),   # sig LARGER post
        )

    # ================= write outputs =================
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    out_csv = out.merge(diffs, on=key, how="left")
    # diff columns are per-(disc,h,ref); keep them only on the 'post' rows to avoid dupes
    for c in ("diff_daily_d", "diff_ci_lo", "diff_ci_hi"):
        out_csv.loc[out_csv.stratum != "post", c] = np.nan
    out_csv.to_csv("results/tables/postcutoff_effects.csv", index=False)

    def sig(dm, p):
        return "**" if (dm < 0 and p < 0.05) else ""

    def e4(x):
        return f"{x * 1e4:+.2f}"

    md = ["# ROW 8 — Post-training-cutoff effect sizes + 95% CIs for the 8-K (C6) "
          "residual\n",
          "## RESTATED vs BEFORE\n",
          "| | BEFORE (llm_contamination.md sec.3) | RESTATED (this table) |",
          "|---|---|---|",
          "| post-cutoff evidence | significance COUNTS only (5/6 Holm fulltext-vs-HAR, "
          "3/6 beyond-identity); point rel% with NO uncertainty | per-cell EFFECT SIZES "
          "with day-block moving-bootstrap 95% CIs of the mean daily loss differential, "
          "pre and post side-by-side, both references |",
          "| pre-vs-post comparison | eyeballed (\"post > pre in 6/6 cells\") | "
          "post-minus-pre difference of the mean daily differential with a bootstrap "
          "95% CI per cell |",
          f"| references | same | same two: (a) single recalibrated HAR; (b) identity-"
          f"augmented [1, log fHAR, log f_datefirm] (same-model date+ticker control) |",
          "",
          "**Protocol (no refit, no look-ahead):** combiner weights are the ORIGINAL "
          "full-validation fit, frozen; ONLY the test evaluation is stratified at "
          f"{CUTOFF.date()} (approx. Qwen3 training-data boundary; caveat in "
          "llm_contamination.md applies). QLIKE in vol units; day-clustered DM "
          "(daily-mean loss differentials over effective_trading_day, HAC lag = h-1 "
          "days). Bootstrap: committed `clustered_dm.mbb_ci_daily` — moving-block "
          "bootstrap over DAYS of the daily-mean loss differential d = QLIKE(U) - "
          f"QLIKE(R), block length h days, B={BOOT_B}, seed {BOOT_SEED} (committed "
          "defaults reused; the brief's 1000-draw fallback applies only when no "
          "committed bootstrap exists). Negative d = text arm better; the rel%-scaled "
          "CI divides by the SUBSAMPLE daily-mean reference QLIKE (fixed denominator), "
          "so it is equal-weighted per day: it brackets the DAY-weighted point "
          "(rel%_daily), NOT the observation-weighted rel% column, and the two can "
          "differ materially where within-day observation counts covary with the loss "
          "differential (e.g. long_form h5 pre: obs +1.25% vs daily +3.08%). No "
          "subsampling anywhere: all C6 test observations are used.\n",
          "## PRE-DECLARED Holm family\n",
          "One family, declared before any results: the **24 stratified cells** "
          "(2 disclosures x 3 horizons x 2 references x 2 strata), Holm within it — "
          "identical to the committed llm_contamination cutoff block (sanity gate C "
          "asserts the Holm column reproduces it exactly). Full-sample rows are "
          "context only; their Holm values are inherited unchanged from the committed "
          "llm_contamination families (18-cell variant block / 6-cell joint block) — "
          "no new inference is run on them. `**` = clustered DM<0 and Holm p<.05.\n"]

    md.append("## 1. Full-sample context rows (sanity anchor; CIs new)\n")
    md.append("| disc | h | ref | n_obs | n_days | rel% | d_daily x1e-4 [95% CI] | "
              "DM(clu) | p raw | p Holm* |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in out[out.stratum == "full"].iterrows():
        md.append(f"| {r.disc} | {r.h} | {r.ref} | {int(r.n_test)} | {int(r.n_days)} | "
                  f"{r.rel_pct:+.2f}%{sig(r.dm_clu, r.p_holm)} | "
                  f"{e4(r.mean_daily_d)} [{e4(r.ci_lo)}, {e4(r.ci_hi)}] | "
                  f"{r.dm_clu:+.2f} | {r.p_raw:.2e} | {r.p_holm:.4f} |")
    md.append("\n*Holm inherited from the committed llm_contamination families "
              "(context only).\n")

    md.append("## 2. Pre/post stratified effect sizes (24 cells, the pre-declared "
              "family)\n")
    md.append("| disc | h | ref | stratum | n_obs | n_days | rel% | "
              "d_daily x1e-4 [95% CI] | rel%_daily [95% CI] | DM(clu) | p raw | "
              "p Holm |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in out[out.stratum != "full"].iterrows():
        md.append(f"| {r.disc} | {r.h} | {r.ref} | {r.stratum} | {int(r.n_test)} | "
                  f"{int(r.n_days)} | {r.rel_pct:+.2f}%{sig(r.dm_clu, r.p_holm)} | "
                  f"{e4(r.mean_daily_d)} [{e4(r.ci_lo)}, {e4(r.ci_hi)}] | "
                  f"{r.rel_daily_pct:+.2f}% [{r.rel_ci_lo:+.2f}, {r.rel_ci_hi:+.2f}] | "
                  f"{r.dm_clu:+.2f} | {r.p_raw:.2e} | {r.p_holm:.4f} |")

    md.append("\n## 3. Pre vs post side-by-side (per cell x reference)\n")
    md.append("`diff` = post mean daily d minus pre mean daily d (negative = LARGER "
              "post-cutoff improvement), bootstrap 95% CI from independent day-block "
              "draws of each stratum (same block scheme, one rng seeded "
              f"{BOOT_SEED} per cell). Each rel% cell shows the obs-weighted point "
              "first, then in parentheses the DAY-weighted point with the CI that "
              "brackets it (the CI is day-weighted, so it brackets the daily point, "
              "not necessarily the obs-weighted one).\n")
    md.append("| disc | h | ref | pre rel% (daily [95% CI]) | "
              "post rel% (daily [95% CI]) | diff x1e-4 [95% CI] | post verdict |")
    md.append("|---|---|---|---|---|---|---|")
    for _, r in sb.iterrows():
        vsig = ("Holm-sig" if (r.dm_clu_post < 0 and r.p_holm_post < 0.05)
                else "raw-sig" if (r.dm_clu_post < 0 and r.p_raw_post < 0.05)
                else "n.s.")
        vdir = "larger" if r.rel_pct_post > r.rel_pct_pre else "smaller"
        md.append(f"| {r.disc} | {r.h} | {r.ref} | "
                  f"{r.rel_pct_pre:+.2f}% ({r.rel_daily_pct_pre:+.2f}% "
                  f"[{r.rel_ci_lo_pre:+.2f}, {r.rel_ci_hi_pre:+.2f}]) | "
                  f"{r.rel_pct_post:+.2f}% ({r.rel_daily_pct_post:+.2f}% "
                  f"[{r.rel_ci_lo_post:+.2f}, {r.rel_ci_hi_post:+.2f}]) | "
                  f"{e4(r.diff_daily_d)} [{e4(r.diff_ci_lo)}, {e4(r.diff_ci_hi)}] | "
                  f"{vsig}; point est {vdir} than pre |")

    ft, bi = stats_["fulltext_vs_har"], stats_["beyond_identity"]
    bi_post = P[P.ref == "beyond_identity"]
    bi_fail = ", ".join(f"{r.disc} h{r.h}" for _, r in bi_post.iterrows()
                        if not (r.dm_clu < 0 and r.p_holm < 0.05))
    n_bi_pos = int((bi_post.rel_pct > 0).sum())
    n_shrink_sig = ft["diff_sig_shrink"] + bi["diff_sig_shrink"]
    n_grow_sig = ft["diff_sig_grow"] + bi["diff_sig_grow"]
    grow_cells = ", ".join(f"{r.disc} h{r.h} ({r.ref})" for _, r in sb.iterrows()
                           if r.diff_ci_hi < 0)
    md.append(
        "\n## Bottom line (honest headline)\n"
        f"- **The post-cutoff residual effect size HOLDS UP — no cell shows "
        f"significant shrinkage.** The post-minus-pre difference CI indicates a "
        f"significantly SMALLER post-cutoff effect in {n_shrink_sig}/12 cells; the "
        f"{n_grow_sig} cell(s) where the difference IS significant go the other way "
        f"(larger post-cutoff effect: {grow_cells or 'none'}).\n"
        f"- **vs single recalibrated HAR (post-cutoff):** Holm-significant in "
        f"{ft['post_holm']}/6 cells (raw {ft['post_raw']}/6); the 95% day-block CI of "
        f"the mean daily differential excludes zero in {ft['post_ci_excl0']}/6.\n"
        f"- **vs identity-augmented reference (post-cutoff):** Holm-significant in "
        f"{bi['post_holm']}/6 (raw {bi['post_raw']}/6); CI excludes zero in "
        f"{bi['post_ci_excl0']}/6 — the beyond-identity residual is estimated with "
        f"positive point effect in {n_bi_pos}/6 post-cutoff cells but with wider "
        f"intervals on the {int(P.n_days.min())}-{int(P.n_days.max())}-day post "
        f"subsamples; Holm-failing post-cutoff cells: {bi_fail or 'none'}.\n"
        f"- **Weighting nuance (disclosed, not hidden):** obs-weighted rel% is larger "
        f"post-cutoff in {ft['post_gt_pre'] + bi['post_gt_pre']}/12 cells, but the "
        f"day-equal-weighted mean daily differential is larger (more negative) post-"
        f"cutoff in {ft['post_dailyd_larger'] + bi['post_dailyd_larger']}/12 — the "
        f"divergent cells (long_form h5, both references) have higher post-cutoff "
        f"volatility inflating the obs-weighted rel% denominator mix. Under NEITHER "
        f"weighting does the effect shrink significantly anywhere.\n"
        f"- **Honest claim:** \"no evidence the residual weakens after the training "
        f"cutoff; where the pre/post difference is individually significant "
        f"({n_grow_sig}/12 cells) it strengthens\" — NOT \"the residual grows\" "
        f"(most differences are individually insignificant).\n"
        f"- Post-cutoff subsamples are {int(P.n_days.min())}-{int(P.n_days.max())} "
        f"days ({int(P.n_test.min())}-{int(P.n_test.max())} obs) — CIs are "
        f"correspondingly wider than full-sample; effect sizes remain economically "
        f"small (post-cutoff rel% max "
        f"{P.rel_pct.max():+.2f}%).\n")

    md.append("## SANITY (gate status: "
              f"{'PASS' if all_pass else 'FAIL'})\n")
    md.append(f"Anchor table: `{ANCHOR}` (committed). All comparisons machine-"
              "precision (tolerance 1e-12; CSV float64 round-trip is lossless).\n")
    for gname, g in gates.items():
        diffs_txt = ", ".join(f"max|d {k[13:]}|={v:.2e}"
                              for k, v in g.items()
                              if k.startswith("max_abs_diff_"))
        md.append(f"- Gate {gname}: {g['n_rows_compared']} rows, {diffs_txt}: "
                  f"**{'PASS' if g['pass'] else 'FAIL'}**")
    md.append("\nGate A is the brief-mandated gate (full-sample fulltext rows == "
              "llm_contamination.csv). Gates B/C additionally pin the identity-"
              "reference rows and the entire stratified block incl. Holm.")

    with open("results/tables/postcutoff_effects.md", "w") as fh:
        fh.write("\n".join(md))

    print(json.dumps({"gates_pass": all_pass, **stats_}, indent=2))
    print("wrote results/tables/postcutoff_effects.{csv,md}")


if __name__ == "__main__":
    main()
