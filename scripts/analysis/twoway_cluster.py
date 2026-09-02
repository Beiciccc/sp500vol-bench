"""P1-b — TWO-WAY (FIRM x DAY) CLUSTER ROBUSTNESS for the three headline DM grids.

Reviewer objection being remediated (REVIEW_ROUND2_GAPS.md P1-4 / REVIEW_BLINDSPOTS
P0#1): all committed inference is day-clustered only, yet the firm-identity control
proves strong WITHIN-FIRM dependence of the loss differentials. This script reports
Cameron-Gelbach-Miller two-way (ticker x effective_trading_day) statistics SIDE BY
SIDE with the committed day-clustered statistics, on byte-identical forecasts:

  (a) the 69-cell M1 grid, single recalibrated-HAR reference (basis of
      results/tables/m1_clustered.csv: seed2026, fc.log_combo val-fit/test-apply);
  (b) the firm-identity-reference grid (basis of
      results/tables/firm_identity_control.csv, produced by
      maximal_reference_firm_control.py: 5-price-model inner-join sample,
      reference = exp OLS[1, log fHAR, log firm_mean_val_RV]);
  (c) the pairwise-vs-A2 headline on squared error (basis of
      results/tables/dm_pairwise_clustered.csv: seed-ensembled test forecasts,
      inner-joined challenger panel), the "0/180 challengers beat A2" claim.

Two-way machinery: scripts/analysis/twoway_dm.py (formula documented there —
V_2way = V_firm + V_day(HAC, lag=h-1) - V_firm∩day; non-PSD guard max(V, eps);
df = min(#firms, #days) - 1).

HARD SANITY: the day-clustered column of every panel is recomputed here through
clustered_dm.dm_test_clustered and asserted equal to the committed tables
(m1_clustered.csv / firm_identity_control.csv / dm_pairwise_clustered.csv).

Discipline: combiner weights val-only, frozen to test (untouched — forecasts are
recomputed through the identical fc/mrf code paths); raw p AND Holm reported for
both clusterings; Holm within the same families as the committed tables
(69-cell grid for a/b; within (disclosure, horizon) challenger set for c).

Outputs (NEW files only):
  results/tables/twoway_cluster.csv
  results/tables/twoway_cluster.md
Run from repo root:  .venv/bin/python scripts/analysis/twoway_cluster.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/analysis")
sys.path.insert(0, "src")
import forecast_combination as fc  # noqa: E402
import m1_clustered as mc  # noqa: E402
import maximal_reference_firm_control as mrf  # noqa: E402
from clustered_dm import dm_test_clustered  # noqa: E402
from twoway_dm import dm_test_2way  # noqa: E402

KEY = fc.KEY
SORT = fc.SORT
HORIZONS = fc.HORIZONS
HAR = "A2_har_rv"
TOL = dict(rtol=1e-9, atol=1e-12)


def tw_fields(tw):
    """Common two-way output fields for one cell."""
    return {
        "dm_2way": tw.stat, "p_2way": tw.p,
        "n_firms": tw.n_firms, "n_days": tw.n_days, "df_2way": tw.df,
        "se_infl_2way_vs_day": float(np.sqrt(tw.V_2way / tw.V_day)) if tw.V_day > 0 else np.nan,
        "share_Vfirm": tw.V_firm / tw.V_2way,
        "share_Vday": tw.V_day / tw.V_2way,
        "share_Vcell": tw.V_cell / tw.V_2way,
        "guard_hit": tw.guard_hit,
    }


# ---------------------------------------------------------------------------
# Panel (a) — 69-cell M1 grid, single recalibrated-HAR reference (m1_clustered basis)
# ---------------------------------------------------------------------------
def panel_a():
    ref = pd.read_csv("results/tables/m1_clustered.csv")
    rows = []
    for disc, models in fc.SETS.items():
        har = fc.load(HAR, disc)[["split"] + KEY + [
            "prediction_realised_vol", "label_realised_vol", "filing_time_utc",
            "effective_trading_day"]].rename(columns={"prediction_realised_vol": "fhar"})
        for m in models:
            txt = fc.load(m, disc)[KEY + ["prediction_realised_vol"]].rename(
                columns={"prediction_realised_vol": "ftext"})
            d = har.merge(txt, on=KEY)
            for h in HORIZONS:
                dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(SORT, kind="mergesort")
                dt = d[(d.horizon_days == h) & (d.split == "test")].sort_values(SORT, kind="mergesort")
                if len(dv) < 100 or len(dt) < 30:
                    continue
                yv, fhv, ftv = dv.label_realised_vol.to_numpy(), dv.fhar.to_numpy(), dv.ftext.to_numpy()
                yt, fhr, ftt = dt.label_realised_vol.to_numpy(), dt.fhar.to_numpy(), dt.ftext.to_numpy()
                days = mc.day_key(dt)
                firms = dt.ticker.to_numpy()

                fR, fU, _ = fc.log_combo(yv, fhv, ftv, fhr, ftt)
                lR, lU = fc.qlike(yt, fR), fc.qlike(yt, fU)
                dm_day, p_day, _nd = dm_test_clustered(lU, lR, days, h)
                tw = dm_test_2way(lU - lR, firms, days, h)

                # placebo gate under the SAME two-way statistic (identical seeds)
                p2 = []
                for s in fc.PLACEBO_SEEDS:
                    rng = np.random.default_rng(s)
                    pR, pU, _ = fc.log_combo(yv, fhv, rng.permutation(ftv), fhr, rng.permutation(ftt))
                    p2.append(dm_test_2way(fc.qlike(yt, pU) - fc.qlike(yt, pR), firms, days, h).stat)

                row = {"panel": "a_m1_grid", "disc": disc, "model": m, "h": h,
                       "n_obs": len(dt), "dm_day": dm_day, "p_day": p_day,
                       "placebo_dm_2way": float(np.mean(p2))}
                row.update(tw_fields(tw))
                rows.append(row)

    df = pd.DataFrame(rows)
    mrg = df.merge(ref[["disc", "model", "h", "dm_q_clust", "p_q_clust", "dmq_holm_clust",
                        "placebo_dm_clust", "genuine_clust"]], on=["disc", "model", "h"])
    if len(mrg) != len(ref):
        raise AssertionError(f"panel (a) cell mismatch: {len(mrg)} vs {len(ref)}")
    if not np.allclose(mrg.dm_day, mrg.dm_q_clust, **TOL):
        raise AssertionError("SANITY FAIL (a): day-clustered DM != m1_clustered.dm_q_clust")
    if not np.allclose(mrg.p_day, mrg.p_q_clust, **TOL):
        raise AssertionError("SANITY FAIL (a): day-clustered p != m1_clustered.p_q_clust")
    mrg["holm_day"] = fc.holm(mrg.p_day.fillna(1.0).values)
    if not np.allclose(mrg.holm_day, mrg.dmq_holm_clust, **TOL):
        raise AssertionError("SANITY FAIL (a): Holm(day) != m1_clustered.dmq_holm_clust")
    mrg["holm_2way"] = fc.holm(mrg.p_2way.fillna(1.0).values)
    mrg["genuine_day"] = mrg.genuine_clust.astype(bool)
    mrg["genuine_2way"] = ((mrg.dm_2way < 0) & (mrg.holm_2way < 0.05)
                           & (mrg.placebo_dm_2way.abs() < 2.0))
    # NOTE: the non-significant label is "ns" (NOT "null" — pandas.read_csv parses
    # the literal string "null" as NaN, silently corrupting the committed CSV).
    mrg["verdict_day"] = np.where(mrg.genuine_day, "genuine", "ns")
    mrg["verdict_2way"] = np.where(mrg.genuine_2way, "genuine", "ns")
    mrg["flip"] = mrg.verdict_day != mrg.verdict_2way
    return mrg


# ---------------------------------------------------------------------------
# Panel (b) — firm-identity-reference grid (firm_identity_control basis)
# ---------------------------------------------------------------------------
def panel_b():
    ref = pd.read_csv("results/tables/firm_identity_control.csv")
    rows = []
    for disc, models in fc.SETS.items():
        panel = mrf.build_price_panel(disc)
        fmap, gmean, _fcov, _ocov = mrf.firm_mean_val(panel)
        panel["firm_mean_val"] = panel.ticker.map(fmap).fillna(gmean).astype(float)
        for m in models:
            txt = fc.load(m, disc)[KEY + ["prediction_realised_vol"]].rename(
                columns={"prediction_realised_vol": "ftext"})
            d = panel.merge(txt, on=KEY, how="inner")
            for h in HORIZONS:
                dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(SORT, kind="mergesort")
                dt = d[(d.horizon_days == h) & (d.split == "test")].sort_values(SORT, kind="mergesort")
                if len(dv) < 100 or len(dt) < 30:
                    continue
                yv = dv.label_realised_vol.to_numpy()
                yt = dt.label_realised_vol.to_numpy()
                fhv, fhr = dv.A2_har_rv.to_numpy(), dt.A2_har_rv.to_numpy()
                ftv, ftt = dv.ftext.to_numpy(), dt.ftext.to_numpy()
                fmv, fmt = dv.firm_mean_val.to_numpy(), dt.firm_mean_val.to_numpy()
                days_t = (dt.effective_trading_day.fillna(dt.filing_time_utc)).to_numpy()
                firms = dt.ticker.to_numpy()

                fRf, _ = mrf.log_ols_frozen(yv, [fhv, fmv], [fhr, fmt])
                fUf, _ = mrf.log_ols_frozen(yv, [fhv, fmv, ftv], [fhr, fmt, ftt])
                lRf, lUf = fc.qlike(yt, fRf), fc.qlike(yt, fUf)
                dm_day, p_day, _nd = dm_test_clustered(lUf, lRf, days_t, h)
                tw = dm_test_2way(lUf - lRf, firms, days_t, h)

                row = {"panel": "b_firm_ref", "disc": disc, "model": m, "h": h,
                       "n_obs": len(dt), "dm_day": dm_day, "p_day": p_day}
                row.update(tw_fields(tw))
                rows.append(row)

    df = pd.DataFrame(rows)
    mrg = df.merge(ref[["disc", "model", "h", "dm_q_clustered", "p_q_clustered",
                        "holm_p", "text_survives_firm_holm"]], on=["disc", "model", "h"])
    if len(mrg) != len(ref):
        raise AssertionError(f"panel (b) cell mismatch: {len(mrg)} vs {len(ref)}")
    if not np.allclose(mrg.dm_day, mrg.dm_q_clustered, **TOL):
        raise AssertionError("SANITY FAIL (b): day DM != firm_identity_control.dm_q_clustered")
    if not np.allclose(mrg.p_day, mrg.p_q_clustered, **TOL):
        raise AssertionError("SANITY FAIL (b): day p != firm_identity_control.p_q_clustered")
    mrg["holm_day"] = fc.holm(mrg.p_day.fillna(1.0).values)
    if not np.allclose(mrg.holm_day, mrg.holm_p, **TOL):
        raise AssertionError("SANITY FAIL (b): Holm(day) != firm_identity_control.holm_p")
    mrg["holm_2way"] = fc.holm(mrg.p_2way.fillna(1.0).values)

    def verdict(dm, hp):
        # "ns" not "null": pandas.read_csv would parse "null" as NaN on re-read.
        return np.where(hp < 0.05, np.where(dm < 0, "text adds", "text HURTS"), "ns")

    mrg["verdict_day"] = verdict(mrg.dm_day, mrg.holm_day)
    mrg["verdict_2way"] = verdict(mrg.dm_2way, mrg.holm_2way)
    # committed survivor flag must equal the recomputed day verdict
    if not (mrg.text_survives_firm_holm.astype(bool)
            == (mrg.verdict_day == "text adds")).all():
        raise AssertionError("SANITY FAIL (b): day survivor set != committed text_survives_firm_holm")
    mrg["flip"] = mrg.verdict_day != mrg.verdict_2way
    return mrg


# ---------------------------------------------------------------------------
# Panel (c) — pairwise vs A2 on squared error (dm_pairwise_clustered basis)
# ---------------------------------------------------------------------------
def panel_c():
    ref = pd.read_csv("results/tables/dm_pairwise_clustered.csv")
    rows = []
    for disc in mc.DISCLOSURES:
        merged, present = mc.build_joined(disc)
        if merged is None or HAR not in present:
            continue
        for h in HORIZONS:
            g = merged[merged.horizon_days == h].sort_values(SORT, kind="mergesort")
            if len(g) < 30:
                continue
            y = g.label_realised_vol.to_numpy()
            days = mc.day_key(g)
            firms = g.ticker.to_numpy()
            se_har = fc.se(y, g[f"pred__{HAR}"].to_numpy())
            for ch in present:
                if ch == HAR:
                    continue
                se_ch = fc.se(y, g[f"pred__{ch}"].to_numpy())
                dm_day, p_day, _nd = dm_test_clustered(se_ch, se_har, days, h)
                tw = dm_test_2way(se_ch - se_har, firms, days, h)
                row = {"panel": "c_pairwise_vsA2", "disc": disc, "model": ch, "h": h,
                       "n_obs": len(g), "dm_day": dm_day, "p_day": p_day}
                row.update(tw_fields(tw))
                rows.append(row)

    df = pd.DataFrame(rows)
    r = ref.rename(columns={"disclosure": "disc", "horizon": "h", "challenger": "model"})
    mrg = df.merge(r[["disc", "h", "model", "dm_clust", "p_clust", "p_holm_clust"]],
                   on=["disc", "h", "model"])
    if len(mrg) != len(ref):
        raise AssertionError(f"panel (c) cell mismatch: {len(mrg)} vs {len(ref)}")
    if not np.allclose(mrg.dm_day, mrg.dm_clust, **TOL):
        raise AssertionError("SANITY FAIL (c): day DM != dm_pairwise_clustered.dm_clust")
    if not np.allclose(mrg.p_day, mrg.p_clust, **TOL):
        raise AssertionError("SANITY FAIL (c): day p != dm_pairwise_clustered.p_clust")
    # Holm within each (disclosure, horizon) challenger group — as committed
    mrg["holm_day"] = mrg.groupby(["disc", "h"])["p_day"].transform(
        lambda s: fc.holm(s.fillna(1.0).to_numpy()))
    if not np.allclose(mrg.holm_day, mrg.p_holm_clust, **TOL):
        raise AssertionError("SANITY FAIL (c): Holm(day) != dm_pairwise_clustered.p_holm_clust")
    mrg["holm_2way"] = mrg.groupby(["disc", "h"])["p_2way"].transform(
        lambda s: fc.holm(s.fillna(1.0).to_numpy()))

    def verdict(dm, hp):
        return np.where(hp < 0.05, np.where(dm < 0, "BETTER", "sig worse"), "ns")

    mrg["verdict_day"] = verdict(mrg.dm_day, mrg.holm_day)
    mrg["verdict_2way"] = verdict(mrg.dm_2way, mrg.holm_2way)
    mrg["flip"] = mrg.verdict_day != mrg.verdict_2way
    return mrg


# ---------------------------------------------------------------------------
def main():
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    a = panel_a()
    print("panel (a) sanity vs m1_clustered.csv: PASS")
    b = panel_b()
    print("panel (b) sanity vs firm_identity_control.csv: PASS")
    c = panel_c()
    print("panel (c) sanity vs dm_pairwise_clustered.csv: PASS")

    cols = ["panel", "disc", "model", "h", "n_obs", "n_firms", "n_days", "df_2way",
            "dm_day", "p_day", "holm_day", "verdict_day",
            "dm_2way", "p_2way", "holm_2way", "verdict_2way", "flip",
            "se_infl_2way_vs_day", "share_Vfirm", "share_Vday", "share_Vcell",
            "guard_hit", "placebo_dm_2way"]
    full = pd.concat([a.reindex(columns=cols), b.reindex(columns=cols),
                      c.reindex(columns=cols)], ignore_index=True)
    full.to_csv("results/tables/twoway_cluster.csv", index=False)

    # headline counts
    a_day, a_2w = int(a.genuine_day.sum()), int(a.genuine_2way.sum())
    b_day = int((b.verdict_day == "text adds").sum())
    b_2w = int((b.verdict_2way == "text adds").sum())
    b_hurt_day = int((b.verdict_day == "text HURTS").sum())
    b_hurt_2w = int((b.verdict_2way == "text HURTS").sum())
    c_day = int((c.verdict_day == "BETTER").sum())
    c_2w = int((c.verdict_2way == "BETTER").sum())
    c_w_day = int((c.verdict_day == "sig worse").sum())
    c_w_2w = int((c.verdict_2way == "sig worse").sum())
    c_better_raw_2w = int(((c.dm_2way < 0) & (c.p_2way < 0.05)).sum())
    n_guard = int(full.guard_hit.sum())
    n_flip = int(full.flip.sum())

    def med(s):
        return float(np.nanmedian(s))

    md = ["# P1-b — TWO-WAY (FIRM x DAY) CLUSTER ROBUSTNESS of the DM inference\n",
          "## RESTATED vs BEFORE\n",
          "| panel (identical forecasts; only the variance estimator changes) | "
          "BEFORE (day-clustered, committed) | RESTATED (two-way firm x day, CGM) | flips |",
          "|---|---|---|---|",
          f"| (a) M1 69-cell grid — genuine text-increment cells (DM<0, Holm<.05, placebo gate) | "
          f"**{a_day}/69** (m1_clustered.csv) | **{a_2w}/69** | {int(a.flip.sum())} |",
          f"| (b) firm-identity-reference grid — text survives (Holm<.05) | **{b_day}/69** "
          f"(firm_identity_control.csv) | **{b_2w}/69** | {int(b.flip.sum())} |",
          f"| (b) — text HURTS (Holm<.05) | {b_hurt_day}/69 | {b_hurt_2w}/69 | — |",
          f"| (c) pairwise vs A2 (SE) — challengers significantly BETTER (Holm<.05) | "
          f"**{c_day}/180** (dm_pairwise_clustered.csv) | **{c_2w}/180** "
          f"(raw p<.05: {c_better_raw_2w}/180) | {int(c.flip.sum())} |",
          f"| (c) — challengers significantly WORSE (Holm<.05) | {c_w_day}/180 | {c_w_2w}/180 | — |",
          "",
          "**Method.** Cameron-Gelbach-Miller two-way clustered variance on the mean "
          "loss differential (equal weight per day, matching the day-clustered primary): "
          "`V_2way = V_firm + V_day - V_firm∩day`, where `V_C = Σ_c (Σ_{i∈c} w_i(d_i - d̄))²` "
          "with `w_i = 1/(T·n_day(i))`, and the day component is the Newey-West HAC "
          "(lag = h-1 trading days) long-run variance of the daily-mean differential "
          "series divided by T — at lag 0 this equals the CGM day component exactly, so "
          "the serial-correlation treatment is identical to the committed day-clustered DM. "
          "Non-PSD guard: `V_2way <- max(V_2way, 1e-30)` (flagged); reference distribution "
          "t(min(#firms, #days) - 1). The firm∩day intersection is subtracted at lag 0 only "
          "(Thompson's lagged own-firm overlap terms omitted), which can only WIDEN the SEs — "
          "conservative for every significance claim. No HLN correction on the two-way stat "
          "(immaterial at n_days≈800; the day column keeps it, as committed). Full details: "
          "`scripts/analysis/twoway_dm.py`.\n",
          "**SANITY (hard assertions, all PASS):** the recomputed day-clustered columns "
          "reproduce the committed tables exactly — (a) `m1_clustered.csv` "
          "(dm_q_clust, p_q_clust, dmq_holm_clust), (b) `firm_identity_control.csv` "
          "(dm_q_clustered, p_q_clustered, holm_p, survivor set), (c) "
          "`dm_pairwise_clustered.csv` (dm_clust, p_clust, p_holm_clust).\n",
          "**Variance anatomy (medians):** SE inflation two-way vs day-only "
          f"sqrt(V_2way/V_day): (a) {med(a.se_infl_2way_vs_day):.3f}, "
          f"(b) {med(b.se_infl_2way_vs_day):.3f}, (c) {med(c.se_infl_2way_vs_day):.3f}. "
          f"Median variance shares (a): firm {med(a.share_Vfirm):.2f}, day {med(a.share_Vday):.2f}, "
          f"intersection (subtracted) {med(a.share_Vcell):.2f}; "
          f"(c): firm {med(c.share_Vfirm):.2f}, day {med(c.share_Vday):.2f}, "
          f"intersection {med(c.share_Vcell):.2f}. "
          f"Non-PSD guard hits: {n_guard}/{len(full)} cells.\n"]

    # ---- flips detail ----
    md.append("## Verdict flips (day-clustered -> two-way)\n")
    flips = full[full.flip]
    if len(flips):
        md.append("| panel | disc | model | h | dm_day | Holm(day) | dm_2way | p_2way | "
                  "Holm(2way) | verdict day -> 2way |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in flips.iterrows():
            md.append(f"| {r.panel} | {r.disc} | {r.model} | {r.h} | {r.dm_day:+.2f} | "
                      f"{r.holm_day:.4f} | {r.dm_2way:+.2f} | {r.p_2way:.4f} | "
                      f"{r.holm_2way:.4f} | {r.verdict_day} -> {r.verdict_2way} |")
    else:
        md.append("None — every committed day-clustered verdict is unchanged under "
                  "two-way (firm x day) clustering.")
    md.append("")

    # ---- panel (a) grid ----
    md.append("\n## (a) M1 69-cell grid — day-clustered vs two-way (seed2026 basis of m1_clustered)\n"
              "| disc | model | h | n_firms | n_days | dm_day | Holm(day) | dm_2way | p_2way | "
              "Holm(2way) | placebo 2way | SEx | genuine day->2way |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in a.sort_values(["disc", "model", "h"]).iterrows():
        md.append(f"| {r.disc} | {r.model} | {r.h} | {r.n_firms} | {r.n_days} | {r.dm_day:+.2f} | "
                  f"{r.holm_day:.3f} | {r.dm_2way:+.2f} | {r.p_2way:.4f} | {r.holm_2way:.3f} | "
                  f"{r.placebo_dm_2way:+.2f} | {r.se_infl_2way_vs_day:.2f} | "
                  f"{'Y' if r.genuine_day else 'n'}->{'Y' if r.genuine_2way else 'n'} |")

    # ---- panel (b) grid ----
    md.append("\n## (b) Firm-identity-reference grid — day-clustered vs two-way\n"
              "| disc | model | h | n_firms | n_days | dm_day | Holm(day) | dm_2way | p_2way | "
              "Holm(2way) | SEx | verdict day -> 2way |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in b.sort_values(["disc", "model", "h"]).iterrows():
        md.append(f"| {r.disc} | {r.model} | {r.h} | {r.n_firms} | {r.n_days} | {r.dm_day:+.2f} | "
                  f"{r.holm_day:.3f} | {r.dm_2way:+.2f} | {r.p_2way:.4f} | {r.holm_2way:.3f} | "
                  f"{r.se_infl_2way_vs_day:.2f} | {r.verdict_day} -> {r.verdict_2way} |")

    # ---- panel (c) ----
    md.append("\n## (c) Pairwise vs A2 on squared error — day-clustered vs two-way "
              "(seed-ensemble basis of dm_pairwise_clustered)\n"
              "| disc | h | challenger | n_firms | n_days | dm_day | Holm(day) | dm_2way | "
              "p_2way | Holm(2way) | SEx | verdict day -> 2way |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in c.sort_values(["disc", "h", "dm_2way"]).iterrows():
        md.append(f"| {r.disc} | {r.h} | {r.model} | {r.n_firms} | {r.n_days} | {r.dm_day:+.2f} | "
                  f"{r.holm_day:.4f} | {r.dm_2way:+.2f} | {r.p_2way:.4f} | {r.holm_2way:.4f} | "
                  f"{r.se_infl_2way_vs_day:.2f} | {r.verdict_day} -> {r.verdict_2way} |")

    # ---- bottom line ----
    md.append("\n## Bottom line\n")
    md.append(f"- Adding the firm clustering dimension changes **{n_flip}** verdict(s) across "
              f"the {len(full)} committed inference cells "
              f"(a: {int(a.flip.sum())}, b: {int(b.flip.sum())}, c: {int(c.flip.sum())}).")
    md.append(f"- (a) M1 grid: genuine cells {a_day}/69 (day) -> **{a_2w}/69** (two-way).")
    md.append(f"- (b) firm-identity control: survivors {b_day}/69 (day) -> **{b_2w}/69** (two-way); "
              f"text-HURTS {b_hurt_day} -> {b_hurt_2w}.")
    md.append(f"- (c) the '0/180 challengers beat A2' headline: **{c_2w}/180** significantly better "
              f"under two-way clustering ({c_better_raw_2w}/180 even at raw p<.05); "
              f"significantly worse {c_w_day} -> {c_w_2w}.")
    md.append("- Risk direction is as pre-registered in REVIEW_ROUND2_GAPS.md P1-4: two-way SEs are "
              "(weakly) wider, so any movement is TOWARD the null — the near-null headline cannot "
              "be an artifact of ignoring within-firm dependence"
              + ("." if n_flip == 0 else "; the flipped cells listed above must be quoted with the "
                 "two-way (weaker) verdict in the paper."))

    with open("results/tables/twoway_cluster.md", "w") as fh:
        fh.write("\n".join(md))

    print("=== P1-b two-way cluster robustness done ===")
    print(f"(a) M1 grid genuine: {a_day}/69 day -> {a_2w}/69 two-way (flips {int(a.flip.sum())})")
    print(f"(b) firm-ref survivors: {b_day}/69 day -> {b_2w}/69 two-way "
          f"(HURTS {b_hurt_day}->{b_hurt_2w}; flips {int(b.flip.sum())})")
    print(f"(c) pairwise better: {c_day}/180 day -> {c_2w}/180 two-way "
          f"(raw {c_better_raw_2w}; sig-worse {c_w_day}->{c_w_2w}; flips {int(c.flip.sum())})")
    print(f"median SE inflation: a {med(a.se_infl_2way_vs_day):.3f} "
          f"b {med(b.se_infl_2way_vs_day):.3f} c {med(c.se_infl_2way_vs_day):.3f}; "
          f"guard hits {n_guard}; total flips {n_flip}")
    print("wrote results/tables/twoway_cluster.{csv,md}")


if __name__ == "__main__":
    main()
