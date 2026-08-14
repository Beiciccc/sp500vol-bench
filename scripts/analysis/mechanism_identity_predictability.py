"""Pre-registered analysis C — mechanism test: the identity control's sign and size
are a function of how much of the entity the baseline already encodes.

Prereg: configs/prereg_mechanism_and_labels.md §C (tag prereg-cd-v1.0, dated
2026-07-15, committed BEFORE any statistic below was computed). Binding commitments
implemented here:

  y (per cell, FROZEN artifacts only — never recomputed from raw predictions):
      y = 100·[L(f_R) − L(f_Re)] / L(f_R)   (positive = identity control helps)
    * SEC   6 cells: results/tables/firm_identity_ensemble.csv,
            rel_impr_firmMeanOnly_vs_fR — verified UNIQUE within each
            (disclosure, horizon) across all models, so the 6 (disc, h) cells are
            well defined; anchored against the committed aggregate
            "53/69 (mean +0.52%)" at machine precision.
    * Yelp  2 cells: results/tables/yelp_cascade.csv row 4 (entity-mean control,
            chronological), full precision; anchored against the committed md
            (−10.58 / −9.48 at 2 dp). The generating protocol_results.json
            (tfidf primary) is no longer on disk; the cascade CSV is the frozen
            per-cell artifact from the same pipeline (yelp_cascade_table.py).
    * MAEC  8 cells: results/second_domain/maec/protocol_tfidf_primary.json,
            horizons[h][ref]["entity"]["delta_rel_pct"] (the STPEV control row,
            field names verified in maec_protocol.py run_ref_ladder), 4 horizons
            × {r_ar, r_har}; anchored against the committed
            results/tables/maec_audit.csv (row3 rel_pct + identity_share).

  x (per cell, the ONLY new computation): R² of regressing the reference's
    log-space VALIDATION predictions on entity dummies (one-way-ANOVA identity;
    exact OLS-on-dummies R²), same panel / horizon / reference as y.
    * SEC : f_R = exp OLS[1, log A2_HAR] fit on the 5-price-model inner-join
            panel's val rows (the committed firm-control code path,
            maximal_reference_firm_control.build_price_panel + fc.log_combo);
            regress the log val fit on ticker dummies.
    * Yelp: f_R = clip(exp OLS[1, log f_AR]) fit on the protocol's AR∩text val
            rows (yelp_protocol.log_ols_frozen); regress log f_R val on
            business dummies.
    * MAEC: labels are already v = log vol, so the stored val predictions of
            preds_r_{ar,har}_primary.parquet ARE the reference's log-space val
            predictions; regress on permno dummies.
    NO TEST LABELS are touched, asserted structurally: every parquet is read
    with a split=='val' predicate pushdown (test rows never materialised), the
    MAEC read omits the label column entirely, and _assert_val_only() guards
    every frame before any label/prediction use. Val labels enter ONLY the
    val-side recalibration fit (allowed by the prereg boundary: "C reads only frozen
    predictions and val labels").

  Statistics (pre-declared): (i) Spearman rho over the 16 points, prediction
  rho < 0; permutation p with 10,000 draws, seed 2026, BOTH global and
  within-panel cell-label permutation (double-reported; prose takes the
  conservative = larger p); one-sided in the registered direction, add-one
  convention p = (1 + #{rho_perm ≤ rho_obs}) / (N + 1). (ii) Median-split
  sign test: proportion of y ≤ 0 among cells with x above vs below the median
  of x, Fisher exact (one-sided 'greater' in the predicted direction;
  two-sided also reported). (iii) Disclosure of non-independence (3 panels).

  Verdict branches (verbatim from the prereg):
    ESTABLISHED : rho < 0 AND conservative permutation p < .05.
    FALSIFIED   : rho >= 0 OR both permutation p's > .10.
    INCONCLUSIVE: anything between (descriptive wording kept, no upgrade).

Single-shot discipline: refuses to run if any output exists (mirror of
maec_protocol §6.5; --force-rerun --reason '...' for logged bug-fixes only).

Outputs:
  results/tables/mechanism_identity.{csv,md}
  writing/paper/figures/mechanism_identity.pdf

Run from repo root:
    .venv/bin/python scripts/analysis/mechanism_identity_predictability.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"                     # thread cap BEFORE numpy import

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
T = REPO / "results/tables"
OUT_CSV = T / "mechanism_identity.csv"
OUT_MD = T / "mechanism_identity.md"
OUT_FIG = REPO / "writing/paper/figures/mechanism_identity.pdf"

EPS = 1e-8                                   # forecast_combination.py convention
SEC_KEY = ["ticker", "accession", "horizon_days"]
SEC_SORT = ["filing_time_utc", "ticker", "accession"]
SEC_PRICE = ["A2_har_rv", "A6_shar", "A3_garch", "A4_egarch", "A5_arima"]
SEC_HORIZONS = (5, 10, 20)
SEC_DISCS = ("long_form", "event_driven")
YELP_KEY = ["entity_id", "event_time", "horizon_months"]
YELP_CLIP = (1.0, 5.0)
MAEC_HORIZONS = (3, 7, 15, 30)
MAEC_REFS = ("r_ar", "r_har")
N_PERM, PERM_SEED = 10_000, 2026

SRC_SEC_Y = "results/tables/firm_identity_ensemble.csv"
SRC_YELP_Y = "results/tables/yelp_cascade.csv"
SRC_MAEC_Y = "results/second_domain/maec/protocol_tfidf_primary.json"
SRC_MAEC_ANCHOR = "results/tables/maec_audit.csv"


# ------------------------------------------------------------------ structural guards
def read_val(path, columns, split_col="split"):
    """Parquet read with a split=='val' predicate pushdown: TEST ROWS ARE NEVER
    MATERIALISED. Returns the val-only frame, asserted."""
    df = pd.read_parquet(path, columns=columns,
                         filters=[(split_col, "==", "val")])
    _assert_val_only(df, split_col)
    return df


def _assert_val_only(df, split_col="split"):
    assert split_col in df.columns and (df[split_col] == "val").all(), \
        "STRUCTURAL GUARD: non-val rows reached a label-bearing computation"


def anova_r2(values, groups):
    """R² of OLS on entity dummies (+intercept) == one-way ANOVA between-group
    share of variance. Exact identity, no dummy matrix built."""
    y = np.asarray(values, float)
    g = pd.Series(np.asarray(groups))
    ybar = y.mean()
    ss_tot = float(((y - ybar) ** 2).sum())
    if ss_tot <= 0:
        return float("nan"), int(g.nunique())
    gm = pd.Series(y).groupby(g.values)
    means, sizes = gm.mean().to_numpy(), gm.size().to_numpy(float)
    ss_between = float((sizes * (means - ybar) ** 2).sum())
    return ss_between / ss_tot, len(means)


# ----------------------------------------------------------------------- y assembly
def y_sec():
    """SEC 6 cells from the frozen firm_identity_ensemble.csv + anchors."""
    fi = pd.read_csv(REPO / SRC_SEC_Y)
    assert len(fi) == 69, "firm_identity_ensemble.csv must be the 69-cell grid"
    col = "rel_impr_firmMeanOnly_vs_fR"
    spread = fi.groupby(["disc", "h"])[col].apply(lambda s: s.max() - s.min())
    assert float(spread.max()) == 0.0, (
        "zero-text firm-mean-vs-f_R values are NOT unique within (disc, h) — "
        "the 6-cell reduction is ill-defined; STOP")
    mean69, beats = float(fi[col].mean()), int(fi.firm_beats_fR.sum())
    assert f"{mean69:+.2f}" == "+0.52" and beats == 53, (
        f"SEC anchor FAILED: mean {mean69:+.6f} / beats {beats} vs committed "
        f"'53/69 (mean +0.52%)'")
    cells = fi.groupby(["disc", "h"])[col].first()
    rows = [dict(panel="SEC", cell=f"{disc}/h{h}", y=float(cells[(disc, h)]),
                 source_y=SRC_SEC_Y)
            for disc in SEC_DISCS for h in SEC_HORIZONS]
    sanity = [f"SEC anchor PASS: 69-cell mean {mean69:+.6f}% prints as +0.52%, "
              f"firm_beats_fR = {beats}/69 (committed md: '53/69 (mean +0.52%)')",
              "SEC per-(disc,h) uniqueness PASS: max within-cell spread of "
              "rel_impr_firmMeanOnly_vs_fR across models = 0.0 (machine exact)"]
    return rows, sanity


def y_yelp():
    """Yelp 2 cells from the frozen yelp_cascade.csv row 4 + md anchor."""
    yc = pd.read_csv(REPO / SRC_YELP_Y)
    r4 = yc[yc.row == 4].set_index("h")
    assert len(r4) == 2 and (yc.tag == "REAL").all()
    vals = {int(h): float(r4.loc[h, "delta_rel_pct"]) for h in (1, 3)}
    assert f"{vals[1]:+.2f}" == "-10.58" and f"{vals[3]:+.2f}" == "-9.48", (
        f"Yelp anchor FAILED: {vals} vs committed md row 4 (−10.58 / −9.48)")
    rows = [dict(panel="Yelp", cell=f"chrono/h{h}m", y=vals[h],
                 source_y=SRC_YELP_Y + " (row 4)") for h in (1, 3)]
    sanity = [f"Yelp anchor PASS: row-4 entity-mean control = "
              f"{vals[1]:+.6f}% / {vals[3]:+.6f}% (md prints −10.58 / −9.48)"]
    return rows, sanity


def y_maec():
    """MAEC 8 cells from protocol_tfidf_primary.json 'entity' rows + anchors
    against the committed maec_audit.csv (row3 rel_pct 4 dp, identity_share 2 dp)."""
    prot = json.loads((REPO / SRC_MAEC_Y).read_text())
    assert prot["arm"] == "tfidf" and prot["alignment"] == "primary" \
        and prot["tag"] == "REAL"
    audit = pd.read_csv(REPO / SRC_MAEC_ANCHOR)
    audit = audit[(audit.arm == "tfidf") & (audit.alignment == "primary")]
    rows, sanity_bad = [], []
    for h in MAEC_HORIZONS:
        for ref in MAEC_REFS:
            r = prot["horizons"][str(h)][ref]
            y = float(r["entity"]["delta_rel_pct"])
            rows.append(dict(panel="MAEC", cell=f"primary/{ref}/h{h}", y=y,
                             source_y=SRC_MAEC_Y +
                             f" horizons.{h}.{ref}.entity.delta_rel_pct"))
            # anchors: the committed audit table is derived from this very json
            a3 = audit[(audit.horizon == h) & (audit.ref == ref)
                       & (audit.stage == "row3")].iloc[0]
            d3 = r["mse"]["R"] - r["mse"]["U"]
            d4 = r["mse"]["R"] - r["mse"]["Re"]
            share = 100.0 * d4 / d3 if d3 > 0 else float("nan")
            if not (abs(float(a3.rel_pct) - r["combined"]["delta_rel_pct"]) < 5e-5
                    and abs(float(a3.identity_share) - share) < 5e-3):
                sanity_bad.append(f"h{h}/{ref}")
    assert not sanity_bad, f"MAEC anchor FAILED for cells {sanity_bad}"
    sanity = ["MAEC anchor PASS: all 8 cells' row3 rel_pct (4 dp) and "
              "identity_share (2 dp) recomputed from the protocol json match "
              "the committed maec_audit.csv"]
    return rows, sanity


# ----------------------------------------------------------------------- x assembly
def x_sec():
    """x per (disc, h): R² of the log val fit of f_R = exp OLS[1, log A2] on
    ticker dummies, on the 5-price-model inner-join panel's val rows (the
    committed firm-control code path, val side)."""
    out = {}
    for disc in SEC_DISCS:
        base = read_val(
            REPO / f"results/runs/A2_har_rv_full_{disc}_seed2026/predictions.parquet",
            ["split"] + SEC_KEY + ["prediction_realised_vol", "label_realised_vol",
                                   "filing_time_utc"]
        ).rename(columns={"prediction_realised_vol": "A2_har_rv"})
        for m in SEC_PRICE[1:]:
            p = read_val(
                REPO / f"results/runs/{m}_full_{disc}_seed2026/predictions.parquet",
                ["split"] + SEC_KEY + ["prediction_realised_vol"]
            ).rename(columns={"prediction_realised_vol": m}).drop(columns=["split"])
            base = base.merge(p, on=SEC_KEY, how="inner")
        assert not base.duplicated(SEC_KEY).any()
        _assert_val_only(base)
        for h in SEC_HORIZONS:
            dv = base[base.horizon_days == h].sort_values(SEC_SORT, kind="mergesort")
            _assert_val_only(dv)
            ly = np.log(np.clip(dv.label_realised_vol.to_numpy(float), EPS, None))
            lh = np.log(np.clip(dv.A2_har_rv.to_numpy(float), EPS, None))
            bR, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(ly)), lh]),
                                     ly, rcond=None)   # fc.log_combo val fit
            lfR_val = bR[0] + bR[1] * lh               # log f_R val predictions
            r2, n_ent = anova_r2(lfR_val, dv.ticker)
            out[f"{disc}/h{h}"] = dict(
                x=r2, n_val=len(dv), n_entities=n_ent,
                source_x=f"results/runs/{{A2,SHAR,GARCH,EGARCH,ARIMA}}_full_{disc}"
                         f"_seed2026/predictions.parquet (val rows, 5-model join)")
    return out


def x_yelp():
    """x per horizon: R² of log f_R (= clip exp OLS[1, log f_AR], the protocol's
    log_ols_frozen applied to val) on business dummies, on the AR∩text val rows."""
    ar = read_val(REPO / "results/second_domain/preds/preds_ar_ridge.parquet",
                  YELP_KEY + ["split", "label", "prediction"])
    tx = read_val(REPO / "results/second_domain/preds/preds_tfidf_chrono.parquet",
                  YELP_KEY + ["split"])
    n_ar = len(ar)
    d = ar.merge(tx[YELP_KEY], on=YELP_KEY, how="inner", validate="1:1")
    assert len(d) == n_ar, "AR∩text val merge dropped rows — protocol row set changed"
    _assert_val_only(d)
    out = {}
    for h in (1, 3):
        dv = d[d.horizon_months == h].sort_values(["event_time", "entity_id"],
                                                  kind="mergesort")
        _assert_val_only(dv)
        ly = np.log(np.clip(dv.label.to_numpy(float), EPS, None))
        la = np.log(np.clip(dv.prediction.to_numpy(float), EPS, None))
        bR, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(ly)), la]),
                                 ly, rcond=None)       # yelp_protocol log_ols_frozen
        fR_val = np.clip(np.exp(bR[0] + bR[1] * la), *YELP_CLIP)
        r2, n_ent = anova_r2(np.log(fR_val), dv.entity_id)
        out[f"chrono/h{h}m"] = dict(
            x=r2, n_val=len(dv), n_entities=n_ent,
            source_x="results/second_domain/preds/preds_ar_ridge.parquet "
                     "(val rows, ∩ preds_tfidf_chrono)")
    return out


def x_maec():
    """x per (ref, h): R² of the STORED reference val predictions (already
    v-space = log vol) on permno dummies. The label column is NEVER read."""
    out = {}
    for ref in MAEC_REFS:
        fp = REPO / f"results/second_domain/maec/preds/preds_{ref}_primary.parquet"
        df = read_val(fp, ["permno", "horizon", "split", "prediction"])
        assert "label" not in df.columns   # structural: labels never loaded
        for h in MAEC_HORIZONS:
            dv = df[df.horizon == h]
            _assert_val_only(dv)
            r2, n_ent = anova_r2(dv.prediction.to_numpy(float), dv.permno)
            out[f"primary/{ref}/h{h}"] = dict(
                x=r2, n_val=len(dv), n_entities=n_ent,
                source_x=str(fp.relative_to(REPO)) + " (val rows, no label col)")
    return out


# ------------------------------------------------------------------------ statistics
def spearman_perm(x, y, panels):
    """Spearman rho + one-sided permutation p (predicted direction rho < 0),
    10,000 draws each for GLOBAL and WITHIN-PANEL cell-label permutation of y.
    Single RNG stream seed 2026: global draws first, then within-panel.
    Add-one convention."""
    rho, _ = stats.spearmanr(x, y)
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)          # rank(perm(y)) == perm(rank(y))
    rxc = rx - rx.mean()
    denom_x = float(np.sqrt((rxc ** 2).sum()))

    def rho_of(ry_perm):
        ryc = ry_perm - ry_perm.mean()
        return float((rxc * ryc).sum() / (denom_x * np.sqrt((ryc ** 2).sum())))

    rng = np.random.default_rng(PERM_SEED)
    n = len(y)
    glob = np.array([rho_of(ry[rng.permutation(n)]) for _ in range(N_PERM)])
    idx_by_panel = [np.flatnonzero(np.asarray(panels) == p)
                    for p in pd.unique(np.asarray(panels))]
    within = np.empty(N_PERM)
    for b in range(N_PERM):
        perm = np.arange(n)
        for idx in idx_by_panel:
            perm[idx] = idx[rng.permutation(len(idx))]
        within[b] = rho_of(ry[perm])
    p_g = (1 + int((glob <= rho).sum())) / (N_PERM + 1)
    p_w = (1 + int((within <= rho).sum())) / (N_PERM + 1)
    return float(rho), float(p_g), float(p_w)


def fisher_median_split(x, y):
    """Median split on x (16 distinct values -> 8 hi / 8 lo); Fisher exact on
    #{y<=0} hi vs lo. One-sided 'greater' = predicted direction (high-x cells
    are more often y<=0); two-sided also reported."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    assert len(np.unique(x)) == len(x), "ties in x — median split ill-defined; STOP"
    med = float(np.median(x))
    hi, lo = x > med, x < med
    assert hi.sum() == lo.sum() == len(x) // 2
    tab = [[int((y[hi] <= 0).sum()), int((y[hi] > 0).sum())],
           [int((y[lo] <= 0).sum()), int((y[lo] > 0).sum())]]
    _, p1 = stats.fisher_exact(tab, alternative="greater")
    _, p2 = stats.fisher_exact(tab, alternative="two-sided")
    return tab, float(p1), float(p2), med


def verdict(rho, p_g, p_w):
    cons = max(p_g, p_w)
    if rho >= 0 or (p_g > 0.10 and p_w > 0.10):
        return ("FALSIFIED", "falsification line fired: rho >= 0 or both permutation p > .10 -- the mechanism claim "
                "does not hold; the paper keeps descriptive wording, no upgrade; this result enters FACTS and the main text (one honest disclosure).")
    if rho < 0 and cons < 0.05:
        return ("ESTABLISHED", "establishment line fired: rho < 0 and conservative permutation p < .05 -- the Discussion "
                "paragraph may be upgraded + the 16-point scatter; wording ceiling: 'the audit's identity term is a "
                "*predictable* correction: its sign and size track how much of the "
                "entity the baseline already encodes'. No causal or universal claim may be made; "
                "Limitations gains one sentence disclosing the panel clustering.")
    return ("INCONCLUSIVE", "neither line fired (rho < 0 but the conservative permutation p falls in the [.05, .10] band, "
            "and at least one permutation p <= .10) -- keep descriptive wording, no upgrade, disclose truthfully.")


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-rerun", action="store_true",
                    help="single-shot override; requires --reason")
    ap.add_argument("--reason", default=None)
    args = ap.parse_args()

    existing = [p for p in (OUT_CSV, OUT_MD, OUT_FIG) if p.exists()]
    if existing:
        if not (args.force_rerun and args.reason):
            sys.exit(f"REFUSED (single-shot, prereg-cd-v1.0 §C): "
                     f"{[str(p) for p in existing]} exist. Reruns are bug-fixes "
                     f"only — pass --force-rerun --reason '...' and log the diff "
                     f"in the prereg revision record.")
        print(f"[single-shot] force-rerun; reason: {args.reason}")

    t0 = time.time()
    sec_rows, s1 = y_sec()
    yelp_rows, s2 = y_yelp()
    maec_rows, s3 = y_maec()
    xs = {**x_sec(), **x_yelp(), **x_maec()}

    rows = []
    for r in sec_rows + yelp_rows + maec_rows:
        xr = xs[r["cell"]]
        rows.append({"panel": r["panel"], "cell": r["cell"],
                     "x_entity_r2_val": xr["x"], "y_identity_gain_relpct": r["y"],
                     "n_val": xr["n_val"], "n_entities": xr["n_entities"],
                     "source_y": r["source_y"], "source_x": xr["source_x"]})
    df = pd.DataFrame(rows)
    assert len(df) == 16 and df[["x_entity_r2_val", "y_identity_gain_relpct"]] \
        .notna().all().all(), "16-point table incomplete"

    x = df.x_entity_r2_val.to_numpy()
    y = df.y_identity_gain_relpct.to_numpy()
    panels = df.panel.to_numpy()
    rho, p_g, p_w = spearman_perm(x, y, panels)
    p_cons = max(p_g, p_w)
    tab, p_fis1, p_fis2, med = fisher_median_split(x, y)
    branch, branch_text = verdict(rho, p_g, p_w)

    # ------------------------------------------------------------------- CSV
    out = df.copy()
    for k, v in [("spearman_rho", rho), ("p_perm_global", p_g),
                 ("p_perm_within_panel", p_w), ("p_perm_conservative", p_cons),
                 ("p_fisher_onesided_greater", p_fis1),
                 ("p_fisher_twosided", p_fis2), ("verdict_branch", branch)]:
        out[k] = v
    T.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    # ---------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    style = {"SEC": dict(marker="o", color="#1f77b4"),
             "Yelp": dict(marker="s", color="#d62728"),
             "MAEC": dict(marker="^", color="#2ca02c")}
    for p in ("SEC", "Yelp", "MAEC"):
        m = df.panel == p
        ax.scatter(df.loc[m, "x_entity_r2_val"], df.loc[m, "y_identity_gain_relpct"],
                   s=34, alpha=0.85, edgecolors="black", linewidths=0.4,
                   label=p, **style[p])
    ax.axhline(0.0, color="gray", lw=0.8, ls="--", zorder=0)
    ax.set_xlabel(r"$x$: entity $R^2$ of reference val predictions")
    ax.set_ylabel(r"$y$: identity-control gain (rel% of reference)")
    ax.legend(frameon=False, fontsize=8, loc="best",
              title=rf"$\rho_S$ = {rho:+.2f}, perm $p$ = {p_cons:.4f}",
              title_fontsize=8)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG)
    plt.close(fig)

    # -------------------------------------------------------------------- md
    md = ["# Pre-registered analysis C — is the identity control's size "
          "predictable from the baseline's entity encoding?\n",
          "> prereg: configs/prereg_mechanism_and_labels.md §C, tag prereg-cd-v1.0 "
          "(committed before any statistic here). Single-shot.\n",
          "## The 16 points\n",
          "| panel | cell | x = entity R² of reference val preds | "
          "y = identity-control gain (rel%) | n_val | n_entities |",
          "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r.panel} | {r.cell} | {r.x_entity_r2_val:.4f} | "
                  f"{r.y_identity_gain_relpct:+.3f} | {r.n_val} | {r.n_entities} |")
    md += ["\n## Pre-registered statistics\n",
           f"- **Spearman rho (16 points)** = **{rho:+.4f}** (prediction: rho < 0)",
           f"- Permutation p (one-sided, {N_PERM:,} draws, seed {PERM_SEED}, "
           f"add-one): **global = {p_g:.4f}**, **within-panel = {p_w:.4f}**; "
           f"conservative (prose) = **{p_cons:.4f}**",
           f"- Median-split Fisher exact (median x = {med:.4f}; table "
           f"[[hi: y<=0 {tab[0][0]}, y>0 {tab[0][1]}], [lo: y<=0 {tab[1][0]}, "
           f"y>0 {tab[1][1]}]]): one-sided (predicted direction) p = "
           f"**{p_fis1:.4f}**, two-sided p = {p_fis2:.4f}",
           f"\n## Verdict — fired branch: **{branch}**\n", branch_text,
           "\n## Disclosures\n",
           "- **Non-independence**: the 16 cells come from **3 panels** (SEC 6, "
           "Yelp 2, MAEC 8); cells share references, entities and overlapping "
           "outcome windows within a panel, so they are not independent draws. "
           "Per the prereg this is met with the double-reported permutation "
           "(within-panel permutation preserves the panel structure under the "
           "null) and the prose takes the conservative p.",
           "- **x definition**: R² of OLS of the reference's log-space "
           "VALIDATION predictions on entity dummies (computed as the exact "
           "one-way-ANOVA between-group variance share), within the same "
           "panel/horizon/reference as y. SEC: log f_R val fit "
           "(f_R = exp OLS[1, log A2-HAR], val-fit on the 5-price-model "
           "inner-join panel — the committed firm-control code path); Yelp: "
           "log of the clipped recalibrated-AR val fit (yelp_protocol "
           "log_ols_frozen applied to val); MAEC: stored val predictions of "
           "the fit-stage reference halves (labels are already v = log vol).",
           "- **No test labels**: every parquet is read with a split=='val' "
           "predicate pushdown (test rows never materialised); the MAEC read "
           "omits the label column entirely; _assert_val_only() guards every "
           "frame. Val labels enter only the val-side recalibration fits "
           "(prereg boundary: 'C reads only frozen predictions and val labels'). y is taken from "
           "frozen artifacts only — nothing on the y side was recomputed from "
           "raw predictions.",
           f"- **Sources, y**: SEC 6 cells = {SRC_SEC_Y} column "
           "rel_impr_firmMeanOnly_vs_fR (verified unique within each "
           "(disc, h) across all 69 rows — max spread exactly 0.0); Yelp 2 "
           f"cells = {SRC_YELP_Y} row 4 (entity-mean control, chronological) "
           "at full CSV precision — the generating protocol_results.json "
           "(tfidf primary) is no longer on disk, and the cascade CSV is the "
           "frozen per-cell artifact of the same pipeline "
           f"(yelp_cascade_table.py); MAEC 8 cells = {SRC_MAEC_Y} "
           "horizons[h][ref].entity.delta_rel_pct (STPEV expanding control, "
           "field names verified in maec_protocol.py).",
           "- **Sources, x**: SEC = results/runs/{A2_har_rv,A6_shar,A3_garch,"
           "A4_egarch,A5_arima}_full_{disc}_seed2026/predictions.parquet val "
           "rows; Yelp = results/second_domain/preds/preds_ar_ridge.parquet "
           "val rows (∩ preds_tfidf_chrono row set, no rows dropped); MAEC = "
           "results/second_domain/maec/preds/preds_r_{ar,har}_primary.parquet "
           "val rows.",
           f"- **Permutation conventions**: {N_PERM:,} draws each scheme from "
           f"a single np.random.default_rng({PERM_SEED}) stream (global draws "
           "first, then within-panel), one-sided in the registered direction "
           "(rho < 0), add-one p = (1 + #{rho_perm <= rho_obs}) / (N + 1).",
           "- **Mechanical caveat on x**: entities with few val rows fit their "
           "dummies near-perfectly, so x levels are panel-size dependent "
           "(n_val/n_entities per cell in the table); the statistic is "
           "rank-based and the within-panel permutation is unaffected.",
           "\n## SANITY (anchors reproduced)\n"]
    for s in s1 + s2 + s3:
        md.append(f"- {s}")
    md.append(f"\nGenerated {time.strftime('%Y-%m-%d %H:%M:%S')} in "
              f"{time.time() - t0:.1f}s by "
              f"scripts/analysis/mechanism_identity_predictability.py; outputs: "
              f"mechanism_identity.csv/.md + writing/paper/figures/"
              f"mechanism_identity.pdf")
    OUT_MD.write_text("\n".join(md))

    # ------------------------------------------------------------------ console
    print("\n=== prereg C — mechanism: identity-control predictability ===")
    print(df[["panel", "cell", "x_entity_r2_val", "y_identity_gain_relpct",
              "n_val", "n_entities"]].to_string(
        index=False, formatters={"x_entity_r2_val": "{:.4f}".format,
                                 "y_identity_gain_relpct": "{:+.3f}".format}))
    print(f"\nSpearman rho = {rho:+.4f}  (prediction rho < 0)")
    print(f"perm p one-sided: global = {p_g:.4f}  within-panel = {p_w:.4f}  "
          f"conservative = {p_cons:.4f}")
    print(f"Fisher median-split: table {tab}, one-sided p = {p_fis1:.4f}, "
          f"two-sided p = {p_fis2:.4f}")
    print(f"VERDICT: {branch}")
    for s in s1 + s2 + s3:
        print(f"  [SANITY] {s}")
    print(f"wrote {OUT_CSV}\nwrote {OUT_MD}\nwrote {OUT_FIG}")


if __name__ == "__main__":
    main()
