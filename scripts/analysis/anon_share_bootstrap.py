"""prereg-ea v1.6 (configs/prereg_swap_lf_and_anon.md §v1.6, tag prereg-ea-v1.6)
— day-block bootstrap 95% CI for the identity share.

UNCERTAINTY QUANTIFICATION ONLY: branch adjudication is permanently locked by
the fired anon_arm.{csv,md}; this analysis cannot and does not alter it
(registered motivation: R12 skeptic — "Bootstrap confidence intervals on the
per-horizon identity shares (0.51/0.56/0.71)").

REGISTERED DESIGN (binding):
  * data plane = the committed ED anon runs, exactly anon_score.py
    --channel ed's: committed unmasked runs (C6_llmtext / C2_finbert_s1) +
    their anonmask runs; B2 exited at G1 and does not participate;
  * resampling unit = effective trading day (the day-clustered DM unit);
    each cell resamples days WITH replacement within its own test panel,
    cells independent; B = 2000, seed = 2026;
  * combiner weights committed (val-fit, test-frozen) and NOT refit inside
    draws — a draw only reweights test days, so the per-observation
    (qR, qU_unmasked, qU_masked) triplets are fixed and the draw statistic is
    a ratio of day-sum aggregates (vectorised below);
  * per draw: rel = 100*sum(qR-qU)/sum(qR) over the drawn days (identical to
    the anon_score cell statistic, whose obs-mean normaliser cancels), and
    share = 1 - rel_masked/rel_unmasked; a draw with unmasked rel <= 0 has an
    UNDEFINED share (v1.1 n/a rule) — the CI is the percentile 95% interval
    over DEFINED draws and the undefined fraction is disclosed; cells with
    undefined fraction > 20% are flagged UNSTABLE;
  * registered readouts: C6 HAR-side 3 cells, C2 HAR-side 3 cells, each arm's
    median (per draw: median over that arm's DEFINED cell shares — the
    point-estimate convention), and the pooled 6-cell median;
  * outputs results/tables/anon_share_ci.{csv,md}, WRITE-ONCE single shot.

GATES (before writing anything): the script's recomputed point estimates
(unmasked/masked rel_har and share per cell) must reproduce the fired
anon_arm.csv at machine precision — same-machinery proof; test-panel KEY
alignment and fR (recalibrated-HAR leg) bit-identity between the unmasked and
masked merges are asserted per cell.

Run ONCE from repo root (CPU, <=4 threads):
    .venv/bin/python scripts/analysis/anon_share_bootstrap.py
Validation (writes nothing under results/): --selftest
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import argparse
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, "scripts/analysis")
sys.path.insert(0, "src")

import forecast_combination as fc  # noqa: E402

KEY = ["ticker", "accession", "horizon_days"]
DISC = "event_driven"
HORIZONS = (5, 10, 20)
B = 2000
SEED = 2026
CI_LO, CI_HI = 2.5, 97.5
UNSTABLE_FRAC = 0.20
RTOL = 1e-12  # machine-precision anchor vs the fired anon_arm.csv
# registered cell order = the RNG stream order (deterministic)
ARMS = [("c6", "C6_llmtext", "C6_llmtext_anonmask"),
        ("c2", "C2_finbert_s1", "C2_finbert_s1_anonmask")]
FINAL_CSV = REPO / "results/tables/anon_share_ci.csv"
FINAL_MD = REPO / "results/tables/anon_share_ci.md"
ANON_ARM = REPO / "results/tables/anon_arm.csv"
DECLARATION = ("UNCERTAINTY QUANTIFICATION ONLY — branch adjudication is "
               "locked by anon_arm.{csv,md} (fired) and is not altered here. "
               "prereg-ea v1.6, day-block bootstrap, B=2000, seed 2026.")


def _fatal(msg: str) -> None:
    raise SystemExit(f"[share-ci] FATAL: {msg}")


def close(a, b):
    a, b = float(a), float(b)
    if np.isnan(a) and np.isnan(b):
        return True
    return abs(a - b) <= RTOL * max(abs(a), abs(b), 1.0)


# --------------------------------------------------------------------------- #
# cell materialisation — the anon_score.m1_cells machinery, day-aggregated     #
# --------------------------------------------------------------------------- #
def cell_series(vu: pd.DataFrame, teu: pd.DataFrame,
                vm: pd.DataFrame, tem: pd.DataFrame) -> dict:
    """From the unmasked (vu/teu) and masked (vm/tem) merged cell frames
    (columns: KEY, label_realised_vol, fh, ft, effective_trading_day),
    compute the committed frozen-weight legs and the per-day aggregates.

    Returns dict with: days (sorted unique), S_qR, S_dU_u, S_dU_m (per-day
    sums of qR and of qR-qU for the unmasked/masked text legs), rel_u, rel_m,
    share (the FULL-panel point statistics, for the anon_arm anchor), n_days,
    n_test.
    """
    if len(teu) != len(tem) or not (
            teu[KEY].to_numpy() == tem[KEY].to_numpy()).all():
        _fatal("unmasked/masked test panels misaligned within a cell")
    y = teu.label_realised_vol.values
    fR_u, fU_u, _ = fc.log_combo(vu.label_realised_vol.values, vu.fh.values,
                                 vu.ft.values, teu.fh.values, teu.ft.values)
    fR_m, fU_m, _ = fc.log_combo(vm.label_realised_vol.values, vm.fh.values,
                                 vm.ft.values, tem.fh.values, tem.ft.values)
    # the recalibrated-HAR leg carries no text: bit-identity is structural
    if not np.array_equal(fR_u, fR_m):
        _fatal("recalibrated-HAR legs differ between unmasked/masked merges")
    qR = fc.qlike(y, fR_u)
    dU_u = qR - fc.qlike(y, fU_u)
    dU_m = qR - fc.qlike(y, fU_m)
    rel_u = 100 * dU_u.mean() / qR.mean()
    rel_m = 100 * dU_m.mean() / qR.mean()
    share = float("nan") if rel_u <= 0 else 1.0 - rel_m / rel_u

    days = pd.to_datetime(teu.effective_trading_day).to_numpy()
    order = np.argsort(days, kind="stable")
    days_sorted = days[order]
    uniq, starts = np.unique(days_sorted, return_index=True)
    S_qR = np.add.reduceat(qR[order], starts)
    S_dU_u = np.add.reduceat(dU_u[order], starts)
    S_dU_m = np.add.reduceat(dU_m[order], starts)
    return dict(days=uniq, S_qR=S_qR, S_dU_u=S_dU_u, S_dU_m=S_dU_m,
                rel_u=float(rel_u), rel_m=float(rel_m), share=float(share),
                n_days=int(len(uniq)), n_test=int(len(teu)))


def load_cell(a2: pd.DataFrame, tu: pd.DataFrame, tm: pd.DataFrame,
              h: int) -> dict:
    """Real-data path: replicate anon_score.m1_cells' merge VERBATIM for the
    unmasked and masked runs, then day-aggregate."""
    a2h = a2[a2.horizon_days == h]
    mu = a2h.merge(tu[tu.horizon_days == h], on=KEY).dropna()
    mm = a2h.merge(tm[tm.horizon_days == h], on=KEY).dropna()
    return cell_series(mu[mu.split == "val"], mu[mu.split == "test"],
                       mm[mm.split == "val"], mm[mm.split == "test"])


# --------------------------------------------------------------------------- #
# bootstrap                                                                    #
# --------------------------------------------------------------------------- #
def bootstrap_draws(cell: dict, b: int, rng: np.random.Generator) -> np.ndarray:
    """B share draws for one cell: resample its test DAYS with replacement,
    recompute both increments on frozen legs, share = 1 - rel_m/rel_u; NaN
    where the drawn unmasked increment <= 0 (v1.1 n/a rule)."""
    n = cell["n_days"]
    idx = rng.integers(0, n, size=(b, n))
    s_qR = cell["S_qR"][idx].sum(axis=1)
    rel_u = 100 * cell["S_dU_u"][idx].sum(axis=1) / s_qR
    rel_m = 100 * cell["S_dU_m"][idx].sum(axis=1) / s_qR
    share = np.full(b, np.nan)
    ok = rel_u > 0
    share[ok] = 1.0 - rel_m[ok] / rel_u[ok]
    return share


def nan_median_rows(mat: np.ndarray) -> np.ndarray:
    """Row-wise median over DEFINED entries (the point-estimate convention);
    all-NaN rows stay NaN silently (that draw is undefined for the median)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(mat, axis=1)


def summarise(name: str, arm: str, h, point: float,
              draws: np.ndarray) -> dict:
    defined = draws[np.isfinite(draws)]
    frac_undef = 1.0 - len(defined) / len(draws)
    if len(defined):
        lo, hi = np.percentile(defined, [CI_LO, CI_HI])
    else:
        lo = hi = float("nan")
    return {
        "readout": name, "arm": arm, "h": h,
        "share_point": point,
        "ci_lo": float(lo), "ci_hi": float(hi),
        "n_draws": int(len(draws)), "n_defined": int(len(defined)),
        "undefined_frac": float(frac_undef),
        "unstable": bool(frac_undef > UNSTABLE_FRAC),
    }


def run_bootstrap(cells: dict, points: dict, b: int,
                  rng: np.random.Generator) -> pd.DataFrame:
    """cells: {(arm,h): cell dict}; points: {(arm,h): point share}.
    Draw order = the registered ARMS x HORIZONS order (RNG determinism)."""
    draws = {}
    rows = []
    for arm, _base, _masked in ARMS:
        for h in HORIZONS:
            draws[(arm, h)] = bootstrap_draws(cells[(arm, h)], b, rng)
    for arm, _base, _masked in ARMS:
        for h in HORIZONS:
            rows.append(summarise("cell", arm, h, points[(arm, h)],
                                  draws[(arm, h)]))
    for arm, _base, _masked in ARMS:
        mat = np.column_stack([draws[(arm, h)] for h in HORIZONS])
        pt = np.nanmedian([points[(arm, h)] for h in HORIZONS]) \
            if np.isfinite([points[(arm, h)] for h in HORIZONS]).any() else float("nan")
        rows.append(summarise("arm_median", arm, "5/10/20", float(pt),
                              nan_median_rows(mat)))
    mat = np.column_stack([draws[(arm, h)]
                           for arm, _b_, _m_ in ARMS for h in HORIZONS])
    pts = [points[(arm, h)] for arm, _b_, _m_ in ARMS for h in HORIZONS]
    pt = np.nanmedian(pts) if np.isfinite(pts).any() else float("nan")
    rows.append(summarise("pooled_median", "c6+c2", "5/10/20", float(pt),
                          nan_median_rows(mat)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# real-data driver                                                             #
# --------------------------------------------------------------------------- #
def main_real() -> int:
    if FINAL_CSV.exists() or FINAL_MD.exists():
        print(f"REFUSED: {FINAL_CSV} / {FINAL_MD} already exists — write-once "
              "single shot (prereg-ea v1.6). A re-run requires a recorded "
              "prereg revision first.")
        return 2
    if not ANON_ARM.exists():
        _fatal("anon_arm.csv missing — the ED table must have fired (it "
               "locks adjudication AND anchors the point estimates)")
    anchor = pd.read_csv(ANON_ARM)
    anchor = anchor[anchor.status == "executed"].set_index(
        ["arm", "h"]) if "status" in anchor.columns else anchor.set_index(["arm", "h"])

    t0 = time.time()
    a2 = fc.load("A2_har_rv", DISC)[KEY + ["split", "label_realised_vol",
                                           "prediction_realised_vol",
                                           "effective_trading_day"]] \
        .rename(columns={"prediction_realised_vol": "fh"})
    cells, points = {}, {}
    for arm, base, masked in ARMS:
        tu = fc.load(base, DISC)[KEY + ["prediction_realised_vol"]] \
            .rename(columns={"prediction_realised_vol": "ft"})
        tm = fc.load(masked, DISC)[KEY + ["prediction_realised_vol"]] \
            .rename(columns={"prediction_realised_vol": "ft"})
        for h in HORIZONS:
            c = load_cell(a2, tu, tm, h)
            # ---- anchor gate: same machinery as the fired anon_arm.csv ----
            ref = anchor.loc[(arm, float(h))]
            for mine, col in ((c["rel_u"], "rel_har_unmasked"),
                              (c["rel_m"], "rel_har_masked"),
                              (c["share"], "share_anon_har")):
                if not close(mine, float(ref[col])):
                    _fatal(f"anchor FAIL {arm}/h{h} {col}: recomputed {mine!r}"
                           f" vs anon_arm.csv {float(ref[col])!r}")
            if int(ref["n_days"]) != c["n_days"]:
                _fatal(f"anchor FAIL {arm}/h{h}: n_days {c['n_days']} vs "
                       f"anon_arm.csv {int(ref['n_days'])}")
            cells[(arm, h)] = c
            points[(arm, h)] = c["share"]
            print(f"[share-ci] cell {arm}/h{h}: n_test={c['n_test']} "
                  f"n_days={c['n_days']} — anchored to anon_arm.csv at "
                  f"machine precision")
    t_prep = time.time() - t0

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    df = run_bootstrap(cells, points, B, rng)
    t_boot = time.time() - t0

    # ---------------- write-once (re-check just before) ----------------
    if FINAL_CSV.exists() or FINAL_MD.exists():
        print("REFUSED at write time: output appeared concurrently")
        return 2
    meta = {
        "prereg": "prereg-ea v1.6 §share-CI (tag prereg-ea-v1.6)",
        "B": B, "seed": SEED,
        "resampling_unit": "effective_trading_day (day-block, per cell "
                           "independent, with replacement, test panel only)",
        "weights": "committed val-fit test-frozen (not refit in draws)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = df.copy()
    for k, v in meta.items():
        out[k] = v
    FINAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(FINAL_CSV, index=False)

    def row_md(r):
        ci = (f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
              if np.isfinite(r.ci_lo) else "n/a")
        pt = f"{r.share_point:+.3f}" if np.isfinite(r.share_point) else "n/a"
        flag = "**UNSTABLE**" if r.unstable else ""
        return (f"| {r.readout} | {r.arm} | {r.h} | {pt} | {ci} | "
                f"{100 * r.undefined_frac:.1f}% | {flag} |")

    md = [
        f"**{DECLARATION}**",
        "",
        "# Identity share — day-block bootstrap 95% CI (prereg-ea v1.6)",
        "",
        f"Registered at tag `prereg-ea-v1.6` BEFORE any share-CI statistic; "
        f"computed once on {meta['generated_utc']}.",
        "",
        f"- **B = {B}**, **seed = {SEED}**; percentile interval "
        f"[{CI_LO}%, {CI_HI}%].",
        "- **Resampling unit = effective trading day** (the day-clustered DM "
        "unit): each cell independently resamples its own test days with "
        "replacement; val is untouched.",
        "- **Combiner weights frozen**: committed val-fit, test-frozen legs; "
        "a draw reweights test days only, so per-observation "
        "(qR, qU_unmasked, qU_masked) are fixed and each draw statistic is a "
        "ratio of day-sum aggregates — identical machinery to the "
        "anon_score cell statistic (obs-mean normalisers cancel).",
        "- **Undefined rule (v1.1)**: a draw with unmasked increment <= 0 "
        "has no defined share; CIs use defined draws only; the undefined "
        f"fraction is disclosed per readout and > {UNSTABLE_FRAC:.0%} is "
        "flagged UNSTABLE.",
        "- **Median readouts**: per draw, the median over the DEFINED cell "
        "shares of that draw (the point-estimate convention); a draw with "
        "zero defined cells is undefined for the median.",
        "- **Gate**: recomputed point estimates (rel_har unmasked/masked, "
        "share) reproduce the fired anon_arm.csv at machine precision "
        f"(rtol {RTOL:g}) in all 6 cells — PASSED before any draw.",
        "- B2 exited at G1 and does not participate (see anon_arm.md); "
        "firmID-side cells are outside the registered readout set.",
        "",
        "| readout | arm | h | point share | 95% CI | undefined % | flag |",
        "|---|---|---|--:|---|--:|---|",
    ]
    for _, r in df.iterrows():
        md.append(row_md(r))
    md += [
        "",
        "## Prose rule (registered)",
        "",
        "- Wherever the paper cites a share point estimate, attach the CI "
        "(the abstract may be exempted for space); CIs change NO locked "
        "branch wording.",
        "",
        f"Runtime: cell prep {t_prep:.1f}s, bootstrap {t_boot:.1f}s "
        f"({len(df)} readouts).",
        "",
        "Single-shot: this file is written once; re-running refuses while "
        "it exists.",
    ]
    FINAL_MD.write_text("\n".join(md) + "\n")
    print(f"[share-ci] wrote {FINAL_CSV} ({len(df)} rows) and {FINAL_MD}")
    print(f"[share-ci] runtime: prep {t_prep:.1f}s, bootstrap {t_boot:.1f}s")
    print(df.to_string(index=False))
    return 0


# --------------------------------------------------------------------------- #
# selftest — synthetic panels; writes nothing under results/                   #
# --------------------------------------------------------------------------- #
def _make_panel(rng, n_days=160, n_firms=8, signal_u=0.5, signal_m=0.25,
                noise=0.30):
    """Synthetic val+test cell frames: labels lognormal around a HAR-ish
    forecast; text forecasts carry a day-level signal of chosen strength
    (unmasked stronger than masked -> positive share)."""
    frames = {}
    for split, nd in (("val", n_days), ("test", n_days)):
        day_shock = rng.normal(0, 0.4, nd)
        rows = []
        for d in range(nd):
            for f in range(n_firms):
                base = 0.15 * np.exp(rng.normal(0, 0.3))
                lab = base * np.exp(day_shock[d] * 0.5 + rng.normal(0, noise))
                fh = base * np.exp(rng.normal(0, 0.25))
                ft_u = lab ** signal_u * fh ** (1 - signal_u) * np.exp(
                    rng.normal(0, 0.10))
                ft_m = lab ** signal_m * fh ** (1 - signal_m) * np.exp(
                    rng.normal(0, 0.10))
                rows.append({"ticker": f"F{f}", "accession": f"{split}{d}_{f}",
                             "horizon_days": 5, "label_realised_vol": lab,
                             "fh": fh, "ft_u": ft_u, "ft_m": ft_m,
                             "effective_trading_day":
                                 pd.Timestamp("2020-01-01")
                                 + pd.Timedelta(days=d)})
        frames[split] = pd.DataFrame(rows)
    def split_ft(df, col):
        out = df.copy()
        out["ft"] = out[col]
        return out
    return (split_ft(frames["val"], "ft_u"), split_ft(frames["test"], "ft_u"),
            split_ft(frames["val"], "ft_m"), split_ft(frames["test"], "ft_m"))


def _selftest() -> int:
    print("[selftest] 1) point recovery + CI behaviour on one synthetic panel")
    rng = np.random.default_rng(7)
    vu, teu, vm, tem = _make_panel(rng)
    cell = cell_series(vu, teu, vm, tem)
    assert cell["rel_u"] > 0 and 0 < cell["share"] < 1, \
        f"synthetic panel degenerate: rel_u={cell['rel_u']}, share={cell['share']}"
    draws = bootstrap_draws(cell, 1500, np.random.default_rng(SEED))
    s = summarise("cell", "syn", 5, cell["share"], draws)
    assert s["ci_lo"] < cell["share"] < s["ci_hi"], "CI must contain the point"
    assert s["undefined_frac"] < 0.5
    # determinism: same seed -> identical draws
    d2 = bootstrap_draws(cell, 1500, np.random.default_rng(SEED))
    assert np.array_equal(draws, d2, equal_nan=True), "seeded draws not deterministic"
    print(f"[selftest]    point {cell['share']:+.3f} in CI "
          f"[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}], undefined "
          f"{s['undefined_frac']:.1%}, deterministic OK")

    print("[selftest] 2) coverage intuition: 40 sims, truth from a mega-panel")
    mega = _make_panel(np.random.default_rng(123), n_days=2500, n_firms=10)
    truth = cell_series(*mega)["share"]
    cover = 0
    for i in range(40):
        p = _make_panel(np.random.default_rng(1000 + i))
        c = cell_series(*p)
        d = bootstrap_draws(c, 400, np.random.default_rng(SEED + i))
        s = summarise("cell", "syn", 5, c["share"], d)
        if s["ci_lo"] <= truth <= s["ci_hi"]:
            cover += 1
    rate = cover / 40
    print(f"[selftest]    truth={truth:+.3f}; coverage {cover}/40 = {rate:.0%}")
    assert rate >= 0.80, f"bootstrap coverage implausibly low: {rate:.0%}"

    print("[selftest] 3) undefined-fraction path: null unmasked signal")
    vu, teu, vm, tem = _make_panel(np.random.default_rng(9), signal_u=0.02,
                                   signal_m=0.01)
    c = cell_series(vu, teu, vm, tem)
    d = bootstrap_draws(c, 800, np.random.default_rng(SEED))
    s = summarise("cell", "syn-null", 5, c["share"], d)
    assert s["undefined_frac"] > UNSTABLE_FRAC and s["unstable"], \
        f"null cell not flagged unstable (undef {s['undefined_frac']:.1%})"
    print(f"[selftest]    undefined {s['undefined_frac']:.1%} -> UNSTABLE flag fires")

    print("[selftest] 4) median readouts: all-NaN draw stays NaN, no warning")
    mat = np.array([[0.5, np.nan, 0.7], [np.nan, np.nan, np.nan]])
    med = nan_median_rows(mat)
    assert med[0] == 0.6 and np.isnan(med[1])
    print("[selftest]    defined-cell median convention verified")

    print("[selftest] 5) misalignment + fR-identity guards")
    try:
        cell_series(vu, teu, vm, tem.iloc[::-1].reset_index(drop=True))
        raise AssertionError("misaligned panels not caught")
    except SystemExit as e:
        assert "misaligned" in str(e)
    print("[selftest]    alignment guard fires")

    print("[selftest] 6) write-once single-shot guard (tmp outputs)")
    import tempfile
    global FINAL_CSV, FINAL_MD
    old = (FINAL_CSV, FINAL_MD)
    try:
        with tempfile.TemporaryDirectory(prefix="share_ci_selftest_") as td:
            FINAL_CSV = Path(td) / "anon_share_ci.csv"
            FINAL_MD = Path(td) / "anon_share_ci.md"
            FINAL_CSV.write_text("fired")
            rc = main_real()
            assert rc == 2, f"guard did not refuse (rc={rc})"
    finally:
        FINAL_CSV, FINAL_MD = old
    print("[selftest]    guard refuses while the output exists")

    print("[selftest] ALL PASS — nothing under results/ was touched")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic validation; writes nothing under results/")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    return main_real()


if __name__ == "__main__":
    sys.exit(main())
