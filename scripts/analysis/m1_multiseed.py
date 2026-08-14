"""M1 multiseed — Family-1 headline (log-space val-fit/test-apply combination) re-run at
EVERY C/D seed in {2026, 2027, 2028}, to kill the "seed2026 cherry-pick" rebuttal.

A/B models are seed-invariant (single seed2026 run) and are therefore out of scope here;
the HAR reference (A2) is likewise seed-invariant and shared across seeds.

For each C/D model x disclosure x horizon x seed:
  f_R = exp(a + b log fHAR)              (val-fit, test-apply)
  f_U = exp(a + b log fHAR + g log fText)
  rel_impr_pct = 100 * (QLIKE_R - QLIKE_U) / QLIKE_R   on test
  DM on fc.qlike loss (two-sided), Holm WITHIN family = within each seed's C/D grid.

Reported per cell: rel_impr_pct per seed, cross-seed mean +/- std, sign agreement
(n seeds with DM<0 & raw p<.05; also Holm), and whether seeds DISAGREE in DM sign.
Aggregate: of the seed2026 "genuine" C/D cells (from forecast_combination_grid.csv),
how many keep a significant same-sign increment in >=2/3 and 3/3 seeds.

SANITY: the seed2026 column must reproduce forecast_combination_grid.csv rel_impr_pct
exactly for the overlapping C/D cells (max abs diff printed).

Run from repo root:  .venv/bin/python scripts/analysis/m1_multiseed.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "scripts/analysis")
sys.path.insert(0, "src")
import forecast_combination as fc
from sp500vol.evaluation.dm_test import dm_test

KEY = fc.KEY
SORT = fc.SORT
HORIZONS = fc.HORIZONS
SEEDS = (2026, 2027, 2028)
DISCS = ("long_form", "event_driven", "combined")
RUNS = Path("results/runs")


def cd_models(disc):
    """All C/D models that have a run dir for this disclosure (any seed)."""
    ms = set()
    for p in RUNS.glob(f"[CD]*_full_{disc}_seed*"):
        stem = p.name
        if not stem.endswith(tuple(f"seed{s}" for s in SEEDS)):
            continue  # skip _smoke etc.
        ms.add(stem.split(f"_full_{disc}_seed")[0])
    return sorted(ms)


def load(run, disc, seed):
    return pd.read_parquet(RUNS / f"{run}_full_{disc}_seed{seed}" / "predictions.parquet")


def main():
    rows = []
    for disc in DISCS:
        har = load("A2_har_rv", disc, 2026)[["split"] + KEY + [
            "prediction_realised_vol", "label_realised_vol", "filing_time_utc"]].rename(
            columns={"prediction_realised_vol": "fhar"})
        for m in cd_models(disc):
            for seed in SEEDS:
                path = RUNS / f"{m}_full_{disc}_seed{seed}" / "predictions.parquet"
                if not path.exists():
                    continue
                txt = pd.read_parquet(path)[KEY + ["prediction_realised_vol"]].rename(
                    columns={"prediction_realised_vol": "ftext"})
                d = har.merge(txt, on=KEY)
                for h in HORIZONS:
                    dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(SORT, kind="mergesort")
                    dt = d[(d.horizon_days == h) & (d.split == "test")].sort_values(SORT, kind="mergesort")
                    if len(dv) < 100 or len(dt) < 30:
                        continue
                    yv, fhv, ftv = dv.label_realised_vol.to_numpy(), dv.fhar.to_numpy(), dv.ftext.to_numpy()
                    yt, fhr, ftt = dt.label_realised_vol.to_numpy(), dt.fhar.to_numpy(), dt.ftext.to_numpy()
                    fR, fU, g_log = fc.log_combo(yv, fhv, ftv, fhr, ftt)
                    lR, lU = fc.qlike(yt, fR), fc.qlike(yt, fU)
                    qR, qU = float(lR.mean()), float(lU.mean())
                    rel = 100.0 * (qR - qU) / qR if qR > 0 else float("nan")
                    dmq, pq = dm_test(lU, lR, h=h)
                    rows.append({"disc": disc, "model": m, "h": h, "seed": seed,
                                 "n_test": len(dt), "qlike_R": qR, "qlike_U": qU,
                                 "rel_impr_pct": rel, "g_log": g_log,
                                 "dm_q": float(dmq), "p_q": float(pq)})

    df = pd.DataFrame(rows)
    # Holm WITHIN family = within each seed's full C/D grid (same multiplicity per seed)
    df["p_holm"] = np.nan
    for seed in SEEDS:
        msk = df.seed == seed
        df.loc[msk, "p_holm"] = fc.holm(df.loc[msk, "p_q"].fillna(1.0).values)
    df["sig_raw"] = (df.dm_q < 0) & (df.p_q < 0.05)
    df["sig_holm"] = (df.dm_q < 0) & (df.p_holm < 0.05)

    # ---- wide per-cell table ------------------------------------------------
    cell_key = ["disc", "model", "h"]
    wide = df.pivot_table(index=cell_key, columns="seed",
                          values=["rel_impr_pct", "dm_q", "p_q", "p_holm"], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    rel_cols = [f"rel_impr_pct_{s}" for s in SEEDS]
    wide["rel_mean"] = wide[rel_cols].mean(axis=1)
    wide["rel_std"] = wide[rel_cols].std(axis=1, ddof=1)
    agg = df.groupby(cell_key).agg(
        n_seeds=("seed", "nunique"),
        n_sig_raw=("sig_raw", "sum"),
        n_sig_holm=("sig_holm", "sum"),
        n_dm_neg=("dm_q", lambda s: int((s < 0).sum())),
        n_dm_pos=("dm_q", lambda s: int((s > 0).sum())),
    ).reset_index()
    wide = wide.merge(agg, on=cell_key)
    wide["sign_disagree"] = (wide.n_dm_neg > 0) & (wide.n_dm_pos > 0)

    # ---- sanity vs seed2026 grid + genuine baseline -------------------------
    grid = pd.read_csv("results/tables/forecast_combination_grid.csv")
    grid_cd = grid[grid.model.str.match(r"^[CD]")][
        ["disc", "model", "h", "rel_impr_pct", "genuine"]].rename(
        columns={"rel_impr_pct": "grid2026_rel_impr_pct", "genuine": "grid2026_genuine"})
    wide = wide.merge(grid_cd, on=cell_key, how="left")
    chk = wide.dropna(subset=["grid2026_rel_impr_pct"])
    max_diff = float((chk["rel_impr_pct_2026"] - chk["grid2026_rel_impr_pct"]).abs().max())
    n_overlap = len(chk)

    # ---- aggregates ----------------------------------------------------------
    gen = wide[wide.grid2026_genuine == True]  # noqa: E712
    n_gen = len(gen)
    keep3_h = int((gen.n_sig_holm == 3).sum())
    keep2_h = int((gen.n_sig_holm >= 2).sum())
    keep3_r = int((gen.n_sig_raw == 3).sum())
    keep2_r = int((gen.n_sig_raw >= 2).sum())
    n_disagree = int(wide.sign_disagree.sum())
    n_disagree_gen = int(gen.sign_disagree.sum())
    n_cells = len(wide)
    all3_sig_holm = int((wide.n_sig_holm == 3).sum())

    # ---- outputs -------------------------------------------------------------
    out = wide.sort_values(cell_key)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    out.to_csv("results/tables/m1_multiseed.csv", index=False)

    md = ["# M1 Family-1 across all three C/D seeds (2026/2027/2028)\n",
          "Same headline computation as forecast_combination.py FAMILY 1 (log-space nested "
          "combination, weights fit on VALIDATION only, applied frozen to TEST; DM on QLIKE "
          "loss, two-sided; Holm within each seed's C/D family). HAR reference = A2 "
          "(seed-invariant). A/B models are seed-invariant and hence excluded.\n",
          f"**Sanity:** seed2026 reproduces forecast_combination_grid.csv rel_impr_pct on all "
          f"{n_overlap} overlapping C/D cells; max abs diff = {max_diff:.6f} pp.\n",
          f"**Aggregate ({n_cells} C/D disclosure-model-horizon cells):**\n",
          f"- Of the **{n_gen}** seed2026 'genuine' C/D cells (grid definition: DM<0, Holm<.05, "
          f"placebo null): **{keep3_h}** keep a significant same-sign increment (DM<0, Holm<.05) "
          f"in 3/3 seeds and **{keep2_h}** in >=2/3 seeds "
          f"(raw p<.05: {keep3_r} in 3/3, {keep2_r} in >=2/3).",
          f"- Cells where seeds DISAGREE in DM sign: **{n_disagree}**/{n_cells} "
          f"({n_disagree_gen} of them among seed2026-genuine cells).",
          f"- Cells significant (DM<0, Holm<.05) in ALL 3 seeds, whole C/D grid: "
          f"{all3_sig_holm}/{n_cells}.\n"]
    for disc in DISCS:
        sub = out[out.disc == disc]
        if sub.empty:
            continue
        md.append(f"\n## {disc}\n"
                  "| model | h | rel% 2026 | rel% 2027 | rel% 2028 | mean±std | "
                  "DM 2026 | DM 2027 | DM 2028 | n sig (raw) | n sig (Holm) | "
                  "sign disagree | genuine@2026 |\n"
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            gtag = ("YES" if r.grid2026_genuine == True else  # noqa: E712
                    ("no" if r.grid2026_genuine == False else "-"))  # noqa: E712
            md.append(
                f"| {r.model} | {r.h} | {r.rel_impr_pct_2026:+.2f} | {r.rel_impr_pct_2027:+.2f} | "
                f"{r.rel_impr_pct_2028:+.2f} | {r.rel_mean:+.2f}±{r.rel_std:.2f} | "
                f"{r.dm_q_2026:+.2f} | {r.dm_q_2027:+.2f} | {r.dm_q_2028:+.2f} | "
                f"{int(r.n_sig_raw)}/3 | {int(r.n_sig_holm)}/3 | "
                f"{'YES' if r.sign_disagree else 'no'} | {gtag} |")
    md.append("\n`genuine@2026 = '-'` marks cells outside the original grid's model set "
              "(no placebo-based genuine flag there); rel% = 100*(QLIKE_R-QLIKE_U)/QLIKE_R on test.")
    with open("results/tables/m1_multiseed.md", "w") as fh:
        fh.write("\n".join(md))

    print("=== M1 multiseed done ===")
    print(f"cells={n_cells} (x3 seeds); sanity overlap n={n_overlap}, max|diff|={max_diff:.6f} pp")
    print(f"seed2026-genuine C/D cells={n_gen}; keep sig same-sign (Holm): 3/3={keep3_h}, >=2/3={keep2_h}; "
          f"(raw p): 3/3={keep3_r}, >=2/3={keep2_r}")
    print(f"sign-disagree cells={n_disagree}/{n_cells} (among genuine: {n_disagree_gen}); "
          f"all-3-seed Holm-sig cells={all3_sig_holm}")
    print("wrote results/tables/m1_multiseed.csv and .md")


if __name__ == "__main__":
    main()
