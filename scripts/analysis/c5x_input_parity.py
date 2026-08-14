"""P0-3(b) — C5x INPUT-PARITY table: prompting vs same-lineage embeddings on
byte-identical excerpts (long_form).

The abstract's sentence "prompting extracts more of this thin signal than pooling
same-lineage embeddings" currently has NO citable table: the C5x_qwen3exc run
(Qwen3-Embedding-8B on the EXACT C6 curated excerpts — 1A/7/7A else head — isolating
input curation from the prompting mechanism) exists under results/runs/ but appears
only in config_fingerprints. This script commits the comparison.

Models (long_form, the only disclosure C5x was run on):
  - C5_qwen3      : Qwen3-Embedding-8B on the standard C-block chunk pooling input.
                    3 seeds -> per-observation SEED-ENSEMBLE mean is the PRIMARY basis
                    (reuses m1_ensemble_primary.ensemble_text); seed2026 shown as check.
  - C5x_qwen3exc  : same embedder, byte-identical C6 excerpts (input parity). Single run.
  - C6_llmtext    : Qwen3-32B zero-shot prompting on those excerpts. Single decode.

Per model x horizon: STANDALONE test quality (QLIKE vol/variance unit, R^2, pred sd,
n_unique of round(pred,2)) and the M1 increment over the single recalibrated HAR
(fc.log_combo: weights fit on val only, frozen to test; vol-unit QLIKE; day-clustered
DM with raw p AND Holm across the table; label-shuffle placebo DM, 5 seeds).

Run from repo root:  .venv/bin/python scripts/analysis/c5x_input_parity.py
Outputs (NEW files): results/tables/c5x_input_parity.{csv,md}
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/analysis")
import clustered_dm as cdm
import forecast_combination as fc
from m1_ensemble_primary import ensemble_text

KEY = fc.KEY
SORT = fc.SORT
DISC = "long_form"
HORIZONS = (5, 10, 20)

MODELS = [
    # (label, run, basis)  basis: "ens" = 3-seed per-obs mean (primary), "s26" = seed2026
    ("C5_qwen3 (ens, PRIMARY)", "C5_qwen3", "ens"),
    ("C5_qwen3 (seed2026)", "C5_qwen3", "s26"),
    ("C5x_qwen3exc (input-parity)", "C5x_qwen3exc", "s26"),
    ("C6_llmtext (prompted)", "C6_llmtext", "s26"),
]


def text_frame(run, basis):
    if basis == "ens":
        ens, used = ensemble_text(run, DISC)
        return ens, "+".join(str(s) for s in used)
    d = fc.load(run, DISC)[KEY + ["prediction_realised_vol"]].rename(
        columns={"prediction_realised_vol": "ftext"})
    return d, "2026"


def standalone_stats(y, f):
    y = np.asarray(y, float)
    f = np.asarray(f, float)
    vals, counts = np.unique(np.round(f, 2), return_counts=True)
    return {
        "qlike_vol_alone": float(fc.qlike(y, f).mean()),
        "qlike_var_alone": float(fc.qlike(y ** 2, f ** 2).mean()),
        "r2_alone": float(1.0 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "pred_sd": float(f.std()),
        "n_unique_2dp": len(vals),
    }


def main():
    har = fc.load("A2_har_rv", DISC)[["split"] + KEY + [
        "prediction_realised_vol", "label_realised_vol",
        "filing_time_utc", "effective_trading_day"]].rename(
        columns={"prediction_realised_vol": "fhar"})

    rows = []
    for label, run, basis in MODELS:
        t, seeds = text_frame(run, basis)
        d = har.merge(t, on=KEY)
        for h in HORIZONS:
            dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(SORT, kind="mergesort")
            dt = d[(d.horizon_days == h) & (d.split == "test")].sort_values(SORT, kind="mergesort")
            yv, fhv, ftv = (dv.label_realised_vol.to_numpy(), dv.fhar.to_numpy(),
                            dv.ftext.to_numpy())
            yt, fhr, ftt = (dt.label_realised_vol.to_numpy(), dt.fhar.to_numpy(),
                            dt.ftext.to_numpy())
            days = dt.effective_trading_day.to_numpy()

            st = standalone_stats(yt, ftt)
            fR, fU, g = fc.log_combo(yv, fhv, ftv, fhr, ftt)
            lR, lU = fc.qlike(yt, fR), fc.qlike(yt, fU)
            qR, qU = float(lR.mean()), float(lU.mean())
            rel = 100.0 * (qR - qU) / qR
            dm, p, nd = cdm.dm_test_clustered(lU, lR, days, h)
            _, lo, hi = cdm.mbb_ci_daily(lU - lR, days, h)
            pdm = []
            for s in fc.PLACEBO_SEEDS:
                rng = np.random.default_rng(s)
                pR, pU, _ = fc.log_combo(yv, fhv, rng.permutation(ftv),
                                         fhr, rng.permutation(ftt))
                pdm.append(cdm.dm_test_clustered(fc.qlike(yt, pU), fc.qlike(yt, pR),
                                                 days, h)[0])
            rows.append({"model": label, "run": run, "basis": basis, "seeds": seeds,
                         "h": h, "n_test": len(dt), "n_days": nd, **st,
                         "qlike_R": qR, "qlike_U": qU, "rel_impr_pct": rel,
                         "g_log": float(g), "dm_q_clu": dm, "p_q_clu": p,
                         "boot_lo_daily": lo, "boot_hi_daily": hi,
                         "placebo_dm_clu": float(np.mean(pdm))})

    df = pd.DataFrame(rows)
    df["dmq_holm_clu"] = fc.holm(df.p_q_clu.fillna(1.0).values)
    df["genuine"] = (df.dm_q_clu < 0) & (df.dmq_holm_clu < 0.05) & (df.placebo_dm_clu.abs() < 2.0)
    df.to_csv("results/tables/c5x_input_parity.csv", index=False)

    c6 = df[df.run == "C6_llmtext"]
    c5x = df[df.run == "C5x_qwen3exc"]
    c5e = df[(df.run == "C5_qwen3") & (df.basis == "ens")]

    md = [
        "# C5x INPUT-PARITY — prompting vs same-lineage embeddings on identical excerpts"
        " (long_form, P0-3b)",
        "",
        "## RESTATED vs BEFORE",
        "",
        "| | BEFORE | RESTATED (this table) |",
        "|---|---|---|",
        "| citable table for the abstract's \"prompting > same-lineage embeddings\" sentence |"
        " **NONE** (C5x_qwen3exc run existed only in results/runs/ + config_fingerprints rows) |"
        " full M1 comparison committed here |",
        "| basis discipline | - | C5_qwen3 uses the declared PRIMARY 3-seed per-observation"
        " ensemble (m1_ensemble_primary loader); C5x/C6 are single-run by design; seed2026 C5"
        " row shown as a check |",
        "| inference | - | day-clustered DM vs single recalibrated HAR, raw p AND Holm (one"
        f" family, {len(df)} cells), 5-seed label-shuffle placebo |",
        "",
        "C5x_qwen3exc = Qwen3-Embedding-8B run on the BYTE-IDENTICAL curated excerpts fed to the"
        " C6 prompt (10-K 1A/7/7A else head-truncation; ridge head on log target with Duan"
        " smearing) — it isolates input curation from the prompting mechanism within the same"
        " Qwen3 lineage. Combiner weights are val-fit and frozen to test throughout; QLIKE in"
        " vol units; `genuine` = clustered DM<0, Holm<.05, |placebo DM|<2.",
        "",
        "| model | h | seeds | n_test | n_days | QLIKE alone (vol) | QLIKE alone (var) | R^2 alone |"
        " pred sd | n_uniq(2dp) | QLIKE(R) | QLIKE(U) | rel% | g_log | DM(clu) | p raw | Holm |"
        " placebo DM | genuine |",
        "|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|",
    ]
    for _, r in df.iterrows():
        md.append(
            f"| {r.model} | {int(r.h)} | {r.seeds} | {int(r.n_test)} | {int(r.n_days)} | "
            f"{r.qlike_vol_alone:.4f} | {r.qlike_var_alone:.3f} | {r.r2_alone:+.3f} | "
            f"{r.pred_sd:.4f} | {int(r.n_unique_2dp)} | {r.qlike_R:.4f} | {r.qlike_U:.4f} | "
            f"{r.rel_impr_pct:+.2f} | {r.g_log:+.3f} | {r.dm_q_clu:+.2f} | {r.p_q_clu:.4f} | "
            f"{r.dmq_holm_clu:.3f} | {r.placebo_dm_clu:+.2f} | "
            f"{'YES' if r.genuine else 'no'} |")

    def rng_str(s):
        return f"{s.min():+.2f}% to {s.max():+.2f}%"

    md += [
        "",
        "## Reading",
        "",
        f"- **Prompting (C6_llmtext)**: M1 increment {rng_str(c6.rel_impr_pct)} across horizons,"
        f" genuine in {int(c6.genuine.sum())}/3 cells.",
        f"- **Same-lineage embeddings on the SAME excerpts (C5x_qwen3exc)**: "
        f"{rng_str(c5x.rel_impr_pct)}, genuine in {int(c5x.genuine.sum())}/3 cells.",
        f"- **Standard-input embeddings (C5_qwen3, primary ensemble)**: "
        f"{rng_str(c5e.rel_impr_pct)}, genuine in {int(c5e.genuine.sum())}/3 cells.",
        "- Input parity means any C6-vs-C5x gap is attributable to the elicitation mechanism"
        " (prompting vs embedding pooling), not to excerpt curation. This is the citable basis"
        " for the abstract's prompting-vs-embedding sentence; cite the exact rel% and Holm p"
        " from this table, and keep the claim limited to long_form (the only C5x cell run).",
        "- Caveat for the paper: this compares mechanisms at parity of INPUT, not of parameter"
        " count (32B decoder vs 8B embedder + ridge); say \"same-lineage\" not \"same-size\".",
        "",
    ]
    Path("results/tables/c5x_input_parity.md").write_text("\n".join(md))
    print(f"wrote results/tables/c5x_input_parity.csv/.md ({len(df)} cells)")
    print(df[["model", "h", "qlike_vol_alone", "rel_impr_pct", "dm_q_clu", "p_q_clu",
              "dmq_holm_clu", "placebo_dm_clu", "genuine"]].to_string(index=False))


if __name__ == "__main__":
    main()
