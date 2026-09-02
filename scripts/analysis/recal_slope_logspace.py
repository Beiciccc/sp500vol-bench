"""The recalibration slope of Eq. (1), on the equation's own scale.

Chapter 3 writes the reference as $f_R = \\exp(a + b\\log f_{price})$ and then
quotes a mean slope for $b$.  The number it quoted came from FAMILY 3 of
`forecast_combination.md`, whose `recal_b` is produced by `level_combo` --- an
OLS of RV on the HAR forecast in LEVELS.  `log_combo` fits the equation the
chapter prints, but it returns only `g_text`; its own $b$ is never stored, so
the printed slope silently swapped scales.

This recomputes $b$ from `log_combo`'s regression on exactly the rows FAMILY 3
uses: the primary text arm's merge with the recalibration reference, validation
split, same sort and same cells.  Nothing is refit and no model is re-run; the
committed prediction artefacts are read as they stand.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forecast_combination as fc  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "results", "tables", "recal_slope_logspace.md")


def main():
    os.chdir(ROOT)
    rows = []
    for disc in fc.SETS:
        har = fc.load("A2_har_rv", disc)[["split"] + fc.KEY +
                                         ["prediction_realised_vol", "label_realised_vol", "filing_time_utc"]]
        har = har.rename(columns={"prediction_realised_vol": "fhar"})
        txt = fc.load(fc.PRIMARY_MODEL, disc)[fc.KEY + ["prediction_realised_vol"]]
        txt = txt.rename(columns={"prediction_realised_vol": "ftext"})
        d = har.merge(txt, on=fc.KEY)
        for h in fc.HORIZONS:
            dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(
                fc.SORT, kind="mergesort")
            if len(dv) < 100:
                continue
            yv = dv.label_realised_vol.to_numpy()
            fhv = dv.fhar.to_numpy()
            ly = np.log(np.clip(yv, fc.EPS, None))
            lh = np.log(np.clip(fhv, fc.EPS, None))
            b_log = float(fc.ols(ly, np.column_stack([np.ones(len(ly)), lh]))[1])
            b_lev = float(fc.ols(yv, np.column_stack([np.ones(len(yv)), fhv]))[1])
            rows.append((disc, h, len(dv), b_log, b_lev))

    logs = [r[3] for r in rows]
    levs = [r[4] for r in rows]
    with open(OUT, "w") as fh:
        fh.write("# Recalibration slope of Eq. (1), log space vs level space\n\n")
        fh.write(f"Primary arm {fc.PRIMARY_MODEL}, validation split, "
                 f"the rows FAMILY 3 of forecast_combination.md is computed on.\n\n")
        fh.write("| disclosure | h | n val | b (log space, Eq. 1) | b (level space, FAMILY 3) |\n")
        fh.write("|---|---|---|---|---|\n")
        for disc, h, n, bl, bv in rows:
            fh.write(f"| {disc} | {h} | {n} | {bl:.3f} | {bv:.3f} |\n")
        fh.write(f"\nmean_b_log= {np.mean(logs):.4f}\n")
        fh.write(f"range_b_log= {min(logs):.4f} {max(logs):.4f}\n")
        fh.write(f"mean_b_level= {np.mean(levs):.4f}\n")
        fh.write(f"range_b_level= {min(levs):.4f} {max(levs):.4f}\n")
    for disc, h, n, bl, bv in rows:
        print(f"  {disc:<14} h={h:<3} n={n:<6} b_log={bl:.3f}  b_level={bv:.3f}")
    print(f"\n  mean b (log, Eq. 1)   {np.mean(logs):.3f}  range {min(logs):.3f}-{max(logs):.3f}")
    print(f"  mean b (level, FAM 3) {np.mean(levs):.3f}  range {min(levs):.3f}-{max(levs):.3f}")
    print(f"  -> {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
