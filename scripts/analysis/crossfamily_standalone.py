"""P0-3(a) — Cross-family table repair: per-family STANDALONE quality columns.

The committed crossfamily_llm.{csv,md} reports only the M1 increment (rel% / clustered
DM) and hides the capability confound: Yi-1.5-34B-Chat is demonstrably broken AT THE
TASK (standalone variance-unit QLIKE 5.25-8.19 vs Qwen3-32B 0.93-1.32; severe forecast
mode-collapse), so "the increment does not replicate in Yi" is unidentified between
"family-specific signal" and "Yi is too weak to attempt the task".

This script adds, per (disclosure x family x horizon), computed on the TEST split of
each run's predictions.parquet (seed2026, single decode):
  - standalone QLIKE in vol units q(y,f) and variance units q(y^2,f^2)
    (variance-unit cross-checked against the stored metrics.json to 1e-3),
  - R^2 on realised vol (metrics.json convention),
  - prediction sd and the mode-collapse diagnostics: n_unique of round(pred,2),
    modal rounded value and its share of test predictions,
and merges the existing M1 columns from crossfamily_llm.csv (unchanged numbers).

Yi long_form rows are FLAGGED 4K-CONTEXT-TRUNCATED (Yi ctx 4K < excerpt cap ~6K prompt
tokens; Qwen3 ran 8K) and must not be cited for the family claim; Yi combined rows are
flagged PARTIAL (contain the truncated long-form subset). The paper may cite only
event_driven (median 8-K ~930 tokens, context not binding) for the cross-family claim,
and the claim wording is downgraded to "does not replicate in the one additional family
tested; capability-confounded at n=2" — NOT "family-specific".

Run from repo root:  .venv/bin/python scripts/analysis/crossfamily_standalone.py
Outputs (NEW files): results/tables/crossfamily_standalone.{csv,md}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/analysis")
import forecast_combination as fc  # noqa: E402

DISCS = ("event_driven", "long_form", "combined")
FAMS = (("qwen3_32b", "C6_llmtext"), ("yi_34b", "C6_llmtext_yi34"))
HORIZONS = (5, 10, 20)


def standalone_stats(y, f):
    y = np.asarray(y, float)
    f = np.asarray(f, float)
    vals, counts = np.unique(np.round(f, 2), return_counts=True)
    i = int(np.argmax(counts))
    return {
        "qlike_vol": float(fc.qlike(y, f).mean()),
        "qlike_var": float(fc.qlike(y ** 2, f ** 2).mean()),
        "r2": float(1.0 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "pred_sd": float(f.std()),
        "n_unique_2dp": int(len(vals)),
        "mode_val_2dp": float(vals[i]),
        "mode_share_pct": float(100.0 * counts[i] / len(f)),
    }


def flag(fam, disc):
    if fam == "yi_34b" and disc == "long_form":
        return "4K-TRUNCATED"
    if fam == "yi_34b" and disc == "combined":
        return "PARTIAL(4K)"
    return "-"


def main():
    rows, sanity_bad = [], []
    for disc in DISCS:
        for fam, run in FAMS:
            p = pd.read_parquet(
                f"results/runs/{run}_full_{disc}_seed2026/predictions.parquet")
            mj = {(r["split"], r["horizon_days"]): r for r in json.load(
                open(f"results/runs/{run}_full_{disc}_seed2026/metrics.json"))}
            te = p[p.split == "test"]
            for h in HORIZONS:
                d = te[te.horizon_days == h]
                st = standalone_stats(d.label_realised_vol.to_numpy(),
                                      d.prediction_realised_vol.to_numpy())
                stored = mj[("test", h)]["qlike"]
                ok = abs(st["qlike_var"] - stored) <= 1e-3 * max(abs(stored), 1.0)
                if not ok:
                    sanity_bad.append((disc, fam, h, st["qlike_var"], stored))
                rows.append({"disc": disc, "family": fam, "h": h, "n_test": len(d),
                             **st, "qlike_var_metricsjson": float(stored),
                             "metrics_sanity": "PASS" if ok else "FAIL",
                             "context_flag": flag(fam, disc)})
    df = pd.DataFrame(rows)

    # merge the existing (unchanged) M1 columns
    m1 = pd.read_csv("results/tables/crossfamily_llm.csv")[
        ["disc", "family", "h", "rel_har", "dm_har", "p_har",
         "rel_firm", "dm_firm", "p_firm"]]
    df = df.merge(m1, on=["disc", "family", "h"], how="left")
    df.to_csv("results/tables/crossfamily_standalone.csv", index=False)

    q = df[df.family == "qwen3_32b"]
    y = df[df.family == "yi_34b"]
    yed = y[y.disc == "event_driven"]
    qed = q[q.disc == "event_driven"]

    md = [
        "# Cross-family table REPAIR — standalone quality + mode-collapse diagnostics (P0-3a)",
        "",
        "## RESTATED vs BEFORE",
        "",
        "| | BEFORE (crossfamily_llm.md) | RESTATED (this table) |",
        "|---|---|---|",
        "| headline | \"the prompted-LLM residual is **family-specific**\" | \"the increment **does not"
        " replicate in the one additional family tested (Yi-1.5-34B)**; at n=2 with the second family"
        " capability-floored at the task, family-specificity is **unidentified** (capability-confounded)\" |",
        "| standalone quality | absent (hidden confound) | per-cell test QLIKE (vol + variance unit),"
        " R^2, prediction sd, n_unique / modal-share of round(pred,2) |",
        f"| Yi capability floor | invisible | variance-unit QLIKE {y.qlike_var.min():.2f}-"
        f"{y.qlike_var.max():.2f} vs Qwen {q.qlike_var.min():.2f}-{q.qlike_var.max():.2f}; "
        f"event_driven h=5 mode-collapse: Yi {yed[yed.h == 5].mode_share_pct.iloc[0]:.1f}% of test"
        f" predictions at {yed[yed.h == 5].mode_val_2dp.iloc[0]:.2f} vs Qwen "
        f"{qed[qed.h == 5].mode_share_pct.iloc[0]:.1f}% at {qed[qed.h == 5].mode_val_2dp.iloc[0]:.2f} |",
        "| Yi long_form rows | in the main table | **demoted: 4K-context-truncated** (Yi ctx 4K <"
        " ~6K-token excerpt cap; Qwen ran 8K) — cite only event_driven (median 8-K ~930 tokens,"
        " context not binding) for the family claim |",
        "",
        "M1 columns (rel% / day-clustered DM vs single recalibrated HAR and vs the firm-identity-"
        "augmented reference) are carried over UNCHANGED from crossfamily_llm.csv; `**` = clustered"
        " DM<0, p<.05. Standalone columns are computed on the TEST split of each run's"
        " predictions.parquet; the variance-unit QLIKE is cross-checked against the stored"
        " metrics.json (sanity column). `combined` M1 cells were not part of the original"
        " cross-family grid (blank).",
        "",
        "| disc | family | h | n_test | QLIKE(vol) | QLIKE(var) | R^2 | pred sd | n_uniq(2dp) |"
        " mode@2dp | mode share% | ctx flag | rel% vs HAR | DM(clu) | rel% vs HAR+firmID | DM(clu) |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|--:|--:|--:|",
    ]
    for _, r in df.iterrows():
        if pd.notna(r.rel_har):
            s1 = "**" if (r.dm_har < 0 and r.p_har < .05) else ""
            s2 = "**" if (r.dm_firm < 0 and r.p_firm < .05) else ""
            m1c = (f"{r.rel_har:+.2f}%{s1} | {r.dm_har:+.2f} | "
                   f"{r.rel_firm:+.2f}%{s2} | {r.dm_firm:+.2f}")
        else:
            m1c = "- | - | - | -"
        md.append(
            f"| {r.disc} | {r.family} | {int(r.h)} | {int(r.n_test)} | {r.qlike_vol:.4f} | "
            f"{r.qlike_var:.3f} | {r.r2:+.3f} | {r.pred_sd:.4f} | {int(r.n_unique_2dp)} | "
            f"{r.mode_val_2dp:.2f} | {r.mode_share_pct:.1f} | {r.context_flag} | {m1c} |")

    md += [
        "",
        "## Honest reading (replaces the \"family-specific\" headline)",
        "",
        "- **The Qwen3-32B increment does not replicate in Yi-1.5-34B** on event_driven, the only"
        " disclosure where the comparison is context-clean (all Yi rel% ~0, no cell significant).",
        "- **But the comparison is capability-confounded at n=2**: Yi's standalone forecasts are"
        f" broken at the task (variance-unit QLIKE {y.qlike_var.min():.2f}-{y.qlike_var.max():.2f}"
        f" vs Qwen {q.qlike_var.min():.2f}-{q.qlike_var.max():.2f}; R^2"
        f" {y.r2.min():+.2f}-{y.r2.max():+.2f} vs Qwen {q.r2.min():+.2f}-{q.r2.max():+.2f}; up to"
        f" {y.mode_share_pct.max():.1f}% of Yi test predictions collapse onto a single rounded"
        " value). A model that cannot produce a calibrated standalone forecast cannot be evidence"
        " that the *signal* is family-specific — only that it fails to replicate in this family.",
        "- **Correct claim for the paper**: \"the prompted-LLM increment does not replicate in a"
        " second model family; because the second family is capability-floored at the task, the"
        " test is confounded and family-specificity remains unidentified at n=2.\" Do NOT write"
        " \"family-specific\".",
        "- **Yi long_form (and the long-form subset of combined) is additionally 4K-context-"
        "truncated** and must not be cited at all for the family claim; the citable cells are the"
        " six event_driven rows.",
        "",
        f"Sanity: recomputed variance-unit QLIKE matches metrics.json within 1e-3 relative in"
        f" {int((df.metrics_sanity == 'PASS').sum())}/{len(df)} cells"
        + ("." if not sanity_bad else f"; FAILURES: {sanity_bad}"),
        "",
    ]
    Path("results/tables/crossfamily_standalone.md").write_text("\n".join(md))
    print(f"wrote results/tables/crossfamily_standalone.csv/.md  "
          f"({len(df)} cells, sanity fails={len(sanity_bad)})")
    print(df[["disc", "family", "h", "qlike_vol", "qlike_var", "r2", "pred_sd",
              "n_unique_2dp", "mode_val_2dp", "mode_share_pct", "metrics_sanity"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
