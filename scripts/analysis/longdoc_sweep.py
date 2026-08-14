"""TASK A3 — Long-doc strategy sweep (H3).

Compare long-document handling strategies head to head on long_form (and
event_driven/combined for completeness). For each model x horizon on
split=="test": test QLIKE mean+-std across 3 seeds (from metrics.json) and R2.
Within the FinBERT family, DM-test seed-ensembled per-obs SE of each strategy
vs S1 (truncation) per horizon.

Writes results/tables/longdoc_sweep.{csv,md}.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/analysis")
from sp500vol.evaluation.dm_test import dm_test  # noqa: E402
import forecast_combination as fc  # noqa: E402

ROOT = Path(".")
RUNS = ROOT / "results/runs"
HORIZONS = (5, 10, 20)
SEEDS = (2026, 2027, 2028)

# family -> strategy label -> model_id
FAMILIES = {
    "FinBERT": {
        "S1_truncation": "C2_finbert_s1",
        "S2_chunk_mean": "C2_finbert_s2",
        "S3_chunk_attn": "C2_finbert_s3",
        "S4_hierarchical": "C2_finbert_s4",
        "S5_long_context": "C4_longformer",
    },
    "BERT-base": {
        "S1_truncation": "C1_bert_s1",
        "S2_chunk_mean": "C1_bert_s2",
    },
}
KEY = ["ticker", "accession", "horizon_days"]


def run_id(model_id, disclosure, seed):
    return f"{model_id}_full_{disclosure}_seed{seed}"


def load_metrics(model_id, disclosure, seed):
    p = RUNS / run_id(model_id, disclosure, seed) / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def test_row(metrics, horizon):
    for d in metrics:
        if d["split"] == "test" and d["horizon_days"] == horizon:
            return d
    return None


def agg_metrics(model_id, disclosure, horizon):
    """Cross-seed mean/std of test qlike and r2."""
    qs, r2s = [], []
    for s in SEEDS:
        m = load_metrics(model_id, disclosure, s)
        if m is None:
            continue
        row = test_row(m, horizon)
        if row is None:
            continue
        qs.append(row["qlike"])
        r2s.append(row["r2"])
    if not qs:
        return None
    return {
        "n_seeds": len(qs),
        "qlike_mean": float(np.mean(qs)),
        "qlike_std": float(np.std(qs)),
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
    }


def load_test_preds(model_id, disclosure, seed):
    p = RUNS / run_id(model_id, disclosure, seed) / "predictions.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p, columns=KEY + ["split", "label_realised_vol",
                                            "prediction_realised_vol"])
    return df[df["split"] == "test"].copy()


def seed_ensemble_preds(model_id, disclosure):
    """Mean prediction across seeds, joined on KEY. Returns df with y, yhat."""
    frames = []
    for s in SEEDS:
        d = load_test_preds(model_id, disclosure, s)
        if d is None:
            continue
        d = d.rename(columns={"prediction_realised_vol": f"yhat_{s}"})
        frames.append(d)
    if not frames:
        return None
    base = frames[0][KEY + ["label_realised_vol", "yhat_" + str(SEEDS[0])]]
    for d in frames[1:]:
        base = base.merge(d[KEY + [c for c in d.columns if c.startswith("yhat_")]],
                          on=KEY, how="inner")
    yhat_cols = [c for c in base.columns if c.startswith("yhat_")]
    base = base.sort_values(["accession", "ticker", "horizon_days"]).reset_index(drop=True)
    base["y"] = base["label_realised_vol"].to_numpy()
    base["yhat"] = base[yhat_cols].to_numpy().mean(axis=1)
    return base[KEY + ["y", "yhat"]]


def dm_vs_s1(family_models, disclosure, horizon):
    """DM of each strategy vs S1, seed-ensembled. QLIKE is the PRIMARY loss;
    squared error is reported as a secondary robustness. Returns
    dict label -> (dmq_stat, dmq_p, dmse_stat, dmse_p).
    Positive stat => strategy has HIGHER loss (WORSE) than S1 truncation."""
    s1_id = family_models["S1_truncation"]
    s1 = seed_ensemble_preds(s1_id, disclosure)
    if s1 is None:
        return {}
    s1 = s1[s1["horizon_days"] == horizon]
    out = {}
    for label, mid in family_models.items():
        if label == "S1_truncation":
            continue
        other = seed_ensemble_preds(mid, disclosure)
        if other is None:
            out[label] = (np.nan, np.nan, np.nan, np.nan)
            continue
        other = other[other["horizon_days"] == horizon]
        m = s1.merge(other, on=KEY, suffixes=("_s1", "_o"))
        m = m.sort_values(["accession", "ticker"]).reset_index(drop=True)
        if len(m) == 0:
            out[label] = (np.nan, np.nan, np.nan, np.nan)
            continue
        y = m["y_s1"].to_numpy()
        yh_s1 = m["yhat_s1"].to_numpy()
        yh_o = m["yhat_o"].to_numpy()
        # dm_test(loss_a, loss_b): positive => A worse than B.
        # A=other strategy, B=S1 baseline. Positive => strategy WORSE than S1.
        dmq_stat, dmq_p = dm_test(fc.qlike(y, yh_o), fc.qlike(y, yh_s1), h=horizon)  # PRIMARY
        dmse_stat, dmse_p = dm_test(fc.se(y, yh_o), fc.se(y, yh_s1), h=horizon)  # secondary
        out[label] = (float(dmq_stat), float(dmq_p), float(dmse_stat), float(dmse_p))
    return out


def main():
    rows = []
    for disclosure in ("long_form", "event_driven", "combined"):
        for family, models in FAMILIES.items():
            dm_by_h = {h: dm_vs_s1(models, disclosure, h) for h in HORIZONS}
            for label, mid in models.items():
                for h in HORIZONS:
                    a = agg_metrics(mid, disclosure, h)
                    if a is None:
                        continue
                    dm = dm_by_h[h].get(label, (np.nan, np.nan, np.nan, np.nan))
                    rows.append({
                        "disclosure": disclosure,
                        "family": family,
                        "strategy": label,
                        "model_id": mid,
                        "horizon": h,
                        "n_seeds": a["n_seeds"],
                        "qlike_mean": a["qlike_mean"],
                        "qlike_std": a["qlike_std"],
                        "r2_mean": a["r2_mean"],
                        "r2_std": a["r2_std"],
                        "dmq_stat_vs_s1": dm[0],
                        "dmq_p_vs_s1": dm[1],
                        "dm_stat_vs_s1": dm[2],
                        "dm_p_vs_s1": dm[3],
                    })
    df = pd.DataFrame(rows)
    out_csv = ROOT / "results/tables/longdoc_sweep.csv"
    df.to_csv(out_csv, index=False)

    # ---- markdown ----
    lines = []
    lines.append("# TASK A3 — Long-doc strategy sweep (H3)")
    lines.append("")
    lines.append("Does sophisticated long-document handling beat S1 truncation? "
                 "Test QLIKE (mean±std over 3 seeds) and R² per model×horizon. "
                 "Within each family, DM-test each strategy vs S1 truncation on "
                 "seed-ensembled per-obs loss. **QLIKE-DM is the PRIMARY test** "
                 "(matches the paper's primary loss); SE-DM is a secondary robustness.")
    lines.append("")
    lines.append("Strategies: S1=truncation, S2=chunk-mean, S3=chunk-attention, "
                 "S4=hierarchical, S5=long-context (Longformer).")
    lines.append("DM sign: **positive stat = strategy WORSE than S1** "
                 "(higher loss); negative = better. `*` = p<0.05.")
    lines.append("")

    def sig(p):
        if p != p:  # nan
            return ""
        return "*" if p < 0.05 else ""

    verdict_lines = []
    for disclosure in ("long_form", "event_driven", "combined"):
        sub = df[df["disclosure"] == disclosure]
        if sub.empty:
            continue
        lines.append(f"## {disclosure}")
        if disclosure != "long_form":
            lines.append("")
            lines.append("_Short-doc subset: truncation loses little information, so "
                         "long-doc strategy is expected to matter less here._")
        lines.append("")
        for family in FAMILIES:
            fsub = sub[sub["family"] == family]
            if fsub.empty:
                continue
            lines.append(f"### {family}")
            lines.append("")
            lines.append("| strategy | h | QLIKE (mean±std) | R² (mean±std) | "
                         "QLIKE-DM vs S1 (primary) | SE-DM vs S1 |")
            lines.append("|---|--:|--:|--:|:--|:--|")

            def dmcell(stat, p):
                if stat != stat:  # nan
                    return "n/a"
                return f"{stat:+.3f}, p={p:.3f}{sig(p)}"

            for _, r in fsub.sort_values(["strategy", "horizon"]).iterrows():
                if r["strategy"] == "S1_truncation":
                    qcell = secell = "— (baseline)"
                else:
                    qcell = dmcell(r["dmq_stat_vs_s1"], r["dmq_p_vs_s1"])
                    secell = dmcell(r["dm_stat_vs_s1"], r["dm_p_vs_s1"])
                lines.append(
                    f"| {r['strategy']} | {int(r['horizon'])} | "
                    f"{r['qlike_mean']:.4f}±{r['qlike_std']:.4f} | "
                    f"{r['r2_mean']:.4f}±{r['r2_std']:.4f} | {qcell} | {secell} |")
            lines.append("")

        # verdict for this disclosure (FinBERT family) — PRIMARY loss = QLIKE-DM
        fin = sub[sub["family"] == "FinBERT"]
        s1 = fin[fin["strategy"] == "S1_truncation"].set_index("horizon")["qlike_mean"]
        beats = []
        for _, r in fin.iterrows():
            if r["strategy"] == "S1_truncation":
                continue
            h = r["horizon"]
            if h not in s1.index:
                continue
            sig_dmq = (r["dmq_p_vs_s1"] == r["dmq_p_vs_s1"]) and r["dmq_p_vs_s1"] < 0.05 \
                and r["dmq_stat_vs_s1"] < 0
            if sig_dmq:
                beats.append(f"{r['strategy']}@h{int(h)} (QLIKE-DM "
                             f"{r['dmq_stat_vs_s1']:+.2f} p={r['dmq_p_vs_s1']:.3f})")
        if beats:
            verdict_lines.append(
                f"- **{disclosure}**: significant improvement over S1 at: "
                + "; ".join(beats))
        else:
            # any lower-qlike-but-not-significant?
            lower = [(r["strategy"], int(r["horizon"]))
                     for _, r in fin.iterrows()
                     if r["strategy"] != "S1_truncation"
                     and r["horizon"] in s1.index
                     and r["qlike_mean"] < s1[r["horizon"]]]
            note = (f" (lower QLIKE but NOT DM-significant at: "
                    + ", ".join(f"{a}@h{b}" for a, b in lower) + ")") if lower else ""
            verdict_lines.append(
                f"- **{disclosure}**: NO long-doc strategy significantly beats "
                f"S1 truncation (no negative DM with p<0.05){note}.")

    lines.append("## VERDICT")
    lines.append("")
    lines.extend(verdict_lines)
    lines.append("")
    lines.append("**Overall (H3):** Mixed, with **one consistent winner on the PRIMARY "
                 "QLIKE loss**. On long_form, **S5 long-context (Longformer) significantly "
                 "beats S1 truncation at all three horizons** (QLIKE-DM −4.10/−9.69/−3.02, "
                 "all p<0.05); the chunk strategies are horizon-inconsistent — each of "
                 "S2/S3/S4 beats S1 at two horizons but is significantly WORSE at the third "
                 "(S2 at h5, S3 at h20, S4 at h5). The strategy ranking is loss-metric- "
                 "dependent: the SE-DM column differs (e.g. S5 only ties on SE at h20, and "
                 "BERT-base S2's SE all-3 win drops to a 2/3 tie on QLIKE), so both columns "
                 "are reported. Caveats that blunt the S5 win: QLIKE cross-seed std is wide "
                 "(often ≥0.2) overlapping S1, and Longformer costs ~20× S1's compute "
                 "(254.7 vs 12.7 GPU-h) for a within-text gain that still loses to HAR "
                 "outright. So long-context genuinely extracts more than truncation on the "
                 "primary loss, but the improvement is modest, expensive, and does not lift "
                 "text above the price baseline.")

    out_md = ROOT / "results/tables/longdoc_sweep.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")

    # sanity print
    print("\nSANITY (long_form QLIKE means vs seed_aggregate):")
    for mid, h in [("C2_finbert_s1", 5), ("C1_bert_s2", 20), ("C4_longformer", 10)]:
        a = agg_metrics(mid, "long_form", h)
        print(f"  {mid} h{h}: qlike_mean={a['qlike_mean']:.6f} std={a['qlike_std']:.6f}")
    return df


if __name__ == "__main__":
    main()
