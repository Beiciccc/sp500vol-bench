"""TASK A2 — Pairwise Diebold-Mariano significance matrix on per-observation squared
error (fc.se), across the key model set, per (disclosure, horizon), on split=="test".

Model set: A2_har_rv, A3_garch, A4_egarch, A5_arima, best B (B2_tfidf_ridge), and every
C and D model. 3-seed models (C*, D*) are SEED-ENSEMBLED: the per-observation prediction
is averaged across the 3 seeds (2026/2027/2028) BEFORE computing SE. A/B baselines are
seed-invariant (only seed2026 exists).

For each ordered pair (challenger, baseline) within a (disclosure, horizon) group:
    dm_test(se_challenger, se_baseline, h=horizon) -> (stat, p)
    POSITIVE stat  => challenger has HIGHER loss (WORSE) than baseline.
    NEGATIVE stat  => challenger LOWER loss (BETTER).
Holm correction (fc.holm) is applied to p-values WITHIN each (disclosure, horizon) group.
A negative stat with holm-p<0.05 => challenger significantly BETTER.

All models are joined on KEY=[ticker, accession, horizon_days] (inner join across the
whole model set within a disclosure) so every pair is compared on an identical sample.

Outputs (NEW files only):
  results/tables/dm_pairwise.csv  — full pairwise long form
  results/tables/dm_pairwise.md   — vs-HAR and vs-best readable summary
Run from repo root:  .venv/bin/python scripts/analysis/dm_pairwise.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/analysis")
import forecast_combination as fc  # se(), holm()
sys.path.insert(0, "src")
from sp500vol.evaluation.dm_test import dm_test

KEY = ["ticker", "accession", "horizon_days"]
SORT = ["filing_time_utc", "ticker", "accession"]
HORIZONS = (5, 10, 20)
DISCLOSURES = ("long_form", "event_driven", "combined")
SEEDS_3 = (2026, 2027, 2028)

# Seed-invariant (single seed2026 run) models.
SEED_INVARIANT = {
    "A2_har_rv", "A3_garch", "A4_egarch", "A5_arima",
    "B1_bow_ridge", "B2_tfidf_ridge", "B3_lm_linear", "B4_lm_features",
}
# 3-seed neural models — seed-ensembled point forecast.
MULTI_SEED = [
    "C1_bert_s1", "C1_bert_s2", "C2_finbert_s1", "C2_finbert_s2", "C2_finbert_s3",
    "C2_finbert_s4", "C3_roberta_s1", "C4_longformer", "C5_qwen3", "C5_gteqwen2",
    "C5_e5mistral", "D1_concat_mlp", "D2_gated_fusion", "D3_qwen3", "D3_gteqwen2",
    "D3_e5mistral",
]
# Classical / baseline model set (best B = B2_tfidf_ridge).
BASELINE_MODELS = ["A2_har_rv", "A3_garch", "A4_egarch", "A5_arima", "B2_tfidf_ridge"]
MODEL_SET = BASELINE_MODELS + MULTI_SEED
HAR = "A2_har_rv"


def _run_path(model, disc, seed):
    return Path(f"results/runs/{model}_full_{disc}_seed{seed}/predictions.parquet")


def load_point(model, disc):
    """Return DataFrame with KEY + label + prediction + filing_time_utc for split==test.
    Multi-seed models: average prediction across the 3 seeds on KEY (seed-ensembled)."""
    if model in SEED_INVARIANT:
        p = _run_path(model, disc, 2026)
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        df = df[df.split == "test"]
        return df[KEY + ["label_realised_vol", "prediction_realised_vol", "filing_time_utc"]]
    # multi-seed
    frames = []
    for s in SEEDS_3:
        p = _run_path(model, disc, s)
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        d = d[d.split == "test"]
        frames.append(d[KEY + ["label_realised_vol", "prediction_realised_vol", "filing_time_utc"]])
    if not frames:
        return None
    cat = pd.concat(frames, ignore_index=True)
    agg = (cat.groupby(KEY, as_index=False)
              .agg(label_realised_vol=("label_realised_vol", "first"),
                   prediction_realised_vol=("prediction_realised_vol", "mean"),
                   filing_time_utc=("filing_time_utc", "first")))
    return agg


def build_joined(disc):
    """Inner-join all available models in MODEL_SET on KEY. Returns (merged_df, present_models).
    merged_df has: KEY, filing_time_utc, label_realised_vol, and pred__<model> per model."""
    present = []
    merged = None
    for m in MODEL_SET:
        d = load_point(m, disc)
        if d is None:
            continue
        present.append(m)
        sub = d[KEY + ["prediction_realised_vol"]].rename(
            columns={"prediction_realised_vol": f"pred__{m}"})
        if merged is None:
            base = d[KEY + ["label_realised_vol", "filing_time_utc"]].copy()
            merged = base.merge(sub, on=KEY)
        else:
            merged = merged.merge(sub, on=KEY)
    return merged, present


def main():
    rows = []
    for disc in DISCLOSURES:
        merged, present = build_joined(disc)
        if merged is None or not present:
            continue
        for h in HORIZONS:
            g = merged[merged.horizon_days == h].sort_values(SORT, kind="mergesort")
            if len(g) < 30:
                continue
            y = g.label_realised_vol.to_numpy()
            # per-obs SE for each present model on the common sample
            se_map = {m: fc.se(y, g[f"pred__{m}"].to_numpy()) for m in present}
            qlike_map = {m: float(np.mean(fc.qlike(y, g[f"pred__{m}"].to_numpy()))) for m in present}
            # all ordered pairs (challenger != baseline)
            group_rows = []
            for ch in present:
                for bl in present:
                    if ch == bl:
                        continue
                    stat, p = dm_test(se_map[ch], se_map[bl], h=h)
                    group_rows.append([disc, h, ch, bl, float(stat), float(p)])
            gdf = pd.DataFrame(group_rows,
                               columns=["disclosure", "horizon", "challenger", "baseline",
                                        "dm_stat", "p_raw"])
            gdf["p_holm"] = fc.holm(gdf.p_raw.fillna(1.0).to_numpy())
            gdf["better"] = (gdf.dm_stat < 0) & (gdf.p_holm < 0.05)
            gdf["_qlike_ch"] = gdf.challenger.map(qlike_map)
            gdf["_qlike_bl"] = gdf.baseline.map(qlike_map)
            gdf["_best_model"] = min(qlike_map, key=qlike_map.get)
            gdf["_n"] = len(g)
            rows.append(gdf)

    if not rows:
        print("no cells"); return None
    full = pd.concat(rows, ignore_index=True)

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    out_cols = ["disclosure", "horizon", "challenger", "baseline",
                "dm_stat", "p_raw", "p_holm", "better"]
    full[out_cols].to_csv("results/tables/dm_pairwise.csv", index=False)

    # ---- readable md: vs-HAR and vs-best per (disclosure, horizon) ----
    md = ["# Pairwise Diebold-Mariano significance matrix (per-obs squared error)\n",
          "DM on per-observation squared error `fc.se` on split==test, joined on "
          "KEY=[ticker,accession,horizon_days] (inner join across the full model set within "
          "each disclosure, so all pairs share one sample). 3-seed models (C*/D*) are "
          "**seed-ensembled**: per-observation prediction averaged across seeds "
          "{2026,2027,2028} BEFORE squared error. A/B are seed-invariant (seed2026 only).\n",
          "**Sign convention:** dm_stat>0 => challenger has HIGHER loss (WORSE) than baseline; "
          "dm_stat<0 => challenger BETTER. Holm applied WITHIN each (disclosure,horizon) group "
          "over all ordered pairs. `sig better` = dm_stat<0 AND p_holm<0.05.\n"]

    for disc in DISCLOSURES:
        sub = full[full.disclosure == disc]
        if sub.empty:
            continue
        md.append(f"\n## {disc}\n")
        for h in HORIZONS:
            hh = sub[sub.horizon == h]
            if hh.empty:
                continue
            best = hh._best_model.iloc[0]
            n = int(hh._n.iloc[0])
            md.append(f"\n### horizon = {h} days (n={n}, best-QLIKE model = **{best}**)\n")
            # vs HAR
            md.append("**Challenger vs A2_har_rv** (does the challenger beat the HAR price baseline?)\n")
            md.append("| challenger | dm_stat | p_raw | p_holm | verdict |\n|---|---|---|---|---|")
            vh = hh[hh.baseline == HAR].sort_values("dm_stat")
            for _, r in vh.iterrows():
                v = "BETTER*" if r.better else ("worse" if r.dm_stat > 0 else "better(ns)")
                md.append(f"| {r.challenger} | {r.dm_stat:+.3f} | {r.p_raw:.4f} | {r.p_holm:.4f} | {v} |")
            # vs best
            md.append(f"\n**Challenger vs best model ({best})** (who is statistically indistinct from / beats the best?)\n")
            md.append("| challenger | dm_stat | p_raw | p_holm | verdict |\n|---|---|---|---|---|")
            vb = hh[hh.baseline == best].sort_values("dm_stat")
            for _, r in vb.iterrows():
                if r.challenger == best:
                    continue
                # here challenger vs best: dm_stat>0 => challenger worse than best
                if r.dm_stat > 0 and r.p_holm < 0.05:
                    v = "sig WORSE than best"
                elif r.better:
                    v = "sig BETTER than best*"
                else:
                    v = "indistinct from best"
                md.append(f"| {r.challenger} | {r.dm_stat:+.3f} | {r.p_raw:.4f} | {r.p_holm:.4f} | {v} |")

    # sanity: text models vs A2 on SE — all should be WORSE (dm_stat>0)
    text_models = [m for m in MULTI_SEED]
    vsA2 = full[(full.baseline == HAR) & (full.challenger.isin(text_models))]
    n_worse = int((vsA2.dm_stat > 0).sum())
    n_tot = int(len(vsA2))
    md.append(f"\n## Sanity — text/neural models vs A2_har_rv on SE\n")
    md.append(f"- {n_worse}/{n_tot} (challenger, A2) cells have dm_stat>0 (challenger WORSE than HAR on SE), "
              f"matching seed_aggregate.md (all text models WORSE than HAR on SE).\n")

    with open("results/tables/dm_pairwise.md", "w") as fh:
        fh.write("\n".join(md))

    print("=== dm_pairwise done ===")
    print(f"pairs={len(full)} disclosures={sorted(full.disclosure.unique())} "
          f"horizons={sorted(full.horizon.unique())}")
    print(f"sanity text-vs-A2 WORSE: {n_worse}/{n_tot}")
    return full, n_worse, n_tot


if __name__ == "__main__":
    main()
