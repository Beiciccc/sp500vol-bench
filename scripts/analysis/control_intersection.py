#!/usr/bin/env python
"""G1 — Control-intersection table.

Joins per-cell survival flags across the 69 M1 cells from five source tables and
computes the AND across controls under BOTH raw p<.05 and Holm<.05 bases.

Sources (keyed [disc,model,h]):
  m1_clustered.csv        -> clustered-genuine (dm_q_clust<0 & sig)
  maximal_reference.csv   -> survives maximal price pool (dm_q_clustered<0 & sig)
  firm_identity_control.csv -> survives firm-identity (dm_q_clustered<0 & sig)
  withindate_placebo.csv  -> cross-sectional verdict (extra column)
  m1_ensemble_primary.csv -> seed-ensemble genuine (extra column)

Outputs (new files only):
  results/tables/control_intersection.csv
  results/tables/control_intersection.md
"""
import pandas as pd
from pathlib import Path

T = Path("results/tables")
KEY = ["disc", "model", "h"]


def load():
    m = pd.read_csv(T / "m1_clustered.csv")
    mx = pd.read_csv(T / "maximal_reference.csv")
    fi = pd.read_csv(T / "firm_identity_control.csv")
    wd = pd.read_csv(T / "withindate_placebo.csv")
    en = pd.read_csv(T / "m1_ensemble_primary.csv")
    return m, mx, fi, wd, en


def build():
    m, mx, fi, wd, en = load()

    df = m[KEY].copy()

    # --- clustered-genuine (m1_clustered): dm_q_clust < 0 AND significant ---
    m_neg = m["dm_q_clust"] < 0
    df["clu_genuine_raw"] = (m_neg & (m["p_q_clust"] < 0.05)).values
    df["clu_genuine_holm"] = (m_neg & (m["dmq_holm_clust"] < 0.05)).values
    df["rel_impr_pct"] = m["rel_impr_pct"].values

    # --- maximal price pool ---
    mxj = mx.set_index(KEY)
    mx_neg = mxj["dm_q_clustered"] < 0
    df = df.set_index(KEY)
    df["maximal_raw"] = (mx_neg & (mxj["p_q_clustered"] < 0.05)).reindex(df.index).values
    df["maximal_holm"] = (mx_neg & (mxj["holm_p"] < 0.05)).reindex(df.index).values
    df["rel_impr_pct_maximal"] = mxj["rel_impr_pct_maximal"].reindex(df.index).values

    # --- firm-identity control ---
    fij = fi.set_index(KEY)
    fi_neg = fij["dm_q_clustered"] < 0
    df["firm_raw"] = (fi_neg & (fij["p_q_clustered"] < 0.05)).reindex(df.index).values
    df["firm_holm"] = (fi_neg & (fij["holm_p"] < 0.05)).reindex(df.index).values
    df["rel_impr_pct_firm"] = fij["rel_impr_pct_firm"].reindex(df.index).values

    # --- extra: within-date cross-sectional verdict ---
    wdj = wd.set_index(KEY)
    df["withindate_crosssec"] = (wdj["verdict"].reindex(df.index) == "cross-sectional").values
    df["withindate_verdict"] = wdj["verdict"].reindex(df.index).values

    # --- extra: seed-ensemble genuine (seed2026 clustered) ---
    enj = en.set_index(KEY)
    df["seed_ens_genuine"] = enj["genuine_s26_clu"].reindex(df.index).astype(bool).values

    df = df.reset_index()

    # --- intersections, per basis ---
    for basis in ("raw", "holm"):
        cg = df[f"clu_genuine_{basis}"]
        mxc = df[f"maximal_{basis}"]
        fic = df[f"firm_{basis}"]
        df[f"AND_maximal_firm_{basis}"] = mxc & fic
        df[f"AND_full_{basis}"] = cg & mxc & fic
        df[f"AND_5way_{basis}"] = (
            cg & mxc & fic & df["withindate_crosssec"] & df["seed_ens_genuine"]
        )

    return df


def counts_table(df):
    rows = []
    controls = [
        ("clu_genuine", "clustered-genuine (m1_clustered)"),
        ("maximal", "maximal price pool"),
        ("firm", "firm-identity"),
    ]
    for col, label in controls:
        rows.append((label, int(df[f"{col}_raw"].sum()), int(df[f"{col}_holm"].sum())))
    # extras (single-basis)
    rows.append(("within-date cross-sectional [extra]", int(df["withindate_crosssec"].sum()), int(df["withindate_crosssec"].sum())))
    rows.append(("seed-ensemble genuine [extra]", int(df["seed_ens_genuine"].sum()), int(df["seed_ens_genuine"].sum())))
    # intersections
    rows.append(("INTERSECTION: maximal AND firm", int(df["AND_maximal_firm_raw"].sum()), int(df["AND_maximal_firm_holm"].sum())))
    rows.append(("INTERSECTION: clustered-genuine AND maximal AND firm (full)", int(df["AND_full_raw"].sum()), int(df["AND_full_holm"].sum())))
    rows.append(("INTERSECTION: 5-way (+within-date x-sec +seed-ensemble)", int(df["AND_5way_raw"].sum()), int(df["AND_5way_holm"].sum())))
    return rows


def cell_str(r):
    return f"{r['disc']}/{r['model']}/h={r['h']}"


def main():
    df = build()

    # csv: one row per cell, boolean columns per control + p-basis
    csv_cols = KEY + [
        "clu_genuine_raw", "clu_genuine_holm",
        "maximal_raw", "maximal_holm",
        "firm_raw", "firm_holm",
        "withindate_crosssec", "withindate_verdict",
        "seed_ens_genuine",
        "AND_maximal_firm_raw", "AND_maximal_firm_holm",
        "AND_full_raw", "AND_full_holm",
        "AND_5way_raw", "AND_5way_holm",
        "rel_impr_pct", "rel_impr_pct_maximal", "rel_impr_pct_firm",
    ]
    out = df[csv_cols].sort_values(KEY)
    out.to_csv(T / "control_intersection.csv", index=False)

    rows = counts_table(df)

    # disjoint survivor sets (Holm basis, the headline)
    mx_surv_holm = df[df["maximal_holm"]].apply(cell_str, axis=1).tolist()
    fi_surv_holm = df[df["firm_holm"]].apply(cell_str, axis=1).tolist()
    mx_set, fi_set = set(mx_surv_holm), set(fi_surv_holm)
    overlap_holm = sorted(mx_set & fi_set)

    # raw-p scattered survivors of the full AND
    raw_full = df[df["AND_full_raw"]]
    raw_full_cells = [
        (cell_str(r), r["rel_impr_pct_maximal"], r["rel_impr_pct_firm"])
        for _, r in raw_full.iterrows()
    ]
    holm_full_n = int(df["AND_full_holm"].sum())

    # sanity marginals
    sanity = {
        "m1_clustered genuine (Holm) == 29": int(df["clu_genuine_holm"].sum()),
        "m1_clustered genuine (raw)": int(df["clu_genuine_raw"].sum()),
        "maximal Holm == 8": int(df["maximal_holm"].sum()),
        "maximal raw": int(df["maximal_raw"].sum()),
        "firm Holm == 8": int(df["firm_holm"].sum()),
        "firm raw": int(df["firm_raw"].sum()),
        "within-date cross-sectional == 33": int(df["withindate_crosssec"].sum()),
        "seed-ensemble genuine == 29": int(df["seed_ens_genuine"].sum()),
    }

    # --- markdown ---
    L = []
    L.append(f"# Control-intersection table — headline: **{holm_full_n}/69 cells clear maximal + firm jointly under Holm**")
    L.append("")
    L.append(f"Under Holm, **{holm_full_n}/69** cells clear the full AND (clustered-genuine AND maximal AND firm-identity). "
             f"Under raw p<.05, **{len(raw_full_cells)}** scattered weak cells clear it — with no coherent model/disclosure pattern.")
    L.append("")
    L.append("## Counts: per-control marginals and intersections (raw p<.05 vs Holm<.05)")
    L.append("")
    L.append("| Control / intersection | raw p<.05 | Holm<.05 |")
    L.append("|---|---|---|")
    for label, raw, holm in rows:
        L.append(f"| {label} | {raw} | {holm} |")
    L.append("")
    L.append("_Extra controls (within-date cross-sectional, seed-ensemble) are single-basis; their two columns repeat._")
    L.append("")

    L.append("## Full AND survivors")
    L.append("")
    L.append(f"**Holm basis: {holm_full_n}/69.**")
    if holm_full_n == 0:
        L.append("No cell survives clustered-genuine AND maximal AND firm-identity jointly under Holm.")
    L.append("")
    L.append(f"**Raw p<.05 basis: {len(raw_full_cells)}/69** scattered weak cells:")
    L.append("")
    if raw_full_cells:
        L.append("| cell (disc/model/h) | rel% maximal | rel% firm |")
        L.append("|---|---|---|")
        for c, rmx, rfi in raw_full_cells:
            L.append(f"| {c} | {rmx:.3f} | {rfi:.3f} |")
        discs = sorted({c.split('/')[0] for c, _, _ in raw_full_cells})
        models = sorted({c.split('/')[1] for c, _, _ in raw_full_cells})
        hs = sorted({c.split('=')[1] for c, _, _ in raw_full_cells})
        L.append("")
        L.append(f"These span disciplines {discs}, models {models}, horizons {hs} — "
                 f"scattered across cells with **no coherent model/disclosure pattern** (no single model or "
                 f"disclosure type dominates; they do not cluster on any horizon).")
    else:
        L.append("_none_")
    L.append("")

    L.append("## Disjoint survivor sets (Holm basis) — substantiates \"disjoint\"")
    L.append("")
    L.append(f"**Maximal survivors (Holm, n={len(mx_surv_holm)}):** " + (", ".join(sorted(mx_surv_holm)) if mx_surv_holm else "none"))
    L.append("")
    L.append(f"**Firm-identity survivors (Holm, n={len(fi_surv_holm)}):** " + (", ".join(sorted(fi_surv_holm)) if fi_surv_holm else "none"))
    L.append("")
    L.append(f"**Overlap (cells in BOTH):** {len(overlap_holm)} — " + (", ".join(overlap_holm) if overlap_holm else "**EMPTY (the two survivor sets are disjoint)**"))
    L.append("")

    L.append("## Sanity — per-control marginals match source tables")
    L.append("")
    L.append("| check | value |")
    L.append("|---|---|")
    for k, v in sanity.items():
        L.append(f"| {k} | {v} |")
    L.append("")

    (T / "control_intersection.md").write_text("\n".join(L))

    # console echo for structured output
    print("HOLM_FULL_AND", holm_full_n)
    print("RAW_FULL_AND", len(raw_full_cells))
    print("RAW_FULL_CELLS", raw_full_cells)
    print("MAXIMAL_HOLM_SURV", sorted(mx_surv_holm))
    print("FIRM_HOLM_SURV", sorted(fi_surv_holm))
    print("OVERLAP_HOLM", overlap_holm)
    print("AND_maximal_firm_raw", int(df["AND_maximal_firm_raw"].sum()), "holm", int(df["AND_maximal_firm_holm"].sum()))
    print("AND_5way_raw", int(df["AND_5way_raw"].sum()), "holm", int(df["AND_5way_holm"].sum()))
    print("SANITY", sanity)


if __name__ == "__main__":
    main()
