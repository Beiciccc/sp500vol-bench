"""ROW 15 companion — does the 3-seed 70B ensemble upgrade the 8-K residual?

Recomputes the matched-class Llama-3.1-70B M1 increment on the event_driven (8-K)
channel for the 3-SEED ENSEMBLE run (C6_llmtext_llama70ens) vs
  (a) the single recalibrated-HAR reference, and
  (b) the firm-identity-augmented reference (val-window firm-mean spec),
day-clustered DM, Holm applied within the pre-declared family of the 6 ensemble
tests (3 horizons x {vs HAR, vs HAR+firmID}) — EXACTLY the crossfamily_llama70
protocol. Reports whether Holm significance improves over the single seed:
  BEFORE (single seed 2026): 1/3 vs-single-HAR cells survive Holm(6); 0/3 firmID
  cells (best firmID Holm p = 0.05001, h=5) — "directionally replicates".
  RESTATED (3-seed ensemble): recomputed here.
The upgrade "directionally replicates" -> "replicates" requires 3/3 firmID cells
at Holm < .05 (the crossfamily_llama70 REPLICATES gate).

SANITY (HARD — aborts before writing): the SINGLE-SEED 2026 rows recomputed by this
script (C6_llmtext_llama70, event_driven) reproduce the committed
results/tables/crossfamily_llama70.csv llama70_awq cells — rel/dm/p vs HAR and vs
HAR+firmID, g_text, n_test, n_days, AND the pre-declared p_har_holm / p_firm_holm —
to machine precision. The M1 math is imported/copied verbatim from
crossfamily_llama70, so this must hold exactly.

Run from repo root:
  .venv/bin/python scripts/analysis/row15_ensemble_m1.py               # full (needs ensemble run dir)
  .venv/bin/python scripts/analysis/row15_ensemble_m1.py --sanity-only # pre-flight reproduction check
Outputs (NEW): results/tables/row15_ensemble_m1.{csv,md}
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/analysis")
import forecast_combination as fc          # noqa: E402  fc.load / log_combo / qlike (VOL-unit)
import clustered_dm as cdm                 # noqa: E402  day-clustered DM
import crossfamily_llama70 as cfl          # noqa: E402  ols / holm / close (verbatim M1 helpers)

KEY = cfl.KEY                              # ["ticker", "accession", "horizon_days"]
EPS = cfl.EPS
HORIZONS = cfl.HORIZONS                    # (5, 10, 20)
DISC = "event_driven"
SINGLE_RUN = "C6_llmtext_llama70"
ENS_RUN = "C6_llmtext_llama70ens"
REF_CSV = "results/tables/crossfamily_llama70.csv"
# columns compared machine-precision against the committed crossfamily row (single seed)
SANITY_COLS = ["n_test", "n_days", "rel_har", "dm_har", "p_har",
               "rel_firm", "dm_firm", "p_firm", "g_text", "p_har_holm", "p_firm_holm"]


def m1_cell(a2, t, h):
    """The crossfamily_llama70 M1 block, VERBATIM (log-space combiner, val-fit /
    test-frozen; single-HAR reference + firm-identity-augmented reference;
    day-clustered DM). VOL-unit QLIKE. Returns one dict per horizon."""
    m = a2[a2.horizon_days == h].merge(t[t.horizon_days == h], on=KEY).dropna()
    v, te = m[m.split == "val"], m[m.split == "test"]
    y = te.label_realised_vol.values
    # --- reference (a): single recalibrated HAR ---
    fR, fU, g = fc.log_combo(v.label_realised_vol.values, v.fh.values, v.ft.values,
                             te.fh.values, te.ft.values)
    qR, qU = fc.qlike(y, fR), fc.qlike(y, fU)
    rel = 100 * np.mean(qR - qU) / np.mean(qR)
    dm, pv, nd = cdm.dm_test_clustered(qU, qR, te.effective_trading_day.values, h)
    # --- reference (b): firm-identity-augmented (val-window firm mean) ---
    fm = v.groupby("ticker").label_realised_vol.mean()
    gmean = v.label_realised_vol.mean()
    fid_v = v.ticker.map(fm).fillna(gmean).values
    fid_t = te.ticker.map(fm).fillna(gmean).values
    L = lambda x: np.log(np.clip(x, EPS, None))  # noqa: E731
    ly = L(v.label_realised_vol.values)
    bR = cfl.ols(ly, np.column_stack([np.ones(len(v)), L(v.fh.values), L(fid_v)]))
    bU = cfl.ols(ly, np.column_stack([np.ones(len(v)), L(v.fh.values), L(fid_v),
                                      L(v.ft.values)]))
    fRf = np.exp(bR[0] + bR[1] * L(te.fh.values) + bR[2] * L(fid_t))
    fUf = np.exp(bU[0] + bU[1] * L(te.fh.values) + bU[2] * L(fid_t) + bU[3] * L(te.ft.values))
    qRf, qUf = fc.qlike(y, fRf), fc.qlike(y, fUf)
    relf = 100 * np.mean(qRf - qUf) / np.mean(qRf)
    dmf, pf, _ = cdm.dm_test_clustered(qUf, qRf, te.effective_trading_day.values, h)
    return dict(h=int(h), n_test=len(te), n_days=nd,
                rel_har=rel, dm_har=dm, p_har=pv,
                rel_firm=relf, dm_firm=dmf, p_firm=pf, g_text=g)


def compute_run(run):
    """M1 cells for one text run on the 8-K channel + the pre-declared Holm(6)."""
    a2 = fc.load("A2_har_rv", DISC)[KEY + ["split", "label_realised_vol",
                                           "prediction_realised_vol",
                                           "effective_trading_day"]] \
        .rename(columns={"prediction_realised_vol": "fh"})
    p = fc.load(run, DISC)
    t = p[KEY + ["prediction_realised_vol"]].rename(columns={"prediction_realised_vol": "ft"})
    rows = [m1_cell(a2, t, h) for h in HORIZONS]
    df = pd.DataFrame(rows).sort_values("h").reset_index(drop=True)
    # Holm within the pre-declared family of 6 tests: [p_har(3), p_firm(3)]
    adj = cfl.holm(np.concatenate([df.p_har.values, df.p_firm.values]))
    df["p_har_holm"] = adj[:3]
    df["p_firm_holm"] = adj[3:]
    return df


def gates(df):
    """crossfamily_llama70 replication counts."""
    return dict(
        n_rep_firm=int(((df.dm_firm < 0) & (df.p_firm_holm < .05)).sum()),   # REPLICATES gate
        n_rep_har=int(((df.dm_har < 0) & (df.p_har_holm < .05)).sum()),
        n_pos_firm=int((df.rel_firm > 0).sum()),
        n_sig_raw_firm=int(((df.dm_firm < 0) & (df.p_firm < .05)).sum()),
        min_firm_holm=float(df.p_firm_holm.min()),
        min_har_holm=float(df.p_har_holm.min()),
    )


def verdict(df):
    g = gates(df)
    if g["n_rep_firm"] == 3:
        return "REPLICATES", ("All 3 firm-identity cells survive the pre-declared Holm(6) "
                              "(dm<0, p<.05) — the 8-K residual is Qwen-independent and Holm-robust "
                              "in the matched-class family.")
    if g["n_pos_firm"] == 3 and (g["n_sig_raw_firm"] >= 2 or g["n_rep_har"] >= 1):
        return "DIRECTIONALLY REPLICATES", (
            f"3/3 firmID cells positive, raw p<.05 in {g['n_sig_raw_firm']}/3, but after Holm(6) "
            f"{g['n_rep_firm']}/3 firmID (min {g['min_firm_holm']:.5f}) and {g['n_rep_har']}/3 "
            f"vs-single-HAR cells survive.")
    if g["n_pos_firm"] == 0 and g["n_sig_raw_firm"] == 0:
        return "DOES NOT REPLICATE", "No positive firmID increment in any horizon."
    return "PARTIAL/MIXED", (f"{g['n_rep_firm']}/3 firmID Holm<.05, {g['n_sig_raw_firm']}/3 raw p<.05, "
                             f"{g['n_pos_firm']}/3 positive; {g['n_rep_har']}/3 vs single HAR after Holm.")


def sanity_single(single):
    """Single-seed 2026 cells must reproduce crossfamily_llama70.csv exactly."""
    ref = pd.read_csv(REF_CSV)
    ref = ref[(ref.family == "llama70_awq") & (ref.disc == DISC)].sort_values("h").reset_index(drop=True)
    if len(ref) != 3:
        return False, [("expected 3 committed llama70_awq event_driven rows", f"got {len(ref)}")]
    bad = []
    for _, r in ref.iterrows():
        mine = single[single.h == int(r.h)]
        if len(mine) != 1:
            bad.append((int(r.h), "row missing")); continue
        for c in SANITY_COLS:
            if not cfl.close(mine[c].iloc[0], r[c]):
                bad.append((int(r.h), c, float(mine[c].iloc[0]), float(r[c])))
    return (len(bad) == 0), bad


def cellstr(r):
    s1 = "**" if (r.dm_har < 0 and r.p_har < .05) else ""
    s2 = "**" if (r.dm_firm < 0 and r.p_firm < .05) else ""
    return (f"{r.rel_har:+.2f}%{s1} | {r.dm_har:+.2f} | {r.p_har_holm:.4g} | "
            f"{r.rel_firm:+.2f}%{s2} | {r.dm_firm:+.2f} | {r.p_firm_holm:.4g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sanity-only", action="store_true",
                    help="reproduce the single-seed crossfamily cells and exit (no ensemble needed)")
    args = ap.parse_args()

    # ---- SANITY (single-seed 2026 reproduces the committed table) ----
    single = compute_run(SINGLE_RUN)
    ok, bad = sanity_single(single)
    if not ok:
        print(f"SANITY FAIL: single-seed 2026 does NOT reproduce {REF_CSV} llama70_awq/event_driven:")
        for b in bad[:20]:
            print("  ", b)
        sys.exit(1)
    print(f"SANITY PASS: single-seed 2026 reproduces {REF_CSV} llama70_awq/event_driven "
          f"to machine precision (rtol {cfl.RTOL:g}) on {SANITY_COLS}.")
    sg = gates(single)
    print(f"  single-seed gates: {sg['n_rep_har']}/3 vs-HAR Holm<.05, {sg['n_rep_firm']}/3 firmID "
          f"Holm<.05, best firmID Holm p={sg['min_firm_holm']:.5f}")

    if args.sanity_only:
        print("--sanity-only: skipping ensemble (no C6_llmtext_llama70ens run required).")
        return

    ens_path = Path(f"results/runs/{ENS_RUN}_full_{DISC}_seed2026/predictions.parquet")
    if not ens_path.exists():
        print(f"ENSEMBLE RUN NOT FOUND: {ens_path}\n"
              f"Run scripts/experiments/row15_llama70_ensemble/launch.sh on the box first, "
              f"then rsync results/runs/{ENS_RUN}_full_{DISC}_seed2026 back. "
              f"(SANITY already passed above; re-run with --sanity-only for pre-flight only.)")
        sys.exit(2)

    ens = compute_run(ENS_RUN)
    sv, sd = verdict(single)
    ev, ed = verdict(ens)
    eg = gates(ens)

    # per-horizon before/after CSV
    out = single.add_suffix("_single").rename(columns={"h_single": "h"}).merge(
        ens.add_suffix("_ens").rename(columns={"h_ens": "h"}), on="h")
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    out.to_csv("results/tables/row15_ensemble_m1.csv", index=False)

    # a Qwen benchmark line for context (committed table)
    ref = pd.read_csv(REF_CSV)
    qe = ref[(ref.family == "qwen3_32b") & (ref.disc == DISC)].sort_values("h")
    _q = "/".join(f"{x:+.2f}" for x in qe.rel_firm)

    upgraded = (sv != "REPLICATES") and (ev == "REPLICATES")
    improved = (eg["n_rep_firm"] > sg["n_rep_firm"]) or (eg["n_rep_har"] > sg["n_rep_har"]) \
        or (eg["min_firm_holm"] < sg["min_firm_holm"])

    md = [
        "# ROW 15 — 3-seed 70B ensemble vs single-seed, matched-class 8-K residual",
        "",
        "## RESTATED vs BEFORE",
        "",
        "| | BEFORE (single seed 2026, crossfamily_llama70) | RESTATED (3-seed ensemble 2026+2027+2028) |",
        "|---|---|---|",
        f"| verdict | {sv} — {sd} | **{ev}** — {ed} |",
        f"| firmID cells surviving Holm(6) | {sg['n_rep_firm']}/3 (best p={sg['min_firm_holm']:.5f}) | "
        f"**{eg['n_rep_firm']}/3** (best p={eg['min_firm_holm']:.5f}) |",
        f"| vs-single-HAR cells surviving Holm(6) | {sg['n_rep_har']}/3 (best p={sg['min_har_holm']:.5f}) | "
        f"**{eg['n_rep_har']}/3** (best p={eg['min_har_holm']:.5f}) |",
        "",
        "Both columns use the identical crossfamily_llama70 protocol: log-space combiner (val-fit / "
        "test-frozen), day-clustered DM, VOL-unit QLIKE, Holm within the pre-declared family of 6 "
        "tests (3 horizons x {vs recalibrated HAR, vs HAR+firm-identity}). The only change is the "
        "forecast object: a per-observation arithmetic mean across vLLM seeds vs seed 2026 alone.",
        "",
        "## Table — M1 increment (rel% vs HAR | DM(clu) | Holm p | rel% vs HAR+firmID | DM(clu) | Holm p)",
        "",
        "`**` = clustered DM<0 & raw p<.05.",
        "",
        "| h | n_test | SINGLE | ENSEMBLE |",
        "|--:|--:|---|---|",
    ]
    for h in HORIZONS:
        rs = single[single.h == h].iloc[0]
        re = ens[ens.h == h].iloc[0]
        md.append(f"| {h} | {int(re.n_test)} | {cellstr(rs)} | {cellstr(re)} |")

    md += [
        "",
        "## HEADLINE (honest)",
        "",
        (f"Ensembling **{'UPGRADES the verdict to REPLICATES' if upgraded else 'does NOT reach REPLICATES'}**. "
         f"Single-seed was {sv} ({sg['n_rep_firm']}/3 firmID, {sg['n_rep_har']}/3 vs-HAR Holm-robust; best "
         f"firmID Holm p={sg['min_firm_holm']:.5f}); the 3-seed ensemble is {ev} "
         f"({eg['n_rep_firm']}/3 firmID, {eg['n_rep_har']}/3 vs-HAR; best firmID Holm p={eg['min_firm_holm']:.5f})."),
        "",
        (f"- Holm significance {'IMPROVES' if improved else 'does not improve'} over the single seed "
         f"(firmID Holm min {sg['min_firm_holm']:.5f} -> {eg['min_firm_holm']:.5f}; vs-HAR "
         f"{sg['n_rep_har']}/3 -> {eg['n_rep_har']}/3)."),
        f"- Ensemble firmID rel%: {'/'.join(f'{ens[ens.h==h].rel_firm.iloc[0]:+.2f}' for h in HORIZONS)} "
        f"(h=5/10/20); Qwen event-driven benchmark was {_q}.",
        (f"- CAVEAT on seed diversity: run_inference.py decodes at temperature 0 and does not pass --seed "
         f"to vLLM, so the 3 seeds diverge only through AWQ-INT4 / TP2 kernel non-determinism. If the "
         f"ensemble ≈ the single seed (rel% and Holm p nearly unchanged), that is the expected signature "
         f"of near-deterministic decoding, NOT a bug — report it as such."),
        "",
        "## SANITY",
        "",
        f"- PASS: single-seed 2026 cells reproduce {REF_CSV} (llama70_awq / event_driven) to machine "
        f"precision (rtol {cfl.RTOL:g}) on {SANITY_COLS}.",
        "",
    ]
    Path("results/tables/row15_ensemble_m1.md").write_text("\n".join(md))

    print(f"single-seed verdict : {sv}")
    print(f"ensemble  verdict   : {ev}")
    print(f"firmID Holm-robust  : {sg['n_rep_firm']}/3 -> {eg['n_rep_firm']}/3 "
          f"(best firmID Holm p {sg['min_firm_holm']:.5f} -> {eg['min_firm_holm']:.5f})")
    print(f"vs-HAR Holm-robust  : {sg['n_rep_har']}/3 -> {eg['n_rep_har']}/3")
    print(f"upgrade to REPLICATES: {upgraded}   Holm improved: {improved}")
    print("wrote results/tables/row15_ensemble_m1.{csv,md}")
    print(ens[["h", "n_test", "rel_har", "dm_har", "p_har_holm",
               "rel_firm", "dm_firm", "p_firm_holm"]].to_string(index=False))


if __name__ == "__main__":
    main()
