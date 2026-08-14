"""Round-3 ROW 11 — 8-K item-code stratification of the surviving event-driven residual.

WHERE does the prompted-LLM 8-K residual live? The cheapest reviewer attack on the
event-driven increment is "the LLM is just reading the earnings number in Item 2.02
(Results of Operations and Financial Condition)". This script locates the residual by
8-K item code to answer that attack head-on, on the ONLY context-clean channel (8-K).

DESIGN (reuses the committed infra verbatim — nothing re-derived):
  * Panel: event_driven (8-K) only.
  * Families: Qwen3-32B (run C6_llmtext) and matched-class Llama-3.1-70B-AWQ
    (run C6_llmtext_llama70) — the two families whose pooled 8-K residual survives the
    firm-identity control (crossfamily_llm.csv / crossfamily_llama70.csv).
  * Reference: the FIRM-IDENTITY-AUGMENTED recalibrated HAR (val-window firm-mean-RV
    spec), i.e. the hardest committed reference — logic copied verbatim from
    crossfamily_llm.py (R=[1,logHAR,logFirmMean], U=+logText). The single-recalibrated-HAR
    increment (fc.log_combo) is carried alongside as an auxiliary column.
  * Combiner discipline: every weight (firm-mean map, both OLS fits) estimated on the
    FULL validation split, applied FROZEN to test. Test residuals are then PARTITIONED by
    8-K item code; NOTHING is refit per stratum (identical leakage discipline to
    m1_stratified.py).

ITEM GROUPING (mutually exclusive, earnings-CONCEDING priority so the reviewer gets the
benefit of the doubt): item_subtype is a comma-separated set of 8-K item codes (e.g.
"2.02,9.01"; 9.01 = boilerplate exhibits, non-informative). A filing is labelled by the
FIRST code present in priority order:
  2.02_earnings   -> ANY filing containing Item 2.02 (earnings release; conceded to the
                     attack even when bundled with narrative items)
  5.02_leadership -> Item 5.02 (departure/appointment of directors & officers)   | narrative
  7.01_regFD      -> Item 7.01 (Regulation FD voluntary disclosure)              | narrative
  8.01_other_events-> Item 8.01 (Other Events catch-all)                          | narrative
  5.07_shareholder_vote -> Item 5.07 (submission of matters to a shareholder vote)| procedural
  other_narrative -> everything else (1.01/2.03/5.03/9.01-only/... rare codes)   | narrative
Because 2.02 is captured FIRST, the five non-2.02 groups contain NO earnings number at
all: if the residual survives there, it cannot be number-parroting.

item_subtype is read directly from each run's predictions.parquet (schema-inspected:
column present, 0 nulls, all forms == 8-K) — this is exactly the aligned_filings item code
already carried on the run keys, so no re-join to aligned_filings is needed and the row set
stays bit-identical to the committed M1 merge (a re-join could silently drop rows and break
the sanity gate).

PER (family, item-group, horizon): n_test, n_days, rel% QLIKE vs firm-ID reference,
day-clustered DM (negative = text better), raw p, Holm p WITHIN the pre-declared family of
the 6 disjoint item-groups x 3 horizons = 18 tests (per LLM family). Plus the signed share
of the pooled absolute QLIKE reduction attributable to each group, and two DERIVED pooled
rows per (family,h): ALL (sanity anchor) and narrative_ALL (union of the 5 non-2.02 groups
= the residual with every earnings filing removed).

READING:
  - concentrates in 2.02, narrative_ALL not significant -> honest deflation (the residual
    is largely the earnings number).
  - survives in narrative_ALL (positive, clustered DM<0, significant) -> the residual is
    genuine EVENT reading, not number parroting (the stronger paper claim).

SANITY (HARD — aborts before writing any table): the pooled ALL cell for each family must
reproduce its committed crossfamily cell to machine precision (rtol 1e-12):
  Qwen  ALL == crossfamily_llm.csv     (family=qwen3_32b,   disc=event_driven)
  70B   ALL == crossfamily_llama70.csv (family=llama70_awq, disc=event_driven)
on {rel_har, dm_har, p_har, rel_firm, dm_firm, p_firm, g_text, n_test, n_days}.

Run from repo root:  .venv/bin/python scripts/analysis/row11_item_stratified.py
Outputs (NEW files): results/tables/row11_item_stratified.{csv,md}
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/analysis")
import clustered_dm as cdm
import forecast_combination as fc

KEY = ["ticker", "accession", "horizon_days"]
DISC = "event_driven"
HORIZONS = (5, 10, 20)
EPS = 1e-8
RTOL = 1e-12  # machine-precision sanity gate

FAMILIES = [("qwen3_32b", "C6_llmtext"), ("llama70_awq", "C6_llmtext_llama70")]

# Mutually-exclusive item groups in earnings-conceding priority order.
GROUP_ORDER = ["2.02_earnings", "5.02_leadership", "7.01_regFD",
               "8.01_other_events", "5.07_shareholder_vote", "other_narrative"]
GROUP_KIND = {  # narrative = contains NO earnings number
    "2.02_earnings": "earnings", "5.02_leadership": "narrative",
    "7.01_regFD": "narrative", "8.01_other_events": "narrative",
    "5.07_shareholder_vote": "procedural", "other_narrative": "narrative",
}

# Committed anchors the pooled ALL cell must reproduce (firm-ID + single-HAR specs).
ANCHOR = {"qwen3_32b": "results/tables/crossfamily_llm.csv",
          "llama70_awq": "results/tables/crossfamily_llama70.csv"}
ANCHOR_COLS = ["rel_har", "dm_har", "p_har", "rel_firm", "dm_firm", "p_firm",
               "g_text", "n_test", "n_days"]


def item_group(subtype):
    codes = set(str(subtype).split(","))
    for g in GROUP_ORDER[:-1]:
        if g.split("_")[0] in codes:
            return g
    return "other_narrative"


def ols(y, X):  # verbatim from crossfamily_llm.py
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def _cell(qU, qR, edt, h, mask=None):
    """rel% (100*mean(qR-qU)/mean(qR)), day-clustered DM(qU,qR), raw p, n_days, abs-sum.
    NEGATIVE DM = text (U) better. mask selects a stratum (None = all)."""
    if mask is not None:
        qU, qR, edt = qU[mask], qR[mask], edt[mask]
    n = len(qU)
    d = qR - qU
    mqR = float(np.mean(qR))
    rel = 100.0 * float(np.mean(d)) / mqR if mqR > 0 else float("nan")
    if n >= max(2 * h, 8):
        dm, p, nd = cdm.dm_test_clustered(qU, qR, edt, h)
    else:
        dm, p, nd = float("nan"), float("nan"), 0
    return dict(n_test=int(n), n_days=int(nd), rel_pct=rel, dm=float(dm),
                p=float(p), abs_reduction=float(np.sum(d)))


def fit_cell(a2, t, h):
    """Reproduce the crossfamily merge/fit exactly; return the per-test-obs qlike arrays,
    item groups, days, and the pooled sanity numbers. Combiner val-fit / test-frozen."""
    m = (a2[a2.horizon_days == h]
         .merge(t[t.horizon_days == h], on=KEY)
         .dropna(subset=["split", "label_realised_vol", "fh",
                         "effective_trading_day", "ft"]))
    v, te = m[m.split == "val"], m[m.split == "test"]
    y = te.label_realised_vol.values

    # --- single recalibrated-HAR reference (fc.log_combo) — auxiliary + sanity ---
    fR, fU, g = fc.log_combo(v.label_realised_vol.values, v.fh.values, v.ft.values,
                             te.fh.values, te.ft.values)
    qR, qU = fc.qlike(y, fR), fc.qlike(y, fU)

    # --- firm-identity-augmented reference (verbatim crossfamily_llm.py) — PRIMARY ---
    fm = v.groupby("ticker").label_realised_vol.mean()
    gmean = v.label_realised_vol.mean()
    fid_v = v.ticker.map(fm).fillna(gmean).values
    fid_t = te.ticker.map(fm).fillna(gmean).values
    L = lambda x: np.log(np.clip(x, EPS, None))
    ly = L(v.label_realised_vol.values)
    bR = ols(ly, np.column_stack([np.ones(len(v)), L(v.fh.values), L(fid_v)]))
    bU = ols(ly, np.column_stack([np.ones(len(v)), L(v.fh.values), L(fid_v),
                                  L(v.ft.values)]))
    fRf = np.exp(bR[0] + bR[1] * L(te.fh.values) + bR[2] * L(fid_t))
    fUf = np.exp(bU[0] + bU[1] * L(te.fh.values) + bU[2] * L(fid_t)
                 + bU[3] * L(te.ft.values))
    qRf, qUf = fc.qlike(y, fRf), fc.qlike(y, fUf)

    edt = te.effective_trading_day.values
    grp = te.item_subtype.map(item_group).to_numpy()
    return dict(qR=qR, qU=qU, qRf=qRf, qUf=qUf, edt=edt, grp=grp, g_text=float(g))


def main():
    rows = []
    pooled_anchor = {}  # (family) -> {h: pooled numbers} for sanity print
    for fam, run in FAMILIES:
        a2 = (fc.load("A2_har_rv", DISC)[KEY + ["split", "label_realised_vol",
                                                "prediction_realised_vol",
                                                "effective_trading_day", "item_subtype"]]
              .rename(columns={"prediction_realised_vol": "fh"}))
        t = (fc.load(run, DISC)[KEY + ["prediction_realised_vol"]]
             .rename(columns={"prediction_realised_vol": "ft"}))
        for h in HORIZONS:
            c = fit_cell(a2, t, h)
            qR, qU, qRf, qUf, edt, grp = (c["qR"], c["qU"], c["qRf"], c["qUf"],
                                          c["edt"], c["grp"])
            n_all = len(qRf)
            total_abs_firm = float(np.sum(qRf - qUf))

            def emit(group, kind, mask, is_partition):
                fcell = _cell(qUf, qRf, edt, h, mask)       # firm-ID (primary)
                hcell = _cell(qU, qR, edt, h, mask)          # single-HAR (aux)
                share = (100.0 * fcell["abs_reduction"] / total_abs_firm
                         if total_abs_firm != 0 else float("nan"))
                rows.append(dict(
                    family=fam, item_group=group, kind=kind, horizon=h,
                    is_partition=is_partition,
                    n_test=fcell["n_test"], n_days=fcell["n_days"],
                    share_of_filings_pct=100.0 * fcell["n_test"] / n_all,
                    rel_firm_pct=fcell["rel_pct"], dm_firm=fcell["dm"],
                    p_firm=fcell["p"], abs_reduction_firm=fcell["abs_reduction"],
                    share_of_pooled_residual_pct=share,
                    rel_har_pct=hcell["rel_pct"], dm_har=hcell["dm"], p_har=hcell["p"],
                    g_text=c["g_text"]))

            emit("ALL", "pooled", None, False)                       # sanity anchor
            for g in GROUP_ORDER:                                    # 6 disjoint groups
                emit(g, GROUP_KIND[g], grp == g, True)
            emit("narrative_ALL", "narrative", grp != "2.02_earnings", False)  # derived

    df = pd.DataFrame(rows)

    # ---- Holm WITHIN each family over the 18 disjoint item-group x horizon cells ----
    df["p_firm_holm"] = np.nan
    for fam, _ in FAMILIES:
        m = (df.family == fam) & df.is_partition
        df.loc[m, "p_firm_holm"] = fc.holm(df.loc[m, "p_firm"].fillna(1.0).values)

    # ================= SANITY GATE (abort before writing) =================
    fails = []
    for fam, _ in FAMILIES:
        ref = pd.read_csv(ANCHOR[fam])
        ref = ref[(ref.family == fam) & (ref.disc == DISC)]
        for h in HORIZONS:
            r = ref[ref.h == h]
            mine = df[(df.family == fam) & (df.item_group == "ALL") & (df.horizon == h)]
            if len(r) != 1 or len(mine) != 1:
                fails.append((fam, h, "row missing", len(r), len(mine)))
                continue
            r, mine = r.iloc[0], mine.iloc[0]
            got = {"rel_har": mine.rel_har_pct, "dm_har": mine.dm_har, "p_har": mine.p_har,
                   "rel_firm": mine.rel_firm_pct, "dm_firm": mine.dm_firm,
                   "p_firm": mine.p_firm, "g_text": mine.g_text,
                   "n_test": mine.n_test, "n_days": mine.n_days}
            pooled_anchor.setdefault(fam, {})[h] = got
            for col in ANCHOR_COLS:
                a, b = float(got[col]), float(r[col])
                if abs(a - b) > RTOL * max(abs(a), abs(b), 1.0):
                    fails.append((fam, h, col, a, b))
    if fails:
        print("SANITY FAIL — pooled ALL cell does NOT reproduce the committed crossfamily "
              "anchor to machine precision; NOT writing tables:")
        for f in fails[:30]:
            print("  ", f)
        sys.exit(1)

    # ---------------- honest headline (data-driven) ----------------
    def narr_share(fam):  # share of pooled residual in narrative_ALL, summed over horizons
        sub = df[(df.family == fam)]
        num = sub[sub.item_group == "narrative_ALL"].abs_reduction_firm.sum()
        den = sub[sub.item_group == "ALL"].abs_reduction_firm.sum()
        return 100.0 * num / den if den != 0 else float("nan")

    def narr_survive(fam):  # horizons where narrative_ALL is +, clustered DM<0, raw p<.05
        s = df[(df.family == fam) & (df.item_group == "narrative_ALL")]
        return int(((s.rel_firm_pct > 0) & (s.dm_firm < 0) & (s.p_firm < 0.05)).sum())

    def narr_partition_holm(fam):  # narrative partition cells surviving Holm<.05
        s = df[(df.family == fam) & df.is_partition
               & df.kind.isin(["narrative", "procedural"])]
        return int(((s.rel_firm_pct > 0) & (s.dm_firm < 0) & (s.p_firm_holm < 0.05)).sum())

    verdicts = {}
    for fam, _ in FAMILIES:
        ns, nv, nph = narr_share(fam), narr_survive(fam), narr_partition_holm(fam)
        if nv >= 2 and ns >= 50:
            v = (f"GENUINE EVENT READING — with every Item-2.02 (earnings) filing removed, "
                 f"the residual survives (narrative_ALL: +, clustered DM<0, raw p<.05 in "
                 f"{nv}/3 horizons) and narrative items carry {ns:.0f}% of the pooled QLIKE "
                 f"reduction. The increment is NOT earnings-number parroting.")
        elif ns < 25 and nv == 0:
            v = (f"DEFLATION — the residual concentrates in Item 2.02 (earnings): narrative "
                 f"items carry only {ns:.0f}% of the pooled reduction and the earnings-free "
                 f"narrative_ALL residual is not significant in any horizon. The 8-K "
                 f"increment is largely the earnings number.")
        else:
            v = (f"MIXED — narrative items carry {ns:.0f}% of the pooled residual and the "
                 f"earnings-free narrative_ALL residual is significant in {nv}/3 horizons "
                 f"({nph}/15 narrative partition cells survive Holm). Partly event reading, "
                 f"partly the earnings number.")
        verdicts[fam] = v

    # ---------------- write CSV ----------------
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    csv_cols = ["family", "item_group", "kind", "horizon", "is_partition", "n_test",
                "n_days", "share_of_filings_pct", "rel_firm_pct", "dm_firm", "p_firm",
                "p_firm_holm", "share_of_pooled_residual_pct", "abs_reduction_firm",
                "rel_har_pct", "dm_har", "p_har", "g_text"]
    df[csv_cols].to_csv("results/tables/row11_item_stratified.csv", index=False)

    # ---------------- write Markdown ----------------
    def famname(f):
        return "Qwen3-32B (C6_llmtext)" if f == "qwen3_32b" \
            else "Llama-3.1-70B-AWQ (C6_llmtext_llama70)"

    md = ["# Round-3 ROW 11 — 8-K item-code stratification of the surviving event-driven "
          "residual", ""]

    md += ["## RESTATED vs BEFORE", "",
           "| | BEFORE (crossfamily_llm.md / crossfamily_llama70.md) | RESTATED (this table) |",
           "|---|---|---|",
           "| unit of analysis | ONE pooled 8-K firm-ID residual per (family, horizon) | the "
           "SAME residual decomposed by 8-K item code (6 disjoint groups) |",
           "| open question | is the increment just the Item-2.02 earnings number? "
           "(unanswered) | share of the residual in 2.02 vs narrative items, and whether the "
           "earnings-free (narrative_ALL) residual survives |",
           "| combiner / reference / DM | val-fit test-frozen, firm-identity-augmented HAR, "
           "day-clustered DM (unchanged) | identical; test residuals PARTITIONED by item "
           "code, nothing refit per stratum |",
           "| multiplicity | Holm across families | Holm WITHIN each family over 6 item-groups "
           "x 3 horizons = 18 disjoint tests |", ""]

    md += ["## Method & disclosures", "",
           "- **Panel** event_driven (8-K only, all forms verified == 8-K). **Families** Qwen3-32B "
           "(`C6_llmtext`) and matched-class Llama-3.1-70B-AWQ (`C6_llmtext_llama70`).",
           "- **Reference** firm-identity-augmented recalibrated HAR "
           "(`R=exp(a+b·logHAR+c·logFirmMeanRV)`, `U=+d·logText`; firm-mean map and both OLS "
           "fits estimated on the FULL validation split, applied frozen to test). The single "
           "recalibrated-HAR increment (`fc.log_combo`) is carried as `rel_har`/`dm_har`.",
           "- **Item grouping** (earnings-CONCEDING priority): a filing is labelled by the first "
           "code present in order 2.02 -> 5.02 -> 7.01 -> 8.01 -> 5.07 -> other. Because Item "
           "2.02 is captured FIRST, the five non-2.02 groups contain NO earnings number, so a "
           "surviving increment there cannot be number-parroting. `item_subtype` read directly "
           "from predictions.parquet (0 nulls) — no re-join to aligned_filings, row set stays "
           "bit-identical to the committed M1 merge.",
           "- **DM** day-clustered on `effective_trading_day`, HAC lag = h-1 DAYS; NEGATIVE "
           "stat = text (U) better. **Holm** within each family over the 18 disjoint "
           "item-group x horizon cells (ALL and narrative_ALL are derived pooled rows, raw p "
           "only, excluded from Holm to avoid double counting).",
           "- **QLIKE unit** every QLIKE (and each `rel%`, `DM`, and `share%` derived from it) "
           "is **volatility-unit** (label and forecasts in realised-vol / sigma units via "
           "`fc.qlike`) — the SAME convention as the committed crossfamily anchor; there is no "
           "variance-unit column in this table.", ""]

    for fam, _ in FAMILIES:
        md += [f"## {famname(fam)} — item-group x horizon (firm-ID reference)", "",
               "`**` = clustered DM<0 & Holm p<.05. rel% > 0 = text lowers volatility-unit "
               "QLIKE vs the firm-ID reference. share% = signed share of the pooled absolute "
               "QLIKE reduction carried by that group.", "",
               "| item group | kind | h | n_test | n_days | % filings | rel% (firm-ID) | "
               "DM(clu) | raw p | Holm p | share% | rel% (vs HAR) |",
               "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        order = {g: i for i, g in enumerate(["ALL"] + GROUP_ORDER + ["narrative_ALL"])}
        sub = df[df.family == fam].copy()
        sub["_ord"] = sub.item_group.map(order)
        for _, r in sub.sort_values(["horizon", "_ord"]).iterrows():
            sig = "**" if (r.dm_firm < 0 and pd.notna(r.p_firm_holm)
                           and r.p_firm_holm < 0.05) else ""
            holm = f"{r.p_firm_holm:.4g}" if pd.notna(r.p_firm_holm) else "—"
            md.append(f"| {r.item_group} | {r.kind} | {int(r.horizon)} | {int(r.n_test)} | "
                      f"{int(r.n_days)} | {r.share_of_filings_pct:.1f} | "
                      f"{r.rel_firm_pct:+.2f}{sig} | {r.dm_firm:+.2f} | {r.p_firm:.4f} | "
                      f"{holm} | {r.share_of_pooled_residual_pct:+.1f} | {r.rel_har_pct:+.2f} |")
        md.append("")

    # pooled residual-share decomposition (2.02 vs narrative), summed over horizons
    md += ["## Where the pooled residual lives — Item 2.02 (earnings) vs narrative "
           "(summed over horizons)", "",
           "| family | 2.02 share of filings | 2.02 share of residual | narrative share of "
           "residual | narrative_ALL survives (h with +,DM<0,p<.05) |",
           "|---|--:|--:|--:|--:|"]
    for fam, _ in FAMILIES:
        sub = df[df.family == fam]
        tot = sub[sub.item_group == "ALL"].abs_reduction_firm.sum()
        e = sub[sub.item_group == "2.02_earnings"].abs_reduction_firm.sum()
        nrow = sub[sub.item_group == "narrative_ALL"]
        n = nrow.abs_reduction_firm.sum()
        fshare = 100.0 * sub[(sub.item_group == "2.02_earnings")
                             & (sub.horizon == 5)].n_test.iloc[0] \
            / sub[(sub.item_group == "ALL") & (sub.horizon == 5)].n_test.iloc[0]
        nv = int(((nrow.rel_firm_pct > 0) & (nrow.dm_firm < 0) & (nrow.p_firm < 0.05)).sum())
        md.append(f"| {famname(fam)} | {fshare:.0f}% | "
                  f"{100.0 * e / tot if tot else float('nan'):+.0f}% | "
                  f"{100.0 * n / tot if tot else float('nan'):+.0f}% | {nv}/3 |")
    md.append("")

    md += ["## HEADLINE (honest)", ""]
    for fam, _ in FAMILIES:
        md += [f"- **{famname(fam)}** — {verdicts[fam]}"]
    md.append("")

    md += ["## SANITY", "",
           "Pooled ALL cell reproduces the committed crossfamily anchor to machine precision "
           f"(rtol {RTOL:g}) on {ANCHOR_COLS}:"]
    for fam, _ in FAMILIES:
        h5 = pooled_anchor[fam][5]
        md.append(f"- **{fam}** vs `{ANCHOR[fam]}`: PASS "
                  f"(h5 rel_firm={h5['rel_firm']:+.4f}%, dm_firm={h5['dm_firm']:+.4f}, "
                  f"n_test={int(h5['n_test'])}, n_days={int(h5['n_days'])}).")
    md.append("")

    Path("results/tables/row11_item_stratified.md").write_text("\n".join(md))

    print("SANITY PASS — pooled ALL cells reproduce crossfamily_llm/crossfamily_llama70 "
          f"to rtol {RTOL:g}")
    print(f"wrote results/tables/row11_item_stratified.csv/.md ({len(df)} rows)")
    for fam, _ in FAMILIES:
        print(f"\n[{fam}] {verdicts[fam]}")
    show = df[df.item_group.isin(["ALL", "2.02_earnings", "narrative_ALL"])][
        ["family", "item_group", "horizon", "n_test", "rel_firm_pct", "dm_firm",
         "p_firm", "share_of_pooled_residual_pct"]]
    print("\n" + show.to_string(index=False))


if __name__ == "__main__":
    main()
