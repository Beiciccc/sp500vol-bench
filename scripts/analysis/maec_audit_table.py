#!/usr/bin/env python
"""MAEC audit final table + pre-registered branch determination
(configs/prereg_maec_audit.md v1.2, §5 / §6.2 / §6.3 / §6.4 / §8).

AGGREGATION ONLY (prereg §6.5 single-shot discipline): this script reads the
NINE frozen jsons —

    results/second_domain/maec/protocol_{tfidf,qwen_emb,prompted_qwen,
                                          identity_probe}_{primary,shifted}.json
    results/second_domain/maec/published_readings.json

— and assembles results/tables/maec_audit.csv + maec_audit.md. No DM, placebo,
MDE, Holm or bootstrap statistic is recomputed from predictions; the only
arithmetic performed here is
  (i)  pass/branch logic on the frozen numbers (§6.2/§6.3/§8), and
  (ii) the §6.4 unit conversion of the published-convention gain into the
       rel-% units the MDE uses (documented in the md, context block).

Field map (maec_protocol.py output):
  row3  (combination increment,        f_U  vs f_X ) = horizons[h][ref]["combined"]
  row5  (identity-controlled residual, f_Ue vs f_Xe) = horizons[h][ref]["residual"]
  identity share (d4/d3)  = horizons[h][ref]["entity"]["identity_share_pct"]
                            (NaN iff d3 <= 0, i.e. no combination increment)
  Holm(8)                 = ...["p_holm8"]   (frozen per §6.2 family F1/F2)
  placebo G4 (label-shuffle, 20 seeds) = ...["placebo"]["row{3,5}_shuffle"]
  placebo G4b (within-date swap, 5 sd) = ...["placebo"]["row{3,5}_swap"]
  MDE (AR / entity stage) = ...["mde"]["{ar,entity}_stage_rel_pct"]
  bootstrap CI            = ...["bootstrap_ci_rel_pct"]

Gate criteria (Yelp precedent, yelp_cascade_table.py):
  G4  pass  per cell = |mean_dm| < 2.0 AND mean_p > .05 on the stage-matched
                       label-shuffle placebo (pre-registered primary gate);
  G4b dirty per cell = mean_p <= .05 OR |mean_dm| >= 2.0 on the stage-matched
                       within-date swap (diagnostic; a dirty cell never enters
                       prose claims, prereg §6.3).

§8 branch (headline arms only; identity_probe is diagnostic, §6.2):
  F2 "pass"  = residual DM < 0  AND  Holm(8) < .05  AND  G4 pass.
  (a) FULLY ABSORBED   : 0/8 F2 passes AND, at every cell whose F1 combination
      increment is Holm-significant, identity share >= 100%  [strict per-cell
      reading — the share clause is evaluated cell-wise, mirroring (b)'s
      "at these horizons"; the alternative headline-R-AR reading is reported
      alongside, never silently substituted].
  (b) PARTIALLY ABSORBED: >=1/8 F2 passes AND share >= 50% at every passing cell.
  (c) SURVIVES          : >=4/8 F2 passes AND share < 50% at the passing cells.
  else MIXED (cell-by-cell reporting, weakest defensible wording).

§6.4 MDE discipline per null F2 cell: an "absorbed / no residual" wording is
licensed only if MDE(ent) <= the repriced published gain in the same units,
where  G_conv(cell) = 100 * Delta_pub(arm, h) / MSE_Re(cell)  and
Delta_pub(arm, h) = row-equal-weight pooled (MSE_raw - MSE_text) over the three
published year panels. Otherwise the wording downgrades to "underpowered to
rule out". qwen_emb has NO published-convention reading (the published scorer
ran tfidf + prompted_qwen only, G1 needed >=1 arm): its comparison uses the
tfidf gain as an explicitly labelled PROXY — disclosed, incl. the direction of
non-conservatism.

Usage:  .venv/bin/python scripts/analysis/maec_audit_table.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import csv
import json
import math
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAEC_DIR = REPO / "results/second_domain/maec"
PREREG = REPO / "configs/prereg_maec_audit.md"
OUT_CSV = REPO / "results/tables/maec_audit.csv"
OUT_MD = REPO / "results/tables/maec_audit.md"

ARMS = ("tfidf", "qwen_emb", "prompted_qwen", "identity_probe")
HEADLINE = ("tfidf", "qwen_emb", "prompted_qwen")   # §6.2; probe = diagnostic
ALIGNS = ("primary", "shifted")
HORIZONS = ("3", "7", "15", "30")
REFS = ("r_ar", "r_har")                             # OPEN-3: r_ar headline
PANELS = ("2015", "2016", "2017-18")                 # published-convention (§4 v1.1)
ALPHA = 0.05
G4_DM_MAX = 2.0                                      # Yelp precedent (yelp_cascade_table.py)


def isnum(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def fnum(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NaN"
    return f"{x:+.{nd}f}" if x < 0 or True else f"{x:.{nd}f}"


def fpos(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NaN"
    return f"{x:.{nd}f}"


# --------------------------------------------------------------- gate logic
def g4_pass(pl):
    """G4 primary gate on the stage-matched label-shuffle placebo."""
    return abs(pl["mean_dm"]) < G4_DM_MAX and pl["mean_p"] > ALPHA


def g4b_dirty(pl):
    """G4b diagnostic flag on the stage-matched within-date swap."""
    return pl["mean_p"] <= ALPHA or abs(pl["mean_dm"]) >= G4_DM_MAX


def stage_flags(block, shuffle, swap):
    """(win, neg_sig, g4, g4b_dirty) for one stage block (combined/residual)."""
    g4 = g4_pass(shuffle)
    dirty = g4b_dirty(swap)
    win = block["dm"] < 0 and block["p_holm8"] < ALPHA and g4
    neg = block["dm"] > 0 and block["p_holm8"] < ALPHA
    return win, neg, g4, dirty


def f1_sig(cell):
    """Holm-significant combination increment (two-sided; direction reported)."""
    return cell["combined"]["p_holm8"] < ALPHA


# ------------------------------------------------------------ §8 branch logic
def branch(run):
    """§8 determination for one (headline arm x alignment) run.
    Returns dict with counts, cell lists and the fired branch under the strict
    per-cell reading + the alternative headline-R-AR reading."""
    cells = [(h, r) for h in HORIZONS for r in REFS]
    f2_pass, f2_neg, f1_win, f1_sig_cells, discord = [], [], [], [], []
    for h, r in cells:
        c = run["horizons"][h][r]
        w5, n5, _, _ = stage_flags(c["residual"], c["placebo"]["row5_shuffle"],
                                   c["placebo"]["row5_swap"])
        w3, _, _, _ = stage_flags(c["combined"], c["placebo"]["row3_shuffle"],
                                  c["placebo"]["row3_swap"])
        share = c["entity"]["identity_share_pct"]
        if w5:
            f2_pass.append((h, r, share))
        if n5:
            f2_neg.append((h, r))
        if w3:
            f1_win.append((h, r))
        if f1_sig(c):
            f1_sig_cells.append((h, r, share, c["combined"]["dm"]))
            # clause-(a) strict per-cell: share >= 100 wherever F1 is Holm-sig
            if not (isnum(share) and share >= 100.0):
                discord.append((h, r, share, c["combined"]["p_holm8"],
                                c["combined"]["dm"]))
    n2 = len(f2_pass)
    # strict per-cell reading
    if n2 == 0 and not discord:
        fired = "(a) FULLY ABSORBED"
    elif n2 >= 4 and all(isnum(s) and s < 50.0 for _, _, s in f2_pass):
        fired = "(c) SURVIVES"
    elif n2 >= 1 and all(isnum(s) and s >= 50.0 for _, _, s in f2_pass):
        fired = "(b) PARTIALLY ABSORBED"
    else:
        fired = "MIXED"
    # alternative reading: clause-(a) share condition at the headline R-AR only
    discord_ar = [d for d in discord if d[1] == "r_ar"]
    if n2 == 0 and not discord_ar:
        alt = "(a) FULLY ABSORBED"
    else:
        alt = fired
    return {"f2_pass": f2_pass, "f2_neg": f2_neg, "f1_win": f1_win,
            "f1_sig": f1_sig_cells, "discord": discord,
            "fired": fired, "alt": alt}


# ----------------------------------------------------- §6.4 published gains
def published_gains(pub):
    """Row-equal-weight pooled per-horizon published-convention text gain
    Delta_pub = MSE_raw - MSE_text (v^2 units), per scored arm."""
    raw_cells = pub["raw_vpast"]["cells"]
    out = {}
    for arm, blk in pub["arms"].items():
        per_h = {}
        for h in HORIZONS:
            num_raw = num_txt = den = 0.0
            for pnl in PANELS:
                cell = blk["cells"][f"{pnl}_h{h}"]
                n = cell["n_test"]
                assert n == raw_cells[f"{pnl}_h{h}"]["n_test"]
                num_raw += n * cell["mse_raw_vpast"]
                num_txt += n * cell["mse_text_standalone"]
                den += n
            raw_p, txt_p = num_raw / den, num_txt / den
            # cross-check the pooled raw MSE against the frozen per-horizon value
            frozen = pub["raw_vpast"]["per_horizon_pooled_mse"][h]
            assert abs(raw_p - frozen) < 1e-9, (h, raw_p, frozen)
            per_h[h] = {"raw_pooled": raw_p, "text_pooled": txt_p,
                        "delta_pub": raw_p - txt_p,
                        "rel_of_raw_pct": 100.0 * (raw_p - txt_p) / raw_p}
        out[arm] = per_h
    return out


def mde_verdict(arm, align, h, ref, cell, gains):
    """§6.4 comparison for one F2 cell: MDE(ent) vs the repriced published
    gain converted into the same rel-% units (denominator = MSE_Re(cell))."""
    mde = cell["mde"]["entity_stage_rel_pct"]
    mse_re = cell["mse"]["Re"]
    if arm in gains:
        g_arm, proxy = arm, False
    else:                       # qwen_emb: no published reading -> tfidf PROXY
        g_arm, proxy = "tfidf", True
    d_pub = gains[g_arm][h]["delta_pub"]
    g_conv = 100.0 * d_pub / mse_re
    if d_pub <= 0:
        verdict = "no published gain at this cell (text loses to raw V_past) — nothing to absorb"
    elif mde <= g_conv:
        verdict = "powered: MDE(ent) <= repriced published gain — 'fully absorbed' wording licensed"
    else:
        verdict = "UNDERPOWERED to rule out — wording must downgrade"
    return {"mde_ent": mde, "mse_re": mse_re, "delta_pub": d_pub,
            "g_conv": g_conv, "proxy": proxy, "verdict": verdict}


# --------------------------------------------------------------------- main
def main():
    runs = {}
    for arm in ARMS:
        for al in ALIGNS:
            fp = MAEC_DIR / f"protocol_{arm}_{al}.json"
            runs[(arm, al)] = json.loads(fp.read_text())
    pub = json.loads((MAEC_DIR / "published_readings.json").read_text())

    # -------- provenance / frozen-spec assertions (read-only, no recompute)
    for (arm, al), r in runs.items():
        assert r["tag"] == "REAL" and r["arm"] == arm and r["alignment"] == al
        assert r["placebo_seeds"] == list(range(1000, 1020))
        assert r["swap_seeds"] == [2000, 2001, 2002, 2003, 2004]
        assert r["embargo_val"] is False
        for h in HORIZONS:
            for ref in REFS:
                blk = r["horizons"][h][ref]
                for stage in ("combined", "residual"):
                    assert "p_holm8" in blk[stage], (arm, al, h, ref, stage)
    # references are text-free -> MSE_Re identical across arms within alignment
    for al in ALIGNS:
        base = runs[("tfidf", al)]
        for arm in ARMS[1:]:
            for h in HORIZONS:
                for ref in REFS:
                    a = runs[(arm, al)]["horizons"][h][ref]["mse"]
                    b = base["horizons"][h][ref]["mse"]
                    assert abs(a["R"] - b["R"]) < 1e-12 and abs(a["Re"] - b["Re"]) < 1e-12

    gains = published_gains(pub)

    # ------------------------------------------------------------------ CSV
    csv_rows = []
    for arm in ARMS:
        for al in ALIGNS:
            r = runs[(arm, al)]
            for h in HORIZONS:
                for ref in REFS:
                    c = r["horizons"][h][ref]
                    share = c["entity"]["identity_share_pct"]
                    mde_e = c["mde"]["entity_stage_rel_pct"]
                    for stage, key in (("row3", "combined"), ("row5", "residual")):
                        blk = c[key]
                        sh = c["placebo"][f"{stage}_shuffle"]
                        sw = c["placebo"][f"{stage}_swap"]
                        win, neg, g4, dirty = stage_flags(blk, sh, sw)
                        if arm == "identity_probe":
                            gate = "diagnostic (§6.2)"
                        else:
                            gate = ("win" if win else ("neg_sig" if neg else "null"))
                            gate += f"|G4:{'pass' if g4 else 'FAIL'}"
                            gate += f"|G4b:{'dirty' if dirty else 'clean'}"
                        ci = blk["bootstrap_ci_rel_pct"]
                        csv_rows.append({
                            "arm": arm, "alignment": al, "horizon": h, "ref": ref,
                            "stage": stage,
                            "rel_pct": f"{blk['delta_rel_pct']:.4f}",
                            "dm": f"{blk['dm']:.4f}",
                            "p_raw": f"{blk['p']:.6f}",
                            "p_holm": f"{blk['p_holm8']:.6f}",
                            "placebo_ls": f"dm={sh['mean_dm']:+.2f},p={sh['mean_p']:.3f}",
                            "placebo_wd": f"dm={sw['mean_dm']:+.2f},p={sw['mean_p']:.3f}",
                            "gate": gate,
                            "identity_share": (f"{share:.2f}" if isnum(share) else "NaN"),
                            "mde_ent_pct": f"{mde_e:.4f}",
                            "boot_ci": f"[{ci['ci_lo']:.2f},{ci['ci_hi']:.2f}]",
                        })
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    # ---------------------------------------------------------------- verdicts
    verdicts = {(arm, al): branch(runs[(arm, al)])
                for arm in HEADLINE for al in ALIGNS}
    mde_table = {}          # (arm, al) -> list of per-F2-cell §6.4 dicts
    for arm in HEADLINE:
        for al in ALIGNS:
            rows = []
            for h in HORIZONS:
                for ref in REFS:
                    cell = runs[(arm, al)]["horizons"][h][ref]
                    rows.append((h, ref, mde_verdict(arm, al, h, ref, cell, gains)))
            mde_table[(arm, al)] = rows

    # §8 verbatim quote from the prereg
    lines = PREREG.read_text().splitlines()
    i8 = next(i for i, l in enumerate(lines) if l.startswith("## 8."))
    i9 = next(i for i, l in enumerate(lines) if l.startswith("## 9."))
    s8 = "\n".join("> " + l for l in lines[i8:i9]).rstrip()

    # ------------------------------------------------------------------- MD
    md = []
    md.append("# MAEC audit master table (single-shot aggregation; prereg-maec v1.2 §5/§6/§8)\n")
    md.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} · script "
              f"`scripts/analysis/maec_audit_table.py` (pure aggregation: all DM/placebo/MDE/"
              f"Holm/bootstrap read from the frozen jsons, zero recomputation; the only arithmetic = gate/branch logic and the "
              f"§6.4 unit conversion, see the §1 conversion note).\n")

    # provenance
    md.append("## 0. Single-shot sources (§6.5 provenance)\n")
    md.append("| json | generated | merge_dropped | crosscheck max\\|Δpred\\| |")
    md.append("|---|---|---|---|")
    for arm in ARMS:
        for al in ALIGNS:
            r = runs[(arm, al)]
            cc = max((v["max_pred_absdiff"] for v in
                      r["crosscheck_vs_baseline_half"].values()), default=float("nan"))
            md.append(f"| protocol_{arm}_{al}.json | {r['generated']} | "
                      f"{r['merge_dropped_rows']} | {cc:.1e} |")
    md.append(f"| published_readings.json | {pub['generated']} | — | — |")
    r0 = runs[("tfidf", "primary")]
    hl = r0["hac_lags_L"]
    md.append("\n- All 8 protocol jsons: tag=REAL, placebo seeds 1000–1019 (20, "
              "label-shuffle), swap seeds 2000–2004 (5, within-date), embargo_val=False, "
              "merge_dropped_rows=0 (incl. shifted: text arms fitted once on primary, §2.3, "
              "complete row set handed over).")
    md.append(f"- Per horizon: n_test=672, 143 call-date clusters, 461 entities, n_val=333; "
              f"HAC lag L_n = {hl['3']}/{hl['7']}/{hl['15']}/{hl['30']} (h=3/7/15/30); "
              f"STPEV (expanding) test prior coverage "
              f"{r0['horizons']['3']['coverage_stpev_prior_test']:.1%}.")
    sef = r0["horizons"]["3"]["r_ar"]["placebo"]["swap_effective_row_frac_test"]
    md.append(f"- within-date swap effective swapped-row fraction (test) = {sef:.1%} (single-call dates unswapped, "
              f"§6.3 disclosure).\n")

    # §8 verbatim
    md.append("## 0b. Decision-ladder original text (prereg §8, verbatim quote)\n")
    md.append(s8 + "\n")

    # published context
    md.append("## 1. published-convention context block (G1/G2, descriptive; §6.2: enters neither Holm nor win wording)\n")
    md.append(f"- **G1 PASS**: tfidf 12/12 cells text standalone beats raw V_past^(n) "
              f"(pooled {pub['arms']['tfidf']['pooled_mse_text']:.4f} vs raw "
              f"{pub['raw_vpast']['pooled_mse']:.4f}); prompted_qwen 4/12 (pooled "
              f"{pub['arms']['prompted_qwen']['pooled_mse_text']:.4f}, pooled loses to raw). "
              f"qwen_emb has no published reading (G1 needs only ≥1 arm; finality note quoted verbatim in the json).")
    md.append(f"- **G2 PASS**: our raw V_past pooled MSE(v) = "
              f"{pub['G2']['our_raw_vpast_mse_pooled']:.4f}, vs the Yu et al. reference 1.12, "
              f"ratio {pub['G2']['ratio_pooled']:.3f} ∈ [1/3, 3] (order-of-magnitude gate; per-horizon ratios "
              + ", ".join(f"h{h}={pub['G2']['ratio_per_horizon'][h]:.2f}" for h in HORIZONS)
              + "; panels differ, no equivalence gate, §0-4 forbids claiming direct comparability).")
    md.append(f"- inference_note verbatim: \"{pub['inference_note']}\"\n")
    md.append("**§6.4 conversion inputs from published-convention gains (per horizon, three yearly panels row-equal-weight pooled)**:\n")
    md.append("| arm | h | raw pooled MSE | text pooled MSE | Δ_pub = raw−text | % of raw |")
    md.append("|---|---|---|---|---|---|")
    for arm in ("tfidf", "prompted_qwen"):
        for h in HORIZONS:
            g = gains[arm][h]
            md.append(f"| {arm} | {h} | {g['raw_pooled']:.4f} | {g['text_pooled']:.4f} | "
                      f"{g['delta_pub']:+.4f} | {g['rel_of_raw_pct']:+.1f}% |")
    md.append("")
    md.append("**Conversion note (§6.4, original wording \"converted into the same units\")**: the unit of MDE(ent) is '% of the entity-stage reference "
              "MSE_Re' (MDE = (1.96+0.84)·SE_date/MSE_Re·100). Hence the published-convention text gain "
              "(text-alone vs raw V_past^(n), the absolute row-equal-weight pooled "
              "ΔMSE over the published three yearly panels, v² units) is converted to **G_conv(cell) = 100·Δ_pub(arm,h) / MSE_Re(cell)**, "
              "directly comparable to that cell's MDE(ent) with the same denominator and units. Verdict: Δ_pub≤0 → this arm/horizon has "
              "no published-convention gain to absorb; MDE(ent) ≤ G_conv → powered, \"fully absorbed\" wording licensed; "
              "MDE(ent) > G_conv → wording downgrades to \"underpowered to rule out\". Note: the published "
              "panels (2015/2016/2017-18, three yearly panels) and the audit primary test (2017-05..2018-06) "
              "differ in period and level — transporting absolute ΔMSE across panels is an inherent approximation of this conversion, disclosed as-is; "
              "the shifted alignment's MSE_Re is the value under shifted labels, same conversion formula.\n")

    # ---- headline: F2 + MDE side-by-side
    md.append("## 2. Per-cell F2 (identity-controlled residual, row5) + MDE side-by-side — headline three arms\n")
    md.append("F2 pass = DM<0 and Holm(8)<.05 and G4 (label-shuffle) clean; G4b (within-date swap) "
              "is diagnostic; dirty cells do not enter prose claims (§6.3, Yelp precedent thresholds |mean DM|<2.0, mean p>.05).\n")
    for arm in HEADLINE:
        for al in ALIGNS:
            md.append(f"### {arm} / {al}\n")
            md.append("| h | ref | row5 rel% | DM | p raw | p Holm(8) | 95% CI (date-block) | G4 | G4b | share d4/d3 | MDE(ent)% | MSE_Re | Δ_pub | G_conv% | §6.4 verdict |")
            md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for (h, ref, mv) in mde_table[(arm, al)]:
                c = runs[(arm, al)]["horizons"][h][ref]
                blk = c["residual"]
                win, neg, g4, dirty = stage_flags(blk, c["placebo"]["row5_shuffle"],
                                                  c["placebo"]["row5_swap"])
                ci = blk["bootstrap_ci_rel_pct"]
                share = c["entity"]["identity_share_pct"]
                mark = " **PASS**" if win else (" **NEG-SIG**" if neg else "")
                proxy = "(proxy: tfidf)" if mv["proxy"] else ""
                md.append(
                    f"| {h} | {ref} | {blk['delta_rel_pct']:+.2f}%{mark} | "
                    f"{blk['dm']:+.2f} | {blk['p']:.4f} | {blk['p_holm8']:.4f} | "
                    f"[{ci['ci_lo']:+.2f}, {ci['ci_hi']:+.2f}] | "
                    f"{'pass' if g4 else 'FAIL'} | {'dirty' if dirty else 'clean'} | "
                    f"{(f'{share:.1f}%' if isnum(share) else 'NaN')} | "
                    f"{mv['mde_ent']:.2f}% | {mv['mse_re']:.4f} | "
                    f"{mv['delta_pub']:+.4f}{proxy} | {mv['g_conv']:+.1f}% | {mv['verdict']} |")
            md.append("")

    # ---- full per-cell table (row3 + row5)
    md.append("## 3. Full per-cell table (row3 combination increment + row5 residual; probe is a diagnostic row)\n")
    for arm in ARMS:
        for al in ALIGNS:
            r = runs[(arm, al)]
            tag = "(diagnostic, §6.2: enters neither Holm nor win)" if arm == "identity_probe" else ""
            md.append(f"### {arm} / {al} {tag}\n")
            md.append("| h | ref | row3 rel% | DM | p raw | p Holm | 95% CI | r3 shuffle | r3 swap | row5 rel% | DM | p raw | p Holm | 95% CI | r5 shuffle | r5 swap | share | MDE(AR)% | MDE(ent)% |")
            md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for h in HORIZONS:
                for ref in REFS:
                    c = r["horizons"][h][ref]
                    c3, c5 = c["combined"], c["residual"]
                    ci3, ci5 = c3["bootstrap_ci_rel_pct"], c5["bootstrap_ci_rel_pct"]
                    p3s, p3w = c["placebo"]["row3_shuffle"], c["placebo"]["row3_swap"]
                    p5s, p5w = c["placebo"]["row5_shuffle"], c["placebo"]["row5_swap"]
                    share = c["entity"]["identity_share_pct"]
                    holm3 = f"{c3['p_holm8']:.4f}" if arm != "identity_probe" else "— (diagnostic)"
                    holm5 = f"{c5['p_holm8']:.4f}" if arm != "identity_probe" else "— (diagnostic)"
                    md.append(
                        f"| {h} | {ref} | {c3['delta_rel_pct']:+.2f}% | {c3['dm']:+.2f} | "
                        f"{c3['p']:.4f} | {holm3} | [{ci3['ci_lo']:+.2f},{ci3['ci_hi']:+.2f}] | "
                        f"dm{p3s['mean_dm']:+.2f}/p{p3s['mean_p']:.3f} | "
                        f"dm{p3w['mean_dm']:+.2f}/p{p3w['mean_p']:.3f} | "
                        f"{c5['delta_rel_pct']:+.2f}% | {c5['dm']:+.2f} | {c5['p']:.4f} | "
                        f"{holm5} | [{ci5['ci_lo']:+.2f},{ci5['ci_hi']:+.2f}] | "
                        f"dm{p5s['mean_dm']:+.2f}/p{p5s['mean_p']:.3f} | "
                        f"dm{p5w['mean_dm']:+.2f}/p{p5w['mean_p']:.3f} | "
                        f"{(f'{share:.1f}%' if isnum(share) else 'NaN')} | "
                        f"{c['mde']['ar_stage_rel_pct']:.2f} | "
                        f"{c['mde']['entity_stage_rel_pct']:.2f} |")
            md.append("")

    # ---- branch determination
    md.append("## 4. §8 branch determination (headline arms × alignments; primary governs paper wording, §2.3)\n")
    for arm in HEADLINE:
        for al in ALIGNS:
            v = verdicts[(arm, al)]
            r = runs[(arm, al)]
            md.append(f"### {arm} / {al}\n")
            md.append(f"- **F2 pass (DM<0 & Holm<.05 & G4): {len(v['f2_pass'])}/8**"
                      + ("; passing cells: " + ", ".join(
                          f"h{h} {ref} (share {fpos(s,1)}%)" for h, ref, s in v["f2_pass"])
                         if v["f2_pass"] else ""))
            if v["f2_neg"]:
                md.append("- **F2 significantly negative (DM>0 & Holm<.05)**: "
                          + ", ".join(f"h{h} {ref}" for h, ref in v["f2_neg"])
                          + " — text is significantly **harmful** after identity control; reported as-is.")
            md.append(f"- F1 combination-increment wins (DM<0 & Holm<.05 & G4): {len(v['f1_win'])}/8"
                      + (": " + ", ".join(f"h{h} {ref}" for h, ref in v["f1_win"])
                         if v["f1_win"] else ""))
            if v["f1_sig"]:
                md.append("- F1 Holm-significant cells and their identity share: "
                          + ", ".join(f"h{h} {ref}: {fpos(s,1) if isnum(s) else 'NaN'}%"
                                      + (" [negative direction]" if dm > 0 else "")
                                      for h, ref, s, dm in v["f1_sig"]))
            if v["discord"]:
                md.append("- **Clause-(a) (strict per-cell reading) non-qualifying cells**: "
                          + "; ".join(
                              f"h{h} {ref} (F1 Holm={fpos(p,4)}, share="
                              f"{fpos(s,1) if isnum(s) else 'NaN'}%"
                              + (", negative direction" if dm > 0 else "") + ")"
                              for h, ref, s, p, dm in v["discord"]))
            md.append(f"- **Fired branch (strict per-cell reading): {v['fired']}**"
                      + (f"; under the headline-R-AR reading = {v['alt']}"
                         if v["alt"] != v["fired"] else " (both readings agree)"))
            md.append("")

    # ---- §6.4 per-arm wording verdict
    md.append("## 5. §6.4 MDE discipline verdicts (per arm; null cells = all F2 non-pass cells)\n")
    for arm in HEADLINE:
        for al in ALIGNS:
            rows = mde_table[(arm, al)]
            null_rows = [(h, ref, mv) for h, ref, mv in rows]
            n_pow = sum(1 for _, _, mv in null_rows
                        if mv["verdict"].startswith("powered"))
            n_und = sum(1 for _, _, mv in null_rows
                        if mv["verdict"].startswith("UNDERPOWERED"))
            n_nog = sum(1 for _, _, mv in null_rows
                        if mv["verdict"].startswith("no published gain"))
            proxy = any(mv["proxy"] for _, _, mv in null_rows)
            if n_und == 0 and n_nog == 0:
                word = "\"fully absorbed\" (relative to the published-convention gain) wording licensed: all 8 cells MDE(ent) ≤ G_conv"
            elif n_und == 0 and n_nog > 0:
                word = (f"cells with a published-convention gain ({n_pow}) are all powered; \"absorbed\" wording licensed at these "
                        f"cells; {n_nog} cells have a negative published-convention gain for this arm (text loses to raw) — nothing to absorb")
            else:
                word = (f"**must downgrade**: {n_und} cells MDE(ent) > G_conv → "
                        f"\"underpowered to rule out\"")
            md.append(f"- **{arm} / {al}**: powered {n_pow}/8, no-gain {n_nog}/8, "
                      f"underpowered {n_und}/8 → {word}"
                      + (" (qwen_emb has no published reading of its own; its G_conv uses the tfidf gain as PROXY, "
                         "see §7-5)" if proxy else "") + ".")
    md.append("\nNote: the §6.4 downgrade rule uses 'the repriced published-convention gain' as the comparator; the row5 "
              "residuals observed at each cell are themselves generally < MDE(ent) (see the §2 table), i.e. this design is underpowered to "
              "rule out residuals at the **observed residual magnitude** — wherever prose writes \"no residual\", the MDE must appear alongside (§6.4 first sentence), and negative conclusions may only target "
              "'residuals at the magnitude of the published-convention gain'.\n")

    # ---- probe readout
    md.append("## 6. identity probe readout (diagnostic, §6.2; OPEN-7: ticker+comnam+date, no transcript)\n")
    md.append("probe share (pre-declared in prereg §5) = probe combination increment / fulltext (its mirror arm = "
              "prompted_qwen) combination increment, a ratio of d3 (same reference, same denominator, hence equal to the ratio of rel%); "
              "undefined when the denominator d3≤0 (n/a). The comparison columns vs the two fitted arms are context (not pre-declared readouts).\n")
    for al in ALIGNS:
        md.append(f"### {al}\n")
        md.append("| h | ref | probe row3 rel% | prompted row3 rel% | probe share (vs prompted) | vs tfidf | vs qwen_emb |")
        md.append("|---|---|---|---|---|---|---|")
        for h in HORIZONS:
            for ref in REFS:
                pr = runs[("identity_probe", al)]["horizons"][h][ref]["combined"]["delta_rel_pct"]
                cols = []
                for tgt in ("prompted_qwen", "tfidf", "qwen_emb"):
                    ft = runs[(tgt, al)]["horizons"][h][ref]["combined"]["delta_rel_pct"]
                    cols.append(f"{100.0 * pr / ft:.0f}%" if ft > 0 else "n/a (d3≤0)")
                pq = runs[("prompted_qwen", al)]["horizons"][h][ref]["combined"]["delta_rel_pct"]
                md.append(f"| {h} | {ref} | {pr:+.2f}% | {pq:+.2f}% | {cols[0]} | "
                          f"{cols[1]} | {cols[2]} |")
        md.append("")
    md.append("**probe honest readout**: under primary the zero-content probe's combination increment (r_ar: +8.55/+5.09/"
              "+1.04/−0.10%) at h3/h7/h15 is **no lower than and even exceeds** its mirror fulltext prompted arm "
              "(+0.19/−0.81/−3.29/+0.41%) — the prompted arm's share ratio is meaningless because its denominator d3≈0; "
              "the honest conclusion is reported as paired increments: the gain available from the identity prior alone ≥ the gain after adding the full transcript. "
              "Against the fitted arms (context columns), the probe reproduces ~64–65% of their combination increment (h3 r_ar), "
              "~29–33% (h7 r_ar), decreasing with horizon (see table).\n")

    # ---- disclosures
    md.append("## 7. Disclosures\n")
    md.append("1. **Alignment discipline (§2.3)**: primary (strict post-call window) governs paper wording; shifted "
              "(day-0 shift) differences:")
    md.append("   - prompted_qwen/shifted shows **significantly negative** cells at h15 (row5: r_ar "
              "Holm=0.0093, r_har Holm=0.0433, DM>0; row3: r_ar Holm=0.0221, r_har "
              "Holm=0.0472) — the text arm significantly **harms** the combination under this alignment; the same cells under primary are merely negative "
              "and not significant. Reported as-is; primary wording unchanged, but claim qualifiers are written per §2.3.")
    md.append("   - tfidf, qwen_emb: F2 is 0/8 under both alignments, determination structure unchanged; the strict clause-(a) "
              "non-qualifying cell moves with alignment (tfidf: primary h7 r_har → shifted h3 r_har; "
              "qwen_emb: primary none → shifted h3 r_har), i.e. under shifted the two arms move from/toward the MIXED "
              "boundary — under the headline-R-AR reading both arms are (a) under both alignments.")
    md.append("2. **Interpretation of share>100% and NaN**: share = d4/d3 = (MSE_R−MSE_Re)/(MSE_R−MSE_U). "
              ">100% = the MSE reduction from STPEV control alone exceeds the reduction from the text combination (the fitted arms' r_ar cells are generally "
              "111–145%); NaN = d3≤0 (no combination increment exists, division undefined; most prompted cells, probe "
              "h30); exploded values (prompted primary h3 r_ar 7890%, h30 r_ar 2807%, h30 r_har "
              "3955%) = artifacts of denominator d3≈0, not cited as readouts.")
    md.append("3. **G4b (within-date swap) dirty cells (mechanical enumeration, Yelp borderline rule)**: "
              "under primary, row3_swap is significant/borderline at tfidf (h3 both refs, h7 both refs, h15 r_ar), qwen_emb "
              "(h3 r_ar, h3 r_har p=.070 borderline, h7 r_ar, h15 r_ar), probe (h3/h7 r_ar) "
              "— after the swap permutation the combination still significantly beats the reference, showing the row3 combination increment contains a **date-common "
              "component** (within-date permutation does not destroy call-date-level information); row5_swap dirty: tfidf h3/h7 "
              "r_ar (p=.047/.043), qwen_emb h3 r_ar (p=.019), h7 r_ar (|DM|=2.05), probe "
              "h3 r_ar, etc. Per §6.3, any swap-borderline-dirty cell does not enter prose claims — this table has no F2 win, "
              "and F1's r_ar wins are all swap-dirty, so **no cell qualifies for \"win\" prose**; "
              "full flags in the CSV `gate` column.")
    md.append("4. **Single-shot provenance**: the generated timestamps of the 9 jsons are in the §0 table; all generated in one pass "
              "(2026-07-15 14:37–14:38), force_rerun_reason all null; protocol reference predictions match the fitted "
              "half row-by-row (max|Δpred|<1e-8, the prediction-level gate after the 2026-07-15 bug-fix).")
    md.append("5. **qwen_emb's §6.4 comparator gap (PROXY disclosure)**: the published scorer ran only "
              "tfidf and prompted_qwen (G1 needs only ≥1 arm through the gate); qwen_emb has no published-convention gain of its own; "
              "its G_conv uses **tfidf's Δ_pub as proxy**. Directionality: the tfidf gain is the largest of the three arms, "
              "so the proxy is biased high → the \"powered\" conclusion is **non-conservative** for this arm; the tightest cell is h30 r_ar "
              "(primary: MDE(ent)=10.07% vs G_conv≈17.4%, margin ≈1.7×) — if qwen_emb's "
              "true published-convention gain is below ~58% of tfidf's, that cell flips to underpowered. This gap is honestly "
              "written into the Limitations.")
    md.append("6. **Clause-(a) reading disclosure**: \"identity share ≥100% or the combination increment itself not significant\" "
              "is read **strictly per cell** (share≥100% required at every F1-Holm-significant cell), aligned with (b)'s "
              "\"at these horizons\" construction; the headline-R-AR reading (the share clause looks only at the OPEN-3 "
              "headline reference) is reported alongside; where the two disagree see §4 — wording takes the weakest defensible "
              "form per the §8 MIXED clause. Significance = Holm(8) two-sided (direction listed separately).")
    # oracle injection: mechanical count (no assertion by hand)
    n_ad = n_det = 0
    inj_fail = []
    for arm in ARMS:
        for al in ALIGNS:
            for h in HORIZONS:
                for ref in REFS:
                    inj = runs[(arm, al)]["horizons"][h][ref]["injection"]
                    ad = next(t for t in inj["targets"] if t["adaptive"])
                    n_ad += 1
                    if ad["ar"]["detect"]:
                        n_det += 1
                    else:
                        inj_fail.append((arm, al, h, ref, ad["converged"],
                                         ad["achieved_rel_pct"]))
    fail_txt = ("" if not inj_fail else "; undetected cells: " + "; ".join(
        f"{arm}/{al} h{h} {ref} (kappa calibration did not converge, converged={c}, "
        f"achieved={a:.0f}% — mechanical artifact of κ/g exploding when g_text≈0, disclosed as-is)"
        for arm, al, h, ref, c, a in inj_fail))
    md.append("7. **oracle injection (power-side evidence)**: disclosure quoted verbatim — \"ORACLE injection — "
              "s uses test labels BY DESIGN; power calibration only, never citable as "
              f"forecast performance\"; adaptive targets detected at the AR stage in {n_det}/{n_ad} cells"
              f"{fail_txt}; numbers serve power calibration only and enter no claims.")
    md.append("8. **probe Holm values**: the protocol json also mechanically computed p_holm8 for the probe, "
              "but per §6.2 the probe is a diagnostic row; this table never displays nor cites its Holm (the §3 tables use \"— (diagnostic)\" "
              "as placeholder), and the CSV gate column is always diagnostic.")
    md.append("")

    OUT_MD.write_text("\n".join(md))

    # -------------------------------------------------------------- console
    print(f"wrote {OUT_CSV}  ({len(csv_rows)} rows)")
    print(f"wrote {OUT_MD}")
    print("\n=== §8 branch verdicts (strict per-cell reading | headline-R-AR reading) ===")
    for arm in HEADLINE:
        for al in ALIGNS:
            v = verdicts[(arm, al)]
            extra = ""
            if v["discord"]:
                extra = "  discord: " + "; ".join(
                    f"h{h} {ref} (F1 Holm={p:.4f}, share="
                    f"{(f'{s:.1f}%' if isnum(s) else 'NaN')}"
                    + (", neg-dir" if dm > 0 else "") + ")"
                    for h, ref, s, p, dm in v["discord"])
            print(f"{arm:14s}/{al:8s}: F2 {len(v['f2_pass'])}/8, F1 wins "
                  f"{len(v['f1_win'])}/8, neg-sig F2 {len(v['f2_neg'])} -> "
                  f"{v['fired']}" + (f"  [alt: {v['alt']}]"
                                     if v["alt"] != v["fired"] else "") + extra)
    print("\n=== §6.4 MDE discipline ===")
    for arm in HEADLINE:
        for al in ALIGNS:
            rows = mde_table[(arm, al)]
            n_pow = sum(mv["verdict"].startswith("powered") for _, _, mv in rows)
            n_und = sum(mv["verdict"].startswith("UNDERPOWERED") for _, _, mv in rows)
            n_nog = sum(mv["verdict"].startswith("no published") for _, _, mv in rows)
            print(f"{arm:14s}/{al:8s}: powered {n_pow}/8, no-gain {n_nog}/8, "
                  f"underpowered {n_und}/8")


if __name__ == "__main__":
    main()
