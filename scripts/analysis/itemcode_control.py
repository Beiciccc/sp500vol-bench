"""Prereg IC (configs/prereg_residual_family_audit.md, tag prereg-rfa-v1.1) —
item-code / earnings-window control for the 8-K residual. Zero GPU.

Object: C6_llmtext (Qwen3-32B, SINGLE seed 2026 — prereg correction: C6 entered with
near-deterministic single-seed decoding, no 3-seed ensemble), event_driven, 3 horizons.

Question: does the 8-K residual over the firm-identity-augmented reference survive an
earnings-window control? has_202 = 1[item_subtype contains "2.02"] (comma-separated
item list carried on predictions.parquet, 0% null) enters the log-space combiner
LINEARLY (dummies are NOT log-transformed):
    restricted   R = [1, L(fh), L(fid), has202]
    unrestricted U = R + [L(ft)]
Val-fit, test-frozen, exp back-transform, QLIKE, day-clustered DM (HAC lag h-1, HLN)
— machinery verbatim from the M1 block of scripts/analysis/crossfamily_llama70.py.
Holm within the pre-declared 3-test family (3 horizons x this one augmented reference).

Secondary spec (REPORT ONLY, not in the decision): dummies for the top-8 individual
items by (train+val)-row frequency (item_subtype parsed by comma, membership counts),
each entering linearly in both R and U. Train rows carry no C6 predictions, so the
frequencies are counted on the A2_har_rv event_driven panel's train+val rows
(item_subtype verified identical to C6's on every merged val/test key).

SANITY ANCHOR (HARD RULE — abort before writing tables on failure): the plain
firm-identity numbers for qwen event_driven (results/tables/crossfamily_llm.csv rows
family=qwen3_32b, disc=event_driven: rel_firm, dm_firm, p_firm) must reproduce on
this code path to machine precision (rtol 1e-12).

Verdict (pre-registered): >=2/3 horizons DM<0 & Holm<.05 under the has_202-augmented
reference -> "not an earnings-window artefact"; otherwise the demotion branch (the
8-K residual is (partly) an earnings-window effect). Both branches go in the paper.

Run from repo root:  .venv/bin/python scripts/analysis/itemcode_control.py
Outputs (NEW files): results/tables/itemcode_control.{csv,md}
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "scripts/analysis")
import forecast_combination as fc  # noqa: E402
import clustered_dm as cdm  # noqa: E402

KEY = ["ticker", "accession", "horizon_days"]
EPS = 1e-8
HORIZONS = (5, 10, 20)
RTOL = 1e-12  # machine-precision anchor gate for CSV float round-trip
DISC = "event_driven"
RUN = "C6_llmtext"          # Qwen3-32B, single seed 2026
FAM = "qwen3_32b"
N_TOP = 8

ANCHOR_COLS = ["n_test", "rel_firm", "dm_firm", "p_firm"]


def ols(y, X):  # verbatim from crossfamily_llama70.py
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def holm(ps):  # verbatim from crossfamily_llama70.py
    ps = np.asarray(ps, float)
    n = len(ps)
    order = np.argsort(ps)
    out = np.empty(n)
    for rank, idx in enumerate(order):
        out[idx] = ps[idx] * (n - rank)
    run = 0.0
    for idx in order:
        run = max(run, out[idx])
        out[idx] = min(run, 1.0)
    return out


def close(a, b):  # verbatim from crossfamily_llama70.py
    a, b = float(a), float(b)
    if np.isnan(a) and np.isnan(b):
        return True
    return abs(a - b) <= RTOL * max(abs(a), abs(b), 1.0)


L = lambda x: np.log(np.clip(x, EPS, None))  # noqa: E731


def _lincomb(b, cols):
    """Accumulate b[0]*cols[0] + b[1]*cols[1] + ... strictly left-to-right, matching
    the committed M1 block's `bR[0] + bR[1]*x1 + bR[2]*x2` evaluation order bitwise
    (cols[0] is the ones column; 0+b0*1.0 == b0 exactly)."""
    acc = 0.0
    for coef, col in zip(b, cols):
        acc = acc + coef * col
    return acc


def combo_qlike(v, te, extras, h):
    """Log-space nested combiner with EXTRA columns entering LINEARLY in both R and U.
    Fit on val, frozen on test, exp back-transform; returns (rel%, dm, p, n_days, g_text).
    extras: list of (val-array, test-array) linear covariates (may be [])."""
    y = te.label_realised_vol.values
    fm = v.groupby("ticker").label_realised_vol.mean()
    gmean = v.label_realised_vol.mean()
    fid_v = v.ticker.map(fm).fillna(gmean).values
    fid_t = te.ticker.map(fm).fillna(gmean).values
    ly = L(v.label_realised_vol.values)
    Rv = [np.ones(len(v)), L(v.fh.values), L(fid_v)] + [ev for ev, _ in extras]
    Rt = [np.ones(len(te)), L(te.fh.values), L(fid_t)] + [et for _, et in extras]
    bR = ols(ly, np.column_stack(Rv))
    bU = ols(ly, np.column_stack(Rv + [L(v.ft.values)]))
    fR = np.exp(_lincomb(bR, Rt))
    fU = np.exp(_lincomb(bU, Rt + [L(te.ft.values)]))
    qR, qU = fc.qlike(y, fR), fc.qlike(y, fU)
    rel = 100 * np.mean(qR - qU) / np.mean(qR)
    dm, p, nd = cdm.dm_test_clustered(qU, qR, te.effective_trading_day.values, h)
    return rel, dm, p, nd, float(bU[-1])


def main():
    a2_full = fc.load("A2_har_rv", DISC)
    a2 = a2_full[KEY + ["split", "label_realised_vol", "prediction_realised_vol",
                        "effective_trading_day"]] \
        .rename(columns={"prediction_realised_vol": "fh"})
    p = fc.load(RUN, DISC)
    t = p[KEY + ["prediction_realised_vol", "item_subtype"]].rename(
        columns={"prediction_realised_vol": "ft"})

    # ---- item_subtype consistency: A2 vs C6 on every merged val/test key ----
    chk = a2_full[a2_full.split != "train"][KEY + ["item_subtype"]].merge(
        p[KEY + ["item_subtype"]], on=KEY, suffixes=("_a2", "_c6"),
        validate="one_to_one")
    if not bool((chk.item_subtype_a2 == chk.item_subtype_c6).all()):
        print("SANITY FAIL: item_subtype differs between A2 and C6 on merged keys")
        sys.exit(1)
    n_null = int(t.item_subtype.isna().sum())
    if n_null:
        print(f"SANITY FAIL: item_subtype has {n_null} nulls (prereg says 0%)")
        sys.exit(1)

    # ---- top-8 items by (train+val)-row frequency (A2 panel; C6 has no train rows) ----
    tv = a2_full[a2_full.split.isin(["train", "val"])]
    freq = tv.item_subtype.str.split(",").explode().str.strip().value_counts()
    top8 = list(freq.head(N_TOP).index)
    print(f"top-{N_TOP} items by (train+val)-row frequency "
          f"({len(tv)} rows, 0% null): "
          + ", ".join(f"{c} ({int(freq[c])})" for c in top8))

    ref = pd.read_csv("results/tables/crossfamily_llm.csv")
    ref = ref[(ref.family == FAM) & (ref.disc == DISC)]

    rows, anchor_bad = [], []
    for h in HORIZONS:
        # ---- same merged val/test frames as the M1 block (+ item_subtype carried) ----
        m = a2[a2.horizon_days == h].merge(t[t.horizon_days == h], on=KEY).dropna()
        m = m.copy()
        m["has_202"] = m.item_subtype.str.contains("2.02", regex=False) \
            .astype(float).values
        items = m.item_subtype.str.split(",").apply(
            lambda lst: {x.strip() for x in lst})
        for c in top8:
            m[f"it_{c}"] = items.apply(lambda s, c=c: float(c in s)).values
        v, te = m[m.split == "val"], m[m.split == "test"]

        # ---- sanity anchor: plain firm-identity spec, must == committed numbers ----
        rel0, dm0, p0, nd, g0 = combo_qlike(v, te, [], h)
        r = ref[ref.h == h].iloc[0]
        mine = dict(n_test=len(te), rel_firm=rel0, dm_firm=dm0, p_firm=p0)
        for c in ANCHOR_COLS:
            if not close(mine[c], r[c]):
                anchor_bad.append((h, c, float(mine[c]), float(r[c])))

        # ---- primary: has_202-augmented reference ----
        ex_v = [(v.has_202.values, te.has_202.values)]
        rel2, dm2, p2, _, g2 = combo_qlike(v, te, ex_v, h)

        # ---- secondary (report only): top-8 item dummies ----
        ex8 = [(v[f"it_{c}"].values, te[f"it_{c}"].values) for c in top8]
        rel8, dm8, p8, _, g8 = combo_qlike(v, te, ex8, h)

        rows.append(dict(
            disc=DISC, family=FAM, h=h, n_test=len(te), n_days=nd,
            frac_has202_test=float(te.has_202.mean()),
            frac_has202_val=float(v.has_202.mean()),
            rel_firm=rel0, dm_firm=dm0, p_firm=p0, g_text_firm=g0,
            rel_202=rel2, dm_202=dm2, p_202=p2, g_text_202=g2,
            rel_top8=rel8, dm_top8=dm8, p_top8=p8, g_text_top8=g8,
        ))

    if anchor_bad:
        print(f"SANITY ANCHOR FAIL ({len(anchor_bad)} mismatches vs committed "
              f"crossfamily_llm.csv qwen event_driven firmID cells):")
        for b in anchor_bad:
            print("  ", b)
        sys.exit(1)
    print(f"SANITY ANCHOR PASS: committed qwen {DISC} firmID cells reproduced to "
          f"machine precision (rtol {RTOL:g}) on columns {ANCHOR_COLS}")

    df = pd.DataFrame(rows)
    # ---- pre-declared Holm(3): 3 horizons x the one has_202-augmented reference ----
    df["p_202_holm"] = holm(df.p_202.values)

    n_pass = int(((df.dm_202 < 0) & (df.p_202_holm < .05)).sum())
    if n_pass >= 2:
        verdict = (f"**NOT an earnings-window artefact.** The 8-K residual survives "
                   f"the has_202-augmented firm-identity reference in {n_pass}/3 "
                   f"horizons (DM<0 & Holm<.05, pre-declared Holm(3), "
                   f"day-clustered).")
        branch = "not an earnings-window artefact"
    else:
        verdict = (f"**DEMOTION BRANCH.** Only {n_pass}/3 horizons survive the "
                   f"has_202-augmented reference (DM<0 & Holm<.05, Holm(3)) — per "
                   f"the prereg the 8-K residual is (partly) an earnings-window "
                   f"effect: the residual paragraph is demoted and the abstract's "
                   f"'what survives' sentence weakened accordingly.")
        branch = "demotion branch (earnings-window effect)"

    md = [
        "# Prereg IC — item-code / earnings-window control for the 8-K residual "
        "(Qwen3-32B, event_driven)",
        "",
        "## Disclosures",
        "",
        "- Object: C6_llmtext (Qwen3-32B, **single seed 2026** — prereg correction: "
        "C6 entered with near-deterministic single-seed decoding, no 3-seed "
        "ensemble), event_driven, 3 horizons.",
        "- `has_202 = 1[item_subtype contains \"2.02\"]` (string contains; "
        "item_subtype is a comma-separated item list carried on "
        "predictions.parquet, 0% null). Dummies enter the log-space combiner "
        "**LINEARLY** (not log-transformed) in BOTH the restricted and "
        "unrestricted design matrices: R = [1, L(fh), L(fid), has202]; "
        "U = R + [L(ft)]. Val-fit, test-frozen, exp back-transform; QLIKE; "
        "day-clustered DM (HAC lag h-1, HLN).",
        "- Holm within the pre-declared 3-test family (3 horizons x this one "
        "augmented reference).",
        f"- Secondary spec (REPORT ONLY, not in the decision): dummies for the "
        f"top-{N_TOP} individual items by (train+val)-row frequency, parsed from "
        f"item_subtype by comma (membership, not substring), each entering "
        f"linearly in both R and U. Frequencies counted on the A2_har_rv "
        f"event_driven panel's train+val rows ({len(tv)} rows; C6 predictions "
        f"carry no train rows); A2 and C6 item_subtype verified identical on "
        f"every merged val/test key. Top-{N_TOP}: "
        + ", ".join(f"{c} ({int(freq[c])})" for c in top8) + ".",
        "",
        "## Table — residual over the firm-identity reference, plain vs augmented",
        "",
        "rel% > 0 = text lowers QLIKE vs the reference; DM<0 = text helps; "
        "`**` = DM<0 & raw p<.05. The plain firmID columns are the sanity anchor "
        "(committed crossfamily_llm.csv values, reproduced).",
        "",
        "| h | n_test | test has_202 frac | rel% firmID | DM | rel% firmID+has202 | "
        "DM | raw p | Holm(3) p | rel% firmID+top8 (secondary) | DM | raw p |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for _, r in df.iterrows():
        s0 = "**" if (r.dm_firm < 0 and r.p_firm < .05) else ""
        s2 = "**" if (r.dm_202 < 0 and r.p_202 < .05) else ""
        s8 = "**" if (r.dm_top8 < 0 and r.p_top8 < .05) else ""
        md.append(
            f"| {int(r.h)} | {int(r.n_test)} | {r.frac_has202_test:.3f} | "
            f"{r.rel_firm:+.2f}%{s0} | {r.dm_firm:+.2f} | "
            f"{r.rel_202:+.2f}%{s2} | {r.dm_202:+.2f} | {r.p_202:.4g} | "
            f"{r.p_202_holm:.4g} | {r.rel_top8:+.2f}%{s8} | {r.dm_top8:+.2f} | "
            f"{r.p_top8:.4g} |")
    md += [
        "",
        "## VERDICT (pre-registered)",
        "",
        verdict,
        "",
        "## SANITY",
        "",
        f"- ANCHOR PASS: the committed plain firm-identity numbers for qwen "
        f"{DISC} (crossfamily_llm.csv, columns {ANCHOR_COLS}) reproduced on this "
        f"code path to machine precision (rtol {RTOL:g}) in 3/3 horizons.",
        "- item_subtype: 0% null on the C6 side; A2 vs C6 identical on all "
        f"{len(chk)} merged val/test keys (1:1 merge verified).",
        "",
    ]

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/tables/itemcode_control.csv", index=False)
    Path("results/tables/itemcode_control.md").write_text("\n".join(md))
    print("wrote results/tables/itemcode_control.csv/.md")
    print(df[["h", "n_test", "n_days", "frac_has202_test", "rel_firm", "dm_firm",
              "p_firm", "rel_202", "dm_202", "p_202", "p_202_holm",
              "rel_top8", "dm_top8", "p_top8"]].to_string(index=False))
    print("\nVERDICT:", verdict.replace("**", ""))
    print("branch fired:", branch)


if __name__ == "__main__":
    main()
