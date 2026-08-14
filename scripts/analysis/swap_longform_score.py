"""E-lf STEP 3/3 — the ONE-SHOT scoring of the long-form document-level swap.

Pre-registration: configs/prereg_swap_lf_and_anon.md §E-lf (tag prereg-ea-v1.0,
amended prereg-ea-v1.2 BEFORE any E-lf statistic).

COVERAGE (prereg-ea-v1.2): C2_finbert_s1 is ARTEFACT-LOST — its three horizon
checkpoints existed only on the box and were deleted (no local copy was ever
held); retraining a stand-in would violate the §E-lf zero-training principle,
so the arm is registered as not-executed. Scoring covers the arms PRESENT
(B2_tfidf_ridge + C5_qwen3); the pre-declared genuine-cell retention medians
are computed over those arms and the degradation is disclosed in the md.

Per long-form cell (arm x horizon; arms = B2_tfidf_ridge, C5_qwen3):
  real increment      : committed predictions, fc.log_combo (val-fit, test-frozen)
                        — asserted equal to forecast_combination_grid.csv where the
                        cell exists (the matched_firm_swap sanity convention);
  forecast-level swap : the committed mechanism (mfs.matched_swap on committed
                        forecasts) — asserted equal to matched_firm_swap.csv;
  document-level swap : FROZEN models re-inferred on swapped documents
                        (swap_longform_infer.py outputs), same combiner refit on
                        swapped val, day-clustered DM;
  retention           : swap increment / real increment (both levels reported).

PRE-DECLARED READOUT (prereg): median + quartiles of document-level retention over
the long-form HAR-genuine set — genuine_ens_vol of m1_ensemble_primary.csv (the
declared M1 primary) restricted to the re-inferred arms — side-by-side with the
committed ED / all-cell forecast-level swap numbers. All three pre-registered
branches are written into the md; the applicable one is marked.

GATES (all mandatory before any statistic is written):
  G1  machine-precision reproduction of the committed matched_firm_swap table on
      the SAME CODE PATH: scripts/analysis/matched_firm_swap.py is re-executed
      unmodified in a scratch workdir and its CSV compared to the committed one;
  G2  level-preservation: the pairing is rebuilt via swap_longform_build
      (imports mfs.matched_swap) and must equal the shipped manifest exactly;
      test swap fractions must equal committed swap_frac_test;
  G3  no-retraining: every infer meta must carry pass=True hash invariance, all
      arms must have used THIS manifest (sha256), and any locally present
      artefact is re-hashed and compared.

SINGLE-SHOT DISCIPLINE: refuses to run if results/tables/swap_longform.csv (or
.md) exists. --i-know-this-violates-prereg forces a rerun and REQUIRES
--rerun-reason; a RERUN DISCLOSURE block is appended to the md.

Validation modes (neither touches results/tables/swap_longform.*):
  --selftest    synthetic universe with a planted LEVEL-channel arm (retention~1)
                and a planted CONTENT arm (retention~0); full pipeline into a
                temp dir; asserts the planted structure is recovered.
  --gates-only  runs G1+G2+G3 only; writes nothing under results/tables.

Run (the one shot, from repo root, after BOTH infer arms B2 + C5):
    .venv/bin/python scripts/analysis/swap_longform_score.py
"""
from __future__ import annotations

# --- thread caps BEFORE numpy/pandas (<=4 cores local discipline) ------------
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, "scripts/analysis")
sys.path.insert(0, "src")

import forecast_combination as fc
import matched_firm_swap as mfs
import numpy as np
import pandas as pd
from clustered_dm import dm_test_clustered
from withindate_placebo import day_key, rel_pct

DISC = "long_form"
KEY, SORT, HORIZONS = fc.KEY, fc.SORT, fc.HORIZONS
# prereg-ea-v1.2: C2_finbert_s1 artefact-lost (checkpoints physically deleted on
# the box, never held locally) -> registered not-executed; score the arms present.
ARMS = ("B2_tfidf_ridge", "C5_qwen3")
C2_LOST_ARM = "C2_finbert_s1"
HAR_COLS = ["split"] + KEY + ["prediction_realised_vol", "label_realised_vol",
            "filing_time_utc", "effective_trading_day"]


def _fatal(msg: str) -> None:
    raise SystemExit(f"[swap_longform_score] FATAL: {msg}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Settings:
    root: Path = REPO                      # results/{runs,tables} live under here
    manifest: Path | None = None           # shipped manifest parquet
    seed: int = 2026
    arms: tuple = ARMS
    horizons: tuple = HORIZONS
    out_csv: Path = REPO / "results/tables/swap_longform.csv"
    out_md: Path = REPO / "results/tables/swap_longform.md"
    out_meta: Path = REPO / "results/tables/swap_longform_meta.json"
    skip_g1: bool = False                  # selftest only
    skip_g3: bool = False                  # selftest only
    require_committed_asserts: bool = True
    gate_notes: dict = field(default_factory=dict)

    def run_pq(self, run: str) -> Path:
        return self.root / "results/runs" / f"{run}_full_{DISC}_seed{self.seed}" / "predictions.parquet"

    def elf_pq(self, arm: str) -> Path:
        return (self.root / "results/runs" / f"ELF_swap_{arm}_full_{DISC}_seed{self.seed}"
                / "predictions_swapped.parquet")

    def elf_meta(self, arm: str) -> Path:
        return self.elf_pq(arm).with_name("predictions_swapped_meta.json")

    def table(self, name: str) -> Path:
        return self.root / "results/tables" / name


# --------------------------------------------------------------------------- #
# GATES                                                                        #
# --------------------------------------------------------------------------- #
def gate_g1_committed_reproduction(st: Settings) -> dict:
    """Re-execute the UNMODIFIED committed matched_firm_swap.py in a scratch
    workdir (symlinked scripts/src/results-runs + the two input tables) and
    compare its CSV to the committed one at machine precision."""
    committed = st.table("matched_firm_swap.csv")
    if not committed.exists():
        _fatal(f"G1: committed table missing: {committed}")
    with tempfile.TemporaryDirectory(prefix="elf_g1_") as td:
        work = Path(td)
        (work / "results" / "tables").mkdir(parents=True)
        os.symlink(REPO / "scripts", work / "scripts")
        os.symlink(REPO / "src", work / "src")
        os.symlink(st.root / "results" / "runs", work / "results" / "runs")
        for name in ("forecast_combination_grid.csv", "withindate_placebo.csv"):
            os.symlink(st.table(name), work / "results" / "tables" / name)
        proc = subprocess.run(
            [sys.executable, "scripts/analysis/matched_firm_swap.py"],
            check=False, cwd=work, capture_output=True, text=True, timeout=3600,
            env={**os.environ},
        )
        if proc.returncode != 0:
            _fatal(f"G1: committed-code rerun failed (rc={proc.returncode}):\n"
                   f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        fresh_path = work / "results" / "tables" / "matched_firm_swap.csv"
        byte_equal = fresh_path.read_bytes() == committed.read_bytes()
        fresh = pd.read_csv(fresh_path)
        ref = pd.read_csv(committed)
    if fresh.shape != ref.shape or list(fresh.columns) != list(ref.columns):
        _fatal(f"G1: shape/columns drifted: fresh {fresh.shape} vs committed {ref.shape}")
    max_abs = 0.0
    for c in ref.columns:
        if pd.api.types.is_numeric_dtype(ref[c]):
            a, b = fresh[c].to_numpy(float), ref[c].to_numpy(float)
            if not np.allclose(a, b, rtol=0.0, atol=1e-12, equal_nan=True):
                _fatal(f"G1: column {c} differs beyond 1e-12 "
                       f"(max abs diff {np.nanmax(np.abs(a - b)):.3e})")
            with np.errstate(invalid="ignore"):
                max_abs = max(max_abs, float(np.nanmax(np.abs(a - b))) if len(a) else 0.0)
        elif not (fresh[c].astype(str) == ref[c].astype(str)).all():
            _fatal(f"G1: non-numeric column {c} differs")
    print(f"[score] G1 PASS — committed matched_firm_swap.csv reproduced on the same "
          f"code path (byte-identical={byte_equal}, max numeric |Δ|={max_abs:.2e})")
    return {"pass": True, "byte_identical": bool(byte_equal), "max_abs_diff": max_abs}


def gate_g2_pairing(st: Settings) -> dict:
    """Rebuild the pairing (same imported mfs.matched_swap code) and require the
    shipped manifest to match EXACTLY; test swap fractions == committed."""
    import swap_longform_build as slb

    if st.manifest is None or not Path(st.manifest).exists():
        _fatal(f"G2: manifest missing: {st.manifest}")
    shipped = pd.read_parquet(st.manifest)
    har = pd.read_parquet(st.run_pq("A2_har_rv"))[slb.PANEL_COLS]
    rebuilt, stats = slb.build_manifest(har, horizons=st.horizons)
    cols = ["split", "horizon_days", "accession", "partner_accession", "swapped"]
    a = shipped[cols].sort_values(cols[:3], kind="mergesort").reset_index(drop=True)
    b = rebuilt[cols].sort_values(cols[:3], kind="mergesort").reset_index(drop=True)
    if len(a) != len(b) or not (a.values == b.values).all():
        _fatal("G2: shipped manifest != freshly rebuilt pairing — manifest is stale "
               "or was built on different data")
    checks = {}
    if st.require_committed_asserts:
        committed = pd.read_csv(st.table("matched_firm_swap.csv"))
        lf = committed[committed.disc == DISC]
        for s in stats:
            if s["split"] != "test":
                continue
            ref = float(lf[lf.h == s["h"]].swap_frac_test.unique()[0])
            if abs(ref - s["swap_frac"]) >= 1e-12:
                _fatal(f"G2: swap_frac mismatch h={s['h']}: {s['swap_frac']} vs {ref}")
            checks[f"test_h{s['h']}"] = s["swap_frac"]
    print(f"[score] G2 PASS — manifest pairing identical to rebuilt committed pairing "
          f"({len(a)} rows); level-preservation stats match the build")
    return {"pass": True, "swap_frac_test": checks,
            "level_preservation_stats": stats}


def gate_g3_no_retraining(st: Settings) -> dict:
    """Checkpoint-hash invariance: every infer meta pass=True; all arms used THIS
    manifest; locally present artefacts re-hashed."""
    manifest_sha = _sha256(Path(st.manifest))
    out = {"manifest_sha256": manifest_sha, "arms": {}}
    for arm in st.arms:
        mp = st.elf_meta(arm)
        if not mp.exists():
            _fatal(f"G3: infer meta missing for {arm}: {mp}")
        meta = json.loads(mp.read_text())
        if not meta.get("g3_hash_invariance", {}).get("pass"):
            _fatal(f"G3: {arm} infer meta does not record hash invariance PASS")
        if meta.get("manifest_sha256") != manifest_sha:
            _fatal(f"G3: {arm} was inferred against a DIFFERENT manifest "
                   f"({meta.get('manifest_sha256', '')[:12]}… vs {manifest_sha[:12]}…)")
        rehash = {}
        for name, rec in meta.get("artefacts", {}).items():
            pre = rec.get("sha256") or rec.get("sha256_pre")
            p = Path(rec.get("path", ""))
            if pre and p.exists():
                now = _sha256(p)
                if now != pre:
                    _fatal(f"G3: artefact {name} of {arm} changed since inference "
                           f"({pre[:12]}… -> {now[:12]}…)")
                rehash[name] = now
        out["arms"][arm] = {"meta_pass": True, "n_rehashed_ok": len(rehash)}
        print(f"[score] G3 {arm}: meta PASS, manifest match, "
              f"{len(rehash)} artefact(s) re-hashed unchanged")
    return out


# --------------------------------------------------------------------------- #
# cell scoring                                                                 #
# --------------------------------------------------------------------------- #
def _cell_frames(har, txt, h):
    d = har.merge(txt, on=KEY)
    dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(
        SORT, kind="mergesort").reset_index(drop=True)
    dt = d[(d.horizon_days == h) & (d.split == "test")].sort_values(
        SORT, kind="mergesort").reset_index(drop=True)
    return dv, dt


def score_cells(st: Settings) -> pd.DataFrame:
    grid = pd.read_csv(st.table("forecast_combination_grid.csv")).set_index(
        ["disc", "model", "h"])
    m1 = pd.read_csv(st.table("m1_ensemble_primary.csv")).set_index(
        ["disc", "model", "h"])
    committed_swap = pd.read_csv(st.table("matched_firm_swap.csv")).set_index(
        ["disc", "model", "h"])

    har = pd.read_parquet(st.run_pq("A2_har_rv"))[HAR_COLS].rename(
        columns={"prediction_realised_vol": "fhar"})
    rows = []
    for arm in st.arms:
        txt = pd.read_parquet(st.run_pq(arm))[KEY + ["prediction_realised_vol"]].rename(
            columns={"prediction_realised_vol": "ftext"})
        swp_path = st.elf_pq(arm)
        if not swp_path.exists():
            _fatal(f"swapped predictions missing for {arm}: {swp_path}")
        swp = pd.read_parquet(swp_path)[KEY + ["prediction_realised_vol"]].rename(
            columns={"prediction_realised_vol": "fdoc"})
        for h in st.horizons:
            dv, dt = _cell_frames(har, txt, h)
            if len(dv) < 100 or len(dt) < 30:
                continue
            yv, fhv, ftv = (dv.label_realised_vol.to_numpy(), dv.fhar.to_numpy(),
                            dv.ftext.to_numpy())
            yt, fht, ftt = (dt.label_realised_vol.to_numpy(), dt.fhar.to_numpy(),
                            dt.ftext.to_numpy())
            days_v, days_t = day_key(dv), day_key(dt)
            rv_map = dv.groupby("ticker")["label_realised_vol"].mean()
            g_mean = float(dv.label_realised_vol.mean())

            # ---- REAL (committed forecasts; grid sanity where the cell exists)
            fR, fU, _ = fc.log_combo(yv, fhv, ftv, fht, ftt)
            real_rel = rel_pct(fc.qlike(yt, fR), fc.qlike(yt, fU))
            in_grid = (DISC, arm, h) in grid.index
            if in_grid and st.require_committed_asserts:
                gref = float(grid.loc[(DISC, arm, h), "rel_impr_pct"])
                assert np.isclose(real_rel, gref, atol=1e-8), \
                    f"real-cell sanity fail {arm}/h{h}: {real_rel} vs grid {gref}"

            # ---- FORECAST-LEVEL swap (the committed mechanism, committed preds)
            ftv_s, _ = mfs.matched_swap(ftv, dv.ticker, days_v, rv_map, g_mean)
            ftt_s, frac_t = mfs.matched_swap(ftt, dt.ticker, days_t, rv_map, g_mean)
            sR, sU, _ = fc.log_combo(yv, fhv, ftv_s, fht, ftt_s)
            slR, slU = fc.qlike(yt, sR), fc.qlike(yt, sU)
            swap_rel_fc = rel_pct(slR, slU)
            dm_fc, p_fc, _ = dm_test_clustered(slU, slR, days_t, h)
            in_committed = (DISC, arm, h) in committed_swap.index
            if in_committed and st.require_committed_asserts:
                cref = committed_swap.loc[(DISC, arm, h)]
                assert np.isclose(swap_rel_fc, float(cref.swap_rel_pct), atol=1e-8), \
                    f"forecast-swap sanity fail {arm}/h{h}"

            # ---- DOCUMENT-LEVEL swap (re-inferred frozen models)
            dv2, dt2 = _cell_frames(har, swp, h)
            if (len(dv2) != len(dv) or len(dt2) != len(dt)
                    or not (dv2.accession.to_numpy() == dv.accession.to_numpy()).all()
                    or not (dt2.accession.to_numpy() == dt.accession.to_numpy()).all()):
                _fatal(f"{arm}/h{h}: swapped-prediction panel misaligned with committed")
            ftv_d, ftt_d = dv2.fdoc.to_numpy(), dt2.fdoc.to_numpy()
            dR, dU, _ = fc.log_combo(yv, fhv, ftv_d, fht, ftt_d)
            dlR, dlU = fc.qlike(yt, dR), fc.qlike(yt, dU)
            swap_rel_doc = rel_pct(dlR, dlU)
            dm_doc, p_doc, n_days = dm_test_clustered(dlU, dlR, days_t, h)
            retention_doc = swap_rel_doc / real_rel if real_rel > 0 else float("nan")
            retention_fc = swap_rel_fc / real_rel if real_rel > 0 else float("nan")

            # equivalence diagnostic (document-level vs forecast-level swap)
            eq_max = float(np.max(np.abs(np.concatenate([ftv_d - ftv_s, ftt_d - ftt_s]))))
            eq_corr = float(np.corrcoef(ftt_d, ftt_s)[0, 1])

            rows.append({
                "disc": DISC, "model": arm, "h": h,
                "n_test": len(dt), "n_days_test": n_days,
                "swap_frac_test": float(frac_t),
                "genuine_orig": bool(grid.loc[(DISC, arm, h), "genuine"]) if in_grid else None,
                "genuine_ens_vol": (bool(m1.loc[(DISC, arm, h), "genuine_ens_vol"])
                                    if (DISC, arm, h) in m1.index else None),
                "real_rel_pct": real_rel,
                "swap_rel_pct_doc": swap_rel_doc,
                "swap_dm_clu_doc": dm_doc, "swap_p_clu_doc": p_doc,
                "retention_doc": retention_doc,
                "swap_rel_pct_fc": swap_rel_fc,
                "swap_dm_clu_fc": dm_fc, "swap_p_clu_fc": p_fc,
                "retention_fc": retention_fc,
                "equiv_max_absdiff": eq_max, "equiv_corr_test": eq_corr,
                "in_committed_grid": in_grid,
            })
            print(f"[score] {arm:16s} h={h:2d}  real={real_rel:+.2f}%  "
                  f"doc-swap={swap_rel_doc:+.2f}% (ret={retention_doc:+.2f})  "
                  f"fc-swap={swap_rel_fc:+.2f}% (ret={retention_fc:+.2f})  "
                  f"equiv max|Δ|={eq_max:.2e}")
    return pd.DataFrame(rows)


def _quart(x: pd.Series) -> tuple[float, float, float]:
    x = x.dropna()
    if not len(x):
        return float("nan"), float("nan"), float("nan")
    return float(x.quantile(0.25)), float(x.median()), float(x.quantile(0.75))


def write_outputs(st: Settings, df: pd.DataFrame, gates: dict,
                  rerun_disclosure: str | None = None) -> dict:
    committed = pd.read_csv(st.table("matched_firm_swap.csv"))
    cgen = committed[committed.genuine & (committed.real_rel_pct > 0)]
    c_all_med = float(cgen.retention_vs_real.median())
    c_ed = cgen[cgen.disc == "event_driven"]
    c_lf = cgen[cgen.disc == "long_form"]

    # PRE-DECLARED set: m1_ensemble_primary genuine_ens_vol long_form cells of the
    # re-inferred arms with a positive real increment (committed swap convention)
    declared = df[(df.genuine_ens_vol == True) & (df.real_rel_pct > 0)]
    q1, med, q3 = _quart(declared.retention_doc)
    o_declared = df[(df.genuine_orig == True) & (df.real_rel_pct > 0)]
    oq1, omed, oq3 = _quart(o_declared.retention_doc)
    all_pos = df[df.real_rel_pct > 0]
    aq1, amed, aq3 = _quart(all_pos.retention_doc)

    branch = "a" if med < 0.5 else "b"
    rets = declared.retention_doc.dropna()
    mixed = bool(len(rets) and rets.min() < 0.4 and rets.max() > 0.6)

    df.to_csv(st.out_csv, index=False)

    md = ["# E-lf — long-form matched-firm swap, DOCUMENT-level re-inference "
          "(one-shot, prereg-ea-v1.0 + v1.2 amendment)\n",
          "Mechanism verbatim from the committed `matched_firm_swap.py` (same pairing "
          "function, same day key, deterministic single draw). NEW: long-form documents "
          "are exchanged between val-RV-matched within-day firm pairs and the FROZEN "
          "models (B2 TF-IDF vectorizer+ridge; C5 frozen-embed + heads reconstructed "
          "through the committed recipe's reproduction gate where the stored files were "
          "lost) are re-run on the swapped documents — zero training on swapped data. "
          "Retention = swapped increment / real increment; day-clustered DM.\n",
          "## Coverage disclosure (prereg-ea-v1.2)\n",
          "**C2_finbert_s1 is artefact-lost and coverage is degraded to B2_tfidf_ridge "
          "+ C5_qwen3 (prereg-ea-v1.2).** The C2 arm's three horizon checkpoints "
          "existed only on the GPU box and were deleted by a disk clean-up; no local "
          "copy was ever held. Retraining a stand-in would violate the §E-lf "
          "zero-training principle, so the arm was pre-registered as not-executed "
          "(v1.2 amendment, committed BEFORE any E-lf statistic). The document-level "
          "readout therefore lacks the deep fine-tuned lineage; the pre-declared "
          "genuine-cell retention medians below are computed over the arms present, "
          "and the committed FORECAST-level swap rows for C2 (matched_firm_swap.csv) "
          "remain in the side-by-side table as the only surviving C2 evidence.\n",
          f"**Gates**: G1 committed-table reproduction "
          f"{'PASS' if gates.get('g1', {}).get('pass') else 'SKIPPED (selftest)'}"
          f" · G2 pairing/level-preservation {'PASS' if gates.get('g2', {}).get('pass') else 'SKIPPED (selftest)'}"
          f" · G3 checkpoint-hash invariance "
          f"{'PASS' if gates.get('g3') else 'SKIPPED (selftest)'}\n",
          "## Per-cell grid (long_form; seed2026 forecasts, committed convention)\n",
          "| model | h | real rel% | doc-swap rel% | DM(clu) | p | retention(doc) | "
          "retention(fc-level) | genuine(m1-ens) | genuine(orig) | equiv max|Δf| |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.sort_values(["model", "h"]).iterrows():
        md.append(
            f"| {r.model} | {r.h} | {r.real_rel_pct:+.3f} | {r.swap_rel_pct_doc:+.3f} | "
            f"{r.swap_dm_clu_doc:+.2f} | {r.swap_p_clu_doc:.4f} | "
            f"{'' if pd.isna(r.retention_doc) else format(r.retention_doc, '+.2f')} | "
            f"{'' if pd.isna(r.retention_fc) else format(r.retention_fc, '+.2f')} | "
            f"{'—' if r.genuine_ens_vol is None else ('YES' if r.genuine_ens_vol else 'no')} | "
            f"{'—' if r.genuine_orig is None else ('YES' if r.genuine_orig else 'no')} | "
            f"{r.equiv_max_absdiff:.1e} |")

    md += ["\n## Pre-declared readout — document-level retention, long-form "
           "HAR-genuine set (m1_ensemble_primary.genuine_ens_vol), arms present "
           "(prereg-ea-v1.2)\n",
           f"- **median {med:.2f}** (Q1 {q1:.2f}, Q3 {q3:.2f}) over "
           f"**{len(declared)}** genuine cells of the re-inferred arms "
           f"(C5_qwen3 has no committed M1 cell — reported per-cell above, "
           f"excluded from this pre-declared median; C2_finbert_s1 artefact-lost, "
           f"see the coverage disclosure);",
           f"- original-grid genuine set (committed-swap convention): median "
           f"{omed:.2f} (Q1 {oq1:.2f}, Q3 {oq3:.2f}) over {len(o_declared)} cells;",
           f"- all re-inferred cells with positive real increment: median {amed:.2f} "
           f"(Q1 {aq1:.2f}, Q3 {aq3:.2f}) over {len(all_pos)} cells.\n",
           "## Side-by-side with the committed forecast-level swap "
           "(matched_firm_swap.csv)\n",
           "| readout | n cells | median retention |", "|---|---|---|",
           f"| committed, ALL genuine cells (forecast-level) | {len(cgen)} | "
           f"{c_all_med:.2f} |",
           f"| committed, event-driven genuine (forecast-level) | {len(c_ed)} | "
           f"{float(c_ed.retention_vs_real.median()):.2f} |",
           f"| committed, long-form genuine (forecast-level) | {len(c_lf)} | "
           f"{float(c_lf.retention_vs_real.median()):.2f} |",
           f"| **THIS ANALYSIS: long-form genuine, DOCUMENT-level re-inference "
           f"(arms present: {', '.join(st.arms)})** | "
           f"{len(declared)} | **{med:.2f}** |\n",
           "## Pre-registered branches (all three committed before any statistic; "
           "the applicable one is marked)\n",
           f"- {'**[APPLIES]** ' if branch == 'a' and not mixed else ''}(a) retention "
           f"median < 50% → the long-form increment is predominantly CONTENT "
           f"(consistent with the ED residual); 'who spoke' wording stays inside the "
           f"existing bracket framing, and the swap tension is reconciled explicitly "
           f"in one paragraph of the main text.",
           f"- {'**[APPLIES]** ' if branch == 'b' and not mixed else ''}(b) retention "
           f"median >= 50% → the level channel dominates; the identity narrative "
           f"receives direct support.",
           f"- {'**[APPLIES]** ' if mixed else ''}(c) mixed → reported cell-by-cell, "
           f"as-is.\n",
           f"**Verdict inputs**: median={med:.2f}; genuine-cell retentions "
           f"{[round(float(x), 2) for x in sorted(rets)]}; "
           f"mixed-flag={'YES' if mixed else 'no'}. Whatever the direction, this "
           f"table feeds the mandatory reconciliation paragraph.\n",
           "## Equivalence note\n",
           "For text-pure frozen models a document swap equals a forecast swap by "
           "construction; the re-inference DEMONSTRATES this through the actual "
           "frozen artefacts (`equiv max|Δf|` per cell above) instead of asserting "
           "it, and any deviation would have flagged hidden non-purity.\n",
           f"Single-shot: this file is written once "
           f"({datetime.now(UTC).isoformat(timespec='seconds')}); reruns "
           f"require --i-know-this-violates-prereg and a disclosure block."]
    if rerun_disclosure:
        md += ["\n## RERUN DISCLOSURE\n", rerun_disclosure]
    st.out_md.write_text("\n".join(md) + "\n")

    summary = {
        "prereg": "configs/prereg_swap_lf_and_anon.md §E-lf (prereg-ea-v1.0 + v1.2 amendment)",
        "declared_set": "m1_ensemble_primary.genuine_ens_vol & long_form & real_rel>0 "
                        "& re-inferred arms",
        "arms_present": list(st.arms),
        "c2_finbert_s1_status": "artefact-lost, not executed (prereg-ea-v1.2); "
                                "coverage degraded to B2+C5",
        "n_declared_cells": len(declared),
        "retention_doc_median": med, "retention_doc_q1": q1, "retention_doc_q3": q3,
        "retention_doc_median_origset": omed,
        "branch": ("c" if mixed else branch), "mixed_flag": mixed,
        "committed_forecast_level": {
            "all_genuine_median": c_all_med,
            "ed_genuine_median": float(c_ed.retention_vs_real.median()),
            "lf_genuine_median": float(c_lf.retention_vs_real.median()),
        },
        "gates": gates,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    st.out_meta.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[score] wrote {st.out_csv}")
    print(f"[score] wrote {st.out_md}")
    print(f"[score] HEADLINE: pre-declared document-level retention median = {med:.2f} "
          f"over {len(declared)} genuine long-form cells -> branch "
          f"{'c (mixed)' if mixed else branch}")
    return summary


def run(st: Settings, rerun_disclosure: str | None = None, force: bool = False) -> dict:
    if (st.out_csv.exists() or st.out_md.exists()) and not force:
        _fatal(f"the one shot has been fired — {st.out_csv} exists; a rerun requires "
               f"--i-know-this-violates-prereg AND --rerun-reason")
    gates = {}
    gates["g2"] = gate_g2_pairing(st)
    gates["g3"] = None if st.skip_g3 else gate_g3_no_retraining(st)
    gates["g1"] = {"skipped": True} if st.skip_g1 else gate_g1_committed_reproduction(st)
    df = score_cells(st)
    if df.empty:
        _fatal("no cells scored")
    return write_outputs(st, df, gates, rerun_disclosure)


# --------------------------------------------------------------------------- #
# selftest — synthetic universe, planted LEVEL vs CONTENT arms                 #
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    print("[selftest] synthetic planted-structure validation (temp dir only)")
    import swap_longform_build as slb

    rng = np.random.default_rng(7)
    firms = [f"F{i}" for i in range(8)]
    base = dict(zip(firms, np.geomspace(0.05, 0.6, len(firms)), strict=False))
    days = pd.bdate_range("2019-01-01", periods=360)
    splits = np.array(["train"] * 120 + ["val"] * 120 + ["test"] * 120)

    rows = []
    acc = 0
    for d, s in zip(days, splits, strict=False):
        shock = float(np.exp(rng.normal(0, 0.3)))
        for f in firms:
            idio = float(np.exp(rng.normal(0, 0.4)))
            rv = base[f] * shock * idio
            rows.append({"ticker": f, "accession": f"a{acc:05d}", "split": s,
                         "filing_time_utc": pd.Timestamp(d, tz="UTC"),
                         "effective_trading_day": d, "rv": rv,
                         "shock": shock, "idio": idio})
            acc += 1
    base_df = pd.DataFrame(rows)

    def per_h(df):
        out = []
        for h in (5, 10):
            x = df.copy()
            x["horizon_days"] = h
            x["label_realised_vol"] = x.rv * np.exp(rng.normal(0, 0.02, len(x)))
            out.append(x)
        return pd.concat(out, ignore_index=True)

    panel = per_h(base_df)
    # HAR: misses part of the firm level (so text level info is incremental)
    panel["fhar"] = (panel.ticker.map(base) ** 0.4) * panel.shock * np.exp(
        rng.normal(0, 0.15, len(panel)))
    # LEVEL arm: text forecast = the firm's level only (survives a level-matched swap)
    lvl = panel.ticker.map(base).to_numpy()
    # CONTENT arm: knows the idiosyncratic piece (dies under any swap)
    cont = (panel.ticker.map(base) * panel.shock * panel.idio ** 0.9).to_numpy() * np.exp(
        rng.normal(0, 0.05, len(panel)))

    with tempfile.TemporaryDirectory(prefix="elf_score_selftest_") as td:
        root = Path(td)
        (root / "results/tables").mkdir(parents=True)

        def write_run(name, pred):
            rd = root / "results/runs" / f"{name}_full_{DISC}_seed2026"
            rd.mkdir(parents=True)
            out = panel[["split"] + KEY + ["label_realised_vol", "filing_time_utc",
                                           "effective_trading_day"]].copy()
            out["prediction_realised_vol"] = pred
            out["text_path"] = "tp_" + out.accession  # unused, for manifest builder
            out.to_parquet(rd / "predictions.parquet", index=False)

        write_run("A2_har_rv", panel.fhar.to_numpy())
        write_run("ARM_LEVEL", lvl)
        write_run("ARM_CONTENT", cont)

        st = Settings(root=root, seed=2026,
                      arms=("ARM_LEVEL", "ARM_CONTENT"), horizons=(5, 10),
                      out_csv=root / "results/tables/swap_longform.csv",
                      out_md=root / "results/tables/swap_longform.md",
                      out_meta=root / "results/tables/swap_longform_meta.json",
                      skip_g1=True, skip_g3=True, require_committed_asserts=True)

        # manifest from the synthetic HAR panel (same builder as production)
        har = pd.read_parquet(st.run_pq("A2_har_rv"))
        har["text_path"] = "tp_" + har.accession
        manifest, _ = slb.build_manifest(har[slb.PANEL_COLS], horizons=(5, 10))
        man_path = root / "manifest.parquet"
        manifest.to_parquet(man_path, index=False)
        st.manifest = man_path

        # fabricate the committed tables via the SAME mechanism so asserts run hot
        grid_rows, cswap_rows, m1_rows = [], [], []
        harp = pd.read_parquet(st.run_pq("A2_har_rv"))[HAR_COLS].rename(
            columns={"prediction_realised_vol": "fhar"})
        for arm in st.arms:
            txt = pd.read_parquet(st.run_pq(arm))[KEY + ["prediction_realised_vol"]].rename(
                columns={"prediction_realised_vol": "ftext"})
            for h in st.horizons:
                dv, dt = _cell_frames(harp, txt, h)
                yv, fhv, ftv = (dv.label_realised_vol.to_numpy(), dv.fhar.to_numpy(),
                                dv.ftext.to_numpy())
                yt, fht, ftt = (dt.label_realised_vol.to_numpy(), dt.fhar.to_numpy(),
                                dt.ftext.to_numpy())
                fR, fU, _ = fc.log_combo(yv, fhv, ftv, fht, ftt)
                rr = rel_pct(fc.qlike(yt, fR), fc.qlike(yt, fU))
                grid_rows.append({"disc": DISC, "model": arm, "h": h,
                                  "rel_impr_pct": rr, "genuine": True})
                m1_rows.append({"disc": DISC, "model": arm, "h": h,
                                "genuine_ens_vol": True})
                rv_map = dv.groupby("ticker")["label_realised_vol"].mean()
                gm = float(dv.label_realised_vol.mean())
                fvs, _ = mfs.matched_swap(ftv, dv.ticker, day_key(dv), rv_map, gm)
                fts, fr = mfs.matched_swap(ftt, dt.ticker, day_key(dt), rv_map, gm)
                sR, sU, _ = fc.log_combo(yv, fhv, fvs, fht, fts)
                sr = rel_pct(fc.qlike(yt, sR), fc.qlike(yt, sU))
                cswap_rows.append({"disc": DISC, "model": arm, "h": h,
                                   "n_test": len(dt), "swap_frac_test": fr,
                                   "real_rel_pct": rr, "swap_rel_pct": sr,
                                   "retention_vs_real": sr / rr if rr > 0 else np.nan,
                                   "genuine": True})
        pd.DataFrame(grid_rows).to_csv(st.table("forecast_combination_grid.csv"), index=False)
        pd.DataFrame(m1_rows).to_csv(st.table("m1_ensemble_primary.csv"), index=False)
        pd.DataFrame(cswap_rows).to_csv(st.table("matched_firm_swap.csv"), index=False)
        (st.table("withindate_placebo.csv")).write_text("disc,model,h\n")  # unused here

        # "re-inferred" swapped predictions == permuted committed (text-pure identity)
        for arm in st.arms:
            pq = pd.read_parquet(st.run_pq(arm))
            vt = pq[pq.split.isin(("val", "test"))].reset_index(drop=True)
            m = vt.merge(manifest[["split", "horizon_days", "accession",
                                   "partner_accession"]],
                         on=["split", "horizon_days", "accession"], validate="one_to_one")
            look = m.set_index(["split", "horizon_days", "accession"])[
                "prediction_realised_vol"]
            keys = list(zip(m.split, m.horizon_days, m.partner_accession, strict=False))
            m["prediction_realised_vol"] = look.loc[keys].to_numpy()
            od = root / "results/runs" / f"ELF_swap_{arm}_full_{DISC}_seed2026"
            od.mkdir(parents=True)
            m[["ticker", "accession", "horizon_days", "split", "label_realised_vol",
               "prediction_realised_vol"]].to_parquet(
                od / "predictions_swapped.parquet", index=False)

        summary = run(st)
        df = pd.read_csv(st.out_csv)
        lvl_ret = df[df.model == "ARM_LEVEL"].retention_doc
        con_ret = df[df.model == "ARM_CONTENT"].retention_doc
        print(f"[selftest] planted LEVEL retentions:   {lvl_ret.round(2).tolist()}")
        print(f"[selftest] planted CONTENT retentions: {con_ret.round(2).tolist()}")
        assert (lvl_ret > 0.6).all(), "LEVEL arm retention not recovered (~1 expected)"
        assert (con_ret < 0.4).all(), "CONTENT arm retention not recovered (~0 expected)"
        assert (df.equiv_max_absdiff < 1e-12).all(), "equivalence identity violated"
        assert summary["branch"] in ("a", "b", "c")

        # single-shot guard must now refuse
        try:
            run(st)
            raise AssertionError("guard did not fire")
        except SystemExit as e:
            assert "one shot" in str(e) or "exists" in str(e), f"wrong guard: {e}"
        print("[selftest] single-shot guard fires on rerun")

    print("[selftest] ALL PASS — nothing under results/tables was touched")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=str, default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gates-only", action="store_true",
                    help="run G1+G2+G3 only; writes nothing under results/tables")
    ap.add_argument("--i-know-this-violates-prereg", dest="force", action="store_true")
    ap.add_argument("--rerun-reason", type=str, default=None,
                    help="required with the override; recorded in the md")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    from sp500vol.utils.paths import data_path
    manifest = (Path(args.manifest) if args.manifest else
                data_path("processed", "swap_lf", "swap_manifest_long_form.parquet"))
    st = Settings(manifest=manifest, seed=args.seed)

    if args.gates_only:
        gate_g2_pairing(st)
        gate_g1_committed_reproduction(st)
        gate_g3_no_retraining(st)  # requires the infer metas; run last
        print("[score] gates-only: ALL GATES PASS — nothing written under results/tables")
        return 0

    disclosure = None
    if args.force:
        if not args.rerun_reason:
            _fatal("--i-know-this-violates-prereg requires --rerun-reason")
        disclosure = (f"Rerun forced at "
                      f"{datetime.now(UTC).isoformat(timespec='seconds')}; "
                      f"reason: {args.rerun_reason}")
    run(st, rerun_disclosure=disclosure, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
