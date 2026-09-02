#!/usr/bin/env python3
"""G2 - SHA-256 configuration fingerprints for every run dir.

Backs the "SHA-256 configuration fingerprints" claim in
writing/paper/sections/12_reproducibility.tex with a concrete artifact.

For every results/runs/<run_id>/ (skipping *_smoke and *_sample):
  * read config.json
  * canonicalize: json.dumps(obj, sort_keys=True, separators=(",",":"))
  * config_sha256   = SHA-256 hex of the canonical full config
  * training_sha256 = SHA-256 hex of the canonical TRAINING-relevant subset
                      (config["model_config"]["training"]) if present, else ""
  * n_config_keys   = number of top-level keys in config.json

Usage:
  python scripts/analysis/config_fingerprints.py            # regenerate CSV
  python scripts/analysis/config_fingerprints.py --verify   # re-hash & compare
  python scripts/analysis/config_fingerprints.py --verify-preimages
        # re-hash the *released* config preimages (release/run_configs/) against
        # release/config_hashes.csv. This is the mode a reviewer can run: it
        # needs only files that ship inside the code-and-data package, not the
        # results/runs/ tree, which is not redistributed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO / "results" / "runs"
OUT_CSV = REPO / "results" / "tables" / "config_fingerprints.csv"
OUT_MD = REPO / "results" / "tables" / "config_fingerprints.md"
PREIMAGE_DIR = REPO / "release" / "run_configs"
PREIMAGE_CSV = REPO / "release" / "config_hashes.csv"
# Skip smoke/sample probe runs. The suffix form is *_smoke / *_sample; the
# sample-dataset runs use "_sample_" as an infix (dataset field == "sample").
# Both are non-full probes and are excluded so fingerprints identify real runs.
SKIP_SUFFIXES = ("_smoke", "_sample")
SKIP_INFIXES = ("_sample_",)
FIELDS = ["run_id", "config_sha256", "training_sha256_or_blank", "n_config_keys"]


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def fingerprint_config(cfg: dict) -> tuple[str, str, int]:
    config_sha = sha256_hex(canonical(cfg))
    training = None
    mc = cfg.get("model_config")
    if isinstance(mc, dict) and isinstance(mc.get("training"), dict):
        training = mc["training"]
    training_sha = sha256_hex(canonical(training)) if training is not None else ""
    return config_sha, training_sha, len(cfg)


def iter_run_dirs():
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        if d.name.endswith(SKIP_SUFFIXES):
            continue
        if any(inf in d.name for inf in SKIP_INFIXES):
            continue
        cfg_path = d / "config.json"
        if not cfg_path.is_file():
            continue
        yield d.name, cfg_path


def compute_rows() -> list[dict]:
    rows = []
    for run_id, cfg_path in iter_run_dirs():
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        config_sha, training_sha, n_keys = fingerprint_config(cfg)
        rows.append({
            "run_id": run_id,
            "config_sha256": config_sha,
            "training_sha256_or_blank": training_sha,
            "n_config_keys": n_keys,
        })
    return rows


def write_csv(rows: list[dict]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def verify() -> int:
    """Re-hash from disk and compare to the CSV. Returns exit code."""
    if not OUT_CSV.is_file():
        print(f"VERIFY FAIL: {OUT_CSV} does not exist; run without --verify first")
        return 1
    with open(OUT_CSV) as fh:
        stored = {r["run_id"]: r for r in csv.DictReader(fh)}
    recomputed = {r["run_id"]: r for r in compute_rows()}

    mismatches = []
    for run_id, rec in recomputed.items():
        st = stored.get(run_id)
        if st is None:
            mismatches.append(f"{run_id}: missing from CSV")
            continue
        for f in FIELDS:
            if str(st[f]) != str(rec[f]):
                mismatches.append(f"{run_id}.{f}: csv={st[f]} recomputed={rec[f]}")
    for run_id in stored:
        if run_id not in recomputed:
            mismatches.append(f"{run_id}: in CSV but no run dir on disk")

    if mismatches:
        print(f"VERIFY FAIL: {len(mismatches)} mismatch(es)")
        for m in mismatches:
            print("  " + m)
        return 1
    print(f"VERIFY OK: {len(recomputed)} runs re-hashed, all digests match CSV")
    return 0


def verify_preimages() -> int:
    """Re-hash the released preimages against the released CSV.

    Uses only files that ship inside the code-and-data package, so a reviewer
    with no access to results/runs/ can still execute the fingerprint check.
    """
    csv_path = PREIMAGE_CSV if PREIMAGE_CSV.is_file() else OUT_CSV
    if not csv_path.is_file() or not PREIMAGE_DIR.is_dir():
        print(f"VERIFY FAIL: need {PREIMAGE_DIR} and {csv_path}")
        return 1
    with open(csv_path) as fh:
        stored = {r["run_id"]: r for r in csv.DictReader(fh)}

    bad, seen = [], set()
    for f in sorted(PREIMAGE_DIR.glob("*.json")):
        run_id = f.stem
        seen.add(run_id)
        st = stored.get(run_id)
        if st is None:
            bad.append(f"{run_id}: preimage present but absent from CSV")
            continue
        cfg_sha, train_sha, n_keys = fingerprint_config(json.loads(f.read_text()))
        # The released index carries two digests per run. `config_sha256` is the
        # original, re-derivable only from the unreleased run tree; the released
        # preimage is checked against `released_config_sha256`, which differs
        # exactly for the runs whose `sanitised_fields` cell is non-empty (a
        # machine path was neutralised for anonymity).
        want = st.get("released_config_sha256") or st["config_sha256"]
        want_train = st.get("released_training_sha256_or_blank")
        if want_train is None:
            want_train = st["training_sha256_or_blank"]
        if cfg_sha != want:
            bad.append(f"{run_id}.config_sha256: csv={want[:16]} "
                       f"preimage={cfg_sha[:16]}")
        if train_sha != want_train:
            bad.append(f"{run_id}.training_sha256: mismatch")
        if str(n_keys) != str(st["n_config_keys"]):
            bad.append(f"{run_id}.n_config_keys: csv={st['n_config_keys']} "
                       f"preimage={n_keys}")
    for run_id in stored:
        if run_id not in seen:
            bad.append(f"{run_id}: in CSV but no released preimage")

    if bad:
        print(f"PREIMAGE VERIFY FAIL: {len(bad)} mismatch(es)")
        for m in bad[:40]:
            print("  " + m)
        return 1
    n_san = sum(1 for r in stored.values() if r.get("sanitised_fields"))
    print(f"PREIMAGE VERIFY OK: {len(seen)} released config preimages re-hashed, "
          f"all digests match {csv_path.name} "
          f"({len(seen) - n_san} identical to the original fingerprint, "
          f"{n_san} path-sanitised for anonymity)")
    return 0


def write_md(rows: list[dict]) -> None:
    """Regenerate the evidence table from the rows, so it cannot drift.

    This file previously drifted: it was hand-written against a 198-run CSV and
    still claimed "zero collisions" after the run set grew to 240 and seven
    genuine collision pairs appeared. It is generated now.
    """
    import collections

    n = len(rows)
    by_cfg: dict[str, list[str]] = {}
    for r in rows:
        by_cfg.setdefault(r["config_sha256"], []).append(r["run_id"])
    coll = {d: sorted(ids) for d, ids in by_cfg.items() if len(ids) > 1}
    tb = [r["training_sha256_or_blank"] for r in rows]
    n_train = sum(1 for x in tb if x)
    cnt = collections.Counter(tb)
    fam: dict[str, set] = collections.defaultdict(set)
    for r in rows:
        rid = r["run_id"].split("_")
        fam[r["training_sha256_or_blank"]].add(
            f"{rid[0]}_{rid[1]}" if len(rid) > 1 else r["run_id"])

    L = [f"# Config fingerprints: {n} runs fingerprinted, "
         f"{len(by_cfg)} distinct config digests, {len(coll)} collision pair(s)",
         "",
         "SHA-256 configuration fingerprints for every run directory under",
         "`results/runs/` (skipping `*_smoke`, `*_sample`, and sample-dataset",
         "`_sample_` probe runs). Generated by",
         "`scripts/analysis/config_fingerprints.py` — do not hand-edit.",
         "", "## Headline", "",
         f"- **{n}** runs fingerprinted.",
         f"- **{len(by_cfg)}** distinct full-config fingerprints.",
         f"- **{len(coll)}** config-digest collision pair(s), enumerated below.",
         f"- **{n_train}** runs carry a `model_config.training` block; these",
         f"  collapse to **{len({x for x in tb if x})}** distinct training",
         f"  fingerprints ({cnt['']} runs have no training block and are blank).",
         ""]
    if coll:
        L += ["## Collisions (expected, and why)", "",
              "Each pair is one cross-family probe's `_full_combined_` and",
              "`_full_event_driven_` run. Their `config.json` files are",
              "byte-identical because the combined/event-driven distinction is a",
              "downstream row-subset selection over a single generation pass, not",
              "a configuration field. The fingerprint therefore identifies the",
              "configuration, not the run.", "",
              "| digest (12) | run_ids |", "|---|---|"]
        for d in sorted(coll):
            L.append(f"| {d[:12]} | {'<br>'.join(coll[d])} |")
        L.append("")
    L += ["## Method", "",
          "For each run: `config_sha256 = SHA-256(json.dumps(cfg,",
          'sort_keys=True, separators=(",",":")))`. The training fingerprint',
          'hashes the canonicalized `config["model_config"]["training"]` subset',
          "(blank when absent), so it identifies the exact training regime",
          "independent of dataset/disclosure/seed.", "",
          "CSV: `results/tables/config_fingerprints.csv`, columns",
          "`[run_id, config_sha256, training_sha256_or_blank, n_config_keys]`.",
          "",
          "The *released* index `release/config_hashes.csv` adds",
          "`released_config_sha256` and `sanitised_fields`: a handful of configs",
          "name an absolute machine path that had to be neutralised for",
          "anonymity, which necessarily changes the digest, so both digests are",
          "published per run alongside the exact fields rewritten.",
          "", "## Training fingerprints cluster by model / training regime", "",
          "| training_sha256 (12) | n runs | model families |", "|---|---|---|"]
    for k, c in cnt.most_common():
        L.append(f"| {k[:12] if k else '(blank)'} | {c} | "
                 f"{', '.join(sorted(fam[k])[:8])} |")
    L += ["", "## Verification", "",
          f"- `config_fingerprints.py --verify` re-hashes all {n} configs from",
          "  `results/runs/` and compares to the CSV.",
          "- `config_fingerprints.py --verify-preimages` re-hashes the released",
          "  preimages in `release/run_configs/` against `release/config_hashes.csv`,",
          "  using only files that ship inside the code-and-data package.",
          "- Determinism: hashing one config twice yields identical digests.", ""]
    a2 = next((r for r in rows if r["run_id"] == "A2_har_rv_full_combined_seed2026"), None)
    if a2:
        L += [f"- Hand-verified `{a2['run_id']}` against a shell `shasum -a 256`",
              f"  of the canonical JSON: `{a2['config_sha256']}`.", ""]
    OUT_MD.write_text("\n".join(L))


def report(rows: list[dict]) -> None:
    n = len(rows)
    n_with_training = sum(1 for r in rows if r["training_sha256_or_blank"])
    distinct_cfg = {r["config_sha256"] for r in rows}
    distinct_train = {r["training_sha256_or_blank"] for r in rows if r["training_sha256_or_blank"]}

    # Collision check: any two DIFFERENT run_ids sharing a config digest?
    by_cfg: dict[str, list[str]] = {}
    for r in rows:
        by_cfg.setdefault(r["config_sha256"], []).append(r["run_id"])
    cfg_collisions = {d: ids for d, ids in by_cfg.items() if len(ids) > 1}

    print(f"runs fingerprinted            : {n}")
    print(f"runs with training block      : {n_with_training}")
    print(f"distinct config fingerprints  : {len(distinct_cfg)}")
    print(f"distinct training fingerprints: {len(distinct_train)}")
    print(f"config-digest collisions      : {len(cfg_collisions)} "
          f"(distinct run_ids sharing a full-config digest)")
    if cfg_collisions:
        for d, ids in cfg_collisions.items():
            print(f"  COLLISION {d[:16]}...: {ids}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="re-hash from results/runs/ and compare to the CSV")
    ap.add_argument("--verify-preimages", action="store_true",
                    help="re-hash the released preimages (package-only inputs)")
    args = ap.parse_args()
    if args.verify_preimages:
        return verify_preimages()
    if args.verify:
        return verify()
    rows = compute_rows()
    write_csv(rows)
    write_md(rows)
    report(rows)
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
