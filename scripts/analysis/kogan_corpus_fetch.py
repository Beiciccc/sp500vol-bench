#!/usr/bin/env python
"""Fetch Kogan et al. (2009)'s public 10-K corpus (prereg configs/prereg_kogan_corpus.md
§"Data", tag prereg-kc-v1.0) into a local scratch directory.

DATA IS NOT REDISTRIBUTED — only this pipeline ships. The corpus is public at
http://www.cs.cmu.edu/~ark/10K/ (Version 1.0, 2009-03-31; addendum 2009-09-18) with
no licence terms beyond a citation request:

  Shimon Kogan, Dimitry Levin, Bryan R. Routledge, Jacob S. Sagi, and Noah A. Smith.
  Predicting Risk from Financial Reports with Regression. NAACL-HLT 2009.

Per year yyyy in 1996..2006 this fetches exactly the four files the prereg names:
  yyyy.meta.txt      key | filing date yyyymmdd | EDGAR URL | company | CIK
  yyyy.tok.tgz       tokenised MD&A sections (README: `tok.tgz` = tokenized MD&A)
  yyyy.logvol.+12.txt  forward 12-month log volatility  = LABEL
  yyyy.logvol.-12.txt  past 12-month log volatility     = the price baseline

G-K0: every downloaded file's SHA-256 is recorded to `manifest.json` in the target
directory; `kogan_corpus_audit.py` re-verifies the hashes and reproduces them into
the output table. Re-running is idempotent: an existing file whose size matches the
server's Content-Length is not re-downloaded (pass --force to override).

Run from repo root:
  .venv/bin/python scripts/analysis/kogan_corpus_fetch.py --dest <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

BASE = "http://www.cs.cmu.edu/~ark/10K/data"
README_URL = "http://www.cs.cmu.edu/~ark/10K/data/README"
YEARS = tuple(range(1996, 2007))
PER_YEAR = ("meta.txt", "tok.tgz", "logvol.+12.txt", "logvol.-12.txt")
TIMEOUT = 180


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remote_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n is not None else None
    except Exception:
        return None


def fetch(url: str, dest: Path, *, force: bool) -> dict:
    """Download `url` to `dest` unless already present at the server's size."""
    exp = remote_size(url)
    if dest.exists() and not force and exp is not None and dest.stat().st_size == exp:
        return {"url": url, "file": dest.name, "bytes": dest.stat().st_size,
                "sha256": sha256(dest), "cached": True}
    t0 = time.time()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r, tmp.open("wb") as out:
        while chunk := r.read(1 << 20):
            out.write(chunk)
    tmp.replace(dest)
    n = dest.stat().st_size
    if exp is not None and n != exp:
        raise RuntimeError(f"{url}: got {n} bytes, server declared {exp}")
    print(f"  fetched {dest.name:<24} {n / 1e6:8.2f} MB  ({time.time() - t0:.1f}s)")
    return {"url": url, "file": dest.name, "bytes": n, "sha256": sha256(dest),
            "cached": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, help="target directory (scratch; NOT the repo)")
    ap.add_argument("--force", action="store_true", help="re-download even if size matches")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    files = [fetch(README_URL, dest / "README.txt", force=args.force)]
    for y in YEARS:
        print(f"[{y}]")
        for suf in PER_YEAR:
            files.append(fetch(f"{BASE}/{y}.{suf}", dest / f"{y}.{suf}", force=args.force))

    manifest = {
        "source": BASE,
        "corpus": "Kogan et al. (2009) 10-K corpus, Version 1.0 (2009-03-31; "
                  "addendum 2009-09-18)",
        "citation": "Kogan, Levin, Routledge, Sagi & Smith. Predicting Risk from "
                    "Financial Reports with Regression. NAACL-HLT 2009.",
        "redistribution": "NOT redistributed — pipeline only (prereg-kc-v1.0)",
        "fetched_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "years": list(YEARS),
        "n_files": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "files": files,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(files)} files, {manifest['total_bytes'] / 1e6:.1f} MB total "
          f"in {time.time() - t0:.1f}s -> {dest / 'manifest.json'}")


if __name__ == "__main__":
    main()
