"""Run an ablation study end-to-end (AB1-AB10, of which 5 are kept under Light Compression).

Usage:
    python scripts/run_ablation.py --ablation AB1
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sp500vol.utils import configure_logging, get_logger

KEPT_ABLATIONS = {"AB1", "AB2", "AB3", "AB4", "AB8"}
STRETCH_ABLATIONS = {"AB10"}
DROPPED_ABLATIONS = {"AB5", "AB6", "AB7", "AB9"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", required=True)
    args = parser.parse_args()

    configure_logging("INFO")
    log = get_logger("ablation")

    ab = args.ablation
    if ab in DROPPED_ABLATIONS:
        log.error("Ablation is dropped under Plan B + Light Compression", ab=ab)
        return 1
    if ab in STRETCH_ABLATIONS:
        log.warning("Stretch goal — only run if time/budget permits at end of W12", ab=ab)
    if ab not in KEPT_ABLATIONS | STRETCH_ABLATIONS:
        log.error("Unknown ablation id", ab=ab, valid=sorted(KEPT_ABLATIONS | STRETCH_ABLATIONS))
        return 1

    log.info("Running ablation", ab=ab)

    # TODO:
    #   1. Load configs/ablations/{ab}_*.yaml
    #   2. For each model in ablation.models, ensure it's trained (or train)
    #   3. Collect predictions from each run
    #   4. Compute pairwise DM tests and bootstrap CIs
    #   5. Write ablation report (markdown + LaTeX table)

    log.warning("Ablation pipeline not yet implemented; this is a skeleton")
    return 0


if __name__ == "__main__":
    sys.exit(main())
