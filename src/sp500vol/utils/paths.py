"""Central filesystem path resolution for generated data products.

Large, regenerable data products — raw EDGAR filings, parsed interim text,
processed datasets, market caches — can be redirected off the repo/internal
disk via the ``SP500VOL_DATA_ROOT`` environment variable. This is needed when:

  - the internal disk is small and the corpus lives on an external drive, or
  - running on a cloud GPU where data is a mounted bucket (compute plan §5.2).

Defaults to ``<repo>/data`` so a fresh clone works with zero configuration.

Inputs that travel WITH the code are NOT redirected and stay under the repo:
index membership tables (``data/universe/``) and the
Loughran-McDonald dictionary (``data/external/``). Those are resolved against
``REPO_ROOT`` by their call sites, never through this module.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT_ENV = "SP500VOL_DATA_ROOT"


def data_root() -> Path:
    """Root for generated data products. Override with ``$SP500VOL_DATA_ROOT``."""
    override = os.environ.get(DATA_ROOT_ENV)
    if override:
        return Path(override).expanduser()
    return REPO_ROOT / "data"


def data_path(*parts: str) -> Path:
    """A path under the data root, e.g. ``data_path('processed', 'full')``."""
    return data_root().joinpath(*parts)


def resolve_data_path(path: str | Path) -> Path:
    """Resolve a config- or parquet-stored data path against the data root.

    Cross-machine / cross-disk safe (pathlib ``is_absolute`` is unreliable for
    POSIX paths on Windows, and processed parquets may store an absolute
    ``text_path`` produced on another machine or disk):

      1. Absolute and existing as-is → use it (typical same-machine case).
      2. Path containing a ``data`` segment (e.g. ``./data/raw`` or a foreign
         ``/old/repo/data/interim/...``) → re-root at ``data_root()/<after data>``.
      3. Otherwise → ``data_root()/<path>``.
    """
    p = Path(path).expanduser()
    if p.is_absolute() and p.exists():
        return p
    parts = [x for x in p.parts if x not in (".", "/", "\\")]
    if "data" in parts:
        idx = parts.index("data")
        return data_root().joinpath(*parts[idx + 1 :])
    if p.is_absolute():
        return p
    return data_root().joinpath(*parts)
