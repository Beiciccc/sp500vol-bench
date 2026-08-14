"""Text-cache shim for boxes without the interim/ filing-text tree.

On the GPU box the data root holds only processed/ (incl. the shared
_text_cache/filing_texts.parquet, 144k rows) — the per-filing interim/*.txt
tree was not synced, and aligned_filings.parquet stores text_path strings from
the ingest machine (/Volumes/Z/...). sp500vol's load_texts() treats such
foreign paths as non-cacheable and falls back to reading the .txt from disk,
which crashes.

ensure_texts_available() makes load_texts() work unchanged: if filing texts
are not readable from disk, it pre-loads the shared parquet store (keyed by
the ORIGINAL text_path strings, which match aligned_filings exactly) and
patches the module's cacheability check so every lookup consults the store.
Missing keys still crash loudly (FileNotFoundError) instead of degrading.

No-op on machines where the text files resolve on disk (e.g. the ingest Mac
with /Volumes/Z mounted).
"""

from __future__ import annotations


def ensure_texts_available(sample_text_path: str) -> str:
    """Call once with any row's text_path before the first load_texts()."""
    from sp500vol.models.classical_text import _text_dataset as tds

    try:
        if tds.resolve_data_path(sample_text_path).exists():
            return "disk"  # texts readable as files — no shim needed
    except (OSError, ValueError):
        pass

    cache_path = tds._default_cache_path()
    if not cache_path.exists():
        raise FileNotFoundError(
            f"filing texts unreachable: {sample_text_path} not on disk and no "
            f"shared text cache at {cache_path}"
        )
    store = tds._shared_store(cache_path)  # loads the parquet into the module store
    if sample_text_path not in store:
        raise KeyError(
            f"text cache at {cache_path} has {len(store)} texts but is missing "
            f"{sample_text_path} — cache and aligned_filings are out of sync"
        )
    # Foreign absolute paths (e.g. /Volumes/Z/... recorded on the ingest machine)
    # are keys of the shared store, not on-disk files here: mark everything
    # cacheable so load_texts() consults the store instead of reading disk.
    tds._is_cacheable = lambda _path: True  # noqa: E731
    return f"parquet-store({len(store)} texts)"
