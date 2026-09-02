"""Shared filing-text loader with a persistent, process-shared cache.

``text_path`` is a stable per-filing key. Across Block-B runs (B1-B4 x disclosure
subsets, each a separate process) the first load reads each parsed ``.txt`` once
and writes a shared parquet store ``(text_path, text)``; every later run reads
that parquet instead of re-opening ~144k small files. A module-level dict shares
within a process so fit and predict do not re-read.

Only paths under the data root are cached. Unit-test corpora live under pytest's
tmp dir (outside the data root), so they stay on the original per-call path and
never touch — or pollute — the shared cache.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from filelock import FileLock

from sp500vol.utils.paths import data_path, data_root, resolve_data_path

# cache_path (str) -> {text_path: text}; shared across fit/predict within a process.
_STORES: dict[str, dict[str, str]] = {}


def load_texts(
    rows: pd.DataFrame,
    *,
    cache_path: Path | None = None,
    persist_new: bool = True,
) -> list[str]:
    """Return one parsed text string per row, via the shared on-disk cache.

    Repeated ``(filing x horizon)`` rows return the same string. Paths under the
    data root are read once and, when ``persist_new`` is true, persisted to a
    shared parquet cache so other Block-B runs reuse them; tmp/test paths fall
    back to a per-call in-memory read.
    """
    if "text_path" not in rows.columns:
        raise ValueError("rows must contain a 'text_path' column")

    paths = rows["text_path"].astype(str).tolist()
    resolved_cache = Path(cache_path) if cache_path is not None else _default_cache_path()
    store = _store_for(paths, resolved_cache)
    if store is None:
        return _load_per_call(paths)
    if not persist_new:
        out: list[str] = []
        for path in paths:
            if path in store:
                out.append(store[path])
            else:
                out.append(resolve_data_path(path).read_text(encoding="utf-8", errors="replace"))
        return out

    new: dict[str, str] = {}
    out: list[str] = []
    for path in paths:
        if path in store:
            out.append(store[path])
        elif path in new:
            out.append(new[path])
        else:
            text = resolve_data_path(path).read_text(encoding="utf-8", errors="replace")
            out.append(text)
            new[path] = text
    if new:
        store.update(new)
        _persist(resolved_cache, store)
    return out


def _store_for(paths: list[str], resolved_cache: Path) -> dict[str, str] | None:
    """The shared store serving these paths, or None for the per-call fallback.

    When no path resolves under THIS machine's data root — a processed parquet
    built on another host stores ``text_path`` strings carrying a foreign
    absolute data root (e.g. ``/path/to/data-root/sp500vol-data/...``) — the shared
    cache may still key those exact strings, so it is used whenever it knows
    any of them. Genuinely uncacheable paths (unit-test tmp corpora) are never
    in the store and stay on the per-call path, exactly as before.
    """
    if any(_is_cacheable(p) for p in paths):
        return _shared_store(resolved_cache)
    if str(resolved_cache) in _STORES or resolved_cache.exists():
        store = _shared_store(resolved_cache)
        if any(p in store for p in paths):
            return store
    return None


def _default_cache_path() -> Path:
    return data_path("processed", "_text_cache", "filing_texts.parquet")


def _load_per_call(paths: list[str]) -> list[str]:
    cache: dict[str, str] = {}
    out: list[str] = []
    for path in paths:
        if path not in cache:
            cache[path] = resolve_data_path(path).read_text(encoding="utf-8", errors="replace")
        out.append(cache[path])
    return out


def _shared_store(cache_path: Path) -> dict[str, str]:
    key = str(cache_path)
    store = _STORES.get(key)
    if store is None:
        store = {}
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            store = dict(zip(df["text_path"].astype(str), df["text"].astype(str), strict=True))
        _STORES[key] = store
    return store


def _persist(cache_path: Path, store: dict[str, str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cache_path) + ".lock"):
        frame = pd.DataFrame({"text_path": list(store), "text": list(store.values())})
        tmp = cache_path.with_suffix(".parquet.tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(cache_path)


def _is_cacheable(text_path: str) -> bool:
    """True when the path resolves under the data root (real ingest, not tmp)."""
    try:
        resolved = resolve_data_path(text_path).resolve()
        root = data_root().resolve()
    except (OSError, ValueError):
        return False
    return resolved == root or root in resolved.parents


def _resolve(text_path: str) -> Path:
    """Back-compat shim: resolve a stored ``text_path`` to a file on disk."""
    return resolve_data_path(text_path)
