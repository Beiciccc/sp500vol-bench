"""Tests for the shared filing-text cache used by Block B classical models."""

from __future__ import annotations

import pandas as pd
import pytest

import sp500vol.models.classical_text._text_dataset as td


@pytest.fixture(autouse=True)
def _clear_store():
    td._STORES.clear()
    yield
    td._STORES.clear()


def test_tmp_paths_stay_per_call_and_do_not_touch_cache(tmp_path):
    # Real _is_cacheable: a pytest tmp path is outside the data root, so it must
    # use the per-call read and never create/populate the shared cache.
    f = tmp_path / "f.txt"
    f.write_text("alpha beta", encoding="utf-8")
    rows = pd.DataFrame({"text_path": [str(f)], "horizon_days": [5]})

    out = td.load_texts(rows)

    assert out == ["alpha beta"]
    assert td._STORES == {}
    assert not (tmp_path / "filing_texts.parquet").exists()


def test_shared_cache_built_and_reused_after_source_deleted(tmp_path, monkeypatch):
    # Force cacheable + cache under tmp; simulate a second process by clearing the
    # in-memory store and deleting the source .txt — must then serve from parquet.
    cache = tmp_path / "cache.parquet"
    monkeypatch.setattr(td, "_is_cacheable", lambda _p: True)
    monkeypatch.setattr(td, "_default_cache_path", lambda: cache)

    src = tmp_path / "f.txt"
    src.write_text("hello world", encoding="utf-8")
    rows = pd.DataFrame({"text_path": [str(src)], "horizon_days": [5]})

    assert td.load_texts(rows) == ["hello world"]
    assert cache.exists()

    td._STORES.clear()  # new process
    src.unlink()  # source gone
    assert td.load_texts(rows) == ["hello world"]  # served from cache parquet


def test_repeated_filing_horizon_rows_return_same_text(tmp_path, monkeypatch):
    cache = tmp_path / "cache.parquet"
    monkeypatch.setattr(td, "_is_cacheable", lambda _p: True)
    monkeypatch.setattr(td, "_default_cache_path", lambda: cache)

    src = tmp_path / "f.txt"
    src.write_text("doc text", encoding="utf-8")
    rows = pd.DataFrame({"text_path": [str(src)] * 3, "horizon_days": [5, 10, 20]})

    assert td.load_texts(rows) == ["doc text"] * 3


def test_second_model_run_reuses_cache_without_reading_source(tmp_path, monkeypatch):
    # First "run" builds the cache; second "run" (fresh store) must not re-read
    # the .txt even if it is replaced with different content.
    cache = tmp_path / "cache.parquet"
    monkeypatch.setattr(td, "_is_cacheable", lambda _p: True)
    monkeypatch.setattr(td, "_default_cache_path", lambda: cache)

    src = tmp_path / "f.txt"
    src.write_text("original", encoding="utf-8")
    rows = pd.DataFrame({"text_path": [str(src)], "horizon_days": [5]})
    assert td.load_texts(rows) == ["original"]

    td._STORES.clear()
    src.write_text("CHANGED", encoding="utf-8")  # cache must win over re-read
    assert td.load_texts(rows) == ["original"]


def test_persist_new_false_reads_but_does_not_write_shared_cache(tmp_path, monkeypatch):
    cache = tmp_path / "cache.parquet"
    monkeypatch.setattr(td, "_is_cacheable", lambda _p: True)
    monkeypatch.setattr(td, "_default_cache_path", lambda: cache)

    src = tmp_path / "f.txt"
    src.write_text("predict only", encoding="utf-8")
    rows = pd.DataFrame({"text_path": [str(src)], "horizon_days": [5]})

    assert td.load_texts(rows, persist_new=False) == ["predict only"]
    assert not cache.exists()
