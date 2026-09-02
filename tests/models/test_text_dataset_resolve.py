"""Path-resolution tests for the classical-text dataset helper.

Cross-machine robustness: a parquet built on machine A (absolute text_path)
must still be readable on machine B with a different data root.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sp500vol.models.classical_text._text_dataset import _resolve, load_texts
from sp500vol.utils.paths import DATA_ROOT_ENV, data_root, resolve_data_path


def test_relative_path_anchored_at_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "sp500vol-data"
    monkeypatch.setenv(DATA_ROOT_ENV, str(data_root))

    p = _resolve("data/interim/sample/AAPL_10-K.txt")
    assert p == data_root / "interim" / "sample" / "AAPL_10-K.txt"


def test_data_root_env_uses_current_external_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external_root = tmp_path / "sp500vol-data"
    external_root.mkdir()
    monkeypatch.setenv(DATA_ROOT_ENV, str(external_root))

    assert data_root() == external_root


def test_absolute_path_returned_as_is_when_exists(tmp_path: Path) -> None:
    real = tmp_path / "filing.txt"
    real.write_text("body", encoding="utf-8")
    p = _resolve(str(real))
    assert p == real


def test_absolute_path_rebased_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "external-data"
    monkeypatch.setenv(DATA_ROOT_ENV, str(data_root))

    # Simulate a parquet from another machine with a foreign data-root prefix.
    foreign = Path("/home/alice/some/where/data/interim/dry_run_medium/8-K/0000320193/x.txt")
    p = _resolve(str(foreign))
    assert p == data_root / "interim" / "dry_run_medium" / "8-K" / "0000320193" / "x.txt"


def test_posix_style_path_rebases_on_windows_style_systems(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On Windows, Path('/home/...').is_absolute() returns False but the path
    still has a 'data' segment. Our resolver must rebase regardless of
    is_absolute() return value — and crucially must not produce
    data_root/home/foreign/... by treating the rooted path as relative.
    """
    data_root = tmp_path / "data-root"
    monkeypatch.setenv(DATA_ROOT_ENV, str(data_root))

    posix_style = "/home/alice/some/foreign/data/interim/dry_run_medium/x.txt"
    p = _resolve(posix_style)
    assert p == data_root / "interim" / "dry_run_medium" / "x.txt"


def test_resolve_data_path_rebases_paths_with_data_segment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "external-data"
    monkeypatch.setenv(DATA_ROOT_ENV, str(data_root))

    assert resolve_data_path(Path("/a/b/c/data/interim/x.txt")) == data_root / "interim" / "x.txt"


def test_resolve_data_path_keeps_missing_absolute_path_without_data_segment() -> None:
    missing = Path("/a/b/c/nothing.txt")
    assert resolve_data_path(missing) == missing


def test_load_texts_reads_from_resolved_path(tmp_path: Path) -> None:
    text_path = tmp_path / "x.txt"
    text_path.write_text("hello world", encoding="utf-8")
    df = pd.DataFrame({"text_path": [str(text_path), str(text_path)]})
    texts = load_texts(df)
    assert texts == ["hello world", "hello world"]
