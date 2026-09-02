"""Tests for the frozen-LLM probe (C5_llm) and its price fusion (D3_llm_fusion).

These never download the 7B encoder: a deterministic fake encoder is injected so
the cache/dedup, early-stopping val curve, predict path, and save/load round-trip
are all exercised on CPU in milliseconds. A separate pure-tensor test pins the
last-token pooling for both padding sides (the single subtlest correctness point).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


def _deterministic_emb(text: str, dim: int) -> np.ndarray:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


class _FakeLLMEncoder:
    """Stand-in for FrozenLLMEncoder: content-deterministic, no model download."""

    hidden_size = 16

    def __init__(self, **_kwargs) -> None:
        pass

    def encode(self, texts: list[str], *, batch_size: int = 8) -> np.ndarray:
        if not texts:
            return np.empty((0, self.hidden_size), dtype=np.float32)
        return np.stack([_deterministic_emb(t, self.hidden_size) for t in texts])


def _corpus(tmp_path: Path, n: int, *, tag: str) -> pd.DataFrame:
    rows = []
    for i in range(n):
        text_path = tmp_path / f"{tag}_{i}.txt"
        text_path.write_text(
            f"{tag} filing {i}: revenue, liquidity risk, leverage, litigation, outlook.",
            encoding="utf-8",
        )
        rows.append(
            {
                "accession": f"{tag}-{i:04d}",
                "horizon_days": 5,
                "text_path": str(text_path),
                "label_realised_vol": 0.18 + 0.01 * i,
                "feature_return_1d": 0.001 * (i + 1),
                "feature_rv_5d": 0.14 + 0.006 * i,
                "feature_rv_22d": 0.16 + 0.007 * i,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def patched_encoder(monkeypatch: pytest.MonkeyPatch):
    import sp500vol.models.neural_text.qwen_llm as qwen_mod

    monkeypatch.setattr(qwen_mod, "FrozenLLMEncoder", _FakeLLMEncoder)
    return _FakeLLMEncoder


def test_last_token_pool_left_and_right_padding():
    from sp500vol.models.neural_text.qwen_llm import last_token_pool

    # (batch=2, seq=3, hidden=2)
    hidden = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
        ]
    )
    # Left padding: last column always real → take [:, -1].
    left_mask = torch.tensor([[0, 1, 1], [0, 0, 1]])
    left = last_token_pool(hidden, left_mask)
    torch.testing.assert_close(left, torch.tensor([[3.0, 3.0], [6.0, 6.0]]))

    # Right padding: last real token per row via mask.
    right_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    right = last_token_pool(hidden, right_mask)
    torch.testing.assert_close(right, torch.tensor([[2.0, 2.0], [4.0, 4.0]]))


def test_c5_llm_fit_predict_save_load(patched_encoder, tmp_path: Path):
    from sp500vol.models.neural_text import C5LLMProbe

    train = _corpus(tmp_path, 6, tag="train")
    val = _corpus(tmp_path, 3, tag="val")
    model = C5LLMProbe(
        device="cpu",
        cache_embeddings=False,
        max_epochs=5,
        es_patience=2,
        batch_size=4,
        hidden_dim=8,
    )
    model.fit(
        train,
        train["label_realised_vol"].to_numpy(),
        X_val=val,
        y_val=val["label_realised_vol"].to_numpy(),
    )
    assert model.embedding_dim_ == _FakeLLMEncoder.hidden_size

    pred = model.predict(train)
    assert pred.shape == (len(train),)
    assert np.isfinite(pred).all() and (pred > 0).all()

    # Early-stopping curve was recorded with real (non-None) validation losses.
    curve = model.val_curves_[5]
    assert len(curve) >= 1
    assert all(c["val_loss"] is not None for c in curve)
    assert sum(int(c["is_best"]) for c in curve) >= 1

    save_path = tmp_path / "c5.pkl"
    model.save(save_path)
    loaded = C5LLMProbe.load(save_path)
    assert loaded.embedding_dim_ == _FakeLLMEncoder.hidden_size
    np.testing.assert_allclose(loaded.predict(train), pred, rtol=1e-5, atol=1e-7)


def test_d3_llm_fusion_fit_predict_save_load(patched_encoder, tmp_path: Path):
    from sp500vol.models.fusion import D3LLMFusion

    train = _corpus(tmp_path, 6, tag="train")
    val = _corpus(tmp_path, 3, tag="val")
    model = D3LLMFusion(
        device="cpu",
        cache_embeddings=False,
        proj_dim=8,
        hidden_dim=8,
        max_epochs=5,
        es_patience=2,
        batch_size=4,
    )
    model.fit(
        train,
        train["label_realised_vol"].to_numpy(),
        X_val=val,
        y_val=val["label_realised_vol"].to_numpy(),
    )
    assert model.embedding_dim_ == _FakeLLMEncoder.hidden_size
    assert 5 in model.price_mean_ and 5 in model.price_std_

    pred = model.predict(train)
    assert pred.shape == (len(train),)
    assert np.isfinite(pred).all() and (pred > 0).all()

    curve = model.val_curves_[5]
    assert all(c["val_loss"] is not None for c in curve)

    save_path = tmp_path / "d3.pkl"
    model.save(save_path)
    loaded = D3LLMFusion.load(save_path)
    assert 5 in loaded.price_mean_
    np.testing.assert_allclose(loaded.predict(train), pred, rtol=1e-5, atol=1e-7)


def test_embed_dataframe_dedups_by_text_path(patched_encoder, tmp_path: Path):
    """Repeated text_path rows (filing x horizon) encode once and gather in order."""
    from sp500vol.models.neural_text.qwen_llm import embed_dataframe

    df = _corpus(tmp_path, 3, tag="dup")
    doubled = pd.concat([df, df], ignore_index=True)  # each text_path appears twice
    emb = embed_dataframe(_FakeLLMEncoder, doubled, cache_path=None, batch_size=4)
    assert emb.shape == (6, _FakeLLMEncoder.hidden_size)
    # Rows 0..2 (first copy) must equal rows 3..5 (second copy) — same text_paths.
    np.testing.assert_array_equal(emb[:3], emb[3:])


def test_embedding_cache_persists_and_skips_encoder_on_hit(tmp_path: Path):
    """A fully-cached run must not invoke the encoder factory (no 7B load)."""
    from sp500vol.models.neural_text import qwen_llm
    from sp500vol.models.neural_text.qwen_llm import embed_dataframe

    df = _corpus(tmp_path, 3, tag="cache")
    cache = tmp_path / "emb_cache.parquet"
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return _FakeLLMEncoder()

    qwen_llm._EMB_STORES.clear()
    emb1 = embed_dataframe(factory, df, cache_path=cache, batch_size=4)
    assert calls["n"] == 1 and cache.exists()

    qwen_llm._EMB_STORES.clear()  # simulate a fresh process reading the disk cache
    emb2 = embed_dataframe(factory, df, cache_path=cache, batch_size=4)
    assert calls["n"] == 1  # factory NOT called again — pure disk-cache hit
    np.testing.assert_array_equal(emb1, emb2)


def test_embedding_cache_merge_keeps_concurrent_writes(tmp_path: Path):
    """Two interleaved writers must not lose each other's entries.

    Reproduces the read-modify-write window: writer A and writer B BOTH load the
    same (empty) snapshot, encode disjoint filings, then persist in sequence.
    Without the merge-under-lock, B's overwrite would clobber A's entry; with it,
    B re-reads the on-disk store inside the FileLock and unions, so both survive.
    Also asserts the in-process _EMB_STORES is refreshed to the union after persist.
    """
    from sp500vol.models.neural_text import qwen_llm

    cache = tmp_path / "merge_cache.parquet"
    a = np.ones(4, dtype=np.float32)
    b = np.full(4, 2.0, dtype=np.float32)

    # Both writers start from the same empty view (nothing on disk yet).
    qwen_llm._EMB_STORES.clear()
    writer_a_store = {"a": a}  # A's in-memory snapshot (loaded empty + encoded "a")
    writer_b_store = {"b": b}  # B's in-memory snapshot (loaded empty + encoded "b")

    # A persists first.
    qwen_llm._persist_emb_store(cache, writer_a_store)
    # B persists from its OWN stale snapshot that never saw "a".
    qwen_llm._EMB_STORES.clear()  # B is a separate process: no shared in-memory store
    qwen_llm._persist_emb_store(cache, writer_b_store)

    # The in-process store B just wrote must already reflect the union (no reload).
    assert set(qwen_llm._EMB_STORES[str(cache)]) == {"a", "b"}

    # And a fresh reader (cleared cache) sees both entries with correct values.
    qwen_llm._EMB_STORES.clear()
    store = qwen_llm._load_emb_store(cache)
    assert set(store) == {"a", "b"}
    np.testing.assert_array_equal(store["a"], a)
    np.testing.assert_array_equal(store["b"], b)

    # Symmetric interleaving (C then D, D persists first) also loses nothing.
    cache2 = tmp_path / "merge_cache2.parquet"
    c = np.full(4, 3.0, dtype=np.float32)
    d = np.full(4, 4.0, dtype=np.float32)
    qwen_llm._EMB_STORES.clear()
    qwen_llm._persist_emb_store(cache2, {"d": d})
    qwen_llm._EMB_STORES.clear()
    qwen_llm._persist_emb_store(cache2, {"c": c})
    qwen_llm._EMB_STORES.clear()
    store2 = qwen_llm._load_emb_store(cache2)
    assert set(store2) == {"c", "d"}
