"""Byte-identity test for cross-horizon pretokenisation reuse (_TokenCache).

The token cache (keyed by text_path, tokenising each unique filing once across the
horizon loop / val epochs / predict) must produce tensors BYTE-IDENTICAL to the legacy
per-call _pretokenized_dataset path — tokenisation is a pure function of
(text, tokenizer, max_length), independent of horizon and encoder weights. Marked slow
because it loads the bert-tiny tokenizer/model.
"""

from __future__ import annotations

import numpy as np
import pytest

_TINY_MODEL = "google/bert_uncased_L-2_H-128_A-2"


@pytest.mark.slow
def test_token_cache_byte_identical_and_dedup() -> None:
    import torch

    from sp500vol.models.neural_text.bert_s1 import _pretokenized_dataset, _TokenCache
    from sp500vol.models.neural_text.encoders import CLSEncoder, EncoderConfig

    encoder = CLSEncoder(EncoderConfig(pretrained=_TINY_MODEL, max_length=48))

    # 5 unique filings, with filing 0 REPEATED (as the same text_path recurs across
    # horizons). Gather order must reproduce the input order exactly.
    base_texts = [f"Filing {i}: revenue growth and risk factors, change {i}%." for i in range(5)]
    texts = [*base_texts, base_texts[0]]
    text_paths = [f"/p/filing_{i}.txt" for i in range(5)] + ["/p/filing_0.txt"]

    legacy = _pretokenized_dataset(
        texts,
        np.zeros(len(texts), dtype=np.float32),
        encoder=encoder,
        tokenization_batch_size=3,
    )
    cache = _TokenCache()
    ids, mask = cache.gather(text_paths, texts, encoder=encoder, tokenization_batch_size=3)

    # Ragged (truncation-only) rows, byte-identical to the legacy path, in input order.
    assert len(ids) == len(legacy.input_ids) == len(texts)
    for got, exp in zip(ids, legacy.input_ids, strict=True):
        assert torch.equal(got, exp)
    for got, exp in zip(mask, legacy.attention_mask, strict=True):
        assert torch.equal(got, exp)
    # The repeated filing was tokenised once: 5 unique keys for 6 rows.
    assert len(cache._memo) == 5  # white-box: the repeated filing was tokenised once


@pytest.mark.slow
def test_token_cache_empty() -> None:
    from sp500vol.models.neural_text.bert_s1 import _TokenCache
    from sp500vol.models.neural_text.encoders import CLSEncoder, EncoderConfig

    encoder = CLSEncoder(EncoderConfig(pretrained=_TINY_MODEL, max_length=48))
    ids, mask = _TokenCache().gather([], [], encoder=encoder, tokenization_batch_size=3)
    assert ids == []
    assert mask == []


@pytest.mark.slow
def test_chunk_token_cache_byte_identical_and_dedup() -> None:
    import torch

    from sp500vol.models.neural_text.bert_s2 import _ChunkedTextDataset, _ChunkTokenCache
    from sp500vol.models.neural_text.encoders import CLSEncoder, EncoderConfig

    encoder = CLSEncoder(EncoderConfig(pretrained=_TINY_MODEL, max_length=32))
    # texts long enough to span several chunks at max_length 32 / stride 8.
    base = [f"Filing {i}: " + " ".join(f"word{j}" for j in range(80)) for i in range(5)]
    texts = [*base, base[0]]  # filing 0 repeated (as the same text_path recurs across horizons)
    text_paths = [f"/p/f{i}.txt" for i in range(5)] + ["/p/f0.txt"]

    legacy = _ChunkedTextDataset(
        texts,
        np.zeros(len(texts), dtype=np.float32),
        encoder,
        chunk_stride=8,
        max_chunks=4,
        tokenization_batch_size=3,
    )
    cache = _ChunkTokenCache()
    items = cache.gather(
        text_paths,
        texts,
        encoder=encoder,
        chunk_stride=8,
        max_chunks=4,
        tokenization_batch_size=3,
    )
    assert len(items) == len(texts)
    for cached_item, legacy_item in zip(items, legacy.items, strict=True):
        assert torch.equal(cached_item["input_ids"], legacy_item["input_ids"])
        assert torch.equal(cached_item["attention_mask"], legacy_item["attention_mask"])
    assert len(cache._memo) == 5  # white-box: the repeated filing was tokenised once


@pytest.mark.slow
def test_chunk_token_disk_cache_roundtrip(tmp_path, monkeypatch) -> None:
    """Disk-backed chunk token cache persists + reloads byte-identically, and a second
    gather reads from disk WITHOUT re-tokenising (proven by feeding empty texts)."""
    import torch

    from sp500vol.models.neural_text import bert_s2
    from sp500vol.models.neural_text.bert_s2 import _chunk_cache_path, _ChunkTokenCache
    from sp500vol.models.neural_text.encoders import CLSEncoder, EncoderConfig

    monkeypatch.setenv("SP500VOL_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("SP500VOL_CHUNK_TOK_DISK_CACHE", "1")
    bert_s2._TOK_STORES.clear()  # drop the process-level store so the test is hermetic

    encoder = CLSEncoder(EncoderConfig(pretrained=_TINY_MODEL, max_length=32))
    base = [f"Filing {i}: " + " ".join(f"word{j}" for j in range(80)) for i in range(5)]
    texts = [*base, base[0]]
    text_paths = [f"/p/f{i}.txt" for i in range(5)] + ["/p/f0.txt"]
    kw = {"encoder": encoder, "chunk_stride": 8, "max_chunks": 4, "tokenization_batch_size": 3}

    # First gather tokenises and persists to disk.
    items1 = _ChunkTokenCache().gather(text_paths, texts, **kw)
    cache_path = _chunk_cache_path(encoder, 4, 8)
    assert cache_path is not None and cache_path.exists()

    # Simulate a fresh process: clear the in-memory store so gather MUST hit disk. Passing
    # empty texts proves it does NOT re-tokenise — the disk store supplies the items.
    bert_s2._TOK_STORES.clear()
    items2 = _ChunkTokenCache().gather(text_paths, [""] * len(texts), **kw)

    assert len(items2) == len(items1) == len(texts)
    for a, b in zip(items1, items2, strict=True):
        assert torch.equal(a["input_ids"], b["input_ids"])
        assert torch.equal(a["attention_mask"], b["attention_mask"])
        assert a["input_ids"].dtype == b["input_ids"].dtype == torch.long


@pytest.mark.slow
def test_pretok_reuse_predict_matches_legacy(tmp_path) -> None:
    """fit+predict with reuse ON vs OFF give identical predictions (same tokens, same
    seed) — a golden check that threading text_paths preserved alignment end-to-end."""
    import os

    import pandas as pd

    from sp500vol.models.neural_text.bert_s1 import BertS1

    rows = []
    for i in range(6):
        tp = tmp_path / f"f{i}.txt"
        tp.write_text(f"Filing {i}: revenue growth and risk factors {i}.", encoding="utf-8")
        for h in (5, 20):
            rows.append(
                {
                    "accession": f"a{i}",
                    "horizon_days": h,
                    "text_path": str(tp),
                    "label_realised_vol": 0.2 + 0.01 * i + 0.001 * h,
                }
            )
    df = pd.DataFrame(rows)
    y = df["label_realised_vol"].to_numpy()

    def run(reuse: str) -> np.ndarray:
        prev = os.environ.get("SP500VOL_PRETOK_REUSE")
        os.environ["SP500VOL_PRETOK_REUSE"] = reuse
        try:
            import torch

            # BertS1 does not seed the RNG itself, so seed here to make the run
            # deterministic — then reuse ON vs OFF must match (same tokens, same RNG).
            torch.manual_seed(0)
            model = BertS1(
                pretrained=_TINY_MODEL,
                max_length=48,
                hidden_dim=16,
                batch_size=4,
                pretokenize=True,
                max_epochs=1,
                early_stopping=False,
                device="cpu",
                seed=123,
            )
            model.fit(df, y)
            return model.predict(df)
        finally:
            if prev is None:
                os.environ.pop("SP500VOL_PRETOK_REUSE", None)
            else:
                os.environ["SP500VOL_PRETOK_REUSE"] = prev

    np.testing.assert_array_equal(run("1"), run("0"))


@pytest.mark.slow
def test_fusion_max_length_padding_invariant() -> None:
    """The fusion refactor switched the text branch from dynamic to max_length padding.
    CLS pooling is padding-invariant under the attention mask, so the text embedding (and
    thus the fusion output) is unchanged. Verify CLS(dynamic) == CLS(max_length)."""
    import torch

    from sp500vol.models.neural_text.encoders import CLSEncoder, EncoderConfig

    enc = CLSEncoder(EncoderConfig(pretrained=_TINY_MODEL, max_length=48))
    enc.eval()
    texts = [f"Filing {i}: revenue and risk factors {i}." for i in range(4)]
    with torch.inference_mode():
        dyn = enc.tokenize(texts)  # dynamic padding (pad to batch max)
        emb_dyn = enc(dyn["input_ids"], dyn["attention_mask"])
        ml = enc.tokenizer(
            texts,
            max_length=48,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
        )
        emb_ml = enc(ml["input_ids"], ml["attention_mask"])
    torch.testing.assert_close(emb_dyn, emb_ml, rtol=1e-4, atol=1e-5)


@pytest.mark.slow
def test_fusion_pretok_reuse_predict_matches(tmp_path) -> None:
    """D2 fusion: reuse ON vs OFF give identical predictions (same tokens, same seed) — an
    end-to-end check that pre-tokenise + token cache + picklable collate kept behaviour."""
    import os

    import pandas as pd
    import torch

    from sp500vol.models.fusion.gated_fusion import GatedFusion

    rows = []
    for i in range(6):
        tp = tmp_path / f"f{i}.txt"
        tp.write_text(f"Filing {i}: revenue growth and risk factors {i}.", encoding="utf-8")
        for h in (5, 20):
            rows.append(
                {
                    "horizon_days": h,
                    "text_path": str(tp),
                    "feature_rv_1d": 0.20 + 0.01 * i,
                    "feature_rv_5d": 0.21 + 0.01 * i,
                    "feature_rv_22d": 0.22 + 0.01 * i,
                    "label_realised_vol": 0.20 + 0.01 * i + 0.001 * h,
                }
            )
    df = pd.DataFrame(rows)
    y = df["label_realised_vol"].to_numpy()

    def run(reuse: str) -> np.ndarray:
        prev = os.environ.get("SP500VOL_PRETOK_REUSE")
        os.environ["SP500VOL_PRETOK_REUSE"] = reuse
        try:
            torch.manual_seed(0)
            model = GatedFusion(
                pretrained=_TINY_MODEL,
                max_length=48,
                proj_dim=16,
                hidden_dim=16,
                batch_size=4,
                max_epochs=1,
                early_stopping=False,
                device="cpu",
            )
            model.fit(df, y)
            return model.predict(df)
        finally:
            if prev is None:
                os.environ.pop("SP500VOL_PRETOK_REUSE", None)
            else:
                os.environ["SP500VOL_PRETOK_REUSE"] = prev

    np.testing.assert_array_equal(run("1"), run("0"))
