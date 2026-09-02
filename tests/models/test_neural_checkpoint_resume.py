"""Per-horizon checkpoint resume tests for neural text and fusion models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_TINY_BERT = "google/bert_uncased_L-2_H-128_A-2"


@pytest.fixture()
def single_horizon_corpus(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for i in range(4):
        text_path = tmp_path / f"filing_{i}.txt"
        text_path.write_text(
            (
                f"Filing {i}: revenue growth, liquidity risk, leverage, litigation, "
                "supplier concentration, and macroeconomic uncertainty. "
            )
            * 4,
            encoding="utf-8",
        )
        rows.append(
            {
                "accession": f"acc-{i:04d}",
                "horizon_days": 5,
                "text_path": str(text_path),
                "label_realised_vol": 0.20 + 0.01 * i,
                "feature_return_1d": 0.001 * (i + 1),
                "feature_rv_1d": 0.12 + 0.005 * i,
                "feature_rv_5d": 0.14 + 0.006 * i,
                "feature_rv_22d": 0.16 + 0.007 * i,
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.slow
def test_bert_s1_checkpoint_resume_skips_completed_horizon_and_checks_fingerprint(
    single_horizon_corpus: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sp500vol.models.neural_text.bert_s1 import BertS1

    checkpoint_dir = tmp_path / "bert_ckpt"
    model = BertS1(
        pretrained=_TINY_BERT,
        max_length=64,
        hidden_dim=16,
        batch_size=2,
        max_epochs=1,
        device="cpu",
        checkpoint_dir=checkpoint_dir,
        seed=2026,
    )
    model.fit(single_horizon_corpus, single_horizon_corpus["label_realised_vol"].to_numpy())
    pred = model.predict(single_horizon_corpus)
    assert (checkpoint_dir / "horizon_5.pt").exists()

    resumed = BertS1(
        pretrained=_TINY_BERT,
        max_length=64,
        hidden_dim=16,
        batch_size=2,
        max_epochs=1,
        device="cpu",
        checkpoint_dir=checkpoint_dir,
        seed=2026,
    )
    original_build = resumed._build_modules

    def fail_build():
        raise AssertionError("fit should skip completed horizon without rebuilding modules")

    monkeypatch.setattr(resumed, "_build_modules", fail_build)
    resumed.fit(single_horizon_corpus, single_horizon_corpus["label_realised_vol"].to_numpy())
    monkeypatch.setattr(resumed, "_build_modules", original_build)
    np.testing.assert_allclose(resumed.predict(single_horizon_corpus), pred, rtol=1e-5, atol=1e-7)

    mismatched = BertS1(
        pretrained=_TINY_BERT,
        max_length=64,
        hidden_dim=16,
        batch_size=2,
        max_epochs=1,
        device="cpu",
        checkpoint_dir=checkpoint_dir,
        seed=999,
    )
    build_calls = 0
    original_mismatch_build = mismatched._build_modules

    def count_build():
        nonlocal build_calls
        build_calls += 1
        return original_mismatch_build()

    monkeypatch.setattr(mismatched, "_build_modules", count_build)
    mismatched.fit(single_horizon_corpus, single_horizon_corpus["label_realised_vol"].to_numpy())
    assert build_calls > 0


@pytest.mark.slow
def test_bert_s4_checkpoint_resume_round_trips_chunk_encoder_state(
    single_horizon_corpus: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    from torch import nn

    import sp500vol.models.neural_text.bert_s1 as bert_s1_mod
    from sp500vol.models.neural_text.bert_s4 import BertS4

    class _FakeTokenizer:
        pad_token_id = 0

        def __call__(self, texts, *, max_length: int, **kwargs):
            values = [texts] if isinstance(texts, str) else list(texts)
            rows = []
            masks = []
            for text in values:
                ids = torch.zeros(max_length, dtype=torch.long)
                mask = torch.zeros(max_length, dtype=torch.long)
                used = min(max_length, 4)
                base = len(str(text)) % 11 + 1
                ids[:used] = torch.arange(base, base + used, dtype=torch.long)
                mask[:used] = 1
                rows.append(ids)
                masks.append(mask)
            return {"input_ids": torch.stack(rows), "attention_mask": torch.stack(masks)}

    class _FakeCLSEncoder(nn.Module):
        def __init__(self, cfg) -> None:
            super().__init__()
            self.cfg = cfg
            self.tokenizer = _FakeTokenizer()
            self.hidden_size = 8
            self.embedding = nn.Embedding(32, self.hidden_size)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            emb = self.embedding(input_ids)
            weights = attention_mask.float().unsqueeze(-1)
            denom = weights.sum(dim=1).clamp(min=1.0)
            return (emb * weights).sum(dim=1) / denom

    monkeypatch.setattr(bert_s1_mod, "CLSEncoder", _FakeCLSEncoder)

    checkpoint_dir = tmp_path / "s4_ckpt"
    model = BertS4(
        pretrained="fake-bert",
        max_length=16,
        chunk_stride=8,
        max_chunks=3,
        chunk_num_heads=2,
        chunk_encoder_layers=1,
        chunk_ff_dim=16,
        hidden_dim=8,
        batch_size=2,
        max_epochs=1,
        device="cpu",
        checkpoint_dir=checkpoint_dir,
        seed=2026,
    )
    model.fit(single_horizon_corpus, single_horizon_corpus["label_realised_vol"].to_numpy())
    pred = model.predict(single_horizon_corpus)

    checkpoint_path = checkpoint_dir / "horizon_5.pt"
    assert checkpoint_path.exists()
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    assert "chunk_encoder_state" in payload["state"]

    resumed = BertS4(
        pretrained="fake-bert",
        max_length=16,
        chunk_stride=8,
        max_chunks=3,
        chunk_num_heads=2,
        chunk_encoder_layers=1,
        chunk_ff_dim=16,
        hidden_dim=8,
        batch_size=2,
        max_epochs=1,
        device="cpu",
        checkpoint_dir=checkpoint_dir,
        seed=2026,
    )
    original_build = resumed._build_modules

    def fail_build():
        raise AssertionError("fit should skip completed S4 horizon")

    monkeypatch.setattr(resumed, "_build_modules", fail_build)
    resumed.fit(single_horizon_corpus, single_horizon_corpus["label_realised_vol"].to_numpy())
    monkeypatch.setattr(resumed, "_build_modules", original_build)

    assert "chunk_encoder_state" in resumed.models_[5]
    np.testing.assert_allclose(resumed.predict(single_horizon_corpus), pred, rtol=1e-5, atol=1e-7)


@pytest.mark.slow
def test_gated_fusion_checkpoint_resume_restores_price_standardisation(
    single_horizon_corpus: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sp500vol.models.fusion.gated_fusion import GatedFusion

    checkpoint_dir = tmp_path / "fusion_ckpt"
    model = GatedFusion(
        pretrained=_TINY_BERT,
        max_length=64,
        proj_dim=8,
        hidden_dim=8,
        batch_size=2,
        max_epochs=1,
        device="cpu",
        checkpoint_dir=checkpoint_dir,
        seed=2026,
    )
    model.fit(single_horizon_corpus, single_horizon_corpus["label_realised_vol"].to_numpy())
    pred = model.predict(single_horizon_corpus)
    assert (checkpoint_dir / "horizon_5.pt").exists()

    resumed = GatedFusion(
        pretrained=_TINY_BERT,
        max_length=64,
        proj_dim=8,
        hidden_dim=8,
        batch_size=2,
        max_epochs=1,
        device="cpu",
        checkpoint_dir=checkpoint_dir,
        seed=2026,
    )
    original_build = resumed._build_modules

    def fail_build():
        raise AssertionError("fit should skip completed fusion horizon")

    monkeypatch.setattr(resumed, "_build_modules", fail_build)
    resumed.fit(single_horizon_corpus, single_horizon_corpus["label_realised_vol"].to_numpy())
    monkeypatch.setattr(resumed, "_build_modules", original_build)

    assert 5 in resumed.price_mean_
    assert 5 in resumed.price_std_
    np.testing.assert_allclose(resumed.predict(single_horizon_corpus), pred, rtol=1e-5, atol=1e-7)
