"""Smoke tests for README in-scope C/D neural/fusion additions.

Each test uses a tiny pretrained checkpoint and is marked slow because it runs a
real fit/predict/save/load cycle. Predictive quality is covered by full runs;
these tests guard model wiring, checkpoint state, and positive finite outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_TINY_BERT = "google/bert_uncased_L-2_H-128_A-2"
_TINY_ROBERTA = "hf-internal-testing/tiny-random-roberta"
_TINY_LONGFORMER = "hf-internal-testing/tiny-random-longformer"


@pytest.fixture()
def tiny_cd_corpus(tmp_path: Path) -> pd.DataFrame:
    rows = []
    horizons = [5, 20]
    for i in range(6):
        text = (
            f"Filing {i}: risk factors include liquidity, leverage, revenue volatility, "
            f"supplier concentration, litigation, and macroeconomic uncertainty. " * 24
        )
        text_path = tmp_path / f"filing_{i}.txt"
        text_path.write_text(text, encoding="utf-8")
        for h in horizons:
            rows.append(
                {
                    "accession": f"acc-{i:04d}",
                    "horizon_days": h,
                    "text_path": str(text_path),
                    "label_realised_vol": 0.20 + 0.01 * i + 0.001 * h,
                    "feature_return_1d": 0.001 * (i + 1),
                    "feature_rv_1d": 0.12 + 0.005 * i,
                    "feature_rv_5d": 0.14 + 0.006 * i,
                    "feature_rv_22d": 0.16 + 0.007 * i,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.slow
def test_c3_roberta_s1_fit_predict_smoke(tiny_cd_corpus: pd.DataFrame, tmp_path: Path) -> None:
    from sp500vol.models.neural_text.roberta_s1 import RobertaS1

    model = RobertaS1(
        pretrained=_TINY_ROBERTA,
        max_length=64,
        hidden_dim=8,
        batch_size=4,
        max_epochs=1,
        device="cpu",
    )
    _assert_roundtrip(model, RobertaS1, tiny_cd_corpus, tmp_path / "roberta_s1.pkl")


@pytest.mark.slow
def test_c2_finbert_s4_fit_predict_smoke(tiny_cd_corpus: pd.DataFrame, tmp_path: Path) -> None:
    from sp500vol.models.neural_text.bert_s4 import FinBertS4

    model = FinBertS4(
        pretrained=_TINY_BERT,
        max_length=64,
        chunk_stride=32,
        max_chunks=3,
        chunk_num_heads=4,
        chunk_encoder_layers=1,
        chunk_ff_dim=64,
        hidden_dim=16,
        batch_size=2,
        grad_accumulation_steps=2,
        max_epochs=1,
        device="cpu",
    )
    _assert_roundtrip(model, FinBertS4, tiny_cd_corpus, tmp_path / "finbert_s4.pkl")


@pytest.mark.slow
def test_c4_longformer_fit_predict_smoke(tiny_cd_corpus: pd.DataFrame, tmp_path: Path) -> None:
    from sp500vol.models.neural_text.longformer import LongformerModel

    model = LongformerModel(
        pretrained=_TINY_LONGFORMER,
        max_length=128,
        hidden_dim=8,
        batch_size=1,
        grad_accumulation_steps=2,
        pretokenize=True,
        tokenization_batch_size=3,
        tokenizer_threads=2,
        max_epochs=1,
        device="cpu",
    )
    _assert_roundtrip(model, LongformerModel, tiny_cd_corpus, tmp_path / "longformer.pkl")


@pytest.mark.slow
def test_d1_concat_mlp_fit_predict_smoke(tiny_cd_corpus: pd.DataFrame, tmp_path: Path) -> None:
    from sp500vol.models.fusion.concat_mlp import ConcatMLP

    model = ConcatMLP(
        pretrained=_TINY_BERT,
        max_length=64,
        proj_dim=8,
        hidden_dim=8,
        batch_size=4,
        max_epochs=1,
        device="cpu",
    )
    _assert_roundtrip(model, ConcatMLP, tiny_cd_corpus, tmp_path / "concat_mlp.pkl")


def _assert_roundtrip(model, cls, data: pd.DataFrame, save_path: Path) -> None:
    model.fit(data, data["label_realised_vol"].to_numpy())
    pred = model.predict(data)
    assert pred.shape == (len(data),)
    assert np.isfinite(pred).all()
    assert (pred > 0).all()

    model.save(save_path)
    loaded = cls.load(save_path)
    pred_loaded = loaded.predict(data)
    np.testing.assert_allclose(pred_loaded, pred, rtol=1e-5, atol=1e-7)
