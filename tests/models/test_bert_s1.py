"""Smoke test for C1 BertS1 — uses a tiny pretrained model (google/bert_uncased_L-2_H-128_A-2).

This is intentionally marked `slow` so it does not run on every CI invocation.
Local runs and full CI matrices will pick it up via `pytest tests/ -m slow`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# canonical bert-tiny (L-2/H-128); ships tokenizer.json so the fast tokenizer
# loads on transformers 5.x without the spurious sentencepiece-conversion error.
_TINY_MODEL = "google/bert_uncased_L-2_H-128_A-2"


@pytest.fixture()
def tiny_corpus(tmp_path: Path) -> pd.DataFrame:
    rows = []
    horizons = [5, 20]
    for i in range(6):
        text = (
            f"Filing {i}: this is a small synthetic 10-K body discussing "
            f"revenue growth and risk factors. Year-over-year change {i}%."
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
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.slow
def test_bert_s1_fit_predict_smoke(tiny_corpus: pd.DataFrame, tmp_path: Path) -> None:
    """End-to-end smoke: fit on tiny pretrained model, predict, save, load.

    Asserts predictions are finite + positive and survive a save/load cycle.
    Does NOT assert predictive quality — that's what the full pipeline is for.
    """
    # Lazy import keeps torch/transformers out of test-collection import time.
    from sp500vol.models.neural_text.bert_s1 import BertS1

    model = BertS1(
        pretrained=_TINY_MODEL,
        max_length=64,
        hidden_dim=16,
        batch_size=4,
        pretokenize=True,
        tokenization_batch_size=3,
        tokenizer_threads=2,
        max_epochs=1,
        device="cpu",
    )
    model.fit(tiny_corpus, tiny_corpus["label_realised_vol"].to_numpy())

    pred = model.predict(tiny_corpus)
    assert pred.shape == (len(tiny_corpus),)
    assert np.isfinite(pred).all()
    assert (pred > 0).all()

    save_path = tmp_path / "bert_s1.pkl"
    model.save(save_path)
    loaded = BertS1.load(save_path)
    pred_loaded = loaded.predict(tiny_corpus)
    np.testing.assert_allclose(pred_loaded, pred, rtol=1e-5, atol=1e-7)
