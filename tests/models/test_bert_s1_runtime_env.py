"""Runtime environment overrides for BERT-S1 training throughput."""

from __future__ import annotations

import os

from sp500vol.models.neural_text.bert_s1 import BertS1


def test_bert_s1_runtime_env_overrides(monkeypatch) -> None:
    model = BertS1(pretokenize=False, tokenization_batch_size=128, tokenizer_threads=None)
    assert not model._use_pretokenize()
    assert model._runtime_tokenization_batch_size() == 128

    monkeypatch.setenv("SP500VOL_FORCE_PRETOKENIZE", "1")
    monkeypatch.setenv("SP500VOL_TOKENIZATION_BATCH_SIZE", "1024")
    monkeypatch.setenv("SP500VOL_TOKENIZER_THREADS", "32")
    monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)

    assert model._use_pretokenize()
    assert model._runtime_tokenization_batch_size() == 1024

    model._configure_tokenizer_runtime()
    assert os.environ["TOKENIZERS_PARALLELISM"] == "true"
    assert os.environ["RAYON_NUM_THREADS"] == "32"
