from __future__ import annotations

import torch

from sp500vol.models.neural_text.bert_s2 import _tokenize_chunk_batch


class _FakeCfg:
    max_length = 4


class _FakeTokenizer:
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        assert kwargs["return_overflowing_tokens"] is True
        assert kwargs["return_tensors"] == "pt"
        counts = [3] if isinstance(texts, str) else [1, 3, 2]

        rows = []
        mapping = []
        for sample_idx, count in enumerate(counts):
            for chunk_idx in range(count):
                rows.append([sample_idx + 1, chunk_idx + 1, 9, 9])
                mapping.append(sample_idx)
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.ones((len(rows), _FakeCfg.max_length), dtype=torch.long),
            "overflow_to_sample_mapping": torch.tensor(mapping, dtype=torch.long),
        }


class _FakeEncoder:
    cfg = _FakeCfg()
    tokenizer = _FakeTokenizer()


def test_chunk_batch_tokenization_groups_and_caps_overflow_chunks() -> None:
    items = _tokenize_chunk_batch(
        ["a", "b", "c"],
        encoder=_FakeEncoder(),
        chunk_stride=2,
        max_chunks=2,
        pad_id=0,
    )

    assert [item["input_ids"].shape[0] for item in items] == [1, 2, 2]
    assert items[0]["input_ids"][:, 0].tolist() == [1]
    assert items[1]["input_ids"][:, 1].tolist() == [1, 2]
    assert items[2]["input_ids"][:, 0].tolist() == [3, 3]
