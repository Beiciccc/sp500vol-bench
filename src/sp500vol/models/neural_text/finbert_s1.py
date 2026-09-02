"""C2 FinBERT-S1 — domain-pretrained BERT with truncation (S1) strategy.

Identical architecture and training loop to BertS1; only the encoder checkpoint
differs (`yiyanghkust/finbert-tone`, 110M params, pretrained on 10-K/10-Q/analyst
reports). This is the headline encoder for AB1 long-doc strategy sweep.
"""

from __future__ import annotations

from sp500vol.models.neural_text.bert_s1 import BertS1


class FinBertS1(BertS1):
    """FinBERT-base + S1 truncation, otherwise identical to BertS1.

    Default checkpoint is ``ProsusAI/finbert`` (Araci 2019) — pretrained on
    Reuters financial news on top of BERT-base. Uses standard BertTokenizer
    (WordPiece), avoiding the sentencepiece/tokenizer conversion issues seen
    with ``yiyanghkust/finbert-tone`` under transformers >= 5.x.
    """

    name = "C2_finbert_s1"

    def __init__(self, *, pretrained: str = "ProsusAI/finbert", **kwargs) -> None:
        super().__init__(pretrained=pretrained, **kwargs)
