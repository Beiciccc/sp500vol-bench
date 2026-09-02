"""C3 RoBERTa-base + S1 truncation strategy."""

from __future__ import annotations

from sp500vol.models.neural_text.bert_s1 import BertS1


class RobertaS1(BertS1):
    """RoBERTa-base S1 control; training loop is identical to BertS1."""

    name = "C3_roberta_s1"

    def __init__(self, *, pretrained: str = "roberta-base", **kwargs) -> None:
        super().__init__(pretrained=pretrained, **kwargs)
