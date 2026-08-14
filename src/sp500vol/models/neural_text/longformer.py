"""C4 Longformer-base native long-context model."""

from __future__ import annotations

from sp500vol.models.neural_text.bert_s1 import _EPSILON, BertS1
from sp500vol.models.neural_text.encoders import LongformerEncoder
from sp500vol.models.neural_text.heads import VolatilityHead


class LongformerModel(BertS1):
    """Longformer-base with S1-style training and 4096-token CLS global attention."""

    name = "C4_longformer"

    def __init__(self, *, pretrained: str = "allenai/longformer-base-4096", **kwargs) -> None:
        super().__init__(pretrained=pretrained, **kwargs)

    def _build_modules(self) -> tuple[LongformerEncoder, VolatilityHead]:
        self._configure_tokenizer_runtime()
        encoder = LongformerEncoder(self.encoder_cfg).to(self.device)
        head = VolatilityHead(
            encoder.hidden_size,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            eps=_EPSILON,
            positive=not self.log_target,
        ).to(self.device)
        return encoder, head
