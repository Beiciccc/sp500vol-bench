"""Block D: fusion models (D1 Concat-MLP, D2 Gated fusion, D3 LLM-embedding fusion)."""

from sp500vol.models.fusion.concat_mlp import ConcatMLP
from sp500vol.models.fusion.gated_fusion import GatedFusion
from sp500vol.models.fusion.llm_fusion import D3LLMFusion

__all__ = ["ConcatMLP", "D3LLMFusion", "GatedFusion"]
