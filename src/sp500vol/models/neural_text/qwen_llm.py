"""C5 — frozen decoder-LLM embedding probe (e.g. gte-Qwen2-7B-instruct).

Unlike the Block-C encoders (C1-C4) which fine-tune a BERT-family CLS encoder,
C5 treats a modern instruction-tuned decoder LLM as a *frozen* feature extractor:

    emb = last_token_pool( LLM(filing_text) )      # frozen, no gradients
    y_hat = head(emb)                              # only this trains

This isolates the question "is the failure in the *information content* of
disclosures, or in *encoder-era representations* of them?" (Conclusions, future
work). The encoder is never fine-tuned, so its embeddings are deterministic and
seed/split-independent — we encode each filing exactly once and cache the vector
on disk, keyed by ``text_path``, shared across horizons, splits and seeds. Only
the small per-horizon regression head trains, via the shared early-stopping loop
in :mod:`sp500vol.models.neural_text._train_utils`.

Pooling follows the gte-Qwen2 model card: left-padding + last-token pooling +
L2 normalisation. ``trust_remote_code=True`` is required for the GTE auto_map.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

from sp500vol.models.base import VolatilityForecaster
from sp500vol.models.classical_text._text_dataset import load_texts
from sp500vol.models.neural_text import _train_utils as train_utils
from sp500vol.models.neural_text.bert_s1 import (
    _EPSILON,
    _TOKENIZER_THREADS_ENV,
    _env_positive_int,
    _horizons,
    _maybe_exp,
    _maybe_log,
    _r2,
    _require_dataframe,
    _resolve_device,
)
from sp500vol.models.neural_text.encoders import EncoderConfig
from sp500vol.models.neural_text.heads import VolatilityHead
from sp500vol.utils.paths import data_path

# cache_path (str) -> {text_path: np.ndarray(embedding_dim)}; shared within a process.
_EMB_STORES: dict[str, dict[str, np.ndarray]] = {}


def _install_dynamic_cache_compat() -> None:
    """gte-Qwen2-7B's bundled (trust_remote_code) modeling code calls the old
    DynamicCache.get_usable_length() API, removed in transformers>=4.x (renamed
    get_seq_length). For a single frozen-encode forward (use_cache=False, empty KV
    cache) aliasing it to get_seq_length() is exact. No-op on newer code paths."""
    try:
        from transformers.cache_utils import DynamicCache

        if not hasattr(DynamicCache, "get_usable_length"):

            def _get_usable_length(self, *args, **kwargs):
                return self.get_seq_length()

            DynamicCache.get_usable_length = _get_usable_length
    except Exception:  # pragma: no cover - defensive; transformers internals may move
        pass


_install_dynamic_cache_compat()


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Hidden state of the final non-pad token (gte-Qwen2 model-card pooling).

    Auto-detects padding side: with left padding the last column is always real,
    so ``[:, -1]`` is correct; with right padding we gather the last real index
    per row from the attention mask.
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


def _encoder_dtype(device: torch.device) -> torch.dtype:
    # bf16 on CUDA (7B in fp32 would not fit a 40G card); fp32 on CPU/MPS.
    return torch.bfloat16 if device.type == "cuda" else torch.float32


# Tokenise this many docs per tokenizer call rather than all-at-once: the fast
# tokenizer's BatchEncoding would otherwise hold every doc's result simultaneously,
# spiking RAM (see FrozenLLMEncoder.encode).
_TOKENIZE_CHUNK = 512


class FrozenLLMEncoder(nn.Module):
    """Frozen decoder-LLM wrapped as a sentence-embedding extractor.

    The model is loaded with ``requires_grad_(False)`` and kept in ``eval()``;
    all encoding runs under ``torch.inference_mode``. It is intentionally NOT
    placed in any optimiser or in the training loop's ``modules`` list.
    """

    def __init__(
        self,
        *,
        pretrained: str,
        max_length: int = 4096,
        instruction: str | None = None,
        normalize: bool = True,
        device: torch.device,
        trust_remote_code: bool = True,
        tokenizer_threads: int | None = None,
    ) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.max_length = int(max_length)
        self.instruction = instruction
        self.normalize = bool(normalize)
        self.device = device
        self.tokenizer_threads = tokenizer_threads
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Force memory-efficient attention. Without this, trust_remote_code encoders
        # (e.g. gte-Qwen2) can default to EAGER attention, materialising the full
        # L x L scores tensor (~8.6 GB at batch 8 / 4096 tokens) and pushing the encode
        # peak toward ~26 GB — marginal on a 32 GB card. SDPA avoids that allocation.
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": _encoder_dtype(device),
        }
        try:
            self.model = AutoModel.from_pretrained(
                pretrained, attn_implementation="sdpa", **model_kwargs
            )
        except (ValueError, TypeError):  # custom remote code may reject the kwarg
            self.model = AutoModel.from_pretrained(pretrained, **model_kwargs)
        self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.hidden_size: int = int(self.model.config.hidden_size)

    def _format(self, text: str) -> str:
        if self.instruction:
            return f"Instruct: {self.instruction}\nQuery: {text}"
        return text

    @torch.inference_mode()
    def encode(self, texts: list[str], *, batch_size: int = 8) -> np.ndarray:
        """Return an (N, hidden_size) float32 array of pooled, normalised embeddings.

        Tokenisation is done up-front for ALL texts via the fast tokenizer's
        multi-threaded batch path (RAYON), instead of one batch at a time between
        GPU forwards — so the CPU tokenisation cost runs in parallel rather than
        serialised against the GPU. Batches are length-sorted so each padded batch
        is near-uniform length (less wasted compute), and the GPU runs back-to-back
        forwards instead of idling while each batch tokenises. This is the fix for
        the CPU-bound encode bottleneck (GPU dropping to 0% between batches).
        """
        if not texts:
            return np.empty((0, self.hidden_size), dtype=np.float32)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        # Pin the fast-tokeniser RAYON thread count so the up-front tokenise does not
        # oversubscribe shared vCPUs under concurrent runs (mirrors BertS1/fusion).
        threads = self.tokenizer_threads or _env_positive_int(_TOKENIZER_THREADS_ENV)
        if threads is not None:
            os.environ["RAYON_NUM_THREADS"] = str(threads)
        # Char-cap each text BEFORE tokenising, and tokenise in chunks rather than all
        # at once. Filings run up to ~2M chars / 442k tokens, and the fast tokenizer
        # tokenises each doc IN FULL before truncating to max_length — doing all ~8k
        # test docs at once peaks ~86 GB RAM (OOMs the 100 GB box). Capping to
        # max_length*16 chars yields token-IDENTICAL output (even at a dense ~4
        # chars/token the first max_length tokens sit well inside the cap) while
        # bounding per-doc tokenisation; chunking stops the BatchEncoding from holding
        # all N results at once. Together this drops the tokenise peak to ~13 GB.
        char_cap = self.max_length * 16
        encoded: list[list[int]] = []
        for chunk_start in range(0, len(texts), _TOKENIZE_CHUNK):
            formatted = [
                self._format(t[:char_cap])
                for t in texts[chunk_start : chunk_start + _TOKENIZE_CHUNK]
            ]
            encoded.extend(
                self.tokenizer(
                    formatted, max_length=self.max_length, truncation=True, padding=False
                )["input_ids"]
            )
        # length-sort to minimise padding within each batch
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
        out = np.empty((len(encoded), self.hidden_size), dtype=np.float32)
        # Token-budget batching: pack each (length-sorted) batch up to batch_size*max_length
        # tokens so short filings (8-K median ~930 tok) share a batch instead of wasting GPU at
        # a fixed `batch_size` rows; long filings fall back to few rows. Padded-token count per
        # batch is bounded by the SAME worst case as the old fixed batch (batch_size*max_length),
        # so peak GPU memory is unchanged. Numerically exact: each row's frozen forward +
        # last-token pool depends only on its own tokens (padding is attention-masked), so batch
        # composition cannot change any embedding.
        token_budget = max(1, batch_size) * self.max_length
        n_docs = len(order)
        start = 0
        while start < n_docs:
            end = start + 1
            while end < n_docs and (end - start + 1) * len(encoded[order[end]]) <= token_budget:
                end += 1
            idx = order[start:end]
            tok = self.tokenizer.pad(
                {"input_ids": [encoded[i] for i in idx]}, padding=True, return_tensors="pt"
            )
            ids = tok["input_ids"].to(self.device)
            mask = tok["attention_mask"].to(self.device)
            hidden = self.model(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).last_hidden_state
            # Cast to fp32 BEFORE normalising so cached vectors are exact unit norm
            # (bf16 L2-normalise leaves ~0.3% norm error in the permanent cache).
            emb = last_token_pool(hidden, mask).float()
            if self.normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out[idx] = emb.float().cpu().numpy()  # scatter back to original order
            start = end
        return out


def _load_emb_store(cache_path: Path) -> dict[str, np.ndarray]:
    key = str(cache_path)
    store = _EMB_STORES.get(key)
    if store is None:
        store = {}
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            store = {
                str(tp): np.asarray(vec, dtype=np.float32)
                for tp, vec in zip(df["text_path"], df["embedding"], strict=True)
            }
        _EMB_STORES[key] = store
    return store


def _persist_emb_store(cache_path: Path, store: dict[str, np.ndarray]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cache_path) + ".lock"):
        # Merge with the current on-disk store INSIDE the lock so concurrent
        # writers (parallel seeds/models sharing this cache) cannot drop each
        # other's freshly-encoded vectors. The frozen encoder is deterministic, so
        # overlapping keys are identical; in-memory values win harmlessly.
        merged = dict(store)
        if cache_path.exists():
            disk = pd.read_parquet(cache_path)
            for tp, vec in zip(disk["text_path"], disk["embedding"], strict=True):
                merged.setdefault(str(tp), np.asarray(vec, dtype=np.float32))
        frame = pd.DataFrame(
            {
                "text_path": list(merged),
                "embedding": [vec.astype(np.float32).tolist() for vec in merged.values()],
            }
        )
        tmp = cache_path.with_suffix(f".parquet.{os.getpid()}.tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(cache_path)
        _EMB_STORES[str(cache_path)] = merged


def _missing_texts(df: pd.DataFrame, missing: list[str]) -> dict[str, str]:
    """Texts for exactly the ``missing`` text_paths, streamed from the shared text
    cache row-group by row-group — so a PARTIAL encode (e.g. predict's test rows over a
    warmed train/val cache) never pulls all 144k texts (~8.4G) into RAM and OOMs. Falls
    back to the full load_texts only for tmp/uncached paths (unit tests, no cache)."""
    from sp500vol.models.classical_text._text_dataset import _default_cache_path

    cache = _default_cache_path()
    if cache.exists():
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        # Stream row-groups, keeping only matching rows. A one-shot read_table with an
        # ("text_path","in",...) filter cannot prune row-groups (text_path is unsorted),
        # so it decompresses the whole 3.5G cache (~18 GB peak). Iterating batches caps
        # the peak to ~one batch plus the kept texts (~1 GB).
        want = pa.array(missing, type=pa.string())
        need = len(set(missing))
        out: dict[str, str] = {}
        for batch in pq.ParquetFile(cache).iter_batches(
            columns=["text_path", "text"], batch_size=10000
        ):
            mask = pc.is_in(batch.column("text_path"), value_set=want)
            if not pc.any(mask).as_py():
                continue
            sub = batch.filter(mask)
            for tp_v, tx_v in zip(
                sub.column("text_path").to_pylist(),
                sub.column("text").to_pylist(),
                strict=True,
            ):
                out.setdefault(tp_v, tx_v)
            if len(out) >= need:
                break
        if all(tp in out for tp in missing):
            return out
    texts = load_texts(df)
    tp_to_text: dict[str, str] = {}
    for tp, tx in zip(df["text_path"].astype(str).tolist(), texts, strict=True):
        tp_to_text.setdefault(tp, tx)
    return tp_to_text


def embed_dataframe(
    encoder_factory,
    df: pd.DataFrame,
    *,
    cache_path: Path | None,
    batch_size: int,
) -> np.ndarray:
    """Embed every row of ``df`` (aligned to row order), deduplicating by
    ``text_path`` so each unique filing is encoded once; reuse a disk cache when
    ``cache_path`` is given (frozen encoder ⇒ cache is seed/split-independent).

    ``encoder_factory`` is a zero-arg callable that builds the (expensive 7B)
    encoder; it is invoked ONLY when there is at least one cache miss, so a fully
    cached run (e.g. later seeds) never loads the encoder at all.
    """
    if "text_path" not in df.columns:
        raise ValueError("rows must contain a 'text_path' column")
    text_paths = df["text_path"].astype(str).tolist()
    store = _load_emb_store(cache_path) if cache_path is not None else {}
    unique_tps = list(dict.fromkeys(text_paths))
    missing = [tp for tp in unique_tps if tp not in store]
    if missing:
        # Read ONLY the missing texts (pyarrow filter) and build the 7B encoder only
        # when there is something to encode: a fully-cached run skips both, and a
        # partial encode (predict's test rows) never loads all 144k texts into RAM.
        tp_to_text = _missing_texts(df, missing)
        encoder = encoder_factory()
        new = encoder.encode([tp_to_text[tp] for tp in missing], batch_size=batch_size)
        for tp, row in zip(missing, new, strict=True):
            store[tp] = np.asarray(row, dtype=np.float32)
        if cache_path is not None:
            _persist_emb_store(cache_path, store)
    return np.stack([store[tp] for tp in text_paths]).astype(np.float32)


def _select_val_emb(
    val_emb: np.ndarray | None,
    val_target: np.ndarray | None,
    val_horizons: np.ndarray | None,
    horizon: Any,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if val_emb is None or val_horizons is None:
        return None, None
    vmask = val_horizons == horizon
    if not bool(vmask.any()):
        return None, None
    return val_emb[vmask], np.asarray(val_target, dtype=float)[vmask]


class _FrozenLLMForecaster(VolatilityForecaster):
    """Shared scaffold for frozen-LLM probes: lazy encoder, embedding cache,
    per-horizon dispatch, and early-stopping integration. Subclasses build the
    trainable module(s) and the per-horizon forward/snapshot."""

    name = "frozen_llm_base"

    def __init__(
        self,
        *,
        pretrained: str,
        max_length: int = 4096,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        lr: float = 1.0e-4,
        weight_decay: float = 0.01,
        batch_size: int = 64,
        max_epochs: int = 20,
        early_stopping: bool = True,
        es_patience: int = 3,
        es_min_delta: float = 0.0,
        mixed_precision: str = "no",
        warmup_ratio: float = 0.0,
        log_target: bool = True,
        instruction: str | None = None,
        normalize_emb: bool = True,
        encode_batch_size: int = 8,
        cache_embeddings: bool = True,
        tokenizer_threads: int | None = None,
        device: str = "auto",
        checkpoint: bool = True,
        checkpoint_dir: str | Path | None = None,
        seed: int | None = None,
        strategy: str | None = None,
    ) -> None:
        self.encoder_cfg = EncoderConfig(pretrained=pretrained, max_length=max_length)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.early_stopping = bool(early_stopping)
        self.es_patience = int(es_patience)
        self.es_min_delta = float(es_min_delta)
        self.mixed_precision = str(mixed_precision)
        self.warmup_ratio = float(warmup_ratio)
        self.log_target = bool(log_target)
        self.instruction = instruction
        self.normalize_emb = bool(normalize_emb)
        self.encode_batch_size = int(encode_batch_size)
        self.cache_embeddings = bool(cache_embeddings)
        self.tokenizer_threads = None if tokenizer_threads is None else int(tokenizer_threads)
        self.device = _resolve_device(device)
        self.checkpoint = bool(checkpoint)
        self.checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self.seed = None if seed is None else int(seed)
        self.strategy = strategy or self.name
        self.embedding_dim_: int | None = None
        self._encoder: FrozenLLMEncoder | None = None
        self.models_: dict[Any, dict[str, Any]] = {}
        self.val_curves_: dict[Any, list[dict[str, Any]]] = {}

    # --- frozen encoder + embedding cache --------------------------------

    def _get_encoder(self) -> FrozenLLMEncoder:
        if self._encoder is None:
            self._encoder = FrozenLLMEncoder(
                pretrained=self.encoder_cfg.pretrained,
                max_length=self.encoder_cfg.max_length,
                instruction=self.instruction,
                normalize=self.normalize_emb,
                device=self.device,
                tokenizer_threads=self.tokenizer_threads,
            )
            self.embedding_dim_ = self._encoder.hidden_size
        return self._encoder

    def _cache_path(self) -> Path | None:
        if not self.cache_embeddings:
            return None
        slug = self.encoder_cfg.pretrained.replace("/", "_")
        dtype = "bf16" if self.device.type == "cuda" else "fp32"
        digest = hashlib.sha256(
            f"{self.instruction or ''}|norm={self.normalize_emb}|dtype={dtype}".encode()
        ).hexdigest()[:8]
        fname = f"{slug}__len{self.encoder_cfg.max_length}__lasttok__{digest}.parquet"
        return data_path("processed", "_llm_emb_cache", fname)

    def _encode(self, df: pd.DataFrame) -> np.ndarray:
        # Pass the encoder *factory* (not a built encoder): embed_dataframe only
        # loads the 7B model if there is a cache miss, so fully-cached runs skip it.
        emb = embed_dataframe(
            self._get_encoder,
            df,
            cache_path=self._cache_path(),
            batch_size=self.encode_batch_size,
        )
        if self.embedding_dim_ is None:
            self.embedding_dim_ = int(emb.shape[1])
        return emb

    @torch.inference_mode()
    def _head_forward(self, module: nn.Module, emb: np.ndarray) -> np.ndarray:
        """Batched forward of a trainable module over cached embeddings (RV-log scale)."""
        tensor = torch.from_numpy(np.asarray(emb, dtype=np.float32))
        out: list[np.ndarray] = []
        for start in range(0, len(tensor), self.batch_size):
            batch = tensor[start : start + self.batch_size].to(self.device)
            out.append(self._forward_emb(module, batch).detach().float().cpu().numpy())
        return np.concatenate(out) if out else np.empty(0, dtype=float)

    # --- hooks subclasses implement --------------------------------------

    def _forward_emb(self, module: nn.Module, emb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        df = _require_dataframe(X, name="X")
        emb = self._encode(df)
        horizons = _horizons(df)
        preds = np.empty(len(df), dtype=float)
        for horizon in sorted(set(horizons.tolist())):
            if horizon not in self.models_:
                raise ValueError(f"no {self.name} model fitted for horizon_days={horizon!r}")
            mask = horizons == horizon
            predict_start = monotonic()
            train_utils.log_predict_horizon_start(self, horizon=horizon, n_rows=int(mask.sum()))
            preds[mask] = self._predict_one(horizon, emb[mask])
            train_utils.log_predict_horizon_done(
                self, horizon=horizon, n_rows=int(mask.sum()), secs=monotonic() - predict_start
            )
        return preds

    def _predict_one(self, horizon: Any, emb: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class C5LLMProbe(_FrozenLLMForecaster):
    """C5 — text-only probe: frozen LLM embedding → per-horizon regression head."""

    name = "C5_llm"

    def __init__(self, *, pretrained: str = "Alibaba-NLP/gte-Qwen2-7B-instruct", **kwargs) -> None:
        super().__init__(pretrained=pretrained, **kwargs)

    # --- forecaster API --------------------------------------------------

    def fit(self, X_train, y_train, *, X_val=None, y_val=None) -> None:
        df = _require_dataframe(X_train, name="X_train")
        target = np.asarray(y_train, dtype=float)
        if len(df) != len(target):
            raise ValueError(f"X_train has {len(df)} rows but y_train has {len(target)} values")
        emb = self._encode(df)
        horizons = _horizons(df)
        val_emb, val_target, val_horizons = self._prepare_val(X_val, y_val)
        self.models_ = {}
        self.val_curves_ = {}
        for horizon in sorted(set(horizons.tolist())):
            mask = horizons == horizon
            h_val_emb, h_val_target = _select_val_emb(val_emb, val_target, val_horizons, horizon)
            self._fit_one(horizon, emb[mask], target[mask], h_val_emb, h_val_target)

    def _prepare_val(self, X_val, y_val):
        if X_val is None or y_val is None:
            return None, None, None
        df = _require_dataframe(X_val, name="X_val")
        if len(df) == 0:
            return None, None, None
        return self._encode(df), np.asarray(y_val, dtype=float), _horizons(df)

    # --- internals -------------------------------------------------------

    def _build_head(self) -> VolatilityHead:
        if self.embedding_dim_ is None:
            raise RuntimeError("embedding_dim_ unknown; encode texts before building the head")
        return VolatilityHead(
            self.embedding_dim_,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            eps=_EPSILON,
            positive=not self.log_target,
        ).to(self.device)

    def _forward_emb(self, module: nn.Module, emb: torch.Tensor) -> torch.Tensor:
        return module(emb)

    def _fit_one(
        self,
        horizon: Any,
        emb: np.ndarray,
        target: np.ndarray,
        val_emb: np.ndarray | None = None,
        val_target: np.ndarray | None = None,
    ) -> None:
        n_train = len(emb)
        horizon_start = monotonic()
        train_utils.log_fit_horizon_start(self, horizon=horizon, n_train=n_train)
        loaded_state = train_utils.maybe_load_horizon_checkpoint(
            self, horizon=horizon, n_train=n_train
        )
        if loaded_state is not None:
            self.models_[horizon] = loaded_state
            checkpoint_path = train_utils.horizon_checkpoint_path(self, horizon)
            if checkpoint_path is None:
                raise RuntimeError("loaded checkpoint without checkpoint path")
            train_utils.log_fit_horizon_skip(
                self, horizon=horizon, n_train=n_train, checkpoint_path=checkpoint_path
            )
            return

        head = self._build_head()
        target_log = _maybe_log(target, log_target=self.log_target)
        dataset = TensorDataset(
            torch.from_numpy(emb.astype(np.float32)),
            torch.from_numpy(target_log.astype(np.float32)),
        )
        # pin_memory speeds the H2D copy of the (large) cached-embedding batches; workers
        # stay at 0 since __getitem__ is a trivial resident-tensor index (IPC would dominate).
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.device.type == "cuda",
        )
        optimiser = torch.optim.AdamW(head.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        scheduler = train_utils.make_scheduler(
            optimiser,
            steps_per_epoch=len(loader),
            max_epochs=self.max_epochs,
            grad_accumulation_steps=1,
            warmup_ratio=self.warmup_ratio,
        )

        def forward_batch(batch):
            emb_b, target_b = batch
            return head(emb_b.to(self.device)), target_b.to(self.device)

        def snapshot_state():
            return {
                "head_state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            }

        val_eval = None
        if val_emb is not None and len(val_emb):
            val_target_arr = np.asarray(val_target, dtype=float)
            val_target_log = _maybe_log(val_target_arr, log_target=self.log_target)

            def val_eval():
                raw = self._head_forward(head, val_emb)
                if raw.size == 0:
                    return float("nan"), float("nan")
                val_loss = float(np.mean((raw - val_target_log) ** 2))
                preds_rv = _maybe_exp(raw, log_target=self.log_target)
                return val_loss, _r2(val_target_arr, preds_rv)

        best_state, curve = train_utils.run_training_loop(
            self,
            horizon=horizon,
            loader=loader,
            optimiser=optimiser,
            loss_fn=loss_fn,
            forward_batch=forward_batch,
            snapshot_state=snapshot_state,
            modules=[head],
            val_eval=val_eval,
            scheduler=scheduler,
            grad_accumulation_steps=1,
            max_epochs=self.max_epochs,
            early_stopping=self.early_stopping,
            patience=self.es_patience,
            min_delta=self.es_min_delta,
        )
        self.models_[horizon] = best_state
        self.val_curves_[horizon] = curve
        train_utils.save_horizon_checkpoint(
            self, horizon=horizon, n_train=n_train, state=self.models_[horizon]
        )
        train_utils.log_fit_horizon_done(
            self, horizon=horizon, n_train=n_train, secs=monotonic() - horizon_start
        )

    @torch.inference_mode()
    def _predict_one(self, horizon: Any, emb: np.ndarray) -> np.ndarray:
        head = self._build_head()
        head.load_state_dict(self.models_[horizon]["head_state"])
        head.eval()
        raw = self._head_forward(head, emb)
        return _maybe_exp(raw, log_target=self.log_target)

    def save(self, path: Path) -> None:
        if not self.models_:
            raise RuntimeError(f"{self.name} must be fitted before save")
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "encoder_cfg": self.encoder_cfg,
            "embedding_dim": self.embedding_dim_,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping": self.early_stopping,
            "es_patience": self.es_patience,
            "es_min_delta": self.es_min_delta,
            "mixed_precision": self.mixed_precision,
            "warmup_ratio": self.warmup_ratio,
            "log_target": self.log_target,
            "instruction": self.instruction,
            "normalize_emb": self.normalize_emb,
            "encode_batch_size": self.encode_batch_size,
            "cache_embeddings": self.cache_embeddings,
            "checkpoint": self.checkpoint,
            "seed": self.seed,
            "strategy": self.strategy,
            "models": self.models_,
        }
        with save_path.open("wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> C5LLMProbe:
        with Path(path).open("rb") as fh:
            state = pickle.load(fh)
        model = cls(
            pretrained=state["encoder_cfg"].pretrained,
            max_length=state["encoder_cfg"].max_length,
            hidden_dim=int(state["hidden_dim"]),
            dropout=float(state["dropout"]),
            lr=float(state["lr"]),
            weight_decay=float(state["weight_decay"]),
            batch_size=int(state["batch_size"]),
            max_epochs=int(state["max_epochs"]),
            early_stopping=bool(state.get("early_stopping", True)),
            es_patience=int(state.get("es_patience", 3)),
            es_min_delta=float(state.get("es_min_delta", 0.0)),
            mixed_precision=str(state.get("mixed_precision", "no")),
            warmup_ratio=float(state.get("warmup_ratio", 0.0)),
            log_target=bool(state["log_target"]),
            instruction=state.get("instruction"),
            normalize_emb=bool(state.get("normalize_emb", True)),
            encode_batch_size=int(state.get("encode_batch_size", 8)),
            cache_embeddings=bool(state.get("cache_embeddings", True)),
            seed=state.get("seed"),
            strategy=state.get("strategy"),
        )
        model.embedding_dim_ = None if state.get("embedding_dim") is None else int(
            state["embedding_dim"]
        )
        model.models_ = state["models"]
        return model
