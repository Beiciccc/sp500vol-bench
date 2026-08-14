"""C-block + S2 sliding-window chunking with mean pooling.

For each filing:
  1. Tokenise the full text WITHOUT truncation, then split into chunks of
     ``max_length`` tokens with stride ``chunk_stride`` (50% overlap by default).
  2. Cap at ``max_chunks`` chunks per filing to bound memory.
  3. Encode every chunk via the underlying encoder → CLS embedding per chunk.
  4. Mean-pool CLS embeddings (mask-aware over real-vs-padded chunks).
  5. Apply the volatility head on the pooled embedding.

This is the headline ``S2 chunk-mean`` strategy in AB1. Comparing it head-to-head
with S1 truncation on the same encoder is the core experiment for H3.
"""

from __future__ import annotations

import functools
import hashlib
import os
import pickle
from collections.abc import Iterable
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sp500vol.models.neural_text import _train_utils as train_utils
from sp500vol.models.neural_text.bert_s1 import (
    BertS1,
    _maybe_exp,
    _maybe_log,
    _r2,
)
from sp500vol.models.neural_text.encoders import CLSEncoder
from sp500vol.utils.paths import data_path


class _ChunkedTextDataset(Dataset):
    """Pre-tokenises filings into chunks; yields one filing per __getitem__.

    Tokenisation happens once at dataset construction (cached in memory) so the
    per-epoch cost is just the forward pass.
    """

    def __init__(
        self,
        texts: list[str],
        targets: np.ndarray,
        encoder: CLSEncoder,
        *,
        chunk_stride: int,
        max_chunks: int,
        tokenization_batch_size: int = 1,
        prebuilt_items: list[dict[str, torch.Tensor]] | None = None,
    ) -> None:
        if len(texts) != len(targets):
            raise ValueError("texts and targets length mismatch")
        self.targets = targets.astype(np.float32)
        self.max_chunks = int(max_chunks)
        self.encoder_max_length = encoder.cfg.max_length

        # Reuse pre-tokenised per-filing chunk items when supplied (cross-horizon cache);
        # otherwise tokenise here. Either way self.items is one chunk-dict per row, in order.
        if prebuilt_items is not None:
            if len(prebuilt_items) != len(targets):
                raise ValueError("prebuilt_items and targets length mismatch")
            self.items = list(prebuilt_items)
            return
        self.items = []
        pad_id = encoder.tokenizer.pad_token_id or 0
        batch_size = max(1, int(tokenization_batch_size))
        for start in range(0, len(texts), batch_size):
            self.items.extend(
                _tokenize_chunk_batch(
                    texts[start : start + batch_size],
                    encoder=encoder,
                    chunk_stride=chunk_stride,
                    max_chunks=self.max_chunks,
                    pad_id=pad_id,
                )
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "n_chunks": int(item["input_ids"].shape[0]),
            "target": float(self.targets[idx]),
        }


def _tokenize_chunk_batch(
    texts: list[str],
    *,
    encoder: CLSEncoder,
    chunk_stride: int,
    max_chunks: int,
    pad_id: int,
) -> list[dict[str, torch.Tensor]]:
    if not texts:
        return []
    tok = encoder.tokenizer(
        texts,
        max_length=encoder.cfg.max_length,
        truncation=True,
        return_overflowing_tokens=True,
        stride=int(chunk_stride),
        padding="max_length",
        return_tensors="pt",
        return_attention_mask=True,
    )
    sample_mapping = tok.get("overflow_to_sample_mapping")
    if sample_mapping is None:
        # Slow tokenizers may omit overflow mapping for batched input; keep the
        # existing correct path rather than silently mixing chunks across docs.
        return [
            _tokenize_single_chunked_text(
                text,
                encoder=encoder,
                chunk_stride=chunk_stride,
                max_chunks=max_chunks,
                pad_id=pad_id,
            )
            for text in texts
        ]

    ids = tok["input_ids"]
    mask = tok["attention_mask"]
    sample_ids = sample_mapping.tolist()
    groups: list[list[int]] = [[] for _ in texts]
    for idx, mapped in enumerate(sample_ids):
        sample_idx = int(mapped)
        if len(groups[sample_idx]) < max_chunks:
            groups[sample_idx].append(idx)

    items: list[dict[str, torch.Tensor]] = []
    for chunk_idx in groups:
        if chunk_idx:
            index = torch.tensor(chunk_idx, dtype=torch.long)
            item_ids = ids.index_select(0, index)
            item_mask = mask.index_select(0, index)
        else:
            item_ids = torch.full((1, encoder.cfg.max_length), pad_id, dtype=torch.long)
            item_mask = torch.zeros((1, encoder.cfg.max_length), dtype=torch.long)
        items.append({"input_ids": item_ids, "attention_mask": item_mask})
    return items


def _tokenize_single_chunked_text(
    text: str,
    *,
    encoder: CLSEncoder,
    chunk_stride: int,
    max_chunks: int,
    pad_id: int,
) -> dict[str, torch.Tensor]:
    tok = encoder.tokenizer(
        text,
        max_length=encoder.cfg.max_length,
        truncation=True,
        return_overflowing_tokens=True,
        stride=int(chunk_stride),
        padding="max_length",
        return_tensors="pt",
        return_attention_mask=True,
    )
    ids = tok["input_ids"]
    mask = tok["attention_mask"]
    if ids.shape[0] == 0:  # empty filing → one all-pad chunk
        ids = torch.full((1, encoder.cfg.max_length), pad_id, dtype=torch.long)
        mask = torch.zeros((1, encoder.cfg.max_length), dtype=torch.long)
    return {"input_ids": ids[:max_chunks], "attention_mask": mask[:max_chunks]}


# --- disk-backed chunk token cache ---------------------------------------------
# Chunk tokenisation is the dominant GPU-idle cost: ~29min for one long_form fit, and
# the in-memory _ChunkTokenCache is rebuilt every run (36 chunk runs => ~71 GPU-idle h
# of redundant tokenisation). A disk cache keyed by the tokenisation params, shared
# across ALL runs/seeds, tokenises each unique filing ONCE matrix-wide (~6h one-time).
# Same correctness argument as the 7-8B embedding disk cache (qwen_llm.py): tokenisation
# is a pure fn of (text, tokenizer, max_length, stride, max_chunks) => byte-identical,
# weight/seed/horizon-independent. FinBERT s2/s3/s4 share one cache (same geometry).
_CHUNK_TOK_DISK_ENV = "SP500VOL_CHUNK_TOK_DISK_CACHE"
_TOK_STORES: dict[str, dict[str, dict[str, torch.Tensor]]] = {}


def _chunk_tok_disk_enabled() -> bool:
    """On by default; a falsy env disables the disk layer (in-memory cache still used)."""
    return os.environ.get(_CHUNK_TOK_DISK_ENV, "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }


def _chunk_cache_path(encoder: CLSEncoder, max_chunks: int, chunk_stride: int) -> Path | None:
    """Disk path keyed by the params that determine the bytes (tokenizer id + max_length
    + max_chunks + stride). Returns None when disabled. Mirrors qwen_llm._cache_path."""
    if not _chunk_tok_disk_enabled():
        return None
    slug = encoder.cfg.pretrained.replace("/", "_")
    digest = hashlib.sha256(
        f"len{encoder.cfg.max_length}|chunks{int(max_chunks)}|stride{int(chunk_stride)}".encode()
    ).hexdigest()[:8]
    return data_path("processed", "_chunk_tok_cache", f"{slug}__chunktok__{digest}.parquet")


def _load_tok_store(cache_path: Path) -> dict[str, dict[str, torch.Tensor]]:
    """Load the on-disk chunk-token store (process-cached in _TOK_STORES). Each row is one
    filing's [n_chunks, max_length] input_ids + attention_mask stored as RAW BYTES
    (int32 ids / uint8 mask) — ~10x less RAM than Python-int lists. astype(int64) copies
    into writable buffers (torch needs long; no read-only aliasing of the parquet buffer)."""
    key = str(cache_path)
    store = _TOK_STORES.get(key)
    if store is None:
        store = {}
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            for tp, ids_b, mask_b, nc in zip(
                df["text_path"],
                df["input_ids"],
                df["attention_mask"],
                df["n_chunks"],
                strict=True,
            ):
                n = int(nc)
                ids = np.frombuffer(ids_b, dtype=np.int32).astype(np.int64).reshape(n, -1)
                mask = np.frombuffer(mask_b, dtype=np.uint8).astype(np.int64).reshape(n, -1)
                store[str(tp)] = {
                    "input_ids": torch.from_numpy(ids),
                    "attention_mask": torch.from_numpy(mask),
                }
        _TOK_STORES[key] = store
    return store


def _flatten_chunk_item(item: dict[str, torch.Tensor]) -> tuple[bytes, bytes, int]:
    """Serialise one filing's chunk tensors to raw bytes (int32 ids, uint8 mask)."""
    ids = item["input_ids"]
    mask = item["attention_mask"]
    return (
        ids.to(torch.int32).reshape(-1).cpu().numpy().tobytes(),
        mask.to(torch.uint8).reshape(-1).cpu().numpy().tobytes(),
        int(ids.shape[0]),
    )


def _persist_tok_store(
    cache_path: Path, new_items: dict[str, dict[str, torch.Tensor]]
) -> None:
    """Persist freshly-tokenised filings, merging with the on-disk store INSIDE a FileLock
    so parallel seeds/models sharing this cache never drop each other's items (overlapping
    keys are identical, so disk values are kept harmlessly). Mirrors qwen_llm._persist_emb_store."""
    if not new_items:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cache_path) + ".lock"):
        rows: dict[str, tuple[bytes, bytes, int]] = {}
        if cache_path.exists():
            disk = pd.read_parquet(cache_path)
            for tp, ids_b, mask_b, nc in zip(
                disk["text_path"],
                disk["input_ids"],
                disk["attention_mask"],
                disk["n_chunks"],
                strict=True,
            ):
                rows[str(tp)] = (bytes(ids_b), bytes(mask_b), int(nc))
        for tp, item in new_items.items():
            rows.setdefault(str(tp), _flatten_chunk_item(item))
        frame = pd.DataFrame(
            {
                "text_path": list(rows),
                "input_ids": [r[0] for r in rows.values()],
                "attention_mask": [r[1] for r in rows.values()],
                "n_chunks": [r[2] for r in rows.values()],
            }
        )
        tmp = cache_path.with_suffix(f".parquet.{os.getpid()}.tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(cache_path)
        merged = _TOK_STORES.get(str(cache_path), {})
        merged.update(new_items)
        _TOK_STORES[str(cache_path)] = merged


class _ChunkTokenCache:
    """Tokenise each unique filing into chunk items ONCE, keyed by text_path; gather
    per-filing items in text_paths order. Byte-identical to _ChunkedTextDataset's
    tokenisation (reuses _tokenize_chunk_batch). In-memory memo avoids re-tokenising a
    filing once per horizon (and per val epoch); a DISK layer (when enabled) additionally
    shares tokenised filings across ALL runs/seeds. Held for one call; never pickled."""

    def __init__(self) -> None:
        self._memo: dict[str, dict[str, torch.Tensor]] = {}
        self._disk_loaded = False
        self._disk_path: Path | None = None

    def gather(
        self,
        text_paths: list[str],
        texts: list[str],
        *,
        encoder: CLSEncoder,
        chunk_stride: int,
        max_chunks: int,
        tokenization_batch_size: int,
    ) -> list[dict[str, torch.Tensor]]:
        # Warm the in-memory memo from the shared on-disk cache on first gather.
        if not self._disk_loaded:
            self._disk_path = _chunk_cache_path(encoder, max_chunks, chunk_stride)
            if self._disk_path is not None:
                for tp, item in _load_tok_store(self._disk_path).items():
                    self._memo.setdefault(tp, item)
            self._disk_loaded = True
        need = list(dict.fromkeys(tp for tp in text_paths if tp not in self._memo))
        if need:
            tp_to_text: dict[str, str] = {}
            for tp, tx in zip(text_paths, texts, strict=True):
                tp_to_text.setdefault(tp, tx)
            pad_id = encoder.tokenizer.pad_token_id or 0
            bs = max(1, int(tokenization_batch_size))
            fresh: dict[str, dict[str, torch.Tensor]] = {}
            for start in range(0, len(need), bs):
                batch_tps = need[start : start + bs]
                items = _tokenize_chunk_batch(
                    [tp_to_text[tp] for tp in batch_tps],
                    encoder=encoder,
                    chunk_stride=chunk_stride,
                    max_chunks=max_chunks,
                    pad_id=pad_id,
                )
                for tp, item in zip(batch_tps, items, strict=True):
                    self._memo[tp] = item
                    fresh[tp] = item
            if self._disk_path is not None:
                _persist_tok_store(self._disk_path, fresh)
        return [self._memo[tp] for tp in text_paths]


def _collate_chunks(
    batch: list[dict[str, Any]],
    *,
    pad_id: int,
    max_length: int,
    max_chunks: int,
) -> dict[str, torch.Tensor]:
    """Pad every filing to (k_max, max_length) and stack along batch dim, where k_max is the
    batch-max REAL chunk count (<= max_chunks) — not the fixed cap. The chunk axis is sized to
    the batch so filings with few chunks don't drag in empty slots. Bit-identical to padding to
    max_chunks: every pooling variant masks chunk slots beyond chunk_counts[i]."""
    b = len(batch)
    k_max = max(1, max(int(item["n_chunks"]) for item in batch))
    out_ids = torch.full((b, k_max, max_length), pad_id, dtype=torch.long)
    out_mask = torch.zeros((b, k_max, max_length), dtype=torch.long)
    chunk_counts = torch.zeros(b, dtype=torch.long)
    targets = torch.zeros(b, dtype=torch.float32)

    for i, item in enumerate(batch):
        k = int(item["n_chunks"])
        out_ids[i, :k] = item["input_ids"]
        out_mask[i, :k] = item["attention_mask"]
        chunk_counts[i] = k
        targets[i] = item["target"]

    return {
        "input_ids": out_ids,
        "attention_mask": out_mask,
        "chunk_counts": chunk_counts,
        "targets": targets,
    }


def encode_real_chunks(
    encoder: CLSEncoder, ids: torch.Tensor, mask: torch.Tensor, chunk_counts: torch.Tensor
) -> torch.Tensor:
    """Encode ONLY the real chunks and scatter the embeddings back to (B, K, H), leaving pad
    chunk slots as zeros. Shared by the S2/S3/S4 forwards.

    BIT-IDENTICAL to the old `encoder(ids.view(B*K, L)).view(B, K, H)` then pool — verified
    max|diff|=0 on the real finbert encoder. The encoder is per-row independent, so encoding the
    real chunks in a smaller (sum_k, L) batch yields the exact same row embeddings as encoding
    them inside the full (B*K, L) batch; the dropped chunk slots are pure all-pad placeholders
    whose embeddings every pooling variant masks out anyway (S2 ×0, S3 −inf softmax, S4
    src_key_padding_mask + ×0). We just stop paying the encoder forward+backward for them.

    NOTE: chunks are kept at the full tokeniser length (no within-chunk token trim). Trimming the
    token axis to the batch-max real length is NOT bit-identical for BERT full attention — the
    Q·Kᵀ matmul reduces in a different order at a different sequence length, giving ~1e-6
    drift — so it is deliberately omitted to keep the change exactly result-preserving."""
    b, k, length = ids.shape
    real = torch.arange(k, device=ids.device).unsqueeze(0) < chunk_counts.unsqueeze(1)  # (B, K)
    real_flat = real.reshape(-1)  # (B*K,)
    hidden = encoder.hidden_size
    if not bool(real_flat.any()):
        return torch.zeros(b, k, hidden, device=ids.device)
    ids_flat = ids.reshape(b * k, length)[real_flat]
    mask_flat = mask.reshape(b * k, length)[real_flat]
    emb_real = encoder(ids_flat, mask_flat)  # (sum_k, H) — only the real chunks
    emb_full = torch.zeros(b * k, emb_real.shape[-1], device=ids.device, dtype=emb_real.dtype)
    emb_full[real_flat] = emb_real
    return emb_full.view(b, k, -1)


class BertS2(BertS1):
    """C-family + S2 chunk-mean strategy."""

    name = "C1_bert_s2"

    def __init__(
        self,
        *,
        chunk_stride: int = 256,
        max_chunks: int = 16,
        batch_size: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(batch_size=batch_size, **kwargs)
        self.chunk_stride = int(chunk_stride)
        self.max_chunks = int(max_chunks)

    # fit() / predict() are inherited from BertS1 — identical text loading and
    # per-horizon dispatch; validation threading is defined once in BertS1.fit.

    # --- internals -------------------------------------------------------

    def _extra_modules(self) -> list[nn.Module]:
        """Modules beyond (encoder, head) to train/checkpoint. S2 has none;
        S3 adds the attention pool, S4 the chunk encoder."""
        return []

    def _extra_params(self) -> list:
        params: list = []
        for module in self._extra_modules():
            params.extend(module.parameters())
        return params

    def _snapshot_chunked(self, encoder, head) -> dict[str, Any]:
        return {
            "encoder_state": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
            "head_state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
        }

    def _new_token_cache(self) -> _ChunkTokenCache:
        return _ChunkTokenCache()

    def _make_chunk_loader(
        self, encoder, texts, targets, *, shuffle: bool, text_paths=None
    ) -> DataLoader:
        cache = getattr(self, "_tok_cache", None)
        prebuilt = None
        if cache is not None and text_paths is not None:
            prebuilt = cache.gather(
                text_paths,
                texts,
                encoder=encoder,
                chunk_stride=self.chunk_stride,
                max_chunks=self.max_chunks,
                tokenization_batch_size=self._runtime_tokenization_batch_size(),
            )
        dataset = _ChunkedTextDataset(
            texts,
            targets,
            encoder,
            chunk_stride=self.chunk_stride,
            max_chunks=self.max_chunks,
            tokenization_batch_size=self._runtime_tokenization_batch_size(),
            prebuilt_items=prebuilt,
        )
        pad_id = encoder.tokenizer.pad_token_id or 0
        # Bind max_length as a plain int (NOT encoder.cfg.max_length) so the partial never
        # captures the encoder object — a captured encoder would serialise full BERT weights
        # into every worker. partial over the top-level _collate_chunks is picklable.
        max_length = int(encoder.cfg.max_length)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=functools.partial(
                _collate_chunks, pad_id=pad_id, max_length=max_length, max_chunks=self.max_chunks
            ),
            **self._loader_kwargs(),
        )

    def _fit_one(
        self,
        horizon: Any,
        texts: list[str],
        target: np.ndarray,
        *,
        text_paths: list[str] | None = None,
        val_texts: list[str] | None = None,
        val_target: np.ndarray | None = None,
        val_text_paths: list[str] | None = None,
    ) -> None:
        self._configure_tokenizer_runtime()
        n_train = len(texts)
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

        encoder, head = self._build_modules()
        extra = self._extra_modules()
        loader = self._make_chunk_loader(
            encoder,
            texts,
            _maybe_log(np.asarray(target, dtype=float), log_target=self.log_target),
            shuffle=True,
            text_paths=text_paths,
        )

        params = list(encoder.parameters()) + list(head.parameters()) + self._extra_params()
        optimiser = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        scheduler = train_utils.make_scheduler(
            optimiser,
            steps_per_epoch=len(loader),
            max_epochs=self.max_epochs,
            grad_accumulation_steps=self.grad_accumulation_steps,
            warmup_ratio=self.warmup_ratio,
        )

        def forward_batch(batch):
            emb = self._forward_chunked(batch, encoder)
            return head(emb), batch["targets"].to(self.device)

        def snapshot_state():
            return self._snapshot_chunked(encoder, head)

        val_eval = None
        if val_texts:
            val_target_arr = np.asarray(val_target, dtype=float)
            val_target_log = _maybe_log(val_target_arr, log_target=self.log_target)

            def val_eval():
                return self._eval_val_chunked(
                    encoder,
                    head,
                    val_texts,
                    val_target_arr,
                    val_target_log,
                    text_paths=val_text_paths,
                )

        best_state, curve = train_utils.run_training_loop(
            self,
            horizon=horizon,
            loader=loader,
            optimiser=optimiser,
            loss_fn=loss_fn,
            forward_batch=forward_batch,
            snapshot_state=snapshot_state,
            modules=[encoder, head, *extra],
            val_eval=val_eval,
            scheduler=scheduler,
            grad_accumulation_steps=self.grad_accumulation_steps,
            max_epochs=self.max_epochs,
            early_stopping=self.early_stopping,
            patience=self.es_patience,
            min_delta=self.es_min_delta,
        )
        self.models_[horizon] = best_state
        self.val_curves_[horizon] = curve
        train_utils.save_horizon_checkpoint(
            self,
            horizon=horizon,
            n_train=n_train,
            state=self.models_[horizon],
        )
        train_utils.log_fit_horizon_done(
            self, horizon=horizon, n_train=n_train, secs=monotonic() - horizon_start
        )

    @torch.inference_mode()
    def _eval_val_chunked(
        self, encoder, head, texts, target_raw, target_log, *, text_paths=None
    ) -> tuple[float, float]:
        raw = self._raw_outputs_chunked(encoder, head, texts, text_paths=text_paths)
        if raw.size == 0:
            return float("nan"), float("nan")
        val_loss = float(np.mean((raw - target_log) ** 2))
        preds_rv = _maybe_exp(raw, log_target=self.log_target)
        return val_loss, _r2(target_raw, preds_rv)

    @torch.inference_mode()
    def _raw_outputs_chunked(self, encoder, head, texts, *, text_paths=None) -> np.ndarray:
        text_list = list(texts)
        loader = self._make_chunk_loader(
            encoder,
            text_list,
            np.zeros(len(text_list), dtype=np.float32),
            shuffle=False,
            text_paths=text_paths,
        )
        out: list[np.ndarray] = []
        for batch in loader:
            emb = self._forward_chunked(batch, encoder)
            out.append(head(emb).detach().float().cpu().numpy())
        return np.concatenate(out) if out else np.empty(0, dtype=float)

    @torch.inference_mode()
    def _predict_one(self, horizon: Any, texts: Iterable[str], *, text_paths=None) -> np.ndarray:
        self._configure_tokenizer_runtime()
        encoder, head = self._build_modules()
        encoder.load_state_dict(self.models_[horizon]["encoder_state"])
        head.load_state_dict(self.models_[horizon]["head_state"])
        encoder.eval()
        head.eval()

        text_list = list(texts)
        # Dummy targets just reuse the same chunk pipeline (and cross-horizon token cache).
        loader = self._make_chunk_loader(
            encoder,
            text_list,
            np.zeros(len(text_list), dtype=np.float32),
            shuffle=False,
            text_paths=text_paths,
        )

        preds: list[float] = []
        for batch in loader:
            emb = self._forward_chunked(batch, encoder)
            raw = head(emb).detach().float().cpu().numpy()
            preds.extend(_maybe_exp(raw, log_target=self.log_target).tolist())
        return np.asarray(preds, dtype=float)

    def _forward_chunked(self, batch: dict[str, torch.Tensor], encoder: CLSEncoder) -> torch.Tensor:
        """Encode (B, K, L) tokens → (B, hidden) via mask-aware mean pool over K."""
        ids = batch["input_ids"].to(self.device)  # (B, K, L)
        mask = batch["attention_mask"].to(self.device)  # (B, K, L)
        chunk_counts = batch["chunk_counts"].to(self.device)  # (B,)
        b, k, length = ids.shape

        emb = encode_real_chunks(encoder, ids, mask, chunk_counts)  # (B, K, H), pad slots = 0

        # Real-chunk mask: chunk_counts[i] is the number of valid chunks for filing i
        chunk_mask = (
            (torch.arange(k, device=self.device).unsqueeze(0) < chunk_counts.unsqueeze(1))
            .float()
            .unsqueeze(-1)
        )  # (B, K, 1)
        emb_sum = (emb * chunk_mask).sum(dim=1)  # (B, H)
        denom = chunk_counts.clamp(min=1).float().unsqueeze(-1)  # (B, 1)
        return emb_sum / denom

    # --- save/load mirror BertS1 but record S2-specific knobs ------------

    def save(self, path: Path) -> None:
        if not self.models_:
            raise RuntimeError(f"{self.name} must be fitted before save")
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "encoder_cfg": self.encoder_cfg,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "grad_accumulation_steps": self.grad_accumulation_steps,
            "log_target": self.log_target,
            "chunk_stride": self.chunk_stride,
            "max_chunks": self.max_chunks,
            "models": {
                horizon: {
                    "encoder_state": entry["encoder_state"],
                    "head_state": entry["head_state"],
                }
                for horizon, entry in self.models_.items()
            },
        }
        with save_path.open("wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> BertS2:
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
            grad_accumulation_steps=int(state.get("grad_accumulation_steps", 1)),
            log_target=bool(state["log_target"]),
            chunk_stride=int(state.get("chunk_stride", 256)),
            max_chunks=int(state.get("max_chunks", 16)),
        )
        model.models_ = {
            horizon: {
                "encoder_state": entry["encoder_state"],
                "head_state": entry["head_state"],
            }
            for horizon, entry in state["models"].items()
        }
        return model


class FinBertS2(BertS2):
    """C2 FinBERT + S2 chunk-mean."""

    name = "C2_finbert_s2"

    def __init__(self, *, pretrained: str = "ProsusAI/finbert", **kwargs) -> None:
        super().__init__(pretrained=pretrained, **kwargs)
