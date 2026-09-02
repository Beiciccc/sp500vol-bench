"""Probe safe CUDA batch sizes for C/D neural models (A100-40G target).

The parent process scans a model/batch grid (fixed grid, or adaptive
explore-then-target on ``--target-gb``) and launches one worker subprocess per
trial. The subprocess boundary keeps CUDA memory state clean after OOMs.

Three probe families, by model:
  * S1 / C4 Longformer / fusion: a real training step (encoder + head + backward)
    over ``training.batch_size``. C4 is a 4096-TOKEN encoder, so it is probed at
    full token length (not the 512-token char cap used for the S1 models).
  * C2 S2/S3/S4: the chunked training step over ``training.batch_size``.
  * C5_* / D3_*: a FROZEN 7-8B decoder-LLM encode forward over
    ``llm.encode_batch_size`` under ``torch.inference_mode`` (no head training,
    which is trivial on cached embeddings). D3_<x> shares C5_<x>'s encoder.

Device labels are read from torch at runtime; nothing here is 3090/24G specific.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

GRID: dict[str, list[int]] = {
    "C1_bert_s1": [8, 12, 16, 24],
    "C2_finbert_s1": [8, 12, 16, 24],
    "C3_roberta_s1": [8, 12, 16, 24],
    "C4_longformer": [2, 3, 4, 6, 8],
    "D1_concat_mlp": [8, 12, 16, 24],
    "D2_gated_fusion": [8, 12, 16, 24],
    "C2_finbert_s2": [2, 4, 6, 8],
    "C2_finbert_s3": [2, 4, 6, 8],
    "C2_finbert_s4": [2, 4, 6, 8],
    # Frozen 7-8B encode probes (encode_batch_size, inference-only forward).
    "C5_qwen3": [2, 4, 6, 8, 12, 16],
    "C5_gteqwen2": [2, 4, 6, 8, 12, 16],
    "C5_e5mistral": [2, 4, 6, 8, 12, 16],
    "D3_qwen3": [2, 4, 6, 8, 12, 16],
    "D3_gteqwen2": [2, 4, 6, 8, 12, 16],
    "D3_e5mistral": [2, 4, 6, 8, 12, 16],
}
CHUNK_MODELS = {"C2_finbert_s2", "C2_finbert_s3", "C2_finbert_s4"}
FUSION_MODELS = {"D1_concat_mlp", "D2_gated_fusion"}
# C5_*/D3_* are frozen decoder-LLM (7-8B) embedding probes. The VRAM-relevant op
# is FrozenLLMEncoder.encode (a forward at encode_batch_size / max_length under
# torch.inference_mode, NO backward), not head training on cached embeddings —
# so they take a dedicated encode-batch worker branch, not the S1 train step.
LLM_ENCODE_MODELS = {
    "C5_qwen3",
    "C5_gteqwen2",
    "C5_e5mistral",
    "D3_qwen3",
    "D3_gteqwen2",
    "D3_e5mistral",
}
# Longformer is a real 4096-TOKEN encoder, so unlike the 512-token S1 models its
# memory scales with sequence length. It must be probed at full token length, not
# the S1_PROBE_CHARS character cap (which would understate its VRAM by ~8x).
LONGFORMER_MODELS = {"C4_longformer"}
S1_PROBE_CHARS = 4_096
TARGET_GRID: dict[str, list[int]] = {
    "C1_bert_s1": [8, 16, 24, 32, 48, 64, 68, 72, 76, 80, 96, 112, 128],
    "C2_finbert_s1": [8, 16, 24, 32, 48, 64, 68, 72, 76, 80, 96, 112, 128],
    "C3_roberta_s1": [8, 16, 24, 32, 48, 64, 68, 72, 76, 80, 96, 112, 128],
    "C4_longformer": [2, 3, 4, 6, 8, 12, 16, 24, 32],
    "D1_concat_mlp": [8, 16, 24, 32, 48, 64, 68, 72, 76, 80, 96, 112, 128],
    "D2_gated_fusion": [8, 16, 24, 32, 48, 64, 68, 72, 76, 80, 96, 112, 128],
    "C2_finbert_s2": [2, 3, 4, 5, 6, 7, 8],
    "C2_finbert_s3": [2, 3, 4, 5, 6, 7, 8],
    "C2_finbert_s4": [2, 3, 4, 5, 6, 7, 8],
    "C5_qwen3": [2, 4, 6, 8, 12, 16, 24, 32],
    "C5_gteqwen2": [2, 4, 6, 8, 12, 16, 24, 32],
    "C5_e5mistral": [2, 4, 6, 8, 12, 16, 24, 32],
    "D3_qwen3": [2, 4, 6, 8, 12, 16, 24, 32],
    "D3_gteqwen2": [2, 4, 6, 8, 12, 16, 24, 32],
    "D3_e5mistral": [2, 4, 6, 8, 12, 16, 24, 32],
}
RAW_FIELDS = [
    "phase",
    "model",
    "batch_size",
    "ok",
    "oom",
    "steps",
    "warmup_steps",
    "mean_step_s",
    "median_step_s",
    "p90_step_s",
    "timed_step_s_total",
    "setup_s",
    "torch_peak_allocated_gb",
    "torch_peak_reserved_gb",
    "nvidia_smi_peak_gb",
    "device_name",
    "cuda_version",
    "text_path",
    "chunk_count",
    "error",
    "stderr_tail",
    "target_gb",
]


def main() -> int:
    args = _parse_args()
    if args.worker:
        print(json.dumps(_run_worker(args), sort_keys=True), flush=True)
        return 0
    return _run_parent(args)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--texts-dir", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results/tables/a100_40g_probe_raw.csv"
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=REPO_ROOT / "results/tables/a100_40g_probe_summary.csv",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=REPO_ROOT / "results/tables/a100_40g_probe_summary.md",
    )
    parser.add_argument("--models", nargs="*", choices=sorted(GRID), default=sorted(GRID))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--explore-steps", type=int, default=30)
    parser.add_argument("--explore-warmup-steps", type=int, default=3)
    # 40G card: leave ~4G for fragmentation/driver/activation spikes vs torch peak.
    parser.add_argument("--headroom-gb", type=float, default=36.0)
    parser.add_argument(
        "--target-gb",
        type=float,
        default=36.0,
        help="Adaptive mode: search batch sizes until torch allocated peak reaches this target "
        "(A100-40G default 36G). Pass an explicit value to override.",
    )
    parser.add_argument("--target-tolerance-gb", type=float, default=0.5)
    parser.add_argument(
        "--fixed-grid",
        action="store_true",
        help="Disable adaptive target mode and scan the full fixed GRID instead.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", choices=sorted(GRID), help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def _run_parent(args: argparse.Namespace) -> int:
    if args.fixed_grid:
        args.target_gb = None
    if args.target_gb is not None:
        return _run_target_parent(args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if args.out.exists():
        args.out.unlink()

    for model in args.models:
        for batch_size in GRID[model]:
            row = _launch_worker(
                args,
                model=model,
                batch_size=batch_size,
                phase="grid",
                steps=args.steps,
                warmup_steps=args.warmup_steps,
            )
            rows.append(row)
            _write_csv(args.out, rows, RAW_FIELDS)
            _write_csv(args.summary_out, _summary_rows(rows, args.headroom_gb), _summary_fields())
            _write_summary_md(args.summary_md, rows, args.headroom_gb)
            print(
                f"{model} batch={batch_size}: ok={row['ok']} oom={row['oom']} "
                f"torch_peak={row['torch_peak_allocated_gb']}GB step={row['mean_step_s']}s",
                flush=True,
            )
            if row["oom"]:
                break
    return 0


def _run_target_parent(args: argparse.Namespace) -> int:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if args.out.exists():
        args.out.unlink()

    for model in args.models:
        model_rows: list[dict[str, Any]] = []
        for batch_size in TARGET_GRID[model]:
            row = _launch_worker(
                args,
                model=model,
                batch_size=batch_size,
                phase="explore",
                steps=args.explore_steps,
                warmup_steps=args.explore_warmup_steps,
            )
            model_rows.append(row)
            rows.append(row)
            _persist_progress(args, rows)
            print(_status_line(row), flush=True)
            if row["oom"]:
                break
            peak = _row_float(row, "torch_peak_allocated_gb")
            if peak >= args.target_gb - args.target_tolerance_gb:
                break

        chosen = _choose_target_row(model_rows, target_gb=args.target_gb)
        if chosen is None:
            continue
        final_row = _launch_worker(
            args,
            model=model,
            batch_size=int(chosen["batch_size"]),
            phase="final",
            steps=args.steps,
            warmup_steps=args.warmup_steps,
        )
        rows.append(final_row)
        _persist_progress(args, rows)
        print(_status_line(final_row), flush=True)
    return 0


def _launch_worker(
    args: argparse.Namespace,
    *,
    model: str,
    batch_size: int,
    phase: str,
    steps: int,
    warmup_steps: int,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--texts-dir",
        str(args.texts_dir),
        "--model",
        model,
        "--batch-size",
        str(batch_size),
        "--steps",
        str(steps),
        "--warmup-steps",
        str(warmup_steps),
        "--seed",
        str(args.seed),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True, check=False)
    row = _parse_worker_stdout(proc.stdout)
    if row is None:
        row = _empty_row(model=model, batch_size=batch_size)
        row["ok"] = False
        row["error"] = "worker produced no JSON"
    row["phase"] = phase
    row["target_gb"] = "" if args.target_gb is None else args.target_gb
    row["stderr_tail"] = "\n".join(proc.stderr.splitlines()[-20:])
    if proc.returncode != 0 and row.get("ok") is not False:
        row["ok"] = False
        row["error"] = f"worker returncode={proc.returncode}"
    return row


def _persist_progress(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    _write_csv(args.out, rows, RAW_FIELDS)
    _write_csv(args.summary_out, _summary_rows(rows, args.headroom_gb), _summary_fields())
    _write_summary_md(args.summary_md, rows, args.headroom_gb)


def _status_line(row: dict[str, Any]) -> str:
    return (
        f"{row['phase']} {row['model']} batch={row['batch_size']}: "
        f"ok={row['ok']} oom={row['oom']} "
        f"torch_peak={row['torch_peak_allocated_gb']}GB step={row['mean_step_s']}s"
    )


def _choose_target_row(rows: list[dict[str, Any]], *, target_gb: float) -> dict[str, Any] | None:
    ok_rows = [r for r in rows if r.get("ok") is True and not r.get("oom")]
    if not ok_rows:
        return None
    return min(
        ok_rows,
        key=lambda r: (
            abs(_row_float(r, "torch_peak_allocated_gb") - target_gb),
            -int(r["batch_size"]),
        ),
    )


def _parse_worker_stdout(stdout: str) -> dict[str, Any] | None:
    for raw_line in reversed(stdout.splitlines()):
        stripped = raw_line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            continue
    return None


def _run_worker(args: argparse.Namespace) -> dict[str, Any]:  # noqa: PLR0915
    row = _empty_row(model=args.model, batch_size=args.batch_size)
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        texts = _load_probe_texts(args.texts_dir)
        model = _build_probe_model(args.model, args.batch_size, args.seed)
        device = torch.device("cuda")
        row["device_name"] = torch.cuda.get_device_name(0)
        row["cuda_version"] = torch.version.cuda

        monitor = _SmiMonitor()
        setup_start = time.perf_counter()
        if args.model in LLM_ENCODE_MODELS:
            # Frozen 7-8B encode probe: no optimiser/backward; the candidate
            # "batch" is encode_batch_size, run under torch.inference_mode.
            step_fn, row["text_path"] = _prepare_llm_encode_step(
                model, texts, device, args.batch_size
            )
        elif args.model in CHUNK_MODELS:
            step_fn, row["text_path"], row["chunk_count"] = _prepare_chunk_step(
                model, texts, device
            )
        elif args.model in FUSION_MODELS:
            step_fn, row["text_path"] = _prepare_fusion_step(model, texts, device)
        else:
            # S1-family and C4 Longformer: train step (encoder + head + backward).
            step_fn, row["text_path"] = _prepare_s1_step(model, texts, device)
        row["setup_s"] = round(time.perf_counter() - setup_start, 4)

        monitor.start()
        for _ in range(args.warmup_steps):
            step_fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        step_times = []
        timed_start = time.perf_counter()
        for _ in range(args.steps):
            t0 = time.perf_counter()
            step_fn()
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)
        row["timed_step_s_total"] = round(time.perf_counter() - timed_start, 4)
        monitor.stop()

        row["ok"] = True
        row["oom"] = False
        row["steps"] = args.steps
        row["warmup_steps"] = args.warmup_steps
        row["mean_step_s"] = round(mean(step_times), 6)
        row["median_step_s"] = round(median(step_times), 6)
        row["p90_step_s"] = round(_percentile(step_times, 0.9), 6)
        row["torch_peak_allocated_gb"] = round(_gb(torch.cuda.max_memory_allocated()), 3)
        row["torch_peak_reserved_gb"] = round(_gb(torch.cuda.max_memory_reserved()), 3)
        row["nvidia_smi_peak_gb"] = round(_gb(monitor.peak_mb * 1024 * 1024), 3)
    except Exception as exc:
        row["ok"] = False
        row["oom"] = _is_oom(exc)
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["stderr_tail"] = "\n".join(traceback.format_exc().splitlines()[-20:])
        try:
            import torch

            if torch.cuda.is_available():
                row["torch_peak_allocated_gb"] = round(_gb(torch.cuda.max_memory_allocated()), 3)
                row["torch_peak_reserved_gb"] = round(_gb(torch.cuda.max_memory_reserved()), 3)
                torch.cuda.empty_cache()
        except Exception:
            pass
    return row


def _build_probe_model(model_id: str, batch_size: int, seed: int):
    from train import _build_model, _load_yaml

    cfg = _load_yaml(REPO_ROOT / "configs" / "models" / f"{model_id}.yaml")
    cfg.setdefault("training", {})
    cfg["training"]["max_epochs"] = 1
    cfg["training"]["grad_accumulation_steps"] = 1
    cfg["training"]["checkpoint"] = False
    if model_id in LLM_ENCODE_MODELS:
        # For the frozen-LLM probe the VRAM-driving knob is the encode forward
        # batch (llm.encode_batch_size), NOT training.batch_size (the head batch
        # over cached embeddings, which is trivial). Drive the probed candidate
        # through encode_batch_size and leave the head batch as configured.
        cfg.setdefault("llm", {})
        cfg["llm"]["encode_batch_size"] = batch_size
        cfg["llm"]["cache_embeddings"] = False  # never touch/write the shared cache
    else:
        cfg["training"]["batch_size"] = batch_size
    with TemporaryDirectory(prefix="sp500vol-probe-") as tmp:
        return _build_model(
            model_id,
            cfg,
            dataset="probe",
            run_dir=Path(tmp),
            seed=seed,
        )


def _prepare_s1_step(model, texts: list[tuple[str, str]], device) -> tuple[Callable[[], None], str]:
    import torch
    from torch import nn

    encoder, head = model._build_modules()
    optimiser = torch.optim.AdamW(
        [*encoder.parameters(), *head.parameters()], lr=model.lr, weight_decay=model.weight_decay
    )
    loss_fn = nn.MSELoss()
    batch_texts = _batch_texts(texts, model.batch_size, max_length=encoder.cfg.max_length)
    targets = _target_tensor(model, model.batch_size, device)

    def step() -> None:
        optimiser.zero_grad(set_to_none=True)
        tok = encoder.tokenize(batch_texts)
        emb = encoder(tok["input_ids"].to(device), tok["attention_mask"].to(device))
        loss = loss_fn(head(emb), targets)
        loss.backward()
        optimiser.step()

    return step, texts[0][0]


def _prepare_fusion_step(
    model, texts: list[tuple[str, str]], device
) -> tuple[Callable[[], None], str]:
    import torch
    from torch import nn

    encoder, fusion, head = model._build_modules()
    optimiser = torch.optim.AdamW(
        [*encoder.parameters(), *fusion.parameters(), *head.parameters()],
        lr=model.lr,
        weight_decay=model.weight_decay,
    )
    loss_fn = nn.MSELoss()
    batch_texts = _batch_texts(texts, model.batch_size, max_length=encoder.cfg.max_length)
    price = torch.zeros((model.batch_size, 3), dtype=torch.float32, device=device)
    targets = _target_tensor(model, model.batch_size, device)

    def step() -> None:
        optimiser.zero_grad(set_to_none=True)
        tok = encoder.tokenize(batch_texts)
        text_emb = encoder(tok["input_ids"].to(device), tok["attention_mask"].to(device))
        loss = loss_fn(head(fusion(price, text_emb)), targets)
        loss.backward()
        optimiser.step()

    return step, texts[0][0]


def _prepare_chunk_step(
    model, texts: list[tuple[str, str]], device
) -> tuple[Callable[[], None], str, int]:
    import torch
    from torch import nn

    encoder, head = model._build_modules()
    params = [*encoder.parameters(), *head.parameters()]
    if hasattr(model, "_current_pool"):
        params.extend(model._current_pool.parameters())
    if hasattr(model, "_current_chunk_encoder"):
        params.extend(model._current_chunk_encoder.parameters())
    optimiser = torch.optim.AdamW(params, lr=model.lr, weight_decay=model.weight_decay)
    loss_fn = nn.MSELoss()
    sample_path, ids, mask = _find_full_chunk_sample(model, encoder, texts)
    chunk_count = int(ids.shape[0])
    ids = ids.unsqueeze(0).repeat(model.batch_size, 1, 1)
    mask = mask.unsqueeze(0).repeat(model.batch_size, 1, 1)
    batch = {
        "input_ids": ids,
        "attention_mask": mask,
        "chunk_counts": torch.full((model.batch_size,), chunk_count, dtype=torch.long),
        "targets": _target_tensor(model, model.batch_size, torch.device("cpu")),
    }

    def step() -> None:
        optimiser.zero_grad(set_to_none=True)
        emb = model._forward_chunked(batch, encoder)
        loss = loss_fn(head(emb), batch["targets"].to(device))
        loss.backward()
        optimiser.step()

    return step, sample_path, chunk_count


def _find_full_chunk_sample(model, encoder, texts: list[tuple[str, str]]):
    for path, text in texts:
        tok = encoder.tokenizer(
            text,
            max_length=encoder.cfg.max_length,
            truncation=True,
            return_overflowing_tokens=True,
            stride=model.chunk_stride,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
        )
        ids = tok["input_ids"][: model.max_chunks]
        mask = tok["attention_mask"][: model.max_chunks]
        if ids.shape[0] >= model.max_chunks:
            return path, ids.contiguous(), mask.contiguous()
    raise RuntimeError(f"no text in probe set filled max_chunks={model.max_chunks}")


def _target_tensor(model, batch_size: int, device):
    import torch

    value = math.log(0.20) if model.log_target else 0.20
    return torch.full((batch_size,), value, dtype=torch.float32, device=device)


# Generous chars/token upper bound for English 10-K text (real ratio is ~4-5
# chars/token; 8 leaves margin so truncation never starves the tokeniser of
# tokens). Used to size the char window we hand the tokeniser per max_length.
_CHARS_PER_TOKEN = 8


def _batch_texts(texts: list[tuple[str, str]], batch_size: int, *, max_length: int) -> list[str]:
    char_budget = _char_budget(max_length)
    return [texts[i % len(texts)][1][:char_budget] for i in range(batch_size)]


def _char_budget(max_length: int) -> int:
    # The encoder truncates to ``max_length`` TOKENS, so memory is set by tokens,
    # not characters. We pre-slice text to a char window that is guaranteed to
    # supply >= max_length tokens, so the tokeniser always fills the sequence and
    # the probe measures memory at the real token length. For 512-token S1 models
    # this keeps the historical ~4k-char window; for the 4096-token Longformer it
    # widens to ~32k chars so the encoder actually sees a full 4096-token sequence
    # (the old fixed 4096-char cap yielded only ~1k tokens, understating VRAM).
    return max(S1_PROBE_CHARS, max_length * _CHARS_PER_TOKEN)


def _prepare_llm_encode_step(
    model, texts: list[tuple[str, str]], device, batch_size: int
) -> tuple[Callable[[], None], str]:
    """Encode-only step for the frozen 7-8B LLM probes (C5_*/D3_*).

    The VRAM-relevant op for these models is FrozenLLMEncoder.encode: a forward
    pass over ``encode_batch_size`` texts at ``max_length`` tokens, run under
    ``torch.inference_mode`` (no optimiser, no backward — the head trains later
    on cheap cached embeddings). We exercise that exact path so the peak-memory
    capture reflects the real encode footprint. ``batch_size`` here is the probed
    encode_batch_size candidate.

    NOTE: D3_<x> shares the SAME frozen encoder + instruction as C5_<x> (same
    pretrained / max_length / encode_batch_size), so the D3 encode footprint is
    identical to the matching C5; both are probed for completeness.
    """
    encoder = model._get_encoder()  # builds + loads the 7-8B model onto CUDA
    char_budget = _char_budget(encoder.max_length)
    batch_texts = [texts[i % len(texts)][1][:char_budget] for i in range(batch_size)]

    def step() -> None:
        # encoder.encode is itself @torch.inference_mode; one call == one encode
        # forward at this candidate batch (encode handles its own batching, and
        # len(batch_texts) == batch_size so it is a single forward).
        encoder.encode(batch_texts, batch_size=batch_size)

    return step, texts[0][0]


def _load_probe_texts(texts_dir: Path) -> list[tuple[str, str]]:
    paths = sorted(Path(texts_dir).glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"no .txt files under {texts_dir}")
    loaded = []
    for path in paths:
        loaded.append((str(path), path.read_text(encoding="utf-8", errors="ignore")))
    return loaded


class _SmiMonitor:
    def __init__(self, interval_s: float = 0.2) -> None:
        self.interval_s = interval_s
        self.peak_mb = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                proc = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=2,
                )
                value = int(proc.stdout.splitlines()[0].strip())
                self.peak_mb = max(self.peak_mb, value)
            except Exception:
                pass
            self._stop.wait(self.interval_s)


def _empty_row(*, model: str, batch_size: int) -> dict[str, Any]:
    return {field: "" for field in RAW_FIELDS} | {
        "model": model,
        "batch_size": batch_size,
        "ok": False,
        "oom": False,
    }


def _summary_rows(rows: list[dict[str, Any]], headroom_gb: float) -> list[dict[str, Any]]:
    out = []
    target_gb = _first_target_gb(rows)
    for model in GRID:
        model_rows = [r for r in rows if r["model"] == model]
        final_rows = [r for r in model_rows if r.get("phase") == "final"]
        candidate_rows = final_rows or model_rows
        ok_rows = [r for r in candidate_rows if r.get("ok") is True]
        if target_gb is None:
            ok_rows = [
                r for r in ok_rows if _row_float(r, "torch_peak_allocated_gb") <= headroom_gb
            ]
            best = (
                max(ok_rows, key=lambda r: _row_float(r, "torch_peak_allocated_gb"))
                if ok_rows
                else None
            )
        else:
            best = _choose_target_row(ok_rows, target_gb=target_gb)
        out.append(
            {
                "model": model,
                # device-generic name; for C5_*/D3_* this is the safe ENCODE batch
                # (encode_batch_size), for the rest the safe TRAIN batch.
                "safe_batch_40g": "" if best is None else best["batch_size"],
                "batch_kind": _batch_kind(model),
                "peak_vram_gb": "" if best is None else best["torch_peak_allocated_gb"],
                "nvidia_smi_peak_gb": "" if best is None else best["nvidia_smi_peak_gb"],
                "step_s": "" if best is None else best["mean_step_s"],
                "device_name": "" if best is None else best.get("device_name", ""),
                "tested_batches": " ".join(str(r["batch_size"]) for r in model_rows),
                "first_oom_batch": _first_oom(model_rows),
            }
        )
    return out


def _batch_kind(model: str) -> str:
    # The "batch" column means different things per model family; label it so the
    # summary is unambiguous (encode forward batch vs training step batch).
    return "encode" if model in LLM_ENCODE_MODELS else "train"


def _summary_fields() -> list[str]:
    return [
        "model",
        "safe_batch_40g",
        "batch_kind",
        "peak_vram_gb",
        "nvidia_smi_peak_gb",
        "step_s",
        "device_name",
        "tested_batches",
        "first_oom_batch",
    ]


def _write_summary_md(path: Path, rows: list[dict[str, Any]], headroom_gb: float) -> None:
    summary = _summary_rows(rows, headroom_gb)
    device = next((r["device_name"] for r in summary if r.get("device_name")), "CUDA device")
    lines = [
        "# A100-40G Batch Probe",
        "",
        f"Device: {device}.",
        f"Headroom rule: largest passing batch with torch peak allocated <= {headroom_gb:.1f} GB.",
        "Batch kind: `encode` = frozen-LLM encode_batch_size (C5_*/D3_*); "
        "`train` = training step batch.",
        "",
        "| model | safe_batch@40G | kind | torch_peak_GB | nvidia_smi_peak_GB | "
        "step_s | first_oom |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {model} | {safe_batch_40g} | {batch_kind} | {peak_vram_gb} | "
            "{nvidia_smi_peak_gb} | {step_s} | {first_oom_batch} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _first_oom(rows: list[dict[str, Any]]) -> str:
    ooms = [int(r["batch_size"]) for r in rows if r.get("oom") is True]
    return "" if not ooms else str(min(ooms))


def _row_float(row: dict[str, Any], field: str) -> float:
    value = row.get(field)
    if value in ("", None):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _first_target_gb(rows: list[dict[str, Any]]) -> float | None:
    for row in rows:
        value = row.get("target_gb")
        if value not in ("", None):
            return float(value)
    return None


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def _gb(num_bytes: int) -> float:
    return num_bytes / 1024**3


def _is_oom(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in text or "cuda error: out of memory" in text


if __name__ == "__main__":
    raise SystemExit(main())
