"""Train a single model by model_id.

Usage:
    python scripts/train.py --model A2_har_rv
    python scripts/train.py --model C2_finbert_s3 --seed 2026
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sp500vol.features.lm_dictionary import LoughranMcDonaldDictionary
from sp500vol.models.base import VolatilityForecaster
from sp500vol.models.classical_text import BoWRidge, LMFeatures, LMLinear, TfidfRidge
from sp500vol.utils import (
    CostTracker,
    configure_logging,
    get_logger,
    seed_everything,
    write_env_snapshot,
)
from sp500vol.utils.paths import data_path

MIN_TRAIN_ROWS = 3
_TRAIN_DATASET_COLUMNS = [
    "cik",
    "ticker",
    "form",
    "item_subtype",
    "filing_time_utc",
    "effective_trading_day",
    "horizon_days",
    "label_realised_vol",
    "text_path",
    "metadata_path",
    "feature_window_end",
    "feature_return_1d",
    "feature_rv_5d",
    "feature_rv_22d",
    "accession",
]

_SUPPORTED_MODELS = {
    "A1_hv",
    "A2_har_rv",
    "A3_garch",
    "A4_egarch",
    "A5_arima",
    "B1_bow_ridge",
    "B2_tfidf_ridge",
    "B3_lm_linear",
    "B4_lm_features",
    "C1_bert_s1",
    "C1_bert_s2",
    "C2_finbert_s1",
    "C2_finbert_s2",
    "C2_finbert_s3",
    "C2_finbert_s4",
    "C3_roberta_s1",
    "C4_longformer",
    "C5_qwen3",
    "C5_gteqwen2",
    "C5_e5mistral",
    "D1_concat_mlp",
    "D2_gated_fusion",
    "D3_qwen3",
    "D3_gteqwen2",
    "D3_e5mistral",
}

# Block A models that need market_returns to fit/predict
_NEEDS_MARKET_RETURNS = {"A3_garch", "A4_egarch", "A5_arima"}


def main() -> int:  # noqa: PLR0915  (CLI entry: argparse + full train/predict/save flow)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model_id (e.g. C2_finbert_s3)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--dataset",
        default="full",
        help="Dataset name under data/processed/{dataset}/ (e.g. sample, dry_run_medium, full)",
    )
    parser.add_argument(
        "--disclosure", default="combined", choices=["long_form", "event_driven", "combined"]
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Pipeline smoke test: tiny per-(split,horizon) subset + capped epochs, but "
        "REAL batch size / sequence length so it still exercises OOM and the full code "
        "path (fit -> predict -> save -> reload). Writes to a *_smoke run dir. This is a "
        "crash check, NOT a convergence check.",
    )
    parser.add_argument(
        "--smoke-rows", type=int, default=64, help="Smoke: rows kept per (split, horizon)."
    )
    parser.add_argument("--smoke-epochs", type=int, default=2, help="Smoke: max_epochs cap.")
    args = parser.parse_args()

    try:  # load .env so HOURLY_RATE_USD reaches CostTracker (cost stays 0 otherwise)
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    configure_logging("INFO")
    log = get_logger("train")

    seed_everything(args.seed)
    run_id = f"{args.model}_{args.dataset}_{args.disclosure}_seed{args.seed}"
    if args.smoke:
        run_id += "_smoke"
    run_dir = REPO_ROOT / "results" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_env_snapshot(run_dir)
    tracker = CostTracker(run_dir=run_dir)

    log.info(
        "Training start",
        model=args.model,
        seed=args.seed,
        dataset=args.dataset,
        disclosure=args.disclosure,
        run_dir=str(run_dir),
    )

    with tracker.timed("training"):
        if args.model not in _SUPPORTED_MODELS:
            raise NotImplementedError(
                f"Model {args.model} not implemented. Supported: {sorted(_SUPPORTED_MODELS)}"
            )

        model_cfg = _load_yaml(REPO_ROOT / "configs" / "models" / f"{args.model}.yaml")
        if args.smoke:
            _cap_smoke_epochs(model_cfg, args.smoke_epochs)
        data = _load_dataset(args.dataset)
        data = _filter_disclosure(data, args.disclosure)
        data = _assign_splits(data, args.dataset)
        data = _drop_invalid_rows(data)
        if args.smoke:
            data = _smoke_subset(data, args.smoke_rows)
            log.info("smoke subset", rows=len(data), per_group=args.smoke_rows)
            _warn_smoke_coverage(data, log)
        _validate_trainable(data)

        model = _build_model(
            args.model,
            model_cfg,
            dataset=args.dataset,
            run_dir=run_dir,
            seed=args.seed,
        )
        train_rows = data[data["split"] == "train"].copy()
        val_rows = data[data["split"] == "val"].copy()
        model.fit(
            train_rows,
            train_rows["label_realised_vol"].to_numpy(),
            X_val=val_rows,
            y_val=val_rows["label_realised_vol"].to_numpy() if not val_rows.empty else None,
        )

        # Write val_curves FIRST — before the predictions/metrics that is_done checks —
        # so a crash mid-output never leaves a predictions-present but curve-missing run
        # that would read as a permanently un-rerunnable SUSPECT.
        val_curves = getattr(model, "val_curves_", None)
        if val_curves:
            (run_dir / "val_curves.json").write_text(
                json.dumps({str(k): v for k, v in val_curves.items()}, indent=2, default=str),
                encoding="utf-8",
            )

        predictions = data.copy()
        predictions["prediction_realised_vol"] = model.predict(predictions)
        predictions["run_id"] = run_id
        predictions["model_id"] = args.model
        predictions["dataset"] = args.dataset
        predictions["seed"] = args.seed
        predictions["disclosure_subset"] = args.disclosure
        predictions["feature_rv_1d"] = _feature_rv_1d(predictions)

        prediction_cols = _prediction_columns(predictions)
        predictions[prediction_cols].to_parquet(run_dir / "predictions.parquet", index=False)
        model.save(run_dir / "model.pkl")

        metrics = _metrics_by_group(predictions)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "model": args.model,
                    "dataset": args.dataset,
                    "disclosure": args.disclosure,
                    "seed": args.seed,
                    "model_config": model_cfg,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        log.info("Training outputs written", predictions=len(predictions), metrics=len(metrics))

    tracker.write_summary()
    log.info("Training done", total_cost_usd=tracker.total_cost_usd)
    return 0


def _cap_smoke_epochs(model_cfg: dict, smoke_epochs: int) -> None:
    """Smoke: cap max_epochs AND es_patience in-place so the smoke also exercises the
    early-stopping / restore-best branch (the riskiest new path). warmup_ratio is left
    as-is, so the short smoke runs near peak lr — fine for a crash check. Neural/fusion
    training configs only."""
    tr = model_cfg.get("training")
    if isinstance(tr, dict):
        tr["max_epochs"] = min(smoke_epochs, int(tr.get("max_epochs", smoke_epochs)))
        if "es_patience" in tr:
            tr["es_patience"] = min(1, int(tr["es_patience"]))


def _smoke_subset(data: pd.DataFrame, rows_per_group: int) -> pd.DataFrame:
    """Smoke: tiny per-(split, horizon) head — keeps up to N rows for each
    (split, horizon) group that EXISTS (a group absent in the data stays absent)."""
    return (
        data.groupby(["split", "horizon_days"], group_keys=False)
        .head(rows_per_group)
        .reset_index(drop=True)
    )


def _warn_smoke_coverage(data: pd.DataFrame, log) -> None:
    """Smoke: warn if any horizon lacks train OR val rows — val being empty silently
    disables early stopping, so the smoke would exercise a different path than the
    real run (and the missing-val curve would later read as SUSPECT in the gate)."""
    for horizon, grp in data.groupby("horizon_days"):
        splits = set(grp["split"].unique())
        if not {"train", "val"}.issubset(splits):
            log.warning("smoke horizon missing a split", horizon=int(horizon), have=sorted(splits))


def _build_model(
    model_id: str,
    cfg: dict,
    *,
    dataset: str,
    run_dir: Path,
    seed: int,
) -> VolatilityForecaster:
    """Factory: instantiate the right model class from its config."""
    training_cfg = cfg.get("training", {})
    # ridge_alpha None → BoWRidge / LMLinear use RidgeCV (auto-pick alpha)
    raw_alpha = training_cfg.get("ridge_alpha")
    ridge_alpha = None if raw_alpha is None else float(raw_alpha)
    log_target = bool(training_cfg.get("log_target", True))

    if model_id == "A1_hv":
        from sp500vol.models.price import NaiveHV

        return NaiveHV()
    if model_id == "A2_har_rv":
        from sp500vol.models.price import HARRV

        smearing = bool(training_cfg.get("smearing", True))  # Duan retransformation, on by default
        return HARRV(log_transform=log_target, smearing=smearing)
    if model_id in {"A3_garch", "A4_egarch"}:
        from sp500vol.models.price import EGARCH, GARCH

        garch_cfg = cfg.get("garch", {})
        market = _load_market_returns(dataset)
        cls = GARCH if model_id == "A3_garch" else EGARCH
        return cls(
            market_returns_df=market,
            p=int(garch_cfg.get("p", 1)),
            q=int(garch_cfg.get("q", 1)),
        )
    if model_id == "A5_arima":
        from sp500vol.models.price import ARIMAVol

        arima_cfg = cfg.get("arima", {})
        market = _load_market_returns(dataset)
        order = tuple(int(v) for v in arima_cfg.get("order", [1, 0, 1]))
        return ARIMAVol(market_returns_df=market, order=order)
    if model_id == "B1_bow_ridge":
        vec = cfg.get("vectoriser", {})
        return BoWRidge(
            max_features=int(vec.get("max_features", 5000)),
            ridge_alpha=ridge_alpha,
            log_target=log_target,
        )
    if model_id == "B2_tfidf_ridge":
        vec = cfg.get("vectoriser", {})
        return TfidfRidge(
            max_features=int(vec.get("max_features", 5000)),
            ridge_alpha=ridge_alpha,
            log_target=log_target,
        )
    if model_id in {"B3_lm_linear", "B4_lm_features"}:
        dictionary = _load_lm_dictionary(cfg.get("dictionary", {}))
        cls = LMLinear if model_id == "B3_lm_linear" else LMFeatures
        return cls(dictionary=dictionary, ridge_alpha=ridge_alpha, log_target=log_target)
    if model_id.startswith(("C5_", "D3_")):
        return _build_llm_model(
            model_id,
            cfg,
            training_cfg,
            log_target=log_target,
            run_dir=run_dir,
            seed=seed,
        )
    if model_id.startswith(("C1_bert", "C2_finbert", "C3_roberta", "C4_longformer")):
        return _build_neural_model(
            model_id,
            cfg,
            training_cfg,
            log_target=log_target,
            run_dir=run_dir,
            seed=seed,
        )
    if model_id in {"D1_concat_mlp", "D2_gated_fusion"}:
        return _build_fusion_model(
            model_id,
            cfg,
            training_cfg,
            log_target=log_target,
            run_dir=run_dir,
            seed=seed,
        )
    raise NotImplementedError(f"no factory for model_id={model_id!r}")


def _build_neural_model(
    model_id: str,
    cfg: dict,
    training_cfg: dict,
    *,
    log_target: bool,
    run_dir: Path,
    seed: int,
) -> VolatilityForecaster:
    neural = _load_neural_components(model_id)
    encoder_cfg = cfg.get("encoder", {})
    head_cfg = cfg.get("head", {})
    chunk_cfg = cfg.get("chunking", {})
    attn_cfg = cfg.get("attention", {})
    hierarchy_cfg = cfg.get("hierarchy", {})
    cls_map = {
        "C1_bert_s1": neural["BertS1"],
        "C2_finbert_s1": neural["FinBertS1"],
        "C1_bert_s2": neural["BertS2"],
        "C2_finbert_s2": neural["FinBertS2"],
        "C1_bert_s3": neural["BertS3"],
        "C2_finbert_s3": neural["FinBertS3"],
        "C2_finbert_s4": neural["FinBertS4"],
        "C3_roberta_s1": neural["RobertaS1"],
        "C4_longformer": neural["LongformerModel"],
    }
    if model_id not in cls_map:
        raise NotImplementedError(f"no neural factory for model_id={model_id!r}")
    checkpoint_enabled = bool(training_cfg.get("checkpoint", True))
    common_kwargs = {
        "pretrained": str(encoder_cfg.get("pretrained", _default_pretrained(model_id))),
        "max_length": int(encoder_cfg.get("max_length", 512)),
        "hidden_dim": int(head_cfg.get("hidden_dim", 128)),
        "dropout": float(head_cfg.get("dropout", 0.1)),
        "lr": float(training_cfg.get("lr", 2.0e-5)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.01)),
        "batch_size": int(training_cfg.get("batch_size", 8 if model_id.endswith("s1") else 2)),
        "max_epochs": int(training_cfg.get("max_epochs", 3)),
        "early_stopping": bool(training_cfg.get("early_stopping", True)),
        "es_patience": int(training_cfg.get("es_patience", 1)),
        "es_min_delta": float(training_cfg.get("es_min_delta", 0.0)),
        "mixed_precision": str(training_cfg.get("mixed_precision", "no")),
        "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.0)),
        "grad_accumulation_steps": int(training_cfg.get("grad_accumulation_steps", 1)),
        "objective": str(training_cfg.get("objective", "mse")),
        "freeze_mode": str(training_cfg.get("freeze_mode", "none")),
        "head_lr_mult": float(training_cfg.get("head_lr_mult", 1.0)),
        "log_target": log_target,
        "pretokenize": bool(training_cfg.get("pretokenize", False)),
        "tokenization_batch_size": int(training_cfg.get("tokenization_batch_size", 128)),
        "tokenizer_threads": _optional_int(training_cfg.get("tokenizer_threads")),
        "dataloader_num_workers": int(training_cfg.get("dataloader_num_workers", 0)),
        "dataloader_persistent_workers": bool(
            training_cfg.get("dataloader_persistent_workers", False)
        ),
        "dataloader_pin_memory": training_cfg.get("dataloader_pin_memory"),
        "dataloader_prefetch_factor": _optional_int(
            training_cfg.get("dataloader_prefetch_factor")
        ),
        "checkpoint": checkpoint_enabled,
        "checkpoint_dir": run_dir / "checkpoints" if checkpoint_enabled else None,
        "seed": seed,
        "strategy": str(cfg.get("strategy", _default_strategy(model_id))),
    }
    if model_id.endswith(("s2", "s3", "s4")):
        common_kwargs["chunk_stride"] = int(chunk_cfg.get("chunk_stride", 256))
        common_kwargs["max_chunks"] = int(chunk_cfg.get("max_chunks", 16))
    if model_id.endswith("s3"):
        common_kwargs["attn_dim"] = int(attn_cfg.get("attn_dim", 128))
    if model_id.endswith("s4"):
        common_kwargs["chunk_num_heads"] = int(hierarchy_cfg.get("chunk_num_heads", 8))
        common_kwargs["chunk_encoder_layers"] = int(hierarchy_cfg.get("chunk_encoder_layers", 1))
        if hierarchy_cfg.get("chunk_ff_dim") is not None:
            common_kwargs["chunk_ff_dim"] = int(hierarchy_cfg["chunk_ff_dim"])
    return cls_map[model_id](**common_kwargs)


def _build_fusion_model(
    model_id: str,
    cfg: dict,
    training_cfg: dict,
    *,
    log_target: bool,
    run_dir: Path,
    seed: int,
) -> VolatilityForecaster:
    fusion_cls = _load_fusion_component(model_id)
    encoder_cfg = cfg.get("encoder", {})
    head_cfg = cfg.get("head", {})
    fusion_cfg = cfg.get("fusion", {})
    checkpoint_enabled = bool(training_cfg.get("checkpoint", True))
    return fusion_cls(
        pretrained=str(encoder_cfg.get("pretrained", "ProsusAI/finbert")),
        max_length=int(encoder_cfg.get("max_length", 512)),
        proj_dim=int(fusion_cfg.get("proj_dim", 128)),
        hidden_dim=int(head_cfg.get("hidden_dim", 128)),
        dropout=float(head_cfg.get("dropout", 0.1)),
        lr=float(training_cfg.get("lr", 2.0e-5)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        batch_size=int(training_cfg.get("batch_size", 8)),
        max_epochs=int(training_cfg.get("max_epochs", 3)),
        early_stopping=bool(training_cfg.get("early_stopping", True)),
        es_patience=int(training_cfg.get("es_patience", 1)),
        es_min_delta=float(training_cfg.get("es_min_delta", 0.0)),
        mixed_precision=str(training_cfg.get("mixed_precision", "no")),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.0)),
        log_target=log_target,
        checkpoint=checkpoint_enabled,
        checkpoint_dir=run_dir / "checkpoints" if checkpoint_enabled else None,
        seed=seed,
        strategy=str(cfg.get("strategy", model_id)),
        pretokenize=bool(training_cfg.get("pretokenize", False)),
        tokenization_batch_size=int(training_cfg.get("tokenization_batch_size", 512)),
        tokenizer_threads=_optional_int(training_cfg.get("tokenizer_threads")),
        dataloader_num_workers=int(training_cfg.get("dataloader_num_workers", 0)),
        dataloader_persistent_workers=bool(
            training_cfg.get("dataloader_persistent_workers", False)
        ),
        dataloader_pin_memory=training_cfg.get("dataloader_pin_memory"),
        dataloader_prefetch_factor=_optional_int(training_cfg.get("dataloader_prefetch_factor")),
    )


def _build_llm_model(
    model_id: str,
    cfg: dict,
    training_cfg: dict,
    *,
    log_target: bool,
    run_dir: Path,
    seed: int,
) -> VolatilityForecaster:
    """Factory for the frozen decoder-LLM probe (C5) and its price fusion (D3)."""
    try:
        from sp500vol.models.fusion import D3LLMFusion
        from sp500vol.models.neural_text import C5LLMProbe
    except ImportError as exc:  # torch / transformers not installed in lint-only env
        raise RuntimeError(f"{model_id} requires torch + transformers. Run `uv sync`.") from exc

    encoder_cfg = cfg.get("encoder", {})
    head_cfg = cfg.get("head", {})
    fusion_cfg = cfg.get("fusion", {})
    llm_cfg = cfg.get("llm", {})
    checkpoint_enabled = bool(training_cfg.get("checkpoint", True))
    common_kwargs = {
        "pretrained": str(encoder_cfg.get("pretrained", "Alibaba-NLP/gte-Qwen2-7B-instruct")),
        "max_length": int(encoder_cfg.get("max_length", 4096)),
        "hidden_dim": int(head_cfg.get("hidden_dim", 128)),
        "dropout": float(head_cfg.get("dropout", 0.1)),
        "lr": float(training_cfg.get("lr", 1.0e-4)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.01)),
        "batch_size": int(training_cfg.get("batch_size", 64)),
        "max_epochs": int(training_cfg.get("max_epochs", 20)),
        "early_stopping": bool(training_cfg.get("early_stopping", True)),
        "es_patience": int(training_cfg.get("es_patience", 3)),
        "es_min_delta": float(training_cfg.get("es_min_delta", 0.0)),
        "mixed_precision": str(training_cfg.get("mixed_precision", "no")),
        "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.0)),
        "log_target": log_target,
        "instruction": llm_cfg.get("instruction"),
        "normalize_emb": bool(llm_cfg.get("normalize_emb", True)),
        "encode_batch_size": int(llm_cfg.get("encode_batch_size", 8)),
        "cache_embeddings": bool(llm_cfg.get("cache_embeddings", True)),
        "tokenizer_threads": _optional_int(training_cfg.get("tokenizer_threads")),
        "checkpoint": checkpoint_enabled,
        "checkpoint_dir": run_dir / "checkpoints" if checkpoint_enabled else None,
        "seed": seed,
        "strategy": str(cfg.get("strategy", model_id)),
    }
    if model_id.startswith("C5_"):
        return C5LLMProbe(**common_kwargs)
    if model_id.startswith("D3_"):
        return D3LLMFusion(proj_dim=int(fusion_cfg.get("proj_dim", 128)), **common_kwargs)
    raise NotImplementedError(f"no llm factory for model_id={model_id!r}")


def _load_neural_components(model_id: str) -> dict[str, type]:
    try:
        from sp500vol.models.neural_text import (
            BertS1,
            BertS2,
            BertS3,
            FinBertS1,
            FinBertS2,
            FinBertS3,
            FinBertS4,
            LongformerModel,
            RobertaS1,
        )
    except ImportError as exc:  # torch / transformers not installed in lint-only env
        raise RuntimeError(
            f"{model_id} requires torch + transformers. Run `uv sync` to install all deps."
        ) from exc
    return {
        "BertS1": BertS1,
        "BertS2": BertS2,
        "BertS3": BertS3,
        "FinBertS1": FinBertS1,
        "FinBertS2": FinBertS2,
        "FinBertS3": FinBertS3,
        "FinBertS4": FinBertS4,
        "LongformerModel": LongformerModel,
        "RobertaS1": RobertaS1,
    }


def _load_fusion_component(model_id: str) -> type:
    try:
        from sp500vol.models.fusion import ConcatMLP, GatedFusion
    except ImportError as exc:  # torch / transformers not installed in lint-only env
        raise RuntimeError(f"{model_id} requires torch + transformers. Run `uv sync`.") from exc
    return {"D1_concat_mlp": ConcatMLP, "D2_gated_fusion": GatedFusion}[model_id]


def _default_pretrained(model_id: str) -> str:
    if model_id.startswith("C1_bert"):
        return "bert-base-uncased"
    if model_id.startswith("C2_finbert"):
        return "ProsusAI/finbert"
    if model_id.startswith("C3_roberta"):
        return "roberta-base"
    if model_id == "C4_longformer":
        return "allenai/longformer-base-4096"
    raise ValueError(f"no default pretrained checkpoint for model_id={model_id!r}")


def _default_strategy(model_id: str) -> str:
    if model_id == "C4_longformer":
        return "S5"
    if model_id.endswith("_s1"):
        return "S1"
    if model_id.endswith("_s2"):
        return "S2"
    if model_id.endswith("_s3"):
        return "S3"
    if model_id.endswith("_s4"):
        return "S4"
    return model_id


def _optional_int(value) -> int | None:
    if value is None:
        return None
    return int(value)


def _load_market_returns(dataset: str) -> pd.DataFrame:
    """Load market_returns.parquet for the GARCH-family models."""
    path = data_path("processed", dataset, "market_returns.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"Market returns not found at {path}. "
            "Run `scripts/build_dataset.py --config configs/data/{dataset}.yaml` first."
        )
    return pd.read_parquet(path)


def _load_lm_dictionary(dictionary_cfg: dict) -> LoughranMcDonaldDictionary:
    """Load real L-M CSV if present; otherwise fall back to mock (with warning)."""
    log = get_logger("train")
    csv_path = dictionary_cfg.get("csv_path")
    if csv_path:
        path = Path(csv_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.exists():
            return LoughranMcDonaldDictionary.from_csv(path)
        if not dictionary_cfg.get("use_mock_if_missing", False):
            raise FileNotFoundError(
                f"L-M dictionary not found at {path}. Set use_mock_if_missing: true "
                "in the config to fall back to the mock dictionary, or download "
                "from https://sraf.nd.edu/loughranmcdonald-master-dictionary/."
            )
        log.warning("L-M CSV missing; falling back to mock dictionary", csv_path=str(path))
    return LoughranMcDonaldDictionary.mock()


def _load_yaml(path: Path, *, resolve: bool = True) -> dict:
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=resolve)
    if not isinstance(cfg, dict):
        raise ValueError(f"invalid YAML config: {path}")
    cfg.pop("defaults", None)
    return cfg


def _load_dataset(dataset: str) -> pd.DataFrame:
    path = data_path("processed", dataset, "aligned_filings.parquet")
    if not path.exists():
        raise FileNotFoundError(path)
    out = pd.read_parquet(path, columns=_TRAIN_DATASET_COLUMNS)
    out["filing_time_utc"] = pd.to_datetime(out["filing_time_utc"], utc=True)
    out["effective_trading_day"] = pd.to_datetime(out["effective_trading_day"])
    return out.sort_values(["filing_time_utc", "accession", "horizon_days"]).reset_index(drop=True)


def _filter_disclosure(data: pd.DataFrame, disclosure: str) -> pd.DataFrame:
    if disclosure == "combined":
        return data.copy()
    if disclosure == "long_form":
        return data[data["form"].isin(["10-K", "10-Q"])].copy()
    if disclosure == "event_driven":
        return data[data["form"] == "8-K"].copy()
    raise ValueError(f"unsupported disclosure subset: {disclosure}")


def _assign_splits(data: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = data.copy()
    # Any dataset other than "full" uses fractional chronological splits
    # (the base.yaml chronological splits assume 2010-2025 coverage).
    if dataset != "full":
        return _assign_fractional_splits(out)

    base_cfg = _load_yaml(REPO_ROOT / "configs" / "base.yaml", resolve=False)
    splits = base_cfg["splits"]
    # PINNED SPLIT CONVENTION (documented in configs/base.yaml): chronological by
    # the filing's EFFECTIVE TRADING DAY — the normalized ET trading day on which
    # the filing becomes actionable (set by alignment) — DAY-INCLUSIVE on both
    # ends. Using the effective trading day rather than the raw UTC filing
    # timestamp removes tz/midnight boundary drift (a filing late on 2019-12-31 ET
    # no longer falls into the unused gap at the 2019->2020 boundary), so split
    # membership is stable, reproducible, and identical across Blocks A-D.
    split_day = out["effective_trading_day"].dt.normalize()
    out["split"] = "unused"
    for name in ("train", "val", "test"):
        lo = pd.Timestamp(splits[f"{name}_start"]).normalize()
        hi = pd.Timestamp(splits[f"{name}_end"]).normalize()
        out.loc[(split_day >= lo) & (split_day <= hi), "split"] = name
    return out[out["split"] != "unused"].copy()


def _assign_fractional_splits(data: pd.DataFrame) -> pd.DataFrame:
    keys = (
        data[["accession", "filing_time_utc"]]
        .drop_duplicates()
        .sort_values(["filing_time_utc", "accession"])
        .reset_index(drop=True)
    )
    n = len(keys)
    train_cut = max(1, int(np.floor(n * 0.6)))
    val_cut = max(train_cut + 1, int(np.floor(n * 0.8)))
    keys["split"] = "test"
    keys.loc[: train_cut - 1, "split"] = "train"
    keys.loc[train_cut : val_cut - 1, "split"] = "val"
    return data.merge(keys[["accession", "split"]], on="accession", how="left")


def _validate_trainable(data: pd.DataFrame) -> None:
    if data.empty:
        raise ValueError("no rows after disclosure/split filtering")
    split_counts = data.groupby("split").size().to_dict()
    if split_counts.get("train", 0) < MIN_TRAIN_ROWS:
        raise ValueError(f"not enough training rows: {split_counts}")
    if split_counts.get("test", 0) < 1:
        raise ValueError(f"not enough test rows: {split_counts}")


def _drop_invalid_rows(data: pd.DataFrame) -> pd.DataFrame:
    required = [
        "label_realised_vol",
        "feature_return_1d",
        "feature_rv_5d",
        "feature_rv_22d",
    ]
    cols = [col for col in required if col in data.columns]
    if not cols:
        return data.copy()
    finite = np.isfinite(data[cols].to_numpy(dtype=float)).all(axis=1)
    return data.loc[finite].copy()


def _feature_rv_1d(data: pd.DataFrame) -> pd.Series:
    if "feature_rv_1d" in data.columns:
        return data["feature_rv_1d"]
    return np.sqrt(252.0) * data["feature_return_1d"].abs()


def _metrics_by_group(predictions: pd.DataFrame) -> list[dict[str, object]]:
    from sp500vol.evaluation.metrics import all_metrics

    rows: list[dict[str, object]] = []
    group_cols = ["split", "disclosure_subset", "horizon_days"]
    for keys, group in predictions.groupby(group_cols, dropna=False):
        y_true = group["label_realised_vol"].to_numpy(dtype=float)
        y_pred = group["prediction_realised_vol"].to_numpy(dtype=float)
        metric_values = all_metrics(y_true, y_pred)
        rows.append(
            {
                "split": keys[0],
                "disclosure_subset": keys[1],
                "horizon_days": int(keys[2]),
                "n": len(group),
                **metric_values,
            }
        )
    return rows


def _prediction_columns(predictions: pd.DataFrame) -> list[str]:
    preferred = [
        "run_id",
        "model_id",
        "dataset",
        "seed",
        "disclosure_subset",
        "split",
        "ticker",
        "form",
        "item_subtype",
        "accession",
        "filing_time_utc",
        "effective_trading_day",
        "horizon_days",
        "label_realised_vol",
        "prediction_realised_vol",
        "feature_rv_1d",
        "feature_rv_5d",
        "feature_rv_22d",
        "text_path",
        "metadata_path",
    ]
    return [col for col in preferred if col in predictions.columns]


if __name__ == "__main__":
    sys.exit(main())
