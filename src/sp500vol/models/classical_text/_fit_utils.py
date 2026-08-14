"""Shared dedup + parallel-CV helpers for Block B classical text models.

Block B trains one ridge per horizon, but a filing's 3 horizon rows share the
same text. These helpers (a) build each filing's representation ONCE by deduping
on accession (3x fewer tokenizations), and (b) run the ridge-alpha cross-
validation across cores. The CV is bit-identical to the prior serial loop: it
uses the same ``KFold(shuffle=True, random_state=0)`` folds and the same
mean-fold-MSE argmin (first-alpha tie-break), only the per-alpha fold fits are
fanned out with joblib.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

CV_FOLDS = 5
RIDGE_SOLVER = "lsqr"
RIDGE_TOL = 1e-4
RIDGE_MAX_ITER = 1000
_EPSILON = 1e-12
# Below this row count the joblib worker-spawn overhead outweighs the CV cost,
# so run serially (also keeps unit tests fast + deterministic).
_PARALLEL_MIN_ROWS = 1000
_DEFAULT_TEXT_N_JOBS = 4
_TEXT_N_JOBS_ENV = "SP500VOL_TEXT_N_JOBS"


def unique_filing_index(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (first_idx, inverse) deduping rows to unique filings.

    Uses ``accession`` (a filing's 3 horizon rows share it); falls back to
    ``text_path``. ``first_idx[j]`` is the original-row index of the j-th unique
    filing; ``inverse[i]`` maps original row i to its unique-filing position, so
    ``per_filing[inverse]`` re-expands a per-filing array back to all rows.
    """
    key_col = "accession" if "accession" in df.columns else "text_path"
    keys = df[key_col].to_numpy().astype(str)
    _, first_idx, inverse = np.unique(keys, return_index=True, return_inverse=True)
    return first_idx, inverse


def build_ridge(alpha: float) -> Ridge:
    return Ridge(alpha=float(alpha), solver=RIDGE_SOLVER, tol=RIDGE_TOL, max_iter=RIDGE_MAX_ITER)


def text_n_jobs() -> int:
    """Return the process cap for classical-text joblib work.

    Full-corpus text matrices are large enough that ``n_jobs=-1`` can multiply
    memory usage past a 16GB laptop's limits. Default to a conservative cap and
    allow explicit override for larger machines.
    """
    raw = os.getenv(_TEXT_N_JOBS_ENV)
    if raw is None or raw.strip() == "":
        return _DEFAULT_TEXT_N_JOBS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_TEXT_N_JOBS_ENV} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{_TEXT_N_JOBS_ENV} must be a positive integer")
    return value


def fit_ridge_cv(
    X, y: np.ndarray, alpha_grid: tuple[float, ...], *, n_jobs: int | None = None
) -> Ridge:
    """Ridge with alpha chosen by parallel 5-fold CV (mean-fold MSE).

    Identical selection to the prior serial loop: same folds, same per-alpha mean
    MSE, first-alpha tie-break; the alpha-level fits run across cores.
    """
    alphas = [float(a) for a in alpha_grid]
    if not alphas:
        raise ValueError("alpha_grid must contain at least one alpha")

    n_splits = min(CV_FOLDS, len(y))
    if n_splits < 2:
        ridge = build_ridge(alphas[0])
        ridge.fit(X, y)
        ridge.alpha_ = alphas[0]
        ridge.cv_mse_ = float("nan")
        return ridge

    splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=0).split(y))

    def _alpha_mean_mse(alpha: float) -> float:
        losses = []
        for train_idx, val_idx in splits:
            ridge = build_ridge(alpha)
            ridge.fit(X[train_idx], y[train_idx])
            pred = ridge.predict(X[val_idx])
            losses.append(float(np.mean((pred - y[val_idx]) ** 2)))
        return float(np.mean(losses))

    requested_jobs = text_n_jobs() if n_jobs is None else int(n_jobs)
    effective_jobs = 1 if len(y) < _PARALLEL_MIN_ROWS else requested_jobs
    losses = Parallel(n_jobs=effective_jobs)(delayed(_alpha_mean_mse)(a) for a in alphas)

    best_i = int(np.argmin(losses))  # argmin returns the first minimum -> first-alpha tie-break
    best_alpha = alphas[best_i]
    ridge = build_ridge(best_alpha)
    ridge.fit(X, y)
    ridge.alpha_ = best_alpha
    ridge.cv_mse_ = float(losses[best_i])
    return ridge


def maybe_log(values: np.ndarray, *, log_target: bool) -> np.ndarray:
    if not log_target:
        return values
    safe = np.where(values >= 0.0, values, np.nan)
    return np.log(safe + _EPSILON)


def maybe_exp(
    values: np.ndarray, *, log_target: bool, min_pred: float, max_pred: float
) -> np.ndarray:
    if not log_target:
        return np.clip(values, min_pred, max_pred)
    clipped = np.clip(values, np.log(min_pred), np.log(max_pred))
    return np.clip(np.exp(clipped), min_pred, max_pred)
