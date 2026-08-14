"""Two-way (FIRM x DAY) cluster-robust DM-style inference — Cameron-Gelbach-Miller.

P1-b remediation. The day-clustered DM (clustered_dm.py) treats same-day filings as
one cluster but still assumes independence ACROSS firms' filings on different days.
The firm-identity finding (firm_identity_control) proves strong within-firm
dependence of the loss differentials; this module adds the firm dimension.

STATISTIC. The point estimate is the mean loss differential with EQUAL WEIGHT PER
DAY — identical to the day-clustered primary:

    dbar = (1/T) * sum_t dbar_t,   dbar_t = mean_{i in day t} d_i
         = sum_i w_i d_i,          w_i = 1 / (T * n_{t(i)})

VARIANCE (Cameron, Gelbach & Miller 2011 two-way decomposition; Thompson 2011):

    V_2way = V_firm + V_day - V_(firm x day)

with, for a clustering C and scores u_i = w_i * (d_i - dbar):

    V_C = sum_{c in C} ( sum_{i in c} u_i )^2

and the DAY component upgraded to the HAC long-run variance of the daily-mean
differential series (Newey-West, lag = h-1 trading days of genuine label overlap),
divided by T:

    V_day = _hac_variance(dbar_t series, lag=h-1) / T

which at lag 0 reduces EXACTLY to the CGM day component sum_t(sum_{i in t} u_i)^2.
This keeps the serial-correlation treatment identical to the day-clustered primary
(Driscoll-Kraay / Thompson style time component).

Deliberate, documented approximations (both push toward WIDER SEs, i.e.
conservative for any significance claim):
  * the firm-x-day intersection is subtracted at lag 0 only; Thompson's lagged
    own-firm overlap terms are omitted, so V_2way is weakly LARGER than the exact
    Thompson estimator;
  * no Harvey-Leybourne-Newbold small-sample correction is applied (the
    day-clustered primary applies HLN with n = n_days; at n_days ~ 800 the factor
    is ~0.98, i.e. immaterial), and the reference distribution is
    t(min(#firms, #days) - 1) — the CGM convention, heavier-tailed than t(T-1).

Non-PSD guard: the CGM difference form can go non-positive in degenerate cells;
we set V_2way <- max(V_2way, EPS_V) and FLAG the cell (guard_hit) instead of
reporting a spurious statistic.

Import from repo root:
    sys.path.insert(0, "scripts/analysis"); from twoway_dm import dm_test_2way
"""
from __future__ import annotations

import sys
from collections import namedtuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "scripts/analysis")
sys.path.insert(0, "src")
from clustered_dm import _day_index  # noqa: E402
from sp500vol.evaluation.dm_test import _hac_variance  # noqa: E402

__all__ = ["dm_test_2way", "TwoWayResult", "EPS_V"]

EPS_V = 1e-30

TwoWayResult = namedtuple(
    "TwoWayResult",
    "stat p n_firms n_days dbar se_2way V_firm V_day V_cell V_2way guard_hit df",
)


def dm_test_2way(d, firms, days, h):
    """Two-way (firm x day) CGM cluster-robust test of H0: E[d] = 0.

    Args:
        d: per-OBSERVATION loss differential (lossA_i - lossB_i);
           positive mean = A worse.
        firms: per-observation firm key (ticker).
        days: per-observation calendar-day key (effective_trading_day, or
              filing_time_utc date fallback) — same convention as clustered_dm.
        h: label horizon in TRADING DAYS -> HAC lag = h-1 on the daily series.

    Returns:
        TwoWayResult(stat, p, n_firms, n_days, dbar, se_2way,
                     V_firm, V_day, V_cell, V_2way, guard_hit, df)
        Positive stat = first loss series WORSE. p is two-sided
        t(df = min(n_firms, n_days) - 1).
    """
    d = np.asarray(d, dtype=np.float64)
    firm = np.asarray(firms)
    day = _day_index(days)
    if not (len(d) == len(firm) == len(day)):
        raise ValueError("d, firms, days must have equal length")

    df_ = pd.DataFrame({"d": d, "firm": firm, "day": day})
    day_means = df_.groupby("day", sort=True)["d"].mean()
    T = int(len(day_means))
    G = int(df_["firm"].nunique())
    dbar = float(day_means.mean())

    # scores u_i = w_i (d_i - dbar), w_i = 1/(T * n_day(i))
    n_t = df_.groupby("day")["d"].transform("size").to_numpy(dtype=np.float64)
    u = (d - dbar) / (T * n_t)
    df_["u"] = u

    V_firm = float((df_.groupby("firm")["u"].sum() ** 2).sum())
    V_cell = float((df_.groupby(["firm", "day"])["u"].sum() ** 2).sum())
    # HAC day component: long-run variance of the daily series / T
    # (lag 0 term == CGM day component sum_t (sum_{i in t} u_i)^2 exactly).
    V_day = float(_hac_variance(day_means.to_numpy(), lag=max(int(h) - 1, 0)) / T)

    V_2way = V_firm + V_day - V_cell
    guard_hit = bool(V_2way <= 0.0)
    V_2way = max(V_2way, EPS_V)

    se = float(np.sqrt(V_2way))
    stat = dbar / se
    dof = min(G, T) - 1
    p = 2.0 * float(stats.t.sf(abs(stat), df=dof)) if dof > 0 else float("nan")
    return TwoWayResult(float(stat), float(p), G, T, dbar, se,
                        V_firm, V_day, V_cell, float(V_2way), guard_hit, int(dof))
