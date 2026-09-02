"""R2 — VaR backtest: the economic value of the text increment for DOWNSIDE risk.

For each (disclosure, model, horizon) we build three h-day volatility forecasts on the
TEST split and turn them into left-tail Value-at-Risk forecasts, then backtest:

  (1) rawHAR = A2_har_rv prediction_realised_vol          (raw, known-miscalibrated HAR)
  (2) fR     = recalibrated price-only forecast           (log-space, fit on val)
  (3) fU     = fR + text                                   (log-space nested, fit on val)

fR and fU come from the M1 combiner fc.log_combo(y_val, fHAR_val, fText_val,
fHAR_test, fText_test), so the text increment is isolated against a PROPERLY CALIBRATED
price baseline (the M1 discipline). All vol forecasts are on the ANNUALIZED RV scale and
are de-annualized to the h-day scale sigma_h = sigma_ann * sqrt(h/252).

Realized h-day LOG return R_h = fwd_logret joined from results/tables/_realized_returns.parquet
on KEY=[ticker,accession,horizon_days]; only ret_match_ok==True rows are kept.

VaR (left tail) at alpha in {0.01, 0.05}:
    VaR_alpha = mu_hat + Phi_inv(alpha) * sigma_h
with mu_hat = pooled mean R_h per horizon (also a zero-mean variant, mu=0).
A violation is R_h < VaR_alpha.

Backtests per forecast:
  * empirical violation rate vs alpha
  * Kupiec (1995) unconditional-coverage LR ~ chi2(1)
  * Christoffersen (1998) independence LR + conditional-coverage LR (= Kupiec + indep) ~ chi2(2)
    Transition counts are built by ordering the violation sequence by filing_time WITHIN
    ticker then concatenating tickers (a within-name ordering; see the overlapping-window
    caveat below).
  * Gonzalez-Rivera et al. (2004) TICK / quantile loss L_alpha(R,VaR) =
    (alpha - 1{R<VaR}) * (R - VaR); lower = better.
  DM on the tick loss: fU vs fR (does text improve VaR?) and fR vs rawHAR (does recalibration
  improve VaR?). dm_test(loss_fU, loss_fR, h): negative stat => text (fU) better.

CAVEAT (overlapping windows): h-day returns from filings closer than h trading days apart
overlap, which induces serial dependence the Christoffersen independence test does not model;
its p-values are therefore only indicative. Kupiec unconditional coverage and the tick loss
are unaffected by this ordering issue. HAC lag = h-1 is used in the DM tests to absorb the
overlap-induced autocorrelation.

Run from repo root:  .venv/bin/python scripts/analysis/var_backtest.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "scripts/analysis")
import forecast_combination as fc  # noqa: E402
sys.path.insert(0, "src")
from sp500vol.evaluation.dm_test import dm_test  # noqa: E402

KEY = ["ticker", "accession", "horizon_days"]
SORT = ["filing_time_utc", "ticker", "accession"]
HORIZONS = (5, 10, 20)
ALPHAS = (0.01, 0.05)
MODELS = ["B2_tfidf_ridge", "C2_finbert_s1", "C2_finbert_s2",
          "C4_longformer", "C5_qwen3", "D2_gated_fusion"]
DISCLOSURES = ["long_form", "event_driven"]
RR_PATH = "results/tables/_realized_returns.parquet"
TRADING_DAYS = 252.0


def deannualize(sigma_ann, h):
    return np.asarray(sigma_ann, float) * np.sqrt(h / TRADING_DAYS)


def kupiec_lr(n, x, alpha):
    """Kupiec (1995) POF unconditional-coverage LR ~ chi2(1). x violations out of n."""
    if n == 0:
        return float("nan"), float("nan")
    pi = x / n
    # log-lik under H0 (rate=alpha) and H1 (rate=pi), guarding the log at the boundaries
    ll0 = x * np.log(alpha) + (n - x) * np.log(1 - alpha)
    if x == 0:
        ll1 = (n - x) * np.log(1 - pi) if pi < 1 else 0.0
    elif x == n:
        ll1 = x * np.log(pi)
    else:
        ll1 = x * np.log(pi) + (n - x) * np.log(1 - pi)
    lr = -2.0 * (ll0 - ll1)
    lr = max(float(lr), 0.0)
    p = float(stats.chi2.sf(lr, df=1))
    return lr, p


def christoffersen(viol_seq, alpha):
    """Christoffersen (1998) independence LR (chi2(1)) and conditional-coverage LR (chi2(2)).

    viol_seq: 0/1 array already ordered in time (within-ticker then across tickers).
    Returns (indep_lr, indep_p, cc_lr, cc_p).
    """
    v = np.asarray(viol_seq, int)
    n = len(v)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    # transition counts n_ij = from state i to state j on consecutive obs
    prev, cur = v[:-1], v[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    x = int(v.sum())

    # independence LR: Markov-chain first order vs iid
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if n > 1 else 0.0

    def _t(k, p):
        if k == 0:
            return 0.0
        if p <= 0:
            return -np.inf  # k>0 but p=0 => impossible under this model
        return k * np.log(p)

    ll_ind = (_t(n00, 1 - pi) + _t(n01, pi) + _t(n10, 1 - pi) + _t(n11, pi))
    ll_mar = (_t(n00, 1 - pi01) + _t(n01, pi01) + _t(n10, 1 - pi11) + _t(n11, pi11))
    if not np.isfinite(ll_ind) or not np.isfinite(ll_mar):
        lr_ind = float("nan")
    else:
        lr_ind = max(-2.0 * (ll_ind - ll_mar), 0.0)
    p_ind = float(stats.chi2.sf(lr_ind, df=1)) if np.isfinite(lr_ind) else float("nan")

    # conditional coverage LR = Kupiec (uc) + independence, ~ chi2(2)
    uc_lr, _ = kupiec_lr(n, x, alpha)
    if np.isfinite(lr_ind) and np.isfinite(uc_lr):
        cc_lr = uc_lr + lr_ind
        cc_p = float(stats.chi2.sf(cc_lr, df=2))
    else:
        cc_lr, cc_p = float("nan"), float("nan")
    return (float(lr_ind) if np.isfinite(lr_ind) else float("nan"),
            p_ind, cc_lr, cc_p)


def tick_loss(R, VaR, alpha):
    """Gonzalez-Rivera (2004) tick / quantile loss for a LEFT-tail VaR. Lower = better."""
    R = np.asarray(R, float)
    VaR = np.asarray(VaR, float)
    ind = (R < VaR).astype(float)
    return (alpha - ind) * (R - VaR)


def main():
    rr = pd.read_parquet(RR_PATH)[KEY + ["fwd_logret", "ret_match_ok"]]
    rr_ok = rr[rr.ret_match_ok].copy()

    rows = []
    tally = []  # per (disc,h,alpha,mu_mode): does text lower tick loss / move viol closer
    kept_total, dropped_total = 0, 0

    for disc in DISCLOSURES:
        har = fc.load("A2_har_rv", disc)[["split"] + KEY +
                                          ["prediction_realised_vol", "label_realised_vol",
                                           "filing_time_utc"]].rename(
            columns={"prediction_realised_vol": "fhar"})
        for m in MODELS:
            txt = fc.load(m, disc)[KEY + ["prediction_realised_vol"]].rename(
                columns={"prediction_realised_vol": "ftext"})
            d = har.merge(txt, on=KEY)
            for h in HORIZONS:
                dv = d[(d.horizon_days == h) & (d.split == "val")].sort_values(
                    SORT, kind="mergesort")
                dt = d[(d.horizon_days == h) & (d.split == "test")].sort_values(
                    SORT, kind="mergesort")
                if len(dv) < 100 or len(dt) < 30:
                    continue
                yv = dv.label_realised_vol.to_numpy()
                fhv = dv.fhar.to_numpy()
                ftv = dv.ftext.to_numpy()
                fhr = dt.fhar.to_numpy()
                ftt = dt.ftext.to_numpy()

                # recalibrated fR and +text fU (annualized RV scale), frozen on val
                fR, fU, _ = fc.log_combo(yv, fhv, ftv, fhr, ftt)

                # attach realized returns; keep ret_match_ok only
                dt_r = dt.merge(rr_ok, on=KEY, how="left")
                kept_mask = dt_r.fwd_logret.notna().to_numpy()
                n_before = len(dt_r)
                dt_r = dt_r[kept_mask].copy()
                fhr_k = fhr[kept_mask]
                fR_k = fR[kept_mask]
                fU_k = fU[kept_mask]
                n_kept = len(dt_r)
                kept_total += n_kept
                dropped_total += (n_before - n_kept)
                if n_kept < 30:
                    continue

                R = dt_r.fwd_logret.to_numpy()
                ticker = dt_r.ticker.to_numpy()
                # de-annualize each vol forecast to h-day
                sig = {
                    "rawHAR": deannualize(fhr_k, h),
                    "fR": deannualize(fR_k, h),
                    "fU": deannualize(fU_k, h),
                }

                # within-ticker time order for the independence transition counts
                order = np.lexsort((dt_r.accession.to_numpy(),
                                    dt_r.filing_time_utc.to_numpy(),
                                    ticker))

                for alpha in ALPHAS:
                    z = stats.norm.ppf(alpha)
                    for mu_mode in ("pooled", "zero"):
                        mu_hat = float(np.mean(R)) if mu_mode == "pooled" else 0.0
                        cell_losses = {}
                        cell_viol = {}
                        for fname in ("rawHAR", "fR", "fU"):
                            VaR = mu_hat + z * sig[fname]
                            viol = (R < VaR)
                            x = int(viol.sum())
                            n = len(R)
                            vr = x / n
                            uc_lr, uc_p = kupiec_lr(n, x, alpha)
                            _, _, cc_lr, cc_p = christoffersen(
                                viol[order].astype(int), alpha)
                            L = tick_loss(R, VaR, alpha)
                            cell_losses[fname] = L
                            cell_viol[fname] = vr
                            rows.append({
                                "disclosure": disc, "model": m, "horizon": h,
                                "alpha": alpha, "mu_mode": mu_mode, "forecast": fname,
                                "n": n, "viol_rate": vr,
                                "kupiec_lr": uc_lr, "kupiec_p": uc_p,
                                "cc_lr": cc_lr, "cc_p": cc_p,
                                "tick_loss": float(L.mean()),
                            })
                        # DM vs fR: negative => the forecast is BETTER than fR on tick loss
                        for fname in ("rawHAR", "fR", "fU"):
                            if fname == "fR":
                                dm_s, dm_p = 0.0, 1.0
                            else:
                                dm_s, dm_p = dm_test(cell_losses[fname],
                                                     cell_losses["fR"], h=h)
                            # write back into the matching row
                            for rr_row in reversed(rows):
                                if (rr_row["disclosure"] == disc and rr_row["model"] == m
                                        and rr_row["horizon"] == h and rr_row["alpha"] == alpha
                                        and rr_row["mu_mode"] == mu_mode
                                        and rr_row["forecast"] == fname
                                        and "dm_tick_vs_fR_stat" not in rr_row):
                                    rr_row["dm_tick_vs_fR_stat"] = float(dm_s)
                                    rr_row["dm_tick_vs_fR_p"] = float(dm_p)
                                    break
                        # tally for the primary question (pooled-mu is the headline)
                        dm_uvr = dm_test(cell_losses["fU"], cell_losses["fR"], h=h)
                        text_tick_better = dm_uvr[0] < 0
                        text_tick_sig = dm_uvr[1] < 0.05
                        viol_closer = (abs(cell_viol["fU"] - alpha)
                                       < abs(cell_viol["fR"] - alpha))
                        tally.append({
                            "disclosure": disc, "model": m, "horizon": h, "alpha": alpha,
                            "mu_mode": mu_mode,
                            "dm_fU_vs_fR_stat": float(dm_uvr[0]),
                            "dm_fU_vs_fR_p": float(dm_uvr[1]),
                            "text_tick_better": bool(text_tick_better),
                            "text_tick_sig_better": bool(text_tick_better and text_tick_sig),
                            "text_tick_sig_worse": bool((dm_uvr[0] > 0) and text_tick_sig),
                            "viol_fR": cell_viol["fR"], "viol_fU": cell_viol["fU"],
                            "viol_rawHAR": cell_viol["rawHAR"],
                            "viol_closer_to_nominal": bool(viol_closer),
                        })

    df = pd.DataFrame(rows)
    tdf = pd.DataFrame(tally)

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    cols = ["disclosure", "model", "horizon", "alpha", "mu_mode", "forecast", "n",
            "viol_rate", "kupiec_lr", "kupiec_p", "cc_lr", "cc_p", "tick_loss",
            "dm_tick_vs_fR_stat", "dm_tick_vs_fR_p"]
    df = df[cols]
    df.to_csv("results/tables/var_backtest.csv", index=False)

    # ---------- markdown ----------
    md = ["# R2 — VaR backtest: economic value of the text increment for downside risk\n"]
    md.append(
        "Three h-day vol forecasts per (disclosure, model, horizon): **rawHAR** (raw A2 HAR, "
        "known to under-forecast vol), **fR** (recalibrated price-only, log-space, frozen on val) "
        "and **fU** (fR + text). Left-tail VaR_alpha = mu_hat + Phi^-1(alpha)*sigma_h, sigma_h = "
        "sigma_ann*sqrt(h/252). Realized R_h = fwd_logret (ret_match_ok only). "
        "DM on the Gonzalez-Rivera tick loss vs fR: **negative = better than fR** (HAC lag h-1).\n")
    md.append(f"**Return-join sanity:** kept (ret_match_ok) = **{kept_total}** filing×cell rows "
              f"across all cells; dropped = **{dropped_total}** "
              f"({100*dropped_total/max(kept_total+dropped_total,1):.1f}% dropped).\n")

    # violation-rate table (pooled mu, headline)
    for alpha in ALPHAS:
        md.append(f"\n## Violation rates vs nominal alpha={alpha:.2f} (pooled-mean mu)\n")
        md.append("A well-calibrated forecast gives viol_rate near alpha. rawHAR under-forecasts "
                  "vol so it should OVER-violate (viol_rate > alpha) more than fR.\n")
        md.append("| disclosure | model | h | viol rawHAR | viol fR | viol fU | Kupiec p (fR) | Kupiec p (fU) |\n"
                  "|---|---|---|---|---|---|---|---|")
        sub = df[(df.alpha == alpha) & (df.mu_mode == "pooled")]
        for (disc, m, h), g in sub.groupby(["disclosure", "model", "horizon"]):
            gg = g.set_index("forecast")
            vraw = gg.loc["rawHAR", "viol_rate"]
            vR = gg.loc["fR", "viol_rate"]
            vU = gg.loc["fU", "viol_rate"]
            kpR = gg.loc["fR", "kupiec_p"]
            kpU = gg.loc["fU", "kupiec_p"]
            md.append(f"| {disc} | {m} | {h} | {vraw:.4f} | {vR:.4f} | {vU:.4f} | "
                      f"{kpR:.3f} | {kpU:.3f} |")

    # rawHAR over-violation check (does rawHAR over-violate more than fR?)
    piv = df[df.mu_mode == "pooled"].pivot_table(
        index=["disclosure", "model", "horizon", "alpha"],
        columns="forecast", values="viol_rate")
    over_raw = ((piv["rawHAR"] - piv.index.get_level_values("alpha"))
                > (piv["fR"] - piv.index.get_level_values("alpha"))).mean()
    raw_over_alpha = (piv["rawHAR"] > piv.index.get_level_values("alpha")).mean()
    fR_over_alpha = (piv["fR"] > piv.index.get_level_values("alpha")).mean()
    md.append(f"\n**Over-violation sanity:** rawHAR viol_rate > alpha in "
              f"{100*raw_over_alpha:.0f}% of cells vs fR in {100*fR_over_alpha:.0f}%; "
              f"rawHAR over-violates by MORE than fR in {100*over_raw:.0f}% of cells "
              f"(expected: rawHAR under-forecasts vol => over-violates).\n")

    # text-vs-recalHAR tick-loss DM verdict per horizon/disclosure (pooled mu, headline)
    md.append("\n## Does text improve VaR? DM on tick loss, fU vs fR (pooled-mean mu)\n")
    md.append("negative DM = text (fU) LOWERS tick loss vs recalibrated HAR (fR). "
              "'closer' = fU violation rate is nearer nominal than fR.\n")
    md.append("| disclosure | h | alpha | model | DM(fU vs fR) | p | text better? | viol closer? |\n"
              "|---|---|---|---|---|---|---|---|")
    th = tdf[tdf.mu_mode == "pooled"].sort_values(
        ["disclosure", "horizon", "alpha", "model"])
    for _, r in th.iterrows():
        verdict = "yes*" if r.text_tick_sig_better else ("yes" if r.text_tick_better else "no")
        md.append(f"| {r.disclosure} | {r.horizon} | {r.alpha:.2f} | {r.model} | "
                  f"{r.dm_fU_vs_fR_stat:+.2f} | {r.dm_fU_vs_fR_p:.3f} | {verdict} | "
                  f"{'yes' if r.viol_closer_to_nominal else 'no'} |")

    # overall tally
    n_cells = len(th)
    n_better = int(th.text_tick_better.sum())
    n_sig_better = int(th.text_tick_sig_better.sum())
    n_sig_worse = int(th.text_tick_sig_worse.sum())
    n_closer = int(th.viol_closer_to_nominal.sum())
    md.append(f"\n## Tally (pooled-mu, {n_cells} disclosure×model×horizon×alpha cells)\n")
    md.append(f"- Text LOWERS tick loss (DM<0) in **{n_better}/{n_cells}** cells; "
              f"significantly (p<.05) in **{n_sig_better}**; significantly WORSE in **{n_sig_worse}**.\n")
    md.append(f"- Text moves the violation rate CLOSER to nominal in **{n_closer}/{n_cells}** cells.\n")

    # recalibration verdict: does fR beat rawHAR on tick loss?
    dR = df[(df.mu_mode == "pooled") & (df.forecast == "rawHAR")]
    n_recal_cells = len(dR)
    n_recal_better = int((dR.dm_tick_vs_fR_stat > 0).sum())  # rawHAR worse than fR => recal helps
    n_recal_sig = int(((dR.dm_tick_vs_fR_stat > 0) & (dR.dm_tick_vs_fR_p < 0.05)).sum())
    md.append(f"- Recalibration (fR vs rawHAR) improves tick loss in **{n_recal_better}/{n_recal_cells}** "
              f"cells (rawHAR worse than fR), significantly in **{n_recal_sig}**.\n")

    md.append("\n## Caveats\n")
    md.append("- **Overlapping windows:** h-day return windows from filings < h trading days apart "
              "overlap, inducing serial dependence the Christoffersen independence test does not model; "
              "its p-values are indicative only. Kupiec UC and tick loss are unaffected; DM uses HAC lag h-1.\n")
    md.append("- Gaussian VaR assumes a Gaussian return given the vol forecast; fat left tails will "
              "raise violations for all three forecasts alike, so the fU-vs-fR comparison stays fair.\n")

    with open("results/tables/var_backtest.md", "w") as fh:
        fh.write("\n".join(md))

    print("=== R2 VaR backtest — done ===")
    print(f"kept={kept_total} dropped={dropped_total} "
          f"({100*dropped_total/max(kept_total+dropped_total,1):.1f}% dropped)")
    print(f"cells (pooled-mu)={n_cells}: text tick-better {n_better} "
          f"(sig {n_sig_better}, sig-worse {n_sig_worse}); viol closer {n_closer}")
    print(f"recalibration better than rawHAR: {n_recal_better}/{n_recal_cells} "
          f"(sig {n_recal_sig})")
    print(f"rawHAR over-violates more than fR in {100*over_raw:.0f}% of cells")
    print("wrote results/tables/var_backtest.csv + .md")
    return dict(kept=kept_total, dropped=dropped_total, n_cells=n_cells,
                n_better=n_better, n_sig_better=n_sig_better, n_sig_worse=n_sig_worse,
                n_closer=n_closer, n_recal_better=n_recal_better,
                n_recal_cells=n_recal_cells, n_recal_sig=n_recal_sig,
                over_raw=float(over_raw), raw_over_alpha=float(raw_over_alpha),
                fR_over_alpha=float(fR_over_alpha))


if __name__ == "__main__":
    main()
