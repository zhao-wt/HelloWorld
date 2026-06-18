"""
forecast/xs_factors.py — three-sleeve cross-sectional sector model.

Three economically-distinct ranking signals ("sleeves"), each scoring the 11
sectors cross-sectionally, then combined by trailing-IC weighting (dropping any
sleeve whose trailing IC<=0) — the same honest rule used elsewhere.

  1. momentum  : own 12-month price momentum (the classic cross-sectional signal).
  2. commodity : each sector's BETA to oil / copper / gas / gold (rolling, lagged)
                 timed by that commodity's momentum —
                     score_i = Σ_c beta_{i,c} · mom_c
                 i.e. overweight high-beta sectors when the commodity is trending up.
  3. rates     : each sector's DURATION (beta to the 12-month change in the 10y
                 yield, rolling, lagged) timed by the yield's momentum —
                     score_i = rate_beta_i · rate_mom
                 i.e. tilt toward rate-sensitive sectors in the direction of the
                 yield trend (Financials when yields rise, Utilities/RE when they fall).

This is a small factor-timing asset-pricing model: betas/durations are the factor
loadings (estimated by rolling regression, strictly trailing → no look-ahead) and
the commodity/rate momentum supplies the timing. Evaluation reuses the rank IC and
rank-weighted long/short from forecast.cross_sectional.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import forecast.sectors as S
import forecast.cross_sectional as X

_DATA_DIR = S._DATA_DIR
BETA_WIN = 36          # rolling window (periods) for beta/duration estimation
BETA_MINP = 24
COMMODITIES = {"oil": "USO", "copper": "HG=F", "gas": "UNG", "gold": "GC=F"}
SLEEVES = ["momentum", "commodity", "rates"]


def _zscore_xs(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score each row (across the sector columns)."""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def _mom(px: pd.Series, n: int) -> pd.Series:
    """n-period return through t-1 (standard single-month skip via the lag)."""
    return (px / px.shift(n) - 1.0).shift(1)


def _rolling_beta(r_i: pd.Series, r_f: pd.Series) -> pd.Series:
    """Trailing beta of sector returns r_i on factor returns r_f, lagged 1 period."""
    cov = r_i.rolling(BETA_WIN, min_periods=BETA_MINP).cov(r_f)
    var = r_f.rolling(BETA_WIN, min_periods=BETA_MINP).var()
    return (cov / var.replace(0, np.nan)).shift(1)


# ---------------------------------------------------------------------------
# Sleeve scores (date x sector)
# ---------------------------------------------------------------------------

def build_sleeves(prices: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
    idx = list(prices.values())[0].index
    w = S._FREQ["win"]
    sectors = list(prices)
    rets = {s: prices[s].pct_change() for s in sectors}

    # ---- 1. Momentum ----
    mom = pd.DataFrame({s: _mom(prices[s], w["mom_long"]) for s in sectors})

    # ---- 2. Commodity: Σ_c beta_{i,c} · mom_c ----
    comm_px, comm_mom = {}, {}
    for name, tic in COMMODITIES.items():
        try:
            cpx = S.fetch_etf_prices(tic).reindex(idx, method="ffill")
        except Exception as exc:
            print(f"  [warn] {tic} ({name}) unavailable: {exc} — skipping")
            continue
        comm_px[name] = cpx
        comm_mom[name] = _mom(cpx, w["mom_long"])
    cmdty = pd.DataFrame(0.0, index=idx, columns=sectors)
    contrib = pd.DataFrame(0.0, index=idx, columns=sectors)
    for name, cpx in comm_px.items():
        cret = cpx.pct_change()
        m = comm_mom[name]
        for s in sectors:
            beta = _rolling_beta(rets[s], cret)
            term = (beta * m)
            cmdty[s] = cmdty[s].add(term.fillna(0.0), fill_value=0.0)
            contrib[s] = contrib[s].add(term.notna().astype(float), fill_value=0.0)
    cmdty = cmdty.where(contrib > 0)            # NaN where no commodity available yet

    # ---- 3. Rates: rate_beta_i · rate_mom ----
    raw = pd.read_csv(_DATA_DIR / "raw_monthly.csv", index_col=0, parse_dates=True)
    dgs10 = raw["DGS10"].reindex(idx, method="ffill")
    dyield = dgs10.diff()
    rate_mom = dgs10.diff(w["mom_long"]).shift(1)        # 12-period yield change, lagged
    rates = pd.DataFrame(index=idx, columns=sectors, dtype=float)
    for s in sectors:
        rbeta = _rolling_beta(rets[s], dyield)
        rates[s] = rbeta * rate_mom

    return {"momentum": _zscore_xs(mom),
            "commodity": _zscore_xs(cmdty),
            "rates": _zscore_xs(rates)}


def realized_rel(prices: dict[str, pd.Series], h: int) -> pd.DataFrame:
    fwd = pd.DataFrame({s: prices[s].shift(-h) / prices[s] - 1.0 for s in prices})
    return fwd.sub(fwd.mean(axis=1), axis=0)             # cross-sectionally demeaned


# ---------------------------------------------------------------------------
# Trailing-IC-weighted ensemble of the three sleeves
# ---------------------------------------------------------------------------

def _ic_row(s, r):
    d = pd.concat([s.rename("s"), r.rename("r")], axis=1).dropna()
    if len(d) < 6 or d["s"].std() == 0:
        return np.nan
    return float(np.corrcoef(d["s"], d["r"])[0, 1])


def combine(scores: dict[str, pd.DataFrame], realized: pd.DataFrame, h: int):
    dates = realized.index
    off = S._offset(h); trail = S._FREQ["wf_trail"]; min_scored = S._FREQ["wf_min_scored"]
    toff = (pd.DateOffset(months=trail) if S._FREQ["unit"] == "m"
            else pd.Timedelta(weeks=trail))
    ic = {m: pd.Series({t: _ic_row(scores[m].loc[t], realized.loc[t]) for t in dates})
          for m in SLEEVES}
    ens = pd.DataFrame(index=dates, columns=realized.columns, dtype=float)
    wlog = {m: [] for m in SLEEVES}
    for t in dates:
        scored = [u for u in dates if (u + off <= t) and (u >= t - toff)]
        w = {}
        if len(scored) >= min_scored:
            for m in SLEEVES:
                ics = ic[m].reindex(scored).dropna()
                if len(ics) >= min_scored // 2 and ics.mean() > 0:
                    w[m] = ics.mean()
        if not w:
            ens.loc[t] = scores["momentum"].loc[t]       # default: momentum sleeve
            for m in SLEEVES:
                wlog[m].append(1.0 if m == "momentum" else 0.0)
        else:
            tot = sum(w.values())
            ens.loc[t] = sum((v / tot) * scores[m].loc[t] for m, v in w.items())
            for m in SLEEVES:
                wlog[m].append(w.get(m, 0.0) / tot)
    avg_w = {m: float(np.mean(wlog[m])) for m in SLEEVES}
    return ens, avg_w


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(freq: str = "M") -> dict:
    S._FREQ = S.FREQS[freq]
    unit = S._FREQ["unit"]; hz = S._FREQ["horizons"]; ppy = S._FREQ["ppy"]
    prices = X.load_universe()
    idx0 = list(prices.values())[0].index
    print(f"\n{'='*78}\n  THREE-SLEEVE CROSS-SECTIONAL MODEL — 11 sectors  [monthly]")
    print(f"  momentum · commodity-beta×mom · rate-duration×mom   "
          f"({idx0[0].date()} → {idx0[-1].date()})\n{'='*78}")

    scores_by_h = {}
    by_h = {}
    for h in hz:
        scores = build_sleeves(prices)
        realized = realized_rel(prices, h)
        # restrict scores/realized to the cross-sectional OOS window
        dates = realized.index[realized.index >= X.XS_OOS_START]
        scores = {m: scores[m].reindex(dates) for m in SLEEVES}
        realized = realized.reindex(dates)
        ens, avg_w = combine(scores, realized, h)
        alls = {**scores, "ensemble": ens}
        test_mask = np.asarray(dates >= S.TEST_START)
        full_mask = np.ones(len(dates), dtype=bool)
        by_h[str(h)] = {
            "test": {m: X.evaluate(alls[m], realized, test_mask) for m in SLEEVES + ["ensemble"]},
            "full": {m: X.evaluate(alls[m], realized, full_mask) for m in SLEEVES + ["ensemble"]},
            "avg_w": avg_w,
        }

    def lab(h):
        return f"{h}{unit}"

    print(f"  IC = cross-sectional corr(score, realized rel. return).  "
          f"L/S = rank-weighted $1-long/$1-short, annualized.\n")
    for win, key in [("2020+ (out-of-sample test)", "test"),
                     (f"Full ({X.XS_OOS_START.year}+)", "full")]:
        print(f"  === {win} ===")
        print(f"  {'Sleeve':<11}" + "".join(f"{lab(h):>21}" for h in hz))
        print(f"  {'':<11}" + "".join(f"{'IC    L/S(ann)  IR':>21}" for h in hz))
        for m in SLEEVES + ["ensemble"]:
            cells = ""
            for h in hz:
                r = by_h[str(h)][key][m]
                ann = r["ls_ret"] * (ppy / h) if not np.isnan(r["ls_ret"]) else np.nan
                cells += f"{r['ic']:>+7.3f}{ann:>+8.1%}{r['ls_ir']:>6.2f}"
            print(f"  {m:<11}{cells}")
        print()

    print(f"  Avg ensemble sleeve weights (2020+):")
    print(f"  {'Sleeve':<11}" + "".join(f"{lab(h):>9}" for h in hz))
    for m in SLEEVES:
        print(f"  {m:<11}" + "".join(f"{by_h[str(h)]['avg_w'][m]:>9.0%}" for h in hz))

    summary = {"model": "xs_factor_sleeves", "frequency": "monthly",
               "sleeves": SLEEVES, "commodities": COMMODITIES,
               "beta_window": BETA_WIN, "oos_start": str(X.XS_OOS_START.date()),
               "horizons": [f"{h}{unit}" for h in hz], "by_horizon": by_h}
    with open(_DATA_DIR / f"xs_factors_{unit}_params.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\n  Wrote data/xs_factors_{unit}_params.json")
    return summary


if __name__ == "__main__":
    import sys
    freq = "W" if any(a.lower() in ("w", "weekly") for a in sys.argv[1:]) else "M"
    run(freq)
