"""
forecast/sector_rotation.py — offline builder for the Sector Rotation tab.

Produces data/sector_rotation_params.json with two independent pieces:

1. MOMENTUM TILT (the one signal that survived every leak-free test) — the 11
   sectors ranked by 12-month cross-sectional price momentum (the score used by
   the `mom` member at the 6-month forecast horizon), with a rank-weighted
   dollar-neutral over/underweight tilt and the signal's historical skill (IC).

2. FACTOR-RISK BETAS (descriptive, NOT a forecast) — each sector's sensitivity
   to four basic factors over a trailing window, from one multivariate
   regression per sector:
       r_sector = a + b_mkt·SPX + b_cmdty·OIL + b_rate·Δ10y + b_emp·EMP + e
   Factors are standardized so the betas are comparable "% sector return per 1σ
   factor shock"; the classic raw market beta and the regression R² are also
   reported. This is cross-sectional RISK analysis (current exposures), separate
   from the momentum forecast.

Run:  python -m forecast.sector_rotation
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import forecast.sectors as S
import forecast.cross_sectional as X

_DATA_DIR = S._DATA_DIR
BETA_WINDOW = 60        # trailing months for the current factor betas

FACTOR_LABELS = {
    "mkt":  "Market beta — sensitivity to the S&P 500 (SPY); ~1.0 = moves with the market",
    "oil":  "Commodity — % sector return per +1% move in crude oil (USO)",
    "dur":  "Duration — % sector return per +1 percentage-point in the 10y Treasury yield",
    "emp":  "Employment — % sector return per +1 percentage-point in the employment rate",
    "infl": "Inflation — % sector return per +1 percentage-point in CPI inflation (YoY)",
}


def _zscore_xs_row(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=1) if s.std(ddof=1) > 0 else s * 0.0


def _mom(px: pd.Series, n: int) -> pd.Series:
    """n-period price return through t-1 (standard single-month skip via the lag)."""
    return (px / px.shift(n) - 1.0).shift(1)


def build_momentum(prices: dict[str, pd.Series]) -> tuple[list[dict], str]:
    """Current 12-month cross-sectional momentum ranking + rank-weighted tilt."""
    w = S._FREQ["win"]
    mom = pd.DataFrame({name: _mom(px, w["mom_long"]) for name, px in prices.items()})
    mom = mom.dropna(how="any")
    as_of = mom.index[-1]
    raw = mom.loc[as_of]
    z = _zscore_xs_row(raw)
    ranks = z.rank(ascending=False)                      # 1 = strongest momentum
    # rank-weighted dollar-neutral tilt (longs sum +1, shorts −1)
    centered = z.rank() - z.rank().mean()
    pos, neg = centered[centered > 0].sum(), -centered[centered < 0].sum()
    weight = centered.where(centered <= 0, centered / pos).where(
        centered >= 0, centered / neg)
    n = len(z)
    rows = []
    for name in prices:
        r = int(ranks[name])
        tilt = "Overweight" if r <= n // 3 + 1 else ("Underweight" if r > n - (n // 3 + 1)
                                                     else "Neutral")
        rows.append({
            "sector": name, "ticker": prices[name].name,
            "score": round(float(z[name]), 3),
            "mom_12m": round(float(raw[name]), 4),
            "rank": r, "tilt": tilt, "weight": round(float(weight[name]), 3),
        })
    rows.sort(key=lambda d: d["rank"])
    return rows, str(as_of.date())


def build_factors(prices: dict[str, pd.Series]) -> tuple[list[dict], str]:
    """
    Trailing-window UNIVARIATE factor betas in NATURAL units (risk analysis).
    Each beta is the slope of a simple regression of the sector return on ONE
    factor (Cov(y,f)/Var(f)) over the window — the sector's actual co-movement
    with that factor, stable under the factors' mutual correlation. Reported:
        beta_mkt  = β to SPX return      (dimensionless market beta, ~1.0)
        beta_oil  = β to oil return      (% sector return per +1% crude-oil move)
        beta_dur  = β to Δ10y · 100      (% sector return per +1pp 10y yield)
        beta_emp  = β to Δemp · 100      (% sector return per +1pp employment rate)
        beta_infl = β to Δinfl · 100     (% sector return per +1pp CPI inflation)
    r2 is the JOINT R² of all five factors together (one multivariate fit), as a
    summary of how much of the sector's variance these factors collectively explain.
    Factor units: oil as a return; yield/employment/inflation as percentage-point
    changes (employment rate = −Δunemployment; inflation = ΔYoY CPI).
    """
    idx = list(prices.values())[0].index
    spy = S.fetch_etf_prices("SPY").reindex(idx, method="ffill")
    oil = S.fetch_etf_prices("USO").reindex(idx, method="ffill")
    raw = pd.read_csv(_DATA_DIR / "raw_monthly.csv", index_col=0, parse_dates=True)
    dgs10 = raw["DGS10"].reindex(idx, method="ffill")
    unrate = raw["UNRATE"].reindex(idx, method="ffill")
    infl_yoy = (raw["CPI"].pct_change(12) * 100.0).reindex(idx, method="ffill")

    F = pd.DataFrame({
        "mkt":  spy.pct_change(),       # market return (decimal)
        "oil":  oil.pct_change(),       # crude-oil return (decimal; 0.01 = +1%)
        "dur":  dgs10.diff(),           # Δ 10y yield (percentage points)
        "emp":  -unrate.diff(),         # Δ employment rate ≈ −Δ unemployment (pp)
        "infl": infl_yoy.diff(),        # Δ CPI inflation, YoY (percentage points)
    })
    cols = ["mkt", "oil", "dur", "emp", "infl"]
    win = F.dropna().index[-BETA_WINDOW:]       # last BETA_WINDOW complete months
    Fw = F.loc[win]

    rows = []
    for name, px in prices.items():
        d = pd.concat([px.pct_change().rename("y"), Fw], axis=1).dropna()
        if len(d) < 30:
            continue
        y = d["y"].values

        def slope(fname):                       # univariate OLS slope Cov/Var
            c = np.cov(y, d[fname].values, ddof=1)
            return c[0, 1] / c[1, 1] if c[1, 1] > 0 else float("nan")

        # joint R² (all five factors together) — a fit summary only
        Xd = np.column_stack([np.ones(len(d)), d[cols].values])
        bj, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        ss_res = float(((y - Xd @ bj) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        rows.append({
            "sector": name, "ticker": px.name,
            "beta_mkt":  round(float(slope("mkt")), 2),          # dimensionless
            "beta_oil":  round(float(slope("oil")), 3),          # % per +1% oil
            "beta_dur":  round(float(slope("dur") * 100), 2),    # % per +1pp yield
            "beta_emp":  round(float(slope("emp") * 100), 2),    # % per +1pp employment
            "beta_infl": round(float(slope("infl") * 100), 2),   # % per +1pp inflation
            "r2": round(float(r2), 3),
        })
    return rows, f"{win[0].date()} → {win[-1].date()}"


def signal_stats() -> dict:
    """Pull the momentum signal's historical IC/IR (6m) from the xsection run."""
    try:
        xs = json.load(open(_DATA_DIR / "xsection_m_params.json"))
        s6 = xs["by_horizon"]["6"]
        return {
            "ic_full": s6["full"]["mom"]["ic"], "ic_test": s6["test"]["mom"]["ic"],
            "ls_ret_full": s6["full"]["mom"]["ls_ret"],
            "ls_ir_full": s6["full"]["mom"]["ls_ir"],
            "horizon": "6m", "signal": "12-month price momentum",
        }
    except Exception:
        return {}


def build() -> dict:
    S._FREQ = S.FREQS["M"]
    prices = X.load_universe()
    ranking, as_of = build_momentum(prices)
    betas, beta_window = build_factors(prices)
    summary = {
        "as_of": as_of,
        "universe_start": str(list(prices.values())[0].index[0].date()),
        "momentum": {"signal": "12-month cross-sectional price momentum",
                     "forecast_horizon": "6 months", "ranking": ranking,
                     "stats": signal_stats()},
        "factor_risk": {"window": beta_window, "window_months": BETA_WINDOW,
                        "factors": FACTOR_LABELS, "betas": betas},
    }
    with open(_DATA_DIR / "sector_rotation_params.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    return summary


if __name__ == "__main__":
    s = build()
    print(f"Sector Rotation — as of {s['as_of']}  (betas: {s['factor_risk']['window']})\n")
    print(f"{'Rank':<5}{'Sector':<15}{'ETF':<6}{'Mom z':>7}{'12m':>9}  Tilt")
    for r in s["momentum"]["ranking"]:
        print(f"{r['rank']:<5}{r['sector']:<15}{r['ticker']:<6}{r['score']:>+7.2f}"
              f"{r['mom_12m']:>+9.1%}  {r['tilt']}")
    st = s["momentum"]["stats"]
    if st:
        print(f"\n  Signal skill (6m): IC full {st['ic_full']:+.3f} · "
              f"test {st['ic_test']:+.3f} · L/S IR {st['ls_ir_full']:.2f}")
    print(f"\n{'Sector':<15}{'Mkt β':>7}{'Oil%/1%':>9}{'Dur%/pp':>9}"
          f"{'Emp%/pp':>9}{'Infl%/pp':>9}{'R²':>7}")
    for b in s["factor_risk"]["betas"]:
        print(f"{b['sector']:<15}{b['beta_mkt']:>7.2f}{b['beta_oil']:>+9.3f}"
              f"{b['beta_dur']:>+9.2f}{b['beta_emp']:>+9.2f}{b['beta_infl']:>+9.2f}"
              f"{b['r2']:>7.2f}")
