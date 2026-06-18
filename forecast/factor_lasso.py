"""
forecast/factor_lasso.py — regularized sector factor betas (Lasso / Elastic Net),
estimated over two windows so coefficient migration over time is visible.

For each sector, the monthly total return is regressed on five factors:
    market (SPY), crude oil, Δ10y yield (duration), Δemployment rate, Δinflation.
Lasso and Elastic Net shrink weak factors to EXACTLY zero (automatic removal of
insignificant exposures). Factors are standardized for the fit (so the L1 penalty
is fair across factors), then coefficients are converted back to NATURAL units:
    market   = dimensionless beta
    oil      = % sector return per +1% crude-oil move
    duration = % return per +1pp 10y yield   (×100)
    employ.  = % return per +1pp employment rate (×100)
    inflation= % return per +1pp CPI inflation   (×100)
Zero = factor dropped by the regularizer.

Two windows:
  * full       — each sector's full history (ETF inception → today).
  * postcovid  — 2020-01 → today.

Run:  python -m forecast.factor_lasso
"""

from __future__ import annotations

import json
import warnings
from datetime import date

import numpy as np
import pandas as pd

import forecast.sectors as S

warnings.filterwarnings("ignore")
_DATA_DIR = S._DATA_DIR
FACTORS = ["mkt", "oil", "dur", "unemp", "infl"]
DISPLAY_SCALE = {"mkt": 1.0, "oil": 1.0, "dur": 100.0, "unemp": 100.0, "infl": 100.0}
POSTCOVID = pd.Timestamp("2020-01-31")


def fetch_oil_monthly() -> pd.Series:
    """Long-history crude oil: FRED WTI (1986) → yfinance CL=F → USO fallback."""
    try:
        from bear.data import fetch_fred_series, _resolve_fred_api_key
        key = _resolve_fred_api_key()
        if key:
            s = fetch_fred_series("MCOILWTICO", date(1980, 1, 1), date.today(),
                                  key, S._CACHE_DIR, frequency="m", aggregation_method="avg")
            return s.resample("ME").last().dropna()
    except Exception as exc:
        print(f"  [oil] FRED WTI unavailable ({exc}); trying futures")
    for tic in ("CL=F", "USO"):
        try:
            return S.fetch_etf_prices(tic)
        except Exception:
            continue
    raise RuntimeError("no oil series available")


def load_factors() -> pd.DataFrame:
    spy = S.fetch_etf_prices("SPY")
    oil = fetch_oil_monthly()
    raw = pd.read_csv(_DATA_DIR / "raw_monthly.csv", index_col=0, parse_dates=True)
    return pd.DataFrame({
        "mkt":   spy.pct_change(),
        "oil":   oil.pct_change(),
        "dur":   raw["DGS10"].diff(),
        "unemp": raw["UNRATE"].diff(),          # +1pp = unemployment rate rising
        "infl":  (raw["CPI"].pct_change(12) * 100.0).diff(),
    }).dropna(how="all")


def _fit(y: pd.Series, F: pd.DataFrame, method: str, start):
    from sklearn.linear_model import LassoCV, ElasticNetCV
    d = pd.concat([y.rename("y"), F[FACTORS]], axis=1).dropna()
    if start is not None:
        d = d[d.index >= start]
    if len(d) < 36:
        return None
    Xr = d[FACTORS].values
    mu, sd = Xr.mean(0), Xr.std(0, ddof=0)
    sd[sd == 0] = 1.0
    Xs = (Xr - mu) / sd
    if method == "lasso":
        m = LassoCV(cv=5, n_alphas=120, max_iter=200000, random_state=42)
    else:
        m = ElasticNetCV(l1_ratio=[0.5, 0.7, 0.9, 0.95], cv=5, n_alphas=120,
                         max_iter=200000, random_state=42)
    m.fit(Xs, d["y"].values)
    coef_nat = m.coef_ / sd                       # back to natural units
    r2 = float(m.score(Xs, d["y"].values))
    return {f: float(coef_nat[i] * DISPLAY_SCALE[f]) for i, f in enumerate(FACTORS)} | {
        "r2": round(r2, 3), "n": len(d),
        "since": str(d.index[0].date()), "to": str(d.index[-1].date())}


def build() -> dict:
    S._FREQ = S.FREQS["M"]
    F = load_factors()
    out = {"full": {"lasso": [], "enet": []}, "postcovid": {"lasso": [], "enet": []}}
    for name in S.SECTORS:
        sym, px = S.choose_etf(*S.SECTORS[name])
        y = px.pct_change()
        for win, start in [("full", None), ("postcovid", POSTCOVID)]:
            for method in ("lasso", "enet"):
                r = _fit(y, F, method, start)
                if r:
                    out[win][method].append({"sector": name, "ticker": sym, **r})
    with open(_DATA_DIR / "factor_lasso_params.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def _print_table(rows: list[dict], title: str) -> None:
    print(f"\n  {title}")
    print(f"  {'Sector':<14}{'Since':>9}{'Mkt β':>8}{'Oil/1%':>8}{'Dur/pp':>8}"
          f"{'Unemp/pp':>9}{'Infl/pp':>9}{'R²':>6}")
    print(f"  {'-'*14}{'-'*9}{'-'*8}{'-'*8}{'-'*8}{'-'*9}{'-'*9}{'-'*6}")
    for r in rows:
        def c(f):
            v = r[f]
            return "  .  " if abs(v) < 1e-9 else f"{v:+.2f}"
        print(f"  {r['sector']:<14}{r['since'][:7]:>9}{c('mkt'):>8}{c('oil'):>8}"
              f"{c('dur'):>8}{c('unemp'):>9}{c('infl'):>9}{r['r2']:>6.2f}")


if __name__ == "__main__":
    res = build()
    for method, mlabel in [("enet", "ELASTIC NET"), ("lasso", "LASSO")]:
        print(f"\n{'='*76}\n  {mlabel} — factor betas ('.' = dropped to zero)\n{'='*76}")
        _print_table(res["full"][method], f"{mlabel}: FULL HISTORY (per-sector inception → 2026)")
        _print_table(res["postcovid"][method], f"{mlabel}: POST-PANDEMIC (2020-01 → 2026)")
    print(f"\n  Units: Mkt β dimensionless; others % sector return per +1% oil / "
          f"+1pp yield / +1pp employment / +1pp inflation.  Wrote data/factor_lasso_params.json")
