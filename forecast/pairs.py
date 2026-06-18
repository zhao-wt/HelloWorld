"""
forecast/pairs.py — relative / cross-sectional sector-pair forecasting.

Instead of forecasting a single sector's absolute return (which the sectors.py
experiments showed is not reliably doable out-of-sample), this forecasts the
RELATIVE return of a pair, A minus B:

    spread_h(t) = ret_A(t->t+h) - ret_B(t->t+h)        (long A / short B)

Why relative is easier
----------------------
The absolute return of any sector is dominated by the common market move, which
is mostly noise at these horizons and drowns the signal. Differencing two
sectors CANCELS that common move, leaving the relative dynamics — where
cross-sectional momentum, relative trend, and DIFFERENTIAL macro/commodity
exposure live. Crucially the spread is ~market-neutral (mean ≈ 0), so its
directional hit-rate ("did we pick the right sector to overweight?") is a clean
skill measure, NOT confounded by the market's up-rate.

Everything else — members (elastic net / random forest / k-NN / RSZ), the
R²-weighted, drop-negative ensemble, expanding-window CV for weights, the strict
pre-2020 / post-2020 split, no-look-ahead lags, and weekly/monthly frequency —
is reused unchanged from forecast.sectors.

Predictors (all relative or differential, lagged one period)
------------------------------------------------------------
* relative technicals from the price RATIO A/B: cross-sectional momentum,
  relative trend, spread reversal, spread volatility;
* commodity momentum/changes (oil, gas, gold, copper) — these hit sectors
  ASYMMETRICALLY (e.g. oil helps Energy, not Health Care), so they should
  predict the spread far better than either sector's level;
* the macro subset (differential sector sensitivities).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import forecast.sectors as S

_DATA_DIR = S._DATA_DIR


def _align(px_a: pd.Series, px_b: pd.Series) -> tuple[pd.Series, pd.Series]:
    idx = px_a.index.intersection(px_b.index)
    return px_a.loc[idx], px_b.loc[idx]


def build_pair_targets(px_a: pd.Series, px_b: pd.Series) -> pd.DataFrame:
    """Forward relative return spread (A - B) at each horizon."""
    out = {}
    for h in S._FREQ["horizons"]:
        ra = px_a.shift(-h) / px_a - 1.0
        rb = px_b.shift(-h) / px_b - 1.0
        out[S._col(h)] = ra - rb
    return pd.DataFrame(out, index=px_a.index)


def build_pair_features(px_a: pd.Series, px_b: pd.Series) -> pd.DataFrame:
    """Relative technicals (from the A/B ratio) + commodities + macro, lagged."""
    w = S._FREQ["win"]
    ratio = (px_a / px_b).rename("ratio")
    ma = ratio.rolling(w["ma"], min_periods=max(w["ma"] // 2, 3)).mean()
    rel_ret = ratio.pct_change()                          # 1-period relative return
    a_ret, b_ret = px_a.pct_change(), px_b.pct_change()

    f = pd.DataFrame(index=ratio.index)
    f["rel_mom_12_1"] = ratio.shift(w["mom_skip"]) / ratio.shift(w["mom_long"]) - 1.0
    f["rel_mom_6_1"]  = ratio.shift(w["mom_skip"]) / ratio.shift(w["mom_mid"]) - 1.0
    f["rel_vs_ma"]    = ratio / ma - 1.0
    f["rel_ret_1"]    = rel_ret                           # short-term reversal
    f["rel_vol"]      = (a_ret - b_ret).rolling(w["vol"], min_periods=w["vol_mp"]) \
        .std(ddof=1) * np.sqrt(S._FREQ["ppy"])
    f = f.shift(1)                                        # publication lag

    # Commodities (asymmetric sector exposure) + macro (differential sensitivity).
    f = f.join(S.build_commodity_features(ratio.index), how="left")
    macro = pd.read_csv(_DATA_DIR / "forecast_features.csv",
                        index_col=0, parse_dates=True).sort_index()
    have = [c for c in S.MACRO_PREDICTORS if c in macro.columns]
    f = f.join(macro[have].reindex(f.index, method="ffill"), how="left")
    return f


def pair_predictor_list(feats: pd.DataFrame) -> list[str]:
    rel = ["rel_mom_12_1", "rel_mom_6_1", "rel_vs_ma", "rel_ret_1", "rel_vol"]
    commodity = [f"{c}_{s}" for c in S.COMMODITY_TICKERS for s in S.COMMODITY_SUFFIXES]
    return [c for c in rel + commodity + S.MACRO_PREDICTORS if c in feats.columns]


def run_pair(name_a: str, name_b: str, freq: str = "M", mode: str = "split") -> dict:
    S._FREQ = S.FREQS[freq]
    unit = S._FREQ["unit"]
    hz = S._FREQ["horizons"]
    ppy = S._FREQ["ppy"]
    fname = "weekly" if unit == "w" else "monthly"
    wf = (mode == "wf")

    sym_a, px_a = S.choose_etf(*S.SECTORS[name_a])
    sym_b, px_b = S.choose_etf(*S.SECTORS[name_b])
    px_a, px_b = _align(px_a, px_b)

    mlabel = "walk-forward (adaptive weights)" if wf else "fixed pre/post-2020 split"
    print(f"\n{'='*72}\n  PAIR  {name_a} ({sym_a})  −  {name_b} ({sym_b})  "
          f"[{fname}, {mlabel}]")
    print(f"  long {sym_a} / short {sym_b};  common history "
          f"{px_a.index[0].date()} → {px_a.index[-1].date()}  ({len(px_a)} obs)\n{'='*72}")

    targets = build_pair_targets(px_a, px_b)
    feats = build_pair_features(px_a, px_b)
    predictors = pair_predictor_list(feats)

    by_h, oos_frames, current = {}, {}, {}
    for h in hz:
        runner = S.run_horizon_walkforward if wf else S.run_horizon
        out, metrics, (cur, as_of) = runner(feats, targets, predictors, h)
        by_h[str(h)] = metrics
        oos_frames[h] = out
        current[str(h)] = cur
        current["as_of"] = as_of

    def lab(h):
        return f"{h}{unit}"

    hdr = "2020+ R2_OS" if wf else "Test R2_OS"
    print(f"  Predictors ({len(predictors)}): {predictors}")
    print(f"\n  Spread = {sym_a} − {sym_b} forward return.  Positive forecast ⇒ "
          f"overweight {sym_a}; negative ⇒ overweight {sym_b}.")
    print(f"\n  {'Horizon':<8}{'Fcst spread':>13}{hdr:>12}"
          f"{'Dir. hit':>10}{'n_test':>8}{'n_train':>9}")
    print(f"  {'-'*8}{'-'*13}{'-'*12}{'-'*10}{'-'*8}{'-'*9}")
    for h in hz:
        em = by_h[str(h)]["ensemble"]
        print(f"  {lab(h):<8}{current[str(h)]:>+12.2%}{em['r2_os']:>+12.3f}"
              f"{em['hit']:>9.1%}{em['n']:>8}{by_h[str(h)]['_n_train']:>9}")

    win = "2020+" if wf else "Test"
    print(f"\n  Per-member {win} R2_OS:")
    print(f"  {'Member':<12}" + "".join(f"{lab(h):>9}" for h in hz))
    for mk in ["enet", "rf", "knn", "rsz", "ensemble"]:
        print(f"  {mk:<12}" + "".join(f"{by_h[str(h)][mk]['r2_os']:>+9.3f}" for h in hz))
    print(f"\n  Per-member {win} directional hit-rate (market-neutral — 50% = no skill):")
    print(f"  {'Member':<12}" + "".join(f"{lab(h):>9}" for h in hz))
    for mk in ["enet", "rf", "knn", "rsz", "ensemble"]:
        print(f"  {mk:<12}" + "".join(f"{by_h[str(h)][mk]['hit']:>8.0%} " for h in hz))

    if wf:
        print(f"\n  Full walk-forward window (2008+) ensemble:")
        print(f"  {'Horizon':<8}{'R2_OS':>9}{'Dir. hit':>10}{'n':>7}")
        for h in hz:
            fm = by_h[str(h)]["_full"]
            print(f"  {lab(h):<8}{fm['r2_os']:>+9.3f}{fm['hit']:>9.1%}{fm['n']:>7}")
        print(f"\n  Avg member weight over 2020+ (adaptive, trailing-skill weighted):")
        print(f"  {'Member':<12}" + "".join(f"{lab(h):>9}" for h in hz))
        for mk in S.ENS_MEMBERS:
            print(f"  {mk:<12}" + "".join(
                f"{by_h[str(h)]['_avg_weights'][mk]:>9.0%}" for h in hz))
    else:
        print(f"\n  Ensemble weights (2010-2019 expanding-window CV R²_OS; "
              f"members with R²_OS<=0 dropped):")
        for h in hz:
            w = by_h[str(h)]["_weights"]; vr = by_h[str(h)]["_val_r2"]
            wstr = ("CV infeasible → RSZ-only" if w is None else
                    "all members failed → mean (no tilt)" if not w else
                    ", ".join(f"{m} {w[m]:.0%}" for m in w))
            print(f"    {lab(h):<5} weights: {wstr}")
            print(f"          val R²_OS: " + ", ".join(
                f"{m} {('NA' if vr[m] is None else format(vr[m],'+.3f'))}" for m in S.ENS_MEMBERS))

    pair = f"{sym_a}_{sym_b}"
    summary = {
        "pair": f"{name_a} - {name_b}", "long": sym_a, "short": sym_b,
        "frequency": fname, "common_start": str(px_a.index[0].date()),
        "test_start": str(S.TEST_START.date()), "as_of": current.get("as_of"),
        "horizons": [f"{h}{unit}" for h in hz], "predictors": predictors,
        "current_forecast": {str(h): current[str(h)] for h in hz},
        "by_horizon": by_h,
    }
    suffix = f"_{unit}" + ("_wf" if wf else "")
    with open(_DATA_DIR / f"pair_{pair}{suffix}_params.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    for h in hz:
        oos_frames[h].to_csv(_DATA_DIR / f"pair_{pair}{suffix}_oos_{h}{unit}.csv",
                             date_format="%Y-%m-%d", float_format="%.6f")
    print(f"\n  Wrote data/pair_{pair}{suffix}_params.json + pair_{pair}{suffix}_oos_*.csv")
    return summary


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    freq, mode = "M", "split"
    keep = []
    for a in args:
        al = a.lower()
        if al in ("w", "weekly", "week"):
            freq = "W"
        elif al in ("m", "monthly", "month"):
            freq = "M"
        elif al in ("wf", "walkforward", "walk-forward"):
            mode = "wf"
        elif al in ("split", "fixed"):
            mode = "split"
        else:
            keep.append(a)
    a, b = (keep + ["Health Care", "Energy"])[:2] if len(keep) < 2 else keep[:2]
    run_pair(a, b, freq, mode)
