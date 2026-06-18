"""
forecast/ensemble.py — combination forecast + out-of-sample metrics.

Combines the member forecasts into one ensemble per horizon by the equal-weight
mean of the model-family members (enet, knn, rf, mlp) — the Rapach-Strauss-Zhou
(2010) combination forecast, the most robust finding in the equity-premium
prediction literature. The prevailing-mean member is kept only as the benchmark.

Metrics (per horizon, on the common OOS window where every member is present):
    R2_OS   Campbell-Thompson out-of-sample R^2 = 1 - MSE(model)/MSE(mean).
            >0 means the model beats the prevailing historical mean.
    RMSE    root mean squared forecast error.
    hit     directional hit-rate (sign of forecast == sign of realized).
    corr    correlation of forecast with realized.

Outputs (read by the app with numpy/pandas only):
    data/forecast_ensemble_oos.csv   per-horizon ensemble forecast + realized
    data/forecast_params.json        current 4 forecasts, member breakdown, metrics
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.models import (
    HORIZONS, MEMBERS, ENSEMBLE_MEMBERS, OOS_START,
)
from forecast.features import PREDICTOR_INFO

_FORECAST_DIR = Path(__file__).resolve().parent
_DATA_DIR = _FORECAST_DIR.parent / "data"

# Shrinkage of the combined model forecast toward the prevailing-mean benchmark.
# SHRINKAGE=0.5 is a textbook 50/50 model-vs-benchmark combination (a stated
# prior, NOT tuned to the data): we only half-trust the models, so we pull every
# forecast halfway back to the historical mean. Because returns are mostly noise,
# this variance reduction sharply improves directional skill and out-of-sample
# accuracy vs the raw model average (Campbell-Thompson 2008; Rapach-Strauss-Zhou
# 2010 combination forecasts). The benchmark itself is unaffected (R²_OS metric).
SHRINKAGE = 0.5


def _shrink(pred: pd.Series, bench: pd.Series, lam: float = SHRINKAGE) -> pd.Series:
    """pull `pred` toward `bench` by (1-lam): bench + lam*(pred - bench)."""
    idx = pred.index
    b = bench.reindex(idx)
    return (b + lam * (pred - b)).rename(pred.name)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def r2_oos(pred: pd.Series, realized: pd.Series, bench: pd.Series) -> float:
    """Campbell-Thompson R^2_OS = 1 - sum (r-pred)^2 / sum (r-bench)^2."""
    idx = pred.dropna().index.intersection(realized.dropna().index).intersection(
        bench.dropna().index)
    if len(idx) < 12:
        return float("nan")
    r = realized.loc[idx].values
    sse_m = float(np.sum((r - pred.loc[idx].values) ** 2))
    sse_b = float(np.sum((r - bench.loc[idx].values) ** 2))
    return 1.0 - sse_m / sse_b if sse_b > 0 else float("nan")


def metrics(pred: pd.Series, realized: pd.Series, bench: pd.Series) -> dict:
    idx = pred.dropna().index.intersection(realized.dropna().index)
    r = realized.loc[idx].values
    p = pred.loc[idx].values
    rmse = float(np.sqrt(np.mean((r - p) ** 2))) if len(idx) else float("nan")
    hit = float(np.mean(np.sign(p) == np.sign(r))) if len(idx) else float("nan")
    corr = float(np.corrcoef(p, r)[0, 1]) if len(idx) > 2 else float("nan")
    return {
        "r2_oos": r2_oos(pred, realized, bench),
        "rmse": rmse,
        "hit_rate": hit,
        "corr": corr,
        "n": int(len(idx)),
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_and_save() -> dict:
    with open(_DATA_DIR / "forecast_current.json") as fh:
        current = json.load(fh)
    as_of = current.get("as_of")

    ens_frames = {}
    horizons_meta = {}
    for h in HORIZONS:
        df = pd.read_csv(_DATA_DIR / f"forecast_members_{h}m.csv",
                         index_col=0, parse_dates=True)
        realized = df["realized"]
        bench = df["mean"]

        # Equal-weight ensemble over the predictive members, on the common
        # window where all of them are present, then shrunk toward the benchmark.
        common = df[ENSEMBLE_MEMBERS].dropna()
        common = common.loc[common.index >= OOS_START]
        ens_raw = common.mean(axis=1).rename("ensemble")
        ens = _shrink(ens_raw, bench).rename("ensemble")

        ens_frames[h] = pd.DataFrame({
            "ensemble": ens,
            "realized": realized.reindex(ens.index),
        })

        # Per-member metrics (raw) + the ensemble's metrics (shrunk).
        member_metrics = {}
        bench_mean_current = current[str(h)]["mean"]
        for m in MEMBERS:
            member_metrics[m] = {
                "label": MEMBERS[m]["label"],
                "current": current[str(h)][m],
                **metrics(df[m], realized, bench),
            }
        ens_metrics = metrics(ens, realized, bench)
        # Current ensemble forecast = shrink the member-average toward the mean.
        members_avg_current = float(np.mean([current[str(h)][m] for m in ENSEMBLE_MEMBERS]))
        ens_current = bench_mean_current + SHRINKAGE * (members_avg_current - bench_mean_current)

        horizons_meta[str(h)] = {
            "horizon_months": h,
            "current_forecast": ens_current,
            "current_forecast_annualized": (1 + ens_current) ** (12 / h) - 1,
            "ensemble_metrics": ens_metrics,
            "members": member_metrics,
            "benchmark_mean": float(bench.dropna().iloc[-1]) if bench.notna().any() else None,
        }

    # Wide ensemble OOS file: one column per horizon (ensemble + realized).
    wide = {}
    for h in HORIZONS:
        wide[f"ens_{h}m"] = ens_frames[h]["ensemble"]
        wide[f"real_{h}m"] = ens_frames[h]["realized"]
    out = pd.DataFrame(wide).sort_index()
    out.to_csv(_DATA_DIR / "forecast_ensemble_oos.csv",
               date_format="%Y-%m-%d", float_format="%.6f")

    summary = {
        "as_of": as_of,
        "horizons": [str(h) for h in HORIZONS],
        "ensemble_members": ENSEMBLE_MEMBERS,
        "method": "equal_weight_combination",
        "shrinkage": SHRINKAGE,
        "predictor_families": sorted({v[1] for v in PREDICTOR_INFO.values()}),
        "oos_start": str(OOS_START.date()),
        "by_horizon": horizons_meta,
    }
    with open(_DATA_DIR / "forecast_params.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _print_report(summary: dict) -> None:
    print(f"\n{'='*72}\n  MARKET FORECAST ENSEMBLE  (as of {summary['as_of']})\n{'='*72}")
    print(f"  {'Horizon':<8}  {'Forecast':>10}  {'Annualized':>11}  "
          f"{'R2_OS':>8}  {'Hit':>6}  {'Corr':>6}  {'n':>5}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*11}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*5}")
    for h in summary["horizons"]:
        m = summary["by_horizon"][h]
        em = m["ensemble_metrics"]
        print(f"  {h+'m':<8}  {m['current_forecast']:>+9.2%}  "
              f"{m['current_forecast_annualized']:>+10.2%}  "
              f"{em['r2_oos']:>+8.3f}  {em['hit_rate']:>5.1%}  "
              f"{em['corr']:>+6.2f}  {em['n']:>5}")

    print(f"\n  Per-member R2_OS (vs prevailing mean):")
    print(f"  {'Member':<18}  " + "  ".join(f"{h+'m':>8}" for h in summary["horizons"]))
    for mk in MEMBERS:
        label = MEMBERS[mk]["label"]
        cells = []
        for h in summary["horizons"]:
            r2 = summary["by_horizon"][h]["members"][mk]["r2_oos"]
            cells.append(f"{r2:>+8.3f}")
        print(f"  {label:<18}  " + "  ".join(cells))


if __name__ == "__main__":
    s = build_and_save()
    _print_report(s)
    print(f"\nWrote data/forecast_ensemble_oos.csv + data/forecast_params.json")
