"""
bear/research_features.py — feature-engineering research for the Bear model.

Builds a broad library of functionals from the long-history raw series
(SPX, DGS10, TB3MS, DFF, UNRATE, SAHM, LEI, NTFS, CAPE, CPI), keeps only those
with real data back to ~1962, then:
  1. univariate AUC + Newey-West HAC screen,
  2. greedy forward selection (path by in-sample AUC),
  3. walk-forward OOS AUC for each prefix → pick the OOS-best model.

Target: P(rolling 12-month forward drawdown > 20%).
Run:  python -m bear.research_features
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from bear.features import apply_publication_lags, _trailing_zscore, _trailing_percentile
from bear.inference import _hac_pvalues

_BEAR_DIR   = Path(__file__).resolve().parent
_DATA_DIR   = _BEAR_DIR.parent / "data"
SAMPLE_START = pd.Timestamp("1962-01-31")   # all pooled features real by here
OOS_START    = pd.Timestamp("1977-01-31")   # 15y initial training
HAC_LAGS     = 12
MAX_FEATURES = 8


def build_library(raw: pd.DataFrame) -> pd.DataFrame:
    """Engineer a wide set of candidate functionals from lag-adjusted raws."""
    p = apply_publication_lags(raw)
    f = pd.DataFrame(index=p.index)
    spx = p["SPX"]
    ret = spx.pct_change()

    # --- Trend / price (SPX, 1871) ---
    f["spx_ret_6m"]       = spx.pct_change(6)
    f["spx_ret_12m"]      = spx.pct_change(12)
    f["spx_mom_12_1"]     = spx.shift(1) / spx.shift(13) - 1
    f["spx_vs_10ma"]      = spx / spx.rolling(10, min_periods=6).mean() - 1
    f["spx_dd_from_high12"] = spx / spx.rolling(12, min_periods=6).max() - 1
    f["spx_rvol_12m"]     = ret.rolling(12, min_periods=6).std() * np.sqrt(12)
    f["spx_accel"]        = spx.pct_change(3) - spx.pct_change(12) / 4.0

    # --- Yield curve (DGS10 1871, TB3MS 1934) ---
    ts = p["DGS10"] - p["TB3MS"]
    f["ts_10y3m_level"]   = ts
    f["ts_10y3m_chg6"]    = ts.diff(6)
    f["ts_10y3m_inv_dummy"] = (ts < 0).astype(float).where(ts.notna())
    f["dgs10_chg12"]      = p["DGS10"].diff(12)
    f["tb3m_chg12"]       = p["TB3MS"].diff(12)

    # --- Inflation (CPI, 1871) — Chen 2009: top bear predictor ---
    infl = p["CPI"].pct_change(12) * 100
    f["infl_yoy"]         = infl
    f["infl_chg12"]       = infl.diff(12)
    f["infl_zscore_120m"] = _trailing_zscore(infl, 120, 60)
    f["real_10y"]         = p["DGS10"] - infl          # real long rate

    # --- Policy (DFF 1954) ---
    f["ffr_6m_chg"]       = p["DFF"].diff(6)
    f["ffr_12m_chg"]      = p["DFF"].diff(12)

    # --- Labor (UNRATE 1948, SAHM 1960) ---
    f["sahm_level"]       = p["SAHMREALTIME"]
    f["sahm_trigger"]     = (p["SAHMREALTIME"] >= 0.5).astype(float).where(p["SAHMREALTIME"].notna())
    f["unrate_12m_chg"]   = p["UNRATE"].diff(12)
    f["unrate_vs_min12"]  = p["UNRATE"] - p["UNRATE"].rolling(12, min_periods=6).min()

    # --- Leading index (LEI 1955) ---
    lei = p["USALOLITOAASTSAM"]
    f["lei_6m_growth"]    = (lei / lei.shift(6)) ** 2 - 1
    f["lei_12m_growth"]   = lei / lei.shift(12) - 1

    # --- Near-term forward spread (NTFS 1961) ---
    f["ntfs_level"]       = p["NTFS"]
    f["ntfs_chg3"]        = p["NTFS"].diff(3)

    # --- Valuation (CAPE 1900) ---
    f["cape_20yr_pct"]    = _trailing_percentile(p["SHILLER_CAPE"], 240, 120)
    f["cape_z_120m"]      = _trailing_zscore(p["SHILLER_CAPE"], 120, 60)

    return f


def _design(X: pd.DataFrame) -> np.ndarray:
    return ((X - X.mean()) / X.std(ddof=1).replace(0.0, np.nan)).fillna(0.0).values


def _is_auc(X: np.ndarray, y: np.ndarray) -> float:
    m = LogisticRegression(C=1e9, solver="lbfgs", max_iter=3000).fit(X, y)
    return roc_auc_score(y, m.predict_proba(X)[:, 1])


def _oos_auc(F: pd.DataFrame, y: pd.Series, cols: list[str]) -> float:
    idx = F.index
    preds, act = [], []
    for t in idx:
        if t < OOS_START:
            continue
        tr = idx < t
        if tr.sum() < 80 or y.loc[tr].sum() < 8:
            continue
        Xtr_raw = F.loc[tr, cols]
        mu = Xtr_raw.mean(); sig = Xtr_raw.std(ddof=1).replace(0.0, np.nan)
        Xtr = ((Xtr_raw - mu) / sig).fillna(0.0).values
        try:
            m = LogisticRegression(C=1e9, solver="lbfgs", max_iter=3000).fit(Xtr, y.loc[tr].values)
        except Exception:
            continue
        xte = ((F.loc[[t], cols] - mu) / sig).fillna(0.0).values
        preds.append(float(m.predict_proba(xte)[0, 1])); act.append(float(y.loc[t]))
    a = np.array(act); pr = np.array(preds)
    if len(a) < 20 or a.sum() == 0 or a.sum() == len(a):
        return float("nan")
    return roc_auc_score(a, pr)


if __name__ == "__main__":
    raw = pd.read_csv(_DATA_DIR / "raw_monthly.csv", index_col=0, parse_dates=True)
    tgt = pd.read_csv(_DATA_DIR / "targets.csv", index_col=0, parse_dates=True)
    lib = build_library(raw)

    y_full = (tgt["mdd_12m"] <= -0.20).astype(float).where(tgt["mdd_12m"].notna())

    # keep features real by SAMPLE_START; sample = SAMPLE_START..target-resolved
    keep = [c for c in lib.columns
            if lib[c].dropna().index.min() <= SAMPLE_START]
    F = lib.loc[lib.index >= SAMPLE_START, keep]
    y = y_full.reindex(F.index)
    rows = F.notna().all(axis=1) & y.notna()
    F, y = F.loc[rows], y.loc[rows]
    yv = y.values
    print(f"Sample: {F.index[0].date()} → {F.index[-1].date()} "
          f"({len(F)} months, {int(yv.sum())} events) | {len(keep)} candidate functionals")

    # --- 1. univariate screen ---
    uni = []
    for c in keep:
        X = _design(F[[c]])
        auc = _is_auc(X, yv)
        Xc = np.column_stack([np.ones(len(yv)), X])
        m = LogisticRegression(C=1e9, solver="lbfgs", max_iter=3000).fit(X, yv)
        pv = _hac_pvalues(Xc, yv, np.concatenate([m.intercept_, m.coef_[0]]), HAC_LAGS)[1]
        uni.append((c, auc, pv))
    uni.sort(key=lambda r: -r[1])
    print("\nUnivariate AUC (top 15):")
    print(f"  {'feature':<22}{'AUC':>7}{'p(HAC)':>9}")
    for c, auc, pv in uni[:15]:
        print(f"  {c:<22}{auc:>7.3f}{pv:>9.3f}")

    # --- 2. greedy forward selection (path by in-sample AUC) ---
    remaining = list(keep); selected = []; path = []
    while remaining and len(selected) < MAX_FEATURES:
        best_c, best_auc = None, -1
        for c in remaining:
            auc = _is_auc(_design(F[selected + [c]]), yv)
            if auc > best_auc:
                best_auc, best_c = auc, c
        selected.append(best_c); remaining.remove(best_c)
        path.append((list(selected), best_auc))

    # --- 3. OOS AUC for each prefix ---
    print("\nForward path (IS) with walk-forward OOS:")
    print(f"  {'k':<3}{'IS AUC':>8}{'OOS AUC':>9}  added")
    best_prefix, best_oos = None, -1
    for cols, is_auc in path:
        oos = _oos_auc(F, y, cols)
        print(f"  {len(cols):<3}{is_auc:>8.3f}{oos:>9.3f}  +{cols[-1]}")
        if oos > best_oos:
            best_oos, best_prefix = oos, list(cols)

    print(f"\nOOS-best model ({len(best_prefix)} factors, OOS AUC {best_oos:.3f}):")
    print(f"  {best_prefix}")
    Xb = _design(F[best_prefix])
    is_b = _is_auc(Xb, yv)
    Xc = np.column_stack([np.ones(len(yv)), Xb])
    m = LogisticRegression(C=1e9, solver="lbfgs", max_iter=3000).fit(Xb, yv)
    pv = _hac_pvalues(Xc, yv, np.concatenate([m.intercept_, m.coef_[0]]), HAC_LAGS)
    print(f"  In-sample AUC {is_b:.3f}")
    print(f"  {'feature':<22}{'coef':>9}{'p(HAC)':>9}")
    for c, co, p in zip(best_prefix, m.coef_[0], pv[1:]):
        star = "*" if p < 0.05 else ("." if p < 0.10 else "")
        print(f"  {c:<22}{co:>+9.3f}{p:>9.3f} {star}")
