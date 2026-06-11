"""
bear/search_bear.py — exhaustive factor search for the long-history Bear model.

Operates on the clean, backfilled training set bear_training_data.csv (8
factors, 1960-2026, no NaN). For every factor combination it fits an
UNCONSTRAINED binary logistic (free signs, free weights) for the bear event
(12-month forward drawdown > 20%), ranks by in-sample AUC, then reports
out-of-sample AUC and Newey-West HAC significance for the leaders.

Run:  python -m bear.search_bear
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from bear.inference import _hac_pvalues

_BEAR_DIR   = Path(__file__).resolve().parent
_DATA_DIR   = _BEAR_DIR.parent / "data"
OOS_START   = pd.Timestamp("1975-01-31")
HAC_LAGS    = 12
TOP_N       = 12


def _load() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(_DATA_DIR / "bear_training_data.csv", index_col=0, parse_dates=True)
    feats = [c for c in df.columns if c not in ("target_event", "mdd_12m")]
    return df[feats], df["target_event"]


def _design(X_raw: pd.DataFrame) -> np.ndarray:
    mu  = X_raw.mean()
    sig = X_raw.std(ddof=1).replace(0.0, np.nan)
    return ((X_raw - mu) / sig).fillna(0.0).values


def _fit_auc(X: np.ndarray, y: np.ndarray) -> float:
    m = LogisticRegression(C=1e9, solver="lbfgs", max_iter=3000)
    m.fit(X, y)
    return roc_auc_score(y, m.predict_proba(X)[:, 1])


def _oos_auc(feats_df: pd.DataFrame, y: pd.Series, cols: list[str]) -> float:
    idx = feats_df.index
    preds, actuals = [], []
    for t in idx:
        if t < OOS_START:
            continue
        tr = idx < t
        if tr.sum() < 60 or y.loc[tr].sum() < 8:
            continue
        Xtr_raw = feats_df.loc[tr, cols]
        mu = Xtr_raw.mean(); sig = Xtr_raw.std(ddof=1).replace(0.0, np.nan)
        Xtr = ((Xtr_raw - mu) / sig).fillna(0.0).values
        try:
            m = LogisticRegression(C=1e9, solver="lbfgs", max_iter=3000)
            m.fit(Xtr, y.loc[tr].values)
        except Exception:
            continue
        xte = ((feats_df.loc[[t], cols] - mu) / sig).fillna(0.0).values
        preds.append(float(m.predict_proba(xte)[0, 1]))
        actuals.append(float(y.loc[t]))
    a = np.array(actuals); p = np.array(preds)
    if len(a) < 20 or a.sum() == 0 or a.sum() == len(a):
        return float("nan")
    return roc_auc_score(a, p)


if __name__ == "__main__":
    feats_df, y = _load()
    pool = list(feats_df.columns)
    yv = y.values
    print(f"Sample: {feats_df.index[0].date()} → {feats_df.index[-1].date()} "
          f"({len(y)} months, {int(yv.sum())} bear-event months)")
    print(f"Factors ({len(pool)}): {pool}")

    total = sum(len(list(combinations(pool, k))) for k in range(1, len(pool) + 1))
    print(f"\nSearching all {total} combinations (sizes 1–{len(pool)}, unconstrained)...")

    results = []
    for k in range(1, len(pool) + 1):
        for combo in combinations(pool, k):
            cols = list(combo)
            try:
                auc = _fit_auc(_design(feats_df[cols]), yv)
            except Exception:
                continue
            results.append((auc, k, cols))

    results.sort(key=lambda r: -r[0])
    print(f"\nTop {TOP_N} by in-sample AUC:")
    print(f"  {'Rank':<5}{'IS AUC':>8}{'k':>3}  Factors  | OOS  sig")
    print(f"  {'-'*5}{'-'*8}{'-'*3}  {'-'*64}")
    deep = []
    for rank, (auc, k, cols) in enumerate(results[:TOP_N], 1):
        oos = _oos_auc(feats_df, y, cols)
        Xc = np.column_stack([np.ones(len(yv)), _design(feats_df[cols])])
        m = LogisticRegression(C=1e9, solver="lbfgs", max_iter=3000).fit(_design(feats_df[cols]), yv)
        pvals = _hac_pvalues(Xc, yv, np.concatenate([m.intercept_, m.coef_[0]]), HAC_LAGS)
        n_sig = int((pvals[1:] < 0.05).sum())
        deep.append((auc, oos, n_sig, k, cols))
        print(f"  {rank:<5}{auc:>8.4f}{k:>3}  {cols}  | OOS={oos:.3f} sig={n_sig}/{k}")

    best = deep[0]
    print(f"\nBest by in-sample AUC: {best[4]}")
    print(f"  In-sample AUC={best[0]:.4f}  OOS AUC={best[1]:.3f}  "
          f"HAC-significant={best[2]}/{best[3]}")
