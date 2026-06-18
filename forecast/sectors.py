"""
forecast/sectors.py — sector-ETF return forecasting (ensemble, fixed split).

Forecasts the forward total return of a SECTOR ETF at four horizons. The ETF
universe is the "Full Investable ETF Universe" on page 22 of
Conditional_Asset_Allocation_v2.pdf: 11 sectors, each with a primary (SPDR) and
a tax-loss-harvest alternate (mostly Vanguard); for each sector we use whichever
of the pair has the LONGER price history.

Frequency
---------
The pipeline runs at MONTHLY or WEEKLY frequency (set by `freq`):
  * monthly : observations month-end, horizons 1/3/6/12 months.
  * weekly  : observations Friday, horizons 4/13/26/52 weeks (~1/3/6/12 months).
Feature windows and the annualization factor scale with the frequency. The
monthly macro panel (data/forecast_features.csv) is forward-filled onto the
weekly grid (each week sees the most recent already-published monthly value).

Methodology
-----------
* Target   : forward h-period TOTAL return of the ETF (adjusted close, so
             dividends are included).
* Split    : TRAIN before 2020-01, TEST from 2020-01 on. Members are refit on the
             full pre-2020 window and frozen; the test period is never used to
             fit or to select.
* Members  : elastic net, random forest, k-NN (multivariate) + the
             Rapach-Strauss-Zhou combination of single-predictor forecasts (rsz).
* Ensemble : members are weighted PROPORTIONAL to their out-of-sample R²_OS,
             measured by EXPANDING-WINDOW walk-forward CV over 2010-2019 (never
             on the test set); members with CV R²_OS<=0 are dropped. If all are
             dropped the ensemble falls back to the prevailing mean.

No-look-ahead discipline
------------------------
* Target ret_h(t) uses only ETF prices strictly AFTER period t.
* Predictors at t use only information available by t: macro predictors are
  pre-lagged (bear/features.py) and forward-filled; ETF own-technicals and
  commodity predictors are shifted one period.
* At the train/test (and each CV) boundary, a training row is kept only if its
  whole forward window realizes BEFORE the cutoff — no future prices leak into
  labels. Standardization uses training means/stds only.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

_FORECAST_DIR = Path(__file__).resolve().parent
_DATA_DIR = _FORECAST_DIR.parent / "data"
_CACHE_DIR = _DATA_DIR / "cache"

TEST_START = pd.Timestamp("2020-01-31")
CV_START = pd.Timestamp("2010-01-31")     # first period of CV predictions
WF_OOS_START = pd.Timestamp("2008-01-31") # first period of walk-forward predictions
ENS_MEMBERS = ["enet", "rf", "knn", "rsz"]

# Frequency specifications. `win` are feature look-back windows IN PERIODS.
FREQS: dict[str, dict] = {
    "M": {
        "resample": "ME", "ppy": 12, "horizons": [1, 3, 6, 12], "unit": "m",
        "min_train": 60, "refit_every": 6, "rsz_min": 24, "cv_min_obs": 12,
        "wf_trail": 60, "wf_min_scored": 24,
        "win": {"mom_long": 12, "mom_skip": 1, "mom_mid": 6, "ma": 10,
                "vol": 12, "vol_mp": 6},
    },
    "W": {
        "resample": "W-FRI", "ppy": 52, "horizons": [4, 13, 26, 52], "unit": "w",
        "min_train": 260, "refit_every": 13, "rsz_min": 100, "cv_min_obs": 52,
        "wf_trail": 260, "wf_min_scored": 104,
        "win": {"mom_long": 52, "mom_skip": 4, "mom_mid": 26, "ma": 40,
                "vol": 52, "vol_mp": 26},
    },
}
_FREQ = FREQS["M"]            # active spec (set by run_sector)

# Page-22 universe: sector -> (primary ETF, TLH alternate).
SECTORS: dict[str, tuple[str, str]] = {
    "Health Care":   ("XLV", "VHT"),
    "Cons. Staples": ("XLP", "VDC"),
    "Utilities":     ("XLU", "VPU"),
    "Energy":        ("XLE", "VDE"),
    "Materials":     ("XLB", "VAW"),
    "Industrials":   ("XLI", "VIS"),
    "Cons. Disc.":   ("XLY", "VCR"),
    "Financials":    ("XLF", "VFH"),
    "Real Estate":   ("XLRE", "VNQ"),
    "Technology":    ("XLK", "VGT"),
    "Semiconductors": ("SMH", "SOXX"),
}

MACRO_PREDICTORS = [
    "ts_10y3m_level", "dgs10_12m_chg", "baa_aaa_spread",
    "indpro_yoy", "unrate_12m_chg", "infl_yoy", "nfci_level",
]

# Commodity-price predictors: crude oil (USO), natural gas (UNG), gold (GC=F),
# copper (HG=F). Momentum + recent change (levels are non-stationary).
COMMODITY_TICKERS = {"oil": "USO", "gas": "UNG", "gold": "GC=F", "copper": "HG=F"}
COMMODITY_SUFFIXES = ["mom_12_1", "ret_1m"]


def _offset(h: int):
    """Calendar offset for h periods at the active frequency."""
    return pd.DateOffset(months=h) if _FREQ["unit"] == "m" else pd.Timedelta(weeks=h)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def fetch_etf_prices(ticker: str) -> pd.Series:
    """Period-end adjusted-close (total-return) price, cached per frequency."""
    safe = ticker.replace("=", "").replace("^", "")
    cache = _CACHE_DIR / f"etf_{safe}_{_FREQ['unit']}.pkl"
    if cache.exists():
        try:
            return pickle.load(open(cache, "rb"))
        except Exception:
            cache.unlink(missing_ok=True)
    import yfinance as yf
    df = yf.download(ticker, start="1990-01-01", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"no price data for {ticker}")
    close = df["Close"].squeeze()
    close.index = pd.to_datetime(close.index)
    px = close.resample(_FREQ["resample"]).last().dropna()
    px.name = ticker
    pickle.dump(px, open(cache, "wb"))
    return px


def choose_etf(primary: str, alternate: str) -> tuple[str, pd.Series]:
    """Return (ticker, price) for whichever of the pair starts earlier."""
    pp = fetch_etf_prices(primary)
    aa = fetch_etf_prices(alternate)
    return (primary, pp) if pp.index[0] <= aa.index[0] else (alternate, aa)


# ---------------------------------------------------------------------------
# Targets + features
# ---------------------------------------------------------------------------

def _col(h: int) -> str:
    return f"ret_{h}{_FREQ['unit']}"


def build_targets(px: pd.Series) -> pd.DataFrame:
    """Forward total returns ret_h(t) = P_{t+h}/P_t - 1 (forward window only)."""
    return pd.DataFrame(
        {_col(h): px.shift(-h) / px - 1.0 for h in _FREQ["horizons"]}, index=px.index)


def build_features(px: pd.Series) -> pd.DataFrame:
    """ETF own-technicals (lagged 1 period) + commodity + forward-filled macro."""
    w = _FREQ["win"]
    f = pd.DataFrame(index=px.index)
    ma = px.rolling(w["ma"], min_periods=max(w["ma"] // 2, 3)).mean()
    mret = px.pct_change()
    f["own_mom_12_1"] = px.shift(w["mom_skip"]) / px.shift(w["mom_long"]) - 1.0
    f["own_mom_6_1"]  = px.shift(w["mom_skip"]) / px.shift(w["mom_mid"]) - 1.0
    f["own_vs_10ma"]  = px / ma - 1.0
    f["own_rvol_12m"] = mret.rolling(w["vol"], min_periods=w["vol_mp"]).std(ddof=1) \
        * np.sqrt(_FREQ["ppy"])
    f["own_ret_1m"]   = mret                                  # 1-period (reversal)
    f = f.shift(1)                                            # publication lag

    f = f.join(build_commodity_features(px.index), how="left")

    macro = pd.read_csv(_DATA_DIR / "forecast_features.csv",
                        index_col=0, parse_dates=True).sort_index()
    have = [c for c in MACRO_PREDICTORS if c in macro.columns]
    # Forward-fill the (already-lagged) monthly macro onto this price grid.
    f = f.join(macro[have].reindex(f.index, method="ffill"), how="left")
    return f


def build_commodity_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Momentum + recent-change per commodity, lagged 1 period, on `index`."""
    w = _FREQ["win"]
    out = pd.DataFrame(index=index)
    for name, tic in COMMODITY_TICKERS.items():
        try:
            px = fetch_etf_prices(tic)
        except Exception as exc:
            print(f"  [warn] {tic} ({name}) unavailable: {exc} — skipping")
            continue
        feat = pd.DataFrame({
            f"{name}_mom_12_1": px.shift(w["mom_skip"]) / px.shift(w["mom_long"]) - 1.0,
            f"{name}_ret_1m":   px.pct_change(),
        })
        out = out.join(feat.shift(1).reindex(index, method="ffill"), how="left")
    return out


def predictor_list(feats: pd.DataFrame) -> list[str]:
    own = ["own_mom_12_1", "own_mom_6_1", "own_vs_10ma", "own_rvol_12m", "own_ret_1m"]
    commodity = [f"{c}_{s}" for c in COMMODITY_TICKERS for s in COMMODITY_SUFFIXES]
    return [c for c in own + commodity + MACRO_PREDICTORS if c in feats.columns]


# ---------------------------------------------------------------------------
# Members
# ---------------------------------------------------------------------------

def _make_estimator(mtype: str):
    if mtype == "enet":
        from sklearn.linear_model import ElasticNetCV
        return ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], n_alphas=40, cv=5,
                            max_iter=10000, random_state=42)
    if mtype == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=400, max_depth=4,
                                     min_samples_leaf=15, max_features="sqrt",
                                     random_state=42, n_jobs=-1)
    if mtype == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_neighbors=20, weights="uniform")
    raise ValueError(mtype)


def _standardize(tr: pd.DataFrame, te: pd.DataFrame):
    mu = tr.mean(); sig = tr.std(ddof=1).replace(0.0, np.nan)
    return (((tr - mu) / sig).fillna(0.0).values,
            ((te - mu) / sig).fillna(0.0).values, mu, sig)


def _fit_multivariate(mtype, Xtr, ytr, Xte, clip):
    est = _make_estimator(mtype)
    est.fit(Xtr, ytr)
    return np.clip(est.predict(Xte), *clip)


def _rsz_combination(feats, y, train_mask, test_idx, predictors, clip):
    """Equal-weight mean of single-predictor OLS forecasts (Rapach-Strauss-Zhou)."""
    cols = []
    for p in predictors:
        x = feats[p]
        m = train_mask & x.notna() & y.notna()
        if m.sum() < _FREQ["rsz_min"] or x.loc[m].std(ddof=1) == 0:
            continue
        mu, sig = x.loc[m].mean(), x.loc[m].std(ddof=1)
        b1, b0 = np.polyfit(((x.loc[m] - mu) / sig).values, y.loc[m].values, 1)
        xt = ((x.reindex(test_idx) - mu) / sig).fillna(0.0).values
        cols.append(pd.Series(b0 + b1 * xt, index=test_idx))
    if not cols:
        return pd.Series(np.nan, index=test_idx)
    return pd.Series(np.clip(pd.concat(cols, axis=1).mean(axis=1), *clip), index=test_idx)


def _member_preds(feats, y, predictors, fit_mask, eval_idx, clip):
    Xtr, Xev, _, _ = _standardize(feats.loc[fit_mask, predictors],
                                  feats.loc[eval_idx, predictors])
    ytr = y.loc[fit_mask].values.astype(float)
    preds = {mt: pd.Series(_fit_multivariate(mt, Xtr, ytr, Xev, clip), index=eval_idx)
             for mt in ["enet", "rf", "knn"]}
    preds["rsz"] = _rsz_combination(feats, y, fit_mask, eval_idx, predictors, clip)
    return preds


def _fit_models(feats, y, predictors, fit_mask, clip):
    Xr = feats.loc[fit_mask, predictors]
    mu = Xr.mean(); sig = Xr.std(ddof=1).replace(0.0, np.nan)
    Xtr = ((Xr - mu) / sig).fillna(0.0).values
    ytr = y.loc[fit_mask].values.astype(float)
    models = {}
    for mt in ["enet", "rf", "knn"]:
        est = _make_estimator(mt); est.fit(Xtr, ytr); models[mt] = est
    return models, mu, sig


def _r2_os(pred: pd.Series, realized: pd.Series, bench: float) -> float:
    d = pd.concat([pred.rename("p"), realized.rename("r")], axis=1).dropna()
    if len(d) < _FREQ["cv_min_obs"]:
        return float("nan")
    sse_m = float(((d["r"] - d["p"]) ** 2).sum())
    sse_b = float(((d["r"] - bench) ** 2).sum())
    return 1.0 - sse_m / sse_b if sse_b > 0 else float("nan")


# ---------------------------------------------------------------------------
# Expanding-window CV (sets ensemble weights, drops weak members)
# ---------------------------------------------------------------------------

def _walk_forward_cv(feats, y, predictors, h):
    idx = feats.index
    window_end = idx + _offset(h)
    feat_ok = feats[predictors].notna().any(axis=1)
    cv_idx = idx[(idx >= CV_START) & (window_end < TEST_START) & feat_ok & y.notna()]

    preds = {m: pd.Series(index=cv_idx, dtype=float) for m in ENS_MEMBERS}
    bench = pd.Series(index=cv_idx, dtype=float)
    cache, last = None, -10**9
    for i, t in enumerate(cv_idx):
        tr = (window_end < t) & y.notna() & feat_ok
        if int(tr.sum()) < _FREQ["min_train"]:
            continue
        ytr = y.loc[tr]
        bench[t] = float(ytr.mean())
        clip = tuple(np.percentile(ytr, [1, 99]))
        if cache is None or (i - last) >= _FREQ["refit_every"]:
            cache = (*_fit_models(feats, y, predictors, tr, clip), clip)
            last = i
        models, mu, sig, clip0 = cache
        x = ((feats.loc[[t], predictors] - mu) / sig).fillna(0.0).values
        for mt in models:
            preds[mt][t] = float(np.clip(models[mt].predict(x)[0], *clip0))
        rsz = _rsz_combination(feats, y, tr, pd.DatetimeIndex([t]), predictors, clip)
        preds["rsz"][t] = float(rsz.iloc[0])
    return preds, bench


def _cv_weights(feats, y, predictors, h):
    preds, bench = _walk_forward_cv(feats, y, predictors, h)
    realized = y.reindex(preds["rsz"].index)

    def _r2(p):
        d = pd.concat([p.rename("p"), realized.rename("r"), bench.rename("b")],
                      axis=1).dropna()
        if len(d) < _FREQ["cv_min_obs"]:
            return float("nan")
        sse_m = float(((d["r"] - d["p"]) ** 2).sum())
        sse_b = float(((d["r"] - d["b"]) ** 2).sum())
        return 1.0 - sse_m / sse_b if sse_b > 0 else float("nan")

    cv_r2 = {m: _r2(preds[m]) for m in ENS_MEMBERS}
    if all(np.isnan(v) for v in cv_r2.values()):
        return None, cv_r2
    survivors = {m: v for m, v in cv_r2.items()
                 if v is not None and not np.isnan(v) and v > 0}
    if not survivors:
        return {}, cv_r2
    tot = sum(survivors.values())
    return {m: survivors[m] / tot for m in survivors}, cv_r2


def _combine(preds: dict, weights, index, bench: float) -> pd.Series:
    if weights is None:                       # CV infeasible -> RSZ only
        return preds["rsz"].rename("ensemble")
    if len(weights) == 0:                     # all dropped -> benchmark mean
        return pd.Series(bench, index=index, name="ensemble")
    return sum(w * preds[m] for m, w in weights.items()).rename("ensemble")


# ---------------------------------------------------------------------------
# Per-horizon train/test
# ---------------------------------------------------------------------------

def run_horizon(feats, targets, predictors, h):
    y = targets[_col(h)]
    idx = feats.index
    feat_ok = feats[predictors].notna().any(axis=1)
    window_end = idx + _offset(h)
    train_mask = (window_end < TEST_START) & y.notna() & feat_ok
    test_idx = idx[(idx >= TEST_START) & feat_ok]

    weights, val_r2 = _cv_weights(feats, y, predictors, h)

    ytr = y.loc[train_mask].values.astype(float)
    clip = tuple(np.percentile(ytr, [1, 99]))
    bench = float(np.mean(ytr))
    preds = {"mean": pd.Series(bench, index=test_idx)}
    preds.update(_member_preds(feats, y, predictors, train_mask, test_idx, clip))
    preds["ensemble"] = _combine(preds, weights, test_idx, bench)

    out = pd.DataFrame(preds)
    out["realized"] = y.reindex(test_idx)

    def _metrics(p):
        d = out[["realized"]].join(p.rename("p")).dropna()
        if len(d) < _FREQ["cv_min_obs"]:
            return {"r2_os": float("nan"), "hit": float("nan"), "n": len(d)}
        r, pv = d["realized"].values, d["p"].values
        return {"r2_os": _r2_os(p, out["realized"], bench),
                "hit": float(np.mean(np.sign(pv) == np.sign(r))), "n": int(len(d))}

    metrics = {m: _metrics(out[m]) for m in ENS_MEMBERS + ["ensemble"]}
    metrics["_n_train"] = int(train_mask.sum())
    metrics["_val_r2"] = {m: (None if (v is None or np.isnan(v)) else round(v, 4))
                          for m, v in val_r2.items()}
    metrics["_weights"] = (None if weights is None else
                           {m: round(w, 3) for m, w in weights.items()})
    cur = _current_forecast(feats, y, predictors, weights)
    return out, metrics, cur


def _current_forecast(feats, y, predictors, weights):
    as_of = feats[predictors].dropna(how="all").index[-1]
    m = y.notna() & feats[predictors].notna().any(axis=1)
    clip = tuple(np.percentile(y.loc[m], [1, 99]))
    bench = float(y.loc[m].mean())
    preds = _member_preds(feats, y, predictors, m, pd.DatetimeIndex([as_of]), clip)
    ens = _combine(preds, weights, pd.DatetimeIndex([as_of]), bench)
    return float(ens.iloc[0]), str(as_of.date())


# ---------------------------------------------------------------------------
# Walk-forward: adaptive ensemble that re-weights members on trailing OOS skill
# ---------------------------------------------------------------------------

def run_horizon_walkforward(feats, targets, predictors, h):
    """
    Expanding-window walk-forward over WF_OOS_START→end. At each period t:
      * members are refit on all data whose forward window realized before t;
      * each member's TRAILING out-of-sample R²_OS (over the last `wf_trail`
        periods of already-realized predictions) sets the weights — members with
        trailing R²_OS<=0 are dropped, survivors weighted ∝ R²_OS. Before enough
        realized history accrues, the ensemble defaults to RSZ.
    This lets the ensemble adapt across regime shifts. Fully no-look-ahead:
    weights at t use only predictions whose targets realized by t (times <= t-h).
    """
    y = targets[_col(h)]
    idx = feats.index
    window_end = idx + _offset(h)
    feat_ok = feats[predictors].notna().any(axis=1)
    pred_idx = idx[(idx >= WF_OOS_START) & feat_ok]

    moos = {m: pd.Series(index=pred_idx, dtype=float) for m in ENS_MEMBERS}
    bench = pd.Series(index=pred_idx, dtype=float)
    cache, last = None, -10**9
    for i, t in enumerate(pred_idx):
        tr = (window_end < t) & y.notna() & feat_ok
        if int(tr.sum()) < _FREQ["min_train"]:
            continue
        ytr = y.loc[tr]
        bench[t] = float(ytr.mean())
        clip = tuple(np.percentile(ytr, [1, 99]))
        if cache is None or (i - last) >= _FREQ["refit_every"]:
            cache = (*_fit_models(feats, y, predictors, tr, clip), clip)
            last = i
        models, mu, sig, clip0 = cache
        x = ((feats.loc[[t], predictors] - mu) / sig).fillna(0.0).values
        for mt in models:
            moos[mt][t] = float(np.clip(models[mt].predict(x)[0], *clip0))
        moos["rsz"][t] = float(
            _rsz_combination(feats, y, tr, pd.DatetimeIndex([t]), predictors, clip).iloc[0])

    realized = y.reindex(pred_idx)
    trail = _offset(_FREQ["wf_trail"])
    min_scored = _FREQ["wf_min_scored"]
    ens = pd.Series(index=pred_idx, dtype=float, name="ensemble")
    wdf = pd.DataFrame(0.0, index=pred_idx, columns=ENS_MEMBERS)
    for t in pred_idx:
        if pd.isna(bench.get(t, np.nan)):
            continue
        sc = pred_idx[(pred_idx + _offset(h) <= t) & (pred_idx >= t - trail)]
        w = {}
        if len(sc) >= min_scored:
            for m in ENS_MEMBERS:
                d = pd.concat([moos[m].reindex(sc).rename("p"),
                               realized.reindex(sc).rename("r"),
                               bench.reindex(sc).rename("b")], axis=1).dropna()
                if len(d) < min_scored:
                    continue
                sse_b = float(((d["r"] - d["b"]) ** 2).sum())
                if sse_b <= 0:
                    continue
                r2 = 1.0 - float(((d["r"] - d["p"]) ** 2).sum()) / sse_b
                if r2 > 0:
                    w[m] = r2
        # only keep survivors whose forecast exists at t
        w = {m: v for m, v in w.items() if pd.notna(moos[m].get(t, np.nan))}
        if not w:
            val = moos["rsz"].get(t, np.nan)
            ens[t] = val if pd.notna(val) else bench[t]
            wdf.loc[t, "rsz"] = 1.0 if pd.notna(val) else 0.0
        else:
            tot = sum(w.values())
            ens[t] = sum((v / tot) * moos[m][t] for m, v in w.items())
            for m, v in w.items():
                wdf.loc[t, m] = v / tot

    out = pd.DataFrame({**moos, "mean": bench, "ensemble": ens, "realized": realized})
    test_mask = (pred_idx >= TEST_START)

    def _eval(p, mask):
        d = pd.concat([p.rename("p"), realized.rename("r"), bench.rename("b")],
                      axis=1).loc[mask].dropna()
        if len(d) < _FREQ["cv_min_obs"]:
            return {"r2_os": float("nan"), "hit": float("nan"), "n": len(d)}
        sse_b = float(((d["r"] - d["b"]) ** 2).sum())
        r2 = 1.0 - float(((d["r"] - d["p"]) ** 2).sum()) / sse_b if sse_b > 0 else float("nan")
        hit = float(np.mean(np.sign(d["p"]) == np.sign(d["r"])))
        return {"r2_os": r2, "hit": hit, "n": int(len(d))}

    metrics = {m: _eval(out[m], test_mask) for m in ENS_MEMBERS + ["ensemble"]}
    metrics["_full"] = _eval(out["ensemble"], pred_idx >= WF_OOS_START)
    metrics["_avg_weights"] = {m: round(float(wdf.loc[test_mask, m].mean()), 3)
                               for m in ENS_MEMBERS}
    metrics["_n_train"] = int(((window_end < TEST_START) & y.notna() & feat_ok).sum())

    live = ens.dropna()
    cur = (float(live.iloc[-1]), str(live.index[-1].date())) if len(live) else (float("nan"), "")
    return out, metrics, cur


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_sector(name: str, freq: str = "M") -> dict:
    global _FREQ
    _FREQ = FREQS[freq]
    unit = _FREQ["unit"]
    hz = _FREQ["horizons"]
    ppy = _FREQ["ppy"]

    primary, alt = SECTORS[name]
    ticker, px = choose_etf(primary, alt)
    fname = "weekly" if unit == "w" else "monthly"
    print(f"\n{'='*72}\n  {name}  —  {ticker}  [{fname}]  "
          f"({px.index[0].date()} → {px.index[-1].date()}, {len(px)} obs)")
    print(f"  primary {primary} vs alternate {alt}: chose longer-history {ticker}\n{'='*72}")

    targets = build_targets(px)
    feats = build_features(px)
    predictors = predictor_list(feats)

    by_h, oos_frames, current = {}, {}, {}
    for h in hz:
        out, metrics, (cur, as_of) = run_horizon(feats, targets, predictors, h)
        by_h[str(h)] = metrics
        oos_frames[h] = out
        current[str(h)] = cur
        current["as_of"] = as_of

    def lab(h):
        return f"{h}{unit}"

    print(f"  Predictors ({len(predictors)}): {predictors}")
    print(f"\n  {'Horizon':<8}{'Ens forecast':>13}{'Test R2_OS':>12}"
          f"{'Test hit':>10}{'n_test':>8}{'n_train':>9}")
    print(f"  {'-'*8}{'-'*13}{'-'*12}{'-'*10}{'-'*8}{'-'*9}")
    for h in hz:
        em = by_h[str(h)]["ensemble"]
        print(f"  {lab(h):<8}{current[str(h)]:>+12.2%}{em['r2_os']:>+12.3f}"
              f"{em['hit']:>9.1%}{em['n']:>8}{by_h[str(h)]['_n_train']:>9}")

    print(f"\n  Per-member Test R2_OS:")
    print(f"  {'Member':<12}" + "".join(f"{lab(h):>9}" for h in hz))
    for mk in ["enet", "rf", "knn", "rsz", "ensemble"]:
        print(f"  {mk:<12}" + "".join(f"{by_h[str(h)][mk]['r2_os']:>+9.3f}" for h in hz))
    print(f"\n  Per-member Test hit-rate:")
    print(f"  {'Member':<12}" + "".join(f"{lab(h):>9}" for h in hz))
    for mk in ["enet", "rf", "knn", "rsz", "ensemble"]:
        print(f"  {mk:<12}" + "".join(f"{by_h[str(h)][mk]['hit']:>8.0%} " for h in hz))

    print(f"\n  Ensemble weights (from 2010-2019 expanding-window CV R²_OS; "
          f"members with R²_OS<=0 dropped):")
    for h in hz:
        w = by_h[str(h)]["_weights"]; vr = by_h[str(h)]["_val_r2"]
        wstr = ("CV infeasible → RSZ-only" if w is None else
                "all members failed → benchmark mean" if not w else
                ", ".join(f"{m} {w[m]:.0%}" for m in w))
        print(f"    {lab(h):<5} weights: {wstr}")
        print(f"          val R²_OS: " + ", ".join(
            f"{m} {('NA' if vr[m] is None else format(vr[m],'+.3f'))}" for m in ENS_MEMBERS))

    sym = ticker
    summary = {
        "sector": name, "ticker": sym, "primary": primary, "alternate": alt,
        "frequency": fname, "history_start": str(px.index[0].date()),
        "test_start": str(TEST_START.date()), "as_of": current.get("as_of"),
        "horizons": [f"{h}{unit}" for h in hz], "predictors": predictors,
        "current_forecast": {str(h): current[str(h)] for h in hz},
        "current_forecast_annualized": {
            str(h): (1 + current[str(h)]) ** (ppy / h) - 1 for h in hz},
        "by_horizon": by_h,
    }
    suffix = f"_{unit}"
    with open(_DATA_DIR / f"sector_{sym}{suffix}_params.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    for h in hz:
        oos_frames[h].to_csv(_DATA_DIR / f"sector_{sym}{suffix}_oos_{h}{unit}.csv",
                             date_format="%Y-%m-%d", float_format="%.6f")
    print(f"\n  Wrote data/sector_{sym}{suffix}_params.json + sector_{sym}{suffix}_oos_*.csv")
    return summary


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:]]
    freq = "M"
    if args and args[-1].lower() in ("w", "weekly", "week"):
        freq = "W"; args = args[:-1]
    elif args and args[-1].lower() in ("m", "monthly", "month"):
        freq = "M"; args = args[:-1]
    name = " ".join(args) if args else "Health Care"
    if name not in SECTORS:
        print(f"Unknown sector '{name}'. Options: {list(SECTORS)}")
        sys.exit(1)
    run_sector(name, freq)
