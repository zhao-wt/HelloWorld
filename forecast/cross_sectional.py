"""
forecast/cross_sectional.py — 11-sector cross-sectional rank model.

Instead of forecasting any single sector's return (shown to be unforecastable
OOS) or one pair's relative return (regime-fragile), this ranks ALL 11 sectors
against each other each period and forms a long/short portfolio: overweight the
top-ranked sectors, underweight the bottom. Single-name instabilities (like the
XLV-XLE leadership flip) diversify away across the cross-section, and
cross-sectional momentum is the one return signal with robust documented
out-of-sample performance (Jegadeesh-Titman; Moskowitz-Grinblatt sector rotation).

Construction
------------
* Universe : the 11 sectors from page 22 (longer-history ETF of each pair).
* Panel    : for each sector and period, technical characteristics
             (12-1 & 6-1 momentum, trend vs MA, realized vol, 1-period reversal),
             lagged one period, then CROSS-SECTIONALLY z-scored each date (so the
             model sees each sector RELATIVE to the others that date).
* Target   : forward h-period return MINUS the equal-weight average across the 11
             sectors that date (the relative return a long/short book captures).

Members (each emits a score per sector per date; higher = expected outperform)
  mom    pure 12-1 cross-sectional momentum (no fit — the classic baseline)
  rsz    combination of single-characteristic cross-sectional regressions
  enet   pooled elastic net on all characteristics
  ensemble — members combined, weighted by TRAILING rank-IC, dropping members
             whose trailing IC<=0 (the cross-sectional analog of the R²-weight /
             drop-negative rule used elsewhere).

Evaluation (walk-forward, expanding window, strict no-look-ahead)
  rank IC      : cross-sectional correlation of score vs realized relative return,
                 averaged over dates (and % of dates with IC>0).
  long/short   : each date, long the top 3 / short the bottom 3 by score; report
                 the average forward relative return per rebalance and the hit-rate.
Reported for the 2020+ window and the full walk-forward window.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import forecast.sectors as S

_DATA_DIR = S._DATA_DIR
XS_OOS_START = pd.Timestamp("2010-01-31")
FEATS = ["mom_12_1", "mom_6_1", "vs_ma", "rvol", "ret_1"]
MEMBERS = ["mom", "rsz", "enet"]
LS_K = 3                       # long top-K, short bottom-K


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

def load_universe() -> dict[str, pd.Series]:
    """Longer-history ETF price for each of the 11 sectors, on a common index."""
    prices = {}
    for name in S.SECTORS:
        sym, px = S.choose_etf(*S.SECTORS[name])
        prices[name] = px.rename(sym)
    idx = None
    for px in prices.values():
        idx = px.index if idx is None else idx.intersection(px.index)
    return {k: v.loc[idx] for k, v in prices.items()}


def build_panel(prices: dict[str, pd.Series], h: int) -> pd.DataFrame:
    """Tidy panel indexed by (date, sector): lagged characteristics + forward ret."""
    w = S._FREQ["win"]
    frames = []
    for name, px in prices.items():
        f = pd.DataFrame(index=px.index)
        ma = px.rolling(w["ma"], min_periods=max(w["ma"] // 2, 3)).mean()
        mret = px.pct_change()
        # Standard 12-1 / 6-1 momentum: 12- and 6-period returns with NO internal
        # skip — the uniform f.shift(1) below supplies the single-month skip, so
        # the signal uses prices through t-1 (the textbook convention).
        f["mom_12_1"] = px / px.shift(w["mom_long"]) - 1.0
        f["mom_6_1"]  = px / px.shift(w["mom_mid"]) - 1.0
        f["vs_ma"]    = px / ma - 1.0
        f["rvol"]     = mret.rolling(w["vol"], min_periods=w["vol_mp"]).std(ddof=1) \
            * np.sqrt(S._FREQ["ppy"])
        f["ret_1"]    = mret
        f = f.shift(1)                                   # single-period publication lag
        f["fwd"] = px.shift(-h) / px - 1.0               # forward return (target)
        f["sector"] = name
        f.index = pd.to_datetime(f.index)
        f.index.name = "date"
        frames.append(f.reset_index())
    panel = pd.concat(frames, ignore_index=True)
    return panel.set_index(["date", "sector"]).sort_index()


def xs_transform(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally z-score characteristics and demean the forward return."""
    g = panel.groupby(level="date")
    z = pd.DataFrame(index=panel.index)
    for c in FEATS:
        mu = g[c].transform("mean"); sd = g[c].transform("std").replace(0.0, np.nan)
        z[c] = ((panel[c] - mu) / sd)
    z[FEATS] = z[FEATS].fillna(0.0)
    z["rel"] = panel["fwd"] - g["fwd"].transform("mean")     # relative forward return
    return z


# ---------------------------------------------------------------------------
# Walk-forward scoring
# ---------------------------------------------------------------------------

def _pivot(series: pd.Series) -> pd.DataFrame:
    return series.unstack("sector")


def walk_forward(z: pd.DataFrame, h: int):
    """Walk-forward member scores (date x sector) — strict no-look-ahead."""
    dates = z.index.get_level_values("date").unique().sort_values()
    pred_dates = dates[dates >= XS_OOS_START]
    sectors = z.index.get_level_values("sector").unique()
    off = S._offset(h)

    scores = {m: pd.DataFrame(index=pred_dates, columns=sectors, dtype=float)
              for m in MEMBERS}
    enet_cache, last = None, -10**9
    for i, t in enumerate(pred_dates):
        # training rows: forward window realized before t
        tr = z[(z.index.get_level_values("date") + off <= t) & z["rel"].notna()]
        if len(tr) < 200:
            continue
        cur = z.xs(t, level="date")                     # this date's cross-section
        Xc = cur[FEATS].values

        scores["mom"].loc[t, cur.index] = cur["mom_12_1"].values

        # rsz: average of single-characteristic cross-sectional OLS forecasts
        rsz = np.zeros(len(cur))
        for j, c in enumerate(FEATS):
            b1, b0 = np.polyfit(tr[c].values, tr["rel"].values, 1)
            rsz += b0 + b1 * cur[c].values
        scores["rsz"].loc[t, cur.index] = rsz / len(FEATS)

        # enet: pooled elastic net (refit on a cadence)
        if enet_cache is None or (i - last) >= S._FREQ["refit_every"]:
            from sklearn.linear_model import ElasticNetCV
            enet_cache = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], n_alphas=40,
                                      cv=5, max_iter=10000, random_state=42)
            enet_cache.fit(tr[FEATS].values, tr["rel"].values)
            last = i
        scores["enet"].loc[t, cur.index] = enet_cache.predict(Xc)

    realized = _pivot(z["rel"]).reindex(pred_dates)
    return scores, realized


# ---------------------------------------------------------------------------
# Ensemble (trailing-IC weighted, drop negative-IC) + metrics
# ---------------------------------------------------------------------------

def _ic(score_row: pd.Series, real_row: pd.Series) -> float:
    d = pd.concat([score_row, real_row], axis=1).dropna()
    if len(d) < 4 or d.iloc[:, 0].std() == 0 or d.iloc[:, 1].std() == 0:
        return np.nan
    return float(np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1])


def build_ensemble(scores, realized, h):
    """Combine members weighted by trailing rank-IC (drop trailing IC<=0)."""
    dates = realized.index
    trail = S._FREQ["wf_trail"]; min_scored = S._FREQ["wf_min_scored"]
    off = S._offset(h)
    # per-member realized IC time series (only dates whose target realized)
    member_ic = {m: pd.Series({t: _ic(scores[m].loc[t], realized.loc[t])
                               for t in dates}) for m in MEMBERS}
    ens = pd.DataFrame(index=dates, columns=realized.columns, dtype=float)
    wlog = {m: [] for m in MEMBERS}
    # cross-sectionally z-score each member's score so they combine on one scale
    zscore = {m: scores[m].sub(scores[m].mean(axis=1), axis=0)
              .div(scores[m].std(axis=1).replace(0, np.nan), axis=0) for m in MEMBERS}
    for t in dates:
        scored = [u for u in dates if (u + off <= t) and (u >= t - pd.DateOffset(months=trail)
                  if S._FREQ["unit"] == "m" else u >= t - pd.Timedelta(weeks=trail))]
        w = {}
        if len(scored) >= min_scored:
            for m in MEMBERS:
                ics = member_ic[m].reindex(scored).dropna()
                if len(ics) >= min_scored // 2 and ics.mean() > 0:
                    w[m] = ics.mean()
        if not w:
            ens.loc[t] = zscore["mom"].loc[t]            # default: pure momentum
            wlog["mom"].append(1.0); [wlog[m].append(0.0) for m in MEMBERS if m != "mom"]
        else:
            tot = sum(w.values())
            ens.loc[t] = sum((v / tot) * zscore[m].loc[t] for m, v in w.items())
            for m in MEMBERS:
                wlog[m].append(w.get(m, 0.0) / tot)
    avg_w = {m: float(np.mean(wlog[m])) for m in MEMBERS}
    return ens, member_ic, avg_w


def evaluate(score: pd.DataFrame, realized: pd.DataFrame, mask) -> dict:
    """Rank IC + RANK-WEIGHTED dollar-neutral long/short performance.

    Each date every sector is weighted by its score rank (centered), with the
    long weights scaled to sum +1 and short weights to sum −1 — a $1-long/$1-short
    book that uses the WHOLE cross-section (not just the extremes). The portfolio
    relative return is w · realized.
    """
    dates = realized.index[mask & realized.notna().any(axis=1).values]
    ics, ls = [], []
    for t in dates:
        d = pd.concat([score.loc[t].rename("sc"), realized.loc[t].rename("rl")],
                      axis=1).dropna()
        if len(d) < 6 or d["sc"].std() == 0:
            continue
        ics.append(float(np.corrcoef(d["sc"], d["rl"])[0, 1]))
        w = d["sc"].rank() - d["sc"].rank().mean()       # centered ranks
        pos, neg = w[w > 0].sum(), -w[w < 0].sum()
        if pos == 0 or neg == 0:
            continue
        w = w.where(w <= 0, w / pos).where(w >= 0, w / neg)   # longs→+1, shorts→−1
        ls.append(float((w * d["rl"]).sum()))
    ics, ls = np.array(ics), np.array(ls)
    if len(ics) == 0:
        return {"ic": np.nan, "ic_pos": np.nan, "ls_ret": np.nan,
                "ls_hit": np.nan, "ls_ir": np.nan, "n": 0}
    ir = float(ls.mean() / ls.std()) if ls.std() > 0 else np.nan   # per-bet info ratio
    return {"ic": float(ics.mean()), "ic_pos": float((ics > 0).mean()),
            "ls_ret": float(ls.mean()), "ls_hit": float((ls > 0).mean()),
            "ls_ir": ir, "n": int(len(ics))}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(freq: str = "M") -> dict:
    S._FREQ = S.FREQS[freq]
    unit = S._FREQ["unit"]; hz = S._FREQ["horizons"]; ppy = S._FREQ["ppy"]
    fname = "weekly" if unit == "w" else "monthly"

    prices = load_universe()
    syms = {k: v.name for k, v in prices.items()}
    idx0 = list(prices.values())[0].index
    print(f"\n{'='*74}\n  CROSS-SECTIONAL RANK — 11 sectors  [{fname}]")
    print(f"  common history {idx0[0].date()} → {idx0[-1].date()}  ({len(idx0)} obs)")
    print(f"  universe: " + ", ".join(f"{k}:{v}" for k, v in syms.items()) + f"\n{'='*74}")

    by_h = {}
    for h in hz:
        panel = build_panel(prices, h)
        z = xs_transform(panel)
        scores, realized = walk_forward(z, h)
        ens, member_ic, avg_w = build_ensemble(scores, realized, h)
        allscores = {**scores, "ensemble": ens}

        dates = realized.index
        test_mask = np.asarray(dates >= S.TEST_START)
        full_mask = np.ones(len(dates), dtype=bool)
        m_test = {m: evaluate(allscores[m], realized, test_mask) for m in MEMBERS + ["ensemble"]}
        m_full = {m: evaluate(allscores[m], realized, full_mask) for m in MEMBERS + ["ensemble"]}
        by_h[str(h)] = {"test": m_test, "full": m_full, "avg_w": avg_w}

    def lab(h):
        return f"{h}{unit}"

    # ---- Report ----
    print(f"  Long/short = top {LS_K} − bottom {LS_K} by rank.  IC = cross-sectional "
          f"corr(score, realized relative return).\n")
    for win, key in [("2020+ (out-of-sample test)", "test"),
                     (f"Full walk-forward ({XS_OOS_START.year}+)", "full")]:
        print(f"  === {win} ===")
        print(f"  {'Member':<10}" + "".join(f"{lab(h):>20}" for h in hz))
        print(f"  {'':<10}" + "".join(f"{'IC   L/S(ann) hit':>20}" for h in hz))
        for m in MEMBERS + ["ensemble"]:
            cells = ""
            for h in hz:
                r = by_h[str(h)][key][m]
                ann = r["ls_ret"] * (ppy / h) if not np.isnan(r["ls_ret"]) else np.nan
                cells += f"{r['ic']:>+6.3f}{ann:>+8.1%}{r['ls_hit']:>6.0%}"
            print(f"  {m:<10}{cells}")
        print()

    print(f"  Rank-weighted L/S info ratio (per-bet mean/std; full walk-forward):")
    print(f"  {'Member':<10}" + "".join(f"{lab(h):>9}" for h in hz))
    for m in MEMBERS + ["ensemble"]:
        print(f"  {m:<10}" + "".join(
            f"{by_h[str(h)]['full'][m]['ls_ir']:>9.2f}" for h in hz))

    print(f"\n  Avg ensemble member weights (2020+):")
    print(f"  {'Member':<10}" + "".join(f"{lab(h):>9}" for h in hz))
    for m in MEMBERS:
        print(f"  {m:<10}" + "".join(f"{by_h[str(h)]['avg_w'][m]:>9.0%}" for h in hz))

    summary = {"model": "cross_sectional_rank", "frequency": fname,
               "universe": syms, "common_start": str(idx0[0].date()),
               "oos_start": str(XS_OOS_START.date()), "long_short_k": LS_K,
               "horizons": [f"{h}{unit}" for h in hz], "by_horizon": by_h}
    suffix = f"_{unit}"
    with open(_DATA_DIR / f"xsection{suffix}_params.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\n  Wrote data/xsection{suffix}_params.json")
    return summary


if __name__ == "__main__":
    import sys
    freq = "W" if any(a.lower() in ("w", "weekly") for a in sys.argv[1:]) else "M"
    run(freq)
