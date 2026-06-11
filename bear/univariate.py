"""
bear/univariate.py — univariate bear-model leaderboard.

For every candidate factor (the engineered features behind the dashboard
indicators, including combinations such as VIX/VIX3M, 10y-3m, price-vs-MA), fit
a single-factor unconstrained logistic model of the bear target
    1{ mdd_12m <= -0.20 }   (>20% drawdown over the next 12 months)
and report, at the latest reading: raw value, z-score, Newey-West HAC p-value,
the model's current bear probability, the direction of its current push, and
in-sample AUC.

Output is written to data/univariate_bear.csv so the dashboard can render the
table without any ML libraries at runtime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bear.inference import (
    _DATA_DIR,
    _apply_target_transform,
    _fit_constrained_core,
    _hac_pvalues,
    _sigmoid,
)

HORIZON = 12

# Candidate factor -> (label, category). Covers the dashboard's raw + derived
# indicators and the requested combinations (ratios, spreads, deviations).
FACTORS: dict[str, tuple[str, str]] = {
    # Yield curve / term structure
    "ntfs_level":         ("Near-term forward spread, level",            "Yield curve"),
    "ntfs_3m_chg":        ("Near-term forward spread, 3m change",        "Yield curve"),
    "ts_10y3m":           ("10y-3m term spread (native, 1982+)",         "Yield curve"),
    "ts_inv_dummy":       ("Yield-curve inversion 10y-3m<0 (native)",    "Yield curve"),
    "ts_10y2y":           ("10y-2y term spread",                         "Yield curve"),
    "ts_10y3m_level":     ("10y-3m term spread (long history)",          "Yield curve"),
    "ts_10y3m_inv_dummy": ("Yield-curve inversion 10y-3m<0 (long)",      "Yield curve"),
    "dgs10_12m_chg":      ("10y Treasury yield, 12m change",             "Rates"),
    # Credit
    "ebp_level":          ("Excess bond premium, level",                 "Credit"),
    "ebp_3m_chg":         ("Excess bond premium, 3m change",             "Credit"),
    "baa_level":          ("BAA-10y default spread, level",              "Credit"),
    "baa_3m_chg":         ("BAA-10y spread, 3m change",                  "Credit"),
    "baa_zscore_60m":     ("BAA-10y spread, 5yr z-score",                "Credit"),
    "baa_zscore_24m":     ("BAA-10y spread, 2yr z-score",                "Credit"),
    "baa_aaa_spread":     ("BAA-AAA quality spread, level",              "Credit"),
    "baa_aaa_chg6":       ("BAA-AAA quality spread, 6m change",          "Credit"),
    "baa_aaa_z24":        ("BAA-AAA quality spread, 2yr z-score",        "Credit"),
    "baa_aaa_z60":        ("BAA-AAA quality spread, 5yr z-score",        "Credit"),
    "baa_10y_spread":     ("BAA-10y credit spread, level",               "Credit"),
    "baa_10y_z24":        ("BAA-10y credit spread, 2yr z-score",         "Credit"),
    "baa_10y_z60":        ("BAA-10y credit spread, 5yr z-score",         "Credit"),
    "baa_yield_chg6":     ("BAA corporate yield, 6m change",             "Credit"),
    # Real economy
    "indpro_yoy":         ("Industrial production, YoY %",               "Real economy"),
    "indpro_6m_growth":   ("Industrial production, 6m ann. growth",      "Real economy"),
    # Financial conditions
    "nfci_level":         ("NFCI financial conditions, level",           "Fin. conditions"),
    "nfci_3m_chg":        ("NFCI, 3m change",                            "Fin. conditions"),
    "anfci_level":        ("ANFCI adj. financial conditions, level",     "Fin. conditions"),
    "anfci_3m_chg":       ("ANFCI, 3m change",                           "Fin. conditions"),
    # Volatility (incl. VIX/VIX3M combinations)
    "vix_level":          ("VIX, level",                                 "Volatility"),
    "vix_zscore_24m":     ("VIX, 2yr z-score",                           "Volatility"),
    "vts_slope":          ("VIX term structure slope (VIX3M-VIX)",       "Volatility"),
    "vts_ratio":          ("VIX / VIX3M ratio",                          "Volatility"),
    "vts_backwardation":  ("VIX>VIX3M backwardation (dummy)",            "Volatility"),
    "vts_slope_zscore":   ("VIX term structure slope, 2yr z-score",      "Volatility"),
    # Labor
    "sahm_level":         ("Sahm rule (real-time)",                      "Labor"),
    "sahm_trigger":       ("Sahm trigger >=0.5 (dummy)",                 "Labor"),
    "icsa_yoy_pct":       ("Initial jobless claims, YoY %",              "Labor"),
    "unrate_12m_chg":     ("Unemployment rate, 12m change",              "Labor"),
    "unrate_sahm":        ("Unemployment Sahm gap (3m avg - 12m min)",   "Labor"),
    # Leading / policy
    "lei_6m_growth":      ("Leading indicator, 6m growth",               "Leading"),
    "lei_stress_dummy":   ("LEI stress <-4% (dummy)",                    "Leading"),
    "ffr_6m_chg":         ("Fed funds rate, 6m change",                  "Policy"),
    # Trend / momentum (incl. price-vs-MA)
    "spx_vs_10ma":        ("S&P 500 vs 10-month MA",                     "Trend"),
    "spx_below_10ma":     ("S&P 500 below 10-month MA (dummy)",          "Trend"),
    "spx_12m_mom":        ("S&P 500 12-month momentum",                  "Trend"),
    "m12_1_mom":          ("12-1 month price momentum",                  "Trend"),
    # Inflation / valuation / sentiment
    "infl_yoy":           ("CPI inflation, YoY %",                       "Inflation"),
    "infl_zscore_120m":   ("CPI inflation, 10yr z-score",                "Inflation"),
    "cape_20yr_pct":      ("Shiller CAPE, 20yr percentile",              "Valuation"),
    "cape_z_120m":        ("Shiller CAPE, 10yr z-score",                 "Valuation"),
    "cpce_low_dummy":     ("Put/call complacency (low CPCE, dummy)",     "Sentiment"),
}


def _load_features() -> pd.DataFrame:
    bf = pd.read_csv(_DATA_DIR / "bear_features.csv", index_col=0, parse_dates=True)
    cf = pd.read_csv(_DATA_DIR / "correction_features.csv", index_col=0, parse_dates=True)
    extra = [c for c in cf.columns if c not in bf.columns]
    return bf.join(cf[extra], how="outer").sort_index()


def build_table(target_col: str = "mdd_12m", transform: str = "exceeds_20",
                horizon: int = 12) -> pd.DataFrame:
    feats = _load_features()
    tg = pd.read_csv(_DATA_DIR / "targets.csv", index_col=0, parse_dates=True)
    y = _apply_target_transform(tg[target_col], transform)

    rows = []
    for col, (label, category) in FACTORS.items():
        if col not in feats.columns:
            continue
        s = feats[col]
        mask = s.notna() & y.notna()
        if mask.sum() < 60:
            continue
        start = s.first_valid_index()           # data-history start of the factor
        X = s.loc[mask].astype(float)
        yy = y.loc[mask].values.astype(float)
        mu, sd = float(X.mean()), float(X.std(ddof=1))
        if sd == 0 or len(set(yy)) < 2:
            continue
        z = ((X - mu) / sd).values
        coef, b = _fit_constrained_core(z.reshape(-1, 1), yy,
                                        np.array([1.0]), 0.0, 1.0, "logistic", True)
        coef0 = float(coef[0])
        # HAC p-value for the single slope
        Xc = np.column_stack([np.ones(len(yy)), z])
        try:
            pv = float(_hac_pvalues(Xc, yy, np.array([b, coef0]), horizon)[1])
        except Exception:
            pv = float("nan")
        auc = float(roc_auc_score(yy, _sigmoid(z * coef0 + b)))

        # Latest reading -> current probability + direction
        last_raw = float(s.dropna().iloc[-1])
        last_z = (last_raw - mu) / sd
        prob = float(_sigmoid(np.array([last_z * coef0 + b]))[0])
        contrib = coef0 * last_z
        direction = "Bearish" if contrib > 0 else "Bullish"
        # Per-factor rule (learned coefficient sign): the call when the factor is
        # ABOVE its historical average. The Z-score column shows where it sits now.
        basis = "Bearish above avg" if coef0 > 0 else "Bullish above avg"

        rows.append({
            "Factor": label,
            "Category": category,
            "Start": start.strftime("%Y-%m"),
            "Raw value": round(last_raw, 4),
            "Z-score": round(last_z, 2),
            "P (HAC)": round(pv, 4),
            "AUC": round(auc, 3),
            "Probability": round(prob, 4),
            "Direction": direction,
            "Basis": basis,
            "n": int(mask.sum()),
            "feature": col,
        })

    df = pd.DataFrame(rows)
    # Organize by category (categories ordered by their strongest factor's AUC),
    # then by AUC within each category.
    cat_max = df.groupby("Category")["AUC"].transform("max")
    df = (df.assign(_catrank=cat_max)
            .sort_values(["_catrank", "Category", "AUC"], ascending=[False, True, False])
            .drop(columns="_catrank")
            .reset_index(drop=True))
    return df


FAMILIES = {
    "bear":       ("univariate_bear.csv",       "mdd_12m", "exceeds_20", 12),
    "correction": ("univariate_correction.csv", "mdd_6m",  "exceeds_10", 6),
}


def build_and_save(family: str = "bear") -> pd.DataFrame:
    out_name, target_col, transform, horizon = FAMILIES[family]
    df = build_table(target_col, transform, horizon)
    df.to_csv(_DATA_DIR / out_name, index=False)
    return df


if __name__ == "__main__":
    import sys
    fams = sys.argv[1:] or ["bear", "correction"]
    pd.set_option("display.width", 180)
    pd.set_option("display.max_rows", 100)
    for fam in fams:
        df = build_and_save(fam)
        print(f"\n===== {fam.upper()} =====")
        print(df.drop(columns=["feature"]).head(12).to_string(index=False))
        print(f"Saved {len(df)} univariate models -> data/{FAMILIES[fam][0]}")
