"""
bear/validation.py — Phase 6: model validation.

Validates both the bear model (Phase 4) and correction model (Phase 5).

Validation checks
-----------------
1. Block-bootstrap 95% CI for OOS AUC
   Resamples in blocks of h months (h=12 bear, h=6 correction) to
   preserve autocorrelation of overlapping targets. Naive SEs on
   overlapping monthly observations understate uncertainty by ~sqrt(h).

2. Non-overlapping subsample robustness
   Every 12th / 6th month gives approximately independent observations.
   Compares non-overlapping AUC vs full-sample OOS AUC — a large gap
   signals look-ahead or target autocorrelation artefacts.

3. Calibration analysis
   Actual event rate vs mean predicted probability in equal-frequency
   decile bins. Well-calibrated: diagonal (actual ≈ predicted). Also
   computes Expected Calibration Error (ECE).

4. Threshold selection (Youden's J)
   Optimal alert threshold is chosen on TRAINING data (all data before
   OOS start). Reports precision, recall, F1, and confusion matrix at
   the chosen threshold — never peeked at OOS.

5. Coefficient sign stability
   Bear model re-fitted on three non-overlapping sub-periods. Signs
   must be consistent across all three for a feature to be considered
   stable. A sign flip across windows → potential spurious predictor.

6. Cumulative OOS log-loss over time
   Tracks whether predictive power is concentrated in specific episodes
   or distributed across the full OOS window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

_BEAR_DIR = Path(__file__).resolve().parent

# Bear Model 1 features (from Phase 4 exhaustive search winner)
BEAR_FEATURES  = ["ntfs_3m_chg", "ts_inv_dummy", "ebp_3m_chg",
                   "baa_zscore_60m", "lei_6m_growth", "ffr_6m_chg"]
BEAR_EXP_SIGNS = {"ntfs_3m_chg":+1,"ts_inv_dummy":+1,"ebp_3m_chg":+1,
                   "baa_zscore_60m":+1,"lei_6m_growth":-1,"ffr_6m_chg":-1}

CORR_FEATURES  = ["vts_slope","spx_vs_10ma","m12_1_mom",
                   "anfci_3m_chg","cape_20yr_pct","baa_zscore_24m"]

BEAR_OOS_START = "2000-01-31"
CORR_OOS_START = "2010-01-31"


# ---------------------------------------------------------------------------
# 1. Block-bootstrap CI for AUC
# ---------------------------------------------------------------------------

def block_bootstrap_auc(
    y_true:     pd.Series,
    y_pred:     pd.Series,
    block_size: int  = 12,
    n_boot:     int  = 2000,
    alpha:      float = 0.05,
    seed:       int  = 42,
) -> tuple[float, float, float]:
    """
    Circular block bootstrap 95% CI for OOS AUC.

    Blocks of consecutive months preserve the autocorrelation structure
    induced by overlapping h-month forward targets.

    Returns (point_estimate, lower_bound, upper_bound).
    """
    mask  = y_true.notna() & y_pred.notna()
    yt    = y_true.loc[mask].values
    yp    = y_pred.loc[mask].values
    n     = len(yt)

    if n < 20 or yt.sum() == 0:
        return np.nan, np.nan, np.nan

    point = roc_auc_score(yt, yp)

    rng   = np.random.default_rng(seed)
    aucs: list[float] = []

    for _ in range(n_boot):
        # Circular block bootstrap
        n_blocks     = int(np.ceil(n / block_size))
        start_idxs   = rng.integers(0, n, size=n_blocks)
        idxs         = np.concatenate([
            np.arange(s, s + block_size) % n for s in start_idxs
        ])[:n]
        yt_b = yt[idxs];  yp_b = yp[idxs]
        if yt_b.sum() == 0 or yt_b.sum() == n:
            continue
        try:
            aucs.append(roc_auc_score(yt_b, yp_b))
        except Exception:
            continue

    if not aucs:
        return point, np.nan, np.nan

    lo = float(np.percentile(aucs, 100 * alpha / 2))
    hi = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return point, lo, hi


# ---------------------------------------------------------------------------
# 2. Non-overlapping subsample AUC
# ---------------------------------------------------------------------------

def non_overlapping_auc(
    y_true: pd.Series,
    y_pred: pd.Series,
    step:   int = 12,
) -> tuple[float, int, int]:
    """
    AUC on every step-th observation (approximately independent sample).

    Returns (auc, n_obs, n_positives).
    """
    mask  = y_true.notna() & y_pred.notna()
    idx   = y_true.loc[mask].index[::step]
    yt    = y_true.loc[idx].values
    yp    = y_pred.loc[idx].values
    if len(yt) < 5 or yt.sum() == 0:
        return np.nan, len(yt), int(yt.sum())
    return roc_auc_score(yt, yp), len(yt), int(yt.sum())


# ---------------------------------------------------------------------------
# 3. Calibration
# ---------------------------------------------------------------------------

def calibration_table(
    y_true: pd.Series,
    y_pred: pd.Series,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, float]:
    """
    Equal-frequency calibration table and Expected Calibration Error (ECE).

    Each bin contains approximately the same number of observations.
    Well-calibrated model: actual_rate ≈ mean_predicted.

    Returns (table_df, ECE).
    """
    mask = y_true.notna() & y_pred.notna()
    yt   = y_true.loc[mask]
    yp   = y_pred.loc[mask]

    quantiles = np.linspace(0, 100, n_bins + 1)
    edges     = np.unique(np.percentile(yp, quantiles))

    rows: list[dict] = []
    for i in range(len(edges) - 1):
        lo, hi   = edges[i], edges[i + 1]
        in_bin   = (yp >= lo) & (yp <= (hi if i < len(edges) - 2 else hi + 1e-9))
        if in_bin.sum() == 0:
            continue
        mean_pred   = float(yp[in_bin].mean())
        actual_rate = float(yt[in_bin].mean())
        rows.append({
            "Bin range":      f"{lo:.0%}–{hi:.0%}",
            "N":              int(in_bin.sum()),
            "Mean predicted": round(mean_pred,   3),
            "Actual rate":    round(actual_rate, 3),
            "Error":          round(actual_rate - mean_pred, 3),
        })

    df  = pd.DataFrame(rows)
    ece = float((df["N"] * df["Error"].abs()).sum() / df["N"].sum()) if not df.empty else np.nan
    return df, ece


# ---------------------------------------------------------------------------
# 4. Threshold selection (Youden's J on training data)
# ---------------------------------------------------------------------------

def youden_threshold(
    y_train: pd.Series,
    p_train: pd.Series,
) -> float:
    """
    Optimal probability threshold by Youden's J = TPR − FPR.
    Must be computed on TRAINING data only — never touch OOS labels.
    """
    mask        = y_train.notna() & p_train.notna()
    fpr, tpr, thresholds = roc_curve(y_train.loc[mask], p_train.loc[mask])
    j_scores    = tpr - fpr
    best_idx    = int(np.argmax(j_scores))
    return float(thresholds[best_idx])


def threshold_report(
    y_true:    pd.Series,
    y_pred:    pd.Series,
    threshold: float,
    label:     str = "",
) -> pd.DataFrame:
    """
    Precision, recall, F1, and confusion matrix at a given threshold.
    """
    mask = y_true.notna() & y_pred.notna()
    yt   = y_true.loc[mask].values.astype(int)
    yhat = (y_pred.loc[mask].values >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(yt, yhat, labels=[0, 1]).ravel()
    prec  = precision_score(yt, yhat, zero_division=0)
    rec   = recall_score(yt, yhat, zero_division=0)
    f1    = f1_score(yt, yhat, zero_division=0)

    rows = [
        {"Metric": "Threshold",        "Value": f"{threshold:.2%}"},
        {"Metric": "Precision",        "Value": f"{prec:.3f}"},
        {"Metric": "Recall (TPR)",     "Value": f"{rec:.3f}"},
        {"Metric": "F1",               "Value": f"{f1:.3f}"},
        {"Metric": "True Positives",   "Value": str(tp)},
        {"Metric": "False Positives",  "Value": str(fp)},
        {"Metric": "True Negatives",   "Value": str(tn)},
        {"Metric": "False Negatives",  "Value": str(fn)},
        {"Metric": "Hit rate",         "Value": f"{tp / max(tp+fn, 1):.1%}"},
        {"Metric": "False alarm rate", "Value": f"{fp / max(fp+tn, 1):.1%}"},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Coefficient sign stability
# ---------------------------------------------------------------------------

def sign_stability(
    features_df:    pd.DataFrame,
    y:              pd.Series,
    features:       list[str],
    expected_signs: dict[str, int],
    split_dates:    list[str],
) -> pd.DataFrame:
    """
    Fit logistic regression on each sub-period defined by split_dates and
    report whether each coefficient's sign is consistent with expectations.

    split_dates should be a list of boundary dates dividing the full
    training history into N windows, e.g. ['1995-12-31', '2008-12-31'].
    A window spans from the start of available data to split_dates[0],
    split_dates[0] to split_dates[1], etc., and split_dates[-1] to end.
    """
    all_dates = sorted(features_df.index)
    boundaries = [pd.Timestamp(d) for d in split_dates]

    # Build window date ranges
    windows: list[tuple[pd.Timestamp | None, pd.Timestamp | None]] = []
    prev = None
    for b in boundaries:
        windows.append((prev, b))
        prev = b
    windows.append((prev, None))

    records: list[dict] = []
    for label, (start, end) in zip(
        ["Pre-" + split_dates[0][:4]] +
        [f"{split_dates[i][:4]}–{split_dates[i+1][:4]}" for i in range(len(split_dates)-1)] +
        ["Post-" + split_dates[-1][:4]],
        windows,
    ):
        idx  = pd.Series(all_dates)
        if start is not None:
            idx = idx[idx > start]
        if end is not None:
            idx = idx[idx <= end]
        idx = list(idx)

        mask = (features_df.index.isin(idx) &
                features_df[features].notna().all(axis=1) &
                y.notna())
        Xw = features_df.loc[mask, features]
        yw = y.loc[mask]

        if len(yw) < 20 or yw.sum() < 3:
            rec = {"Window": label, "N": len(yw)}
            for f in features:
                rec[f] = "insuf. data"
            records.append(rec)
            continue

        mu  = Xw.mean();  sig = Xw.std(ddof=1).replace(0.0, np.nan)
        Xsc = ((Xw - mu) / sig).fillna(0.0)

        try:
            m = LogisticRegression(C=1.0, class_weight="balanced",
                                   solver="lbfgs", max_iter=1000, random_state=42)
            m.fit(Xsc.values, yw.values)
            coef_map = dict(zip(features, m.coef_[0]))
        except Exception:
            rec = {"Window": label, "N": len(yw)}
            for f in features:
                rec[f] = "fit failed"
            records.append(rec)
            continue

        rec = {"Window": label, "N": int(mask.sum())}
        for f in features:
            c      = coef_map[f]
            exp_s  = expected_signs.get(f, 0)
            act_s  = int(np.sign(c))
            ok     = "✓" if act_s == exp_s else "✗"
            rec[f] = f"{ok} {c:+.3f}"
        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 6. Cumulative OOS log-loss
# ---------------------------------------------------------------------------

def cumulative_log_loss(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> pd.Series:
    """
    Cumulative sum of per-observation log-loss over time.
    A flat line = steady predictive performance.
    A sudden jump = model struggled during a specific episode.
    """
    mask  = y_true.notna() & y_pred.notna()
    yt    = y_true.loc[mask]
    yp    = y_pred.loc[mask].clip(1e-7, 1 - 1e-7)
    ll    = -(yt * np.log(yp) + (1 - yt) * np.log(1 - yp))
    return ll.cumsum().rename("cumulative_log_loss")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print(f"\n{'='*70}")
    print("  Phase 6 — Validation Report")
    print(f"{'='*70}")

    # -- Load OOS probability series --
    bear_out = pd.read_csv(_BEAR_DIR / "bear_model_output.csv",
                           index_col=0, parse_dates=True)
    corr_out = pd.read_csv(_BEAR_DIR / "correction_model_output.csv",
                           index_col=0, parse_dates=True)
    bear_f   = pd.read_csv(_BEAR_DIR / "bear_features.csv",
                           index_col=0, parse_dates=True)
    corr_f   = pd.read_csv(_BEAR_DIR / "correction_features.csv",
                           index_col=0, parse_dates=True)
    targets  = pd.read_csv(_BEAR_DIR / "targets.csv",
                           index_col=0, parse_dates=True)

    y_bear   = targets["y_bear"]
    y_corr   = targets["y_corr"]

    # Use the Phase 4 exhaustive-search winner (prob_bear_selected_oos)
    p_bear   = bear_out["prob_bear_selected_oos"].dropna()
    p_corr   = corr_out["prob_corr_oos"].dropna()

    oos_bear_start = pd.Timestamp(BEAR_OOS_START)
    oos_corr_start = pd.Timestamp(CORR_OOS_START)

    # Training-period predictions (IS, before OOS start) — used for threshold selection
    p_bear_train = bear_out["prob_bear_is"].loc[bear_out.index < oos_bear_start].dropna()
    y_bear_train = y_bear.loc[y_bear.index < oos_bear_start]

    p_corr_train = corr_out["prob_corr_oos"].loc[corr_out.index < oos_corr_start].dropna()
    y_corr_train = y_corr.loc[y_corr.index < oos_corr_start]

    # ================================================================
    # Bear model validation
    # ================================================================

    print(f"\n{'='*70}")
    print(f"  BEAR MODEL  (target: >20 % drawdown / 12 months)")
    print(f"  OOS window: {BEAR_OOS_START} to present")
    print(f"{'='*70}")

    # -- 1. Block-bootstrap CI --
    print(f"\n  [1] Block-bootstrap 95% CI  (block_size=12, n_boot=2000)")
    pt, lo, hi = block_bootstrap_auc(y_bear, p_bear, block_size=12, n_boot=2000)
    print(f"      OOS AUC = {pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    # -- 2. Non-overlapping subsample --
    print(f"\n  [2] Non-overlapping subsample  (every 12th month ≈ independent)")
    auc_nol, n_nol, n_pos_nol = non_overlapping_auc(y_bear, p_bear, step=12)
    print(f"      OOS AUC (full)          = {pt:.4f}  (N={p_bear.notna().sum()})")
    print(f"      OOS AUC (non-overlap)  = {auc_nol:.4f}  (N={n_nol}, positives={n_pos_nol})")
    gap = pt - auc_nol
    flag = "  *** LARGE GAP — possible overfit/autocorrelation ***" if abs(gap) > 0.05 else ""
    print(f"      Δ AUC = {gap:+.4f}{flag}")

    # -- 3. Calibration --
    print(f"\n  [3] Calibration  (equal-frequency deciles)")
    cal_df, ece = calibration_table(y_bear, p_bear, n_bins=10)
    print(f"      Expected Calibration Error (ECE) = {ece:.4f}")
    print(f"\n      {'Bin range':<14}  {'N':>5}  {'Predicted':>10}  {'Actual':>10}  {'Error':>8}")
    print(f"      {'-'*14}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*8}")
    for _, row in cal_df.iterrows():
        bar = "█" * int(abs(row["Error"]) * 50)
        sign = "+" if row["Error"] > 0 else "-"
        print(f"      {row['Bin range']:<14}  {row['N']:>5}  "
              f"{row['Mean predicted']:>10.1%}  {row['Actual rate']:>10.1%}  "
              f"{row['Error']:>+8.3f}  {sign}{bar}")

    # -- 4. Threshold selection (Youden on training) --
    print(f"\n  [4] Threshold selection  (Youden's J on training data)")
    if p_bear_train.notna().sum() > 10:
        thresh_bear = youden_threshold(y_bear_train, p_bear_train)
    else:
        thresh_bear = 0.30   # fallback
    print(f"      Optimal threshold (training) = {thresh_bear:.1%}")
    t_df = threshold_report(y_bear, p_bear, thresh_bear, "Bear OOS")
    for _, row in t_df.iterrows():
        print(f"      {row['Metric']:<22}  {row['Value']}")

    # -- 5. Coefficient sign stability --
    print(f"\n  [5] Coefficient sign stability  (sub-period re-fits)")
    stab_bear = sign_stability(
        bear_f, y_bear, BEAR_FEATURES, BEAR_EXP_SIGNS,
        split_dates=["2008-12-31", "2015-12-31"],
    )
    cols = ["Window", "N"] + BEAR_FEATURES
    avail_cols = [c for c in cols if c in stab_bear.columns]
    print(f"\n      {stab_bear[avail_cols].to_string(index=False)}")
    # Count consistent features
    feat_cols = [c for c in avail_cols if c not in ("Window", "N")]
    if len(stab_bear) > 0:
        for feat in feat_cols:
            signs = [str(v)[0] for v in stab_bear[feat] if isinstance(v, str) and v[0] in ("✓","✗")]
            all_ok = all(s == "✓" for s in signs)
            status = "STABLE" if all_ok else "UNSTABLE"
            print(f"      {feat:<22}  {status}")

    # -- 6. Cumulative log-loss --
    cum_ll_bear = cumulative_log_loss(y_bear, p_bear)
    cum_ll_valid = cum_ll_bear.dropna()
    print(f"\n  [6] Cumulative log-loss  (steady = stable; jump = struggled episode)")
    print(f"      Total OOS log-loss = {float(cum_ll_valid.iloc[-1]):.3f}  "
          f"({len(cum_ll_valid)} months)")
    # Flag the 3 months with largest single-period loss
    single_ll = cum_ll_valid.diff().fillna(cum_ll_valid.iloc[0])
    top3 = single_ll.nlargest(3)
    print(f"      Worst 3 months (highest per-month loss):")
    for dt, val in top3.items():
        yb = y_bear.get(dt, np.nan)
        pp = p_bear.get(dt, np.nan)
        print(f"        {str(dt.date()):>12}  loss={val:.3f}  "
              f"y_bear={int(yb) if pd.notna(yb) else 'NaN'}  "
              f"prob={pp:.1%}" if pd.notna(pp) else f"prob=NaN")

    # ================================================================
    # Correction model validation
    # ================================================================

    print(f"\n\n{'='*70}")
    print(f"  CORRECTION MODEL  (target: 10–20 % drawdown / 6 months)")
    print(f"  OOS window: {CORR_OOS_START} to present")
    print(f"{'='*70}")

    # -- 1. Block-bootstrap CI --
    print(f"\n  [1] Block-bootstrap 95% CI  (block_size=6, n_boot=2000)")
    pt_c, lo_c, hi_c = block_bootstrap_auc(y_corr, p_corr, block_size=6, n_boot=2000)
    print(f"      OOS AUC = {pt_c:.4f}  95% CI [{lo_c:.4f}, {hi_c:.4f}]")

    # -- 2. Non-overlapping subsample --
    print(f"\n  [2] Non-overlapping subsample  (every 6th month ≈ independent)")
    auc_nol_c, n_nol_c, n_pos_nol_c = non_overlapping_auc(y_corr, p_corr, step=6)
    print(f"      OOS AUC (full)         = {pt_c:.4f}  (N={p_corr.notna().sum()})")
    print(f"      OOS AUC (non-overlap)  = {auc_nol_c:.4f}  "
          f"(N={n_nol_c}, positives={n_pos_nol_c})")
    gap_c = pt_c - auc_nol_c
    flag_c = "  *** LARGE GAP ***" if abs(gap_c) > 0.05 else ""
    print(f"      Δ AUC = {gap_c:+.4f}{flag_c}")

    # -- 3. Calibration --
    print(f"\n  [3] Calibration  (equal-frequency deciles)")
    cal_df_c, ece_c = calibration_table(y_corr, p_corr, n_bins=10)
    print(f"      Expected Calibration Error (ECE) = {ece_c:.4f}")
    print(f"\n      {'Bin range':<14}  {'N':>5}  {'Predicted':>10}  {'Actual':>10}  {'Error':>8}")
    print(f"      {'-'*14}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*8}")
    for _, row in cal_df_c.iterrows():
        bar  = "█" * int(abs(row["Error"]) * 50)
        sign = "+" if row["Error"] > 0 else "-"
        print(f"      {row['Bin range']:<14}  {row['N']:>5}  "
              f"{row['Mean predicted']:>10.1%}  {row['Actual rate']:>10.1%}  "
              f"{row['Error']:>+8.3f}  {sign}{bar}")

    # -- 4. Threshold selection --
    print(f"\n  [4] Threshold selection  (Youden's J on training data)")
    if p_corr_train.notna().sum() > 10:
        thresh_corr = youden_threshold(y_corr_train, p_corr_train)
    else:
        thresh_corr = 0.35  # fallback
    print(f"      Optimal threshold (training) = {thresh_corr:.1%}")
    t_df_c = threshold_report(y_corr, p_corr, thresh_corr, "Correction OOS")
    for _, row in t_df_c.iterrows():
        print(f"      {row['Metric']:<22}  {row['Value']}")

    # -- 5. Coefficient sign stability --
    print(f"\n  [5] Coefficient sign stability")
    stab_corr = sign_stability(
        corr_f, y_corr, CORR_FEATURES, {},   # no strict expected signs for correction
        split_dates=["2014-12-31", "2019-12-31"],
    )
    avail_corr = [c for c in ["Window", "N"] + CORR_FEATURES if c in stab_corr.columns]
    print(f"\n      {stab_corr[avail_corr].to_string(index=False)}")

    # ================================================================
    # Summary comparison
    # ================================================================

    print(f"\n\n{'='*70}")
    print("  Summary — Model Comparison")
    print(f"{'='*70}\n")

    print(f"  {'Metric':<35}  {'Bear':>10}  {'Correction':>12}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*12}")
    rows_summary = [
        ("OOS AUC  (full overlap)",       f"{pt:.4f}",       f"{pt_c:.4f}"),
        ("OOS AUC  95% CI",               f"[{lo:.3f},{hi:.3f}]", f"[{lo_c:.3f},{hi_c:.3f}]"),
        ("OOS AUC  (non-overlapping)",     f"{auc_nol:.4f}",  f"{auc_nol_c:.4f}"),
        ("Δ AUC  (overlap − non-overlap)", f"{pt-auc_nol:+.4f}", f"{pt_c-auc_nol_c:+.4f}"),
        ("ECE  (calibration error)",       f"{ece:.4f}",      f"{ece_c:.4f}"),
        ("Alert threshold  (Youden)",      f"{thresh_bear:.1%}", f"{thresh_corr:.1%}"),
        ("OOS horizon",                    "12 months",       "6 months"),
        ("HAC max_lags",                   "12",              "6"),
        ("OOS start",                      BEAR_OOS_START,    CORR_OOS_START),
    ]
    for label, bear_val, corr_val in rows_summary:
        print(f"  {label:<35}  {bear_val:>10}  {corr_val:>12}")
