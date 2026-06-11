"""
bear/ensemble.py — Phase E4: combine the four era-trained members into one
bear-market ensemble.

Members (built in inference.py, fitted by fit_and_save):
    modela  1920s   Trend + Valuation + Credit                 OOS 1950+
    modelb  1950s   Trend + Inflation + Valuation + Rates       OOS 1970+
    modelc  1960s   + yield-curve inversion                     OOS 1985+
    modeld  1980s   Policy + Inflation + Credit (modern)        OOS 2005+

Each member is an "era expert": trained on the longest history its feature set
permits, and out-of-sample from its own oos_start. The ensemble lets every
member that is already out-of-sample at month t cast a vote.

Two probability series per member:
  * in-sample `history`     — full-fit params applied from train_start (used for
                              the production / current ensemble reading).
  * walk-forward `oos`      — honest expanding-window predictions (used for all
                              evaluation; no look-ahead).

Combination methods compared (out-of-sample):
  * equal      — simple mean of the available members' probabilities.
  * auc        — weight each member by its (common-window) OOS AUC skill.
  * stacked    — walk-forward logistic meta-model on the members' OOS probs.

The default production ensemble is the EQUAL-WEIGHT mean: it needs no extra
estimation (no overfitting), is robust to the members' very different sample
lengths, and matches the panel-of-experts intuition.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from bear.inference import (
    _DATA_DIR,
    _SPECS,
    _apply_target_transform,
    _oos_path,
    _sigmoid,
    load_assessment,
)

COMMON_START = pd.Timestamp("2005-01-31")   # first month all four are out-of-sample

# Ensemble families. Each is four era-trained members sharing a target.
FAMILIES = {
    "bear": {
        "members": ["modela", "modelb", "modelc", "modeld"],
        "target_col": "mdd_12m", "transform": "exceeds_20",
        "params_file": "ensemble_params.json", "oos_file": "ensemble_oos.csv",
    },
    "correction": {
        "members": ["corra", "corrb", "corrc", "corrd"],
        "target_col": "mdd_6m", "transform": "exceeds_10",
        "params_file": "correction_ensemble_params.json",
        "oos_file": "correction_ensemble_oos.csv",
    },
}

MEMBERS = FAMILIES["bear"]["members"]   # back-compat default (bear)


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _target(family: str = "bear") -> pd.Series:
    cfg = FAMILIES[family]
    tg = pd.read_csv(_DATA_DIR / "targets.csv", index_col=0, parse_dates=True)
    return _apply_target_transform(tg[cfg["target_col"]], cfg["transform"])


def member_oos(family: str = "bear") -> pd.DataFrame:
    """Walk-forward OOS probability of each member, aligned on a union index."""
    cols = {}
    for k in FAMILIES[family]["members"]:
        s = pd.read_csv(_oos_path(k), index_col=0, parse_dates=True).iloc[:, 0]
        cols[k] = s
    return pd.DataFrame(cols).sort_index()


def member_history(family: str = "bear") -> pd.DataFrame:
    """In-sample fitted probability of each member (for the production series)."""
    cols = {k: load_assessment(k)["history"] for k in FAMILIES[family]["members"]}
    return pd.DataFrame(cols).sort_index()


def _auc(y: pd.Series, p: pd.Series) -> float:
    from sklearn.metrics import roc_auc_score
    idx = y.dropna().index.intersection(p.dropna().index)
    yy = y.loc[idx]
    if yy.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(yy.values, p.loc[idx].values))


# ---------------------------------------------------------------------------
# Combination methods
# ---------------------------------------------------------------------------

def ensemble_equal(probs: pd.DataFrame) -> pd.Series:
    """Mean of available members at each date (growing membership over time)."""
    return probs.mean(axis=1, skipna=True).rename("ensemble")


def ensemble_auc_weighted(probs: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Skill-weighted mean; weights renormalized over the members present each row."""
    w = pd.Series(weights)
    num = (probs * w).sum(axis=1, skipna=True)
    den = probs.notna().mul(w, axis=1).sum(axis=1)
    return (num / den.replace(0.0, np.nan)).rename("ensemble_auc")


def ensemble_stacked_walkforward(
    oos: pd.DataFrame, y: pd.Series, start: pd.Timestamp, init_years: int = 7
) -> pd.Series:
    """
    Walk-forward logistic meta-model on the four members' OOS probabilities.
    At each month t >= start+init_years, fit logit(y) ~ members on rows < t
    (where all members and y are present), predict t. No look-ahead.
    """
    from sklearn.linear_model import LogisticRegression

    rows = oos.dropna()                       # all four present
    idx = rows.index[rows.index >= start]
    out = pd.Series(np.nan, index=oos.index, name="ensemble_stacked")
    meta_start = start + pd.DateOffset(years=init_years)
    for t in idx:
        if t < meta_start:
            continue
        tr = rows.index[(rows.index < t)]
        yy = y.reindex(tr).dropna()
        tr = yy.index
        if len(tr) < 24 or yy.sum() < 4:
            continue
        m = LogisticRegression(C=1.0, max_iter=1000)
        m.fit(rows.loc[tr].values, yy.values)
        out.loc[t] = float(m.predict_proba(rows.loc[[t]].values)[0, 1])
    return out.dropna()


# ---------------------------------------------------------------------------
# Calibration (Platt scaling of the equal-weight ensemble)
# ---------------------------------------------------------------------------
# Averaging member probabilities is under-confident (it compresses toward the
# middle): the Cox calibration slope of the raw ensemble is ~1.3-1.8 (>1). A
# Platt map  p_cal = sigmoid(a + b*logit(p))  fit on realized OOS history
# restores calibration (slope ~1.1) without changing the ranking (AUC fixed).

def fit_global_platt(ens_oos: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Fit logit(y) ~ a + b*logit(ensemble) on all realized OOS history."""
    from sklearn.linear_model import LogisticRegression
    idx = ens_oos.dropna().index.intersection(y.dropna().index)
    X = _logit(ens_oos.loc[idx].values).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(X, y.loc[idx].values.astype(float))
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def apply_platt(p, a: float, b: float):
    return _sigmoid(a + b * _logit(p))


def platt_walkforward(p: pd.Series, y: pd.Series, start: pd.Timestamp,
                      init_years: int = 7) -> pd.Series:
    """Expanding-window Platt scaling: logit(y)~logit(p) fit only on rows < t."""
    from sklearn.linear_model import LogisticRegression
    out = pd.Series(np.nan, index=p.index, name="ensemble_calibrated")
    ms = start + pd.DateOffset(years=init_years)
    pv = p.dropna()
    for t in pv.index:
        if t < ms:
            continue
        tr = pv.index[pv.index < t]
        ytr = y.reindex(tr).dropna()
        tr = ytr.index
        if len(tr) < 24 or ytr.sum() < 4:
            continue
        m = LogisticRegression(C=1e6, max_iter=1000).fit(
            _logit(pv.loc[tr].values).reshape(-1, 1), ytr.values)
        out.loc[t] = float(m.predict_proba(_logit([pv.loc[t]]).reshape(-1, 1))[0, 1])
    return out.dropna()


# ---------------------------------------------------------------------------
# Build + evaluate
# ---------------------------------------------------------------------------

def build_and_save(family: str = "bear") -> dict:
    """Compute the production ensemble series + summary and persist them."""
    cfg = FAMILIES[family]
    members = cfg["members"]
    y = _target(family)
    oos = member_oos(family)
    hist = member_history(family)

    # Production (in-sample) ensemble + current reading (all four available now)
    ens_hist = ensemble_equal(hist)
    current = {k: float(load_assessment(k)["current_prob"]) for k in members}
    ens_current = float(np.mean(list(current.values())))

    # Honest OOS ensemble (equal weight, growing membership)
    ens_oos = ensemble_equal(oos)

    # Calibration: global Platt map (for the current reading) + walk-forward
    # Platt (for an honest historical calibrated curve).
    a, b = fit_global_platt(ens_oos, y)
    ens_current_cal = float(apply_platt(ens_current, a, b))
    ens_oos_cal = platt_walkforward(ens_oos, y, COMMON_START)

    out = pd.DataFrame({"ensemble": ens_oos, "ensemble_calibrated": ens_oos_cal})
    out.to_csv(_DATA_DIR / cfg["oos_file"])

    # Per-member metadata + OOS skill (computed offline so the app needs no ML).
    from sklearn.metrics import roc_auc_score

    def _auc_on(s, window_start=None):
        ss = s if window_start is None else s.loc[s.index >= window_start]
        idx = ss.dropna().index.intersection(y.dropna().index)
        yy = y.loc[idx]
        return float(roc_auc_score(yy.values, ss.loc[idx].values)) if yy.nunique() > 1 else None

    members_meta = {}
    for k in members:
        am = load_assessment(k)
        spec = _SPECS[k]
        members_meta[k] = {
            "title": am["title"],
            "subtitle": am["subtitle"],
            "train_start": spec.get("train_start"),
            "oos_start": spec.get("oos_start"),
            "n_factors": len(am["features"]),
            "categories": [am["labels"][f][1] for f in am["features"]],
            "current_prob": current[k],
            "current_prob_calibrated": float(apply_platt(current[k], a, b)),
            "base_rate": float(am["base_rate"]),
            "oos_auc_native": _auc_on(oos[k]),
            "oos_auc_common": _auc_on(oos[k], COMMON_START),
        }

    ens_common = ens_oos.loc[ens_oos.index >= COMMON_START]
    metrics = {
        "ensemble_auc_full": _auc_on(ens_oos),
        "ensemble_auc_common": _auc_on(ens_common),
        "realized_base_rate": float(y.reindex(ens_oos.dropna().index).dropna().mean()),
    }

    summary = {
        "family": family,
        "members": members,
        "method": "equal_weight",
        "calibration": "platt",
        "platt_a": a,
        "platt_b": b,
        "current_member_probs": current,
        "current_ensemble_prob": ens_current,
        "current_ensemble_prob_calibrated": ens_current_cal,
        "as_of": str(hist.dropna(how="all").index[-1].date()),
        "common_oos_start": str(COMMON_START.date()),
        "members_meta": members_meta,
        "metrics": metrics,
    }
    with open(_DATA_DIR / cfg["params_file"], "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def evaluate(family: str = "bear") -> None:
    members = FAMILIES[family]["members"]
    y = _target(family)
    oos = member_oos(family)

    print("=" * 72)
    print("MEMBER OOS coverage")
    print("=" * 72)
    for k in members:
        s = oos[k].dropna()
        print(f"  {k:8s} {s.index[0].date()} -> {s.index[-1].date()}  (n={len(s)})")

    # ---- Common window (2005+): every member predicts -> fair comparison ----
    print("\n" + "=" * 72)
    print(f"COMMON WINDOW {COMMON_START.date()}+  (all four out-of-sample) — AUC")
    print("=" * 72)
    common = oos.loc[oos.index >= COMMON_START].dropna()
    yc = y.reindex(common.index)
    member_auc = {}
    for k in members:
        member_auc[k] = _auc(yc, common[k])
        print(f"  member {k:8s} AUC={member_auc[k]:.4f}")
    eq = ensemble_equal(common)
    print(f"  ENSEMBLE equal-weight     AUC={_auc(yc, eq):.4f}")
    w = {k: max(member_auc[k] - 0.5, 0.0) for k in members}   # skill above chance
    aw = ensemble_auc_weighted(common, w)
    print(f"  ENSEMBLE AUC-weighted     AUC={_auc(yc, aw):.4f}   weights="
          f"{ {k: round(v/sum(w.values()),2) for k,v in w.items()} }")
    st = ensemble_stacked_walkforward(oos, y, COMMON_START)
    if len(st):
        ys = y.reindex(st.index)
        print(f"  ENSEMBLE stacked (WF)     AUC={_auc(ys, st):.4f}   "
              f"(from {st.index[0].date()}, n={len(st)})")

    # ---- Long window: growing-membership equal-weight ensemble ----
    print("\n" + "=" * 72)
    print("LONG WINDOW — equal-weight ensemble, growing membership")
    print("=" * 72)
    ens = ensemble_equal(oos)
    yl = y.reindex(ens.index)
    print(f"  ensemble OOS span {ens.index[0].date()} -> {ens.index[-1].date()} (n={len(ens)})")
    print(f"  ensemble AUC (full span)        = {_auc(yl, ens):.4f}")
    for cut in ("1970-01-31", "1985-01-31", "2005-01-31"):
        c = pd.Timestamp(cut)
        sub = ens.loc[ens.index >= c]
        print(f"    AUC on {cut[:4]}+ : {_auc(y.reindex(sub.index), sub):.4f}  (n={len(sub)})")


if __name__ == "__main__":
    import sys
    fams = sys.argv[1:] or ["bear", "correction"]
    for fam in fams:
        s = build_and_save(fam)
        print(f"[{fam}] saved {FAMILIES[fam]['oos_file']} + {FAMILIES[fam]['params_file']}")
        print(f"  current ensemble (raw → calibrated) = "
              f"{s['current_ensemble_prob']:.4f} → {s['current_ensemble_prob_calibrated']:.4f}")
        print(f"  members: { {k: round(v,3) for k,v in s['current_member_probs'].items()} }")
        print()
        evaluate(fam)
        print()
