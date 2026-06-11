"""
bear/make_docs.py — generate technical documentation for the Bear and
Correction models.

Produces, under bear/docs/:
  bear_model_tech_doc.md          correction_model_tech_doc.md
  <kind>_insample.png             in-sample fitted series + event shading
  <kind>_oos.png                  walk-forward out-of-sample series
  <kind>_roc.png                  in-sample & OOS ROC curves

Each doc contains: Feature Engineering, Constrained Logistic Model, Regression
Performance (Specification / Sensitivity / AUC), and the in-sample + OOS charts.

Run:  python -m bear.make_docs   (needs scikit-learn + matplotlib)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from bear.inference import load_assessment, _SPECS

_BEAR_DIR = Path(__file__).resolve().parent
_DATA_DIR = _BEAR_DIR.parent / "data"
_DOCS_DIR = _BEAR_DIR / "docs"

CORR_SHADE = "#f9a8d4"
BEAR_SHADE = "#ef4444"
LINE_COLOR = "#173f2a"

# ---------------------------------------------------------------------------
# Per-factor feature-engineering detail (raw series / transform / rationale)
# ---------------------------------------------------------------------------

FEATURE_DETAILS: dict[str, dict[str, str]] = {
    "ntfs_3m_chg": {
        "raw": "Near-term forward spread (6-quarter-ahead 3m forward rate − 3m T-bill), GSW curve",
        "transform": "3-month change",
        "rationale": "Rising NTFS reflects markets pricing imminent Fed cuts — a late-cycle recession signal (Engstrom-Sharpe 2019).",
    },
    "ts_inv_dummy": {
        "raw": "10y−3m Treasury term spread (FRED T10Y3M)",
        "transform": "Inversion dummy 1{spread < 0}",
        "rationale": "Curve inversion is the canonical recession predictor (Estrella-Mishkin 1998).",
    },
    "ebp_3m_chg": {
        "raw": "Gilchrist-Zakrajšek Excess Bond Premium (Fed)",
        "transform": "3-month change",
        "rationale": "Rapidly rising EBP signals deteriorating credit supply / risk appetite beyond default risk (GZ 2012).",
    },
    "baa_zscore_60m": {
        "raw": "Moody's BAA − 10y default spread (FRED BAA10Y)",
        "transform": "60-month trailing z-score",
        "rationale": "Credit spreads elevated vs a 5-year norm indicate financial stress.",
    },
    "lei_6m_growth": {
        "raw": "OECD US composite leading indicator (FRED USALOLITOAASTSAM)",
        "transform": "Annualized 6-month growth",
        "rationale": "A falling leading index foreshadows slowing growth ~7 months ahead.",
    },
    "ffr_6m_chg": {
        "raw": "Effective federal funds rate (FRED DFF)",
        "transform": "6-month change",
        "rationale": "Sharp Fed easing tends to precede recessions (policy-response signal; Tokic-Jackson 2023).",
    },
    "sahm_level": {
        "raw": "Sahm rule, real-time vintage (FRED SAHMREALTIME)",
        "transform": "Level",
        "rationale": "Rising unemployment momentum marks recession onset (Sahm 2019).",
    },
    "vts_slope": {
        "raw": "VIX term structure (VIX3M − VIX)",
        "transform": "Level (slope)",
        "rationale": "Steep contango = complacency that often precedes corrections; backwardation = stress.",
    },
    "spx_vs_10ma": {
        "raw": "S&P 500 price (monthly close)",
        "transform": "% deviation from 10-month MA (≈ 200-day)",
        "rationale": "Trend deterioration; price below the long MA is risk-off (Moskowitz-Ooi-Pedersen 2012).",
    },
    "m12_1_mom": {
        "raw": "S&P 500 price (monthly close)",
        "transform": "12-1 momentum (12m return excluding last month)",
        "rationale": "Stretched momentum raises mean-reversion correction risk.",
    },
    "anfci_3m_chg": {
        "raw": "Chicago Fed Adjusted NFCI (FRED ANFCI)",
        "transform": "3-month change",
        "rationale": "Tightening financial conditions (orthogonalized to the economy) precede pullbacks.",
    },
    "cape_20yr_pct": {
        "raw": "Shiller CAPE (P/E10)",
        "transform": "20-year trailing percentile",
        "rationale": "Valuation extreme — a severity conditioner; weak standalone timer (Goyal-Welch 2008).",
    },
    "baa_zscore_24m": {
        "raw": "Moody's BAA − 10y default spread (FRED BAA10Y)",
        "transform": "24-month trailing z-score",
        "rationale": "Fast credit-stress signal (the Tokic-Jackson correction→bear bridge).",
    },
}


def _event_series(kind: str) -> pd.Series:
    """Binary outcome used for AUC: the rolling-window drawdown event."""
    targets = pd.read_csv(_DATA_DIR / "targets.csv", index_col=0, parse_dates=True)
    if _SPECS[kind]["horizon"] == 12:
        ev = (targets["mdd_12m"] <= -0.20).astype(float).where(targets["mdd_12m"].notna())
        label = "Bear event: 12-month drawdown > 20%"
    else:
        ev = (targets["mdd_6m"] <= -0.10).astype(float).where(targets["mdd_6m"].notna())
        label = "Correction event: 6-month drawdown > 10%"
    ev.name = "event"
    ev.attrs["label"] = label
    return ev


def _event_episodes(kind: str, index: pd.DatetimeIndex,
                    event: pd.Series) -> tuple[list[dict], int]:
    """
    Group the event months (within `index`) into contiguous episodes.

    Returns (episodes, total_event_months) where each episode dict has
    start, end, n_months, and worst (deepest forward drawdown over the run).
    """
    mdd_col = "mdd_12m" if _SPECS[kind]["horizon"] == 12 else "mdd_6m"
    mdd = pd.read_csv(_DATA_DIR / "targets.csv", index_col=0,
                      parse_dates=True)[mdd_col]

    ev = event.reindex(index).fillna(0.0)
    episodes: list[dict] = []
    in_run = False
    start = prev = None
    for d, flag in ev.items():
        if flag >= 0.5 and not in_run:
            in_run, start = True, d
        elif flag < 0.5 and in_run:
            episodes.append({"start": start, "end": prev})
            in_run = False
        prev = d
    if in_run:
        episodes.append({"start": start, "end": prev})

    total = int((ev >= 0.5).sum())
    for ep in episodes:
        seg = ev.loc[ep["start"]:ep["end"]]
        ep["n_months"] = int((seg >= 0.5).sum())
        ep["worst"] = float(mdd.loc[ep["start"]:ep["end"]].min())
    return episodes, total


def _auc(pred: pd.Series, event: pd.Series) -> tuple[float, int, int]:
    mask = pred.notna() & event.notna()
    y = event.loc[mask].values
    p = pred.loc[mask].values
    if len(y) < 10 or y.sum() == 0 or y.sum() == len(y):
        return float("nan"), len(y), int(y.sum())
    return float(roc_auc_score(y, p)), len(y), int(y.sum())


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _plot_series(series: pd.Series, base: float, event: pd.Series,
                 title: str, ylabel: str, out_path: Path,
                 shade_color: str, is_drawdown_scale: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    s = series.dropna()
    ax.plot(s.index, s.values, color=LINE_COLOR, lw=1.6, label=ylabel)
    ax.axhline(base, ls="--", color="#6b7280", lw=1,
               label=f"Historical mean ({base:.0%})")

    # Shade event months (the realized rolling-window drawdown event)
    ev = event.reindex(s.index).fillna(0.0)
    in_run = False
    run_start = None
    prev = None
    for d, flag in ev.items():
        if flag >= 0.5 and not in_run:
            in_run = True; run_start = d
        elif flag < 0.5 and in_run:
            ax.axvspan(run_start, prev, color=shade_color, alpha=0.25, lw=0)
            in_run = False
        prev = d
    if in_run:
        ax.axvspan(run_start, prev, color=shade_color, alpha=0.25, lw=0)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(0, max(0.6, float(s.max()) * 1.15))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_roc(pred_is: pd.Series, pred_oos: pd.Series, event: pd.Series,
              title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for pred, name, color in ((pred_is, "In-sample", "#173f2a"),
                              (pred_oos, "Out-of-sample", "#b68a35")):
        mask = pred.notna() & event.notna()
        y = event.loc[mask].values
        p = pred.loc[mask].values
        if len(y) < 10 or y.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#9ca3af", lw=1)
    ax.set_xlabel("False positive rate", fontsize=9)
    ax.set_ylabel("True positive rate", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------

def build_doc(kind: str) -> Path:
    a = load_assessment(kind)
    spec = _SPECS[kind]
    event = _event_series(kind)

    horizon   = a["horizon"]
    value_kind = a["value_kind"]
    is_sev    = value_kind == "severity"
    feats     = a["features"]
    coef      = a["coef"]
    intercept = a["intercept"]
    mu, sigma = a["mu"], a["sigma"]
    factors   = a["factors"]
    base      = a["base_rate"]
    current   = a["current_prob"]
    as_of     = a["as_of"]
    hac_lags  = a["hac_maxlags"]

    hist = a["history"]
    oos  = a["history_oos"]

    auc_is, n_is, pos_is   = _auc(hist, event)
    auc_oos, n_oos, pos_oos = _auc(oos, event)

    # ---- charts ----
    _DOCS_DIR.mkdir(parents=True, exist_ok=True)
    shade = BEAR_SHADE if horizon == 12 else CORR_SHADE
    ylab = "Expected drawdown severity" if is_sev else "P(drawdown event)"
    _plot_series(hist, base, event,
                 f"{a['title']} model — in-sample fitted ({hist.index[0].year}–{hist.index[-1].year})",
                 ylab, _DOCS_DIR / f"{kind}_insample.png", shade, is_sev)
    _plot_series(oos, base, event,
                 f"{a['title']} model — out-of-sample, walk-forward",
                 ylab, _DOCS_DIR / f"{kind}_oos.png", shade, is_sev)
    _plot_roc(hist, oos, event,
              f"{a['title']} model — ROC", _DOCS_DIR / f"{kind}_roc.png")

    # ---- sensitivity (Δ output per +1 SD factor move, at current reading) ----
    slope = current * (1 - current)   # sigmoid derivative at current prediction
    sens_rows = []
    for _, r in factors.iterrows():
        d_out = slope * r["Coefficient"]
        sens_rows.append((r["Description"], r["Coefficient"], r["Weight %"],
                          r["P (HAC)"], d_out, r["Direction"]))

    # ---- target wording ----
    if is_sev:
        target_def = (f"the **{horizon}-month rolling drawdown severity** "
                      f"$s_t = -\\mathrm{{MDD}}_t \\in [0,1]$, the worst peak-to-trough "
                      f"decline over the rolling {horizon}-month forward window")
        link_line = (r"\hat{s}_t = \mathbb{E}[s_t] = \sigma(z_t) = \frac{1}{1+e^{-z_t}}")
        model_kind = "fractional logistic (quasi-binomial)"
    else:
        target_def = (f"a binary **{horizon}-month rolling correction**, "
                      f"$y_t = \\mathbf{{1}}\\{{\\mathrm{{MDD}}_t \\le -10\\%\\}}$, i.e. any "
                      f"drawdown deeper than 10% over the rolling {horizon}-month forward window")
        link_line = (r"\hat{p}_t = \Pr(y_t=1) = \sigma(z_t) = \frac{1}{1+e^{-z_t}}")
        model_kind = "binary logistic"

    # ---- MDD definition ----
    mdd_line = (r"\mathrm{MDD}_t = \min_{t < u \le t+" + str(horizon) + r"}"
                r"\left(\frac{P_u}{\max_{t<v\le u}P_v}-1\right)")

    # ---- fitted z ----
    z_terms = [f"{intercept:+.3f}"] + [
        f"{c:+.3f}\\,\\tilde{{x}}_{{{i}}}" for i, c in enumerate(coef, 1)
    ]

    # ---- build markdown ----
    lines: list[str] = []
    lines.append(f"# {a['title']} Model — Technical Documentation\n")
    lines.append(f"*Generated from `bear/{spec['features_csv']}` and "
                 f"`bear/targets.csv`. Reproduce with `python -m bear.make_docs`.*\n")

    # Overview
    lines.append("## 1. Overview\n")
    lines.append(f"- **Objective:** estimate {target_def}.")
    lines.append(f"- **Model:** weight-constrained {model_kind} on standardized factors, "
                 f"signs fixed to economic priors, calibrated to the base rate.")
    lines.append(f"- **Horizon:** {horizon} months (rolling forward window).")
    lines.append(f"- **Sample:** {hist.index[0].date()} → {hist.index[-1].date()} "
                 f"({n_is} complete monthly observations).")
    base_word = "mean severity" if is_sev else "base rate"
    cur_word  = "Expected drawdown severity" if is_sev else "P(event)"
    lines.append(f"- **Historical {base_word}:** {base:.1%}.")
    lines.append(f"- **Current reading ({as_of.date()}):** {cur_word} = **{current:.1%}**.")
    lines.append(f"- **Inference:** Newey-West HAC, max lag = {hac_lags} months.\n")

    # Feature engineering
    lines.append("## 2. Feature Engineering\n")
    lines.append("All raw series are monthly (daily/weekly series are sampled to "
                 "month-end), shifted forward by their real-world publication lag to "
                 "prevent look-ahead, and transformed to stationary form. Each factor "
                 "is then standardized on the training sample, "
                 r"$\tilde{x}_{i,t} = (x_{i,t}-\mu_i)/\sigma_i$." + "\n")
    lines.append("| # | Factor | Raw series | Transformation | Rationale |")
    lines.append("|---|---|---|---|---|")
    for i, f in enumerate(feats, 1):
        d = FEATURE_DETAILS.get(f, {"raw": "—", "transform": "—", "rationale": "—"})
        lines.append(f"| {i} | `{f}` | {d['raw']} | {d['transform']} | {d['rationale']} |")
    lines.append("")
    lines.append("Forward drawdown target (rolling window):\n")
    lines.append("$$" + mdd_line + "$$\n")

    # Constrained logistic
    lines.append("## 3. Constrained Logistic Model\n")
    lines.append("**Link / specification:**\n")
    lines.append("$$" + link_line + "$$\n")
    lines.append("$$z_t = \\beta_0 + \\sum_{i=1}^{" + str(len(feats)) +
                 "} \\beta_i\\,\\tilde{x}_{i,t}$$\n")
    lines.append("**Estimation:** coefficients are written as "
                 r"$\beta_i = \mathrm{sign}_i \cdot \gamma_i$ with $\gamma_i \ge 0$ "
                 "(signs fixed to economic priors), fitted by maximizing the "
                 + ("Bernoulli quasi-likelihood (continuous $[0,1]$ target)"
                    if is_sev else "Bernoulli likelihood") +
                 " subject to per-factor weight bounds:\n")
    lines.append("$$w_i = \\frac{|\\beta_i|}{\\sum_j |\\beta_j|}, \\qquad "
                 f"{a['min_w']*100:.0f}\\% \\le w_i \\le {a['max_w']*100:.0f}\\%$$\n")
    lines.append("**Fitted model:**\n")
    lines.append("$$z_t = " + " ".join(z_terms) + "$$\n")
    lines.append("**Fitted coefficients** (sorted by weight):\n")
    lines.append("| Factor | Coefficient $\\beta_i$ | Weight $w_i$ | p (HAC) |")
    lines.append("|---|---:|---:|---:|")
    for _, r in factors.iterrows():
        pv = r["P (HAC)"]
        star = " \\*" if (pd.notna(pv) and pv < 0.05) else (" ." if (pd.notna(pv) and pv < 0.10) else "")
        ptxt = "—" if pd.isna(pv) else f"{pv:.3f}{star}"
        lines.append(f"| `{r['Feature']}` | {r['Coefficient']:+.4f} | "
                     f"{r['Weight %']:.1f}% | {ptxt} |")
    lines.append(f"| _intercept_ | {intercept:+.4f} | — | — |")
    lines.append("\n`*` p<0.05  ·  `.` p<0.10  (Newey-West HAC, "
                 f"max lag = {hac_lags}).\n")

    # Performance
    lines.append("## 4. Regression Performance\n")
    lines.append("### 4.1 Specification\n")
    lines.append(f"- **Form:** {model_kind}, identity-of-weights constraint "
                 f"$w_i \\in [{a['min_w']*100:.0f}\\%, {a['max_w']*100:.0f}\\%]$, "
                 f"{len(feats)} factors, signs fixed.")
    lines.append(f"- **Autocorrelation:** the {horizon}-month rolling target overlaps "
                 f"across consecutive months (consecutive observations share "
                 f"{horizon-1}/{horizon} of their window). Standard errors are "
                 f"Newey-West **HAC**-corrected with a Bartlett kernel and max lag = "
                 f"{hac_lags} months.")
    lines.append(f"- **Calibration:** fitted by natural-weight likelihood, so the output "
                 f"is calibrated to the {base:.1%} {base_word}.\n")

    lines.append("### 4.2 Sensitivity\n")
    lines.append("Marginal effect of a **+1 standard-deviation** move in each factor on "
                 "the model output, evaluated at the current reading "
                 f"($\\partial \\hat{{y}} = \\hat{{y}}(1-\\hat{{y}})\\,\\beta_i$, "
                 f"$\\hat{{y}}={current:.3f}$):\n")
    lines.append("| Factor | $\\beta_i$ | Weight | Δ output / +1 SD | p (HAC) | Push |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for desc, c, w, pv, d_out, direction in sorted(sens_rows, key=lambda x: -abs(x[4])):
        ptxt = "—" if pd.isna(pv) else f"{pv:.3f}"
        lines.append(f"| {desc} | {c:+.4f} | {w:.1f}% | {d_out:+.4f} "
                     f"({d_out*100:+.2f} pp) | {ptxt} | {direction} |")
    lines.append("")

    lines.append("### 4.3 Area Under the Curve (AUC)\n")
    score_note = ("the continuous severity prediction is used as the ranking score"
                  if is_sev else "the predicted probability is the ranking score")
    lines.append(f"Discrimination is measured against the realized event "
                 f"(*{event.attrs['label']}*); {score_note}.\n")
    lines.append("| Sample | AUC | N | Events |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| In-sample | {auc_is:.3f} | {n_is} | {pos_is} |")
    lines.append(f"| Out-of-sample (walk-forward) | {auc_oos:.3f} | {n_oos} | {pos_oos} |")
    lines.append("")
    lines.append(f"![ROC curve]({kind}_roc.png)\n")

    # Charts
    lines.append("## 5. Charts\n")
    lines.append("### 5.1 In-sample fit\n")
    lines.append("Fitted values across the full sample (parameters estimated on the "
                 "full sample). Shaded spans mark the realized rolling-window event.\n")
    lines.append(f"![In-sample]({kind}_insample.png)\n")
    lines.append("### 5.2 Out-of-sample (walk-forward)\n")
    lines.append("Expanding-window estimate: at each month the model is re-fit on prior "
                 "data only, then predicts that month (no look-ahead).\n")
    lines.append(f"![Out-of-sample]({kind}_oos.png)\n")

    # Appendix — the realized event episodes behind the AUC event count
    episodes, total_evt = _event_episodes(kind, hist.index, event)
    evt_word = "bear" if horizon == 12 else "correction"
    thr = "20%" if horizon == 12 else "10%"
    lines.append(f"## 6. Appendix — Realized {evt_word.title()} Events\n")
    lines.append(f"The {total_evt} event-months in the AUC sample (signal months whose "
                 f"{horizon}-month forward drawdown exceeded {thr}) group into "
                 f"**{len(episodes)} distinct episodes**. *Start*/*End* are the first and "
                 f"last signal months of each run; *Worst drawdown* is the deepest "
                 f"{horizon}-month forward drawdown observed during the episode.\n")
    lines.append("| # | Start | End | Signal months | Worst drawdown |")
    lines.append("|---|---|---|---:|---:|")
    for i, ep in enumerate(episodes, 1):
        lines.append(f"| {i} | {ep['start'].strftime('%Y-%m')} | "
                     f"{ep['end'].strftime('%Y-%m')} | {ep['n_months']} | "
                     f"{ep['worst']:.1%} |")
    lines.append(f"| | | **Total** | **{total_evt}** | |")
    lines.append("")

    out_path = _DOCS_DIR / f"{kind}_model_tech_doc.md"
    out_path.write_text("\n".join(lines))
    return out_path


if __name__ == "__main__":
    mapping = {"bear": "bearplus", "correction": "correctionplus"}
    for doc_name, kind in mapping.items():
        path = build_doc(kind)
        # rename file to friendly doc name
        friendly = _DOCS_DIR / f"{doc_name}_model_tech_doc.md"
        if path != friendly:
            path.rename(friendly)
        a = load_assessment(kind)
        print(f"  {a['title']:<12} -> {friendly.relative_to(_BEAR_DIR.parent)}")
    print(f"\nDocs + charts written to {_DOCS_DIR}")
