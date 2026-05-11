"""
src/visualization/plots.py
===========================
Visualisation Layer — 6-Panel Publication-Grade Figures

Figure inventory
----------------
1. Learning Curve         — train/val loss vs epoch  -> diagnose over/underfitting
2. Residual Distribution  — histogram + KDE of (ŷ−y) -> check normality (OLS assumption)
3. Actual vs Predicted    — scatter + identity line  -> detect systematic bias
4. Feature Weights        — top-k |w_j| bar chart    -> model interpretability
5. Time-Series Plot       — actual vs predicted over time  -> visual inspection
6. Residual vs Predicted  — homoscedasticity diagnostic  -> detect heteroscedasticity

All figures use Seaborn's publication style and save to outputs/figures/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess  # type: ignore[import]
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

logger = logging.getLogger(__name__)

# ── Global style ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGSIZE_DEFAULT = (10, 6)
DPI = 150


def _save(fig: plt.Figure, path: Path, fname: str) -> None:
    """Save figure as high-resolution PNG."""
    path.mkdir(parents=True, exist_ok=True)
    fpath = path / fname
    fig.savefig(fpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved -> %s", fpath)


# ---------------------------------------------------------------------------
# 1. Learning Curve
# ---------------------------------------------------------------------------

def plot_learning_curve(
    train_losses: List[float],
    val_losses:   List[float],
    best_epoch:   int,
    save_dir:     Optional[Path] = None,
) -> plt.Figure:
    """
    Plot Ridge loss over training epochs.

    Interpretation:
      • Diverging val_loss -> overfitting -> increase λ or reduce features
      • High both losses   -> underfitting -> reduce λ or add features
      • Best epoch marked with a vertical dashed line (early stopping point)
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)

    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss (Ridge Objective)", lw=2)
    ax.plot(epochs, val_losses,   label="Val Loss (MSE)",                lw=2, linestyle="--")
    ax.axvline(best_epoch + 1, color="red", linestyle=":", lw=1.5,
               label=f"Best Epoch ({best_epoch + 1})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Learning Curve — Ridge Regression (Mini-Batch GD)")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    if save_dir:
        _save(fig, save_dir, "01_learning_curve.png")
    return fig


# ---------------------------------------------------------------------------
# 2. Residual Distribution
# ---------------------------------------------------------------------------

def plot_residual_distribution(
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    save_dir: Optional[Path] = None,
) -> plt.Figure:
    """
    Histogram + KDE of residuals e = ŷ − y.

    OLS assumption: residuals ~ 𝒩(0, σ²).
    Heavy tails or systematic skew indicate model misspecification.
    """
    residuals = y_pred - y_true
    mu, sigma = residuals.mean(), residuals.std()

    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    sns.histplot(residuals, kde=True, ax=ax, bins=60, color="steelblue",
                 stat="density", alpha=0.6)

    # Overlay theoretical normal
    x = np.linspace(residuals.min(), residuals.max(), 300)
    from scipy.stats import norm
    ax.plot(x, norm.pdf(x, mu, sigma), "r--", lw=2, label=f"Normal(mu={mu:.2f}, sigma={sigma:.2f})")

    ax.axvline(0, color="black", lw=1, linestyle=":")
    ax.set_xlabel("Residual (ŷ − y)  [µg/m³]")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution — Normality Check")
    ax.legend()

    if save_dir:
        _save(fig, save_dir, "02_residual_distribution.png")
    return fig


# ---------------------------------------------------------------------------
# 3. Actual vs Predicted Scatter
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    save_dir: Optional[Path] = None,
) -> plt.Figure:
    """
    Scatter plot of ŷ vs y with the ideal 45° identity line.

    Systematic deviation above/below the identity line indicates:
      • Over-prediction or under-prediction bias
    Colour-coded by density using a 2D histogram proxy.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Hexbin density
    hb = ax.hexbin(y_true, y_pred, gridsize=60, cmap="Blues",
                   mincnt=1, linewidths=0.2)
    plt.colorbar(hb, ax=ax, label="Count")

    # Identity line
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Ideal (ŷ = y)")

    # US EPA threshold
    ax.axhline(35.4, color="orange", lw=1.2, linestyle="--", label="EPA Threshold 35.4 µg/m³")
    ax.axvline(35.4, color="orange", lw=1.2, linestyle="--")

    ax.set_xlabel("Actual PM2.5 [µg/m³]")
    ax.set_ylabel("Predicted PM2.5 [µg/m³]")
    ax.set_title("Actual vs Predicted — Regression Performance")
    ax.legend()

    if save_dir:
        _save(fig, save_dir, "03_actual_vs_predicted.png")
    return fig


# ---------------------------------------------------------------------------
# 4. Feature Weights (Top-K)
# ---------------------------------------------------------------------------

def plot_feature_weights(
    weights:       np.ndarray,
    feature_names: List[str],
    top_k:         int = 25,
    save_dir:      Optional[Path] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of the top-k features by |w_j|.

    Positive weights -> PM2.5 increases with feature.
    Negative weights -> PM2.5 decreases with feature.
    Magnitude -> relative influence under L2 regularisation.
    """
    idx     = np.argsort(np.abs(weights))[::-1][:top_k]
    vals    = weights[idx]
    names   = [feature_names[i] for i in idx]
    colors  = ["#d73027" if v > 0 else "#4575b4" for v in vals]

    fig, ax = plt.subplots(figsize=(9, top_k * 0.38 + 1))
    bars = ax.barh(range(top_k), vals[::-1], color=colors[::-1])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Feature Weight (wⱼ)")
    ax.set_title(f"Top-{top_k} Feature Weights — Ridge Regression")

    if save_dir:
        _save(fig, save_dir, "04_feature_weights.png")
    return fig


# ---------------------------------------------------------------------------
# 5. Time-Series Plot
# ---------------------------------------------------------------------------

def plot_time_series(
    timestamps: np.ndarray,
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    city:       str = "All cities",
    n_hours:    int = 720,
    save_dir:   Optional[Path] = None,
) -> plt.Figure:
    """
    Overlay actual and predicted PM2.5 over the last `n_hours` of test set.

    Shaded region above EPA threshold (35.4 µg/m³) highlights alert periods.
    """
    n = min(n_hours, len(y_true))
    ts   = timestamps[-n:]
    yt   = y_true[-n:]
    yp   = y_pred[-n:]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ts, yt, label="Actual PM2.5",    lw=1.5, color="royalblue", alpha=0.85)
    ax.plot(ts, yp, label="Predicted PM2.5", lw=1.2, color="tomato",    alpha=0.85,
            linestyle="--")
    ax.axhline(35.4, color="darkorange", lw=1.0, linestyle=":", label="EPA Threshold")
    _fill_max = np.maximum(yt.astype(float), yp.astype(float))
    ax.fill_between(ts, 35.4, _fill_max, where=_fill_max >= 35.4,
                    alpha=0.12, color="red", label="Unhealthy zone")

    ax.set_xlabel("Timestamp")
    ax.set_ylabel("PM2.5 [µg/m³]")
    ax.set_title(f"Time-Series Forecast — {city} (last {n}h of test set)")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()

    if save_dir:
        _save(fig, save_dir, f"05_time_series_{city.replace(' ', '_')}.png")
    return fig


# ---------------------------------------------------------------------------
# 6. Residual vs Predicted (Homoscedasticity Check)
# ---------------------------------------------------------------------------

def plot_residual_vs_predicted(
    y_pred:   np.ndarray,
    y_true:   np.ndarray,
    save_dir: Optional[Path] = None,
) -> plt.Figure:
    """
    Scatter of residuals (e = ŷ−y) vs fitted values ŷ.

    Ideal behaviour: random scatter around e = 0 (homoscedastic).
    Funnel shape (variance grows with ŷ) -> heteroscedasticity -> consider
    log-transform or weighted regression.
    """
    residuals = y_pred - y_true

    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    ax.scatter(y_pred, residuals, alpha=0.25, s=8, color="steelblue", edgecolors="none")

    # LOWESS smoother for trend detection (requires statsmodels)
    if _HAS_STATSMODELS:
        sm = _lowess(residuals, y_pred, frac=0.2, return_sorted=True)
        ax.plot(sm[:, 0], sm[:, 1], "r-", lw=2, label="LOWESS trend")

    ax.axhline(0, color="black", lw=1.0, linestyle="--")
    ax.set_xlabel("Fitted Values ŷ [µg/m³]")
    ax.set_ylabel("Residual (ŷ − y) [µg/m³]")
    ax.set_title("Residual vs Predicted — Homoscedasticity Diagnostic")
    ax.legend()

    if save_dir:
        _save(fig, save_dir, "06_residual_vs_predicted.png")
    return fig


# ---------------------------------------------------------------------------
# Per-City RMSE/F1 Summary
# ---------------------------------------------------------------------------

def plot_city_comparison(
    city_metrics: "pd.DataFrame",
    save_dir:     Optional[Path] = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing RMSE and F1-Bad across cities.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    city_metrics["RMSE"].plot.bar(ax=axes[0], color="steelblue", edgecolor="black")
    axes[0].set_title("RMSE by City")
    axes[0].set_ylabel("RMSE [µg/m³]")
    axes[0].set_xlabel("")

    city_metrics["F1_Bad"].plot.bar(ax=axes[1], color="tomato", edgecolor="black")
    axes[1].set_title("F1-Score (Unhealthy class) by City")
    axes[1].set_ylabel("F1-Score")
    axes[1].set_xlabel("")

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Regional Performance Analysis", fontweight="bold")
    plt.tight_layout()

    if save_dir:
        _save(fig, save_dir, "07_city_comparison.png")
    return fig


# ---------------------------------------------------------------------------
# Master plotting driver
# ---------------------------------------------------------------------------

def generate_all_figures(
    model,
    history:       Dict,
    eval_results:  Dict,
    feature_names: List[str],
    df_test_raw:   "pd.DataFrame",
    city_metrics:  "pd.DataFrame",
    save_dir:      Path,
) -> None:
    """
    Generate and save all 7 publication-grade figures.

    Parameters
    ----------
    model         : fitted RidgeRegression instance
    history       : dict from model.training_history()
    eval_results  : dict from metrics.full_evaluation()
    feature_names : list of feature column names
    df_test_raw   : original-scale test DataFrame
    city_metrics  : output of evaluate_by_city()
    save_dir      : output directory for figures
    """
    save_dir = Path(save_dir)

    plot_learning_curve(
        history["train_loss"], history["val_loss"],
        model.best_epoch_, save_dir
    )

    y_true = eval_results["y_orig"]
    y_pred = eval_results["yp_orig"]

    plot_residual_distribution(y_true, y_pred, save_dir)
    plot_actual_vs_predicted(y_true, y_pred, save_dir)
    plot_feature_weights(model.feature_weights, feature_names, save_dir=save_dir)

    # Time-series for all cities combined
    ts = df_test_raw["timestamp"].values
    plot_time_series(ts, y_true, y_pred, city="All Cities", save_dir=save_dir)

    plot_residual_vs_predicted(y_pred, y_true, save_dir)
    plot_city_comparison(city_metrics, save_dir)

    logger.info("All figures saved to %s", save_dir)