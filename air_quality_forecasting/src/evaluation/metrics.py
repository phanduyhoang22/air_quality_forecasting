"""
src/evaluation/metrics.py
==========================
Evaluation Layer — Regression · Classification · Regional Analysis

Metrics implemented
-------------------

Regression
~~~~~~~~~~
• RMSE (Root Mean Squared Error):
    RMSE = √[ (1/n) Σ (ŷᵢ − yᵢ)² ]
  Penalises large errors quadratically — important for pollution spikes.

• R² (Coefficient of Determination):
    R² = 1 − SS_res / SS_tot
       = 1 − Σ(ŷᵢ−yᵢ)² / Σ(yᵢ−ȳ)²
  Measures fraction of variance explained.  R²=1 → perfect; R²=0 → baseline mean.

• RMSE Improvement over Persistence Baseline:
    Δ% = (1 − RMSE_model / RMSE_baseline) × 100

Classification (US EPA threshold τ = 35.4 µg/m³)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Binary labels:
    y_bin = 1  if PM2.5 ≥ τ  (Unhealthy / "Bad" air)
    y_bin = 0  if PM2.5 < τ  (Good / Moderate)

Priority metric: Recall (Sensitivity) for class 1:
    Recall = TP / (TP + FN)
  False Negatives (missed "Bad" days) carry highest health cost.

Full report via sklearn.metrics.classification_report.

Regional Analysis
~~~~~~~~~~~~~~~~~
RMSE and F1 computed per city to detect spatial generalisation failure.
Cities with few "Bad" events (e.g. Đà Nẵng) may show low Recall due to
class imbalance — identified here for further investigation.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

logger = logging.getLogger(__name__)

PM25_THRESHOLD = 35.4   # US EPA 24h NAAQS standard (µg/m³)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    RMSE = √[ (1/n) Σ (ŷᵢ − yᵢ)² ]
    """
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of Determination R².

    R² = 1 − SS_res / SS_tot
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_pred - y_true)))


# ---------------------------------------------------------------------------
# Improvement over baseline
# ---------------------------------------------------------------------------

def rmse_improvement(rmse_model: float, rmse_baseline: float) -> float:
    """
    Percentage improvement of model over baseline:

        Δ% = (1 − RMSE_model / RMSE_baseline) × 100
    """
    return (1.0 - rmse_model / (rmse_baseline + 1e-8)) * 100.0


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------

def classify(y: np.ndarray, threshold: float = PM25_THRESHOLD) -> np.ndarray:
    """
    Convert continuous PM2.5 values to binary health labels.

    Label mapping:
        0 → Good / Moderate  (PM2.5 < threshold)
        1 → Unhealthy / Bad  (PM2.5 ≥ threshold)
    """
    return (y >= threshold).astype(np.int8)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = PM25_THRESHOLD,
) -> Dict[str, object]:
    """
    Compute binary classification metrics using EPA threshold.

    Parameters
    ----------
    y_true    : ground-truth PM2.5 values (µg/m³, original scale)
    y_pred    : predicted PM2.5 values
    threshold : decision boundary (default 35.4 µg/m³)

    Returns
    -------
    dict with 'report' (str) and 'f1_bad' (float for class 1)
    """
    y_true_bin = classify(y_true, threshold)
    y_pred_bin = classify(y_pred, threshold)

    report = classification_report(
        y_true_bin, y_pred_bin,
        target_names=["Good/Moderate", "Unhealthy"],
        zero_division=0,
    )
    f1_bad = f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    recall_bad = _recall(y_true_bin, y_pred_bin)

    return {
        "report":     report,
        "f1_bad":     float(f1_bad),
        "recall_bad": float(recall_bad),
    }


def _recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall for class 1 (Unhealthy air)."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return float(tp / (tp + fn + 1e-8))


# ---------------------------------------------------------------------------
# Regional (per-city) breakdown
# ---------------------------------------------------------------------------

def evaluate_by_city(
    df_test: pd.DataFrame,
    y_pred_original_scale: np.ndarray,
    target_col: str = "pm2_5",
    city_col:   str = "city",
    threshold:  float = PM25_THRESHOLD,
) -> pd.DataFrame:
    """
    Compute RMSE, R², MAE, F1-Bad and Recall-Bad for each city.

    Parameters
    ----------
    df_test               : raw (original scale) test DataFrame with city labels
    y_pred_original_scale : model predictions on original PM2.5 scale
    target_col            : name of target column
    city_col              : name of city column
    threshold             : EPA PM2.5 decision threshold

    Returns
    -------
    pd.DataFrame indexed by city with columns: RMSE, R2, MAE, F1_Bad, Recall_Bad
    """
    df = df_test.copy().reset_index(drop=True)
    df["y_pred"] = y_pred_original_scale

    results = []
    for city, grp in df.groupby(city_col):
        yt = grp[target_col].values
        yp = grp["y_pred"].values

        y_t_bin = classify(yt, threshold)
        y_p_bin = classify(yp, threshold)

        results.append({
            "city":       city,
            "RMSE":       rmse(yt, yp),
            "R2":         r_squared(yt, yp),
            "MAE":        mae(yt, yp),
            "F1_Bad":     f1_score(y_t_bin, y_p_bin, pos_label=1, zero_division=0),
            "Recall_Bad": _recall(y_t_bin, y_p_bin),
            "n_samples":  len(yt),
            "pct_bad":    float(y_t_bin.mean() * 100),
        })

    result_df = pd.DataFrame(results).set_index("city")
    logger.info("Per-city evaluation:\n%s", result_df.to_string())
    return result_df


# ---------------------------------------------------------------------------
# Consolidated evaluation report
# ---------------------------------------------------------------------------

def full_evaluation(
    y_true:         np.ndarray,
    y_pred:         np.ndarray,
    y_baseline:     np.ndarray,
    df_test_raw:    pd.DataFrame,
    target_scaler,
    feature_names:  Optional[list] = None,
) -> Dict[str, object]:
    """
    Run the complete evaluation suite and return all metrics.

    This function operates on SCALED predictions and back-transforms to
    original scale before computing classification metrics.

    Parameters
    ----------
    y_true        : scaled ground-truth array
    y_pred        : scaled model predictions
    y_baseline    : scaled persistence baseline predictions
    df_test_raw   : original-scale test DataFrame
    target_scaler : fitted ZScoreScaler for the target column
    feature_names : optional list of feature names (for logging)

    Returns
    -------
    dict of all metrics
    """
    # ── Back-transform to µg/m³ ──────────────────────────────────────────
    def inv(arr):
        return target_scaler.inverse_transform(
            arr.reshape(-1, 1), ["pm2_5"]
        ).ravel()

    y_orig      = inv(y_true)
    yp_orig     = inv(y_pred)
    yb_orig     = inv(y_baseline)

    # ── Regression ────────────────────────────────────────────────────────
    model_rmse    = rmse(y_orig, yp_orig)
    baseline_rmse = rmse(y_orig, yb_orig)
    model_r2      = r_squared(y_orig, yp_orig)
    model_mae     = mae(y_orig, yp_orig)
    improvement   = rmse_improvement(model_rmse, baseline_rmse)

    # ── Classification ────────────────────────────────────────────────────
    cls_metrics = evaluate_classification(y_orig, yp_orig)

    logger.info("── Regression Metrics ──────────────────────────────")
    logger.info("  RMSE (model)    : %.4f µg/m³", model_rmse)
    logger.info("  RMSE (baseline) : %.4f µg/m³", baseline_rmse)
    logger.info("  RMSE improvement: %.2f%%",      improvement)
    logger.info("  R²              : %.4f",         model_r2)
    logger.info("  MAE             : %.4f µg/m³",  model_mae)
    logger.info("── Classification Report ────────────────────────────")
    logger.info("\n%s", cls_metrics["report"])

    return {
        "rmse_model":    model_rmse,
        "rmse_baseline": baseline_rmse,
        "rmse_improv":   improvement,
        "r2":            model_r2,
        "mae":           model_mae,
        "cls_report":    cls_metrics["report"],
        "f1_bad":        cls_metrics["f1_bad"],
        "recall_bad":    cls_metrics["recall_bad"],
        "y_orig":        y_orig,
        "yp_orig":       yp_orig,
        "yb_orig":       yb_orig,
    }