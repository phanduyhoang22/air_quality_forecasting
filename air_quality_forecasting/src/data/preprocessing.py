"""
src/data/preprocessing.py
==========================
Preprocessing Layer — Chronological Split · Z-Score Scaling · One-Hot Encoding

Mathematical rationale
----------------------

Chronological Split
~~~~~~~~~~~~~~~~~~~
Unlike i.i.d. data, time-series violates the exchangeability assumption.
Random shuffling creates "future leakage": the model trains on t+k while
predicting t, artificially inflating validation metrics.
The correct partition is:

    [0 … 70%)  -> Train    (historical signal)
    [70%…85%)  -> Val      (hyper-parameter tuning / early stopping)
    [85%…100%) -> Test     (held-out, never seen during development)

Z-Score Normalisation
~~~~~~~~~~~~~~~~~~~~~
Ridge Regression's L2 penalty is scale-sensitive.  A feature x_j with large
variance dominates the squared residual, biasing the regularisation path.
Standardisation transforms each feature to zero mean and unit variance:

    x̃_j = (x_j − μ_j) / σ_j ,   μ_j, σ_j computed on TRAIN only.

Applying train statistics to val/test prevents leakage of future distributional
information.  Saved as `scaler_stats.pkl` for production inference.

One-Hot Encoding (Drop-First)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
With C = 4 cities, full OHE creates 4 binary columns that are perfectly
collinear (their sum = 1 always), causing rank deficiency in X^T X.
Dropping one reference category (e.g. 'hanoi') yields C−1 = 3 orthogonal
dummy variables that encode the same information without multicollinearity.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COL = "pm2_5"
CITY_COL   = "city"


# ---------------------------------------------------------------------------
# Chronological Split
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partition `df` into train / val / test by temporal order.

    Indices:
        train  -> rows [0,   n*train_ratio)
        val    -> rows [n*train_ratio, n*(train_ratio+val_ratio))
        test   -> rows [n*(train_ratio+val_ratio), n)

    Parameters
    ----------
    df          : full DataFrame sorted chronologically
    train_ratio : fraction allocated to training
    val_ratio   : fraction allocated to validation

    Returns
    -------
    (df_train, df_val, df_test) tuple
    """
    n = len(df)
    i_val  = int(n * train_ratio)
    i_test = int(n * (train_ratio + val_ratio))

    df_train = df.iloc[:i_val].copy()
    df_val   = df.iloc[i_val:i_test].copy()
    df_test  = df.iloc[i_test:].copy()

    logger.info(
        "Split -> Train: %d  Val: %d  Test: %d  (total: %d)",
        len(df_train), len(df_val), len(df_test), n,
    )
    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# Z-Score Scaler
# ---------------------------------------------------------------------------

class ZScoreScaler:
    """
    Column-wise Z-Score scaler.

    Only `fit` on training data; `transform` applies the same (μ, σ) to
    all subsequent sets to prevent distributional leakage.

    Attributes
    ----------
    mean_ : pd.Series  — per-column mean (μ)
    std_  : pd.Series  — per-column std  (σ), clipped to ≥ 1e-8 to avoid /0
    """

    def __init__(self) -> None:
        self.mean_: Optional[pd.Series] = None
        self.std_:  Optional[pd.Series] = None
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, columns: List[str]) -> "ZScoreScaler":
        """
        Compute μ and σ from `df[columns]`.  Only call on training split.

        Parameters
        ----------
        df      : training DataFrame
        columns : list of numeric columns to scale

        Returns
        -------
        self
        """
        self.mean_ = df[columns].mean()
        self.std_  = df[columns].std().clip(lower=1e-8)
        self._fitted = True
        logger.info("Scaler fitted on %d columns.", len(columns))
        return self

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply z-score transform: x̃ = (x − μ) / σ.

        Parameters
        ----------
        df : DataFrame to transform (in-place on a copy)

        Returns
        -------
        pd.DataFrame : standardised copy
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .transform().")

        cols = self.mean_.index.tolist()
        out  = df.copy()
        out[cols] = (df[cols] - self.mean_) / self.std_
        return out

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Convenience: fit then transform."""
        return self.fit(df, columns).transform(df)

    # ------------------------------------------------------------------
    def inverse_transform(self, arr: np.ndarray, columns: List[str]) -> np.ndarray:
        """
        Undo z-score: x = x̃ · σ + μ.

        Useful for back-transforming model predictions to original scale.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .inverse_transform().")

        mean = self.mean_[columns].values
        std  = self.std_[columns].values
        return arr * std + mean

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Serialise scaler statistics to pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"mean": self.mean_, "std": self.std_}, f)
        logger.info("Scaler saved -> %s", path)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "ZScoreScaler":
        """Deserialise scaler from pickle."""
        with open(path, "rb") as f:
            stats = pickle.load(f)
        scaler = cls()
        scaler.mean_   = stats["mean"]
        scaler.std_    = stats["std"]
        scaler._fitted = True
        return scaler


# ---------------------------------------------------------------------------
# One-Hot Encoding (Drop-First)
# ---------------------------------------------------------------------------

def one_hot_encode_city(df: pd.DataFrame,
                         reference: str = "hanoi") -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encode the 'city' column, dropping the reference category.

    With cities {hanoi, haiphong, danang, hcmc}, we get three binary columns:
        city_haiphong, city_danang, city_hcmc
    The intercept term in Ridge Regression implicitly captures 'hanoi'.

    Parameters
    ----------
    df        : DataFrame with a 'city' string column
    reference : city key to drop (serves as baseline)

    Returns
    -------
    (df_encoded, dummy_columns) tuple
    """
    dummies = pd.get_dummies(df[CITY_COL], prefix="city", drop_first=False)
    # Manually drop reference to be explicit about which category is baseline
    drop_col = f"city_{reference}"
    if drop_col in dummies.columns:
        dummies = dummies.drop(columns=[drop_col])

    dummy_cols = dummies.columns.tolist()
    df_out = pd.concat([df.drop(columns=[CITY_COL]), dummies], axis=1)
    return df_out, dummy_cols


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(
    df: pd.DataFrame,
    cfg: dict,
    save_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Orchestrate split -> OHE -> scale -> return artefacts.

    Parameters
    ----------
    df       : feature-engineered DataFrame (output of src.features.engineering)
    cfg      : full config dict
    save_dir : directory for scaler_stats.pkl

    Returns
    -------
    dict with keys:
        X_train, y_train, X_val, y_val, X_test, y_test  (np.ndarray)
        scaler    : fitted ZScoreScaler instance
        feature_names : list[str]
        df_test   : raw test DataFrame (for region-level evaluation)
    """
    # ── Sort chronologically ─────────────────────────────────────────────
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Drop rows where target is NaN (edges from lag construction) ──────
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # ── One-hot encode city ──────────────────────────────────────────────
    # Preserve city labels BEFORE OHE (needed for per-city evaluation)
    city_labels = df[CITY_COL].copy()

    df, dummy_cols = one_hot_encode_city(df, reference="hanoi")
    logger.info("City dummy columns: %s", dummy_cols)

    # ── Identify feature columns ─────────────────────────────────────────
    non_feature_cols = {TARGET_COL, "timestamp", "pm10",
                        "nitrogen_dioxide", "ozone"}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    # ── Drop any remaining NaN rows (lag/rolling edges) ──────────────────
    df = df.dropna(subset=feature_cols + [TARGET_COL]).reset_index(drop=True)

    # ── Chronological split ───────────────────────────────────────────────
    df_train, df_val, df_test = chronological_split(
        df,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio  =cfg["data"]["val_ratio"],
    )

    # ── Z-Score scaling (fit on train only) ──────────────────────────────
    scaler = ZScoreScaler()
    scaler.fit(df_train, columns=feature_cols)

    df_train_s = scaler.transform(df_train)
    df_val_s   = scaler.transform(df_val)
    df_test_s  = scaler.transform(df_test)

    # ── Also scale the TARGET for model training ─────────────────────────
    # (predictions are back-transformed before evaluation)
    target_scaler = ZScoreScaler()
    target_scaler.fit(df_train, columns=[TARGET_COL])

    df_train_s = target_scaler.transform(df_train_s)
    df_val_s   = target_scaler.transform(df_val_s)
    df_test_s  = target_scaler.transform(df_test_s)

    # ── Extract numpy arrays ──────────────────────────────────────────────
    def _to_arrays(frame: pd.DataFrame):
        X = frame[feature_cols].values.astype(np.float64)
        y = frame[TARGET_COL].values.astype(np.float64)
        return X, y

    X_train, y_train = _to_arrays(df_train_s)
    X_val,   y_val   = _to_arrays(df_val_s)
    X_test,  y_test  = _to_arrays(df_test_s)

    # ── Save scaler ───────────────────────────────────────────────────────
    if save_dir is not None:
        save_dir = Path(save_dir)
        scaler.save(save_dir / "scaler_stats.pkl")
        target_scaler.save(save_dir / "target_scaler_stats.pkl")

    logger.info("X_train: %s  X_val: %s  X_test: %s",
                X_train.shape, X_val.shape, X_test.shape)

    return {
        "X_train":       X_train,
        "y_train":       y_train,
        "X_val":         X_val,
        "y_val":         y_val,
        "X_test":        X_test,
        "y_test":        y_test,
        "scaler":        scaler,
        "target_scaler": target_scaler,
        "feature_names": feature_cols,
        "df_test_raw":   df_test.assign(
            city=city_labels.iloc[len(df_train)+len(df_val):len(df_train)+len(df_val)+len(df_test)].values
        ),   # original scale test set with city labels restored
    }