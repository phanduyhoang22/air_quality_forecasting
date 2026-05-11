"""
src/features/engineering.py
============================
Feature Engineering Layer — Temporal · Meteorological · Time-Series Memory

Mathematical foundations
------------------------

1. Trigonometric Temporal Encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Cyclic variables (hour h ∈ [0,23], month m ∈ [1,12]) have a boundary
discontinuity if used as raw integers: hour 23 is "far" from hour 0 in
Euclidean space but physically adjacent.  Trigonometric projection onto the
unit circle resolves this:

    sin_h = sin(2π · h / 24),   cos_h = cos(2π · h / 24)

The two components uniquely identify each hour while preserving periodicity
(‖[sin_h, cos_h]‖₂ = 1 always), and the dot-product between consecutive
hours ≈ 1 (close) vs. distant hours ≈ −1 (far).

2. Meteorological Interaction Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
• Heat-Humidity Index: T × RH — proxy for the thermal-moisture energy that
  promotes aerosol nucleation and hygroscopic growth of PM2.5 particles.

• Pressure Derivative (Δp): Change in surface pressure over lag_1 window
  (past only).  Falling pressure signals an approaching low-pressure system
  associated with increased pollutant accumulation.
  Anti-leakage: Δp = pressure_lag1 − pressure_lag2  (no current value used).

• Stagnation Coefficient: S = 1 / (wind_speed + ε)  where ε = 1e-8.
  Under stagnant conditions (calm wind), boundary-layer dilution fails,
  causing pollutant build-up.  S is a physically motivated proxy for
  atmospheric mixing capacity.

3. Lag Features & Rolling Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The PM2.5 process exhibits strong autocorrelation at hours 1–3 (short-term
persistence), diurnal cycles (lag 24), weekly patterns (lag 168).  Including
raw lags y_{t−k} provides the model with direct historical signal.

Rolling statistics (mean, std over windows w) capture the local trend and
volatility regime — essential for non-stationary air-quality episodes.

4. Anti-Leakage Guarantee
~~~~~~~~~~~~~~~~~~~~~~~~~~
Delta / percentage-change features derived from the CURRENT timestep t would
be equivalent to using the target y_t itself (since y_t appears in lags once
computed).  ALL derived features use only y_{t−1}, y_{t−2}, … ensuring strict
causal ordering:

    Δ₁₋₂ = lag_1 − lag_2   ✓  (valid: fully in the past)
    Δ₀₋₁ = y_t − lag_1     ✗  (FORBIDDEN: reveals current target)
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COL = "pm2_5"


# ---------------------------------------------------------------------------
# Temporal Features
# ---------------------------------------------------------------------------

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode cyclic temporal components via trigonometric projection.

    Derived columns:
        sin_hour, cos_hour       — diurnal cycle (period 24h)
        sin_month, cos_month     — seasonal cycle (period 12 months)
        sin_dow, cos_dow         — day-of-week cycle (period 7 days)
        is_weekend               — binary: Saturday/Sunday

    Parameters
    ----------
    df : DataFrame with a 'timestamp' datetime column (tz-aware)

    Returns
    -------
    df : DataFrame with new temporal columns
    """
    ts = pd.to_datetime(df["timestamp"])

    # Diurnal
    df["sin_hour"]  = np.sin(2 * np.pi * ts.dt.hour  / 24)
    df["cos_hour"]  = np.cos(2 * np.pi * ts.dt.hour  / 24)

    # Seasonal (month)
    df["sin_month"] = np.sin(2 * np.pi * ts.dt.month / 12)
    df["cos_month"] = np.cos(2 * np.pi * ts.dt.month / 12)

    # Weekly
    df["sin_dow"]   = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    df["cos_dow"]   = np.cos(2 * np.pi * ts.dt.dayofweek / 7)

    # Calendar flag
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(np.int8)

    logger.debug("Temporal features added.")
    return df


# ---------------------------------------------------------------------------
# Meteorological Derived Features
# ---------------------------------------------------------------------------

def add_meteo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct physically motivated meteorological interaction features.

    Derived columns:
        heat_humidity   — T × RH (thermal-moisture coupling)
        stagnation      — 1 / (wind_speed + ε)
        pressure_delta  — Δp from past lags (anti-leakage)
        wind_u          — zonal wind component: U = -V · sin(θ)
        wind_v          — meridional component: V = -V · cos(θ)

    Parameters
    ----------
    df : DataFrame with meteorological columns

    Returns
    -------
    df : DataFrame with new derived columns
    """
    eps = 1e-8

    # Heat-Humidity index
    df["heat_humidity"] = (
        df["temperature_2m"] * df["relative_humidity_2m"]
    )

    # Stagnation coefficient  S = 1/(v + ε)
    df["stagnation"] = 1.0 / (df["wind_speed_10m"].clip(lower=0.0) + eps)

    # Wind vector decomposition (θ in degrees → radians)
    theta_rad  = np.deg2rad(df["wind_direction_10m"])
    df["wind_u"] = -df["wind_speed_10m"] * np.sin(theta_rad)
    df["wind_v"] = -df["wind_speed_10m"] * np.cos(theta_rad)

    # Dew-point depression (proxy for relative humidity in physical units)
    df["dewpoint_depression"] = df["temperature_2m"] - df["dew_point_2m"]

    logger.debug("Meteorological features added.")
    return df


# ---------------------------------------------------------------------------
# Lag & Rolling Features (Anti-Leakage enforced)
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame,
                      lag_hours: List[int],
                      group_col: str = "city") -> pd.DataFrame:
    """
    Construct lag and delta features per city group.

    Lag features:
        pm25_lag_{k} = y_{t−k}   for k ∈ lag_hours

    Delta features (anti-leakage — computed only from past lags):
        pm25_delta_{k1}_{k2} = pm25_lag_{k1} − pm25_lag_{k2}
        Only created for consecutive pairs in lag_hours.

    Parameters
    ----------
    df        : DataFrame sorted chronologically, grouped by city
    lag_hours : list of integer lag offsets (e.g. [1, 2, 3, 6, 12, 24, 48, 72, 168])
    group_col : column to group by before shifting

    Returns
    -------
    df : DataFrame with new lag/delta columns
    """
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    lag_col_names = []
    for k in lag_hours:
        col = f"pm25_lag_{k}"
        df[col] = df.groupby(group_col)[TARGET_COL].shift(k)
        lag_col_names.append(col)

    # Delta features between consecutive lags (all strictly in the past)
    sorted_lags = sorted(lag_hours)
    for i in range(len(sorted_lags) - 1):
        k1, k2 = sorted_lags[i], sorted_lags[i + 1]
        col_delta = f"pm25_delta_{k1}_{k2}"
        df[col_delta] = df[f"pm25_lag_{k1}"] - df[f"pm25_lag_{k2}"]

    logger.debug("Lag features added for lags: %s", lag_hours)
    return df


def add_rolling_features(df: pd.DataFrame,
                           windows: List[int],
                           group_col: str = "city") -> pd.DataFrame:
    """
    Compute rolling mean and standard deviation of PM2.5 per city.

    Rolling statistics are shifted by 1 to prevent current-hour leakage:
        rolling_mean_w(t) = mean(y_{t−w}, …, y_{t−1})

    This is equivalent to a closed-right, open-current window.

    Parameters
    ----------
    df      : DataFrame with 'pm2_5' column
    windows : rolling window sizes in hours
    group_col : grouping column

    Returns
    -------
    df : DataFrame with rolling mean/std columns
    """
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    for w in windows:
        grp = df.groupby(group_col)[TARGET_COL]
        # Shift(1) ensures the window does NOT include current observation
        df[f"pm25_roll_mean_{w}h"] = (
            grp.shift(1).rolling(window=w, min_periods=1).mean().values
        )
        df[f"pm25_roll_std_{w}h"] = (
            grp.shift(1).rolling(window=w, min_periods=2).std().fillna(0).values
        )

    logger.debug("Rolling features added for windows: %s", windows)
    return df


# ---------------------------------------------------------------------------
# Pressure Derivative (Anti-Leakage)
# ---------------------------------------------------------------------------

def add_pressure_features(df: pd.DataFrame,
                            group_col: str = "city") -> pd.DataFrame:
    """
    Compute rate-of-change of surface pressure from lagged values only.

    Δp = pressure_lag1 − pressure_lag2
       = p_{t−1} − p_{t−2}

    This proxy captures baric tendency — a key meteorological signal:
        Δp < 0  ↔  falling pressure → approaching trough → poor mixing
        Δp > 0  ↔  rising pressure  → anticyclone → cleaner air

    Parameters
    ----------
    df        : DataFrame with 'surface_pressure'
    group_col : city grouping

    Returns
    -------
    df : DataFrame with 'pressure_lag1', 'pressure_lag2', 'pressure_delta'
    """
    grp = df.groupby(group_col)["surface_pressure"]
    df["pressure_lag1"]  = grp.shift(1)
    df["pressure_lag2"]  = grp.shift(2)
    df["pressure_delta"] = df["pressure_lag1"] - df["pressure_lag2"]
    return df


# ---------------------------------------------------------------------------
# Master Feature Pipeline
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Orchestrate the full feature engineering pipeline.

    Pipeline order:
        1. Temporal encoding
        2. Meteorological interactions
        3. Pressure derivatives (anti-leakage)
        4. Lag features (anti-leakage)
        5. Rolling statistics (anti-leakage)

    Parameters
    ----------
    df  : raw ingested DataFrame (output of src.data.ingestion)
    cfg : full config dict

    Returns
    -------
    pd.DataFrame : enriched feature matrix, NaN rows at edges (will be
                   dropped downstream in preprocessing)
    """
    logger.info("Building features. Input shape: %s", df.shape)

    df = add_temporal_features(df)
    df = add_meteo_features(df)
    df = add_pressure_features(df)
    df = add_lag_features(df,
                           lag_hours=cfg["features"]["lag_hours"])
    df = add_rolling_features(df,
                               windows=cfg["features"]["rolling_windows"])

    # Drop any intermediate helper columns not needed as features
    drop_candidates = ["wind_direction_10m"]
    df = df.drop(columns=[c for c in drop_candidates if c in df.columns])

    logger.info("Feature engineering complete. Output shape: %s", df.shape)
    return df