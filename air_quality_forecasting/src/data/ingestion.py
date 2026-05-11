"""
src/data/ingestion.py
=====================
Data Ingestion Layer — Open-Meteo API (Meteorology + Air Quality)

Mathematical rationale
----------------------
Hourly meteorological time-series must be *complete* (no missing timestamps)
before constructing lag features.  A single missing observation at hour t
causes lag_k(t+k) to silently reference the wrong physical timestep, biasing
the entire feature matrix.

Solution — "Time Backbone" technique:
  1. Generate a contiguous hourly DatetimeIndex (the backbone).
  2. Left-join raw API response onto the backbone.
  3. Gaps are NaN -> filled by linear interpolation (1st-order polynomial
     between neighboring observations), which preserves local gradient.
  4. Remaining edge-NaN (start/end) are forward/backward filled.

All timestamps are normalised to Asia/Ho_Chi_Minh (UTC+7) immediately after
download so downstream modules never handle timezone-naïve objects.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METEO_API   = "https://archive-api.open-meteo.com/v1/archive"
AQ_API      = "https://air-quality-api.open-meteo.com/v1/air-quality"

METEO_VARS  = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
    "boundary_layer_height",
    "shortwave_radiation",
]

AQ_VARS = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone"]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get_json(url: str, params: dict, retries: int = 5, backoff: float = 2.0) -> dict:
    """
    Robust HTTP GET with exponential backoff.

    Parameters
    ----------
    url     : endpoint URL
    params  : query-string parameters
    retries : number of retry attempts
    backoff : base wait (seconds), doubles each failure

    Returns
    -------
    dict : parsed JSON body
    """
    wait = backoff
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning("Attempt %d/%d failed: %s — retrying in %.1fs",
                           attempt + 1, retries, exc, wait)
            time.sleep(wait)
            wait *= 2.0
    raise RuntimeError(f"All {retries} attempts to {url} failed.")


# ---------------------------------------------------------------------------
# Time-Backbone builder
# ---------------------------------------------------------------------------

def _apply_time_backbone(df: pd.DataFrame,
                          start: str,
                          end: str,
                          tz: str = "Asia/Ho_Chi_Minh") -> pd.DataFrame:
    """
    Left-join `df` onto a complete hourly backbone, then interpolate gaps.

    The backbone guarantees:
        ∀ t ∈ [start, end],  ∃ exactly one row with timestamp t.

    Parameters
    ----------
    df    : raw DataFrame with DatetimeIndex (may contain gaps)
    start : ISO-8601 date string
    end   : ISO-8601 date string
    tz    : IANA timezone string

    Returns
    -------
    pd.DataFrame : gap-free DataFrame
    """
    backbone = pd.date_range(start=start, end=end, freq="h", tz=tz, name="timestamp")
    df = df.reindex(backbone)                        # left-join
    df = df.interpolate(method="linear", limit_direction="both")  # fill interior
    df = df.ffill().bfill()                          # fill edges
    df.index.name = "timestamp"                      # guarantee name survives reindex
    return df


# ---------------------------------------------------------------------------
# City-level fetcher
# ---------------------------------------------------------------------------

def fetch_city(city_key: str,
               cfg: dict,
               save_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Download and assemble a complete hourly DataFrame for one city.

    Data sources merged:
        • Open-Meteo Archive API  -> meteorological predictors
        • Open-Meteo AQ API       -> PM2.5 target + ancillary pollutants

    Parameters
    ----------
    city_key : key from config['cities'] (e.g. 'hanoi')
    cfg      : full config dict
    save_dir : if provided, serialise raw parquet here

    Returns
    -------
    pd.DataFrame : hourly, UTC+7, gap-free, with city column
    """
    city_cfg  = cfg["cities"][city_key]
    lat, lon  = city_cfg["lat"], city_cfg["lon"]
    start     = cfg["data"]["start_date"]
    end       = cfg["data"]["end_date"]
    tz        = cfg["timezone"]

    logger.info("Fetching meteorological data for %s ...", city_key)

    # ── Meteorological ──────────────────────────────────────────────────────
    meteo_params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start,
        "end_date":   end,
        "hourly":     ",".join(METEO_VARS),
        "timezone":   tz,
        "wind_speed_unit": "ms",
    }
    meteo_raw = _get_json(METEO_API, meteo_params)
    df_meteo  = _parse_hourly_response(meteo_raw, tz)

    logger.info("Fetching air-quality data for %s ...", city_key)

    # ── Air Quality ─────────────────────────────────────────────────────────
    aq_params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start,
        "end_date":   end,
        "hourly":     ",".join(AQ_VARS),
        "timezone":   tz,
    }
    aq_raw  = _get_json(AQ_API, aq_params)
    df_aq   = _parse_hourly_response(aq_raw, tz)

    # ── Merge on common backbone ─────────────────────────────────────────
    df = df_meteo.join(df_aq, how="outer")

    # ── Apply Time Backbone ──────────────────────────────────────────────
    df = _apply_time_backbone(df, start, end, tz)

    # ── Metadata ─────────────────────────────────────────────────────────
    df["city"] = city_key

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{city_key}_raw.csv"
        df.to_csv(out_path, encoding="utf-8")
        logger.info("Saved raw data -> %s", out_path)

    return df


def _parse_hourly_response(raw: dict, tz: str) -> pd.DataFrame:
    """
    Convert Open-Meteo JSON response to a timezone-aware DataFrame.

    Open-Meteo returns:
        { "hourly": { "time": [...], "var1": [...], "var2": [...] } }

    Parameters
    ----------
    raw : parsed JSON dict from Open-Meteo
    tz  : IANA timezone to localise timestamps

    Returns
    -------
    pd.DataFrame with DatetimeIndex (tz-aware)
    """
    hourly = raw.get("hourly", {})
    times  = pd.to_datetime(hourly.pop("time"))
    df     = pd.DataFrame(hourly, index=times)
    df.index = df.index.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    df.index.name = "timestamp"
    return df


# ---------------------------------------------------------------------------
# Multi-city ingestion
# ---------------------------------------------------------------------------

def ingest_all_cities(cfg: dict, save_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Fetch all four cities and concatenate into a single long-format DataFrame.

    The resulting DataFrame has a (timestamp, city) MultiIndex suitable for
    group-wise operations during feature engineering.

    Parameters
    ----------
    cfg      : full config dict
    save_dir : directory to persist raw parquet files

    Returns
    -------
    pd.DataFrame : concatenated, chronologically sorted
    """
    frames: List[pd.DataFrame] = []
    for city_key in cfg["cities"]:
        df = fetch_city(city_key, cfg, save_dir=save_dir)
        frames.append(df)

    combined = pd.concat(frames).sort_index()
    combined = combined.reset_index()  # timestamp -> column for easier manipulation

    logger.info("Ingestion complete. Shape: %s", combined.shape)
    return combined


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str = "config/config.yaml") -> dict:
    """Load YAML config and return as nested dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)