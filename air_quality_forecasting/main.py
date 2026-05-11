"""
main.py
=======
Master Pipeline — Air Quality PM2.5 Forecasting
================================================
Orchestrates the 6-stage research pipeline:

    Stage 1 -> Data Ingestion       (src.data.ingestion)
    Stage 2 -> Feature Engineering  (src.features.engineering)
    Stage 3 -> Preprocessing        (src.data.preprocessing)
    Stage 4 -> Model Training       (src.models.ridge_regression)
    Stage 5 -> Evaluation           (src.evaluation.metrics)
    Stage 6 -> Visualisation        (src.visualization.plots)

Usage:
    python main.py [--config config/config.yaml] [--skip-fetch]

Flags:
    --config      : path to YAML configuration file (default: config/config.yaml)
    --skip-fetch  : load raw data from disk instead of calling the API
    --alpha       : override Ridge λ (e.g. --alpha 0.5)
    --lr          : override learning rate (e.g. --lr 0.005)
"""

from __future__ import annotations

import argparse
import io
import logging
import sys

import pandas as pd
from pathlib import Path

import numpy as np

# ── Logging setup — force UTF-8 on Windows (avoids cp1252 UnicodeEncodeError) ──
Path("outputs").mkdir(exist_ok=True)

_stream_handler = logging.StreamHandler(
    stream=io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "buffer") else sys.stdout
)
_file_handler = logging.FileHandler("outputs/pipeline.log", mode="w", encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[_stream_handler, _file_handler],
)
logger = logging.getLogger("main")

# ── Project imports ────────────────────────────────────────────────────────
from src.data.ingestion     import load_config, ingest_all_cities
from src.features.engineering import build_features
from src.data.preprocessing  import preprocess
from src.models.ridge_regression import RidgeRegression
from src.models.baseline         import PersistenceBaseline
from src.evaluation.metrics      import full_evaluation, evaluate_by_city
from src.visualization.plots     import generate_all_figures


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PM2.5 Forecasting Pipeline")
    p.add_argument("--config",     default="config/config.yaml")
    p.add_argument("--skip-fetch", action="store_true",
                   help="Skip API calls; load parquet from data/raw/")
    p.add_argument("--alpha",      type=float, default=None,
                   help="Override Ridge α (L2 strength)")
    p.add_argument("--lr",         type=float, default=None,
                   help="Override learning rate η")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def stage1_ingest(cfg: dict, skip_fetch: bool) -> "pd.DataFrame":
    """Download or load raw data."""
    raw_dir = Path(cfg["paths"]["raw_data"])

    if skip_fetch:
        logger.info("Stage 1 -- Loading cached raw data from %s", raw_dir)
        frames = []
        for city_key in cfg["cities"]:
            path = raw_dir / f"{city_key}_raw.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"Cache miss: {path}. Run without --skip-fetch first."
                )
            _df = pd.read_csv(path, index_col=0, parse_dates=True)
            # When saved from a DatetimeIndex, timestamp is the index col.
            # reset_index() promotes it back to a regular column.
            if _df.index.name == "timestamp":
                _df = _df.reset_index()
            frames.append(_df)
        df = pd.concat(frames).sort_values("timestamp").reset_index(drop=True)
    else:
        logger.info("Stage 1 — Fetching data from Open-Meteo API ...")
        df = ingest_all_cities(cfg, save_dir=raw_dir)

    logger.info("Stage 1 complete. Shape: %s", df.shape)
    return df


def stage2_features(df: "pd.DataFrame", cfg: dict) -> "pd.DataFrame":
    """Construct the full feature matrix."""
    logger.info("Stage 2 — Feature Engineering ...")
    df_feat = build_features(df, cfg)
    logger.info("Stage 2 complete. Shape: %s", df_feat.shape)
    return df_feat


def stage3_preprocess(df: "pd.DataFrame", cfg: dict) -> dict:
    """Split, scale, encode."""
    logger.info("Stage 3 — Preprocessing (split + scale + OHE) ...")
    model_dir = Path(cfg["paths"]["models"])
    artefacts = preprocess(df, cfg, save_dir=model_dir)
    logger.info("Stage 3 complete.")
    return artefacts


def stage4_train(artefacts: dict, cfg: dict, alpha_override=None, lr_override=None):
    """Train Ridge and Baseline models."""
    logger.info("Stage 4 — Model Training ...")

    alpha = alpha_override or cfg["model"]["alpha"]
    lr    = lr_override    or cfg["model"]["learning_rate"]

    model = RidgeRegression(
        alpha         = alpha,
        learning_rate = lr,
        max_epochs    = cfg["model"]["max_epochs"],
        batch_size    = cfg["model"]["batch_size"],
        patience      = cfg["model"]["patience"],
        tol           = cfg["model"]["tol"],
    )
    model.fit(
        artefacts["X_train"], artefacts["y_train"],
        artefacts["X_val"],   artefacts["y_val"],
    )

    # Baseline
    baseline = PersistenceBaseline()
    baseline.fit(artefacts["y_train"])

    # Save model
    model_path = Path(cfg["paths"]["models"]) / "ridge_model.pkl"
    model.save(model_path)

    logger.info("Stage 4 complete. Best epoch: %d", model.best_epoch_)
    return model, baseline


def stage5_evaluate(model, baseline, artefacts: dict, cfg: dict) -> dict:
    """Run all evaluation metrics."""
    logger.info("Stage 5 — Evaluation ...")

    X_test  = artefacts["X_test"]
    y_test  = artefacts["y_test"]

    y_pred     = model.predict(X_test)
    y_baseline = baseline.predict(y_test)

    eval_results = full_evaluation(
        y_true      = y_test,
        y_pred      = y_pred,
        y_baseline  = y_baseline,
        df_test_raw = artefacts["df_test_raw"],
        target_scaler = artefacts["target_scaler"],
        feature_names = artefacts["feature_names"],
    )

    city_metrics = evaluate_by_city(
        df_test               = artefacts["df_test_raw"],
        y_pred_original_scale = eval_results["yp_orig"],
        threshold             = cfg["threshold"]["pm25_bad"],
    )

    # Save summary CSV
    report_dir = Path(cfg["paths"]["reports"])
    report_dir.mkdir(parents=True, exist_ok=True)
    city_metrics.to_csv(report_dir / "city_metrics.csv")
    _write_summary(eval_results, report_dir / "summary_metrics.txt")

    logger.info("Stage 5 complete.")
    return eval_results, city_metrics


def stage6_visualise(model, eval_results, city_metrics, artefacts, cfg) -> None:
    """Generate all publication-grade figures."""
    logger.info("Stage 6 — Visualisation ...")
    fig_dir = Path(cfg["paths"]["figures"])
    generate_all_figures(
        model         = model,
        history       = model.training_history(),
        eval_results  = eval_results,
        feature_names = artefacts["feature_names"],
        df_test_raw   = artefacts["df_test_raw"],
        city_metrics  = city_metrics,
        save_dir      = fig_dir,
    )
    logger.info("Stage 6 complete.")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _write_summary(eval_results: dict, path: Path) -> None:
    """Write a human-readable metrics summary."""
    lines = [
        "=" * 60,
        "  PM2.5 Forecasting — Evaluation Summary",
        "=" * 60,
        f"  RMSE (model)     : {eval_results['rmse_model']:.4f} µg/m³",
        f"  RMSE (baseline)  : {eval_results['rmse_baseline']:.4f} µg/m³",
        f"  RMSE improvement : {eval_results['rmse_improv']:.2f}%",
        f"  R²               : {eval_results['r2']:.4f}",
        f"  MAE              : {eval_results['mae']:.4f} µg/m³",
        "",
        "  Classification (EPA threshold = 35.4 µg/m³):",
        f"  F1-Score (Bad)   : {eval_results['f1_bad']:.4f}",
        f"  Recall   (Bad)   : {eval_results['recall_bad']:.4f}",
        "",
        "  Full Classification Report:",
        eval_results["cls_report"],
        "=" * 60,
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Summary saved -> %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    # Ensure output dirs exist
    for key in ("raw_data", "processed_data", "models", "figures", "reports"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)

    df          = stage1_ingest(cfg, args.skip_fetch)
    df_feat     = stage2_features(df, cfg)
    artefacts   = stage3_preprocess(df_feat, cfg)
    model, base = stage4_train(artefacts, cfg, args.alpha, args.lr)
    evals, city = stage5_evaluate(model, base, artefacts, cfg)
    stage6_visualise(model, evals, city, artefacts, cfg)

    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()