"""
models/dashboard_export.py
Backend export helpers — write dashboard-ready CSVs without building the UI.

The dashboard reads these files only; it does not run training live.

Exports:
  data/dashboard/latest_predictions.csv
  data/dashboard/latest_rl_trades.csv      (populated by rl_agent.py)
  data/dashboard/model_metrics.csv
  data/dashboard/feature_status.csv
  data/dashboard/latest_news_sentiment.csv  (populated by gdelt_sentiment.py)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2.extras
from loguru import logger

DASHBOARD_DIR = Path(__file__).parent.parent / "data" / "dashboard"
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

SAVED_DIR     = Path(__file__).parent / "saved"


# ── 1. latest_predictions.csv ─────────────────────────────────────────────────

def export_predictions(assets: list[str] = None) -> None:
    """
    Read transformer_predictions from PostgreSQL and write latest_predictions.csv.
    Columns: timestamp_utc, asset, transformer_prediction, transformer_confidence,
             actual_direction, sentiment_score, news_volume.
    """
    assets = assets or ["BTC", "ETH", "GOLD", "OIL"]
    frames = []

    try:
        from db.connection import get_conn
        for asset in assets:
            sql = """
                SELECT tp.ts,
                       tp.direction   AS transformer_prediction,
                       tp.confidence  AS transformer_confidence,
                       tp.p_down, tp.p_neutral, tp.p_up,
                       t.direction_1h AS actual_direction
                FROM   transformer_predictions tp
                LEFT JOIN targets t
                       ON t.asset = tp.asset AND t.ts = tp.ts
                WHERE  tp.asset = %s
                ORDER  BY tp.ts DESC
                LIMIT  2000
            """
            with get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql, (asset,))
                    rows = cur.fetchall()
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["asset"] = asset
            df["ts"]    = pd.to_datetime(df["ts"], utc=True)
            df.rename(columns={"ts": "timestamp_utc"}, inplace=True)

            # Add GDELT sentiment if available
            proc_path = (
                Path(__file__).parent.parent / "data" / "processed"
                / f"gdelt_{asset.lower()}_sentiment_features.csv"
            )
            if proc_path.exists():
                gdelt_df = pd.read_csv(proc_path)
                gdelt_df["timestamp_utc"] = pd.to_datetime(
                    gdelt_df["timestamp_utc"], utc=True, errors="coerce"
                )
                gdelt_df = gdelt_df.set_index("timestamp_utc")
                pfx = asset.lower()
                if f"{pfx}_gdelt_sentiment_mean" in gdelt_df.columns:
                    df = df.set_index("timestamp_utc")
                    df["sentiment_score"] = gdelt_df[f"{pfx}_gdelt_sentiment_mean"].reindex(
                        df.index
                    ).fillna(0.0)
                    df["news_volume"] = gdelt_df[f"{pfx}_gdelt_news_volume"].reindex(
                        df.index
                    ).fillna(0.0)
                    df = df.reset_index()
                else:
                    df["sentiment_score"] = 0.0
                    df["news_volume"]     = 0.0
            else:
                df["sentiment_score"] = 0.0
                df["news_volume"]     = 0.0

            frames.append(df)

    except Exception as e:
        logger.warning(f"Could not load transformer_predictions: {e}")

    if not frames:
        logger.warning("No predictions to export.")
        return

    out = pd.concat(frames, ignore_index=True)
    cols = [
        "timestamp_utc", "asset",
        "transformer_prediction", "transformer_confidence",
        "actual_direction", "sentiment_score", "news_volume",
    ]
    out = out[[c for c in cols if c in out.columns]]
    path = DASHBOARD_DIR / "latest_predictions.csv"
    out.to_csv(path, index=False)
    logger.success(f"Predictions → {path} ({len(out)} rows)")


# ── 2. model_metrics.csv ──────────────────────────────────────────────────────

def export_model_metrics(assets: list[str] = None) -> None:
    """
    Aggregate metrics from saved model metadata and write model_metrics.csv.
    """
    assets = assets or ["BTC", "ETH", "GOLD", "OIL"]
    rows   = []

    for asset in assets:
        # Ridge
        ridge_path = SAVED_DIR / f"ridge_{asset.lower()}.pkl"
        if ridge_path.exists():
            try:
                import joblib
                ridge_data = joblib.load(ridge_path)
                metrics    = ridge_data.get("metrics", {})
                test_m     = metrics.get("test", {})
                rows.append({
                    "asset":                asset,
                    "model":               "Ridge",
                    "accuracy":            None,
                    "macro_f1":            None,
                    "weighted_f1":         None,
                    "mae":                 test_m.get("mae"),
                    "directional_accuracy": test_m.get("directional_accuracy"),
                    "total_return":        None,
                    "max_drawdown":        None,
                    "sharpe_like":         None,
                    "win_rate":            None,
                })
            except Exception:
                pass

        # Transformer
        trans_path = SAVED_DIR / f"transformer_{asset.lower()}.pt"
        if trans_path.exists():
            try:
                import torch
                ckpt = torch.load(str(trans_path), map_location="cpu")
                rows.append({
                    "asset":                asset,
                    "model":               "Transformer",
                    "accuracy":            ckpt.get("test_accuracy"),
                    "macro_f1":            ckpt.get("test_f1_macro"),
                    "weighted_f1":         ckpt.get("test_f1_weighted"),
                    "mae":                 None,
                    "directional_accuracy": None,
                    "total_return":        None,
                    "max_drawdown":        None,
                    "sharpe_like":         None,
                    "win_rate":            None,
                })
            except Exception:
                pass

        # RL PPO
        ppo_meta = SAVED_DIR / f"ppo_{asset.lower()}_metadata.json"
        if ppo_meta.exists():
            try:
                with open(ppo_meta) as f:
                    meta = json.load(f)
                rows.append({
                    "asset":                asset,
                    "model":               "PPO_RL",
                    "accuracy":            None,
                    "macro_f1":            None,
                    "weighted_f1":         None,
                    "mae":                 None,
                    "directional_accuracy": None,
                    "total_return":        meta.get("total_pnl"),
                    "max_drawdown":        meta.get("max_drawdown"),
                    "sharpe_like":         meta.get("sharpe"),
                    "win_rate":            meta.get("win_rate"),
                })
            except Exception:
                pass

    if not rows:
        logger.warning("No model metrics to export.")
        return

    df   = pd.DataFrame(rows)
    path = DASHBOARD_DIR / "model_metrics.csv"
    df.to_csv(path, index=False)
    logger.success(f"Model metrics → {path} ({len(df)} rows)")


# ── 3. feature_status.csv ─────────────────────────────────────────────────────

def export_feature_status(
    assets: list[str] = None,
    use_gdelt_sentiment: bool = False,
    use_kaggle_sentiment: bool = False,
) -> None:
    """
    Check feature schema health per asset and write feature_status.csv.
    """
    from models.feature_schema import (
        get_feature_columns, GDELT_FEATURES,
    )
    assets = assets or ["BTC", "ETH", "GOLD", "OIL"]
    rows   = []

    for asset in assets:
        expected = get_feature_columns(
            asset, model="rl",
            use_gdelt_sentiment=use_gdelt_sentiment,
            use_kaggle_sentiment=use_kaggle_sentiment,
        )

        # Check scaler schema
        meta_path = SAVED_DIR / f"feature_scaler_{asset.lower()}_schema.json"
        missing_features = []
        extra_features   = []
        schema_valid     = True

        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            saved_cols   = meta.get("feature_columns", [])
            missing_features = [c for c in expected  if c not in saved_cols]
            extra_features   = [c for c in saved_cols if c not in expected]
            schema_valid     = not missing_features and not extra_features

        # Check processed GDELT file
        proc_path = (
            Path(__file__).parent.parent / "data" / "processed"
            / f"gdelt_{asset.lower()}_sentiment_features.csv"
        )
        gdelt_available = proc_path.exists()

        rows.append({
            "asset":                    asset,
            "feature_count":           len(expected),
            "sentiment_enabled":       use_gdelt_sentiment or use_kaggle_sentiment,
            "gdelt_sentiment_enabled": use_gdelt_sentiment,
            "kaggle_sentiment_enabled": use_kaggle_sentiment,
            "gdelt_file_exists":       gdelt_available,
            "missing_features":        str(missing_features) if missing_features else "",
            "extra_features":          str(extra_features)   if extra_features   else "",
            "schema_valid":            schema_valid,
        })

    df   = pd.DataFrame(rows)
    path = DASHBOARD_DIR / "feature_status.csv"
    df.to_csv(path, index=False)
    logger.success(f"Feature status → {path}")


# ── Orchestration ──────────────────────────────────────────────────────────────

def export_all(
    assets: list[str] = None,
    use_gdelt_sentiment: bool = False,
    use_kaggle_sentiment: bool = False,
) -> None:
    """Export all dashboard backend files."""
    logger.info("Exporting dashboard backend files…")
    export_predictions(assets)
    export_model_metrics(assets)
    export_feature_status(
        assets,
        use_gdelt_sentiment=use_gdelt_sentiment,
        use_kaggle_sentiment=use_kaggle_sentiment,
    )
    logger.success("Dashboard export complete.")
    logger.info(f"Files in {DASHBOARD_DIR}:")
    for f in sorted(DASHBOARD_DIR.iterdir()):
        logger.info(f"  {f.name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export dashboard backend CSVs")
    parser.add_argument("--use-gdelt-sentiment",  action="store_true")
    parser.add_argument("--use-kaggle-sentiment", action="store_true")
    args = parser.parse_args()
    export_all(
        use_gdelt_sentiment=args.use_gdelt_sentiment,
        use_kaggle_sentiment=args.use_kaggle_sentiment,
    )
