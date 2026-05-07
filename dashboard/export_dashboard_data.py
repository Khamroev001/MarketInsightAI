"""
dashboard/export_dashboard_data.py
Export dashboard-ready CSVs from PostgreSQL (or model files as fallback).

Usage:
    python -m dashboard.export_dashboard_data
    python -m dashboard.export_dashboard_data --use-gdelt-sentiment
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR      = Path(__file__).parent.parent
DASHBOARD_DIR = ROOT_DIR / "data" / "dashboard"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
SAVED_DIR     = ROOT_DIR / "models" / "saved"

DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT_DIR))

ALL_ASSETS = ["BTC", "ETH", "GOLD", "OIL"]

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("export")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ── DB helper ─────────────────────────────────────────────────────────────────

def _get_conn():
    try:
        from db.connection import get_conn
        return get_conn
    except Exception as e:
        logger.warning(f"DB not available: {e}")
        return None


def _query(get_conn, sql: str, params=()) -> pd.DataFrame:
    try:
        import psycopg2.extras
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                return pd.DataFrame(cur.fetchall())
    except Exception as e:
        logger.warning(f"Query failed: {e}")
        return pd.DataFrame()


# ── 1. latest_prices.csv ──────────────────────────────────────────────────────

def export_prices(get_conn) -> None:
    if get_conn is None:
        logger.warning("Skipping prices export — no DB connection")
        return
    frames = []
    for asset in ALL_ASSETS:
        df = _query(get_conn, """
            SELECT ts AS timestamp_utc, asset, open, high, low, close, volume
            FROM   raw_price_bars
            WHERE  asset = %s
            ORDER  BY ts DESC
            LIMIT  2880
        """, (asset,))
        if not df.empty:
            frames.append(df)
    if not frames:
        logger.warning("No price data found in DB")
        return
    out = pd.concat(frames, ignore_index=True)
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    out = out.sort_values(["asset", "timestamp_utc"])
    path = DASHBOARD_DIR / "latest_prices.csv"
    out.to_csv(path, index=False)
    logger.info(f"Prices → {path} ({len(out)} rows)")


# ── 2. latest_news_sentiment.csv ──────────────────────────────────────────────

def export_news_sentiment() -> None:
    """Merge raw GDELT article CSVs; already exists but refresh if newer files found."""
    frames = []
    for asset in ALL_ASSETS:
        p = ROOT_DIR / "data" / "raw" / f"gdelt_{asset.lower()}_articles.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["asset"] = asset
            frames.append(df)
            logger.info(f"GDELT articles: {asset} → {len(df)} rows")
    if not frames:
        logger.warning("No raw GDELT articles found")
        return
    out = pd.concat(frames, ignore_index=True)
    if "timestamp_utc" in out.columns:
        out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    path = DASHBOARD_DIR / "latest_news_sentiment.csv"
    out.to_csv(path, index=False)
    logger.info(f"News sentiment → {path} ({len(out)} rows)")


# ── 3. latest_predictions.csv ─────────────────────────────────────────────────

# Threshold used only to derive a low-noise visualization direction for charts.
# The raw predicted_log_return is always exported unchanged.
VIZ_DIRECTION_THRESHOLD = 0.00001


def export_predictions(get_conn, use_gdelt_sentiment: bool = False,
                       horizon: int = 1, target_mode: str = "vol_norm") -> None:
    frames = []
    for asset in ALL_ASSETS:
        df = pd.DataFrame()
        if get_conn is not None:
            # Select regression outputs with new vol_norm schema columns.
            # Falls back to a secondary query with old columns if new ones are absent.
            df = _query(get_conn, """
                SELECT tp.ts                              AS timestamp_utc,
                       tp.asset,
                       tp.prediction_horizon,
                       tp.target_mode,
                       tp.predicted_norm_return,
                       tp.rolling_volatility,
                       tp.predicted_log_return,
                       tp.calibrated_predicted_log_return,
                       tp.signal_strength,
                       tp.predicted_direction,
                       tp.actual_log_return,
                       tp.actual_direction,
                       tp.regression_error,
                       tp.abs_error
                FROM   transformer_predictions tp
                WHERE  tp.asset = %s
                  AND  tp.target_mode = %s
                  AND  tp.prediction_horizon = %s
                ORDER  BY tp.ts DESC
                LIMIT  2000
            """, (asset, target_mode, horizon))

            # If new columns missing (old schema), fall back and join targets for actuals
            if df.empty or "predicted_log_return" not in df.columns:
                logger.warning(f"{asset}: new regression columns not found — trying legacy schema")
                df = _query(get_conn, """
                    SELECT tp.ts          AS timestamp_utc,
                           tp.asset,
                           tp.direction   AS predicted_direction,
                           tp.confidence  AS signal_strength,
                           t.ret_1h       AS actual_log_return,
                           t.direction_1h AS actual_direction
                    FROM   transformer_predictions tp
                    LEFT JOIN targets t ON t.asset = tp.asset AND t.ts = tp.ts
                    WHERE  tp.asset = %s
                    ORDER  BY tp.ts DESC
                    LIMIT  2000
                """, (asset,))
                # Legacy: derive a rough proxy only when real column is absent
                if not df.empty:
                    df["predicted_log_return"]  = None
                    df["predicted_norm_return"]  = None
                    df["rolling_volatility"]     = None
                    df["prediction_horizon"]     = 1
                    df["target_mode"]            = "raw"
                    df["regression_error"]       = None
                    df["abs_error"]              = None

        if df.empty:
            logger.warning(f"No transformer predictions for {asset}")
            continue

        df["asset"] = asset
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

        # Coerce numeric columns — do NOT replace non-NULL values with 0
        for col in ["predicted_norm_return", "rolling_volatility",
                    "predicted_log_return", "calibrated_predicted_log_return",
                    "signal_strength", "actual_log_return",
                    "regression_error", "abs_error", "prediction_horizon"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Visualization direction: low threshold so small real predictions show a signal.
        # This is separate from predicted_direction (which uses the model's own threshold).
        if "predicted_log_return" in df.columns:
            plr = df["predicted_log_return"]
            df["viz_direction"] = 0
            df.loc[plr >  VIZ_DIRECTION_THRESHOLD, "viz_direction"] =  1
            df.loc[plr < -VIZ_DIRECTION_THRESHOLD, "viz_direction"] = -1

        # ── Per-asset diagnostics ─────────────────────────────────────────────
        if "predicted_log_return" in df.columns:
            plr = df["predicted_log_return"].dropna()
            sig = df["signal_strength"].dropna() if "signal_strength" in df.columns else pd.Series(dtype=float)
            logger.info(
                f"  {asset} predictions: n={len(plr)}  "
                f"plr  mean={plr.mean():.6f}  std={plr.std():.6f}  "
                f"min={plr.min():.7f}  max={plr.max():.7f}"
            )
            if not sig.empty:
                logger.info(
                    f"  {asset} signal_str: n={len(sig)}  "
                    f"mean={sig.mean():.6f}  std={sig.std():.6f}  "
                    f"min={sig.min():.7f}  max={sig.max():.7f}"
                )
            # Consistency check: signal_strength should equal abs(predicted_log_return)
            if not sig.empty and not plr.empty:
                aligned = pd.concat([plr.rename("plr"), sig.rename("sig")], axis=1).dropna()
                if not aligned.empty:
                    drift = (aligned["sig"] - aligned["plr"].abs()).abs().max()
                    if drift > 1e-9:
                        logger.warning(
                            f"  {asset} signal/abs(plr) max drift = {drift:.2e}  "
                            "(signal_strength != abs(predicted_log_return))"
                        )
                    else:
                        logger.info(f"  {asset} signal consistency OK (drift={drift:.2e})")

        # Attach GDELT sentiment if requested
        if use_gdelt_sentiment:
            proc = PROCESSED_DIR / f"gdelt_{asset.lower()}_sentiment_features.csv"
            if proc.exists():
                gdelt = pd.read_csv(proc)
                gdelt["timestamp_utc"] = pd.to_datetime(gdelt["timestamp_utc"], utc=True, errors="coerce")
                pfx      = asset.lower()
                sent_col = f"{pfx}_gdelt_sentiment_mean"
                vol_col  = f"{pfx}_gdelt_news_volume"
                if sent_col in gdelt.columns:
                    gdelt = gdelt.set_index("timestamp_utc")[[sent_col, vol_col]]
                    df    = df.set_index("timestamp_utc")
                    df["sentiment_score"] = gdelt[sent_col].reindex(df.index)   # keep NaN, no fillna
                    df["news_volume"]     = gdelt[vol_col].reindex(df.index).fillna(0.0)
                    df = df.reset_index()

        frames.append(df)

    if not frames:
        logger.warning("No predictions to export")
        return

    out = pd.concat(frames, ignore_index=True)
    path = DASHBOARD_DIR / "latest_predictions.csv"
    # float_format preserves sub-thousandth precision for log returns
    out.to_csv(path, index=False, float_format="%.8f")
    logger.info(f"Predictions → {path} ({len(out)} rows)")


# ── 4. model_metrics.csv ──────────────────────────────────────────────────────

FALLBACK_METRICS = {
    "BTC":  {"MAE": 0.001218, "RMSE": 0.002886, "R2":  0.0000, "Corr":  0.0210, "DirAcc": 0.174, "DerivedAcc": 0.806, "DerivedF1": 0.298},
    "ETH":  {"MAE": 0.001617, "RMSE": 0.003780, "R2": -0.0021, "Corr":  0.0291, "DirAcc": 0.828, "DerivedAcc": 0.782, "DerivedF1": 0.293},
    "GOLD": {"MAE": 0.000704, "RMSE": 0.002404, "R2":  0.0000, "Corr":  0.0080, "DirAcc": 0.753, "DerivedAcc": 0.893, "DerivedF1": 0.315},
    "OIL":  {"MAE": 0.001532, "RMSE": 0.005747, "R2": -0.0003, "Corr": -0.0012, "DirAcc": 0.878, "DerivedAcc": 0.847, "DerivedF1": 0.306},
}


def export_model_metrics() -> None:
    rows = []
    for asset in ALL_ASSETS:
        base = {"asset": asset, "model": "Transformer", **FALLBACK_METRICS[asset]}

        # Try loading richer metrics from transformer checkpoint
        ckpt_path = SAVED_DIR / f"transformer_{asset.lower()}.pt"
        if ckpt_path.exists():
            try:
                import torch
                ckpt = torch.load(str(ckpt_path), map_location="cpu")
                # Pull any stored regression metrics from checkpoint
                for key in ["mae", "rmse", "r2", "corr", "dir_acc"]:
                    if key in ckpt:
                        col = {"mae": "MAE", "rmse": "RMSE", "r2": "R2",
                               "corr": "Corr", "dir_acc": "DirAcc"}.get(key, key)
                        base[col] = float(ckpt[key])
            except Exception:
                pass

        rows.append(base)

        # Ridge metrics from pkl
        ridge_path = SAVED_DIR / f"ridge_{asset.lower()}.pkl"
        if ridge_path.exists():
            try:
                import joblib
                ridge_data = joblib.load(ridge_path)
                m = ridge_data.get("metrics", {}).get("test", {})
                rows.append({
                    "asset": asset, "model": "Ridge",
                    "MAE": m.get("mae"), "RMSE": m.get("rmse"),
                    "R2": m.get("r2"), "Corr": m.get("corr"),
                    "DirAcc": m.get("directional_accuracy"),
                    "DerivedAcc": None, "DerivedF1": None,
                })
            except Exception:
                pass

        # PPO from metadata JSON
        ppo_meta = SAVED_DIR / f"ppo_{asset.lower()}_metadata.json"
        if ppo_meta.exists():
            try:
                with open(ppo_meta) as f:
                    meta = json.load(f)
                rows.append({
                    "asset": asset, "model": "PPO_RL",
                    "MAE": None, "RMSE": None, "R2": None, "Corr": None,
                    "DirAcc": None, "DerivedAcc": None, "DerivedF1": None,
                    "total_return": meta.get("total_pnl"),
                    "max_drawdown": meta.get("max_drawdown"),
                    "win_rate":     meta.get("win_rate"),
                })
            except Exception:
                pass

    df   = pd.DataFrame(rows)
    path = DASHBOARD_DIR / "model_metrics.csv"
    df.to_csv(path, index=False)
    logger.info(f"Model metrics → {path} ({len(df)} rows)")


# ── 5. feature_status.csv ─────────────────────────────────────────────────────

def export_feature_status(get_conn, use_gdelt_sentiment: bool = False) -> None:
    rows = []
    for asset in ALL_ASSETS:
        row = {
            "asset":             asset,
            "gdelt_file_exists": (PROCESSED_DIR / f"gdelt_{asset.lower()}_sentiment_features.csv").exists(),
            "transformer_saved": (SAVED_DIR / f"transformer_{asset.lower()}.pt").exists(),
            "ridge_saved":       (SAVED_DIR / f"ridge_{asset.lower()}.pkl").exists(),
            "ppo_saved":         (SAVED_DIR / f"ppo_{asset.lower()}.zip").exists(),
            "scaler_saved":      (SAVED_DIR / f"feature_scaler_{asset.lower()}.pkl").exists(),
        }

        # Row counts from DB
        if get_conn is not None:
            for table, key in [
                ("raw_price_bars", "price_rows"),
                ("features",       "feature_rows"),
                ("targets",        "target_rows"),
                ("transformer_predictions", "prediction_rows"),
            ]:
                cnt_df = _query(get_conn,
                    f"SELECT COUNT(*) AS cnt FROM {table} WHERE asset = %s", (asset,))
                row[key] = int(cnt_df["cnt"].iloc[0]) if not cnt_df.empty else 0

        rows.append(row)

    df   = pd.DataFrame(rows)
    path = DASHBOARD_DIR / "feature_status.csv"
    df.to_csv(path, index=False)
    logger.info(f"Feature status → {path}")


# ── 6. model_comparison.csv ───────────────────────────────────────────────────

def export_model_comparison() -> None:
    metrics_path = DASHBOARD_DIR / "model_metrics.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
    else:
        rows = [{
            "asset": a, "model": "Transformer", **FALLBACK_METRICS[a],
            "total_return": None, "max_drawdown": None, "win_rate": None,
        } for a in ALL_ASSETS]
        df = pd.DataFrame(rows)
    path = DASHBOARD_DIR / "model_comparison.csv"
    df.to_csv(path, index=False)
    logger.info(f"Model comparison → {path}")


# ── Orchestration ─────────────────────────────────────────────────────────────

def export_all(use_gdelt_sentiment: bool = False,
               horizon: int = 1, target_mode: str = "vol_norm") -> None:
    logger.info("=" * 60)
    logger.info("MarketInsight AI — Dashboard Export")
    logger.info(f"  horizon={horizon}h  target_mode={target_mode}")
    logger.info("=" * 60)

    get_conn = _get_conn()
    export_prices(get_conn)
    export_news_sentiment()
    export_predictions(get_conn, use_gdelt_sentiment=use_gdelt_sentiment,
                       horizon=horizon, target_mode=target_mode)
    export_model_metrics()
    export_feature_status(get_conn, use_gdelt_sentiment=use_gdelt_sentiment)
    export_model_comparison()

    logger.info("=" * 60)
    logger.info(f"Dashboard data directory: {DASHBOARD_DIR}")
    for f in sorted(DASHBOARD_DIR.iterdir()):
        size = f.stat().st_size
        logger.info(f"  {f.name:45s}  {size:>8,} bytes")
    logger.info("=" * 60)
    logger.info("Run the dashboard:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export dashboard backend CSVs")
    parser.add_argument("--use-gdelt-sentiment", action="store_true",
                        help="Attach GDELT sentiment scores to predictions CSV")
    parser.add_argument("--horizon",     type=int, default=1,
                        help="Prediction horizon in hours: 1 or 4 (default: 1)")
    parser.add_argument("--target-mode", default="vol_norm",
                        choices=["raw", "vol_norm"],
                        help="Target mode to export (default: vol_norm)")
    args = parser.parse_args()
    export_all(use_gdelt_sentiment=args.use_gdelt_sentiment,
               horizon=args.horizon, target_mode=args.target_mode)
