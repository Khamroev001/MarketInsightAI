"""
dashboard/data_loader.py
Load dashboard data from CSVs → PostgreSQL → built-in fallback.
All public loaders are decorated with @st.cache_data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent.parent
DASHBOARD_DIR = ROOT_DIR / "data" / "dashboard"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RAW_DIR       = ROOT_DIR / "data" / "raw"
SAVED_DIR     = ROOT_DIR / "models" / "saved"

sys.path.insert(0, str(ROOT_DIR))

ALL_ASSETS = ["BTC", "ETH", "GOLD", "OIL"]

# ── Fallback transformer metrics (provided by project) ────────────────────────
FALLBACK_METRICS: dict[str, dict] = {
    "BTC":  {"asset": "BTC",  "model": "Transformer", "MAE": 0.001218, "RMSE": 0.002886, "R2":  0.0000, "Corr":  0.0210, "DirAcc": 0.174, "DerivedAcc": 0.806, "DerivedF1": 0.298},
    "ETH":  {"asset": "ETH",  "model": "Transformer", "MAE": 0.001617, "RMSE": 0.003780, "R2": -0.0021, "Corr":  0.0291, "DirAcc": 0.828, "DerivedAcc": 0.782, "DerivedF1": 0.293},
    "GOLD": {"asset": "GOLD", "model": "Transformer", "MAE": 0.000704, "RMSE": 0.002404, "R2":  0.0000, "Corr":  0.0080, "DirAcc": 0.753, "DerivedAcc": 0.893, "DerivedF1": 0.315},
    "OIL":  {"asset": "OIL",  "model": "Transformer", "MAE": 0.001532, "RMSE": 0.005747, "R2": -0.0003, "Corr": -0.0012, "DirAcc": 0.878, "DerivedAcc": 0.847, "DerivedF1": 0.306},
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_conn():
    """Return psycopg2 get_conn factory or None on failure."""
    try:
        from db.connection import get_conn
        return get_conn
    except Exception:
        return None


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame()


# ── Public loaders ────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_prices() -> pd.DataFrame:
    """OHLCV price bars for all assets."""
    csv_path = DASHBOARD_DIR / "latest_prices.csv"
    df = _read_csv_safe(csv_path)
    if not df.empty:
        return _parse_ts(df).sort_values(["asset", "timestamp_utc"])

    get_conn = _get_conn()
    if get_conn is None:
        return pd.DataFrame()

    try:
        import psycopg2.extras
        frames = []
        for asset in ALL_ASSETS:
            sql = """
                SELECT ts AS timestamp_utc, asset, open, high, low, close, volume
                FROM   raw_price_bars
                WHERE  asset = %s
                ORDER  BY ts DESC
                LIMIT  1440
            """
            with get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql, (asset,))
                    rows = cur.fetchall()
            if rows:
                frames.append(pd.DataFrame(rows))
        if frames:
            df = pd.concat(frames, ignore_index=True)
            return _parse_ts(df).sort_values(["asset", "timestamp_utc"])
    except Exception as e:
        st.warning(f"Prices: DB query failed — {e}")

    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_news_sentiment() -> pd.DataFrame:
    """GDELT + FinBERT article-level news data."""
    csv_path = DASHBOARD_DIR / "latest_news_sentiment.csv"
    df = _read_csv_safe(csv_path)
    if not df.empty:
        return _parse_ts(df).sort_values("timestamp_utc", ascending=False)

    # Fall back to raw GDELT article CSVs
    frames = []
    for asset in ALL_ASSETS:
        p = RAW_DIR / f"gdelt_{asset.lower()}_articles.csv"
        sub = _read_csv_safe(p)
        if not sub.empty:
            sub["asset"] = asset
            frames.append(sub)
    if frames:
        df = pd.concat(frames, ignore_index=True)
        return _parse_ts(df)

    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_gdelt_features() -> pd.DataFrame:
    """Hourly aggregated GDELT sentiment features per asset."""
    frames = []
    for asset in ALL_ASSETS:
        p = PROCESSED_DIR / f"gdelt_{asset.lower()}_sentiment_features.csv"
        sub = _read_csv_safe(p)
        if not sub.empty:
            sub["asset"] = asset
            frames.append(sub)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return _parse_ts(df)


@st.cache_data(ttl=300)
def load_predictions() -> pd.DataFrame:
    """Transformer predictions joined with actual returns."""
    csv_path = DASHBOARD_DIR / "latest_predictions.csv"
    df = _read_csv_safe(csv_path)
    if not df.empty:
        df = _parse_ts(df)
        df = _normalize_predictions(df)
        return df.sort_values(["asset", "timestamp_utc"])

    get_conn = _get_conn()
    if get_conn is None:
        return pd.DataFrame()

    try:
        import psycopg2.extras
        frames = []
        for asset in ALL_ASSETS:
            # Use new regression columns directly; fall back to legacy on error
            sql = """
                SELECT tp.ts                              AS timestamp_utc,
                       tp.asset,
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
                ORDER  BY tp.ts DESC
                LIMIT  1440
            """
            try:
                with get_conn() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute(sql, (asset,))
                        rows = cur.fetchall()
            except Exception:
                # Legacy schema fallback
                with get_conn() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute("""
                            SELECT tp.ts          AS timestamp_utc,
                                   tp.asset,
                                   tp.direction   AS predicted_direction,
                                   tp.confidence  AS signal_strength,
                                   t.ret_1h       AS actual_log_return,
                                   t.direction_1h AS actual_direction
                            FROM   transformer_predictions tp
                            LEFT JOIN targets t
                                   ON t.asset = tp.asset AND t.ts = tp.ts
                            WHERE  tp.asset = %s
                            ORDER  BY tp.ts DESC
                            LIMIT  1440
                        """, (asset,))
                        rows = cur.fetchall()
            if rows:
                frames.append(pd.DataFrame(rows))
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df = _parse_ts(df)
            df = _normalize_predictions(df)
            return df.sort_values(["asset", "timestamp_utc"])
    except Exception as e:
        st.warning(f"Predictions: DB query failed — {e}")

    return pd.DataFrame()


# Matches the threshold used in export_dashboard_data.py
_VIZ_DIRECTION_THRESHOLD = 0.00001


def _normalize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonise column names; never overwrite real predicted_log_return values."""
    # Rename old-format columns only if the canonical names are absent
    if "transformer_prediction" in df.columns and "predicted_direction" not in df.columns:
        df = df.rename(columns={"transformer_prediction": "predicted_direction"})
    if "transformer_confidence" in df.columns and "signal_strength" not in df.columns:
        df = df.rename(columns={"transformer_confidence": "signal_strength"})

    # Coerce to numeric; preserve NaN, do not fill with 0
    for col in ["predicted_log_return", "calibrated_predicted_log_return",
                "signal_strength", "actual_log_return",
                "predicted_direction", "actual_direction"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Basis-point versions for chart readability (1 bps = 0.0001 = 0.01%)
    if "predicted_log_return" in df.columns:
        df["predicted_log_return_bps"] = df["predicted_log_return"] * 10000
    if "calibrated_predicted_log_return" in df.columns:
        df["calibrated_predicted_log_return_bps"] = df["calibrated_predicted_log_return"] * 10000
    if "actual_log_return" in df.columns:
        df["actual_log_return_bps"] = df["actual_log_return"] * 10000

    # Derive predicted_log_return ONLY when the column is genuinely absent
    # (never when it exists but is small — those are real model outputs)
    if "predicted_log_return" not in df.columns:
        # No real regression output available; leave as NaN so charts show "no data"
        df["predicted_log_return"] = float("nan")

    # signal_strength fallback: use abs(predicted_log_return) only when column absent
    if "signal_strength" not in df.columns:
        df["signal_strength"] = df["predicted_log_return"].abs()

    # actual_log_return: only derive proxy if column is truly absent
    if "actual_log_return" not in df.columns:
        df["actual_log_return"] = float("nan")

    # viz_direction: low-threshold direction for display (not the model's own threshold)
    if "viz_direction" not in df.columns:
        plr = df["predicted_log_return"]
        df["viz_direction"] = 0
        df.loc[plr >  _VIZ_DIRECTION_THRESHOLD, "viz_direction"] =  1
        df.loc[plr < -_VIZ_DIRECTION_THRESHOLD, "viz_direction"] = -1

    return df


@st.cache_data(ttl=300)
def load_model_metrics() -> pd.DataFrame:
    """Model evaluation metrics with fallback to hard-coded values."""
    csv_path = DASHBOARD_DIR / "model_metrics.csv"
    df = _read_csv_safe(csv_path)
    if not df.empty:
        df = _normalize_metrics_cols(df)
        return df

    # Return fallback transformer-only metrics
    return pd.DataFrame(list(FALLBACK_METRICS.values()))


def _normalize_metrics_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Map alternative column names used by models/dashboard_export.py → canonical."""
    rename = {
        "mae":                  "MAE",
        "rmse":                 "RMSE",
        "r2":                   "R2",
        "corr":                 "Corr",
        "directional_accuracy": "DirAcc",
        "derived_acc":          "DerivedAcc",
        "derived_f1":           "DerivedF1",
    }
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

def _parse_ts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Normalize possible timestamp column names
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    elif "ts" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    elif "timestamp" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    return df

@st.cache_data(ttl=300)
def load_rl_trades() -> pd.DataFrame:
    """PPO RL trade log."""
    csv_path = DASHBOARD_DIR / "latest_rl_trades.csv"
    df = _read_csv_safe(csv_path)

    if df.empty:
        return pd.DataFrame()

    # RL trade log uses timestamp_utc; keep this name because charts.py expects it.
    if "timestamp_utc" not in df.columns:
        if "ts" in df.columns:
            df = df.rename(columns={"ts": "timestamp_utc"})
        elif "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "timestamp_utc"})

    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(
            df["timestamp_utc"], errors="coerce", utc=True
        )
        df = df.dropna(subset=["timestamp_utc"])

    return df


@st.cache_data(ttl=300)
def load_rl_trade_history() -> pd.DataFrame:
    """Completed-trade history — one row per closed trade."""
    csv_path = DASHBOARD_DIR / "latest_rl_trade_history.csv"
    df = _read_csv_safe(csv_path)
    if df.empty:
        return pd.DataFrame()
    for col in ("entry_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


@st.cache_data(ttl=300)
def load_feature_status() -> pd.DataFrame:
    """Feature pipeline health per asset."""
    csv_path = DASHBOARD_DIR / "feature_status.csv"
    df = _read_csv_safe(csv_path)
    if not df.empty:
        return df

    # Derive from filesystem
    rows = []
    for asset in ALL_ASSETS:
        rows.append({
            "asset":         asset,
            "gdelt_file":    (PROCESSED_DIR / f"gdelt_{asset.lower()}_sentiment_features.csv").exists(),
            "transformer_pt": (SAVED_DIR / f"transformer_{asset.lower()}.pt").exists(),
            "scaler_pkl":    (SAVED_DIR / f"feature_scaler_{asset.lower()}.pkl").exists(),
            "ridge_pkl":     (SAVED_DIR / f"ridge_{asset.lower()}.pkl").exists(),
            "ppo_zip":       (SAVED_DIR / f"ppo_{asset.lower()}.zip").exists(),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_model_comparison() -> pd.DataFrame:
    """Multi-model comparison table."""
    csv_path = DASHBOARD_DIR / "model_comparison.csv"
    df = _read_csv_safe(csv_path)
    if not df.empty:
        return df
    # Fallback: transformer-only
    rows = [{**m, "total_return": None, "max_drawdown": None, "win_rate": None}
            for m in FALLBACK_METRICS.values()]
    return pd.DataFrame(rows)


# ── Filtering utilities ───────────────────────────────────────────────────────

def filter_by_asset(df: pd.DataFrame, asset: str, col: str = "asset") -> pd.DataFrame:
    if asset == "All" or df.empty or col not in df.columns:
        return df
    return df[df[col] == asset].copy()


def filter_by_days(df: pd.DataFrame, days: int | None, col: str = "timestamp_utc") -> pd.DataFrame:
    if days is None or df.empty or col not in df.columns:
        return df
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    mask = df[col] >= cutoff
    return df[mask].copy()
