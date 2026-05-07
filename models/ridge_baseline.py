"""
models/ridge_baseline.py
Ridge regression baseline — 13 price features only, predicts ret_4h magnitude.

Pipeline:
  1. Load features + targets from PostgreSQL, join on (asset, ts).
  2. Chronological 70 / 15 / 15 train / val / test split.
  3. Train Ridge(alpha=1.0) on 13 price-derived features.
  4. Evaluate: MAE and directional accuracy on val and test sets.
  5. Save model to models/saved/ridge_{asset}.pkl.

Usage:
    python -m models.ridge_baseline
    python -m models.ridge_baseline --asset BTC
"""

import argparse, sys, os, warnings
from xml.parsers.expat import model
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings("ignore", category=UserWarning)

from dotenv import parser
import joblib
import numpy as np
import pandas as pd
import psycopg2.extras
from loguru import logger
from pathlib import Path

from db.connection import get_conn
from config import ALL_ASSETS

try:
    from sklearn.linear_model  import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics       import mean_absolute_error
    _SKLEARN = True
except ImportError:
    logger.error("scikit-learn not found: pip install scikit-learn")
    _SKLEARN = False

SAVED_DIR = Path(__file__).parent / "saved"
SAVED_DIR.mkdir(exist_ok=True)

# Exactly 13 price-derived features (no technical indicators, no macro, no LLM)
PRICE_FEATURES = [
    "ret_1",  "ret_4",  "ret_8",  "ret_16",
    "mean_4", "mean_8", "mean_16",
    "std_4",  "std_8",  "std_16",
    "mom_4",  "mom_16",
    "vol_20",
]


# ── Utilities (shared by transformer_model and rl_agent) ──────────────────────

def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70 / 15 / 15 chronological split — no shuffling, no look-ahead leakage."""
    n       = len(df)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    return df.iloc[:n_train], df.iloc[n_train:n_train + n_val], df.iloc[n_train + n_val:]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(conn, asset: str) -> pd.DataFrame:
    """Join features and targets on (asset, ts), return chronological DataFrame."""
    feat_cols = ", ".join(f"f.{c}" for c in PRICE_FEATURES)
    sql = f"""
        SELECT f.ts,
               {feat_cols},
               t.ret_4h, t.direction_4h
        FROM   features f
        JOIN   targets  t USING (asset, ts)
        WHERE  f.asset = %s
          AND  t.direction_4h IS NOT NULL
        ORDER  BY f.ts
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (asset,))
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _prepare(df: pd.DataFrame, feature_cols: list[str],
             target_col: str,
             medians: "pd.Series | None" = None,
             ) -> tuple[pd.DataFrame, pd.Series]:
    """Return X, y with NaN imputation.  medians=None → fit mode; else transform mode."""
    if medians is not None:
        cols = [c for c in medians.index if c in df.columns]
        sub  = df[cols + [target_col]].dropna(subset=[target_col]).copy()
        for c in cols:
            sub[c] = sub[c].fillna(medians[c])
        return sub[cols], sub[target_col].astype(float)

    cols = [c for c in feature_cols if c in df.columns]
    sub  = df[cols + [target_col]].copy()
    all_null = [c for c in cols if sub[c].isna().all()]
    if all_null:
        sub  = sub.drop(columns=all_null)
        cols = [c for c in cols if c not in all_null]
    sub = sub.dropna(subset=[target_col])
    for c in cols:
        if sub[c].isna().any():
            sub[c] = sub[c].fillna(sub[c].median())
    return sub[cols], sub[target_col].astype(float)


# ── Training ──────────────────────────────────────────────────────────────────

def train_ridge(asset: str,
                train: pd.DataFrame,
                val:   pd.DataFrame,
                test:  pd.DataFrame) -> dict:
    """Train Ridge on 13 price features, evaluate, save."""
    if not _SKLEARN:
        return {}

    X_tr, y_tr = _prepare(train, PRICE_FEATURES, "ret_4h")
    if X_tr.empty:
        logger.warning(f"{asset} Ridge: insufficient training data")
        return {}

    tr_med     = X_tr.median()
    X_vl, y_vl = _prepare(val,  PRICE_FEATURES, "ret_4h", medians=tr_med)
    X_te, y_te = _prepare(test, PRICE_FEATURES, "ret_4h", medians=tr_med)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_vl_s = scaler.transform(X_vl) if not X_vl.empty else np.empty((0, X_tr_s.shape[1]))
    X_te_s = scaler.transform(X_te) if not X_te.empty else np.empty((0, X_tr_s.shape[1]))

    model = Ridge(alpha=1.0)
    model.fit(X_tr_s, y_tr)

    metrics: dict = {}
    for split_name, X_s, y_s in [("val", X_vl_s, y_vl), ("test", X_te_s, y_te)]:
        if len(X_s) == 0:
            continue
    preds = model.predict(X_s)
    
    mae = mean_absolute_error(y_s, preds)
    rmse = float(np.sqrt(np.mean((y_s.values - preds) ** 2)))
    
    ss_res = float(np.sum((y_s.values - preds) ** 2))
    ss_tot = float(np.sum((y_s.values - y_s.values.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    
    corr = float(np.corrcoef(preds, y_s.values)[0, 1]) if len(preds) > 1 else np.nan
    dir_acc = float((np.sign(preds) == np.sign(y_s.values)).mean())
    std_ratio = float(np.std(preds) / np.std(y_s.values)) if np.std(y_s.values) > 0 else np.nan
    
    metrics[split_name] = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "corr": corr,
        "directional_accuracy": dir_acc,
        "std_ratio": std_ratio,
    }
    
    logger.info(
        f"{asset} Ridge [{split_name}] "
        f"MAE={mae:.6f} RMSE={rmse:.6f} R2={r2:.4f} "
        f"Corr={corr:.4f} DirAcc={dir_acc:.3f} StdRatio={std_ratio:.3f}"
    )
    
    joblib.dump(
        {"model": model, "scaler": scaler, "features": PRICE_FEATURES},
        SAVED_DIR / f"ridge_{asset.lower()}.pkl",
    )
    logger.success(f"{asset}: Ridge model saved → {SAVED_DIR / f'ridge_{asset.lower()}.pkl'}")
    return metrics


# ── Load-or-train ──────────────────────────────────────────────────────────────

def load_or_train(asset: str, X_train, y_train, X_val=None, y_val=None,
                  force_retrain: bool = False):
    """
    Load an existing Ridge model from disk or train from scratch.

    Parameters
    ----------
    asset : str
    X_train, y_train : training data (already scaled)
    force_retrain : bool
        When True, always retrain regardless of saved state.

    Returns
    -------
    Fitted Ridge model.
    """
    path = SAVED_DIR / f"ridge_{asset.lower()}.pkl"

    if path.exists() and not force_retrain:
        bundle = joblib.load(path)
        model  = bundle.get("model", bundle) if isinstance(bundle, dict) else bundle
        logger.info(f"[Ridge/{asset}] Loaded from {path} — skipping training")
        return model

    logger.info(f"[Ridge/{asset}] Training from scratch …")
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    joblib.dump(model, path)
    logger.success(f"[Ridge/{asset}] Saved to {path}")
    return model


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_asset(asset: str, force_retrain: bool = False) -> dict:
    # Skip training if saved model exists (unless forced)
    path = SAVED_DIR / f"ridge_{asset.lower()}.pkl"
    if path.exists() and not force_retrain:
        logger.info(f"[Ridge/{asset}] Model exists at {path} — skipping (use force_retrain=True to retrain)")
        return {"ridge": {"skipped": True}}

    with get_conn() as conn:
        df = load_dataset(conn, asset)
    if df.empty or len(df) < 200:
        logger.warning(f"{asset}: not enough data ({len(df)} rows) — skipping Ridge")
        return {}
    train, val, test = chronological_split(df)
    logger.info(f"{asset}: {len(train)} train / {len(val)} val / {len(test)} test rows")
    return {"ridge": train_ridge(asset, train, val, test)}


def run(assets: list = None, force_retrain: bool = False) -> dict[str, dict]:
    assets = assets or ALL_ASSETS
    results: dict[str, dict] = {}
    for asset in assets:
        try:
            results[asset] = run_asset(asset, force_retrain=force_retrain)
        except Exception as e:
            logger.error(f"{asset}: Ridge baseline failed — {e}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Ridge baseline (13 price features)")
    parser.add_argument("--asset", default=None, help="BTC|ETH|GOLD|OIL (default: all)")
    parser.add_argument("--retrain", action="store_true", help="Force retrain Ridge models")
    args = parser.parse_args()
    
    assets  = [args.asset] if args.asset else ALL_ASSETS
    results = run(assets=assets, force_retrain=args.retrain)

    for asset, res in results.items():
        r = res.get("ridge", {}).get("test", {})
        if r:
            logger.info(
                f"{asset} Ridge test → "
                f"MAE={r.get('mae', float('nan')):.6f}  "
                f"DirAcc={r.get('directional_accuracy', float('nan')):.3f}"
            )


if __name__ == "__main__":
    main()
