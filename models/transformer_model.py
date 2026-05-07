"""
models/transformer_model.py
Transformer encoder — regression-first design for log-return forecasting.

Target modes
  raw      : predict raw future_log_return (ret_1h / ret_4h)
  vol_norm : predict norm_ret = future_log_return / trailing_vol (clipped ±5)
             then invert: predicted_log_return = predicted_norm * vol_target_20

Usage:
    python -m models.transformer_model --asset BTC --retrain
    python -m models.transformer_model --asset BTC --debug-transformer
    python -m models.transformer_model --asset BTC --overfit-test
    python -m models.transformer_model --asset BTC --use-gdelt-sentiment \\
        --target-horizon 4 --target-mode vol_norm --alpha-move 1.0 \\
        --selection-metric signal_score
"""

import argparse, sys, os, math, warnings
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import psycopg2.extras
from loguru import logger
from pathlib import Path

from db.connection import get_conn
from config import ALL_ASSETS
from models.ridge_baseline import chronological_split
from models.feature_schema import (
    get_feature_columns, save_schema_metadata,
    TRANSFORMER_BASE_FEATURES,
)

SAVED_DIR = Path(__file__).parent / "saved"
SAVED_DIR.mkdir(exist_ok=True)

WINDOW_SIZE = 64
D_MODEL     = 128
N_HEAD      = 4
N_LAYERS    = 4
DIM_FF      = 256
DROPOUT     = 0.1
EPOCHS      = 20
BATCH_SIZE  = 64
LR          = 1e-4
THRESHOLD   = 0.002   # ±0.2 % boundary for derived direction labels

# ── Loss config ────────────────────────────────────────────────────────────────
ALPHA_MOVE = 1.0   # weight = 1 + alpha_move * |y_norm|; 0 = plain SmoothL1

# ── Target mode ────────────────────────────────────────────────────────────────
TARGET_MODE = "vol_norm"   # "raw" | "vol_norm"

# ── Checkpoint selection metric ────────────────────────────────────────────────
SELECTION_METRIC = "signal_score"   # "val_loss" | "signal_score"

# ── Pooling ────────────────────────────────────────────────────────────────────
POOLING = "last_token"   # "last_token" | "mean"

# Debug mode
DEBUG_EPOCHS     = 2
DEBUG_BATCH_SIZE = 32
DEBUG_ROW_LIMIT  = 2000
DEBUG_WINDOW     = 32

TRANSFORMER_FEATURES = list(TRANSFORMER_BASE_FEATURES)

# ── Feature-set ablation ────────────────────────────────────────────────────────
_GDELT_PATS = ("gdelt", "sentiment", "finbert", "news_count", "tone",
               "goldstein", "quad_class", "avg_tone", "num_articles")
_MACRO_PATS = ("dxy", "vix", "spy", "_etf", "macro", "tbill", "yield",
               "dollar", "crb", "ted_spread")
_PRICE_KEEP = ("ret_", "log_ret", "pct_", "close", "vol_", "volatility",
               "return", "rsi", "macd", "bb_", "atr", "ema", "sma",
               "stoch", "adx", "cci", "obv", "log_volume")


def _filter_feature_set(cols: list, feature_set: str) -> list:
    """Return feature column subset for the requested ablation mode."""
    if not feature_set or feature_set == "full":
        return cols

    def _has(c, pats):
        cl = c.lower()
        return any(p in cl for p in pats)

    if feature_set == "no_gdelt":
        return [c for c in cols if not _has(c, _GDELT_PATS)]
    if feature_set == "technical_macro":
        return [c for c in cols if not _has(c, _GDELT_PATS)]
    if feature_set == "technical_only":
        return [c for c in cols if not _has(c, _GDELT_PATS + _MACRO_PATS)]
    if feature_set == "price_only":
        return [c for c in cols
                if _has(c, _PRICE_KEEP) and not _has(c, _GDELT_PATS + _MACRO_PATS)]
    return cols


def _derive_direction(arr: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    return np.where(arr > threshold, 1, np.where(arr < -threshold, -1, 0)).astype(np.int8)


# ── Optional imports ───────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    _TORCH = True
except ImportError:
    logger.error("torch not found: pip install torch")
    _TORCH = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    _SKLEARN = True
except ImportError:
    logger.error("scikit-learn not found: pip install scikit-learn")
    _SKLEARN = False


# ── Architecture ───────────────────────────────────────────────────────────────

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class MarketTransformerRegressor(nn.Module):
    def __init__(self, n_features: int,
                 d_model: int = D_MODEL, nhead: int = N_HEAD,
                 num_layers: int = N_LAYERS, dim_feedforward: int = DIM_FF,
                 dropout: float = DROPOUT, pooling: str = POOLING):
        super().__init__()
        self.pooling     = pooling
        self.input_proj  = nn.Linear(n_features, d_model)
        self.pos_enc     = _PositionalEncoding(d_model, dropout=dropout)
        enc_layer        = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.reg_head    = nn.Linear(d_model, 1)
        nn.init.normal_(self.reg_head.weight, std=0.1)
        nn.init.zeros_(self.reg_head.bias)

    def encode(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return x[:, -1, :] if self.pooling == "last_token" else x.mean(dim=1)

    def forward(self, x):
        return self.reg_head(self.encode(x)).squeeze(-1)


# ── Dataset ────────────────────────────────────────────────────────────────────

class WindowDatasetRegression(Dataset):
    """
    Sample i → (X[i:i+window], y[i+window-1]).
    Alignment: window ending at t predicts the target labelled at t.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, window: int = WINDOW_SIZE):
        self.X      = torch.tensor(X, dtype=torch.float32)
        self.y      = torch.tensor(y, dtype=torch.float32)
        self.window = window

    def __len__(self):
        return max(0, len(self.X) - self.window + 1)

    def __getitem__(self, idx: int):
        return self.X[idx : idx + self.window], self.y[idx + self.window - 1]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_dataset(conn, asset: str, row_limit: int = None,
                 target_horizon: int = 1,
                 target_mode: str = "vol_norm",
                 last_n: int = None) -> pd.DataFrame:
    """
    Load features + target columns.

    In vol_norm mode we load both the normalized target (training signal) and
    the raw return + volatility (for inverse conversion and reporting).
    """
    sfx = f"_{target_horizon}h"
    ret_col  = f"ret{sfx}"
    dir_col  = f"direction{sfx}"
    norm_col = f"norm_ret{sfx}"

    feat_sel_main = ", ".join(
        f"f.{c}" for c in TRANSFORMER_FEATURES
        if c not in ("btc_ret_lag_1", "btc_ret_lag_4", "oil_vol_lag_1",
                     "dxy", "vix", "spy")
    )
    optional = {
        "btc_ret_lag_1": 0, "btc_ret_lag_4": 0, "oil_vol_lag_1": 0,
        "dxy": 0, "vix": 0, "spy": 0,
    }
    optional_sel  = ", ".join(
        f"COALESCE(f.{c}, {v}) AS {c}" for c, v in optional.items()
    )
    if last_n:
        order_clause = "ORDER BY f.ts DESC"
        limit_clause = f"LIMIT {last_n}"
    else:
        order_clause = "ORDER BY f.ts"
        limit_clause = f"LIMIT {row_limit}" if row_limit else ""

    if target_mode == "vol_norm":
        target_sel = (
            f"t.{norm_col}   AS future_log_return, "
            f"t.{ret_col}    AS raw_log_return, "
            f"t.vol_target_20 AS rolling_volatility, "
            f"t.{dir_col}    AS direction_class"
        )
        not_null = f"t.{norm_col} IS NOT NULL AND t.vol_target_20 IS NOT NULL"
    else:
        target_sel = (
            f"t.{ret_col}  AS future_log_return, "
            f"t.{dir_col}  AS direction_class"
        )
        not_null = f"t.{ret_col} IS NOT NULL"

    sql = f"""
        SELECT f.ts,
               {feat_sel_main},
               {optional_sel},
               {target_sel}
        FROM   features f
        JOIN   targets  t USING (asset, ts)
        WHERE  f.asset = %s
          AND  {not_null}
        {order_clause}
        {limit_clause}
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


# ── Feature preparation ────────────────────────────────────────────────────────

def prepare_arrays(df: pd.DataFrame, feat_cols: list,
                   target_mode: str = "vol_norm") -> dict:
    """
    Returns dict with:
      X         : (n, n_features) float32  NaN-imputed features
      y_train   : (n,) float32             training target (norm or raw)
      y_raw     : (n,) float32             raw log return (for reporting)
      vol       : (n,) float32 or None     rolling_volatility (vol_norm only)
    """
    need_cols = feat_cols + ["future_log_return"]
    if target_mode == "vol_norm":
        need_cols += ["raw_log_return", "rolling_volatility"]

    sub = df[need_cols].copy()
    sub = sub.dropna(subset=["future_log_return"])
    if target_mode == "vol_norm":
        sub = sub.dropna(subset=["raw_log_return", "rolling_volatility"])

    for c in feat_cols:
        if sub[c].isna().any():
            fill = float(sub[c].median()) if pd.notna(sub[c].median()) else 0.0
            sub[c] = sub[c].fillna(fill)

    X       = sub[feat_cols].values.astype(np.float32)
    y_train = sub["future_log_return"].values.astype(np.float32)

    if target_mode == "vol_norm":
        y_raw = sub["raw_log_return"].values.astype(np.float32)
        vol   = sub["rolling_volatility"].values.astype(np.float32)
    else:
        y_raw = y_train.copy()
        vol   = None

    return {"X": X, "y_train": y_train, "y_raw": y_raw, "vol": vol,
            "index": sub.index}


# ── Loss ───────────────────────────────────────────────────────────────────────

def _weighted_smooth_l1(pred, target, alpha: float):
    """weight[i] = 1 + alpha * |y[i]|.  alpha=0 reduces to plain SmoothL1."""
    base = F.smooth_l1_loss(pred, target, reduction="none")
    if alpha <= 0:
        return base.mean()
    w = 1.0 + alpha * target.abs()
    return (w * base).mean()


# ── Signal score ───────────────────────────────────────────────────────────────

def _signal_score(preds: np.ndarray, trues: np.ndarray) -> float:
    """
    Composite score that prioritises Corr, SignAcc>0.5, std_ratio in [0.3, 1.0],
    low bias, and beating the zero-return baseline.
    std_ratio below 0.3 (too flat) or above 1.0 (too noisy) incurs a penalty.
    Model MAE worse than zero-return baseline incurs a heavy -2.0× penalty.
    """
    if len(preds) < 2 or np.std(preds) < 1e-12 or np.std(trues) < 1e-12:
        return -1.0
    corr = float(np.corrcoef(preds, trues)[0, 1])
    if not np.isfinite(corr):
        corr = 0.0
    ratio     = float(np.std(preds) / np.std(trues))
    zero_mae  = float(np.mean(np.abs(trues)))
    model_mae = float(np.mean(np.abs(preds - trues)))
    zero_fail = max(0.0, (model_mae - zero_mae) / (zero_mae + 1e-10))
    sign_acc  = float(np.mean(np.sign(preds) == np.sign(trues)))
    sign_bonus = max(0.0, sign_acc - 0.5)
    bias      = abs(float(np.mean(preds)) - float(np.mean(trues)))
    bias_pen  = bias / (float(np.std(trues)) + 1e-10)
    # std_ratio reward: [0.3, 1.0] is ideal → zero penalty
    # below 0.3 (over-shrunk) or above 1.0 (over-noisy) → linearly penalised
    if ratio < 0.3:
        ratio_pen = (0.3 - ratio) / 0.3      # 1.0 at ratio=0,  0 at ratio=0.3
    elif ratio > 1.0:
        ratio_pen = min(ratio - 1.0, 1.0)    # 0 at ratio=1.0, capped at 1.0
    else:
        ratio_pen = 0.0
    return (corr + 0.5 * sign_bonus
            - 0.5 * ratio_pen - 2.0 * zero_fail - 0.3 * bias_pen)


# ── Calibration / lag helpers ──────────────────────────────────────────────────

def _calibrate_on_val(p_val: np.ndarray, a_val: np.ndarray,
                      shrinkage_grid: tuple = (0.0, 0.1, 0.2, 0.5, 0.8, 1.0),
                      min_std_ratio: float = 0.2,
                      ) -> tuple:
    """
    Fit linear recalibration a*pred+b on val via OLS, then search shrinkage s
    towards the identity transform.  Effective params: slope=a_ols*s+(1-s)*1,
    intercept=b_ols*s.  Returns (a_ols, b_ols, best_s).
    Rejects any shrinkage level that would reduce the effective std_ratio below
    min_std_ratio (default 0.2) — prevents the calibrated output going flat.
    """
    if len(p_val) < 10 or np.std(p_val) < 1e-12:
        return 1.0, 0.0, 0.0
    A      = np.column_stack([p_val, np.ones(len(p_val))])
    result = np.linalg.lstsq(A, a_val, rcond=None)
    a_ols, b_ols  = float(result[0][0]), float(result[0][1])
    val_std_pred  = float(np.std(p_val))
    val_std_act   = float(np.std(a_val))
    best_s, best_score = 0.0, float("-inf")
    for s in shrinkage_grid:
        a_eff     = a_ols * s + (1.0 - s) * 1.0
        b_eff     = b_ols * s
        eff_ratio = abs(a_eff) * val_std_pred / (val_std_act + 1e-10)
        if eff_ratio < min_std_ratio:
            continue   # skip — would make predictions too flat
        p_cal = p_val * a_eff + b_eff
        score = _signal_score(p_cal, a_val)
        if score > best_score:
            best_score = score
            best_s     = s
    return a_ols, b_ols, best_s


def _lag_diagnostic(pred_raw: np.ndarray, actual_raw: np.ndarray,
                    asset: str, split: str) -> tuple:
    """Compute corr(pred[i], actual[i+lag]) for lag in [-12, +12]. Returns (best_lag, best_corr)."""
    best_lag, best_corr = 0, 0.0
    results = []
    for lag in range(-12, 13):
        if lag < 0:
            p, a = pred_raw[:lag], actual_raw[-lag:]
        elif lag > 0:
            p, a = pred_raw[lag:], actual_raw[:-lag]
        else:
            p, a = pred_raw, actual_raw
        if len(p) < 5 or np.std(p) < 1e-12 or np.std(a) < 1e-12:
            c = 0.0
        else:
            c = float(np.corrcoef(p, a)[0, 1])
            if not np.isfinite(c):
                c = 0.0
        results.append((lag, c))
        if abs(c) > abs(best_corr):
            best_corr, best_lag = c, lag
    corr_at_0 = next(c for l, c in results if l == 0)
    logger.info(
        f"{asset} [{split}] lag diagnostic: corr@0={corr_at_0:.4f}  "
        f"best_lag={best_lag:+d}  best_corr={best_corr:.4f}"
    )
    detail = "  ".join(f"lag{l:+d}={c:.3f}" for l, c in results if l % 4 == 0)
    logger.info(f"  {detail}")
    return best_lag, best_corr


def _save_experiment_summary(
    asset: str,
    feature_set: str,
    target_horizon: int,
    target_mode: str,
    selection_metric: str,
    n_features: int,
    val_metrics: dict,
    test_metrics: dict,
    calibrated_test_metrics: dict = None,
) -> None:
    """Append one row to data/dashboard/transformer_experiment_summary.csv."""
    dash_dir = Path(__file__).parent.parent / "data" / "dashboard"
    dash_dir.mkdir(parents=True, exist_ok=True)
    out_path = dash_dir / "transformer_experiment_summary.csv"
    row = {
        "run_at":           datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "asset":            asset,
        "feature_set":      feature_set or "full",
        "target_horizon":   target_horizon,
        "target_mode":      target_mode,
        "selection_metric": selection_metric,
        "n_features":       n_features,
        "val_mae":          val_metrics.get("mae",              float("nan")),
        "val_corr":         val_metrics.get("pearson_corr",     float("nan")),
        "val_r2":           val_metrics.get("r2",               float("nan")),
        "val_sign_acc":     val_metrics.get("directional_accuracy", float("nan")),
        "val_zero_mae":     val_metrics.get("zero_mae",         float("nan")),
        "val_mae_vs_zero":  val_metrics.get("mae_vs_zero_pct",  float("nan")),
        "test_mae":         test_metrics.get("mae",             float("nan")),
        "test_corr":        test_metrics.get("pearson_corr",    float("nan")),
        "test_r2":          test_metrics.get("r2",              float("nan")),
        "test_sign_acc":    test_metrics.get("directional_accuracy", float("nan")),
        "test_zero_mae":    test_metrics.get("zero_mae",        float("nan")),
        "test_mae_vs_zero": test_metrics.get("mae_vs_zero_pct", float("nan")),
    }
    if calibrated_test_metrics:
        row["cal_test_mae"]  = calibrated_test_metrics.get("mae",          float("nan"))
        row["cal_test_corr"] = calibrated_test_metrics.get("pearson_corr", float("nan"))
        row["cal_test_r2"]   = calibrated_test_metrics.get("r2",           float("nan"))
    df_row = pd.DataFrame([row])
    if out_path.exists():
        df_row.to_csv(out_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(out_path, index=False)
    logger.info(f"Experiment summary appended → {out_path}")


# ── Training ───────────────────────────────────────────────────────────────────

def train_regressor(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    asset:   str,
    device:  "torch.device",
    epochs:          int   = EPOCHS,
    batch_size:      int   = BATCH_SIZE,
    window:          int   = WINDOW_SIZE,
    lr:              float = LR,
    dropout:         float = DROPOUT,
    pooling:         str   = POOLING,
    alpha_move:      float = ALPHA_MOVE,
    selection_metric:str   = SELECTION_METRIC,
) -> "MarketTransformerRegressor":
    """Train; return best checkpoint selected by val_loss or signal_score."""
    n_features = X_train.shape[1]
    model      = MarketTransformerRegressor(
        n_features, dropout=dropout, pooling=pooling
    ).to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info(
        f"{asset}: pooling={pooling}  alpha_move={alpha_move}  "
        f"selection={selection_metric}  lr={lr}"
    )

    train_ds = WindowDatasetRegression(X_train, y_train, window)
    val_ds   = WindowDatasetRegression(X_val,   y_val,   window)

    if len(train_ds) == 0:
        logger.warning(f"{asset}: not enough rows for window")
        return model

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    best_val_loss    = float("inf")
    best_sig_score   = float("-inf")
    best_loss_state  = None
    best_sig_state   = None
    ckpt_path        = SAVED_DIR / f"transformer_{asset.lower()}.pt"

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        tr_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = _weighted_smooth_l1(model(xb), yb, alpha_move)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= max(len(train_ds), 1)

        # ── val — compute loss AND collect preds for signal_score ──
        model.eval()
        vl_loss   = 0.0
        vl_preds  = []
        vl_trues  = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred   = model(xb)
                vl_loss += _weighted_smooth_l1(pred, yb, alpha_move).item() * len(xb)
                vl_preds.extend(pred.cpu().numpy().tolist())
                vl_trues.extend(yb.cpu().numpy().tolist())
        vl_loss /= max(len(val_ds), 1)

        vp  = np.array(vl_preds, dtype=float)
        vt  = np.array(vl_trues, dtype=float)
        sig = _signal_score(vp, vt)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"{asset} epoch {epoch:3d}/{epochs} "
                f"| tr={tr_loss:.5f}  vl={vl_loss:.5f}  sig={sig:.4f}"
            )

        state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if vl_loss < best_val_loss:
            best_val_loss   = vl_loss
            best_loss_state = state
        if sig > best_sig_score:
            best_sig_score  = sig
            best_sig_state  = state

    # Select checkpoint
    chosen_state = (
        best_sig_state  if selection_metric == "signal_score" and best_sig_state is not None
        else best_loss_state
    )
    if chosen_state is None:
        chosen_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(chosen_state)
    torch.save({
        "model_state_dict": chosen_state,
        "task_type":        "regression",
        "config": {
            "input_dim":          n_features,
            "d_model":            D_MODEL,
            "nhead":              N_HEAD,
            "num_encoder_layers": N_LAYERS,
            "dim_feedforward":    DIM_FF,
            "dropout":            dropout,
            "window_size":        window,
            "pooling":            pooling,
        },
        "loss":               "WeightedSmoothL1",
        "alpha_move":         alpha_move,
        "pooling":            pooling,
        "selection_metric":   selection_metric,
        "best_val_loss":      best_val_loss,
        "best_signal_score":  best_sig_score,
        "trained_at":         datetime.utcnow().isoformat(),
        "n_train_samples":    len(X_train),
    }, ckpt_path)
    logger.success(
        f"{asset}: saved (selection={selection_metric}  "
        f"val_loss={best_val_loss:.5f}  sig_score={best_sig_score:.4f})"
    )
    return model


# ── Inference ──────────────────────────────────────────────────────────────────

def predict_all_bars(
    model:        "MarketTransformerRegressor",
    X_all:        np.ndarray,
    ts_all:       pd.DatetimeIndex,
    y_train_all:  np.ndarray,   # scaled training target (norm or raw)
    device:       "torch.device",
    threshold:    float = THRESHOLD,
    batch_size:   int   = 64,
    window:       int   = WINDOW_SIZE,
    target_mean:  float = 0.0,
    target_std:   float = 1.0,
    vol_all:      np.ndarray = None,  # rolling_volatility per bar (vol_norm only)
    raw_ret_all:  np.ndarray = None,  # actual raw log return per bar (vol_norm only)
) -> tuple:
    """
    Returns (predicted_norm_return, predicted_log_return, signal_strength,
             predicted_direction, actual_log_return, rolling_volatility, valid_ts)
    All arrays length m = n - window + 1.
    """
    model.eval()
    n = len(X_all)
    m = n - window + 1
    if m <= 0:
        empty = np.array([], dtype=float)
        return (empty, empty, empty, np.array([], dtype=np.int8),
                empty, empty, pd.DatetimeIndex([]))

    valid_ts = ts_all[window - 1:]

    logger.info(
        "Transformer alignment: window end at t predicts target labelled at t."
    )

    # Use Dataset+DataLoader to match evaluate_regression (produces contiguous batches).
    _dummy_y = np.zeros(len(X_all), dtype=np.float32)
    _ds      = WindowDatasetRegression(X_all.astype(np.float32), _dummy_y, window)
    _dl      = DataLoader(_ds, batch_size=batch_size, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for xb, _ in _dl:
            all_preds.extend(model(xb.to(device)).cpu().numpy().tolist())

    preds_scaled = np.array(all_preds, dtype=float)

    # Inverse-transform from scaled training-target space to norm space
    pred_norm = preds_scaled * target_std + target_mean

    # Inverse-transform from norm space to raw log-return space
    if vol_all is not None:
        vol_window  = vol_all[window - 1:].astype(float)
        pred_raw    = pred_norm * vol_window
        actual_raw  = raw_ret_all[window - 1:].astype(float) if raw_ret_all is not None \
                      else np.full(m, np.nan)
        vol_out     = vol_window
    else:
        pred_raw    = pred_norm          # norm == raw in raw mode
        actual_raw  = (
            (y_train_all[window - 1:].copy() * target_std + target_mean).astype(float)
            if y_train_all is not None else np.full(m, np.nan)
        )
        vol_out     = np.full(m, np.nan)

    signal_strength     = np.abs(pred_raw)
    predicted_direction = _derive_direction(pred_raw, threshold)

    assert len(pred_raw) == len(valid_ts) == len(actual_raw), (
        f"length mismatch: preds={len(pred_raw)} ts={len(valid_ts)} actual={len(actual_raw)}"
    )
    return pred_norm, pred_raw, signal_strength, predicted_direction, actual_raw, vol_out, valid_ts


# ── Prediction table ───────────────────────────────────────────────────────────

_CREATE_PRED_TABLE = """
CREATE TABLE IF NOT EXISTS transformer_predictions (
    id                    BIGSERIAL    PRIMARY KEY,
    asset                 TEXT         NOT NULL,
    ts                    TIMESTAMPTZ  NOT NULL,
    prediction_horizon    INT          NOT NULL DEFAULT 1,
    target_mode           TEXT         NOT NULL DEFAULT 'raw',
    predicted_norm_return FLOAT,
    predicted_log_return  FLOAT,
    rolling_volatility    FLOAT,
    signal_strength       FLOAT,
    predicted_direction   INT,
    actual_log_return              FLOAT,
    regression_error               FLOAT,
    abs_error                      FLOAT,
    actual_direction               INT,
    direction                      INT,
    confidence                     FLOAT,
    prob_down                      FLOAT,
    prob_neutral                   FLOAT,
    prob_up                        FLOAT,
    calibrated_predicted_log_return FLOAT,
    inserted_at                    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (asset, ts, prediction_horizon, target_mode)
);
CREATE INDEX IF NOT EXISTS idx_tp_asset_ts
    ON transformer_predictions (asset, ts DESC);
CREATE INDEX IF NOT EXISTS idx_tp_asset_horizon_mode
    ON transformer_predictions (asset, prediction_horizon, target_mode, ts DESC);
"""

_PRED_MIGRATIONS = [
    # New columns for vol_norm mode
    "ALTER TABLE transformer_predictions ADD COLUMN IF NOT EXISTS prediction_horizon    INT  DEFAULT 1",
    "ALTER TABLE transformer_predictions ADD COLUMN IF NOT EXISTS target_mode           TEXT DEFAULT 'raw'",
    "ALTER TABLE transformer_predictions ADD COLUMN IF NOT EXISTS predicted_norm_return FLOAT",
    "ALTER TABLE transformer_predictions ADD COLUMN IF NOT EXISTS rolling_volatility    FLOAT",
    "ALTER TABLE transformer_predictions ADD COLUMN IF NOT EXISTS predicted_log_return  FLOAT",
    "ALTER TABLE transformer_predictions ADD COLUMN IF NOT EXISTS signal_strength       FLOAT",
    "ALTER TABLE transformer_predictions ADD COLUMN IF NOT EXISTS predicted_direction   INT",
    "ALTER TABLE transformer_predictions ADD COLUMN IF NOT EXISTS actual_log_return     FLOAT",
    "ALTER TABLE transformer_predictions ADD COLUMN IF NOT EXISTS regression_error      FLOAT",
    "ALTER TABLE transformer_predictions ADD COLUMN IF NOT EXISTS abs_error             FLOAT",
    "ALTER TABLE transformer_predictions ADD COLUMN IF NOT EXISTS actual_direction      INT",
    "ALTER TABLE transformer_predictions ADD COLUMN IF NOT EXISTS direction             INT",
    "ALTER TABLE transformer_predictions ADD COLUMN IF NOT EXISTS confidence            FLOAT",
    "ALTER TABLE transformer_predictions ADD COLUMN IF NOT EXISTS prob_down             FLOAT",
    "ALTER TABLE transformer_predictions ADD COLUMN IF NOT EXISTS prob_neutral          FLOAT",
    "ALTER TABLE transformer_predictions ADD COLUMN IF NOT EXISTS prob_up               FLOAT",
    "ALTER TABLE transformer_predictions ADD COLUMN IF NOT EXISTS calibrated_predicted_log_return FLOAT",
    # Drop old (asset, ts) unique if it exists — use DO block to survive missing constraint
    """DO $$ BEGIN
         ALTER TABLE transformer_predictions
             DROP CONSTRAINT IF EXISTS transformer_predictions_asset_ts_key;
     EXCEPTION WHEN others THEN NULL; END $$""",
    # Add new compound unique if absent
    """DO $$ BEGIN
         ALTER TABLE transformer_predictions
             ADD CONSTRAINT tp_asset_ts_horizon_mode_key
             UNIQUE (asset, ts, prediction_horizon, target_mode);
     EXCEPTION WHEN duplicate_table THEN NULL;
               WHEN others THEN NULL; END $$""",
    """DO $$ BEGIN
         ALTER TABLE transformer_predictions ALTER COLUMN direction    DROP NOT NULL;
     EXCEPTION WHEN undefined_column THEN NULL; WHEN others THEN NULL; END $$""",
    """DO $$ BEGIN
         ALTER TABLE transformer_predictions ALTER COLUMN confidence   DROP NOT NULL;
     EXCEPTION WHEN undefined_column THEN NULL; WHEN others THEN NULL; END $$""",
]


def _ensure_pred_table(conn) -> None:
    """Create table and run column migrations. Each migration runs in its own
    get_conn() transaction so a failed statement never aborts a prior success."""
    with conn.cursor() as cur:
        cur.execute(_CREATE_PRED_TABLE)
    for stmt in _PRED_MIGRATIONS:
        try:
            with get_conn() as _c:
                with _c.cursor() as _cur:
                    _cur.execute(stmt)
        except Exception:
            pass  # column/constraint already exists or not applicable


def save_predictions(
    conn, asset: str,
    pred_norm:   np.ndarray,
    pred_raw:    np.ndarray,
    vol_out:     np.ndarray,
    signal_strength:     np.ndarray,
    predicted_direction: np.ndarray,
    actual_log_return:   np.ndarray,
    ts_index:    pd.DatetimeIndex,
    prediction_horizon: int        = 1,
    target_mode:        str        = "vol_norm",
    threshold:          float      = THRESHOLD,
    cal_pred_raw:       np.ndarray = None,
) -> int:
    """Save raw transformer predictions; calibrated_predicted_log_return stored alongside for
    reference only — PPO and dashboard main line always use pred_raw."""
    logger.info("Entered save_predictions()")
    logger.info("Skipping transformer_predictions schema migration during save; table already initialized.")
    if len(pred_raw) == 0:
        return 0

    actual_direction = _derive_direction(actual_log_return, threshold)
    regression_error = pred_raw - actual_log_return
    abs_error        = np.abs(regression_error)

    def _f(v):
        try:
            fv = float(v)
            return fv if np.isfinite(fv) else None
        except Exception:
            return None

    def _i(v):
        try:
            return int(v)
        except Exception:
            return None

    _cal = cal_pred_raw if cal_pred_raw is not None else np.full(len(pred_raw), np.nan)

    records = []
    for ts, pn, pr, vo, ss, pd_, alr, re_, ae, ad, cal in zip(
        ts_index, pred_norm, pred_raw, vol_out, signal_strength, predicted_direction,
        actual_log_return, regression_error, abs_error, actual_direction, _cal
    ):
        ts_py = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        records.append((
            asset, ts_py, int(prediction_horizon), target_mode,
            _f(pn), _f(pr), _f(vo),
            _f(ss), _i(pd_),
            _f(alr), _f(re_), _f(ae), _i(ad),
            _i(pd_), _f(ss), 0.0, 0.0, 0.0,
            _f(cal),
        ))

    total = len(records)

    # DELETE existing rows, then INSERT fresh — no ON CONFLICT.
    try:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '60s'")
            logger.info(
                f"Deleting old predictions for asset={asset} "
                f"horizon={prediction_horizon} target_mode={target_mode}"
            )
            cur.execute(
                "DELETE FROM transformer_predictions "
                "WHERE asset=%s AND prediction_horizon=%s AND target_mode=%s",
                (asset, int(prediction_horizon), target_mode),
            )
            deleted = cur.rowcount
            logger.info(f"Deleted {deleted} rows")
        conn.commit()
    except Exception:
        logger.exception(
            f"{asset}: DELETE failed "
            f"(horizon={prediction_horizon} target_mode={target_mode})"
        )
        raise

    sql = """
        INSERT INTO transformer_predictions
               (asset, ts, prediction_horizon, target_mode,
                predicted_norm_return, predicted_log_return, rolling_volatility,
                signal_strength, predicted_direction,
                actual_log_return, regression_error, abs_error, actual_direction,
                direction, confidence, prob_down, prob_neutral, prob_up,
                calibrated_predicted_log_return)
        VALUES %s
    """
    CHUNK = 500
    inserted = 0
    try:
        for start in range(0, total, CHUNK):
            chunk = records[start : start + CHUNK]
            with conn.cursor() as cur:
                cur.execute("SET statement_timeout = '60s'")
                psycopg2.extras.execute_values(cur, sql, chunk, page_size=CHUNK)
            conn.commit()
            inserted += len(chunk)
            logger.info(f"Inserted {inserted}/{total} rows")
    except Exception:
        logger.exception(
            f"{asset}: INSERT failed at row {inserted}/{total} "
            f"(horizon={prediction_horizon} target_mode={target_mode})"
        )
        raise

    return total


# ── Evaluation ─────────────────────────────────────────────────────────────────

def _meaningful_move_eval(pred_raw: np.ndarray, actual_raw: np.ndarray,
                          asset: str, split: str) -> dict:
    """Evaluate only rows where |actual_log_return| exceeds a threshold. Returns results dict."""
    results = {}
    for thr in (0.0005, 0.001, 0.002):
        mask = np.abs(actual_raw) > thr
        n    = int(mask.sum())
        if n < 5:
            logger.info(f"  MeaningfulMove thr={thr:.4f}: n={n} (too few to report)")
            continue
        p = pred_raw[mask]
        a = actual_raw[mask]
        mae  = float(np.mean(np.abs(p - a)))
        rmse = float(np.sqrt(np.mean((p - a) ** 2)))
        corr = float(np.corrcoef(p, a)[0, 1]) if np.std(p) > 0 and np.std(a) > 0 else 0.0
        dacc = float(np.mean(np.sign(p) == np.sign(a)))
        ratio = float(np.std(p) / np.std(a)) if np.std(a) > 0 else 0.0
        logger.info(
            f"  [{split}] |ret|>{thr:.4f}: n={n}  MAE={mae:.6f}  RMSE={rmse:.6f}  "
            f"Corr={corr:.4f}  DirAcc={dacc:.3f}  std_ratio={ratio:.4f}"
        )
        results[thr] = {"n": n, "mae": mae, "rmse": rmse, "corr": corr,
                        "dir_acc": dacc, "std_ratio": ratio}
    return results


def evaluate_regression(
    model:       "MarketTransformerRegressor",
    X:           np.ndarray,
    y_scaled:    np.ndarray,    # scaled training target
    device:      "torch.device",
    split:       str,
    asset:       str,
    threshold:   float = THRESHOLD,
    window:      int   = WINDOW_SIZE,
    target_mean: float = 0.0,
    target_std:  float = 1.0,
    vol_arr:     np.ndarray = None,   # rolling_vol per sample (vol_norm only)
    raw_ret_arr: np.ndarray = None,   # actual raw log return per sample (vol_norm only)
) -> dict:
    """
    Computes metrics in raw log-return space (after inverse vol transform when
    vol_arr is provided).  Also logs normalized-space metrics and meaningful-move
    breakdown.  Returns metrics dict.
    """
    ds = WindowDatasetRegression(X, y_scaled, window)
    if len(ds) == 0:
        return {}

    dl = DataLoader(ds, batch_size=256, shuffle=False)
    preds_sc, trues_sc = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in dl:
            preds_sc.extend(model(xb.to(device)).cpu().numpy().tolist())
            trues_sc.extend(yb.numpy().tolist())

    p_sc = np.array(preds_sc, dtype=float)
    t_sc = np.array(trues_sc, dtype=float)

    # ── Normalized-target metrics (model's training objective space) ──
    norm_mae  = float(np.mean(np.abs(p_sc - t_sc)))
    norm_rmse = float(np.sqrt(np.mean((p_sc - t_sc) ** 2)))
    norm_corr = float(np.corrcoef(p_sc, t_sc)[0, 1]) if (
        len(p_sc) > 1 and np.std(p_sc) > 0 and np.std(t_sc) > 0
    ) else 0.0
    norm_sig  = _signal_score(p_sc, t_sc)

    # ── Inverse-transform to norm space ──
    p_norm = p_sc * target_std + target_mean   # predicted normalized return

    # ── Inverse-transform to raw log-return space ──
    m_samples = len(p_sc)        # number of windowed samples
    if vol_arr is not None and raw_ret_arr is not None:
        # vol_arr and raw_ret_arr are per-original-bar.
        # The windowed samples correspond to bars [window-1 : window-1+m_samples]
        start_idx  = window - 1
        end_idx    = start_idx + m_samples
        vol_window = vol_arr[start_idx : end_idx].astype(float)
        p_raw      = p_norm * vol_window
        a_raw      = raw_ret_arr[start_idx : end_idx].astype(float)
    else:
        # raw mode: norm == raw (target_mean/std are of raw returns)
        p_raw = p_sc * target_std + target_mean
        a_raw = t_sc * target_std + target_mean

    mae  = float(np.mean(np.abs(p_raw - a_raw)))
    rmse = float(np.sqrt(np.mean((p_raw - a_raw) ** 2)))
    ss_res = float(np.sum((a_raw - p_raw) ** 2))
    ss_tot = float(np.sum((a_raw - a_raw.mean()) ** 2))
    r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    corr   = float(np.corrcoef(p_raw, a_raw)[0, 1]) if (
        len(p_raw) > 1 and np.std(p_raw) > 0 and np.std(a_raw) > 0
    ) else 0.0
    dir_acc = float(np.mean(np.sign(p_raw) == np.sign(a_raw)))
    pred_dir = _derive_direction(p_raw, threshold)
    true_dir = _derive_direction(a_raw, threshold)
    der_acc  = float(np.mean(pred_dir == true_dir))
    der_f1   = float(f1_score(true_dir, pred_dir, average="macro", zero_division=0)) if _SKLEARN else 0.0

    pred_std  = float(np.std(p_raw))
    true_std  = float(np.std(a_raw))
    std_ratio = pred_std / true_std if true_std > 0 else 0.0

    # Zero-return baseline comparison
    zero_mae  = float(np.mean(np.abs(a_raw)))
    zero_rmse = float(np.sqrt(np.mean(a_raw ** 2)))
    mae_improvement_pct  = (zero_mae  - mae)  / (zero_mae  + 1e-10) * 100.0
    rmse_improvement_pct = (zero_rmse - rmse) / (zero_rmse + 1e-10) * 100.0

    logger.info(
        f"{asset} [{split}]  "
        f"MAE={mae:.6f}  RMSE={rmse:.6f}  R²={r2:.4f}  "
        f"Corr={corr:.4f}  DirAcc={dir_acc:.3f}"
    )
    logger.info(
        f"  zero-return baseline: MAE={zero_mae:.6f}  RMSE={zero_rmse:.6f}  "
        f"MAE_improvement={mae_improvement_pct:+.1f}%  RMSE_improvement={rmse_improvement_pct:+.1f}%"
    )
    logger.info(f"  DerivedAcc={der_acc:.3f}  DerivedF1={der_f1:.3f}")
    logger.info(
        f"  std_ratio={std_ratio:.4f}  "
        f"actual_std={true_std:.6f}  pred_std={pred_std:.6f}"
    )
    logger.info(
        f"  actual  [mean={a_raw.mean():.6f}  "
        f"min={a_raw.min():.6f}  max={a_raw.max():.6f}]"
    )
    logger.info(
        f"  predict [mean={p_raw.mean():.6f}  "
        f"min={p_raw.min():.6f}  max={p_raw.max():.6f}]"
    )
    logger.info(
        f"  norm-space: MAE={norm_mae:.4f}  RMSE={norm_rmse:.4f}  "
        f"Corr={norm_corr:.4f}  sig_score={norm_sig:.4f}"
    )

    _meaningful_move_eval(p_raw, a_raw, asset, split)

    return {
        "mae":             mae,  "rmse":           rmse,
        "r2":              r2,   "pearson_corr":   corr,
        "directional_accuracy": dir_acc,
        "derived_accuracy":     der_acc,
        "derived_f1_macro":     der_f1,
        "actual_std":      true_std,  "pred_std":   pred_std,
        "std_ratio":       std_ratio,
        "actual_mean":     float(a_raw.mean()),
        "pred_mean":       float(p_raw.mean()),
        "actual_min":      float(a_raw.min()),  "actual_max": float(a_raw.max()),
        "pred_min":        float(p_raw.min()),  "pred_max":   float(p_raw.max()),
        "norm_mae":        norm_mae,  "norm_rmse": norm_rmse,
        "norm_corr":       norm_corr, "norm_signal_score": norm_sig,
        "zero_mae":        zero_mae,  "zero_rmse": zero_rmse,
        "mae_vs_zero_pct": mae_improvement_pct,
        "rmse_vs_zero_pct":rmse_improvement_pct,
        "_p_raw":          p_raw,     "_a_raw":    a_raw,
    }


# ── Baselines ──────────────────────────────────────────────────────────────────

def _baseline_metrics(y_true: np.ndarray, train_mean: float,
                      asset: str, split: str) -> None:
    def _m(p, label):
        mae  = float(np.mean(np.abs(p - y_true)))
        rmse = float(np.sqrt(np.mean((p - y_true) ** 2)))
        ss_res = float(np.sum((y_true - p) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2  = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        corr = float(np.corrcoef(p, y_true)[0, 1]) if (
            len(p) > 1 and np.std(p) > 0 and np.std(y_true) > 0
        ) else 0.0
        ratio = float(np.std(p) / np.std(y_true)) if np.std(y_true) > 0 else 0.0
        logger.info(
            f"  Baseline [{label}] MAE={mae:.6f}  RMSE={rmse:.6f}  "
            f"R²={r2:.4f}  Corr={corr:.4f}  std_ratio={ratio:.4f}"
        )
    logger.info(f"{asset} Baselines [{split}]:")
    _m(np.zeros_like(y_true),           "zero-return")
    _m(np.full_like(y_true, train_mean),"train-mean ")


# ── Overfit test ───────────────────────────────────────────────────────────────

def run_overfit_test(
    asset:               str,
    use_gdelt_sentiment: bool = False,
    use_kaggle_sentiment:bool = False,
    target_horizon:      int  = 1,
    target_mode:         str  = "vol_norm",
) -> None:
    """Sanity check: model should achieve train R² > 0.05 on 1000 rows / 200 epochs."""
    if not _TORCH or not _SKLEARN:
        logger.error("Missing torch/sklearn")
        return

    OVERFIT_ROWS   = 1000; OVERFIT_EPOCHS = 200
    OVERFIT_WINDOW = 32;   OVERFIT_LR     = 1e-3
    OVERFIT_BS     = 32;   OVERFIT_DM     = 64
    OVERFIT_LAYERS = 2;    OVERFIT_DIMFF  = 128

    logger.info(
        f"[Overfit/{asset}] rows={OVERFIT_ROWS} epochs={OVERFIT_EPOCHS} "
        f"window={OVERFIT_WINDOW} lr={OVERFIT_LR} dropout=0 "
        f"target_mode={target_mode}"
    )
    feat_cols_full = get_feature_columns(
        asset, model="transformer",
        use_gdelt_sentiment=use_gdelt_sentiment,
        use_kaggle_sentiment=use_kaggle_sentiment,
    )
    with get_conn() as conn:
        df = load_dataset(conn, asset, row_limit=OVERFIT_ROWS,
                          target_horizon=target_horizon, target_mode=target_mode)
    if df.empty or len(df) < OVERFIT_WINDOW + 20:
        logger.error(f"[Overfit/{asset}] not enough data ({len(df)} rows)")
        return
    if use_gdelt_sentiment:
        from models.gdelt_sentiment import merge_gdelt_features
        df = merge_gdelt_features(df, asset)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_cols = [c for c in feat_cols_full if c in df.columns]
    arrays    = prepare_arrays(df, feat_cols, target_mode=target_mode)
    X, y_tr   = arrays["X"], arrays["y_train"]
    y_raw, vol = arrays["y_raw"], arrays["vol"]

    logger.info(
        f"[Overfit/{asset}] {len(X)} rows, {X.shape[1]} features  "
        f"y_train std={y_tr.std():.4f}  range=[{y_tr.min():.4f}, {y_tr.max():.4f}]"
    )
    n_train     = max(OVERFIT_WINDOW + 1, int(len(X) * 0.90))
    X_tr        = X[:n_train];    y_sc_tr = y_tr[:n_train]
    vol_tr      = vol[:n_train]   if vol is not None else None
    raw_tr      = y_raw[:n_train] if y_raw is not None else None

    target_mean = float(np.mean(y_sc_tr))
    target_std  = max(float(np.std(y_sc_tr)), 1e-8)
    y_sc_tr     = ((y_sc_tr - target_mean) / target_std).astype(np.float32)

    scaler  = StandardScaler()
    X_sc_tr = np.nan_to_num(scaler.fit_transform(X_tr).astype(np.float32), 0.0, 0.0, 0.0)

    feat_diag_zero = int((X_sc_tr.var(axis=0) < 1e-10).sum())
    logger.info(f"[Overfit/{asset}] zero_var_cols={feat_diag_zero}")

    model     = MarketTransformerRegressor(
        X_sc_tr.shape[1], d_model=OVERFIT_DM, nhead=2,
        num_layers=OVERFIT_LAYERS, dim_feedforward=OVERFIT_DIMFF, dropout=0.0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=OVERFIT_LR)
    train_ds  = WindowDatasetRegression(X_sc_tr, y_sc_tr, OVERFIT_WINDOW)
    if len(train_ds) == 0:
        logger.error(f"[Overfit/{asset}] empty dataset after windowing")
        return
    train_dl = DataLoader(train_ds, batch_size=OVERFIT_BS, shuffle=True)

    for epoch in range(1, OVERFIT_EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = _weighted_smooth_l1(model(xb), yb, ALPHA_MOVE)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= max(len(train_ds), 1)
        if epoch % 20 == 0 or epoch == 1:
            logger.info(f"[Overfit/{asset}] epoch {epoch:3d}/{OVERFIT_EPOCHS}  loss={tr_loss:.5f}")

    logger.info(f"[Overfit/{asset}] Evaluating train split:")
    m = evaluate_regression(
        model, X_sc_tr, y_sc_tr, device, "overfit-train", asset,
        window=OVERFIT_WINDOW, target_mean=target_mean, target_std=target_std,
        vol_arr=vol_tr, raw_ret_arr=raw_tr,
    )
    r2 = m.get("r2", float("nan")); sr = m.get("std_ratio", float("nan"))
    if r2 > 0.05:
        logger.success(f"[Overfit/{asset}] PASS — R²={r2:.4f}  std_ratio={sr:.4f}")
    else:
        logger.error(f"[Overfit/{asset}] FAIL — R²={r2:.4f}  std_ratio={sr:.4f}")


# ── Per-asset pipeline ─────────────────────────────────────────────────────────

def load_or_train(asset: str, force_retrain: bool = False,
                  device: "torch.device" = None):
    if not _TORCH:
        return None
    ckpt_path = SAVED_DIR / f"transformer_{asset.lower()}.pt"
    if not ckpt_path.exists() or force_retrain:
        return None
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if ckpt.get("task_type", "classification") != "regression":
        raise ValueError(f"[Transformer/{asset}] Old classifier checkpoint — use --retrain.")
    cfg = ckpt.get("config", {})
    model = MarketTransformerRegressor(
        n_features=cfg.get("input_dim", 26),
        d_model=cfg.get("d_model", D_MODEL), nhead=cfg.get("nhead", N_HEAD),
        num_layers=cfg.get("num_encoder_layers", N_LAYERS),
        dim_feedforward=cfg.get("dim_feedforward", DIM_FF),
        dropout=cfg.get("dropout", DROPOUT), pooling=cfg.get("pooling", POOLING),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device)


def run_asset(
    asset:               str,
    epochs:              int   = EPOCHS,
    force_retrain:       bool  = False,
    use_gdelt_sentiment: bool  = False,
    use_kaggle_sentiment:bool  = False,
    debug:               bool  = False,
    target_horizon:      int   = 1,
    target_mode:         str   = TARGET_MODE,
    alpha_move:          float = ALPHA_MOVE,
    selection_metric:    str   = SELECTION_METRIC,
    feature_set:         str   = "full",
) -> dict:
    if not _TORCH or not _SKLEARN:
        logger.error(f"{asset}: missing torch/sklearn — skipping")
        return {}

    target_name = f"{'norm_ret' if target_mode == 'vol_norm' else 'ret'}_{target_horizon}h"

    if debug:
        epochs = DEBUG_EPOCHS; batch_size = DEBUG_BATCH_SIZE
        row_limit = DEBUG_ROW_LIMIT; window = DEBUG_WINDOW
        logger.info(
            f"{asset}: DEBUG  epochs={epochs} rows<={row_limit} "
            f"window={window} target={target_name} mode={target_mode}"
        )
    else:
        batch_size = BATCH_SIZE; row_limit = None; window = WINDOW_SIZE

    ckpt_path  = SAVED_DIR / f"transformer_{asset.lower()}.pt"
    _skip_train = False
    _ckpt       = {}

    if ckpt_path.exists() and not force_retrain:
        try:
            _ckpt = torch.load(str(ckpt_path), map_location="cpu")
        except Exception as e:
            logger.warning(f"{asset}: bad checkpoint ({e}); retraining.")
            _ckpt = {}
        if _ckpt.get("task_type", "classification") != "regression":
            raise ValueError(
                f"[Transformer/{asset}] Old classifier checkpoint — use --retrain."
            )
        logger.info(f"[Transformer/{asset}] Checkpoint exists — loading for inference.")
        _skip_train = True

    feat_cols_full = get_feature_columns(
        asset, model="transformer",
        use_gdelt_sentiment=use_gdelt_sentiment,
        use_kaggle_sentiment=use_kaggle_sentiment,
    )
    feat_cols_full = _filter_feature_set(feat_cols_full, feature_set)
    if feature_set and feature_set != "full":
        logger.info(f"{asset}: feature_set={feature_set!r} → {len(feat_cols_full)} candidate cols")

    with get_conn() as conn:
        df = load_dataset(conn, asset, row_limit=row_limit,
                          target_horizon=target_horizon, target_mode=target_mode)

    if df.empty or len(df) < window + 100:
        logger.warning(f"{asset}: not enough data ({len(df)} rows)")
        return {}

    if use_gdelt_sentiment:
        from models.gdelt_sentiment import merge_gdelt_features
        df = merge_gdelt_features(df, asset)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_cols = [c for c in feat_cols_full if c in df.columns]
    arrays    = prepare_arrays(df, feat_cols, target_mode=target_mode)
    X         = arrays["X"]
    y_train   = arrays["y_train"]   # norm or raw
    y_raw     = arrays["y_raw"]     # always raw log return
    vol       = arrays["vol"]       # rolling_volatility or None

    n_features = X.shape[1]
    logger.info(
        f"{asset}: {len(X)} rows  {n_features} features  "
        f"device={device}  target={target_name}  mode={target_mode}"
    )
    logger.info(
        f"  y_train range=[{y_train.min():.4f}, {y_train.max():.4f}]  "
        f"std={y_train.std():.4f}"
    )

    n       = len(X)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    val_end = n_train + n_val

    # Scale training target (normalised returns or raw returns)
    target_mean = float(np.mean(y_train[:n_train]))
    target_std  = max(float(np.std(y_train[:n_train])), 1e-8)
    # When loading a checkpoint, use the stored scaling to keep predictions consistent
    if _skip_train and "target_mean" in _ckpt:
        target_mean = float(_ckpt["target_mean"])
        target_std  = max(float(_ckpt.get("target_std", target_std)), 1e-8)

    logger.info(
        f"  target scaling: mean={target_mean:.8f}  std={target_std:.8f}"
    )
    y_scaled = ((y_train - target_mean) / target_std).astype(np.float32)

    scaler   = StandardScaler()
    scaler.fit(X[:n_train])
    X_scaled = np.nan_to_num(scaler.transform(X), nan=0.0, posinf=0.0, neginf=0.0)

    if _skip_train:
        cfg = _ckpt.get("config", {})
        model = MarketTransformerRegressor(
            n_features,
            d_model=cfg.get("d_model", D_MODEL),
            nhead=cfg.get("nhead", N_HEAD),
            num_layers=cfg.get("num_encoder_layers", N_LAYERS),
            dim_feedforward=cfg.get("dim_feedforward", DIM_FF),
            dropout=cfg.get("dropout", DROPOUT),
            pooling=cfg.get("pooling", POOLING),
        ).to(device)
        model.load_state_dict(
            {k: v.to(device) for k, v in _ckpt["model_state_dict"].items()}
        )
        model.eval()
    else:
        model = train_regressor(
            X_scaled[:n_train],        y_scaled[:n_train],
            X_scaled[n_train:val_end], y_scaled[n_train:val_end],
            asset, device,
            epochs=epochs, batch_size=batch_size, window=window,
            alpha_move=alpha_move, selection_metric=selection_metric,
        )

    # ── Evaluation per split ──
    metrics   = {}
    train_met = {}
    splits = [
        ("train", X_scaled[:n_train],       y_scaled[:n_train],
         vol[:n_train] if vol is not None else None,
         y_raw[:n_train]),
        ("val",   X_scaled[n_train:val_end], y_scaled[n_train:val_end],
         vol[n_train:val_end] if vol is not None else None,
         y_raw[n_train:val_end]),
        ("test",  X_scaled[val_end:],        y_scaled[val_end:],
         vol[val_end:] if vol is not None else None,
         y_raw[val_end:]),
    ]
    for sname, X_s, y_s, vol_s, raw_s in splits:
        m = evaluate_regression(
            model, X_s, y_s, device, sname, asset,
            window=window, target_mean=target_mean, target_std=target_std,
            vol_arr=vol_s, raw_ret_arr=raw_s,
        )
        if sname == "train":
            train_met = m
        else:
            metrics[sname] = m
        # Lag diagnostic on val and test only (skip train — too noisy)
        if sname in ("val", "test") and "_p_raw" in m and "_a_raw" in m:
            _lag_diagnostic(m["_p_raw"], m["_a_raw"], asset, sname)

    # Raw-return baseline on test split
    if len(y_raw) > val_end + window:
        test_raw = y_raw[val_end + window - 1:]
        train_raw_mean = float(np.mean(y_raw[:n_train]))
        _baseline_metrics(test_raw, train_raw_mean, asset, "test")

    # ── Validation calibration ──
    val_met      = metrics.get("val",  {})
    test_met_raw = metrics.get("test", {})
    cal_test_met = {}
    cal_a_ols = cal_b_ols = cal_s = None
    if "_p_raw" in val_met and "_a_raw" in val_met:
        p_val_raw = val_met["_p_raw"]
        a_val_raw = val_met["_a_raw"]
        cal_a_ols, cal_b_ols, cal_s = _calibrate_on_val(p_val_raw, a_val_raw)
        a_eff = cal_a_ols * cal_s + (1.0 - cal_s) * 1.0
        b_eff = cal_b_ols * cal_s
        logger.info(
            f"{asset} calibration: OLS slope={cal_a_ols:.4f}  intercept={cal_b_ols:.6f}  "
            f"best_shrink={cal_s:.2f} → effective slope={a_eff:.4f}  intercept={b_eff:.6f}"
        )
        # Apply calibration to test predictions (evaluation only — raw remains primary)
        if "_p_raw" in test_met_raw and "_a_raw" in test_met_raw:
            p_test_cal    = test_met_raw["_p_raw"] * a_eff + b_eff
            a_test        = test_met_raw["_a_raw"]
            cal_mae       = float(np.mean(np.abs(p_test_cal - a_test)))
            cal_corr      = float(np.corrcoef(p_test_cal, a_test)[0, 1]) if (
                np.std(p_test_cal) > 0 and np.std(a_test) > 0
            ) else 0.0
            ss_res_c      = float(np.sum((a_test - p_test_cal) ** 2))
            ss_tot_c      = float(np.sum((a_test - a_test.mean()) ** 2))
            cal_r2        = float(1.0 - ss_res_c / ss_tot_c) if ss_tot_c > 0 else 0.0
            cal_sign_acc  = float(np.mean(np.sign(p_test_cal) == np.sign(a_test)))
            cal_std_ratio = float(np.std(p_test_cal) / np.std(a_test)) if np.std(a_test) > 0 else 0.0
            cal_test_met  = {
                "mae": cal_mae, "pearson_corr": cal_corr, "r2": cal_r2,
                "sign_acc": cal_sign_acc, "std_ratio": cal_std_ratio,
            }
            base_mae = test_met_raw.get("mae", float("nan"))
            logger.info(
                f"{asset} calibrated test MAE={cal_mae:.6f}  R²={cal_r2:.4f}  Corr={cal_corr:.4f}  "
                f"({'improved' if cal_mae < base_mae else 'no improvement'} vs raw MAE={base_mae:.6f})"
            )

    # ── Task 10: Final raw vs calibrated comparison log ──
    _nan = float("nan")
    raw_sign_acc = test_met_raw.get("directional_accuracy", _nan)
    raw_std_ratio = test_met_raw.get("std_ratio", _nan)
    logger.info(
        f"\n{asset} ══ Final test-split comparison ══\n"
        f"  Raw        : MAE={test_met_raw.get('mae', _nan):.6f}  "
        f"Corr={test_met_raw.get('pearson_corr', _nan):.4f}  "
        f"SignAcc={raw_sign_acc:.3f}  StdRatio={raw_std_ratio:.4f}\n"
        f"  Calibrated : MAE={cal_test_met.get('mae', _nan):.6f}  "
        f"Corr={cal_test_met.get('pearson_corr', _nan):.4f}  "
        f"SignAcc={cal_test_met.get('sign_acc', _nan):.3f}  "
        f"StdRatio={cal_test_met.get('std_ratio', _nan):.4f}"
    )

    # ── Update checkpoint ──
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        for prefix, met in [("train", train_met),
                             ("val",   val_met),
                             ("test",  test_met_raw)]:
            for k, v in met.items():
                if k.startswith("_"):
                    continue  # skip internal _p_raw/_a_raw arrays
                try:
                    ckpt[f"{prefix}_{k}"] = float(v)
                except (TypeError, ValueError):
                    pass
        ckpt.update({
            "feature_columns":      feat_cols,
            "feature_count":        len(feat_cols),
            "feature_set":          feature_set or "full",
            "use_gdelt_sentiment":  use_gdelt_sentiment,
            "use_kaggle_sentiment": use_kaggle_sentiment,
            "prediction_horizon":   target_horizon,
            "target_mode":          target_mode,
            "target_name":          target_name,
            "target_mean":          target_mean,
            "target_std":           target_std,
            "direction_threshold":  THRESHOLD,
        })
        if cal_a_ols is not None:
            ckpt.update({
                "cal_a_ols": cal_a_ols, "cal_b_ols": cal_b_ols,
                "cal_shrink": cal_s,
            })
        torch.save(ckpt, str(ckpt_path))
        t = test_met_raw
        logger.info(
            f"[Transformer/{asset}] test_MAE={t.get('mae', float('nan')):.6f}  "
            f"test_R²={t.get('r2', float('nan')):.4f}  "
            f"test_Corr={t.get('pearson_corr', float('nan')):.4f}  "
            f"test_std_ratio={t.get('std_ratio', float('nan')):.4f}"
        )

    # ── Experiment summary ──
    _save_experiment_summary(
        asset=asset, feature_set=feature_set or "full",
        target_horizon=target_horizon, target_mode=target_mode,
        selection_metric=selection_metric, n_features=n_features,
        val_metrics=val_met, test_metrics=test_met_raw,
        calibrated_test_metrics=cal_test_met if cal_test_met else None,
    )

    t = metrics.get("test", {})
    save_schema_metadata(
        SAVED_DIR / f"transformer_{asset.lower()}_schema.json",
        asset, "transformer", feat_cols,
        use_kaggle_sentiment=use_kaggle_sentiment,
        use_gdelt_sentiment=use_gdelt_sentiment,
        extra={
            "task_type":          "regression",
            "target_name":        target_name,
            "target_mode":        target_mode,
            "prediction_horizon": target_horizon,
            "alpha_move":         alpha_move,
            "pooling":            POOLING,
            "selection_metric":   selection_metric,
            "target_mean":        target_mean, "target_std": target_std,
            "feature_set":  feature_set or "full",
            "val_mae":     float(val_met.get("mae",          float("nan"))),
            "val_r2":      float(val_met.get("r2",           float("nan"))),
            "val_std_ratio":float(val_met.get("std_ratio",   float("nan"))),
            "test_mae":    float(test_met_raw.get("mae",         float("nan"))),
            "test_r2":     float(test_met_raw.get("r2",          float("nan"))),
            "test_corr":   float(test_met_raw.get("pearson_corr",float("nan"))),
            "test_std_ratio":float(test_met_raw.get("std_ratio", float("nan"))),
        },
    )

    # ── Full inference (most-recent bars only; free large arrays first) ──
    # Dashboard queries LIMIT 2000; window=64 needs 64 extra rows.
    import gc
    _MAX_INF = 2500
    if len(X_scaled) > _MAX_INF:
        _s      = len(X_scaled) - _MAX_INF
        X_inf   = X_scaled[_s:].copy()
        ts_inf  = df.index[_s:]
        vol_inf = vol[_s:].copy() if vol is not None else None
        raw_inf = y_raw[_s:].copy()
    else:
        X_inf   = X_scaled.copy()
        ts_inf  = df.index
        vol_inf = vol.copy() if vol is not None else None
        raw_inf = y_raw.copy()

    # Explicitly release large arrays held from training/evaluation before inference.
    del X_scaled, y_scaled, y_train, y_raw, X, vol, df, arrays
    gc.collect()

    pred_norm, pred_raw, ss, pd_, alr, vol_out, valid_ts = predict_all_bars(
        model, X_inf, ts_inf, None, device,
        window=window, target_mean=target_mean, target_std=target_std,
        vol_all=vol_inf, raw_ret_all=raw_inf,
    )

    # Compute calibrated inference predictions (saved as secondary column; raw stays primary)
    cal_pred_raw_inf = None
    if cal_a_ols is not None and len(pred_raw) > 0:
        a_eff_inf     = cal_a_ols * cal_s + (1.0 - cal_s) * 1.0
        b_eff_inf     = cal_b_ols * cal_s
        cal_candidate = pred_raw * a_eff_inf + b_eff_inf
        # Enforce min std_ratio of 0.2: if effective ratio too low, fall back to raw
        inf_std_act = float(np.std(alr)) if len(alr) > 1 else 1.0
        inf_ratio   = float(np.std(cal_candidate)) / (inf_std_act + 1e-10)
        if inf_ratio >= 0.2:
            cal_pred_raw_inf = cal_candidate
        else:
            logger.warning(
                f"{asset}: calibrated std_ratio={inf_ratio:.4f} < 0.2 — "
                "storing calibrated_predicted_log_return as NaN (raw used as fallback)"
            )

    # Mandatory pre-save diagnostics
    if len(pred_norm) > 0:
        _fin_vol = vol_out[np.isfinite(vol_out)] if vol_out is not None and len(vol_out) else np.array([float("nan")])
        logger.info(
            f"{asset} prediction save diagnostics:\n"
            f"  horizon={target_horizon}  target_mode={target_mode}\n"
            f"  rows={len(pred_norm)}\n"
            f"  predicted_norm_return  min={float(pred_norm.min()):.6f}  max={float(pred_norm.max()):.6f}\n"
            f"  rolling_volatility     min={float(_fin_vol.min()):.6f}  max={float(_fin_vol.max()):.6f}\n"
            f"  predicted_log_return   min={float(pred_raw.min()):.6f}  max={float(pred_raw.max()):.6f}\n"
            f"  actual_log_return      min={float(alr.min()):.6f}  max={float(alr.max()):.6f}"
        )
        if cal_pred_raw_inf is not None:
            logger.info(
                f"  cal_predicted_log_return  min={float(cal_pred_raw_inf.min()):.6f}  "
                f"max={float(cal_pred_raw_inf.max()):.6f}"
            )

    try:
        logger.info("Opening DB connection for prediction save...")
        with get_conn() as conn:
            logger.info("DB connection opened. Calling save_predictions()...")
            n_saved = save_predictions(
                conn, asset, pred_norm, pred_raw, vol_out, ss, pd_, alr, valid_ts,
                prediction_horizon=target_horizon, target_mode=target_mode,
                cal_pred_raw=cal_pred_raw_inf,
            )
            logger.info("Returned from save_predictions().")
        logger.success(
            f"{asset}: {n_saved} predictions saved to transformer_predictions "
            f"(horizon={target_horizon}, target_mode={target_mode})"
        )
    except Exception as e:
        logger.exception(f"{asset}: save_predictions failed — {e}")
        raise

    # Strip internal numpy arrays before returning
    for split_met in metrics.values():
        split_met.pop("_p_raw", None)
        split_met.pop("_a_raw", None)
    return metrics


# ── Orchestration ──────────────────────────────────────────────────────────────

def run(
    assets:              list  = None,
    epochs:              int   = EPOCHS,
    force_retrain:       bool  = False,
    use_gdelt_sentiment: bool  = False,
    use_kaggle_sentiment:bool  = False,
    debug:               bool  = False,
    target_horizon:      int   = 1,
    target_mode:         str   = TARGET_MODE,
    alpha_move:          float = ALPHA_MOVE,
    selection_metric:    str   = SELECTION_METRIC,
    feature_set:         str   = "full",
) -> dict:
    assets = assets or ALL_ASSETS
    results: dict = {}
    for asset in assets:
        try:
            results[asset] = run_asset(
                asset, epochs=epochs, force_retrain=force_retrain,
                use_gdelt_sentiment=use_gdelt_sentiment,
                use_kaggle_sentiment=use_kaggle_sentiment,
                debug=debug, target_horizon=target_horizon,
                target_mode=target_mode, alpha_move=alpha_move,
                selection_metric=selection_metric,
                feature_set=feature_set,
            )
        except Exception as e:
            logger.error(f"{asset}: pipeline failed — {e}")
            results[asset] = {}
    return results


def main():
    p = argparse.ArgumentParser(description="Transformer log-return forecaster")
    p.add_argument("--asset",               default=None)
    p.add_argument("--epochs",              type=int,   default=EPOCHS)
    p.add_argument("--use-gdelt-sentiment",   action="store_true")
    p.add_argument("--use-kaggle-sentiment",  action="store_true")
    p.add_argument("--debug-transformer",     action="store_true")
    p.add_argument("--overfit-test",          action="store_true")
    p.add_argument("--retrain",               action="store_true")
    p.add_argument("--target-horizon",        type=int,   default=1, choices=[1, 4])
    p.add_argument("--target-mode",           default=TARGET_MODE,
                   choices=["raw", "vol_norm"])
    p.add_argument("--alpha-move",            type=float, default=ALPHA_MOVE)
    p.add_argument("--selection-metric",      default=SELECTION_METRIC,
                   choices=["val_loss", "signal_score"])
    p.add_argument("--feature-set",           default="full",
                   choices=["full", "no_gdelt", "technical_macro",
                            "technical_only", "price_only"])
    args = p.parse_args()

    assets = [args.asset] if args.asset else ALL_ASSETS

    if args.overfit_test:
        for asset in assets:
            run_overfit_test(
                asset,
                use_gdelt_sentiment=args.use_gdelt_sentiment,
                use_kaggle_sentiment=args.use_kaggle_sentiment,
                target_horizon=args.target_horizon,
                target_mode=args.target_mode,
            )
        return

    results = run(
        assets=assets, epochs=args.epochs,
        force_retrain=args.retrain,
        use_gdelt_sentiment=args.use_gdelt_sentiment,
        use_kaggle_sentiment=args.use_kaggle_sentiment,
        debug=args.debug_transformer,
        target_horizon=args.target_horizon,
        target_mode=args.target_mode,
        alpha_move=args.alpha_move,
        selection_metric=args.selection_metric,
        feature_set=args.feature_set,
    )

    for asset, res in results.items():
        t = res.get("test", {})
        if t:
            logger.info(
                f"{asset} final test  "
                f"MAE={t.get('mae', float('nan')):.6f}  "
                f"RMSE={t.get('rmse', float('nan')):.6f}  "
                f"R²={t.get('r2', float('nan')):.4f}  "
                f"Corr={t.get('pearson_corr', float('nan')):.4f}  "
                f"std_ratio={t.get('std_ratio', float('nan')):.4f}"
            )


if __name__ == "__main__":
    main()
