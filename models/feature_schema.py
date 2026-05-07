"""
models/feature_schema.py
Central feature schema — single source of truth for all models.

Every model (Ridge, Transformer, RL scaler, RL env, RL agent, dashboard export)
must call get_feature_columns() instead of maintaining its own hard-coded list.
"""
from __future__ import annotations
from loguru import logger

# ── Base market features for RL (includes close, excludes LLM alphas) ──────────
RL_BASE_FEATURES: list[str] = [
    "close",
    "ret_1", "ret_4", "ret_8", "ret_16",
    "mean_4", "mean_8", "mean_16",
    "std_4",  "std_8",  "std_16",
    "mom_4",  "mom_16",
    "vol_20",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_pct", "atr_14",
    "btc_ret_lag_1", "btc_ret_lag_4", "oil_vol_lag_1",
    "dxy", "vix", "spy",
]  # 27 features

# ── Base transformer features (excludes close, excludes LLM alphas) ────────────
TRANSFORMER_BASE_FEATURES: list[str] = [
    "ret_1", "ret_4", "ret_8", "ret_16",
    "mean_4", "mean_8", "mean_16",
    "std_4",  "std_8",  "std_16",
    "mom_4",  "mom_16",
    "vol_20",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_pct", "atr_14",
    "btc_ret_lag_1", "btc_ret_lag_4", "oil_vol_lag_1",
    "dxy", "vix", "spy",
]  # 26 features (alpha_1..5 removed; GDELT added via get_feature_columns)

# ── Ridge price-only features ──────────────────────────────────────────────────
RIDGE_BASE_FEATURES: list[str] = [
    "ret_1", "ret_4", "ret_8", "ret_16",
    "mean_4", "mean_8", "mean_16",
    "std_4",  "std_8",  "std_16",
    "mom_4",  "mom_16", "vol_20",
]  # 13 features

# ── GDELT FinBERT sentiment features — one set per asset ──────────────────────
GDELT_FEATURES: dict[str, list[str]] = {
    "BTC": [
        "btc_gdelt_sentiment_mean", "btc_gdelt_sentiment_std",
        "btc_gdelt_news_volume",    "btc_gdelt_positive_ratio",
        "btc_gdelt_negative_ratio", "btc_gdelt_avg_confidence",
    ],
    "ETH": [
        "eth_gdelt_sentiment_mean", "eth_gdelt_sentiment_std",
        "eth_gdelt_news_volume",    "eth_gdelt_positive_ratio",
        "eth_gdelt_negative_ratio", "eth_gdelt_avg_confidence",
    ],
    "GOLD": [
        "gold_gdelt_sentiment_mean", "gold_gdelt_sentiment_std",
        "gold_gdelt_news_volume",    "gold_gdelt_positive_ratio",
        "gold_gdelt_negative_ratio", "gold_gdelt_avg_confidence",
    ],
    "OIL": [
        "oil_gdelt_sentiment_mean", "oil_gdelt_sentiment_std",
        "oil_gdelt_news_volume",    "oil_gdelt_positive_ratio",
        "oil_gdelt_negative_ratio", "oil_gdelt_avg_confidence",
    ],
}

# ── Kaggle/external sentiment features (extend as needed) ─────────────────────
KAGGLE_SENTIMENT_FEATURES: dict[str, list[str]] = {
    "BTC":  [],
    "ETH":  [],
    "GOLD": [],
    "OIL":  [],
}

# ── Transformer regression signal columns appended to RL state ───────────────
# These replace the old classifier probability columns (p_down/p_neutral/p_up/confidence).
TRANSFORMER_OUTPUT_COLS: list[str] = [
    "predicted_log_return",   # raw regression forecast
    "signal_strength",        # abs(predicted_log_return)
    "derived_direction",      # +1 / 0 / -1 from threshold comparison
]

# ── Portfolio state appended to RL observation after market + transformer cols ─
PORTFOLIO_STATE_DIM: int = 7
# Indices: current_position_pct(0), unrealised_pnl(1), cash_ratio(2),
#          portfolio_value_norm(3), drawdown(4), steps_in_position_norm(5),
#          time_of_day_norm(6)


def get_feature_columns(
    asset: str,
    model: str = "rl",
    use_kaggle_sentiment: bool = False,
    use_gdelt_sentiment: bool = False,
) -> list[str]:
    """
    Canonical ordered feature list for a given asset and model.

    Parameters
    ----------
    asset   : BTC | ETH | GOLD | OIL
    model   : "rl" | "transformer" | "ridge"
    use_kaggle_sentiment : include Kaggle sentiment cols for this asset
    use_gdelt_sentiment  : include GDELT FinBERT sentiment cols for this asset

    Returns
    -------
    Ordered list of column names.  Every model must use this instead of
    maintaining its own hard-coded list.
    """
    asset = asset.upper()

    if model == "rl":
        cols = list(RL_BASE_FEATURES)
    elif model == "transformer":
        cols = list(TRANSFORMER_BASE_FEATURES)
    elif model == "ridge":
        cols = list(RIDGE_BASE_FEATURES)
    else:
        raise ValueError(f"Unknown model type: {model!r}. Use 'rl', 'transformer', or 'ridge'.")

    if use_kaggle_sentiment:
        for c in KAGGLE_SENTIMENT_FEATURES.get(asset, []):
            if c not in cols:
                cols.append(c)

    if use_gdelt_sentiment:
        for c in GDELT_FEATURES.get(asset, []):
            if c not in cols:
                cols.append(c)

    return cols


def validate_feature_schema(
    current_columns: list[str],
    expected_columns: list[str],
    context: str = "",
    force_retrain: bool = False,
) -> list[str]:
    """
    Validate that current_columns matches expected_columns exactly.

    Returns expected_columns (use it to subset your dataframe).
    Raises ValueError on mismatch unless force_retrain=True (which just logs).

    Parameters
    ----------
    current_columns  : columns present in df / scaler
    expected_columns : from get_feature_columns()
    context          : label for error messages, e.g. "BTC RL scaler"
    force_retrain    : if True, log warning instead of raising

    Returns
    -------
    expected_columns — always use this to subset X.
    """
    current_set  = set(current_columns)
    expected_set = set(expected_columns)
    missing = [c for c in expected_columns if c not in current_set]
    extra   = [c for c in current_columns  if c not in expected_set]
    order_ok = (
        not missing and not extra
        and [c for c in current_columns if c in expected_set] == expected_columns
    )

    if not missing and not extra and order_ok:
        return expected_columns

    lines = [f"Feature schema mismatch for {context}:"]
    if missing:
        lines.append(f"  Missing features : {missing}")
    if extra:
        lines.append(f"  Extra features   : {extra}")
    if not missing and not extra:
        lines.append("  Order mismatch (same features, wrong order).")
    lines.append(f"  Expected count   : {len(expected_columns)}")
    lines.append(f"  Current count    : {len(current_columns)}")
    lines.append(
        "  Suggested fix    : regenerate features and retrain with "
        "--retrain rl or --retrain-all."
    )
    msg = "\n".join(lines)

    if force_retrain:
        logger.warning(msg + "\n  force_retrain=True → proceeding with refit.")
    else:
        logger.error(msg)
        raise ValueError(msg)

    return expected_columns


def get_rl_state_dim(
    asset: str,
    use_kaggle_sentiment: bool = False,
    use_gdelt_sentiment: bool = False,
) -> int:
    """Total RL observation vector length for the given feature config."""
    market_cols = get_feature_columns(
        asset, model="rl",
        use_kaggle_sentiment=use_kaggle_sentiment,
        use_gdelt_sentiment=use_gdelt_sentiment,
    )
    return len(market_cols) + len(TRANSFORMER_OUTPUT_COLS) + PORTFOLIO_STATE_DIM


def save_schema_metadata(path, asset: str, model_type: str,
                          feature_columns: list[str],
                          use_kaggle_sentiment: bool = False,
                          use_gdelt_sentiment: bool = False,
                          extra: dict = None) -> None:
    """Persist schema metadata alongside every model checkpoint."""
    import json
    from datetime import datetime
    meta = {
        "asset":                 asset,
        "model_type":            model_type,
        "feature_columns":       feature_columns,
        "feature_count":         len(feature_columns),
        "use_kaggle_sentiment":  use_kaggle_sentiment,
        "use_gdelt_sentiment":   use_gdelt_sentiment,
        "created_at":            datetime.utcnow().isoformat(),
    }
    if extra:
        meta.update(extra)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
