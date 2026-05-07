"""
pipeline/features.py
Phase 4 — Build feature matrix directly from raw_price_bars + news.

Computes per-asset:
  • Technical indicators  : RSI(14), MACD(12,26,9), Bollinger %B(20), ATR(14),
                            rolling vol(20)
  • Price features        : lagged returns ×4, rolling mean/std ×3, momentum ×2
  • LLM alpha signals     : Claude API called only when news exists for the asset
                            in the preceding 24 h.  No news → alpha_1..5 = 0.0.
  • Cross-asset features  : BTC lag→ETH; OIL vol→GOLD
  • Macro features        : DXY, VIX, SPY fetched daily from yfinance,
                            forward-filled to bar frequency.
                            Labeled as daily signals — NOT hourly.
  • Regime               : 4 one-hot boolean columns
                            (regime_low_vol, regime_bull, regime_bear, regime_high_vol)

Reads from:  raw_price_bars
Writes to:   features table

Usage:
    python -m pipeline.features
    python -m pipeline.features --asset BTC
    python -m pipeline.features --start 2024-01-01 --end 2024-06-01
"""

import argparse, sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_GDELT_FILES = {
    "BTC":  os.path.join(_PROJECT_ROOT, "data", "processed", "gdelt_btc_sentiment_features.csv"),
    "ETH":  os.path.join(_PROJECT_ROOT, "data", "processed", "gdelt_eth_sentiment_features.csv"),
    "GOLD": os.path.join(_PROJECT_ROOT, "data", "processed", "gdelt_gold_sentiment_features.csv"),
    "OIL":  os.path.join(_PROJECT_ROOT, "data", "processed", "gdelt_oil_sentiment_features.csv"),
}

_GDELT_COLS = {
    "BTC":  ["btc_gdelt_sentiment_mean", "btc_gdelt_sentiment_std",
             "btc_gdelt_news_volume",    "btc_gdelt_positive_ratio",
             "btc_gdelt_negative_ratio", "btc_gdelt_avg_confidence"],
    "ETH":  ["eth_gdelt_sentiment_mean", "eth_gdelt_sentiment_std",
             "eth_gdelt_news_volume",    "eth_gdelt_positive_ratio",
             "eth_gdelt_negative_ratio", "eth_gdelt_avg_confidence"],
    "GOLD": ["gold_gdelt_sentiment_mean", "gold_gdelt_sentiment_std",
             "gold_gdelt_news_volume",    "gold_gdelt_positive_ratio",
             "gold_gdelt_negative_ratio", "gold_gdelt_avg_confidence"],
    "OIL":  ["oil_gdelt_sentiment_mean", "oil_gdelt_sentiment_std",
             "oil_gdelt_news_volume",    "oil_gdelt_positive_ratio",
             "oil_gdelt_negative_ratio", "oil_gdelt_avg_confidence"],
}

import numpy as np
import pandas as pd
import psycopg2.extras
from loguru import logger

from db.connection import get_conn, log_ingestion
from config import ALL_ASSETS

# ── Optional heavy imports ─────────────────────────────────────────────────────
try:
    import ta
    _TA_AVAILABLE = True
except ImportError:
    logger.warning("'ta' not found — technical indicators disabled. pip install ta")
    _TA_AVAILABLE = False

try:
    import yfinance as yf
    _YFINANCE = True
except ImportError:
    logger.warning("'yfinance' not found — macro features disabled. pip install yfinance")
    _YFINANCE = False

# DeepSeek/LLM alpha disabled — sentiment comes from GDELT + FinBERT only
_DEEPSEEK_AVAILABLE = False
_deepseek_client    = None

# ── LLM alpha config ───────────────────────────────────────────────────────────
_LLM_SYSTEM = (
    "You are a quantitative finance analyst. You read news headlines and recent "
    "price data for a financial asset and output exactly 5 numerical signals as JSON. "
    "Output ONLY valid JSON with keys alpha_1 through alpha_5. No explanation, no markdown, "
    "no preamble. Values must be between -1.0 and +1.0."
)
_ZERO_ALPHAS     = [0.0, 0.0, 0.0, 0.0, 0.0]
# Within a news window, sample every N news-adjacent bars to control API cost.
LLM_SAMPLE_EVERY = 4

# ── Macro tickers ──────────────────────────────────────────────────────────────
_MACRO_TICKERS = {
    "dxy": "DX-Y.NYB",   # US Dollar Index
    "vix": "^VIX",        # CBOE Volatility Index
    "spy": "SPY",         # S&P 500 ETF
}

# ── Schema DDL ─────────────────────────────────────────────────────────────────
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS features (
    id               BIGSERIAL    PRIMARY KEY,
    asset            VARCHAR(10)  NOT NULL,
    ts               TIMESTAMPTZ  NOT NULL,
    close            NUMERIC(20,8),
    volume           NUMERIC(30,8),
    market_open      BOOLEAN,
    rsi_14           NUMERIC(12,6),
    macd             NUMERIC(12,6),
    macd_signal      NUMERIC(12,6),
    macd_hist        NUMERIC(12,6),
    bb_upper         NUMERIC(20,8),
    bb_middle        NUMERIC(20,8),
    bb_lower         NUMERIC(20,8),
    bb_width         NUMERIC(12,6),
    bb_pct           NUMERIC(12,6),
    atr_14           NUMERIC(20,8),
    vol_20           NUMERIC(12,6),
    ret_1            NUMERIC(12,6),
    ret_4            NUMERIC(12,6),
    ret_8            NUMERIC(12,6),
    ret_16           NUMERIC(12,6),
    mean_4           NUMERIC(20,8),
    mean_8           NUMERIC(20,8),
    mean_16          NUMERIC(20,8),
    std_4            NUMERIC(12,6),
    std_8            NUMERIC(12,6),
    std_16           NUMERIC(12,6),
    mom_4            NUMERIC(12,6),
    mom_16           NUMERIC(12,6),
    btc_ret_lag_1    NUMERIC(12,6),
    btc_ret_lag_4    NUMERIC(12,6),
    oil_vol_lag_1    NUMERIC(12,6),
    alpha_1          NUMERIC(8,4),
    alpha_2          NUMERIC(8,4),
    alpha_3          NUMERIC(8,4),
    alpha_4          NUMERIC(8,4),
    alpha_5          NUMERIC(8,4),
    dxy              NUMERIC(12,6),
    vix              NUMERIC(12,6),
    spy              NUMERIC(12,6),
    regime_low_vol   BOOLEAN,
    regime_bull      BOOLEAN,
    regime_bear      BOOLEAN,
    regime_high_vol  BOOLEAN,
    btc_gdelt_sentiment_mean   NUMERIC(12,6),
    btc_gdelt_sentiment_std    NUMERIC(12,6),
    btc_gdelt_news_volume      NUMERIC(12,6),
    btc_gdelt_positive_ratio   NUMERIC(12,6),
    btc_gdelt_negative_ratio   NUMERIC(12,6),
    btc_gdelt_avg_confidence   NUMERIC(12,6),
    eth_gdelt_sentiment_mean   NUMERIC(12,6),
    eth_gdelt_sentiment_std    NUMERIC(12,6),
    eth_gdelt_news_volume      NUMERIC(12,6),
    eth_gdelt_positive_ratio   NUMERIC(12,6),
    eth_gdelt_negative_ratio   NUMERIC(12,6),
    eth_gdelt_avg_confidence   NUMERIC(12,6),
    gold_gdelt_sentiment_mean  NUMERIC(12,6),
    gold_gdelt_sentiment_std   NUMERIC(12,6),
    gold_gdelt_news_volume     NUMERIC(12,6),
    gold_gdelt_positive_ratio  NUMERIC(12,6),
    gold_gdelt_negative_ratio  NUMERIC(12,6),
    gold_gdelt_avg_confidence  NUMERIC(12,6),
    oil_gdelt_sentiment_mean   NUMERIC(12,6),
    oil_gdelt_sentiment_std    NUMERIC(12,6),
    oil_gdelt_news_volume      NUMERIC(12,6),
    oil_gdelt_positive_ratio   NUMERIC(12,6),
    oil_gdelt_negative_ratio   NUMERIC(12,6),
    oil_gdelt_avg_confidence   NUMERIC(12,6),
    inserted_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (asset, ts)
);
CREATE INDEX IF NOT EXISTS idx_features_asset_ts ON features (asset, ts DESC);
"""

_CREATE_LLM_CACHE_TABLE = """
CREATE TABLE IF NOT EXISTS llm_alpha_cache (
    id          BIGSERIAL    PRIMARY KEY,
    asset       VARCHAR(10)  NOT NULL,
    ts          TIMESTAMPTZ  NOT NULL,
    alpha_1     NUMERIC(8,4),
    alpha_2     NUMERIC(8,4),
    alpha_3     NUMERIC(8,4),
    alpha_4     NUMERIC(8,4),
    alpha_5     NUMERIC(8,4),
    inserted_at TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (asset, ts)
);
CREATE INDEX IF NOT EXISTS idx_llm_cache_asset_ts ON llm_alpha_cache (asset, ts DESC);
"""

_FEATURE_COLS = [
    "close", "volume", "market_open",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct",
    "atr_14", "vol_20",
    "ret_1", "ret_4", "ret_8", "ret_16",
    "mean_4", "mean_8", "mean_16",
    "std_4", "std_8", "std_16",
    "mom_4", "mom_16",
    "btc_ret_lag_1", "btc_ret_lag_4", "oil_vol_lag_1",
    "alpha_1", "alpha_2", "alpha_3", "alpha_4", "alpha_5",
    "dxy", "vix", "spy",
    "regime_low_vol", "regime_bull", "regime_bear", "regime_high_vol",
]


# ── Schema ─────────────────────────────────────────────────────────────────────

def ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(_CREATE_TABLE)
        cur.execute(_CREATE_LLM_CACHE_TABLE)
        # Idempotent migrations: add new columns to existing tables
        for col, defn in [
            ("alpha_1",         "NUMERIC(8,4)"),
            ("alpha_2",         "NUMERIC(8,4)"),
            ("alpha_3",         "NUMERIC(8,4)"),
            ("alpha_4",         "NUMERIC(8,4)"),
            ("alpha_5",         "NUMERIC(8,4)"),
            ("dxy",             "NUMERIC(12,6)"),
            ("vix",             "NUMERIC(12,6)"),
            ("spy",             "NUMERIC(12,6)"),
            ("regime_low_vol",  "BOOLEAN"),
            ("regime_bull",     "BOOLEAN"),
            ("regime_bear",     "BOOLEAN"),
            ("regime_high_vol", "BOOLEAN"),
            # GDELT FinBERT sentiment columns
            ("btc_gdelt_sentiment_mean",  "NUMERIC(12,6)"),
            ("btc_gdelt_sentiment_std",   "NUMERIC(12,6)"),
            ("btc_gdelt_news_volume",     "NUMERIC(12,6)"),
            ("btc_gdelt_positive_ratio",  "NUMERIC(12,6)"),
            ("btc_gdelt_negative_ratio",  "NUMERIC(12,6)"),
            ("btc_gdelt_avg_confidence",  "NUMERIC(12,6)"),
            ("eth_gdelt_sentiment_mean",  "NUMERIC(12,6)"),
            ("eth_gdelt_sentiment_std",   "NUMERIC(12,6)"),
            ("eth_gdelt_news_volume",     "NUMERIC(12,6)"),
            ("eth_gdelt_positive_ratio",  "NUMERIC(12,6)"),
            ("eth_gdelt_negative_ratio",  "NUMERIC(12,6)"),
            ("eth_gdelt_avg_confidence",  "NUMERIC(12,6)"),
            ("gold_gdelt_sentiment_mean", "NUMERIC(12,6)"),
            ("gold_gdelt_sentiment_std",  "NUMERIC(12,6)"),
            ("gold_gdelt_news_volume",    "NUMERIC(12,6)"),
            ("gold_gdelt_positive_ratio", "NUMERIC(12,6)"),
            ("gold_gdelt_negative_ratio", "NUMERIC(12,6)"),
            ("gold_gdelt_avg_confidence", "NUMERIC(12,6)"),
            ("oil_gdelt_sentiment_mean",  "NUMERIC(12,6)"),
            ("oil_gdelt_sentiment_std",   "NUMERIC(12,6)"),
            ("oil_gdelt_news_volume",     "NUMERIC(12,6)"),
            ("oil_gdelt_positive_ratio",  "NUMERIC(12,6)"),
            ("oil_gdelt_negative_ratio",  "NUMERIC(12,6)"),
            ("oil_gdelt_avg_confidence",  "NUMERIC(12,6)"),
        ]:
            cur.execute(
                f"ALTER TABLE features ADD COLUMN IF NOT EXISTS {col} {defn}"
            )


# ── Database I/O ───────────────────────────────────────────────────────────────

def load_raw_bars(conn, asset: str,
                  start: str = None, end: str = None) -> pd.DataFrame:
    """Load raw_price_bars for one asset into a UTC DatetimeIndex DataFrame."""
    conditions = ["asset = %s"]
    params     = [asset]
    if start:
        conditions.append("ts >= %s")
        params.append(start)
    if end:
        conditions.append("ts < %s")
        params.append(end)
    sql = f"""
        SELECT ts, open, high, low, close, volume
        FROM   raw_price_bars
        WHERE  {' AND '.join(conditions)}
        ORDER  BY ts
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    if not rows:
        logger.warning(f"{asset}: no raw price bars found")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # raw_price_bars has no market_open; treat all bars as market-open
    df["market_open"] = True
    return df


def load_news(conn, start: str = None, end: str = None) -> pd.DataFrame:
    """Load raw_news titles with timestamps and asset_tags."""
    conditions: list[str] = []
    params: list = []
    if start:
        conditions.append("published_at >= %s")
        params.append(start)
    if end:
        conditions.append("published_at < %s")
        params.append(end)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"""
        SELECT published_at, asset_tags, title
        FROM   raw_news
        {where}
        ORDER  BY published_at
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["published_at", "asset_tags", "title"])
    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df


def upsert_features(conn, asset: str, df: pd.DataFrame) -> int:
    """Upsert feature rows (index = ts) into the features table."""
    if df.empty:
        return 0
    df = df.reset_index()

    gdelt_cols = _GDELT_COLS.get(asset, [])

    def _f(val): return float(val) if pd.notna(val) else None
    def _b(val): return bool(val)  if pd.notna(val) else None

    records = []
    for row in df.itertuples(index=False):
        base = (
            asset, row.ts,
            _f(getattr(row, "close",           None)),
            _f(getattr(row, "volume",          None)),
            _b(getattr(row, "market_open",     None)),
            _f(getattr(row, "rsi_14",          None)),
            _f(getattr(row, "macd",            None)),
            _f(getattr(row, "macd_signal",     None)),
            _f(getattr(row, "macd_hist",       None)),
            _f(getattr(row, "bb_upper",        None)),
            _f(getattr(row, "bb_middle",       None)),
            _f(getattr(row, "bb_lower",        None)),
            _f(getattr(row, "bb_width",        None)),
            _f(getattr(row, "bb_pct",          None)),
            _f(getattr(row, "atr_14",          None)),
            _f(getattr(row, "vol_20",          None)),
            _f(getattr(row, "ret_1",           None)),
            _f(getattr(row, "ret_4",           None)),
            _f(getattr(row, "ret_8",           None)),
            _f(getattr(row, "ret_16",          None)),
            _f(getattr(row, "mean_4",          None)),
            _f(getattr(row, "mean_8",          None)),
            _f(getattr(row, "mean_16",         None)),
            _f(getattr(row, "std_4",           None)),
            _f(getattr(row, "std_8",           None)),
            _f(getattr(row, "std_16",          None)),
            _f(getattr(row, "mom_4",           None)),
            _f(getattr(row, "mom_16",          None)),
            _f(getattr(row, "btc_ret_lag_1",   None)),
            _f(getattr(row, "btc_ret_lag_4",   None)),
            _f(getattr(row, "oil_vol_lag_1",   None)),
            _f(getattr(row, "alpha_1",         None)),
            _f(getattr(row, "alpha_2",         None)),
            _f(getattr(row, "alpha_3",         None)),
            _f(getattr(row, "alpha_4",         None)),
            _f(getattr(row, "alpha_5",         None)),
            _f(getattr(row, "dxy",             None)),
            _f(getattr(row, "vix",             None)),
            _f(getattr(row, "spy",             None)),
            _b(getattr(row, "regime_low_vol",  None)),
            _b(getattr(row, "regime_bull",     None)),
            _b(getattr(row, "regime_bear",     None)),
            _b(getattr(row, "regime_high_vol", None)),
        )
        gdelt_vals = tuple(_f(getattr(row, c, None)) for c in gdelt_cols)
        records.append(base + gdelt_vals)

    base_cols = (
        "asset,ts,"
        "close,volume,market_open,"
        "rsi_14,macd,macd_signal,macd_hist,"
        "bb_upper,bb_middle,bb_lower,bb_width,bb_pct,"
        "atr_14,vol_20,"
        "ret_1,ret_4,ret_8,ret_16,"
        "mean_4,mean_8,mean_16,"
        "std_4,std_8,std_16,"
        "mom_4,mom_16,"
        "btc_ret_lag_1,btc_ret_lag_4,oil_vol_lag_1,"
        "alpha_1,alpha_2,alpha_3,alpha_4,alpha_5,"
        "dxy,vix,spy,"
        "regime_low_vol,regime_bull,regime_bear,regime_high_vol"
    )
    cols = base_cols + ("," + ",".join(gdelt_cols) if gdelt_cols else "")
    update_cols = [c for c in cols.split(",") if c not in ("asset", "ts")]
    update_set  = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)

    sql = f"""
        INSERT INTO features ({cols})
        VALUES %s
        ON CONFLICT (asset, ts) DO UPDATE SET {update_set}
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, records, page_size=500)
    return len(records)


# ── Technical Indicators ───────────────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Add RSI(14), MACD(12,26,9), Bollinger %B(20), ATR(14), rolling vol(20)."""
    if not _TA_AVAILABLE:
        for col in ["rsi_14", "macd", "macd_signal", "macd_hist",
                    "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct",
                    "atr_14", "vol_20"]:
            df[col] = np.nan
        return df

    close = df["close"]
    high  = df["high"] if "high" in df.columns else close
    low   = df["low"]  if "low"  in df.columns else close

    try:
        df["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    except Exception as e:
        logger.warning(f"{asset}: RSI failed — {e}")
        df["rsi_14"] = np.nan

    try:
        _macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        df["macd"]        = _macd.macd()
        df["macd_signal"] = _macd.macd_signal()
        df["macd_hist"]   = _macd.macd_diff()
    except Exception as e:
        logger.warning(f"{asset}: MACD failed — {e}")
        df[["macd", "macd_signal", "macd_hist"]] = np.nan

    try:
        _bb = ta.volatility.BollingerBands(close=close, window=20)
        df["bb_upper"]  = _bb.bollinger_hband()
        df["bb_middle"] = _bb.bollinger_mavg()
        df["bb_lower"]  = _bb.bollinger_lband()
        df["bb_width"]  = _bb.bollinger_wband()
        df["bb_pct"]    = _bb.bollinger_pband()
    except Exception as e:
        logger.warning(f"{asset}: Bollinger Bands failed — {e}")
        df[["bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct"]] = np.nan

    try:
        df["atr_14"] = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range()
    except Exception as e:
        logger.warning(f"{asset}: ATR failed — {e}")
        df["atr_14"] = np.nan

    log_ret      = np.log(close / close.shift(1))
    df["vol_20"] = log_ret.rolling(20).std()

    return df


# ── Price Features ─────────────────────────────────────────────────────────────

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lagged log returns ×4, rolling mean/std ×3, momentum ×2."""
    log_ret = np.log(df["close"] / df["close"].shift(1))

    for lag in [1, 4, 8, 16]:
        df[f"ret_{lag}"] = log_ret.shift(lag - 1).rolling(lag).sum()

    for w in [4, 8, 16]:
        df[f"mean_{w}"] = df["close"].rolling(w).mean()
        df[f"std_{w}"]  = df["close"].rolling(w).std()

    df["mom_4"]  = df["close"] / df["close"].shift(4)  - 1
    df["mom_16"] = df["close"] / df["close"].shift(16) - 1

    return df


# ── Macro Features ─────────────────────────────────────────────────────────────

def fetch_macro_daily(time_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Fetch DXY, VIX, SPY from Yahoo Finance as daily close prices.
    Forward-fill to match the bar time_index.
    These remain daily signals — NOT hourly.  Column names: dxy, vix, spy.
    """
    macro = pd.DataFrame(
        {"dxy": np.nan, "vix": np.nan, "spy": np.nan},
        index=time_index,
    )

    if not _YFINANCE or time_index.empty:
        return macro

    start = (time_index.min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end   = (time_index.max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    for col, ticker in _MACRO_TICKERS.items():
        try:
            raw = yf.download(ticker, start=start, end=end,
                              progress=False, auto_adjust=True)
            if raw.empty:
                logger.warning(f"Macro {ticker}: empty download")
                continue
            # Handle MultiIndex columns from yfinance
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            close_col = next((c for c in raw.columns if c.lower() == "close"), None)
            if close_col is None:
                continue
            series = raw[close_col].copy()
            series.index = pd.to_datetime(series.index, utc=True)
            # Forward-fill daily close to every bar in time_index
            series = series.reindex(
                series.index.union(time_index)
            ).ffill().reindex(time_index)
            macro[col] = series.values
        except Exception as e:
            logger.warning(f"Macro {ticker}: {e}")

    return macro


# ── LLM Alpha Signals ──────────────────────────────────────────────────────────

def _load_cached_alpha_series(conn, asset: str, ts_list: list) -> pd.DataFrame:
    if not ts_list:
        return pd.DataFrame(columns=["ts","alpha_1","alpha_2","alpha_3","alpha_4","alpha_5"])
    sql = """
        SELECT ts, alpha_1, alpha_2, alpha_3, alpha_4, alpha_5
        FROM   llm_alpha_cache
        WHERE  asset = %s AND ts = ANY(%s::timestamptz[])
        ORDER  BY ts
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (asset, ts_list))
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["ts","alpha_1","alpha_2","alpha_3","alpha_4","alpha_5"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for c in [f"alpha_{i}" for i in range(1, 6)]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("ts")


def _save_alpha_cache_batch(conn, asset: str, records: list) -> None:
    if not records:
        return
    sql = """
        INSERT INTO llm_alpha_cache
               (asset, ts, alpha_1, alpha_2, alpha_3, alpha_4, alpha_5)
        VALUES %s
        ON CONFLICT (asset, ts) DO NOTHING
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, records, page_size=200)


def _call_llm(asset: str, ohlcv_df: pd.DataFrame, headlines_text: str) -> list[float]:
    """Call DeepSeek to generate 5 alpha signals. Returns [0.0]*5 silently on any error."""
    if _deepseek_client is None:
        return _ZERO_ALPHAS[:]

    user_prompt = (
        f"Asset: {asset}\n"
        f"Recent OHLCV (last 5 bars, hourly):\n"
        f"{ohlcv_df.tail(5).to_string()}\n\n"
        f"Recent news headlines (last 24 hours):\n"
        f"{headlines_text}\n\n"
        f"Output JSON with exactly these keys and float values between -1.0 and +1.0:\n"
        f"{{\n"
        f'  "alpha_1": <momentum signal from news context>,\n'
        f'  "alpha_2": <mean-reversion probability>,\n'
        f'  "alpha_3": <news impact magnitude and direction>,\n'
        f'  "alpha_4": <volatility regime assessment>,\n'
        f'  "alpha_5": <cross-asset sentiment context>\n'
        f"}}"
    )

    for attempt in range(5):
        try:
            resp = _deepseek_client.chat.completions.create(
                model="deepseek-chat",
                max_tokens=256,
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            text = resp.choices[0].message.content.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                vals = [parsed.get(f"alpha_{i}", 0.0) for i in range(1, 6)]
                return [max(-1.0, min(1.0, float(v))) for v in vals]
            logger.warning(f"DeepSeek returned unexpected format: {text[:80]}")
            return _ZERO_ALPHAS[:]
        except Exception as e:
            err_str = str(e).lower()
            if any(k in err_str for k in ("authentication", "api_key", "auth_token",
                                           "authorization", "401", "403",
                                           "could not resolve")):
                return _ZERO_ALPHAS[:]
            wait = 2 ** attempt
            logger.warning(f"DeepSeek attempt {attempt + 1}/5 failed ({e}); retrying in {wait}s")
            if attempt < 4:
                time.sleep(wait)
    return _ZERO_ALPHAS[:]


def build_llm_alpha_series(
    df: pd.DataFrame,
    news_df: pd.DataFrame,
    asset: str,
    time_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Call DeepSeek ONLY when news articles for this asset exist in the preceding 24 h.
    If article_count == 0 for a bar's 24-hour window → alpha_1..5 = 0.0, API skipped.

    Within news windows, sample every LLM_SAMPLE_EVERY bars to control API cost.
    Results are cached in llm_alpha_cache and forward-filled within news windows.
    Alphas are zeroed for bars that have no recent news.
    """
    zero_df = pd.DataFrame(
        {f"alpha_{i}": 0.0 for i in range(1, 6)},
        index=time_index,
    )

    if not _DEEPSEEK_AVAILABLE or _deepseek_client is None:
        return zero_df

    # Pre-filter news to this asset
    if news_df.empty:
        return zero_df

    mask = news_df["asset_tags"].apply(
        lambda tags: bool(tags) and asset in {t.upper() for t in (tags or [])}
    )
    asset_news = news_df[mask].copy()
    if asset_news.empty:
        return zero_df

    asset_news_sorted = asset_news.set_index("published_at").sort_index()

    # Build a boolean Series: has any news been published for this asset in [t-24h, t]?
    # Mark bars that align with (or follow) a news article publication
    bar_arr   = time_index.values  # numpy array of timestamps
    art_times = asset_news_sorted.index.values

    news_indicator = np.zeros(len(time_index), dtype=int)
    idxs = np.searchsorted(bar_arr, art_times, side="left")
    valid = idxs < len(bar_arr)
    np.add.at(news_indicator, idxs[valid], 1)

    news_series  = pd.Series(news_indicator, index=time_index, dtype=int)
    # Rolling 24-bar window (= 24 h at hourly cadence)
    has_news_24h = news_series.rolling(24, min_periods=1).sum() > 0

    # Sample timestamps: bars inside news windows, every LLM_SAMPLE_EVERY-th
    news_ts_all = time_index[has_news_24h.values]
    sample_ts   = [ts for i, ts in enumerate(news_ts_all) if i % LLM_SAMPLE_EVERY == 0]

    if not sample_ts:
        return zero_df

    # Load cache
    try:
        with get_conn() as conn:
            cached = _load_cached_alpha_series(conn, asset, sample_ts)
    except Exception as e:
        logger.warning(f"{asset}: LLM cache load failed — {e}")
        cached = pd.DataFrame()

    cached_set  = set(cached.index) if not cached.empty else set()
    uncached_ts = [ts for ts in sample_ts if ts not in cached_set]
    logger.info(
        f"{asset}: LLM alphas — {len(cached_set)} cached, "
        f"{len(uncached_ts)} to compute (news-conditioned)"
    )

    new_records = []

    for ts in uncached_ts:
        window = df[df.index <= ts].tail(5)
        ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in window.columns]
        ohlcv_window = window[ohlcv_cols]

        recent = asset_news_sorted[asset_news_sorted.index <= ts].tail(5)
        heads  = "\n".join(recent["title"].fillna("").tolist()) or "No recent news"

        # Skip API call if no articles in window (article_count == 0)
        article_count = int(has_news_24h.loc[ts]) if ts in has_news_24h.index else 0
        if article_count == 0:
            alphas = _ZERO_ALPHAS[:]
        else:
            alphas = _call_llm(asset, ohlcv_window, heads)
        new_records.append((asset, ts, *alphas))

    # Persist new cache entries
    if new_records:
        try:
            with get_conn() as conn:
                _save_alpha_cache_batch(conn, asset, new_records)
        except Exception as e:
            logger.warning(f"{asset}: LLM cache save failed — {e}")

    # Build sample-level DataFrame
    newly = {rec[1]: list(rec[2:]) for rec in new_records}
    sample_rows = {}
    for ts in sample_ts:
        if ts in cached_set and not cached.empty and ts in cached.index:
            row = cached.loc[ts]
            sample_rows[ts] = [float(row[f"alpha_{i}"]) for i in range(1, 6)]
        elif ts in newly:
            sample_rows[ts] = newly[ts]
        else:
            sample_rows[ts] = _ZERO_ALPHAS[:]

    sample_df = pd.DataFrame(
        [sample_rows[ts] for ts in sample_ts],
        index=pd.DatetimeIndex(sample_ts),
        columns=[f"alpha_{i}" for i in range(1, 6)],
    )

    # Forward-fill to all bars; then zero out bars that have no news in 24 h
    alpha_df = sample_df.reindex(time_index).ffill().fillna(0.0)
    for col in [f"alpha_{i}" for i in range(1, 6)]:
        alpha_df.loc[~has_news_24h, col] = 0.0

    return alpha_df


# ── Regime Detection ───────────────────────────────────────────────────────────

def detect_regime(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Classify each bar into one of 4 market regimes, encoded as 4 one-hot columns:
      regime_low_vol  — rolling std < 25th percentile
      regime_bull     — rolling mean > 0 AND std in [p25, p75)
      regime_bear     — rolling mean ≤ 0 AND std in [p25, p75)
      regime_high_vol — rolling std ≥ 75th percentile

    Returns a DataFrame with those 4 boolean columns.
    NaN during the warm-up period is represented as all-False.
    """
    log_ret      = np.log(df["close"] / df["close"].shift(1))
    rolling_mean = log_ret.rolling(window).mean()
    rolling_std  = log_ret.rolling(window).std()

    try:
        p25 = float(rolling_std.quantile(0.25))
        p75 = float(rolling_std.quantile(0.75))
    except Exception:
        p25, p75 = 0.0, float("inf")

    warm_up = rolling_std.isna()

    regime_low_vol  = (~warm_up) & (rolling_std < p25)
    regime_high_vol = (~warm_up) & (rolling_std >= p75)
    regime_bull     = (~warm_up) & (~regime_low_vol) & (~regime_high_vol) & (rolling_mean > 0)
    regime_bear     = (~warm_up) & (~regime_low_vol) & (~regime_high_vol) & (rolling_mean <= 0)

    return pd.DataFrame({
        "regime_low_vol":  regime_low_vol.astype("boolean"),
        "regime_bull":     regime_bull.astype("boolean"),
        "regime_bear":     regime_bear.astype("boolean"),
        "regime_high_vol": regime_high_vol.astype("boolean"),
    }, index=df.index)


# ── Cross-asset Features ───────────────────────────────────────────────────────

def add_cross_asset_features(
    asset_dfs: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    BTC return lags → ETH columns  (btc_ret_lag_1, btc_ret_lag_4)
    OIL volatility  → GOLD columns (oil_vol_lag_1)
    All other assets get NaN for these columns.
    """
    for df in asset_dfs.values():
        for col in ["btc_ret_lag_1", "btc_ret_lag_4", "oil_vol_lag_1"]:
            if col not in df.columns:
                df[col] = np.nan

    btc_df = asset_dfs.get("BTC")
    oil_df = asset_dfs.get("OIL")

    if btc_df is not None and "ETH" in asset_dfs:
        btc_ret = np.log(btc_df["close"] / btc_df["close"].shift(1))
        eth_df  = asset_dfs["ETH"]
        eth_df["btc_ret_lag_1"] = btc_ret.shift(1).reindex(eth_df.index)
        eth_df["btc_ret_lag_4"] = btc_ret.shift(4).reindex(eth_df.index)
        asset_dfs["ETH"] = eth_df

    if oil_df is not None and "GOLD" in asset_dfs:
        oil_log_ret          = np.log(oil_df["close"] / oil_df["close"].shift(1))
        oil_vol              = oil_log_ret.rolling(20).std()
        gold_df              = asset_dfs["GOLD"]
        gold_df["oil_vol_lag_1"] = oil_vol.shift(1).reindex(gold_df.index)
        asset_dfs["GOLD"]   = gold_df

    return asset_dfs


# ── Per-asset pipeline ─────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Compute all features for one asset DataFrame (index = ts)."""
    df = df.copy()
    df = add_technical_indicators(df, asset)
    df = add_price_features(df)

    # Alpha columns: zeroed — DeepSeek/LLM disabled; sentiment via GDELT + FinBERT
    for col in [f"alpha_{i}" for i in range(1, 6)]:
        df[col] = 0.0

    # Regime: 4 one-hot boolean columns
    regime_df = detect_regime(df)
    for col in ["regime_low_vol", "regime_bull", "regime_bear", "regime_high_vol"]:
        df[col] = regime_df[col]

    return df


# ── GDELT Sentiment Merge ──────────────────────────────────────────────────────

def merge_gdelt_features(asset_df: pd.DataFrame, asset: str,
                          use_gdelt_sentiment: bool = False) -> pd.DataFrame:
    """Left-join the GDELT FinBERT sentiment CSV into the feature DataFrame."""
    if not use_gdelt_sentiment:
        return asset_df

    csv_path = _GDELT_FILES.get(asset)
    if csv_path is None:
        logger.warning(f"{asset}: no GDELT CSV path configured — skipping")
        return asset_df

    if not os.path.exists(csv_path):
        logger.warning(f"{asset}: GDELT file not found at {csv_path} — skipping")
        return asset_df

    try:
        gdelt = pd.read_csv(csv_path)
        logger.info(f"{asset}: GDELT file loaded — {len(gdelt)} rows from {csv_path}")
    except Exception as e:
        logger.error(f"{asset}: GDELT CSV load failed — {e}")
        return asset_df

    gdelt["timestamp_utc"] = pd.to_datetime(gdelt["timestamp_utc"], utc=True)
    gdelt = gdelt.set_index("timestamp_utc").sort_index()
    logger.info(f"{asset}: GDELT rows available — {len(gdelt)}")

    gdelt_cols = _GDELT_COLS.get(asset, [])
    present = [c for c in gdelt_cols if c in gdelt.columns]
    gdelt = gdelt[present]

    asset_df = asset_df.join(gdelt, how="left")

    total_filled = 0
    for col in gdelt_cols:
        if col in asset_df.columns:
            n_missing = int(asset_df[col].isna().sum())
            total_filled += n_missing
            asset_df[col] = asset_df[col].fillna(0.0)
        else:
            asset_df[col] = 0.0
            total_filled += len(asset_df)

    logger.info(f"{asset}: GDELT columns merged — {gdelt_cols}")
    logger.info(f"{asset}: GDELT missing values filled with 0 ({total_filled} cells)")
    return asset_df


# ── Orchestration ──────────────────────────────────────────────────────────────

def run(assets: list = None, start: str = None, end: str = None,
        use_gdelt_sentiment: bool = False) -> dict[str, int]:
    """Phase 4 pipeline: raw bars → features → cross-asset → macro → save."""
    assets = assets or ALL_ASSETS

    with get_conn() as conn:
        ensure_table(conn)

    # Build per-asset feature DataFrames
    asset_dfs: dict[str, pd.DataFrame] = {}
    for asset in assets:
        try:
            with get_conn() as conn:
                raw = load_raw_bars(conn, asset, start=start, end=end)
            if raw.empty:
                logger.warning(f"{asset}: no raw bars — skipping features")
                continue
            feat = build_features(raw, asset)
            asset_dfs[asset] = feat
            logger.info(f"{asset}: {len(feat)} feature rows computed")
        except Exception as e:
            logger.error(f"{asset}: feature build failed — {e}")
            log_ingestion("features", asset, "error", error_msg=str(e))

    # Cross-asset enrichment (needs all assets loaded)
    if len(asset_dfs) > 1:
        asset_dfs = add_cross_asset_features(asset_dfs)

    # Macro features (DXY, VIX, SPY) — fetched once, shared across all assets
    all_ts = pd.DatetimeIndex(
        sorted({ts for df in asset_dfs.values() for ts in df.index})
    )
    if not all_ts.empty:
        macro_df = fetch_macro_daily(all_ts)
        logger.info(
            f"Macro: dxy={macro_df['dxy'].notna().sum()} "
            f"vix={macro_df['vix'].notna().sum()} "
            f"spy={macro_df['spy'].notna().sum()} bars with data"
        )
        for asset, df in asset_dfs.items():
            for col in ["dxy", "vix", "spy"]:
                df[col] = macro_df[col].reindex(df.index)
            asset_dfs[asset] = df

    # GDELT sentiment merge — runs after cross-asset and macro, before upsert
    for asset in list(asset_dfs.keys()):
        asset_dfs[asset] = merge_gdelt_features(
            asset_dfs[asset], asset, use_gdelt_sentiment
        )

    # Save
    results: dict[str, int] = {}
    for asset, df in asset_dfs.items():
        try:
            with get_conn() as conn:
                n = upsert_features(conn, asset, df)
            logger.success(f"{asset}: saved {n} feature rows")
            log_ingestion("features", asset, "success", rows_saved=n)
            results[asset] = n
        except Exception as e:
            logger.error(f"{asset}: feature save failed — {e}")
            log_ingestion("features", asset, "error", error_msg=str(e))
            results[asset] = 0

    # Report feature matrix dimensions for scaler validation (Task 3A)
    feat_col_count = len(_FEATURE_COLS)
    if asset_dfs:
        sample_df = next(iter(asset_dfs.values()))
        actual_feat_cols = [c for c in _FEATURE_COLS if c in sample_df.columns]
        print(
            f"Feature matrix shape: {sample_df.shape} — "
            f"{feat_col_count} feature columns defined, "
            f"{len(actual_feat_cols)} present in output"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 4: compute feature matrix")
    parser.add_argument("--asset", default=None, help="BTC|ETH|GOLD|OIL (default: all)")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD inclusive")
    parser.add_argument("--end",   default=None, help="YYYY-MM-DD exclusive")
    args = parser.parse_args()

    assets  = [args.asset] if args.asset else ALL_ASSETS
    results = run(assets=assets, start=args.start, end=args.end)

    total = sum(results.values())
    logger.info(f"Done. Total feature rows saved: {total}")
    for asset, n in results.items():
        logger.info(f"  {asset}: {n}")


if __name__ == "__main__":
    main()
