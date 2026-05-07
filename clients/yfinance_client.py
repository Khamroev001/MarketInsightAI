"""
yfinance_client.py
Yahoo Finance market data client for MarketInsight AI.

Handles OHLCV data for: BTC-USD, ETH-USD, GC=F (Gold), CL=F (WTI Oil)
"""

import logging
import pandas as pd
import yfinance as yf
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REQUIRED_COLUMNS = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]

SUPPORTED_SYMBOLS = {
    "BTC":  "BTC-USD",
    "ETH":  "ETH-USD",
    "GOLD": "GC=F",
    "OIL":  "CL=F",
}


class YFinanceClient:
    """
    Client for fetching, normalizing, and validating OHLCV market data
    from Yahoo Finance via the yfinance library.
    """

    def __init__(self, timeout: int = 30, default_interval: str = "1h"):
        self.timeout          = timeout
        self.default_interval = default_interval

    def get_history(self, symbol: str, interval: str = "1h",
                    start: Optional[str] = None, end: Optional[str] = None,
                    period: Optional[str] = None) -> pd.DataFrame:
        """Fetch historical OHLCV data for one symbol."""
        if period is None and (start is None or end is None):
            period = "30d"
        logger.info(f"Fetching {symbol} | interval={interval} | "
                    f"{'period=' + period if period else f'start={start} end={end}'}")
        try:
            if period:
                raw = yf.download(symbol, period=period, interval=interval,
                                  progress=False, auto_adjust=True, timeout=self.timeout)
            else:
                raw = yf.download(symbol, start=start, end=end, interval=interval,
                                  progress=False, auto_adjust=True, timeout=self.timeout)
        except Exception as e:
            raise RuntimeError(f"yfinance download failed for {symbol}: {e}") from e

        df = self.normalize_ohlcv(raw, symbol)
        df = self.validate_price_frame(df, symbol)
        self._log_frame_summary(df, symbol)
        return df

    def download_batch(self, symbols: list[str], interval: str = "1h",
                       start: Optional[str] = None, end: Optional[str] = None,
                       period: Optional[str] = None) -> dict[str, pd.DataFrame]:
        """Download multiple symbols at once, skipping failures gracefully."""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_history(symbol, interval=interval,
                                                   start=start, end=end, period=period)
            except Exception as e:
                logger.error(f"Batch download failed for {symbol}: {e}")
        logger.info(f"Batch complete — {len(results)}/{len(symbols)} symbols succeeded")
        return results

    def normalize_ohlcv(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Transform raw yfinance output into standardized schema."""
        if df.empty:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        for candidate in ["datetime", "date", "index"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "timestamp"})
                break
        if "timestamp" not in df.columns and df.columns[0] not in REQUIRED_COLUMNS:
            df = df.rename(columns={df.columns[0]: "timestamp"})
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = self._safe_numeric_cast(df[col])
        df["symbol"] = symbol
        df = self._standardize_timestamp_column(df)
        cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
        df = df[cols]
        df = df.drop_duplicates(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def validate_price_frame(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate a normalized OHLCV DataFrame."""
        if df.empty:
            raise ValueError(f"Empty DataFrame for {symbol} — no data returned.")
        missing = [c for c in ["timestamp", "open", "high", "low", "close"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for {symbol}: {missing}")
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            raise ValueError(f"timestamp column is not datetime for {symbol}")
        for col in ["open", "high", "low", "close"]:
            if col in df.columns and df[col].isna().all():
                raise ValueError(f"Column '{col}' is entirely null for {symbol}")
        null_pct = df[["open", "high", "low", "close"]].isna().mean().mean()
        if null_pct > 0.1:
            logger.warning(f"{symbol}: {null_pct:.1%} null values in OHLC columns")
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def resample_bars(self, df: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
        """Resample OHLCV data to a target frequency."""
        if df.empty:
            return df
        symbol = df["symbol"].iloc[0] if "symbol" in df.columns else ""
        df = df.set_index("timestamp")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        agg = {k: v for k, v in
               {"open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum"}.items()
               if k in df.columns}
        resampled = df.resample(freq).agg(agg)
        resampled = resampled.dropna(subset=["open", "high", "low", "close"])
        resampled = resampled.reset_index()
        resampled["symbol"] = symbol
        cols = [c for c in REQUIRED_COLUMNS if c in resampled.columns]
        return resampled[cols].reset_index(drop=True)

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Fetch the latest available close price for a symbol."""
        try:
            raw = yf.download(symbol, period="5d", interval="1h",
                              progress=False, auto_adjust=True, timeout=self.timeout)
            if raw.empty:
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            close_col = next((c for c in raw.columns if c.lower() == "close"), None)
            if close_col is None:
                return None
            return float(raw[close_col].dropna().iloc[-1])
        except Exception as e:
            logger.error(f"get_latest_price failed for {symbol}: {e}")
            return None

    def get_ticker_metadata(self, symbol: str) -> dict:
        """Return lightweight metadata for a symbol."""
        meta = {"symbol": symbol, "currency": None,
                "exchange": None, "quote_type": None, "timezone": None}
        try:
            info = yf.Ticker(symbol).info
            meta["currency"]   = info.get("currency")
            meta["exchange"]   = info.get("exchange")
            meta["quote_type"] = info.get("quoteType")
            meta["timezone"]   = info.get("exchangeTimezoneName")
        except Exception as e:
            logger.warning(f"Could not fetch metadata for {symbol}: {e}")
        return meta

    def fetch_and_prepare(self, symbol: str, interval: str = "1h",
                          start: Optional[str] = None, end: Optional[str] = None,
                          period: Optional[str] = None,
                          resample_freq: Optional[str] = None) -> pd.DataFrame:
        """Convenience wrapper: fetch history and optionally resample."""
        df = self.get_history(symbol, interval=interval,
                              start=start, end=end, period=period)
        if resample_freq:
            df = self.resample_bars(df, freq=resample_freq)
        return df

    def _safe_numeric_cast(self, series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    def _standardize_timestamp_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" not in df.columns:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        return df

    def _log_frame_summary(self, df: pd.DataFrame, symbol: str) -> None:
        if df.empty:
            logger.info(f"{symbol}: 0 rows")
            return
        logger.info(f"{symbol}: {len(df)} rows | "
                    f"{df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()} | "
                    f"latest close: {df['close'].iloc[-1]:,.2f}")


if __name__ == "__main__":
    client = YFinanceClient()

    print("=== BTC-USD (30 days, hourly) ===")
    btc = client.get_history("BTC-USD", interval="1h", period="30d")
    print(btc.head())
    print()

    print("=== Batch download ===")
    batch = client.download_batch(["BTC-USD", "ETH-USD", "GC=F", "CL=F"],
                                  interval="1d", period="30d")
    for sym, df in batch.items():
        print(f"{sym}: {len(df)} rows")
    print()

    print("=== Latest prices ===")
    for sym in ["BTC-USD", "ETH-USD", "GC=F", "CL=F"]:
        price = client.get_latest_price(sym)
        print(f"  {sym}: {price:,.2f}" if price else f"  {sym}: unavailable")
    print()

    print("=== BTC resampled to 15min ===")
    btc_15m = client.fetch_and_prepare("BTC-USD", interval="1h",
                                        period="7d", resample_freq="15min")
    print(btc_15m.tail())