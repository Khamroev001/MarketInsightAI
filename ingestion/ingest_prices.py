"""
ingestion/ingest_prices.py
Phase 2 — Ingest ALL four assets via yfinance → PostgreSQL.

Usage:
    python -m ingestion.ingest_prices
    python -m ingestion.ingest_prices --asset BTC --days 90
"""

import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import psycopg2.extras
from loguru import logger
from clients.yfinance_client import YFinanceClient, SUPPORTED_SYMBOLS
from db.connection import get_conn, log_ingestion

ALL_ASSETS = ["BTC", "ETH", "GOLD", "OIL"]


def upsert_price_bars(conn, asset, df) -> int:
    if df.empty:
        return 0
    records = [
        (asset, "yfinance", row.timestamp,
         float(row.open), float(row.high), float(row.low), float(row.close),
         float(row.volume) if row.volume is not None else None)
        for row in df.itertuples(index=False)
    ]
    sql = """INSERT INTO raw_price_bars (asset, source, ts, open, high, low, close, volume)
             VALUES %s ON CONFLICT (asset, ts) DO NOTHING"""
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, records, page_size=500)
    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default=None, help="BTC, ETH, GOLD, or OIL (default: all)")
    parser.add_argument("--days",  type=int, default=90)
    args = parser.parse_args()

    client = YFinanceClient()

    assets = [args.asset] if args.asset else ALL_ASSETS
    total  = 0
    for asset in assets:
        try:
            symbol = SUPPORTED_SYMBOLS[asset]
            df = client.get_history(symbol, interval="1h", period=f"{args.days}d")
            with get_conn() as conn:
                n = upsert_price_bars(conn, asset, df)
            logger.success(f"Saved {n} rows for {asset}")
            log_ingestion("yfinance", asset, "success", rows_saved=n)
            total += n
        except Exception as e:
            logger.error(f"Failed {asset}: {e}")
            log_ingestion("yfinance", asset, "error", error_msg=str(e))

    logger.info(f"Done. Total rows saved: {total}")


def run(*args, **kwargs):
    """
    Pipeline-compatible entrypoint for run_pipeline.py.

    Prevents ingest_prices.main() from accidentally parsing
    run_pipeline.py arguments like --use-gdelt-sentiment.
    """
    import sys

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0]]
        return main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
