"""
ingestion/ingest_news.py
Phase 2 — News ingestion via yfinance → PostgreSQL.

Usage:
    python -m ingestion.ingest_news
    python -m ingestion.ingest_news --asset BTC
"""

import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import psycopg2.extras
from loguru import logger
from clients.yfinance_news import YFinanceNewsClient, NEWS_SYMBOLS
from db.connection import get_conn, log_ingestion


def upsert_news(conn, df) -> int:
    if df.empty:
        return 0
    records = [
        (
            "yfinance",
            [row.asset_tag],
            row.published_at,
            row.title,
            row.url,
        )
        for row in df.itertuples(index=False)
    ]
    sql = """
        INSERT INTO raw_news (source, asset_tags, published_at, title, url)
        VALUES %s
        ON CONFLICT (url) DO NOTHING
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, records, page_size=200)
    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default=None, help="BTC, ETH, GOLD, or OIL (default: all)")
    args = parser.parse_args()

    client = YFinanceNewsClient()
    if not client.ping():
        logger.error("yfinance news ping failed")
        sys.exit(1)

    total = 0
    assets = [args.asset] if args.asset else list(NEWS_SYMBOLS.keys())

    for asset in assets:
        try:
            df = client.get_news(asset)
            with get_conn() as conn:
                n = upsert_news(conn, df)
            logger.success(f"Saved {n} articles for {asset}")
            log_ingestion("yfinance_news", asset, "success", rows_saved=n)
            total += n
        except Exception as e:
            logger.error(f"Failed {asset}: {e}")
            log_ingestion("yfinance_news", asset, "error", error_msg=str(e))

    logger.info(f"Done. Total articles saved: {total}")


if __name__ == "__main__":
    main()
