"""
ingestion/ingest_news_finnhub.py
Fetch Finnhub news for the last N days in monthly (~30-day) chunks and
upsert to raw_news.  ON CONFLICT (url) DO NOTHING makes re-runs safe.

Usage:
    python -m ingestion.ingest_news_finnhub --days 365
    python -m ingestion.ingest_news_finnhub --asset BTC --days 30
"""

import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime, timedelta, timezone

import psycopg2.extras
from loguru import logger

from config import ALL_ASSETS, FINNHUB_KEY
from db.connection import get_conn, log_ingestion
from clients.finnhub_news import FinnhubNewsClient

# 12 chunks × ~30 days each ≈ 365 days of history per asset
_CHUNK_DAYS = 30


def upsert_articles(conn, articles: list[dict]) -> int:
    """
    Upsert news articles into raw_news.
    Silently skips records with no URL or no timestamp.
    Returns the number of rows attempted (conflicts excluded by the DB).
    """
    if not articles:
        return 0

    records = []
    for a in articles:
        if not a.get("url") or not a.get("published_at"):
            continue
        records.append((
            "finnhub",          # source
            [a["asset"]],       # asset_tags  (TEXT[])
            a["published_at"],  # published_at
            a["title"],         # title
            a["url"],           # url
            None,               # raw_json
        ))

    if not records:
        return 0

    sql = """
        INSERT INTO raw_news (source, asset_tags, published_at, title, url, raw_json)
        VALUES %s
        ON CONFLICT (url) DO NOTHING
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, records, page_size=200)
    return len(records)


def run(assets: list = None, days: int = 365) -> dict[str, int]:
    """Fetch and upsert Finnhub news in monthly chunks for each asset."""
    if not FINNHUB_KEY:
        logger.error("FINNHUB_KEY is not set in .env — aborting ingest_news_finnhub")
        return {}

    assets = assets or ALL_ASSETS
    client = FinnhubNewsClient(FINNHUB_KEY)

    now       = datetime.now(tz=timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0)
    start_all = now - timedelta(days=days)

    results: dict[str, int] = {}
    for asset in assets:
        total     = 0
        chunk_end = now

        while chunk_end > start_all:
            chunk_start = max(chunk_end - timedelta(days=_CHUNK_DAYS), start_all)
            from_str    = chunk_start.strftime("%Y-%m-%d")
            to_str      = chunk_end.strftime("%Y-%m-%d")

            try:
                articles = client.get_news(asset, from_str, to_str)
                with get_conn() as conn:
                    n = upsert_articles(conn, articles)
                total += n
                month_label = chunk_start.strftime("%Y-%m")
                print(f"{asset}: fetched {len(articles)} articles for month {month_label}")
                logger.info(
                    f"{asset} [{from_str} → {to_str}]: "
                    f"{len(articles)} fetched, {n} submitted"
                )
            except Exception as exc:
                logger.error(f"{asset} [{from_str} → {to_str}]: {exc}")
                log_ingestion("ingest_news_finnhub", asset, "error",
                              error_msg=str(exc))

            chunk_end = chunk_start

        log_ingestion("ingest_news_finnhub", asset, "success", rows_saved=total)
        results[asset] = total
        logger.success(f"{asset}: {total} total article rows submitted")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Backfill Finnhub news into raw_news (monthly chunks)"
    )
    parser.add_argument(
        "--days", type=int, default=365,
        help="Number of days of history to backfill (default: 365)"
    )
    parser.add_argument(
        "--asset", default=None,
        help="Single asset to ingest: BTC|ETH|GOLD|OIL (default: all)"
    )
    args = parser.parse_args()

    assets  = [args.asset.upper()] if args.asset else ALL_ASSETS
    results = run(assets=assets, days=args.days)
    logger.info(f"Done. Total articles submitted: {sum(results.values()):,}")


if __name__ == "__main__":
    main()
