"""
clients/finnhub_news.py
Finnhub REST client for fetching market news articles.

Supported assets and endpoints:
  BTC  → /news?category=crypto  (free-tier compatible; filtered by keyword)
  ETH  → /news?category=crypto  (free-tier compatible; filtered by keyword)
  GOLD → /news?category=forex   (filtered by keyword)
  OIL  → /news?category=forex   (filtered by keyword)

The company-news endpoint is NOT used for any asset (403 on free tier
for BINANCE:BTCUSDT / BINANCE:ETHUSDT).
"""

import time
from datetime import datetime

import requests
from loguru import logger


# ── Relevance keyword filters per asset ────────────────────────────────────────
_KEYWORDS: dict[str, list[str]] = {
    "BTC":  ["bitcoin", "btc", "crypto"],
    "ETH":  ["ethereum", "eth", "ether"],
    "GOLD": ["gold", "xau", "precious"],
    "OIL":  ["oil", "crude", "wti", "opec"],
}

# ── Finnhub category per asset ────────────────────────────────────────────────
_CATEGORY: dict[str, str] = {
    "BTC":  "crypto",
    "ETH":  "crypto",
    "GOLD": "forex",
    "OIL":  "forex",
}


class FinnhubNewsClient:
    """Fetch news from the Finnhub REST API (free-tier compatible)."""

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str):
        self.api_key  = api_key
        self._session = requests.Session()

    def get_news(self, asset: str, from_date: str, to_date: str) -> list[dict]:
        """
        Fetch news for *asset* between from_date and to_date (YYYY-MM-DD strings).

        Uses the general /news?category=<cat> endpoint for all assets.
        Articles are filtered by relevance keywords so only the most
        pertinent stories are returned.

        Returns a list of article dicts with keys:
            title, published_at, source, url, summary, asset

        Rate-limited: always sleeps 1.1 s before returning.
        """
        asset = asset.upper()
        if asset not in _CATEGORY:
            logger.warning(f"FinnhubNewsClient: unknown asset '{asset}'")
            time.sleep(1.1)
            return []

        articles = self._market_news(asset)
        time.sleep(1.1)
        return articles

    # ── private helpers ───────────────────────────────────────────────────────

    def _market_news(self, asset: str) -> list[dict]:
        category = _CATEGORY[asset]
        keywords = _KEYWORDS.get(asset, [])
        try:
            resp = self._session.get(
                f"{self.BASE_URL}/news",
                params={"category": category, "token": self.api_key},
                timeout=15,
            )
            resp.raise_for_status()
            raw = resp.json()
        except Exception as exc:
            logger.warning(f"Finnhub market-news {asset} (category={category}): {exc}")
            return []

        results = []
        for item in raw:
            parsed = self._parse(item, asset)
            title_lower = (parsed.get("title") or "").lower()
            if keywords and not any(kw in title_lower for kw in keywords):
                continue
            results.append(parsed)

        return results

    @staticmethod
    def _parse(item: dict, asset: str) -> dict:
        ts = item.get("datetime") or 0
        try:
            published_at = datetime.utcfromtimestamp(int(ts)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        except Exception:
            published_at = None
        return {
            "title":        item.get("headline") or item.get("title") or "",
            "published_at": published_at,
            "source":       item.get("source", "finnhub"),
            "url":          item.get("url") or "",
            "summary":      item.get("summary") or "",
            "asset":        asset,
        }
