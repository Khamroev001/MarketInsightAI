import pandas as pd
import yfinance as yf
from datetime import timezone, datetime
from loguru import logger

NEWS_SYMBOLS = {
    "BTC":  "BTC-USD",
    "ETH":  "ETH-USD",
    "GOLD": "GLD",
    "OIL":  "CL=F",
}


class YFinanceNewsClient:

    def ping(self) -> bool:
        try:
            articles = yf.Ticker("BTC-USD").news
            ok = len(articles) > 0
            if ok:
                logger.info(f"yfinance news ping OK — {len(articles)} articles for BTC")
            return ok
        except Exception as e:
            logger.error(f"yfinance news ping failed: {e}")
            return False

    def get_news(self, asset: str) -> pd.DataFrame:
        symbol = NEWS_SYMBOLS.get(asset)
        if not symbol:
            raise ValueError(f"Unknown asset: {asset}. Use {list(NEWS_SYMBOLS)}")

        logger.info(f"Fetching news for {asset} ({symbol})")

        try:
            articles = yf.Ticker(symbol).news
        except Exception as e:
            logger.error(f"Failed to fetch news for {asset}: {e}")
            return pd.DataFrame()

        if not articles:
            logger.warning(f"No news returned for {asset}")
            return pd.DataFrame()

        rows = []
        for a in articles:
            # new nested structure under 'content'
            content = a.get("content", {})
            title   = content.get("title", "")
            url     = content.get("canonicalUrl", {}).get("url", "")
            pub     = content.get("pubDate", "")
            publisher = content.get("provider", {}).get("displayName", "")
            summary = content.get("summary", "")

            # parse pubDate string e.g. "2026-03-28T10:00:05Z"
            try:
                published_at = datetime.strptime(pub, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            except Exception:
                published_at = datetime.fromtimestamp(0, tz=timezone.utc)

            rows.append({
                "asset_tag":    asset,
                "published_at": published_at,
                "title":        title,
                "summary":      summary,
                "url":          url,
                "publisher":    publisher,
            })

        df = pd.DataFrame(rows)
        logger.info(f"  {len(df)} articles fetched for {asset}")
        return df

    def get_all_news(self) -> pd.DataFrame:
        frames = []
        for asset in NEWS_SYMBOLS:
            df = self.get_news(asset)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["url"])
        return combined.sort_values("published_at", ascending=False).reset_index(drop=True)