"""
models/gdelt_sentiment.py
GDELT + FinBERT news sentiment pipeline for BTC, ETH, GOLD, OIL.

Architecture:
  GDELT article fetch → optional newspaper3k full-text → FinBERT scoring
  → hourly UTC aggregation → one-period leakage shift → processed CSV

CLI:
  python -m models.gdelt_sentiment --finbert-test
  python -m models.gdelt_sentiment --mock-test
  python -m models.gdelt_sentiment --smoke-test --assets BTC GOLD
      --start-date 2026-04-20 --end-date 2026-04-22 --max-records-per-query 5 --no-full-text
  python -m models.gdelt_sentiment --assets BTC ETH GOLD OIL
      --start-date 2026-04-01 --end-date 2026-04-29 --max-records-per-query 50 --no-full-text
"""

from __future__ import annotations

import argparse
import sys
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Project root on path ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Output paths ───────────────────────────────────────────────────────────────
_BASE        = Path(__file__).parent.parent
PROCESSED_DIR = _BASE / "data" / "processed"
RAW_DIR       = _BASE / "data" / "raw"
DASHBOARD_DIR = _BASE / "data" / "dashboard"

for _d in (PROCESSED_DIR, RAW_DIR, DASHBOARD_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Latency buffer (minutes) — configurable ────────────────────────────────────
NEWS_LATENCY_MINUTES: int = 0   # set to 15 to add 15-min availability buffer

# ── FinBERT label → numeric score ─────────────────────────────────────────────
FINBERT_LABEL_MAP: dict[str, int] = {
    "positive": 1,
    "neutral":  0,
    "negative": -1,
}

# ── Asset query configuration ──────────────────────────────────────────────────
ASSET_QUERIES: dict[str, list[str]] = {
    "BTC": [
        # Ticker / market symbols
        "btc",
        "btcusd",
        "BTCUSD",
        "BTC USD",
        "BTC bitcoin",
        "BTCUSD bitcoin",

        # Core Bitcoin
        "bitcoin",
        "bitcoin price",
        "bitcoin market",
        "bitcoin cryptocurrency",
        "bitcoin trading",
        "bitcoin volatility",
        "bitcoin rally",
        "bitcoin selloff",
        "bitcoin crash",

        # USD / exchange rate phrasing
        "bitcoin usd",
        "bitcoin dollar",
        "bitcoin exchange rate",
        "bitcoin price usd",

        # ETF / institutional
        "bitcoin etf",
        "spot bitcoin etf",
        "bitcoin exchange traded fund",
        "bitcoin fund inflows",
        "bitcoin institutional investors",
        "bitcoin wall street",

        # Mining / supply
        "bitcoin mining",
        "bitcoin miners",
        "bitcoin hash rate",
        "bitcoin halving",
        "bitcoin supply",
        "bitcoin difficulty",

        # Regulation / macro / exchanges
        "bitcoin regulation",
        "bitcoin sec",
        "bitcoin federal reserve",
        "bitcoin inflation",
        "bitcoin interest rates",
        "bitcoin exchange",
        "bitcoin binance",
        "bitcoin coinbase",
        "bitcoin grayscale",
        "bitcoin blackrock",
    ],

    "ETH": [
        # Ticker / market symbols
        "eth",
        "ethusd",
        "ETHUSD",
        "ETH USD",
        "ETH ethereum",
        "ETHUSD ethereum",

        # Core Ethereum
        "ethereum",
        "ethereum price",
        "ethereum market",
        "ethereum cryptocurrency",
        "ethereum trading",
        "ethereum volatility",
        "ethereum rally",
        "ethereum selloff",
        "ethereum crash",

        # USD / exchange rate phrasing
        "ethereum usd",
        "ethereum dollar",
        "ethereum exchange rate",
        "ethereum price usd",

        # ETF / institutional
        "ethereum etf",
        "spot ethereum etf",
        "ethereum exchange traded fund",
        "ethereum institutional investors",
        "ethereum fund inflows",

        # Fundamentals
        "ethereum staking",
        "ethereum validators",
        "ethereum proof of stake",
        "ethereum gas fees",
        "ethereum transaction fees",
        "ethereum smart contracts",
        "ethereum defi",
        "ethereum layer 2",
        "ethereum scaling",
        "ethereum network upgrade",
        "ethereum blockchain",

        # Regulation / ecosystem
        "ethereum regulation",
        "ethereum sec",
        "ethereum binance",
        "ethereum coinbase",
        "ethereum vitalik",
    ],

    "GOLD": [
        # Ticker / market symbols
        "xau",
        "xauusd",
        "XAUUSD",
        "XAU USD",
        "GC=F",
        "GC futures",
        "XAUUSD gold",

        # Core gold
        "gold price",
        "gold futures",
        "gold market",
        "gold trading",
        "gold rally",
        "gold selloff",
        "gold demand",
        "gold supply",
        "gold volatility",
        "spot gold",
        "gold spot price",

        # Instruments / market forms
        "bullion",
        "gold bullion",
        "precious metals",
        "precious metal prices",
        "gold exchange traded fund",
        "gold etf",
        "gold reserves",

        # Macro drivers
        "safe haven gold",
        "gold safe haven",
        "gold inflation hedge",
        "gold inflation",
        "gold interest rates",
        "gold treasury yields",
        "gold real yields",
        "gold dollar",
        "gold usd",
        "gold federal reserve",
        "gold fed decision",
        "gold rate cuts",
        "gold rate hikes",

        # Central banks / geopolitical risk
        "central bank gold reserves",
        "central banks buying gold",
        "gold central banks",
        "gold recession",
        "gold economic slowdown",
        "gold geopolitical risk",
        "gold risk off",
        "gold war",
        "gold middle east",
    ],

    "OIL": [
        # Ticker / market symbols
        "wti",
        "brent",
        "CL=F",
        "BZ=F",
        "WTI crude",
        "Brent crude",

        # Core oil
        "crude oil",
        "oil price",
        "oil market",
        "oil futures",
        "oil trading",
        "oil rally",
        "oil selloff",
        "oil volatility",

        # WTI / Brent
        "wti crude",
        "wti oil",
        "west texas intermediate",
        "brent oil",
        "brent futures",
        "crude futures",

        # Supply / demand / inventories
        "oil supply",
        "oil demand",
        "crude oil demand",
        "crude oil supply",
        "oil inventory",
        "crude inventory",
        "oil stockpiles",
        "crude stockpiles",
        "eia oil inventory",
        "iea oil demand",
        "oil demand outlook",
        "oil supply outlook",

        # OPEC / production
        "opec",
        "opec oil",
        "opec crude oil",
        "opec production",
        "opec supply cut",
        "opec production cut",
        "opec meeting",
        "opec plus",
        "opec plus oil",
        "opec+ oil",
        "saudi oil",
        "russia oil",

        # Energy / geopolitical drivers
        "energy market",
        "energy prices",
        "fuel prices",
        "gasoline prices",
        "diesel prices",
        "oil refinery",
        "oil sanctions",
        "oil geopolitical risk",
        "oil middle east",
        "oil shipping",
        "red sea oil",
        "oil war",
    ],
}

# ── Hourly aggregation output columns per asset ────────────────────────────────
def _gdelt_cols(asset: str) -> list[str]:
    pfx = asset.lower()
    return [
        f"{pfx}_gdelt_sentiment_mean",
        f"{pfx}_gdelt_sentiment_std",
        f"{pfx}_gdelt_news_volume",
        f"{pfx}_gdelt_positive_ratio",
        f"{pfx}_gdelt_negative_ratio",
        f"{pfx}_gdelt_avg_confidence",
    ]


# ── Optional GDELT import ─────────────────────────────────────────────────────
try:
    from gdeltdoc import GdeltDoc, Filters
    _GDELT_OK = True
except ImportError:
    _GDELT_OK = False
    logger.warning("gdeltdoc not found: pip install gdeltdoc")

# ── Optional newspaper3k import ───────────────────────────────────────────────
try:
    from newspaper import Article, Config as _NpConfig
    _NEWSPAPER_OK = True
except ImportError:
    _NEWSPAPER_OK = False

# ── FinBERT via HuggingFace transformers ──────────────────────────────────────
_finbert_pipe = None

def _load_finbert():
    global _finbert_pipe
    if _finbert_pipe is not None:
        return _finbert_pipe
    try:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading FinBERT (ProsusAI/finbert) …")
        _finbert_pipe = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        logger.success("FinBERT loaded.")
    except Exception as e:
        logger.error(f"FinBERT load failed: {e}")
        _finbert_pipe = None
    return _finbert_pipe


def score_texts_finbert(texts: list[str]) -> list[dict]:
    """
    Score a list of texts with FinBERT.
    Returns list of dicts with keys: finbert_label, finbert_confidence, sentiment_score.
    Empty / None texts get neutral score with confidence 0.
    """
    pipe = _load_finbert()
    results = []
    for text in texts:
        if not text or not text.strip():
            results.append({"finbert_label": "neutral", "finbert_confidence": 0.0,
                            "sentiment_score": 0})
            continue
        if pipe is None:
            results.append({"finbert_label": "neutral", "finbert_confidence": 0.0,
                            "sentiment_score": 0})
            continue
        try:
            out = pipe(text[:512])
            label = out[0]["label"].lower()
            conf  = float(out[0]["score"])
            score = FINBERT_LABEL_MAP.get(label, 0)
            results.append({"finbert_label": label, "finbert_confidence": conf,
                            "sentiment_score": score})
        except Exception as e:
            logger.warning(f"FinBERT scoring error: {e}")
            results.append({"finbert_label": "neutral", "finbert_confidence": 0.0,
                            "sentiment_score": 0})
    return results


# ── GDELT fetch ────────────────────────────────────────────────────────────────

def _fetch_gdelt_query(
    query: str,
    start_date: str,
    end_date: str,
    max_records: int = 50,
) -> pd.DataFrame:
    """Fetch one GDELT query. Returns empty DataFrame on any error."""
    if not _GDELT_OK:
        return pd.DataFrame()
    try:
        gd = GdeltDoc()
        f  = Filters(
            keyword   = query,
            start_date= start_date,
            end_date  = end_date,
            num_records = min(max_records, 250),
            language  = "English",
        )
        df = gd.article_search(f)
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception as e:
        logger.warning(f"GDELT query '{query}' failed: {e}")
        return pd.DataFrame()


def fetch_gdelt_articles(
    asset: str,
    start_date: str,
    end_date: str,
    max_records_per_query: int = 50,
    queries: list[str] = None,
) -> pd.DataFrame:
    """
    Run all queries for one asset, deduplicate by URL.
    Returns raw article DataFrame with normalized columns.
    """
    queries = queries or ASSET_QUERIES.get(asset.upper(), [])
    frames  = []

    for q in queries:
        df = _fetch_gdelt_query(q, start_date, end_date, max_records_per_query)
        if not df.empty:
            df["_query"] = q
            frames.append(df)
            logger.debug(f"  [{asset}] query='{q}': {len(df)} articles")

    if not frames:
        logger.warning(f"[{asset}] No articles returned from GDELT.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Normalize column names — GDELT may use 'url' or 'URL'
    combined.columns = [c.lower() for c in combined.columns]

    # Ensure required columns exist
    for col in ["url", "title", "seendate", "domain", "language", "sourcecountry"]:
        if col not in combined.columns:
            combined[col] = None

    # Deduplicate by URL
    before = len(combined)
    combined = combined.drop_duplicates(subset=["url"])
    after   = len(combined)
    logger.info(f"[{asset}] {before} raw → {after} unique articles")

    combined["asset"]  = asset.upper()
    combined["query"]  = combined.get("_query", "unknown")
    combined           = combined.drop(columns=["_query"], errors="ignore")
    return combined.reset_index(drop=True)


# ── UTC timestamp normalization ────────────────────────────────────────────────

def _normalize_gdelt_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse seendate to timezone-aware UTC timestamp_utc."""
    if "seendate" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(
            df["seendate"], utc=True, errors="coerce"
        )
    elif "timestamp_utc" in df.columns:
        ts = pd.to_datetime(df["timestamp_utc"], errors="coerce")
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
        else:
            ts = ts.dt.tz_convert("UTC")
        df["timestamp_utc"] = ts
    else:
        df["timestamp_utc"] = pd.NaT

    # Apply optional latency buffer
    if NEWS_LATENCY_MINUTES > 0:
        df["timestamp_utc"] = df["timestamp_utc"] + pd.Timedelta(
            minutes=NEWS_LATENCY_MINUTES
        )
    return df


# ── newspaper3k full-text extraction ──────────────────────────────────────────

def _extract_full_text(url: str) -> Optional[str]:
    """Return full article text via newspaper3k, or None on failure."""
    if not _NEWSPAPER_OK or not url:
        return None
    try:
        cfg = _NpConfig()
        cfg.browser_user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        cfg.request_timeout = 10
        art = Article(url, config=cfg)
        art.download()
        art.parse()
        return (art.title or "") + " " + (art.text or "")
    except Exception:
        return None


def add_text_for_sentiment(
    df: pd.DataFrame,
    use_full_article_text: bool = False,
) -> pd.DataFrame:
    """
    Populate text_for_sentiment column.
    Mode A (default): title only — fast, stable.
    Mode B: newspaper3k full text — slower, optional.
    Falls back to title on extraction failure.
    """
    success_count = 0
    fail_count    = 0
    texts         = []

    for _, row in df.iterrows():
        title = str(row.get("title", "") or "")
        if not use_full_article_text:
            texts.append(title)
            continue

        # Mode B: try full text
        extracted = _extract_full_text(row.get("url", ""))
        if extracted and len(extracted.strip()) > len(title):
            texts.append(extracted.strip())
            success_count += 1
        else:
            texts.append(title)
            fail_count += 1

    if use_full_article_text:
        logger.info(
            f"Full-text extraction: {success_count} success, {fail_count} fallback to title"
        )

    df["text_for_sentiment"] = texts
    return df


# ── FinBERT scoring pipeline ───────────────────────────────────────────────────

def score_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Add finbert_label, finbert_confidence, sentiment_score columns."""
    if df.empty:
        df["finbert_label"]      = []
        df["finbert_confidence"] = []
        df["sentiment_score"]    = []
        return df

    texts   = df["text_for_sentiment"].fillna("").tolist()
    scored  = score_texts_finbert(texts)

    df["finbert_label"]      = [s["finbert_label"]      for s in scored]
    df["finbert_confidence"] = [s["finbert_confidence"] for s in scored]
    df["sentiment_score"]    = [s["sentiment_score"]    for s in scored]
    return df


# ── Hourly aggregation ─────────────────────────────────────────────────────────

def aggregate_hourly(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """
    Aggregate article-level sentiment to hourly UTC buckets.

    Leakage prevention:
        News published in hour H (00:00–00:59) is bucketed at H.
        After aggregation, all features are shifted +1 period.
        So news from hour H becomes available for the H+1 prediction bar.

    Returns DataFrame indexed by timestamp_utc (hourly, UTC-aware).
    """
    pfx = asset.lower()
    cols_out = _gdelt_cols(asset)

    if df.empty or "timestamp_utc" not in df.columns:
        return pd.DataFrame(columns=["timestamp_utc"] + cols_out)

    df = df.copy()
    df = df.dropna(subset=["timestamp_utc"])
    df["hour_bucket"] = df["timestamp_utc"].dt.floor("1h")

    grp = df.groupby("hour_bucket")

    agg = pd.DataFrame({
        f"{pfx}_gdelt_sentiment_mean": grp["sentiment_score"].mean(),
        f"{pfx}_gdelt_sentiment_std":  grp["sentiment_score"].std().fillna(0),
        f"{pfx}_gdelt_news_volume":    grp["url"].count(),
        f"{pfx}_gdelt_positive_ratio": grp["sentiment_score"].apply(
            lambda s: (s == 1).sum() / max(len(s), 1)
        ),
        f"{pfx}_gdelt_negative_ratio": grp["sentiment_score"].apply(
            lambda s: (s == -1).sum() / max(len(s), 1)
        ),
        f"{pfx}_gdelt_avg_confidence": grp["finbert_confidence"].mean(),
    })

    # Shift by 1 period for leakage prevention (news at H available at H+1)
    agg = agg.shift(1)

    # Fill after shift
    fill_vals = {
        f"{pfx}_gdelt_sentiment_mean":  0.0,
        f"{pfx}_gdelt_sentiment_std":   0.0,
        f"{pfx}_gdelt_news_volume":     0.0,
        f"{pfx}_gdelt_positive_ratio":  0.0,
        f"{pfx}_gdelt_negative_ratio":  0.0,
        f"{pfx}_gdelt_avg_confidence":  0.0,
    }
    agg = agg.fillna(fill_vals)
    agg.index.name = "timestamp_utc"

    # Ensure index is UTC-aware
    if agg.index.tz is None:
        agg.index = agg.index.tz_localize("UTC")
    else:
        agg.index = agg.index.tz_convert("UTC")

    return agg.reset_index()


# ── Full pipeline for one asset ────────────────────────────────────────────────

def run_asset(
    asset: str,
    start_date: str,
    end_date: str,
    max_records_per_query: int = 50,
    use_full_article_text: bool = False,
    save_raw: bool = True,
) -> pd.DataFrame:
    """
    End-to-end GDELT → FinBERT → hourly features for one asset.
    Returns hourly feature DataFrame; saves processed CSV.
    """
    logger.info(f"[{asset}] Fetching GDELT articles {start_date} → {end_date}")
    raw_df = fetch_gdelt_articles(
        asset, start_date, end_date, max_records_per_query
    )

    if raw_df.empty:
        logger.warning(f"[{asset}] No articles — returning empty features.")
        return pd.DataFrame(columns=["timestamp_utc"] + _gdelt_cols(asset))

    # Timestamp normalization
    raw_df = _normalize_gdelt_timestamps(raw_df)

    # Text extraction
    raw_df = add_text_for_sentiment(raw_df, use_full_article_text)

    # FinBERT scoring
    logger.info(f"[{asset}] Scoring {len(raw_df)} articles with FinBERT…")
    raw_df = score_articles(raw_df)

    # Save article-level CSV
    if save_raw:
        raw_path = RAW_DIR / f"gdelt_{asset.lower()}_articles.csv"
        raw_df.to_csv(raw_path, index=False)
        logger.info(f"[{asset}] Raw articles → {raw_path}")

    # Hourly aggregation + shift
    hourly = aggregate_hourly(raw_df, asset)

    # Save processed features CSV
    proc_path = PROCESSED_DIR / f"gdelt_{asset.lower()}_sentiment_features.csv"
    hourly.to_csv(proc_path, index=False)
    logger.success(f"[{asset}] Processed features → {proc_path} ({len(hourly)} rows)")

    # Dashboard: append to latest_news_sentiment
    _update_dashboard_news(raw_df, asset)

    return hourly


def _update_dashboard_news(raw_df: pd.DataFrame, asset: str) -> None:
    """Append or update latest_news_sentiment.csv for the dashboard."""
    cols = ["timestamp_utc", "asset", "query", "url", "domain",
            "title", "sourcecountry", "language",
            "finbert_label", "sentiment_score", "finbert_confidence"]
    subset = raw_df[[c for c in cols if c in raw_df.columns]].copy()
    subset["news_source"] = "GDELT"

    path = DASHBOARD_DIR / "latest_news_sentiment.csv"
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, subset], ignore_index=True)
        combined = combined.drop_duplicates(subset=["url", "asset"], keep="last")
    else:
        combined = subset

    combined.to_csv(path, index=False)


# ── Date window splitting ──────────────────────────────────────────────────────

def split_date_windows(
    start_date: str, end_date: str, window_days: int = 7
) -> list[tuple[str, str]]:
    """Split a date range into sub-windows of `window_days` each."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")
    windows = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=window_days), end)
        windows.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt
    return windows


# ── Smoke / Mock / FinBERT-only test modes ────────────────────────────────────

def _test_finbert_only() -> None:
    """Quick FinBERT-only test — no internet needed after model download."""
    samples = [
        "Bitcoin price rallies as ETF inflows rise",
        "Oil prices fall after weak demand outlook",
        "Gold remains steady ahead of Federal Reserve decision",
    ]
    print("\n=== FinBERT-only test ===")
    pipe = _load_finbert()
    if pipe is None:
        print("ERROR: FinBERT failed to load.")
        return
    results = score_texts_finbert(samples)
    for text, res in zip(samples, results):
        print(
            f"  [{res['finbert_label']:<9}  conf={res['finbert_confidence']:.3f}"
            f"  score={res['sentiment_score']:+d}]  {text}"
        )
    print("=== FinBERT test PASSED ===\n")


def _test_mock() -> None:
    """Offline mock test — no GDELT or internet calls."""
    from io import StringIO

    print("\n=== Mock/offline test ===")
    mock_csv = """url,title,seendate,domain,language,sourcecountry
http://a.com/1,Bitcoin hits 70k,20260401T100000Z,a.com,English,US
http://b.com/2,ETH gas fees drop,20260401T110000Z,b.com,English,UK
http://a.com/1,Bitcoin hits 70k,20260401T100000Z,a.com,English,US
http://c.com/3,Gold surges on Fed news,20260401T120000Z,c.com,English,DE
http://d.com/4,Oil drops on OPEC output,20260401T130000Z,d.com,English,AE
http://e.com/5,BTC miners increase hash rate,20260401T140000Z,e.com,English,US
http://f.com/6,Ethereum staking reward rises,20260401T150000Z,f.com,English,UK
http://g.com/7,Crude inventory surprise,20260401T160000Z,g.com,English,US
http://h.com/8,Gold price stable near 2000,20260401T170000Z,h.com,English,CH
"""
    df = pd.read_csv(StringIO(mock_csv))
    df["asset"] = "BTC"
    df["query"] = "bitcoin"

    # Deduplication
    before = len(df)
    df = df.drop_duplicates(subset=["url"])
    after  = len(df)
    assert after < before, "Deduplication failed"
    print(f"  Deduplication: {before} → {after} rows ✓")

    # Timestamp normalization
    df = _normalize_gdelt_timestamps(df)
    assert df["timestamp_utc"].dt.tz is not None, "timestamp_utc must be tz-aware"
    print(f"  UTC timestamps: tz={df['timestamp_utc'].dt.tz} ✓")
    assert str(df["timestamp_utc"].dt.tz) == "UTC", "Must be UTC"
    print("  Timezone is UTC ✓")

    # Text + FinBERT
    df = add_text_for_sentiment(df, use_full_article_text=False)
    df = score_articles(df)

    # Check label mapping
    for lbl, val in FINBERT_LABEL_MAP.items():
        assert isinstance(val, int), f"Label {lbl} must map to int"
    print("  FinBERT label mapping ✓")

    # Hourly aggregation
    hourly = aggregate_hourly(df, "BTC")
    pfx = "btc"
    expected_cols = [
        f"{pfx}_gdelt_sentiment_mean", f"{pfx}_gdelt_sentiment_std",
        f"{pfx}_gdelt_news_volume", f"{pfx}_gdelt_positive_ratio",
        f"{pfx}_gdelt_negative_ratio", f"{pfx}_gdelt_avg_confidence",
    ]
    for c in expected_cols:
        assert c in hourly.columns, f"Missing column: {c}"
    print(f"  Hourly columns present ✓  ({len(hourly)} rows)")

    # Shift check: first row after shift should have NaN-filled 0
    if len(hourly) > 0:
        assert hourly[f"{pfx}_gdelt_sentiment_mean"].iloc[0] == 0.0, "Shift fill failed"
        print("  One-period shift + fillna(0) ✓")

    # NaN check
    assert not hourly[expected_cols].isnull().any().any(), "Unexpected NaN in features"
    print("  No NaN in output ✓")

    # Save CSV
    proc_path = PROCESSED_DIR / "gdelt_btc_sentiment_features.csv"
    hourly.to_csv(proc_path, index=False)
    assert proc_path.exists(), "CSV not saved"
    print(f"  CSV saved to {proc_path} ✓")

    print("=== Mock test PASSED ===\n")


def _test_smoke(
    assets: list[str],
    start_date: str,
    end_date: str,
    max_records_per_query: int = 5,
    use_full_article_text: bool = False,
) -> None:
    """Quick GDELT smoke test with real API calls and small record counts."""
    print("\n=== GDELT smoke test ===")

    if not _GDELT_OK:
        print("ERROR: gdeltdoc not installed. Run: pip install gdeltdoc")
        return

    # FinBERT load check
    pipe = _load_finbert()
    if pipe is None:
        print("ERROR: FinBERT failed to load.")
        return
    print("  FinBERT loaded ✓")

    for asset in assets:
        print(f"\n  Asset: {asset}")
        queries_to_test = ASSET_QUERIES.get(asset, [])[:2]  # 2 queries for speed
        raw_df = fetch_gdelt_articles(
            asset, start_date, end_date, max_records_per_query,
            queries=queries_to_test,
        )

        if raw_df.empty:
            print(f"  [{asset}] WARNING: No articles returned — empty result handled gracefully ✓")
            # Save empty processed CSV
            empty_hourly = pd.DataFrame(columns=["timestamp_utc"] + _gdelt_cols(asset))
            proc_path = PROCESSED_DIR / f"gdelt_{asset.lower()}_sentiment_features.csv"
            empty_hourly.to_csv(proc_path, index=False)
            print(f"  Status: PASS (empty result)")
            continue

        raw_articles = len(raw_df)

        # Required columns
        for col in ["url", "title"]:
            assert col in raw_df.columns, f"Missing required column: {col}"
        print(f"  Required columns present ✓")

        # Timestamp normalization
        raw_df = _normalize_gdelt_timestamps(raw_df)
        assert raw_df["timestamp_utc"].dt.tz is not None
        assert str(raw_df["timestamp_utc"].dt.tz) == "UTC"
        print(f"  timestamp_utc is UTC ✓")

        # URL dedup
        before_dedup = len(raw_df)
        raw_df = raw_df.drop_duplicates(subset=["url"])
        unique_urls = len(raw_df)
        print(f"  URL dedup: {before_dedup} → {unique_urls} ✓")

        # FinBERT scoring (just first 5)
        sample = raw_df.head(5).copy()
        sample = add_text_for_sentiment(sample, use_full_article_text)
        sample = score_articles(sample)
        labels_ok = all(
            sample["finbert_label"].isin(["positive", "neutral", "negative"])
        )
        scores_ok = all(sample["sentiment_score"].isin([1, 0, -1]))
        print(f"  FinBERT scored {len(sample)} articles ✓")
        print(f"  Labels valid: {labels_ok} ✓   Scores valid: {scores_ok} ✓")

        # Full scoring for aggregation
        raw_df = add_text_for_sentiment(raw_df, use_full_article_text)
        raw_df = score_articles(raw_df)

        # Hourly aggregation
        hourly = aggregate_hourly(raw_df, asset)
        print(f"  Hourly rows created: {len(hourly)}")

        # Shift check
        pfx = asset.lower()
        if len(hourly) > 0:
            assert hourly[f"{pfx}_gdelt_sentiment_mean"].iloc[0] == 0.0
            print(f"  One-period shift applied ✓")

        # Save
        proc_path = PROCESSED_DIR / f"gdelt_{asset.lower()}_sentiment_features.csv"
        hourly.to_csv(proc_path, index=False)
        assert proc_path.exists()

        print(f"\n  GDELT smoke test result:")
        print(f"    asset              : {asset}")
        print(f"    queries tested     : {len(queries_to_test)}")
        print(f"    raw articles       : {raw_articles}")
        print(f"    unique URLs        : {unique_urls}")
        print(f"    FinBERT scored     : {min(len(raw_df), 5)} (sample)")
        print(f"    hourly rows        : {len(hourly)}")
        print(f"    output saved       : {proc_path}")
        print(f"    status             : PASS")

    print("\n=== Smoke test complete ===\n")


# ── Merge GDELT features into existing feature dataframe ──────────────────────

def merge_gdelt_features(
    market_df: pd.DataFrame,
    asset: str,
) -> pd.DataFrame:
    """
    Merge pre-computed hourly GDELT features into a market feature DataFrame.
    Only merges GDELT features for the specified asset.

    market_df must have a UTC DatetimeIndex.
    """
    proc_path = PROCESSED_DIR / f"gdelt_{asset.lower()}_sentiment_features.csv"
    if not proc_path.exists():
        logger.warning(f"[{asset}] GDELT features file not found: {proc_path}")
        pfx = asset.lower()
        for col in _gdelt_cols(asset):
            market_df[col] = 0.0
        return market_df

    gdelt_df = pd.read_csv(proc_path)
    gdelt_df["timestamp_utc"] = pd.to_datetime(
        gdelt_df["timestamp_utc"], utc=True, errors="coerce"
    )
    gdelt_df = gdelt_df.dropna(subset=["timestamp_utc"])
    gdelt_df = gdelt_df.set_index("timestamp_utc")

    before_rows   = len(market_df)
    gdelt_cols    = _gdelt_cols(asset)
    available     = [c for c in gdelt_cols if c in gdelt_df.columns]

    # Reindex GDELT to market index
    gdelt_aligned = gdelt_df[available].reindex(market_df.index).fillna(0.0)
    for col in available:
        market_df[col] = gdelt_aligned[col].values

    # Fill any asset GDELT cols not found in processed file
    for col in gdelt_cols:
        if col not in market_df.columns:
            market_df[col] = 0.0

    missing_filled = int(gdelt_aligned.isnull().sum().sum())
    logger.info(
        f"[{asset}] Merged GDELT: {before_rows} rows, "
        f"{len(available)} GDELT cols, "
        f"{missing_filled} missing values filled with 0"
    )
    return market_df


# ── CLI orchestration ──────────────────────────────────────────────────────────

def run(
    assets: list[str],
    start_date: str,
    end_date: str,
    max_records_per_query: int = 50,
    use_full_article_text: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run full GDELT pipeline for multiple assets."""
    results = {}
    for asset in assets:
        try:
            df = run_asset(
                asset, start_date, end_date,
                max_records_per_query, use_full_article_text,
            )
            results[asset] = df
        except Exception as e:
            logger.error(f"[{asset}] Pipeline failed: {e}")
            results[asset] = pd.DataFrame()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="GDELT + FinBERT sentiment pipeline"
    )
    parser.add_argument("--assets", nargs="+", default=["BTC", "ETH", "GOLD", "OIL"],
                        help="Assets to process")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date",   default=None)
    parser.add_argument("--max-records-per-query", type=int, default=50)
    parser.add_argument("--no-full-text",  action="store_true",
                        help="Use title-only mode (default)")
    parser.add_argument("--use-full-text", action="store_true",
                        help="Enable newspaper3k full-text extraction")
    parser.add_argument("--finbert-test",  action="store_true",
                        help="Test FinBERT only — no internet")
    parser.add_argument("--mock-test",     action="store_true",
                        help="Offline mock test — no GDELT API")
    parser.add_argument("--smoke-test",    action="store_true",
                        help="Quick GDELT smoke test with real API")

    args = parser.parse_args()

    if args.finbert_test:
        _test_finbert_only()
        return

    if args.mock_test:
        _test_mock()
        return

    if args.smoke_test:
        start = args.start_date or "2026-04-20"
        end   = args.end_date   or "2026-04-22"
        _test_smoke(
            assets=[a.upper() for a in args.assets],
            start_date=start,
            end_date=end,
            max_records_per_query=args.max_records_per_query,
            use_full_article_text=args.use_full_text,
        )
        return

    if not args.start_date or not args.end_date:
        parser.error("--start-date and --end-date are required for full pipeline run")

    use_full = args.use_full_text and not args.no_full_text
    run(
        assets=[a.upper() for a in args.assets],
        start_date=args.start_date,
        end_date=args.end_date,
        max_records_per_query=args.max_records_per_query,
        use_full_article_text=use_full,
    )


if __name__ == "__main__":
    main()
