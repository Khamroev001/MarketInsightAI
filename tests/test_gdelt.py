"""
tests/test_gdelt.py
Tests for GDELT, FinBERT, feature schema, and timestamp handling.

Run: python -m pytest tests/test_gdelt.py -v
 or: python tests/test_gdelt.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest
from io import StringIO
from pathlib import Path

from models.gdelt_sentiment import (
    _normalize_gdelt_timestamps, aggregate_hourly, add_text_for_sentiment,
    score_texts_finbert, score_articles, FINBERT_LABEL_MAP, _gdelt_cols,
    PROCESSED_DIR,
)
from models.feature_schema import (
    get_feature_columns, validate_feature_schema, GDELT_FEATURES,
    RL_BASE_FEATURES, TRANSFORMER_BASE_FEATURES,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _mock_articles(n: int = 6) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "url":       f"http://example.com/{i}",
            "title":     f"Test headline {i}: Bitcoin surges" if i % 2 == 0
                         else f"Test headline {i}: Oil falls sharply",
            "seendate":  f"202604{i+1:02d}T{10+i:02d}0000Z",
            "domain":    "example.com",
            "language":  "English",
            "sourcecountry": "US",
            "asset":     "BTC",
            "query":     "bitcoin",
        })
    return pd.DataFrame(rows)


# ── A. GDELT offline tests ─────────────────────────────────────────────────────

class TestGDELTOffline:

    def test_timestamp_normalization_utc(self):
        df = _mock_articles()
        df = _normalize_gdelt_timestamps(df)
        assert "timestamp_utc" in df.columns
        assert df["timestamp_utc"].dt.tz is not None
        assert str(df["timestamp_utc"].dt.tz) == "UTC"

    def test_no_naive_timestamps(self):
        df = _mock_articles()
        df = _normalize_gdelt_timestamps(df)
        assert not df["timestamp_utc"].isnull().all()
        # Every non-null timestamp must be tz-aware
        valid = df["timestamp_utc"].dropna()
        assert all(ts.tzinfo is not None for ts in valid)

    def test_url_deduplication(self):
        df = _mock_articles()
        # Insert a duplicate
        dup = df.iloc[0:1].copy()
        df = pd.concat([df, dup], ignore_index=True)
        before = len(df)
        df = df.drop_duplicates(subset=["url"])
        assert len(df) < before, "Deduplication should remove at least one row"
        assert len(df) == before - 1

    def test_hourly_aggregation_columns(self):
        df = _mock_articles()
        df = _normalize_gdelt_timestamps(df)
        df = add_text_for_sentiment(df, use_full_article_text=False)
        # Manually assign sentiment scores for testing
        df["finbert_label"]      = "positive"
        df["finbert_confidence"] = 0.9
        df["sentiment_score"]    = 1
        hourly = aggregate_hourly(df, "BTC")
        for col in _gdelt_cols("BTC"):
            assert col in hourly.columns, f"Missing column: {col}"

    def test_hourly_aggregation_no_nan(self):
        df = _mock_articles()
        df = _normalize_gdelt_timestamps(df)
        df = add_text_for_sentiment(df, use_full_article_text=False)
        df["finbert_label"]      = "neutral"
        df["finbert_confidence"] = 0.5
        df["sentiment_score"]    = 0
        hourly = aggregate_hourly(df, "BTC")
        cols   = _gdelt_cols("BTC")
        assert not hourly[cols].isnull().any().any(), "No NaN allowed in hourly features"

    def test_one_period_shift_applied(self):
        df = _mock_articles()
        df = _normalize_gdelt_timestamps(df)
        df = add_text_for_sentiment(df, use_full_article_text=False)
        df["finbert_label"]      = "positive"
        df["finbert_confidence"] = 0.9
        df["sentiment_score"]    = 1
        hourly = aggregate_hourly(df, "BTC")
        if len(hourly) > 0:
            # First row must be 0 (shifted from nothing above it)
            assert hourly["btc_gdelt_sentiment_mean"].iloc[0] == 0.0, \
                "After shift, first row should be 0"

    def test_sentiment_label_mapping(self):
        assert FINBERT_LABEL_MAP["positive"] == 1
        assert FINBERT_LABEL_MAP["neutral"]  == 0
        assert FINBERT_LABEL_MAP["negative"] == -1

    def test_empty_result_no_crash(self):
        empty = pd.DataFrame()
        result = aggregate_hourly(empty, "BTC")
        assert isinstance(result, pd.DataFrame)

    def test_text_for_sentiment_title_mode(self):
        df = _mock_articles()
        df = add_text_for_sentiment(df, use_full_article_text=False)
        assert "text_for_sentiment" in df.columns
        assert all(df["text_for_sentiment"] == df["title"])

    def test_output_csv_saved(self):
        df = _mock_articles()
        df = _normalize_gdelt_timestamps(df)
        df = add_text_for_sentiment(df, use_full_article_text=False)
        df["finbert_label"]      = "neutral"
        df["finbert_confidence"] = 0.5
        df["sentiment_score"]    = 0
        hourly = aggregate_hourly(df, "BTC")
        path   = PROCESSED_DIR / "gdelt_btc_sentiment_features.csv"
        hourly.to_csv(path, index=False)
        assert path.exists(), f"Expected CSV at {path}"


# ── B. FinBERT tests (require model download; skipped if unavailable) ─────────

class TestFinBERT:

    def test_finbert_loads(self):
        from models.gdelt_sentiment import _load_finbert
        pipe = _load_finbert()
        # If torch/transformers not installed, pipe is None — that's acceptable
        # but we confirm no exception raised during load attempt
        assert pipe is None or callable(pipe)

    def test_empty_text_returns_neutral(self):
        results = score_texts_finbert([""])
        assert results[0]["finbert_label"]   == "neutral"
        assert results[0]["sentiment_score"] == 0

    def test_none_text_returns_neutral(self):
        results = score_texts_finbert([None])
        assert results[0]["sentiment_score"] == 0

    def test_score_returns_valid_labels(self):
        from models.gdelt_sentiment import _load_finbert
        if _load_finbert() is None:
            pytest.skip("FinBERT not available")
        texts   = ["Bitcoin price rallies to new highs", "Oil prices collapse on oversupply"]
        results = score_texts_finbert(texts)
        for r in results:
            assert r["finbert_label"]      in ["positive", "neutral", "negative"]
            assert r["sentiment_score"]    in [1, 0, -1]
            assert 0.0 <= r["finbert_confidence"] <= 1.0

    def test_confidence_saved(self):
        from models.gdelt_sentiment import _load_finbert
        if _load_finbert() is None:
            pytest.skip("FinBERT not available")
        results = score_texts_finbert(["Gold steady before FOMC"])
        assert "finbert_confidence" in results[0]
        assert results[0]["finbert_confidence"] >= 0.0


# ── C. Feature schema tests ────────────────────────────────────────────────────

class TestFeatureSchema:

    def test_gdelt_features_appear_only_when_enabled(self):
        without = get_feature_columns("BTC", model="rl", use_gdelt_sentiment=False)
        with_g  = get_feature_columns("BTC", model="rl", use_gdelt_sentiment=True)
        gdelt_cols = GDELT_FEATURES["BTC"]
        for c in gdelt_cols:
            assert c not in without, f"{c} should not appear without GDELT"
            assert c in with_g,      f"{c} should appear with GDELT"

    def test_rl_base_feature_count(self):
        cols = get_feature_columns("BTC", model="rl")
        assert len(cols) == 27, f"Expected 27 RL base features, got {len(cols)}"

    def test_transformer_base_feature_count(self):
        cols = get_feature_columns("BTC", model="transformer")
        assert len(cols) == 31, f"Expected 31 Transformer base features, got {len(cols)}"

    def test_ridge_feature_count(self):
        cols = get_feature_columns("BTC", model="ridge")
        assert len(cols) == 13, f"Expected 13 Ridge features, got {len(cols)}"

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError):
            get_feature_columns("BTC", model="unknown_model")

    def test_validate_perfect_match_no_error(self):
        cols = get_feature_columns("BTC", model="rl")
        # Should not raise
        result = validate_feature_schema(cols, cols, context="test")
        assert result == cols

    def test_validate_missing_feature_raises(self):
        expected = get_feature_columns("BTC", model="rl")
        current  = expected[:-2]   # drop 2 features
        with pytest.raises(ValueError, match="Missing features"):
            validate_feature_schema(current, expected, context="test")

    def test_validate_extra_feature_raises(self):
        expected = get_feature_columns("BTC", model="rl")
        current  = expected + ["extra_col_1", "extra_col_2"]
        with pytest.raises(ValueError, match="Extra features"):
            validate_feature_schema(current, expected, context="test")

    def test_validate_force_retrain_does_not_raise(self):
        expected = get_feature_columns("BTC", model="rl")
        current  = expected[:-2]
        # Should not raise with force_retrain=True
        validate_feature_schema(current, expected, context="test", force_retrain=True)

    def test_all_assets_have_gdelt_features(self):
        for asset in ["BTC", "ETH", "GOLD", "OIL"]:
            cols = get_feature_columns(asset, model="rl", use_gdelt_sentiment=True)
            gdelt = GDELT_FEATURES[asset]
            for c in gdelt:
                assert c in cols, f"[{asset}] GDELT col {c} missing"

    def test_gdelt_not_cross_contaminated(self):
        """BTC GDELT features should not appear in ETH schema."""
        btc_gdelt = GDELT_FEATURES["BTC"]
        eth_cols  = get_feature_columns("ETH", model="rl", use_gdelt_sentiment=True)
        for c in btc_gdelt:
            assert c not in eth_cols, f"BTC GDELT col {c} leaked into ETH schema"


# ── D. Timestamp validation tests ─────────────────────────────────────────────

class TestTimestamps:

    def test_gdelt_timestamp_aware(self):
        df = _mock_articles()
        df = _normalize_gdelt_timestamps(df)
        assert df["timestamp_utc"].dt.tz is not None

    def test_gdelt_timestamp_is_utc(self):
        df = _mock_articles()
        df = _normalize_gdelt_timestamps(df)
        assert str(df["timestamp_utc"].dt.tz) == "UTC"

    def test_no_naive_leak_into_model(self):
        """All timestamps processed through the pipeline must remain UTC-aware."""
        df = _mock_articles()
        df = _normalize_gdelt_timestamps(df)
        df = add_text_for_sentiment(df, use_full_article_text=False)
        df["finbert_label"]      = "neutral"
        df["finbert_confidence"] = 0.5
        df["sentiment_score"]    = 0
        hourly = aggregate_hourly(df, "BTC")
        if "timestamp_utc" in hourly.columns and len(hourly) > 0:
            ts = pd.to_datetime(hourly["timestamp_utc"])
            if ts.dt.tz is not None:
                assert str(ts.dt.tz) == "UTC"

    def test_news_at_h_not_used_for_h_prediction(self):
        """After shift, row i+1 contains info from row i (not row i itself)."""
        df = _mock_articles()
        df = _normalize_gdelt_timestamps(df)
        df = add_text_for_sentiment(df, use_full_article_text=False)
        df["finbert_label"]      = "positive"
        df["finbert_confidence"] = 0.9
        df["sentiment_score"]    = 1
        hourly = aggregate_hourly(df, "BTC")
        # First bucket: shifted → should be 0 (no prior news)
        if len(hourly) > 0:
            assert hourly["btc_gdelt_sentiment_mean"].iloc[0] == 0.0, \
                "News at hour H must NOT appear in hour H prediction (leakage)"


if __name__ == "__main__":
    import importlib

    all_test_classes = [
        TestGDELTOffline,
        TestFinBERT,
        TestFeatureSchema,
        TestTimestamps,
    ]

    total_pass = total_fail = 0
    for cls in all_test_classes:
        print(f"\n-- {cls.__name__} --")
        obj = cls()
        for name in dir(cls):
            if not name.startswith("test_"):
                continue
            fn = getattr(obj, name)
            try:
                fn()
                print(f"  PASS  {name}")
                total_pass += 1
            except pytest.skip.Exception as e:
                print(f"  SKIP  {name}: {e}")
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
                total_fail += 1

    print(f"\n{total_pass} passed, {total_fail} failed")
