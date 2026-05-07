-- ─────────────────────────────────────────────────────────────────────────────
-- market_predictor schema
-- Run once: psql -U <user> -d market_predictor -f schema.sql
-- ─────────────────────────────────────────────────────────────────────────────

-- Remove unused / superseded tables
DROP TABLE IF EXISTS raw_social;
DROP TABLE IF EXISTS clean_price_bars;

-- Raw OHLCV bars (crypto + commodities)
CREATE TABLE IF NOT EXISTS raw_price_bars (
    id          BIGSERIAL PRIMARY KEY,
    asset       VARCHAR(10)   NOT NULL,           -- BTC, ETH, GOLD, OIL
    source      VARCHAR(30)   NOT NULL,           -- coingecko | alpha_vantage
    ts          TIMESTAMPTZ   NOT NULL,
    open        NUMERIC(20,8),
    high        NUMERIC(20,8),
    low         NUMERIC(20,8),
    close       NUMERIC(20,8) NOT NULL,
    volume      NUMERIC(30,8),
    inserted_at TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    UNIQUE (asset, ts)
);

CREATE INDEX IF NOT EXISTS idx_price_bars_asset_ts ON raw_price_bars (asset, ts DESC);

-- Raw news articles
CREATE TABLE IF NOT EXISTS raw_news (
    id           BIGSERIAL PRIMARY KEY,
    source       VARCHAR(30)  NOT NULL,           -- finnhub | gdelt | ...
    asset_tags   TEXT[],                          -- ['BTC','ETH'] etc.
    published_at TIMESTAMPTZ  NOT NULL,
    title        TEXT,
    url          TEXT         UNIQUE,
    raw_json     JSONB,
    inserted_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_news_published ON raw_news (published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_tags      ON raw_news USING GIN (asset_tags);

-- Feature matrix (output of pipeline/features.py)
-- Reads from raw_price_bars; no clean_price_bars intermediate table.
-- Regime is encoded as 4 one-hot boolean columns (not a single integer).
-- Macro signals (dxy, vix, spy) are fetched daily and forward-filled.
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
    dxy              NUMERIC(12,6),  -- US Dollar Index (daily, forward-filled)
    vix              NUMERIC(12,6),  -- CBOE Volatility Index (daily, forward-filled)
    spy              NUMERIC(12,6),  -- S&P 500 ETF close (daily, forward-filled)
    regime_low_vol   BOOLEAN,        -- one-hot regime: rolling std < p25
    regime_bull      BOOLEAN,        -- one-hot regime: mean > 0 AND std < p75
    regime_bear      BOOLEAN,        -- one-hot regime: mean < 0 AND std < p75
    regime_high_vol  BOOLEAN,        -- one-hot regime: rolling std >= p75
    inserted_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (asset, ts)
);

CREATE INDEX IF NOT EXISTS idx_features_asset_ts ON features (asset, ts DESC);

-- Prediction targets (output of pipeline/targets.py)
CREATE TABLE IF NOT EXISTS targets (
    id           BIGSERIAL    PRIMARY KEY,
    asset        VARCHAR(10)  NOT NULL,
    ts           TIMESTAMPTZ  NOT NULL,
    ret_1h       NUMERIC(12,6),
    ret_4h       NUMERIC(12,6),
    direction_1h SMALLINT,                        -- 1=up, 0=neutral, -1=down
    direction_4h SMALLINT,
    inserted_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (asset, ts)
);

CREATE INDEX IF NOT EXISTS idx_targets_asset_ts ON targets (asset, ts DESC);

-- Transformer classifier per-bar predictions (output of models/transformer_model.py)
-- Read by the RL agent to enrich its state vector (p_down, p_neutral, p_up, confidence).
CREATE TABLE IF NOT EXISTS transformer_predictions (
    id          BIGSERIAL PRIMARY KEY,
    asset       TEXT          NOT NULL,
    ts          TIMESTAMPTZ   NOT NULL,
    direction   INT           NOT NULL,   -- predicted class: -1, 0, or 1
    confidence  FLOAT         NOT NULL,   -- max softmax probability [0, 1]
    p_down      FLOAT,                    -- P(direction == -1)
    p_neutral   FLOAT,                    -- P(direction ==  0)
    p_up        FLOAT,                    -- P(direction == +1)
    inserted_at TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    UNIQUE (asset, ts)
);

CREATE INDEX IF NOT EXISTS idx_transformer_pred_asset_ts
    ON transformer_predictions (asset, ts DESC);

-- LLM alpha signal cache (avoids re-calling Claude for the same bar)
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

-- Ingestion log (track every run)
CREATE TABLE IF NOT EXISTS ingestion_log (
    id          BIGSERIAL PRIMARY KEY,
    source      VARCHAR(30)   NOT NULL,
    asset       VARCHAR(10),
    status      VARCHAR(10)   NOT NULL,           -- success | error
    rows_saved  INTEGER       DEFAULT 0,
    error_msg   TEXT,
    ran_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);
