# Market Predictor

Multi-asset financial market prediction system targeting BTC, ETH, Gold, and WTI Oil.

## Data Sources

| Asset      | Source   | Key needed |
|------------|----------|------------|
| BTC, ETH   | Finnhub  | ✅ Yes     |
| GOLD, OIL  | yfinance | ❌ Free    |
| News       | Finnhub  | ✅ Yes     |

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
copy .env.example .env
# edit .env with your Postgres credentials and Finnhub key

# 3. Create the database
psql -U postgres -c "CREATE DATABASE market_predictor;"
psql -U postgres -d market_predictor -f db/schema.sql

# 4. Test all connections
python test_phase1.py

# 5. Ingest data
python -m ingestion.ingest_prices --days 90
python -m ingestion.ingest_commodities --days 90
```

## Project Structure

```
market-predictor/
├── config.py                    # all settings and env vars
├── requirements.txt
├── test_phase1.py               # run this first
│
├── db/
│   ├── schema.sql               # PostgreSQL table definitions
│   └── connection.py            # connection context manager
│
├── clients/
│   ├── finnhub.py               # BTC, ETH candles + news
│   ├── yfinance_commodities.py  # Gold, Oil candles
│   ├── gdelt.py                 # news stub (Phase 3)
│   └── social.py                # Stocktwits stub (Phase 3)
│
└── ingestion/
    ├── ingest_prices.py         # BTC + ETH → PostgreSQL
    └── ingest_commodities.py    # Gold + Oil → PostgreSQL
```

## Phase Status

| Phase | Description               | Status       |
|-------|---------------------------|--------------|
| 1     | Scope + architecture      | ✅ Done      |
| 2     | Data ingestion            | ✅ Done      |
| 3     | Cleaning + alignment      | 🔜 Next      |
| 4     | Feature engineering       | 🔜 Done      |
| 5     | Target creation           | ✅ Done      |
| 6     | Modeling                  | ✅ Done      |
| 7     | Evaluation + backtest     | ✅ Done      |
| 8     | Explainability (SHAP)     | 🔜 Upcoming  |
| 9     | Dashboard                 | 🔜 Upcoming  |
| 10    | Report + presentation     | 🔜 Upcoming  |
