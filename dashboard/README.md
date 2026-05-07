# MarketInsight AI — Dashboard

Professional Streamlit + Plotly dashboard for the MarketInsight AI final report demo.

## Install dependencies

```bash
pip install streamlit plotly pandas psycopg2-binary python-dotenv loguru
```

## Export dashboard data (recommended first step)

Pulls data from PostgreSQL and writes dashboard-ready CSVs to `data/dashboard/`:

```bash
python -m dashboard.export_dashboard_data
```

With GDELT sentiment attached to predictions:

```bash
python -m dashboard.export_dashboard_data --use-gdelt-sentiment
```

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`

---

## Data sources (priority order)

| CSV | Fallback | Description |
|-----|----------|-------------|
| `data/dashboard/latest_prices.csv` | PostgreSQL `raw_price_bars` | OHLCV bars |
| `data/dashboard/latest_news_sentiment.csv` | `data/raw/gdelt_*_articles.csv` | GDELT + FinBERT |
| `data/dashboard/latest_predictions.csv` | PostgreSQL `transformer_predictions` | Forecasts |
| `data/dashboard/latest_rl_trades.csv` | — | PPO trade log |
| `data/dashboard/model_metrics.csv` | Hard-coded fallback values | Eval metrics |
| `data/dashboard/feature_status.csv` | Filesystem check | Pipeline health |
| `data/dashboard/model_comparison.csv` | model_metrics.csv | Model comparison |

If a CSV is missing the dashboard shows a warning and continues.

---

## Tabs

| Tab | Contents |
|-----|----------|
| Executive Overview | Price cards, summary table, pipeline status |
| Market Monitor | Candlestick, normalised price, returns, volatility, correlation heatmap |
| News & Sentiment | News feed, sentiment timeline, volume, heatmap, top domains, choropleth |
| Transformer Forecasts | Predicted vs actual return, scatter, signal strength, direction chart, price markers |
| Prediction Accuracy | MAE/RMSE/R²/Corr/DirAcc bars, error histogram, rolling MAE, scatter |
| PPO Trading Agent | Portfolio curve, drawdown, action distribution, reward components |
| Model Comparison | Multi-model table, grouped bars, SHAP images |
| Pipeline Health | Checklist, file status, row counts, news coverage |

---

## Fallback metrics (Transformer Regressor)

These are embedded as defaults when `model_metrics.csv` is absent:

| Asset | MAE | RMSE | R² | Corr | DirAcc | DerivedAcc | DerivedF1 |
|-------|-----|------|-----|------|--------|------------|-----------|
| BTC   | 0.001218 | 0.002886 | 0.000 | 0.021 | 0.174 | 0.806 | 0.298 |
| ETH   | 0.001617 | 0.003780 | -0.002 | 0.029 | 0.828 | 0.782 | 0.293 |
| GOLD  | 0.000704 | 0.002404 | 0.000 | 0.008 | 0.753 | 0.893 | 0.315 |
| OIL   | 0.001532 | 0.005747 | 0.000 | -0.001 | 0.878 | 0.847 | 0.306 |

> Low MAE/RMSE should be interpreted carefully because hourly log returns are naturally small.
> R² and correlation near zero indicate limited explanatory power for short-horizon returns.

---

## Sidebar controls

- **Asset**: BTC / ETH / GOLD / OIL / All
- **Date range**: 24H / 7D / 30D / 90D / All
- **Model**: Transformer / Ridge / PPO/RL / All
- **Refresh**: clears Streamlit cache and reloads
- **Show raw tables**: reveals debug DataFrames in each tab

---

## Generate missing backend data

```bash
# Full pipeline (includes GDELT sentiment)
python run_pipeline.py --use-gdelt-sentiment

# Only RL training
python run_pipeline.py --use-gdelt-sentiment --only 7 --retrain rl

# Re-export after pipeline run
python -m dashboard.export_dashboard_data --use-gdelt-sentiment
```
