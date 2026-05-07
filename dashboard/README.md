# MarketInsight AI

**Adaptive Multi-Asset Financial Market Prediction and PPO Trading Dashboard**

MarketInsight AI is an end-to-end research project for forecasting and evaluating short-horizon market behavior across four assets:

- **BTC** — Bitcoin
- **ETH** — Ethereum
- **GOLD** — Gold futures proxy
- **OIL** — Crude oil futures proxy

The system combines market data, engineered technical features, GDELT/FinBERT sentiment features, baseline regression models, a Transformer return forecaster, a PPO reinforcement-learning trading agent, and a Streamlit dashboard for model diagnostics.

> **Important note:** This is a data science research project, not a production trading system. The strongest result is asset-specific: BTC shows the clearest profitable PPO behavior, while ETH/GOLD/OIL require stricter deployment gates and further tuning.

---

## 1. Project Overview

The project studies whether weak short-horizon forecasting signals can be converted into useful trading decisions.

The pipeline is organized as:

```text
1. Ingest hourly OHLCV price data
2. Add technical, macro, cross-asset, and sentiment features
3. Build forward-return targets
4. Train Ridge baseline models
5. Train Transformer return forecasting models
6. Feed forecasts into a PPO trading agent
7. Export results to a Streamlit dashboard
```

The dashboard includes:

- Executive overview
- Market monitor
- GDELT + FinBERT news sentiment
- Transformer forecasts
- Prediction accuracy
- PPO trading agent analysis
- Model comparison
- Pipeline health checks

---

## 2. Repository Structure

```text
MarketInsightAI/
│
├── clients/                  # External data/API clients
├── dashboard/                # Streamlit dashboard and chart utilities
│   ├── app.py                # Main dashboard app
│   ├── charts.py             # Plotly chart functions
│   └── data_loader.py        # CSV/PostgreSQL dashboard data loader
│
├── data/
│   ├── raw/                  # Raw GDELT article CSVs
│   ├── processed/            # Processed GDELT sentiment feature CSVs
│   └── dashboard/            # Exported dashboard CSVs and PPO results
│
├── db/                       # Database helpers/schema
├── ingestion/                # Price/news/sentiment ingestion modules
├── models/                   # Ridge, Transformer, PPO/RL models
│   ├── feature_schema.py     # Central feature schema
│   ├── ridge_baseline.py
│   ├── transformer_model.py
│   ├── rl_env.py
│   └── rl_agent.py
│
├── pipeline/                 # Feature engineering and target construction
├── tests/                    # Unit tests
├── config.py                 # Environment variable configuration
├── run_pipeline.py           # Main pipeline runner
├── requirements.txt
└── README.md
```

---

## 3. Quick Start for Grading / Dashboard Demo

This is the easiest way to inspect the project outputs without retraining all models.

### Step 1 — Clone the repository

```bash
git clone https://github.com/Khamroev001/MarketInsightAI.git
cd MarketInsightAI
```

### Step 2 — Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Streamlit is missing, install the dashboard dependencies directly:

```bash
pip install streamlit plotly pandas numpy scikit-learn torch stable-baselines3 gymnasium python-dotenv loguru yfinance transformers
```

### 4. — Run the dashboard

```bash
python -m streamlit run dashboard/app.py
```

Then open the browser link printed by Streamlit, usually:

```text
http://localhost:8501
```

The dashboard should load from the exported CSV files in:

```text
data/dashboard/
```

No retraining is required for basic grading/demo inspection if the exported CSVs are present.

---

## 5. Running the Full Pipeline

The full pipeline can be run with:

```bash
python run_pipeline.py --use-gdelt-sentiment
```

Common development commands:

```bash
# Run pipeline without RL
python run_pipeline.py --use-gdelt-sentiment --no-rl

# Start from feature engineering / later steps depending on CLI support
python run_pipeline.py --use-gdelt-sentiment --from 3 --no-rl

# Run model stages and dashboard export after data exists
python run_pipeline.py --use-gdelt-sentiment
```

If the local database/API setup is not available, use the already exported dashboard CSVs in `data/dashboard/` and run only the dashboard.

---

## 6. Running PPO / RL Experiments

The PPO agent is run per asset. The main experiment configuration uses:

```text
target_horizon = 4
target_mode    = vol_norm
```

Example commands:

```bash
python -m models.rl_agent --asset BTC  --target-horizon 4 --target-mode vol_norm --retrain
python -m models.rl_agent --asset ETH  --target-horizon 4 --target-mode vol_norm --retrain
python -m models.rl_agent --asset GOLD --target-horizon 4 --target-mode vol_norm --retrain
python -m models.rl_agent --asset OIL  --target-horizon 4 --target-mode vol_norm --retrain
```

To avoid overwriting dashboard results, use an output suffix:

```bash
python -m models.rl_agent --asset BTC  --target-horizon 4 --target-mode vol_norm --retrain --output-suffix rl_test_v3
python -m models.rl_agent --asset ETH  --target-horizon 4 --target-mode vol_norm --retrain --output-suffix rl_test_v3
python -m models.rl_agent --asset GOLD --target-horizon 4 --target-mode vol_norm --retrain --output-suffix rl_test_v3
python -m models.rl_agent --asset OIL  --target-horizon 4 --target-mode vol_norm --retrain --output-suffix rl_test_v3
```

This saves outputs into:

```text
data/dashboard/rl_test_v3/
```

instead of overwriting the main dashboard files.

---

## 7. Expected Dashboard Results

The project should be evaluated primarily through completed-trade PPO metrics, because completed trade history gives the clearest realized PnL after transaction costs.

### Final PPO Interpretation

The final project conclusion is asset-specific:

| Asset | Interpretation |
|---|---|
| BTC | Strongest PPO result; deployable candidate in this experiment |
| ETH | Weak / watchlist; small loss after costs |
| GOLD | Original PPO failed, but v3 filtering reduced loss substantially |
| OIL | Original PPO failed, but v3 filtering brought result close to breakeven |

### Expected PPO Results Used in the Final Discussion

| Asset | Selected Run | Completed Trades | Net PnL | Return | Win Rate | Profit Factor | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| BTC | Original strong PPO | 13 | +$2,343.38 | +23.43% | 61.5% | 8.61 | Best result |
| ETH | Original PPO | 23 | -$232.72 | -2.33% | 21.7% | 0.72 | Watchlist / weak |
| GOLD | v3 filtered PPO | 71 | about -$903 | about -9.03% | 33.8% | 0.73 | Improved but not profitable |
| OIL | v3 filtered PPO | 51 | about -$158 | about -1.58% | 39.2% | 0.68 | Near breakeven, best weak-asset improvement |

The key conclusion is not that the system is universally profitable. The correct conclusion is:

```text
PPO can convert weak Transformer signals into useful decisions for selected assets,
but the strategy does not generalize equally across all markets. A professional
version should use asset-specific deployment gates and reject assets where realized
trade PnL, profit factor, or drawdown are not acceptable.
```

---

## 8. Transformer / Forecasting Results

The Transformer predicts short-horizon future log returns. The target design uses volatility-normalized 4-hour returns to reduce near-zero prediction collapse:

```text
future_log_return = log(P[t+4]) - log(P[t])
vol_norm_target   = future_log_return / rolling_volatility_20
```

The Transformer should be interpreted as a noisy signal generator, not as a standalone price predictor.

Dashboard metrics to inspect:

- MAE
- RMSE
- R²
- Pearson correlation
- Direction accuracy
- Prediction standard-deviation ratio

Expected qualitative result:

```text
MAE is small because hourly returns are naturally small.
R² and correlation are weak, which is expected in short-horizon financial prediction.
Directional accuracy and signal strength are more useful for trading analysis than price-level prediction.
```

---

## 9. Running Tests

Run the RL/unit tests with:

```bash
python -m pytest tests/test_rl_env.py
```

Run all tests:

```bash
python -m pytest tests
```

A recent PPO v3 test run passed all 39 RL environment tests before the final v3 experiments.

---

## 10. Dashboard Troubleshooting

### Problem: dashboard still shows old PPO results

Clear Streamlit cache and hard refresh the browser:

```bash
python -m streamlit cache clear
python -m streamlit run dashboard/app.py
```

Then press:

```text
Ctrl + F5
```

### Problem: PPO tab says “No step data”

The dashboard may need the generic combined files:

```text
data/dashboard/latest_rl_trades.csv
data/dashboard/latest_rl_trade_history.csv
```

If only per-asset files exist, rebuild the generic files by concatenating:

```text
latest_rl_trades_btc_h4_volnorm.csv
latest_rl_trades_eth_h4_volnorm.csv
latest_rl_trades_gold_h4_volnorm.csv
latest_rl_trades_oil_h4_volnorm.csv
```

and the corresponding `latest_rl_trade_history_*` files.

### Problem: charts appear in the wrong dashboard tab

This was caused by a Streamlit indentation/container issue in `dashboard/app.py`. The “Per-Asset Metric Charts” block must remain inside the Prediction Accuracy tab:

```python
with tabs[4]:
    st.subheader("Model Evaluation — Transformer Regression")
    ...
    st.markdown("#### Per-Asset Metric Charts")
```

Do not place tab-specific chart blocks at top level.

---

## 11. Reproducibility Notes

PPO training is stochastic. Results can vary across random seeds, retraining runs, and small environment changes. Therefore:

- Dashboard CSVs are used as the submitted/reference outputs.
- Completed-trade net PnL is the primary PPO metric.
- Step-level portfolio curves are useful for visualization but should be reconciled with completed trade history.
- Profit factor and realized net PnL matter more than win rate alone.

A win rate below 50% is not automatically bad if average winners are much larger than average losers. In this project, however, several non-BTC experiments had profit factors below 1, meaning the low win rates were not fully compensated by larger winning trades.

---

## 12. Recommended Grading Path

For the fastest review:

```bash
git clone https://github.com/Khamroev001/MarketInsightAI.git
cd MarketInsightAI
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows PowerShell
pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

Then inspect these dashboard tabs:

1. **Executive Overview** — project-level snapshot
2. **Transformer Forecasts** — forecast timeline and signal strength
3. **Prediction Accuracy** — MAE/RMSE/R²/correlation/direction accuracy
4. **PPO Trading Agent** — realized trade results and trade history
5. **Model Comparison** — Ridge vs Transformer summary
6. **Pipeline Health** — data/model availability checks

---

## 13. Final Research Conclusion

MarketInsight AI demonstrates a full forecasting-to-decision pipeline. The system successfully integrates market data, engineered features, GDELT/FinBERT sentiment, Transformer forecasts, and PPO trading evaluation in one dashboard.

The best result is asset-specific rather than universal. BTC produced the strongest PPO result, while ETH, GOLD, and OIL show that weak predictive signals and high transaction costs can quickly reduce profitability. The v3 filtered PPO experiment improved GOLD and OIL losses substantially, but did not make the system consistently profitable across all assets.

The professional conclusion is:

```text
MarketInsight AI is a transparent research system for testing financial forecasting
and RL-based trading decisions. It is not a production trading bot. Future work should
focus on supervised trade-quality filtering, stronger walk-forward validation,
multiple random seeds, and asset-specific deployment gates.
```
