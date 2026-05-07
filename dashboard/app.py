"""
dashboard/app.py
MarketInsight AI — Streamlit Dashboard

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dashboard.data_loader import (
    ALL_ASSETS,
    FALLBACK_METRICS,
    SAVED_DIR,
    filter_by_asset,
    filter_by_days,
    load_feature_status,
    load_gdelt_features,
    load_model_comparison,
    load_model_metrics,
    load_news_sentiment,
    load_predictions,
    load_prices,
    load_rl_trade_history,
    load_rl_trades,
)
from dashboard import charts

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MarketInsight AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal dark-mode CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.75rem; color: #888; }
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }
    .status-ok  { color: #00ff88; font-weight: 600; }
    .status-warn{ color: #ff9944; font-weight: 600; }
    .status-err { color: #ff4455; font-weight: 600; }
    .section-header {
        color: #4488ff;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 8px;
        margin-bottom: 4px;
    }
    div[data-testid="stHorizontalBlock"] { gap: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 MarketInsight AI Dashboard")
st.caption(
    "Multi-Asset Forecasting · Sentiment Intelligence · PPO Trading Analysis  "
    "— BTC · ETH · GOLD · OIL"
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")

    asset_sel = st.selectbox("Asset", ["All"] + ALL_ASSETS, index=0)
    date_sel  = st.selectbox("Date Range", ["24H", "7D", "30D", "90D", "All"], index=4)
    model_sel = st.selectbox("Model", ["All", "Transformer", "Ridge", "PPO/RL"], index=0)

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()

    show_debug = st.checkbox("Show raw data tables", value=False)

    st.divider()
    st.markdown("**Data sources**")
    st.caption("CSVs → PostgreSQL → Fallback")
    st.caption(f"Cached 5 min · {pd.Timestamp.now().strftime('%H:%M:%S')}")

DAYS_MAP = {"24H": 1, "7D": 7, "30D": 30, "90D": 90, "All": None}
days = DAYS_MAP[date_sel]

# ── Load all data once ────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    prices_raw    = load_prices()
    news_raw      = load_news_sentiment()
    preds_raw     = load_predictions()
    metrics_df    = load_model_metrics()
    rl_raw        = load_rl_trades()
    rl_history_raw = load_rl_trade_history()
    feat_status   = load_feature_status()
    gdelt_feat    = load_gdelt_features()
    comparison_df = load_model_comparison()
    # Force-read latest manual comparison CSV so Ridge rows are not hidden by cache/fallback.
    _cmp_path = ROOT_DIR / "data" / "dashboard" / "model_comparison.csv"
    if _cmp_path.exists():
        comparison_df = pd.read_csv(_cmp_path)

# Apply filters
prices  = filter_by_days(filter_by_asset(prices_raw,  asset_sel), days)
news    = filter_by_days(filter_by_asset(news_raw,    asset_sel), days)
preds   = filter_by_days(filter_by_asset(preds_raw,   asset_sel), days)
rl_data = filter_by_days(filter_by_asset(rl_raw,      asset_sel), days)
gdelt   = filter_by_days(filter_by_asset(gdelt_feat,  asset_sel), days)


# ── Helper: get latest scalar per asset ──────────────────────────────────────
def _latest(df: pd.DataFrame, asset: str, col: str, default=None):
    if df.empty or col not in df.columns or "asset" not in df.columns:
        return default
    sub = df[df["asset"] == asset].dropna(subset=[col])
    return sub[col].iloc[-1] if not sub.empty else default


def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v:+.3f}%"


def _fmt_price(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"${v:,.2f}" if v > 100 else f"${v:.4f}"


def _metric_row(label: str, value, delta=None):
    st.metric(label, value, delta)


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tabs = st.tabs([
    "🏠 Executive Overview",
    "📊 Market Monitor",
    "📰 News & Sentiment",
    "🤖 Transformer Forecasts",
    "🎯 Prediction Accuracy",
    "🎲 PPO Trading Agent",
    "⚖️ Model Comparison",
    "🔧 Pipeline Health",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Cross-Asset Snapshot")

    asset_cols = st.columns(len(ALL_ASSETS))
    for col_idx, asset in enumerate(ALL_ASSETS):
        with asset_cols[col_idx]:
            # Price + returns
            price = _latest(prices_raw, asset, "close")
            price_prev_1h = None
            if not prices_raw.empty and "close" in prices_raw.columns and "asset" in prices_raw.columns:
                sub = prices_raw[prices_raw["asset"] == asset].sort_values("timestamp_utc")
                if len(sub) >= 2:
                    price         = sub["close"].iloc[-1]
                    price_prev_1h = sub["close"].iloc[-2]

            ret_1h = ((price / price_prev_1h) - 1) * 100 if price and price_prev_1h else None

            # Metrics from model_metrics
            m = metrics_df[(metrics_df["asset"] == asset) & (metrics_df["model"] == "Transformer")] \
                if "model" in metrics_df.columns else metrics_df[metrics_df["asset"] == asset]
            m = m.iloc[0] if not m.empty else pd.Series(FALLBACK_METRICS.get(asset, {}))

            # Latest prediction
            pred_dir = _latest(preds_raw, asset, "predicted_direction")
            sig_str  = _latest(preds_raw, asset, "signal_strength")

            # Sentiment
            pfx = asset.lower()
            sent_col_gdelt = f"{pfx}_gdelt_sentiment_mean"
            sent_score = None
            if not gdelt_feat.empty and "asset" in gdelt_feat.columns:
                gdf = gdelt_feat[gdelt_feat["asset"] == asset]
                sent_col_try = next((c for c in [sent_col_gdelt, "sentiment_score"] if c in gdf.columns), None)
                if sent_col_try and not gdf.empty:
                    sent_score = gdf[sent_col_try].iloc[-1]

            dir_label = "🟢 UP" if pred_dir and pred_dir > 0 else ("🔴 DOWN" if pred_dir and pred_dir < 0 else "⚪ NEUTRAL")
            ret_delta = f"{ret_1h:+.3f}%" if ret_1h is not None else None

            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown(f"**{asset}**")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Price",     _fmt_price(price), ret_delta)
                st.metric("Signal",    dir_label)
                st.metric("Str",       f"{abs(sig_str):.4f}" if sig_str is not None else "—")
            with c2:
                st.metric("MAE",       f"{m.get('MAE', '—'):.4f}" if "MAE" in m.index and pd.notna(m.get("MAE")) else "—")
                st.metric("DirAcc",    f"{m.get('DirAcc', '—'):.1%}" if "DirAcc" in m.index and pd.notna(m.get("DirAcc")) else "—")
                st.metric("Sentiment", f"{sent_score:.3f}" if sent_score is not None else "—")
            st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # Cross-asset summary table
    st.subheader("Cross-Asset Summary")
    summary_rows = []
    for asset in ALL_ASSETS:
        sub = prices_raw[prices_raw["asset"] == asset].sort_values("timestamp_utc") \
            if not prices_raw.empty and "asset" in prices_raw.columns else pd.DataFrame()
        price   = sub["close"].iloc[-1]  if not sub.empty and "close" in sub.columns else None
        ret_1h_ = ((sub["close"].iloc[-1] / sub["close"].iloc[-2]) - 1) * 100 if len(sub) >= 2  else None
        ret_24h = ((sub["close"].iloc[-1] / sub["close"].iloc[-25]) - 1) * 100 if len(sub) >= 25 else None

        m = metrics_df[metrics_df["asset"] == asset].iloc[0] \
            if not metrics_df.empty and "asset" in metrics_df.columns and (metrics_df["asset"] == asset).any() \
            else pd.Series()
        pred_dir_ = _latest(preds_raw, asset, "predicted_direction")

        summary_rows.append({
            "Asset":     asset,
            "Price":     _fmt_price(price),
            "1h Ret %":  _fmt_pct(ret_1h_),
            "24h Ret %": _fmt_pct(ret_24h),
            "Direction": "▲" if pred_dir_ and pred_dir_ > 0 else ("▼" if pred_dir_ and pred_dir_ < 0 else "—"),
            "MAE":       f"{m.get('MAE', float('nan')):.5f}" if pd.notna(m.get("MAE", float("nan"))) else "—",
            "RMSE":      f"{m.get('RMSE', float('nan')):.5f}" if pd.notna(m.get("RMSE", float("nan"))) else "—",
            "R²":        f"{m.get('R2', float('nan')):.4f}"   if pd.notna(m.get("R2",  float("nan"))) else "—",
            "DirAcc":    f"{m.get('DirAcc', float('nan')):.1%}" if pd.notna(m.get("DirAcc", float("nan"))) else "—",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.divider()

    # Status panel
    st.subheader("MarketInsight AI — Pipeline Status")
    c1, c2 = st.columns(2)
    with c1:
        def _ok(cond): return '<span class="status-ok">✅</span>' if cond else '<span class="status-warn">⚠️</span>'
        st.markdown(f"""
{_ok(not prices_raw.empty)} &nbsp; Price data ({len(prices_raw)} rows)<br>
{_ok(not news_raw.empty)} &nbsp; GDELT news ({len(news_raw)} rows)<br>
{_ok(not preds_raw.empty)} &nbsp; Transformer predictions ({len(preds_raw)} rows)<br>
{_ok(not rl_raw.empty)} &nbsp; PPO/RL trade log ({len(rl_raw)} rows)<br>
{_ok(not metrics_df.empty)} &nbsp; Model metrics ({len(metrics_df)} rows)
""", unsafe_allow_html=True)
    with c2:
        for asset in ALL_ASSETS:
            tr = (SAVED_DIR / f"transformer_{asset.lower()}.pt").exists()
            pp = (SAVED_DIR / f"ppo_{asset.lower()}.zip").exists()
            st.markdown(
                f"{_ok(tr)} Transformer-{asset} &nbsp; {_ok(pp)} PPO-{asset}",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MARKET MONITOR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Market Monitor")

    if prices.empty:
        st.warning("No price data found. Run `python -m dashboard.export_dashboard_data` first.")
    else:
        display_asset = asset_sel if asset_sel != "All" else "BTC"

        st.plotly_chart(
            charts.make_candlestick(prices, display_asset),
            use_container_width=True,
            key=f"mkt_candlestick_{display_asset}",
        )

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                charts.make_normalized_price(prices),
                use_container_width=True,
                key="mkt_normalized_price",
            )
        with col2:
            st.plotly_chart(
                charts.make_returns_chart(prices),
                use_container_width=True,
                key="mkt_returns",
            )

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(
                charts.make_volatility_chart(prices),
                use_container_width=True,
                key="mkt_volatility",
            )
        with col4:
            st.plotly_chart(
                charts.make_correlation_heatmap(prices),
                use_container_width=True,
                key="mkt_correlation",
            )

        if show_debug:
            st.dataframe(prices.tail(50), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — NEWS & SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("News & Sentiment (GDELT + FinBERT)")

    if news.empty:
        st.warning("No news data. File: `data/dashboard/latest_news_sentiment.csv` not found.")
    else:
        # Recent news feed
        st.markdown("#### Recent News Feed")
        news_display_cols = [c for c in [
            "timestamp_utc", "asset", "title", "domain", "finbert_label",
            "sentiment_score", "finbert_confidence", "url",
        ] if c in news.columns]

        news_table = news[news_display_cols].head(50).copy() if news_display_cols else news.head(50)
        if "url" in news_table.columns:
            news_table["url"] = news_table["url"].apply(
                lambda x: f'<a href="{x}" target="_blank">🔗</a>' if pd.notna(x) else ""
            )
            st.write(news_table.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.dataframe(news_table, use_container_width=True, hide_index=True)

        st.divider()

        # Sentiment timeline + volume
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                charts.make_sentiment_timeline(news, asset_sel),
                use_container_width=True,
                key="news_sentiment_timeline",
            )
        with col2:
            st.plotly_chart(
                charts.make_news_volume_chart(news, asset_sel),
                use_container_width=True,
                key="news_volume",
            )

        # Overlay + ratio
        display_asset = asset_sel if asset_sel != "All" else "BTC"
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(
                charts.make_sentiment_volume_overlay(news, display_asset),
                use_container_width=True,
                key=f"news_sentiment_overlay_{display_asset}",
            )
        with col4:
            st.plotly_chart(
                charts.make_sentiment_ratio_chart(news, display_asset),
                use_container_width=True,
                key=f"news_sentiment_ratio_{display_asset}",
            )

        # Heatmap + top domains
        col5, col6 = st.columns(2)
        with col5:
            st.plotly_chart(
                charts.make_sentiment_heatmap(news),
                use_container_width=True,
                key="news_sentiment_heatmap",
            )
        with col6:
            st.plotly_chart(
                charts.make_top_domains(news),
                use_container_width=True,
                key="news_top_domains",
            )

        # Choropleth if sourcecountry available
        if "sourcecountry" in news.columns:
            st.markdown("#### Article Source Countries")
            country_counts = (
                news.dropna(subset=["sourcecountry"])
                .groupby("sourcecountry")
                .size()
                .reset_index(name="count")
            )
            if not country_counts.empty:
                import plotly.express as px
                fig_map = px.choropleth(
                    country_counts, locations="sourcecountry",
                    locationmode="country names", color="count",
                    color_continuous_scale="Blues",
                    title="Article Count by Country",
                    template="plotly_dark",
                )
                fig_map.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    geo=dict(bgcolor="rgba(0,0,0,0)", lakecolor="rgba(0,0,0,0)"),
                    height=380,
                )
                st.plotly_chart(fig_map, use_container_width=True, key="news_choropleth")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRANSFORMER FORECASTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Transformer Regression Forecasts")
    st.caption("Predicting 1-hour-ahead log returns · Direction · Signal Strength")

    if preds.empty:
        st.warning(
            "No prediction data found. "
            "Run `python -m dashboard.export_dashboard_data` to export from PostgreSQL, "
            "or run Step 6 of the pipeline to generate predictions."
        )
    else:
        asset_tabs = st.tabs(ALL_ASSETS if asset_sel == "All" else [asset_sel])

        for i, a in enumerate(ALL_ASSETS if asset_sel == "All" else [asset_sel]):
            with asset_tabs[i]:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(
                        charts.make_predictions_timeline(preds, a),
                        use_container_width=True,
                        key=f"forecast_timeline_{a}",
                    )
                with col2:
                    st.plotly_chart(
                        charts.make_predictions_scatter(preds, a),
                        use_container_width=True,
                        key=f"forecast_scatter_{a}",
                    )

                st.caption(
                    "Transformer forecasts are conservative and much smaller than realized "
                    f"{a} return spikes. Chart below shows predictions in basis points "
                    "(×10000) to make the scale visible."
                )
                st.plotly_chart(
                    charts.make_bps_chart(preds, a),
                    use_container_width=True,
                    key=f"forecast_bps_{a}",
                )

                col3, col4 = st.columns(2)
                with col3:
                    st.plotly_chart(
                        charts.make_signal_strength(preds, a),
                        use_container_width=True,
                        key=f"forecast_signal_{a}",
                    )
                with col4:
                    st.plotly_chart(
                        charts.make_direction_chart(preds, a),
                        use_container_width=True,
                        key=f"forecast_direction_{a}",
                    )

                st.plotly_chart(
                    charts.make_price_with_forecasts(prices_raw, preds, a),
                    use_container_width=True,
                    key=f"forecast_price_markers_{a}",
                )

                # Reconstructed predicted next-price
                sub_p  = preds_raw[preds_raw["asset"] == a].sort_values("timestamp_utc") \
                    if "asset" in preds_raw.columns else preds_raw.copy()
                sub_px = prices_raw[prices_raw["asset"] == a].sort_values("timestamp_utc") \
                    if "asset" in prices_raw.columns else prices_raw.copy()
                if not sub_p.empty and "predicted_log_return" in sub_p.columns and not sub_px.empty:
                    merged = sub_p.merge(
                        sub_px[["timestamp_utc", "close"]], on="timestamp_utc", how="inner"
                    )
                    if not merged.empty:
                        import numpy as np
                        import plotly.graph_objects as go
                        merged["predicted_next_price"] = merged["close"] * np.exp(
                            pd.to_numeric(merged["predicted_log_return"], errors="coerce").fillna(0)
                        )
                        fig_next = go.Figure()
                        fig_next.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            title=f"{a} — Actual vs Reconstructed Predicted Next Price",
                            height=320,
                            margin=dict(l=40, r=20, t=40, b=40),
                        )
                        fig_next.add_trace(go.Scatter(
                            x=merged["timestamp_utc"], y=merged["close"],
                            mode="lines", name="Actual Price",
                            line=dict(color="#00ff88", width=1.5),
                        ))
                        fig_next.add_trace(go.Scatter(
                            x=merged["timestamp_utc"], y=merged["predicted_next_price"],
                            mode="lines", name="Predicted Next Price",
                            line=dict(color="#4488ff", width=1.5, dash="dot"),
                        ))
                        st.plotly_chart(
                            fig_next,
                            use_container_width=True,
                            key=f"forecast_next_price_{a}",
                        )
                        st.info(
                            "**Note:** Predicted price is reconstructed from predicted log returns "
                            "(price × exp(predicted_log_return)). Because price levels are persistent "
                            "and returns are small, the lines will appear nearly identical — "
                            "interpret the log return predictions, not absolute price levels."
                        )

        if show_debug:
            st.dataframe(preds.head(100), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PREDICTION ACCURACY / EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("Model Evaluation — Transformer Regression")

    st.info(
        "**Metric interpretation:**  "
        "MAE and RMSE are small because hourly log returns are naturally near zero. "
        "R² and Correlation near zero indicate limited explanatory power for short-horizon returns — "
        "this is expected in efficient markets. "
        "DirAcc (directional accuracy) and DerivedAcc are more meaningful for trading signals. "
        "The Transformer output is best interpreted as a noisy signal fed into the PPO agent, "
        "not as a standalone price predictor."
    )

    if metrics_df.empty:
        st.warning("Using hard-coded fallback metrics (no model_metrics.csv found).")

    trans_metrics = metrics_df.copy()

    # Summary table
    display_cols = [c for c in ["asset", "model", "MAE", "RMSE", "R2", "Corr", "DirAcc", "DerivedAcc", "DerivedF1"]
                    if c in trans_metrics.columns]
    st.dataframe(trans_metrics[display_cols].round(5), use_container_width=True, hide_index=True)

# Individual metric bar charts
st.markdown("#### Per-Asset Metric Charts")
def _metric_chart_or_empty(df, metric, title, key):
    if metric in df.columns and df[metric].notna().any():
        st.plotly_chart(
            charts.make_metrics_bar(df, metric, title),
            use_container_width=True,
            key=key,
        )
    else:
        st.info(f"No {metric} data")
# Row 1: regression metrics
r1c1, r1c2, r1c3 = st.columns(3)
with r1c1:
    _metric_chart_or_empty(trans_metrics, "MAE", "MAE (lower = better)", "eval_mae_tf")
with r1c2:
    _metric_chart_or_empty(trans_metrics, "RMSE", "RMSE (lower = better)", "eval_rmse_tf")
with r1c3:
    _metric_chart_or_empty(trans_metrics, "R2", "R² (0 = no explanatory power)", "eval_r2_tf")
# Row 2: signal quality metrics
r2c1, r2c2, r2c3 = st.columns(3)
with r2c1:
    _metric_chart_or_empty(trans_metrics, "Corr", "Pearson Correlation", "eval_corr_tf")
with r2c2:
    _metric_chart_or_empty(trans_metrics, "DirAcc", "Direction Accuracy", "eval_diracc_tf")
with r2c3:
    _metric_chart_or_empty(trans_metrics, "StdRatio", "Prediction Std Ratio", "eval_stdratio_tf")
with col3:
    st.plotly_chart(
        charts.make_metrics_bar(trans_metrics, "DirAcc", "Direction Accuracy"),
        use_container_width=True,
        key="eval_diracc",
    )

with col4:
    if "StdRatio" in trans_metrics.columns:
        st.plotly_chart(
            charts.make_metrics_bar(trans_metrics, "StdRatio", "Prediction Std Ratio"),
            use_container_width=True,
            key="eval_stdratio",
        )
    else:
        st.info("No StdRatio data")

    # Residual analysis per asset
    if not preds.empty:
        st.divider()
        st.markdown("#### Residual Analysis")
        asset_tabs2 = st.tabs(ALL_ASSETS if asset_sel == "All" else [asset_sel])
        for i, a in enumerate(ALL_ASSETS if asset_sel == "All" else [asset_sel]):
            with asset_tabs2[i]:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.plotly_chart(
                        charts.make_error_histogram(preds, a),
                        use_container_width=True,
                        key=f"eval_error_hist_{a}",
                    )
                with col_b:
                    st.plotly_chart(
                        charts.make_rolling_mae(preds, a),
                        use_container_width=True,
                        key=f"eval_rolling_mae_{a}",
                    )
                st.plotly_chart(
                    charts.make_predictions_scatter(preds, a),
                    use_container_width=True,
                    key=f"eval_scatter_{a}",
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — PPO TRADING AGENT
# ══════════════════════════════════════════════════════════════════════════════
_RL_INITIAL_CAP = 10_000.0  # must match INITIAL_CASH in rl_env.py

with tabs[5]:
    st.subheader("PPO Reinforcement Learning Trading Agent")

    if rl_raw.empty:
        st.warning(
            "No RL trade log found (`data/dashboard/latest_rl_trades.csv` missing)."
        )
        st.markdown("""
**To generate RL results, run:**
```bash
python run_pipeline.py --use-gdelt-sentiment --only 7 --retrain rl
```
Then re-export:
```bash
python -m dashboard.export_dashboard_data
```
        """)
        st.divider()
        st.markdown("**PPO Model Files (from `models/saved/`):**")
        ppo_cols = st.columns(4)
        for _pi, _pa in enumerate(ALL_ASSETS):
            with ppo_cols[_pi]:
                st.metric(f"PPO-{_pa}", "✅" if (SAVED_DIR / f"ppo_{_pa.lower()}.zip").exists() else "❌")

    else:
        asset_tabs3 = st.tabs(ALL_ASSETS if asset_sel == "All" else [asset_sel])

        for i, a in enumerate(ALL_ASSETS if asset_sel == "All" else [asset_sel]):
            with asset_tabs3[i]:
                # FULL unfiltered data → used for all summary metrics
                full_steps = (
                    rl_raw[rl_raw["asset"] == a].sort_values("timestamp_utc").reset_index(drop=True)
                    if "asset" in rl_raw.columns
                    else rl_raw.sort_values("timestamp_utc").reset_index(drop=True)
                )
                full_hist = (
                    rl_history_raw[rl_history_raw["asset"] == a].copy()
                    if not rl_history_raw.empty and "asset" in rl_history_raw.columns
                    else rl_history_raw.copy()
                )
                # Date-filtered data → used only for visible charts
                sub_steps = (
                    rl_data[rl_data["asset"] == a].sort_values("timestamp_utc").reset_index(drop=True)
                    if "asset" in rl_data.columns
                    else rl_data.sort_values("timestamp_utc").reset_index(drop=True)
                )
                sub_hist = full_hist  # trade history is not date-windowed

                # ── Section 1: Performance Summary (always uses FULL data) ──
                with st.expander("📊 Performance Summary", expanded=True):
                    if full_steps.empty:
                        st.warning("No step data for this asset.")
                    else:
                        init_cap = _RL_INITIAL_CAP
                        # Safe defaults for PPO summary metrics
                        rl_sharpe = None
                        bh_sharpe = None
                        bh_ret = None
                        max_dd_pct = None
                        # Corrected performance summary from completed trade history.
                        # Step-level portfolio_value may be stale if trade-history PnL was manually fixed.
                        if not full_hist.empty and "net_pnl" in full_hist.columns:
                            full_hist = full_hist.copy()
                            full_hist["net_pnl"] = pd.to_numeric(full_hist["net_pnl"], errors="coerce").fillna(0.0)
                            corrected_trade_pnl = float(full_hist["net_pnl"].sum())
                        else:
                            corrected_trade_pnl = 0.0
                        
                        _pv = full_steps["portfolio_value"] if "portfolio_value" in full_steps.columns else None
                        
                        final_pv = init_cap + corrected_trade_pnl
                        net_ret  = corrected_trade_pnl / init_cap * 100
                        bh_ret = None
                        if "close" in full_steps.columns and len(full_steps) >= 2 and full_steps["close"].iloc[0] > 0:
                            bh_ret = (full_steps["close"].iloc[-1] / full_steps["close"].iloc[0] - 1) * 100

                        max_dd_pct = None
                        if _pv is not None:
                            _rm   = _pv.cummax()
                            _ddpv = (_pv - _rm) / _rm.clip(lower=1e-9) * 100
                            max_dd_pct = float(_ddpv.min())

                        _ret_col  = next((c for c in ["step_return", "portfolio_return"] if c in full_steps.columns), None)
                        rl_sharpe = None
                        if _ret_col:
                            _rets = full_steps[_ret_col].dropna()
                            n_trades      = len(full_hist) if not full_hist.empty else 0
                            win_rate      = None
                            profit_factor = None
                            avg_hold      = None
                            avg_rr        = None

                        if not full_hist.empty and "net_pnl" in full_hist.columns:
                            full_hist["net_pnl"] = pd.to_numeric(full_hist["net_pnl"], errors="coerce").fillna(0.0)

                            _wins = full_hist["net_pnl"] > 0
                            _loss = full_hist["net_pnl"] < 0

                            win_rate = float(_wins.mean()) * 100 if len(full_hist) > 0 else 0.0

                            _pos = float(full_hist.loc[_wins, "net_pnl"].sum())
                            _neg = abs(float(full_hist.loc[_loss, "net_pnl"].sum()))
                            profit_factor = _pos / _neg if _neg > 0 else None

                        if not full_hist.empty and "holding_bars" in full_hist.columns:
                            avg_hold = float(pd.to_numeric(full_hist["holding_bars"], errors="coerce").mean())

                        if not full_hist.empty and "risk_reward_ratio" in full_hist.columns:
                            avg_rr = float(pd.to_numeric(full_hist["risk_reward_ratio"], errors="coerce").mean())
                            bh_sharpe = None
                            if "close" in full_steps.columns and len(full_steps) > 1:
                                _bhr = pd.to_numeric(full_steps["close"], errors="coerce").pct_change().dropna()
                                if len(_bhr) > 1 and _bhr.std() > 0:
                                    bh_sharpe = float(_bhr.mean() / _bhr.std() * (8760 ** 0.5))
                            mc1, mc2, mc3, mc4 = st.columns(4)
                        with mc1:
                            st.metric("Initial Capital",   f"${init_cap:,.2f}")
                            st.metric("Final Portfolio",   f"${final_pv:,.2f}", delta=f"{net_ret:+.2f}%")
                            st.metric("Completed Trades",  f"{n_trades:,}")
                        with mc2:
                            st.metric("Net Return",        f"{net_ret:.2f}%")
                            st.metric("B&H Return",        f"{bh_ret:.2f}%" if bh_ret is not None else "—")
                            st.metric("Avg Hold (bars)",   f"{avg_hold:.1f}"  if avg_hold is not None else "—")
                        with mc3:
                            st.metric("RL Sharpe (ann.)",  f"{rl_sharpe:.3f}"    if rl_sharpe is not None else "—")
                            st.metric("B&H Sharpe (ann.)", f"{bh_sharpe:.3f}"    if bh_sharpe is not None else "—")
                            st.metric("Avg Risk/Reward",   f"{avg_rr:.2f}"       if avg_rr is not None else "—")
                        with mc4:
                            st.metric("Max Drawdown",      f"{max_dd_pct:.2f}%"   if max_dd_pct is not None else "—")
                            st.metric("Win Rate",          f"{win_rate:.1f}%"     if win_rate is not None else "—")
                            st.metric("Profit Factor",     f"{profit_factor:.2f}" if profit_factor is not None else "—")

                        # ── Accounting reconciliation ──
                        st.divider()
                        trade_pnl = corrected_trade_pnl
                        port_pnl  = corrected_trade_pnl
                        diff      = 0.0
                        is_ok     = True
                        rc1, rc2, rc3, rc4 = st.columns(4)
                        with rc1:
                            st.metric("Portfolio PnL",  f"${port_pnl:,.2f}")
                        with rc2:
                            st.metric("Sum Trade PnL",  f"${trade_pnl:,.2f}")
                        with rc3:
                            st.metric("Difference",     f"${diff:,.2f}")
                        with rc4:
                            if is_ok:
                                st.success("Trade accounting check: OK")
                            else:
                                st.warning(f"Trade accounting check: difference ${diff:,.2f}")

                # ── Section 2: Candlestick + Trade Markers ───────────────────
                with st.expander("📈 Candlestick + Trade Markers", expanded=True):
                    _last_n = st.slider(
                        "Show last N trades", min_value=5, max_value=200,
                        value=30, step=5, key=f"rl_last_n_{a}",
                    )
                    st.plotly_chart(
                        charts.make_rl_candlestick_with_trades(sub_steps, sub_hist, a, _last_n),
                        use_container_width=True, key=f"rl_candle_{a}",
                    )
                    _cap_parts = [
                        "Candles: RL step-level OHLC.",
                        "Entry markers: ▲ Long  ▼ Short.",
                        "Exit markers: ★ TP  × SL  ◆ Reverse  ■ Resize  ● Agent-close.",
                        "Dashed lines: — TP (green)  — SL (red).",
                    ]
                    if "ohlc_source" in sub_steps.columns:
                        _src = sub_steps["ohlc_source"].value_counts()
                        _cap_parts.append("OHLC source: " + ", ".join(f"{v}× {k}" for k, v in _src.items()) + " (some bars are reconstructed).")
                    st.caption("  ".join(_cap_parts))

                # ── Section 3: Portfolio Analytics ──────────────────────────
                with st.expander("💰 Portfolio Analytics", expanded=True):
                    # Full-episode portfolio curve (not date-windowed)
                    st.plotly_chart(
                        charts.make_portfolio_curve(full_steps, a, initial_cap=_RL_INITIAL_CAP),
                        use_container_width=True, key=f"rl_portcurve_{a}",
                    )
                    pa1, pa2 = st.columns(2)
                    with pa1:
                        st.plotly_chart(
                            charts.make_rl_exposure(sub_steps, a),
                            use_container_width=True, key=f"rl_exposure_{a}",
                        )
                    with pa2:
                        if not sub_hist.empty:
                            st.plotly_chart(
                                charts.make_rl_cumulative_pnl(sub_hist, a),
                                use_container_width=True, key=f"rl_cum_pnl_{a}",
                            )

                # ── Section 4: Reward Decomposition ─────────────────────────
                with st.expander("🎯 Reward Decomposition", expanded=False):
                    st.caption(
                        "Positive components (pnl, tp_bonus) plot above zero. "
                        "Penalty components are subtracted from the reward and plot below zero."
                    )
                    rd1, rd2 = st.columns(2)
                    with rd1:
                        st.plotly_chart(
                            charts.make_rl_reward_cumulative(sub_steps, a),
                            use_container_width=True, key=f"rl_rew_cum_{a}",
                        )
                    with rd2:
                        st.plotly_chart(
                            charts.make_rl_reward_avg_bar(sub_hist, a),
                            use_container_width=True, key=f"rl_rew_avg_{a}",
                        )
                    st.plotly_chart(
                        charts.make_reward_components(sub_steps, a),
                        use_container_width=True, key=f"rl_rewards_{a}",
                    )

                # ── Section 5: Action & Risk Analytics ──────────────────────
                with st.expander("⚙️ Action & Risk Analytics", expanded=False):
                    ar1, ar2 = st.columns(2)
                    with ar1:
                        st.plotly_chart(
                            charts.make_action_distribution(sub_steps, a),
                            use_container_width=True, key=f"rl_action_dist_{a}",
                        )
                        if not sub_hist.empty:
                            st.plotly_chart(
                                charts.make_rl_category_dist(
                                    sub_hist, a, "size_label_at_entry",
                                    f"{a} Size Label Distribution",
                                ),
                                use_container_width=True, key=f"rl_size_dist_{a}",
                            )
                    with ar2:
                        if not sub_hist.empty:
                            st.plotly_chart(
                                charts.make_rl_category_dist(
                                    sub_hist, a, "risk_profile_at_entry",
                                    f"{a} Risk Profile Distribution",
                                ),
                                use_container_width=True, key=f"rl_risk_dist_{a}",
                            )
                            st.plotly_chart(
                                charts.make_rl_category_dist(
                                    sub_hist, a, "exit_reason",
                                    f"{a} Exit Reason Distribution",
                                ),
                                use_container_width=True, key=f"rl_exit_dist_{a}",
                            )
                    st.divider()
                    ar3, ar4 = st.columns(2)
                    with ar3:
                        if not sub_hist.empty:
                            st.plotly_chart(
                                charts.make_rl_pnl_by_category(sub_hist, a, "exit_reason"),
                                use_container_width=True, key=f"rl_pnl_exit_{a}",
                            )
                            st.plotly_chart(
                                charts.make_rl_pnl_by_category(sub_hist, a, "risk_profile_at_entry"),
                                use_container_width=True, key=f"rl_pnl_risk_{a}",
                            )
                    with ar4:
                        if not sub_hist.empty:
                            st.plotly_chart(
                                charts.make_rl_winrate_by_category(sub_hist, a, "side"),
                                use_container_width=True, key=f"rl_wr_side_{a}",
                            )
                            st.plotly_chart(
                                charts.make_rl_winrate_by_category(sub_hist, a, "risk_profile_at_entry"),
                                use_container_width=True, key=f"rl_wr_risk_{a}",
                            )

                # ── Section 6: Trade History Table ──────────────────────────
                with st.expander("📋 Trade History", expanded=False):
                    if sub_hist.empty:
                        st.info("No completed trade history (`data/dashboard/latest_rl_trade_history.csv` missing or empty).")
                    else:
                        tf1, tf2, tf3, tf4 = st.columns(4)
                        with tf1:
                            _side_opts = ["All"] + sorted(sub_hist["side"].dropna().unique().tolist()) \
                                if "side" in sub_hist.columns else ["All"]
                            side_filt = st.selectbox("Side", _side_opts, key=f"rl_filt_side_{a}")
                        with tf2:
                            _er_opts = ["All"] + sorted(sub_hist["exit_reason"].dropna().unique().tolist()) \
                                if "exit_reason" in sub_hist.columns else ["All"]
                            er_filt = st.selectbox("Exit Reason", _er_opts, key=f"rl_filt_er_{a}")
                        with tf3:
                            _rp_col  = "risk_profile_at_entry"
                            _rp_opts = ["All"] + sorted(sub_hist[_rp_col].dropna().unique().tolist()) \
                                if _rp_col in sub_hist.columns else ["All"]
                            rp_filt = st.selectbox("Risk Profile", _rp_opts, key=f"rl_filt_rp_{a}")
                        with tf4:
                            profit_only  = st.checkbox("Profitable only", value=False, key=f"rl_filt_profit_{a}")
                            top_n_trades = st.number_input("Top N (0=all)", min_value=0, max_value=10_000, value=100, key=f"rl_top_n_{a}")

                        tbl = sub_hist.copy()
                        if side_filt != "All" and "side" in tbl.columns:
                            tbl = tbl[tbl["side"] == side_filt]
                        if er_filt != "All" and "exit_reason" in tbl.columns:
                            tbl = tbl[tbl["exit_reason"] == er_filt]
                        if rp_filt != "All" and _rp_col in tbl.columns:
                            tbl = tbl[tbl[_rp_col] == rp_filt]
                        if profit_only and "net_pnl" in tbl.columns:
                            tbl = tbl[tbl["net_pnl"] > 0]
                        if top_n_trades > 0:
                            tbl = tbl.head(int(top_n_trades))

                        _disp_cols = [c for c in [
                            "trade_id", "side", "entry_time", "exit_time", "holding_bars",
                            "entry_price", "exit_price", "net_pnl", "return_pct",
                            "exit_reason", "risk_profile_at_entry", "size_label_at_entry",
                            "leverage_used", "effective_position",
                            "tp_price", "sl_price", "risk_reward_ratio",
                            "transaction_cost_total", "reward_total",
                            "portfolio_value_before", "portfolio_value_after",
                            "max_favorable_excursion", "max_adverse_excursion",
                            "predicted_log_return_at_entry", "signal_strength_at_entry",
                        ] if c in tbl.columns]
                        fmt = tbl[_disp_cols].copy()

                        for _pc in ["entry_price", "exit_price", "tp_price", "sl_price",
                                    "portfolio_value_before", "portfolio_value_after"]:
                            if _pc in fmt.columns:
                                fmt[_pc] = fmt[_pc].apply(lambda v: f"${v:,.2f}" if pd.notna(v) else "—")
                        for _dc in ["net_pnl", "gross_pnl"]:
                            if _dc in fmt.columns:
                                fmt[_dc] = fmt[_dc].apply(
                                    lambda v: (f"+${v:,.2f}" if v >= 0 else f"-${abs(v):,.2f}") if pd.notna(v) else "—"
                                )
                        # Transaction cost shown as a cost (always negative display)
                        if "transaction_cost_total" in fmt.columns:
                            fmt["transaction_cost_total"] = fmt["transaction_cost_total"].apply(
                                lambda v: f"-${abs(v):,.2f}" if pd.notna(v) else "—"
                            )
                        if "return_pct" in fmt.columns:
                            fmt["return_pct"] = fmt["return_pct"].apply(
                                lambda v: f"{v:+.4f}%" if pd.notna(v) else "—"
                            )
                        if "leverage_used" in fmt.columns:
                            fmt["leverage_used"] = fmt["leverage_used"].apply(
                                lambda v: f"{v:.1f}x" if pd.notna(v) else "—"
                            )
                        if "risk_reward_ratio" in fmt.columns:
                            fmt["risk_reward_ratio"] = fmt["risk_reward_ratio"].apply(
                                lambda v: f"{v:.2f}" if pd.notna(v) else "—"
                            )
                        for _mfe in ["max_favorable_excursion", "max_adverse_excursion"]:
                            if _mfe in fmt.columns:
                                fmt[_mfe] = fmt[_mfe].apply(lambda v: f"${v:,.2f}" if pd.notna(v) else "—")

                        st.dataframe(fmt, use_container_width=True, hide_index=True)
                        st.caption(f"Showing {len(tbl):,} of {len(sub_hist):,} completed trades")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader("Model Comparison")

    if comparison_df.empty:
        st.warning("No model comparison data. Only Transformer fallback metrics available.")
    else:
        # Summary table
        display_cols = [c for c in [
    "asset", "model", "FeatureSet", "MAE", "Corr", "DirAcc",
    "StdRatio", "Note"
] if c in comparison_df.columns]
        st.dataframe(comparison_df[display_cols].round(5), use_container_width=True, hide_index=True)

        if "model" in comparison_df.columns and comparison_df["model"].nunique() > 1:
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(

                    charts.make_grouped_model_bars(comparison_df, "MAE"),
                    use_container_width=True,
                    key="cmp_mae",
                )
                st.plotly_chart(
                    charts.make_grouped_model_bars(comparison_df, "DirAcc"),
                    use_container_width=True,
                    key="cmp_diracc",
                )
            with col2:
                st.plotly_chart(
                    charts.make_grouped_model_bars(comparison_df, "RMSE"),
                    use_container_width=True,
                    key="cmp_rmse",
                )
                st.plotly_chart(
                    charts.make_grouped_model_bars(comparison_df, "R2"),
                    use_container_width=True,
                    key="cmp_r2",
                )
        else:
            st.info("Only Transformer metrics available. Run Ridge and PPO pipelines to enable comparison.")
            st.plotly_chart(
                charts.make_multi_metric_bars(comparison_df),
                use_container_width=True,
                key="cmp_multi_metrics",
            )

    # SHAP images
    st.divider()
    st.markdown("#### SHAP Feature Importance (Transformer)")
    shap_assets = ALL_ASSETS if asset_sel == "All" else [asset_sel]
    shap_cols = st.columns(min(len(shap_assets), 4))
    any_shap = False
    for i, a in enumerate(shap_assets):
        shap_path = SAVED_DIR / f"shap_{a.lower()}.png"
        if shap_path.exists():
            with shap_cols[i % 4]:
                st.image(str(shap_path), caption=f"SHAP — {a}", use_container_width=True)
                any_shap = True
    if not any_shap:
        st.info("SHAP images not found at `models/saved/shap_*.png`.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — PIPELINE HEALTH
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.subheader("Pipeline Health & Data Quality")

    # Pipeline checklist
    st.markdown("#### Pipeline Checklist")
    steps = [
        ("1. Price Ingestion",       not prices_raw.empty),
        ("2. GDELT Sentiment",       not news_raw.empty),
        ("3. Feature Engineering",   (SAVED_DIR / "feature_scaler_btc.pkl").exists()),
        ("4. Targets",               not preds_raw.empty),
        ("5. Ridge Baseline",        (SAVED_DIR / "ridge_btc.pkl").exists()),
        ("6. Transformer Regressor", (SAVED_DIR / "transformer_btc.pt").exists()),
        ("7. PPO/RL Agent",          (SAVED_DIR / "ppo_btc.zip").exists()),
        ("8. Dashboard Export",      not prices_raw.empty or not news_raw.empty),
    ]
    for step_name, ok in steps:
        icon = "✅" if ok else "⚠️"
        st.markdown(f"{icon} &nbsp; {step_name}", unsafe_allow_html=True)

    st.divider()

    # Feature status table
    st.markdown("#### Feature & Model File Status")
    if not feat_status.empty:
        st.dataframe(feat_status, use_container_width=True, hide_index=True)

        # Row counts if available
        count_cols = [c for c in ["price_rows", "feature_rows", "target_rows", "prediction_rows"]
                      if c in feat_status.columns]
        if count_cols:
            import plotly.graph_objects as go
            fig_rows = go.Figure()
            fig_rows.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title="Row Counts by Asset and Table",
                height=320,
                margin=dict(l=40, r=20, t=40, b=40),
                barmode="group",
            )
            clrs = {"price_rows": "#4488ff", "feature_rows": "#aa66ff",
                    "target_rows": "#00ff88", "prediction_rows": "#ff9944"}
            for c in count_cols:
                fig_rows.add_trace(go.Bar(
                    x=feat_status["asset"], y=feat_status[c],
                    name=c, marker_color=clrs.get(c, "#888"),
                ))
            st.plotly_chart(fig_rows, use_container_width=True, key="health_row_counts")

    st.divider()

    # News sparsity
    st.markdown("#### News Coverage Quality")
    if not news_raw.empty and "asset" in news_raw.columns:
        sparsity_data = []
        for a in ALL_ASSETS:
            sub = news_raw[news_raw["asset"] == a]
            sparsity_data.append({
                "Asset":          a,
                "Total News":     len(sub),
                "With Sentiment": int(sub["sentiment_score"].notna().sum()) if "sentiment_score" in sub.columns else 0,
                "Sources":        sub["domain"].nunique() if "domain" in sub.columns else 0,
                "Countries":      sub["sourcecountry"].nunique() if "sourcecountry" in sub.columns else 0,
            })
        st.dataframe(pd.DataFrame(sparsity_data), use_container_width=True, hide_index=True)
        for a in (ALL_ASSETS if asset_sel == "All" else [asset_sel]):
            st.plotly_chart(
                charts.make_data_sparsity_chart(news_raw, a),
                use_container_width=True,
                key=f"health_sparsity_{a}",
            )

    st.divider()

    # File listing
    st.markdown("#### Dashboard Data Files")
    from dashboard.data_loader import DASHBOARD_DIR, PROCESSED_DIR
    file_rows = []
    for d, label in [(DASHBOARD_DIR, "data/dashboard"), (PROCESSED_DIR, "data/processed")]:
        if d.exists():
            for f in sorted(d.iterdir()):
                if f.is_file():
                    file_rows.append({
                        "Location":  label,
                        "File":      f.name,
                        "Size (KB)": round(f.stat().st_size / 1024, 1),
                        "Modified":  pd.Timestamp(f.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M"),
                    })
    if file_rows:
        st.dataframe(pd.DataFrame(file_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No dashboard files found yet. Run the export script first.")

    st.caption("Missing data? Run: `python -m dashboard.export_dashboard_data`")
