"""
dashboard/charts.py
Plotly chart factories for MarketInsight AI Dashboard.
All functions return a go.Figure with the dark financial theme.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Shared style constants ─────────────────────────────────────────────────────
TEMPLATE   = "plotly_dark"
GREEN      = "#00ff88"
RED        = "#ff4455"
NEUTRAL    = "#888899"
BLUE       = "#4488ff"
PURPLE     = "#aa66ff"
GOLD_CLR   = "#ffd700"
ORANGE     = "#ff9944"

ASSET_COLORS = {"BTC": BLUE, "ETH": PURPLE, "GOLD": GOLD_CLR, "OIL": ORANGE}

_LAYOUT_BASE = dict(
    template=TEMPLATE,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#444"),
)


def _fig(**kwargs) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**_LAYOUT_BASE, **kwargs)
    return fig


def _sub(rows=1, cols=1, **kwargs) -> go.Figure:
    fig = make_subplots(rows=rows, cols=cols, **kwargs)
    fig.update_layout(**_LAYOUT_BASE)
    return fig


# ── Market Monitor ────────────────────────────────────────────────────────────

def make_candlestick(df: pd.DataFrame, asset: str) -> go.Figure:
    sub = df[df["asset"] == asset].sort_values("timestamp_utc") if "asset" in df.columns else df.copy()
    if sub.empty:
        return _empty("No price data")
    has_ohlc = all(c in sub.columns for c in ["open", "high", "low", "close"])
    has_vol  = "volume" in sub.columns

    rows = 2 if has_vol else 1
    heights = [0.75, 0.25] if has_vol else [1.0]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=heights,
                        vertical_spacing=0.03)
    fig.update_layout(**_LAYOUT_BASE, title=f"{asset} Price", height=480)

    if has_ohlc:
        fig.add_trace(go.Candlestick(
            x=sub["timestamp_utc"], open=sub["open"], high=sub["high"],
            low=sub["low"],  close=sub["close"],
            name=asset, increasing_line_color=GREEN, decreasing_line_color=RED,
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=sub["timestamp_utc"], y=sub["close"],
            mode="lines", line=dict(color=ASSET_COLORS.get(asset, BLUE), width=1.5),
            name=asset,
        ), row=1, col=1)

    if "close" in sub.columns and len(sub) >= 20:
        ma20 = sub["close"].rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=sub["timestamp_utc"], y=ma20, mode="lines",
            line=dict(color=NEUTRAL, width=1, dash="dot"), name="MA-20",
        ), row=1, col=1)

    if has_vol:
        colors = [GREEN if r >= 0 else RED
                  for r in sub["close"].pct_change().fillna(0)]
        fig.add_trace(go.Bar(
            x=sub["timestamp_utc"], y=sub["volume"],
            marker_color=colors, name="Volume", opacity=0.7,
        ), row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def make_normalized_price(df: pd.DataFrame) -> go.Figure:
    if df.empty or "close" not in df.columns:
        return _empty("No price data")
    fig = _fig(title="Normalized Price (base = 100)", height=380)
    for asset in df["asset"].unique():
        sub = df[df["asset"] == asset].sort_values("timestamp_utc")
        if sub.empty or sub["close"].iloc[0] == 0:
            continue
        norm = sub["close"] / sub["close"].iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=sub["timestamp_utc"], y=norm, mode="lines",
            name=asset, line=dict(color=ASSET_COLORS.get(asset, BLUE), width=2),
        ))
    return fig


def make_returns_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty or "close" not in df.columns:
        return _empty("No price data")
    fig = _fig(title="Hourly Returns by Asset (%)", height=320,
               barmode="group")
    for asset in df["asset"].unique():
        sub = df[df["asset"] == asset].sort_values("timestamp_utc").copy()
        sub["ret"] = sub["close"].pct_change() * 100
        colors = [GREEN if r >= 0 else RED for r in sub["ret"].fillna(0)]
        fig.add_trace(go.Bar(
            x=sub["timestamp_utc"], y=sub["ret"],
            name=asset, marker_color=colors, opacity=0.8,
        ))
    return fig


def make_volatility_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty or "close" not in df.columns:
        return _empty("No price data")
    fig = _fig(title="Rolling 20-Bar Volatility (annualised, %)", height=320)
    for asset in df["asset"].unique():
        sub = df[df["asset"] == asset].sort_values("timestamp_utc").copy()
        sub["vol"] = sub["close"].pct_change().rolling(20).std() * (24 ** 0.5) * 100
        fig.add_trace(go.Scatter(
            x=sub["timestamp_utc"], y=sub["vol"], mode="lines",
            name=asset, line=dict(color=ASSET_COLORS.get(asset, BLUE), width=1.5),
        ))
    return fig


def make_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    if df.empty or "close" not in df.columns:
        return _empty("Not enough price data")
    pivot = (
        df.pivot_table(index="timestamp_utc", columns="asset", values="close")
        .pct_change().dropna(how="all")
    )
    if pivot.shape[0] < 5:
        return _empty("Not enough data for correlation")
    corr = pivot.corr().round(3)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdYlGn", zmid=0, zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        hovertemplate="(%{x}, %{y}): %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(**_LAYOUT_BASE, title="Return Correlation Heatmap", height=350)
    return fig


# ── News & Sentiment ──────────────────────────────────────────────────────────

def make_sentiment_timeline(df: pd.DataFrame, asset: str = "All") -> go.Figure:
    fig = _fig(title="Sentiment Score Over Time", height=340)
    if df.empty:
        return _empty("No sentiment data")

    ts_col  = "timestamp_utc" if "timestamp_utc" in df.columns else None
    sent_col = next((c for c in ["sentiment_score", "gdelt_sentiment_mean"] if c in df.columns), None)
    if ts_col is None or sent_col is None:
        return _empty("Missing sentiment columns")

    assets = [asset] if asset != "All" else df["asset"].unique().tolist() if "asset" in df.columns else ["All"]
    for a in assets:
        sub = df[df["asset"] == a] if "asset" in df.columns else df
        sub = sub.dropna(subset=[ts_col, sent_col]).sort_values(ts_col)
        if sub.empty:
            continue
        # Resample to hourly mean
        sub2 = sub.set_index(ts_col)[sent_col].resample("1h").mean().reset_index()
        fig.add_trace(go.Scatter(
            x=sub2[ts_col], y=sub2[sent_col],
            mode="lines", name=a,
            line=dict(color=ASSET_COLORS.get(a, BLUE), width=1.5),
        ))
    fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL, opacity=0.5)
    return fig


def make_news_volume_chart(df: pd.DataFrame, asset: str = "All") -> go.Figure:
    if df.empty or "timestamp_utc" not in df.columns:
        return _empty("No news data")
    fig = _fig(title="News Volume Over Time", height=280)
    assets = [asset] if asset != "All" else (df["asset"].unique().tolist() if "asset" in df.columns else ["All"])
    for a in assets:
        sub = df[df["asset"] == a].copy() if "asset" in df.columns else df.copy()
        sub = sub.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
        sub = sub.set_index("timestamp_utc").resample("6h").size().reset_index(name="count")
        fig.add_trace(go.Bar(
            x=sub["timestamp_utc"], y=sub["count"],
            name=a, marker_color=ASSET_COLORS.get(a, BLUE), opacity=0.8,
        ))
    fig.update_layout(barmode="group")
    return fig


def make_sentiment_volume_overlay(df: pd.DataFrame, asset: str) -> go.Figure:
    if df.empty or "timestamp_utc" not in df.columns:
        return _empty("No data")
    sub = df[df["asset"] == asset].copy() if "asset" in df.columns else df.copy()
    sub = sub.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    hourly_vol  = sub.set_index("timestamp_utc").resample("6h").size().rename("count")
    sent_col = next((c for c in ["sentiment_score", "gdelt_sentiment_mean"] if c in sub.columns), None)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(**_LAYOUT_BASE, title=f"{asset} — Sentiment & Volume", height=320)
    fig.add_trace(go.Bar(
        x=hourly_vol.index, y=hourly_vol.values,
        name="News Volume", marker_color=BLUE, opacity=0.5,
    ), secondary_y=False)
    if sent_col:
        hourly_sent = sub.set_index("timestamp_utc")[sent_col].resample("6h").mean()
        fig.add_trace(go.Scatter(
            x=hourly_sent.index, y=hourly_sent.values,
            mode="lines", name="Sentiment", line=dict(color=GREEN, width=2),
        ), secondary_y=True)
        fig.update_yaxes(title_text="Sentiment", secondary_y=True)
    fig.update_yaxes(title_text="Article Count", secondary_y=False)
    return fig


def make_sentiment_ratio_chart(df: pd.DataFrame, asset: str) -> go.Figure:
    if df.empty or "timestamp_utc" not in df.columns:
        return _empty("No data")
    sub = df[df["asset"] == asset].copy() if "asset" in df.columns else df.copy()
    sub = sub.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    if "finbert_label" not in sub.columns:
        return _empty("No FinBERT labels")
    hourly = sub.set_index("timestamp_utc")
    pos = hourly["finbert_label"].resample("6h").apply(lambda x: (x == "positive").mean())
    neg = hourly["finbert_label"].resample("6h").apply(lambda x: (x == "negative").mean())
    neu = 1 - pos - neg
    fig = _fig(title=f"{asset} Sentiment Ratio", height=300)
    fig.add_trace(go.Bar(x=pos.index, y=pos.values, name="Positive", marker_color=GREEN, opacity=0.85))
    fig.add_trace(go.Bar(x=neg.index, y=neg.values, name="Negative", marker_color=RED,   opacity=0.85))
    fig.add_trace(go.Bar(x=neu.index, y=neu.values, name="Neutral",  marker_color=NEUTRAL, opacity=0.6))
    fig.update_layout(barmode="stack")
    return fig


def make_sentiment_heatmap(df: pd.DataFrame) -> go.Figure:
    if df.empty or "timestamp_utc" not in df.columns:
        return _empty("No data")
    sent_col = next((c for c in ["sentiment_score", "gdelt_sentiment_mean"] if c in df.columns), None)
    if sent_col is None or "asset" not in df.columns:
        return _empty("Missing columns")
    df2 = df.dropna(subset=["timestamp_utc", sent_col]).copy()
    df2["day"] = df2["timestamp_utc"].dt.floor("D")
    pivot = df2.groupby(["asset", "day"])[sent_col].mean().unstack("day").fillna(0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=[str(d)[:10] for d in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[[0, RED], [0.5, "#222"], [1, GREEN]],
        zmid=0, zmin=-1, zmax=1,
        hovertemplate="%{y} | %{x}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(**_LAYOUT_BASE, title="Sentiment Heatmap (Daily Mean)", height=260)
    return fig


def make_top_domains(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    if df.empty or "domain" not in df.columns:
        return _empty("No domain data")
    counts = df["domain"].value_counts().head(top_n).reset_index()
    counts.columns = ["domain", "count"]
    fig = px.bar(counts, x="count", y="domain", orientation="h",
                 template=TEMPLATE, color_discrete_sequence=[BLUE],
                 title=f"Top {top_n} News Sources")
    fig.update_layout(**_LAYOUT_BASE, height=360)
    fig.update_yaxes(categoryorder="total ascending")
    return fig


# ── Transformer Forecasts ─────────────────────────────────────────────────────

def make_predictions_timeline(df: pd.DataFrame, asset: str) -> go.Figure:
    sub = df[df["asset"] == asset].sort_values("timestamp_utc") if "asset" in df.columns else df.copy()
    if sub.empty:
        return _empty("No predictions")
    fig = _fig(title=f"{asset} — Predicted vs Actual Log Return (raw transformer output)", height=380)
    if "actual_log_return" in sub.columns:
        fig.add_trace(go.Scatter(
            x=sub["timestamp_utc"], y=sub["actual_log_return"],
            mode="lines", name="Actual", line=dict(color=GREEN, width=1.5),
        ))
    if "predicted_log_return" in sub.columns:
        fig.add_trace(go.Scatter(
            x=sub["timestamp_utc"], y=sub["predicted_log_return"],
            mode="lines", name="Predicted (raw)", line=dict(color=BLUE, width=1.5, dash="dot"),
        ))
    # Secondary: calibrated output (optional — only if column present and non-trivially non-null)
    cal_col = "calibrated_predicted_log_return"
    if cal_col in sub.columns and sub[cal_col].notna().sum() > 0:
        fig.add_trace(go.Scatter(
            x=sub["timestamp_utc"], y=sub[cal_col],
            mode="lines", name="Predicted (calibrated)",
            line=dict(color=GOLD_CLR, width=1.2, dash="dash"), opacity=0.6,
        ))
    if "actual_log_return" in sub.columns and "predicted_log_return" in sub.columns:
        residual = sub["actual_log_return"] - sub["predicted_log_return"]
        fig.add_trace(go.Scatter(
            x=sub["timestamp_utc"], y=residual,
            mode="lines", name="Residual",
            line=dict(color=NEUTRAL, width=0.8), opacity=0.5,
        ))
    fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL, opacity=0.4)
    return fig


def make_bps_chart(df: pd.DataFrame, asset: str) -> go.Figure:
    """Predicted vs actual log return in basis points (×10000) for readability."""
    sub = df[df["asset"] == asset].sort_values("timestamp_utc") if "asset" in df.columns else df.copy()
    if sub.empty or "predicted_log_return_bps" not in sub.columns:
        return _empty("No BPS data (run export first)")
    fig = _fig(title=f"{asset} — Forecast in Basis Points (1 bps = 0.01%)", height=360)
    if "actual_log_return_bps" in sub.columns:
        fig.add_trace(go.Scatter(
            x=sub["timestamp_utc"], y=sub["actual_log_return_bps"],
            mode="lines", name="Actual (bps)",
            line=dict(color=GREEN, width=1.2), opacity=0.7,
        ))
    fig.add_trace(go.Scatter(
        x=sub["timestamp_utc"], y=sub["predicted_log_return_bps"],
        mode="lines", name="Predicted raw (bps)",
        line=dict(color=BLUE, width=1.5, dash="dot"),
    ))
    # Calibrated secondary line
    cal_bps = "calibrated_predicted_log_return_bps"
    if cal_bps in sub.columns and sub[cal_bps].notna().sum() > 0:
        fig.add_trace(go.Scatter(
            x=sub["timestamp_utc"], y=sub[cal_bps],
            mode="lines", name="Predicted calibrated (bps)",
            line=dict(color=GOLD_CLR, width=1.2, dash="dash"), opacity=0.6,
        ))
    fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL, opacity=0.4)
    fig.update_yaxes(title_text="Log Return (bps)")
    return fig


def make_predictions_scatter(df: pd.DataFrame, asset: str) -> go.Figure:
    sub = df[df["asset"] == asset].copy() if "asset" in df.columns else df.copy()
    if sub.empty or "predicted_log_return" not in sub.columns or "actual_log_return" not in sub.columns:
        return _empty("Missing predicted/actual columns")
    sub = sub.dropna(subset=["predicted_log_return", "actual_log_return"])
    lim = max(sub["predicted_log_return"].abs().max(), sub["actual_log_return"].abs().max()) * 1.1
    fig = _fig(title=f"{asset} — Predicted vs Actual Scatter", height=380)
    fig.add_trace(go.Scatter(
        x=sub["predicted_log_return"], y=sub["actual_log_return"],
        mode="markers", marker=dict(color=BLUE, opacity=0.5, size=4), name="Points",
    ))
    fig.add_trace(go.Scatter(
        x=[-lim, lim], y=[-lim, lim], mode="lines",
        line=dict(color=NEUTRAL, dash="dot"), name="y=x",
    ))
    fig.update_xaxes(title_text="Predicted Log Return")
    fig.update_yaxes(title_text="Actual Log Return")
    return fig


def make_signal_strength(df: pd.DataFrame, asset: str) -> go.Figure:
    sub = df[df["asset"] == asset].sort_values("timestamp_utc") if "asset" in df.columns else df.copy()
    if sub.empty or "signal_strength" not in sub.columns:
        return _empty("No signal data")
    fig = _fig(title=f"{asset} — Signal Strength (|Predicted Return|)", height=280)
    fig.add_trace(go.Scatter(
        x=sub["timestamp_utc"], y=sub["signal_strength"].abs(),
        mode="lines", fill="tozeroy", name="Signal",
        line=dict(color=PURPLE, width=1.5), fillcolor="rgba(170,102,255,0.15)",
    ))
    return fig


def make_direction_chart(df: pd.DataFrame, asset: str) -> go.Figure:
    sub = df[df["asset"] == asset].sort_values("timestamp_utc") if "asset" in df.columns else df.copy()
    if sub.empty:
        return _empty("No data")
    fig = _fig(title=f"{asset} — Direction: Predicted vs Actual", height=280)
    pd_col = next((c for c in ["predicted_direction", "transformer_prediction"] if c in sub.columns), None)
    if pd_col:
        fig.add_trace(go.Scatter(
            x=sub["timestamp_utc"], y=sub[pd_col],
            mode="markers+lines", name="Predicted",
            marker=dict(color=BLUE, size=4), line=dict(color=BLUE, width=0.8),
        ))
    if "actual_direction" in sub.columns:
        fig.add_trace(go.Scatter(
            x=sub["timestamp_utc"], y=sub["actual_direction"],
            mode="markers+lines", name="Actual",
            marker=dict(color=GREEN, size=4), line=dict(color=GREEN, width=0.8),
        ))
    fig.update_yaxes(tickvals=[-1, 0, 1], ticktext=["Down", "Neutral", "Up"])
    return fig


def make_price_with_forecasts(prices_df: pd.DataFrame, preds_df: pd.DataFrame, asset: str) -> go.Figure:
    p = prices_df[prices_df["asset"] == asset].sort_values("timestamp_utc") if "asset" in prices_df.columns else prices_df.copy()
    pr = preds_df[preds_df["asset"] == asset].sort_values("timestamp_utc") if "asset" in preds_df.columns else preds_df.copy()
    if p.empty:
        return _empty("No price data")
    fig = _fig(title=f"{asset} Price with Forecast Signals", height=400)
    fig.add_trace(go.Scatter(
        x=p["timestamp_utc"], y=p["close"],
        mode="lines", name="Price", line=dict(color=NEUTRAL, width=1.5),
    ))
    # Prefer viz_direction (threshold=0.00005) so small real predictions show signal.
    # Fall back to predicted_direction (model's own threshold) if viz_direction absent.
    dir_col = next((c for c in ["viz_direction", "predicted_direction"] if c in pr.columns), None)
    if not pr.empty and dir_col is not None:
        merged = pr.merge(p[["timestamp_utc", "close"]], on="timestamp_utc", how="left")
        long_  = merged[merged[dir_col] > 0]
        short_ = merged[merged[dir_col] < 0]
        neut_  = merged[merged[dir_col] == 0]
        if not long_.empty:
            fig.add_trace(go.Scatter(
                x=long_["timestamp_utc"], y=long_["close"],
                mode="markers", name="Long Signal",
                marker=dict(color=GREEN, size=7, symbol="triangle-up"),
            ))
        if not short_.empty:
            fig.add_trace(go.Scatter(
                x=short_["timestamp_utc"], y=short_["close"],
                mode="markers", name="Short Signal",
                marker=dict(color=RED, size=7, symbol="triangle-down"),
            ))
        if not neut_.empty:
            fig.add_trace(go.Scatter(
                x=neut_["timestamp_utc"], y=neut_["close"],
                mode="markers", name="Neutral",
                marker=dict(color=NEUTRAL, size=4, symbol="circle"),
            ))
    return fig


# ── Model Evaluation ──────────────────────────────────────────────────────────

def make_metrics_bar(df: pd.DataFrame, metric: str, title: str | None = None) -> go.Figure:
    if df.empty or metric not in df.columns:
        return _empty(f"No {metric} data")
    sub = df[df["model"] == "Transformer"] if "model" in df.columns else df
    sub = sub.dropna(subset=[metric])
    colors = [GREEN if v >= 0 else RED for v in sub[metric]]
    fig = _fig(title=title or metric, height=300)
    fig.add_trace(go.Bar(
        x=sub["asset"], y=sub[metric], marker_color=colors, name=metric,
        text=sub[metric].round(4), textposition="outside",
    ))
    return fig


def make_error_histogram(preds_df: pd.DataFrame, asset: str) -> go.Figure:
    sub = preds_df[preds_df["asset"] == asset].copy() if "asset" in preds_df.columns else preds_df.copy()
    if sub.empty or "actual_log_return" not in sub.columns or "predicted_log_return" not in sub.columns:
        return _empty("No residual data")
    sub = sub.dropna(subset=["actual_log_return", "predicted_log_return"])
    residuals = sub["actual_log_return"] - sub["predicted_log_return"]
    fig = _fig(title=f"{asset} Prediction Error Distribution", height=300)
    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=50,
        marker_color=BLUE, opacity=0.8, name="Residuals",
    ))
    fig.add_vline(x=0, line_dash="dot", line_color=NEUTRAL)
    fig.update_xaxes(title_text="Actual − Predicted")
    fig.update_yaxes(title_text="Count")
    return fig


def make_rolling_mae(preds_df: pd.DataFrame, asset: str, window: int = 24) -> go.Figure:
    sub = preds_df[preds_df["asset"] == asset].sort_values("timestamp_utc") if "asset" in preds_df.columns else preds_df.sort_values("timestamp_utc")
    if sub.empty or "actual_log_return" not in sub.columns or "predicted_log_return" not in sub.columns:
        return _empty("No prediction data for rolling MAE")
    sub = sub.dropna(subset=["actual_log_return", "predicted_log_return"])
    sub["abs_err"] = (sub["actual_log_return"] - sub["predicted_log_return"]).abs()
    sub["rolling_mae"] = sub["abs_err"].rolling(window).mean()
    fig = _fig(title=f"{asset} Rolling {window}-Bar MAE", height=280)
    fig.add_trace(go.Scatter(
        x=sub["timestamp_utc"], y=sub["rolling_mae"],
        mode="lines", name=f"MAE-{window}", line=dict(color=ORANGE, width=1.5),
    ))
    return fig


def make_multi_metric_bars(metrics_df: pd.DataFrame) -> go.Figure:
    """One grouped bar chart showing MAE, RMSE, DirAcc for each asset."""
    if metrics_df.empty:
        return _empty("No metrics")
    sub = metrics_df[metrics_df["model"] == "Transformer"] if "model" in metrics_df.columns else metrics_df
    fig = _fig(title="Transformer Regression Metrics by Asset", height=360, barmode="group")
    for metric, color in [("MAE", ORANGE), ("RMSE", RED), ("DirAcc", GREEN)]:
        if metric in sub.columns:
            fig.add_trace(go.Bar(
                x=sub["asset"], y=sub[metric], name=metric,
                marker_color=color, opacity=0.85,
            ))
    return fig


# ── PPO Trading Agent ─────────────────────────────────────────────────────────

def _shorten_action(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    if name in ("HOLD", "FLAT"):
        return name.capitalize()
    _d = {"LONG": "L", "SHORT": "S"}
    _s = {"SMALL": "Small", "MEDIUM": "Med", "LARGE": "Large"}
    _r = {"CONSERVATIVE": "Cons", "BALANCED": "Bal", "AGGRESSIVE": "Agg"}
    parts = name.split("_")
    if len(parts) == 2:
        return f"{_d.get(parts[0], parts[0])}-{_s.get(parts[1], parts[1])}"
    if len(parts) == 3:
        return f"{_d.get(parts[0], parts[0])}-{_s.get(parts[1], parts[1])}-{_r.get(parts[2], parts[2])}"
    return name


def make_portfolio_curve(df: pd.DataFrame, asset: str, initial_cap: float = 10_000.0) -> go.Figure:
    sub = df[df["asset"] == asset].sort_values("timestamp_utc") if "asset" in df.columns else df.sort_values("timestamp_utc")
    if sub.empty:
        return _empty("No RL trade data")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                        vertical_spacing=0.05)
    fig.update_layout(**_LAYOUT_BASE, title=f"{asset} PPO Portfolio vs Buy & Hold", height=480)

    if "portfolio_value" in sub.columns:
        pv = sub["portfolio_value"]

        # Portfolio value curve
        fig.add_trace(go.Scatter(
            x=sub["timestamp_utc"], y=pv,
            mode="lines", name="RL Portfolio", line=dict(color=GREEN, width=2),
        ), row=1, col=1)

        # Initial capital horizontal reference
        fig.add_hline(
            y=initial_cap, line_dash="dot", line_color=NEUTRAL, opacity=0.5,
            annotation_text=f"Initial ${initial_cap:,.0f}",
            annotation_position="bottom right",
            row=1, col=1,
        )

        # Buy & Hold benchmark curve
        if "close" in sub.columns and len(sub) >= 2 and sub["close"].iloc[0] > 0:
            bh = sub["close"] / sub["close"].iloc[0] * initial_cap
            fig.add_trace(go.Scatter(
                x=sub["timestamp_utc"], y=bh,
                mode="lines", name="Buy & Hold",
                line=dict(color=NEUTRAL, width=1.5, dash="dot"),
            ), row=1, col=1)

        # Drawdown (%)
        running_max = pv.cummax()
        drawdown_pct = (pv - running_max) / running_max.clip(lower=1e-9) * 100
        fig.add_trace(go.Scatter(
            x=sub["timestamp_utc"], y=drawdown_pct,
            mode="lines", name="Drawdown (%)", fill="tozeroy",
            line=dict(color=RED, width=1), fillcolor="rgba(255,68,85,0.2)",
        ), row=2, col=1)

    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    return fig


def make_action_distribution(df: pd.DataFrame, asset: str) -> go.Figure:
    sub = df[df["asset"] == asset].copy() if "asset" in df.columns else df.copy()
    if sub.empty or "action_name" not in sub.columns:
        return _empty("No action data")
    counts = sub["action_name"].value_counts().reset_index()
    counts.columns = ["action", "count"]
    _action_colors = {
        "HOLD": NEUTRAL, "FLAT": NEUTRAL,
        "LONG_SMALL":  "#66cc88", "LONG_MEDIUM":  "#33bb66", "LONG_LARGE":  GREEN,
        "SHORT_SMALL": "#cc6666", "SHORT_MEDIUM": "#bb3333", "SHORT_LARGE": RED,
    }
    short_labels = [_shorten_action(a) for a in counts["action"]]
    fig = _fig(title=f"{asset} PPO Action Distribution", height=320)
    fig.add_trace(go.Bar(
        x=short_labels, y=counts["count"],
        marker_color=[_action_colors.get(a, BLUE) for a in counts["action"]],
        name="Actions",
        customdata=counts["action"],
        hovertemplate="%{customdata}: %{y:,}<extra></extra>",
    ))
    return fig


_REWARD_POS_COLS = {"reward_pnl", "reward_tp_bonus"}
_REWARD_NEG_COLS = {
    "reward_transaction_cost", "reward_drawdown_penalty",
    "reward_overtrading_penalty", "reward_missed_opportunity_penalty",
    "reward_wrong_side_penalty", "reward_sl_penalty",
}
_REWARD_COLORS = {
    "reward_pnl":                        GREEN,
    "reward_tp_bonus":                   GOLD_CLR,
    "reward_transaction_cost":           ORANGE,
    "reward_drawdown_penalty":           RED,
    "reward_overtrading_penalty":        "#cc4466",
    "reward_missed_opportunity_penalty": NEUTRAL,
    "reward_wrong_side_penalty":         "#dd3355",
    "reward_sl_penalty":                 "#ff6655",
    "reward_total":                      BLUE,
}


def make_reward_components(df: pd.DataFrame, asset: str) -> go.Figure:
    sub = df[df["asset"] == asset].sort_values("timestamp_utc") if "asset" in df.columns else df.sort_values("timestamp_utc")
    if sub.empty:
        return _empty("No reward data")
    reward_cols = [c for c in [
        "reward_pnl", "reward_tp_bonus",
        "reward_transaction_cost", "reward_drawdown_penalty",
        "reward_overtrading_penalty", "reward_missed_opportunity_penalty",
        "reward_wrong_side_penalty", "reward_sl_penalty", "reward_total",
    ] if c in sub.columns]
    if not reward_cols:
        return _empty("No reward component columns found")
    fig = _fig(title=f"{asset} Reward Components per Step  (penalties plot below zero)", height=360)
    for col in reward_cols:
        dash = "dot" if col == "reward_total" else "solid"
        width = 2 if col == "reward_total" else 1.2
        fig.add_trace(go.Scatter(
            x=sub["timestamp_utc"], y=sub[col],
            mode="lines", name=col.replace("reward_", ""),
            line=dict(color=_REWARD_COLORS.get(col, NEUTRAL), width=width, dash=dash),
        ))
    fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL, opacity=0.4)
    return fig


# ── PPO RL — new chart functions ─────────────────────────────────────────────

def _trade_hover_text(rows: pd.DataFrame, event: str) -> list[str]:
    """Build rich hover strings for entry or exit scatter markers."""
    texts = []
    for _, r in rows.iterrows():
        tid  = r.get("trade_id", "?")
        side = r.get("side", "")
        pnl  = r.get("net_pnl")
        ret  = r.get("return_pct")
        risk = r.get("risk_profile_at_entry", "")
        lev  = r.get("leverage_used")
        tp   = r.get("tp_price")
        sl   = r.get("sl_price")
        ex   = r.get("exit_reason", "")
        hb   = r.get("holding_bars")
        lines = [f"<b>Trade #{tid} — {side} {event}</b>"]
        if event == "Entry" and "entry_price" in r.index:
            lines.append(f"Entry: ${r['entry_price']:,.2f}")
            if pd.notna(tp):  lines.append(f"TP: ${tp:,.2f}")
            if pd.notna(sl):  lines.append(f"SL: ${sl:,.2f}")
        else:
            if "exit_price" in r.index: lines.append(f"Exit: ${r['exit_price']:,.2f}")
            if ex:                      lines.append(f"Reason: {ex}")
        if pd.notna(pnl):  lines.append(f"Net PnL: ${pnl:,.2f}")
        if pd.notna(ret):  lines.append(f"Return: {ret:+.4f}%")
        if risk:           lines.append(f"Risk: {risk}")
        if pd.notna(lev):  lines.append(f"Leverage: {lev:.1f}x")
        if pd.notna(hb):   lines.append(f"Hold: {int(hb)} bars")
        texts.append("<br>".join(lines))
    return texts


def make_rl_candlestick_with_trades(
    df_steps: pd.DataFrame,
    df_hist: pd.DataFrame,
    asset: str,
    last_n: int = 30,
) -> go.Figure:
    sub = (
        df_steps[df_steps["asset"] == asset].sort_values("timestamp_utc")
        if "asset" in df_steps.columns
        else df_steps.sort_values("timestamp_utc")
    )
    if sub.empty:
        return _empty("No RL step data")

    has_ohlc = all(c in sub.columns for c in ["open", "high", "low", "close"])
    fig = _fig(title=f"{asset} PPO — Candlestick + Trades (last {last_n})", height=580)

    if has_ohlc:
        fig.add_trace(go.Candlestick(
            x=sub["timestamp_utc"], open=sub["open"], high=sub["high"],
            low=sub["low"], close=sub["close"],
            name="Price", increasing_line_color=GREEN, decreasing_line_color=RED,
        ))
    elif "close" in sub.columns:
        fig.add_trace(go.Scatter(
            x=sub["timestamp_utc"], y=sub["close"],
            mode="lines", name="Close", line=dict(color=NEUTRAL, width=1),
        ))

    if not df_hist.empty and "entry_time" in df_hist.columns:
        hist_a = (
            df_hist[df_hist["asset"] == asset].copy()
            if "asset" in df_hist.columns
            else df_hist.copy()
        )
        hist_a = hist_a.sort_values("entry_time").tail(last_n)

        # Entry markers with rich hover
        for side, color, sym in [("LONG", GREEN, "triangle-up"), ("SHORT", RED, "triangle-down")]:
            s = hist_a[hist_a["side"] == side] if "side" in hist_a.columns else pd.DataFrame()
            if not s.empty and "entry_price" in s.columns:
                fig.add_trace(go.Scatter(
                    x=s["entry_time"], y=s["entry_price"],
                    mode="markers", name=f"{side} Entry",
                    marker=dict(color=color, symbol=sym, size=11),
                    text=_trade_hover_text(s, "Entry"),
                    hoverinfo="text",
                ))

        # Exit markers with rich hover
        if "exit_reason" in hist_a.columns and "exit_price" in hist_a.columns:
            _exit_styles = {
                "TP":             dict(color=GREEN,   symbol="star",          size=11),
                "SL":             dict(color=RED,     symbol="x",             size=11),
                "AGENT_CLOSE":    dict(color=NEUTRAL, symbol="circle",        size=9),
                "AGENT_REVERSE":  dict(color=BLUE,    symbol="diamond",       size=9),
                "AGENT_RESIZE":   dict(color=PURPLE,  symbol="square",        size=9),
                "MAX_HOLD":       dict(color=ORANGE,  symbol="triangle-left", size=9),
                "END_OF_EPISODE": dict(color=NEUTRAL, symbol="circle-open",   size=8),
            }
            for reason, style in _exit_styles.items():
                sx = hist_a[hist_a["exit_reason"] == reason]
                if not sx.empty:
                    fig.add_trace(go.Scatter(
                        x=sx["exit_time"], y=sx["exit_price"],
                        mode="markers", name=f"Exit:{reason}",
                        marker=dict(**style),
                        text=_trade_hover_text(sx, "Exit"),
                        hoverinfo="text",
                    ))

        # TP/SL dashed lines
        if all(c in hist_a.columns for c in ["tp_price", "sl_price", "exit_time"]):
            for _, row in hist_a.iterrows():
                et, xt = row["entry_time"], row["exit_time"]
                if pd.isna(et) or pd.isna(xt):
                    continue
                if pd.notna(row.get("tp_price")) and row["tp_price"] > 0:
                    fig.add_trace(go.Scatter(
                        x=[et, xt], y=[row["tp_price"], row["tp_price"]],
                        mode="lines", line=dict(color=GREEN, dash="dash", width=0.7),
                        showlegend=False, hoverinfo="skip",
                    ))
                if pd.notna(row.get("sl_price")) and row["sl_price"] > 0:
                    fig.add_trace(go.Scatter(
                        x=[et, xt], y=[row["sl_price"], row["sl_price"]],
                        mode="lines", line=dict(color=RED, dash="dash", width=0.7),
                        showlegend=False, hoverinfo="skip",
                    ))

    fig.update_xaxes(rangeslider_visible=False)
    return fig


def make_rl_exposure(df_steps: pd.DataFrame, asset: str) -> go.Figure:
    sub = (
        df_steps[df_steps["asset"] == asset].sort_values("timestamp_utc")
        if "asset" in df_steps.columns
        else df_steps.sort_values("timestamp_utc")
    )
    if sub.empty or "effective_position" not in sub.columns:
        return _empty("No exposure data (effective_position column missing)")
    ep = sub["effective_position"]
    colors = [GREEN if v > 0 else (RED if v < 0 else NEUTRAL) for v in ep]
    fig = _fig(title=f"{asset} Position Exposure", height=280)
    fig.add_trace(go.Bar(
        x=sub["timestamp_utc"], y=ep,
        marker_color=colors, name="Exposure", opacity=0.75,
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL, opacity=0.5)
    fig.update_yaxes(title_text="Effective Position")
    return fig


def make_rl_cumulative_pnl(df_hist: pd.DataFrame, asset: str) -> go.Figure:
    sub = (
        df_hist[df_hist["asset"] == asset].copy()
        if "asset" in df_hist.columns
        else df_hist.copy()
    )
    if sub.empty or "net_pnl" not in sub.columns or "exit_time" not in sub.columns:
        return _empty("No trade history (net_pnl / exit_time missing)")
    sub = sub.sort_values("exit_time").dropna(subset=["net_pnl", "exit_time"])
    sub["cum_pnl"] = sub["net_pnl"].cumsum()
    fig = _fig(title=f"{asset} Cumulative Net PnL (Closed Trades)", height=280)
    fig.add_trace(go.Scatter(
        x=sub["exit_time"], y=sub["cum_pnl"],
        mode="lines", fill="tozeroy", name="Cum PnL",
        line=dict(color=GREEN, width=2), fillcolor="rgba(0,255,136,0.10)",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL, opacity=0.5)
    fig.update_yaxes(title_text="Cumulative PnL ($)")
    return fig


def make_rl_reward_cumulative(df_steps: pd.DataFrame, asset: str) -> go.Figure:
    sub = (
        df_steps[df_steps["asset"] == asset].sort_values("timestamp_utc")
        if "asset" in df_steps.columns
        else df_steps.sort_values("timestamp_utc")
    )
    if sub.empty:
        return _empty("No reward data")
    component_cols = [c for c in [
        "reward_pnl", "reward_tp_bonus", "reward_transaction_cost",
        "reward_drawdown_penalty", "reward_overtrading_penalty",
        "reward_missed_opportunity_penalty", "reward_wrong_side_penalty", "reward_sl_penalty",
    ] if c in sub.columns]
    if not component_cols:
        return _empty("No reward component columns found")
    _colors = {
        "reward_pnl":                        GREEN,
        "reward_tp_bonus":                   GOLD_CLR,
        "reward_transaction_cost":           ORANGE,
        "reward_drawdown_penalty":           RED,
        "reward_overtrading_penalty":        "#cc4466",
        "reward_missed_opportunity_penalty": NEUTRAL,
        "reward_wrong_side_penalty":         "#dd3355",
        "reward_sl_penalty":                 "#ff6655",
    }
    fig = _fig(
        title=f"{asset} Cumulative Reward Components  (penalties accumulate below zero)",
        height=380,
    )
    for col in component_cols:
        fig.add_trace(go.Scatter(
            x=sub["timestamp_utc"], y=sub[col].cumsum(),
            mode="lines", name=col.replace("reward_", ""),
            line=dict(color=_colors.get(col, NEUTRAL), width=1.5),
        ))
    if "reward_total" in sub.columns:
        fig.add_trace(go.Scatter(
            x=sub["timestamp_utc"], y=sub["reward_total"].cumsum(),
            mode="lines", name="total",
            line=dict(color=BLUE, width=2, dash="dot"),
        ))
    fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL, opacity=0.4)
    return fig


def make_rl_reward_avg_bar(df_hist: pd.DataFrame, asset: str) -> go.Figure:
    sub = (
        df_hist[df_hist["asset"] == asset].copy()
        if "asset" in df_hist.columns
        else df_hist.copy()
    )
    if sub.empty:
        return _empty("No trade history data")
    total_cols = [c for c in [
        "reward_pnl_total", "reward_tp_bonus_total", "reward_transaction_cost_total",
        "reward_drawdown_penalty_total", "reward_overtrading_penalty_total",
        "reward_missed_opportunity_penalty_total", "reward_wrong_side_penalty_total",
        "reward_sl_penalty_total",
    ] if c in sub.columns]
    if not total_cols:
        return _empty("No reward_*_total columns in trade history")
    means  = sub[total_cols].mean()
    labels = [c.replace("reward_", "").replace("_total", "") for c in total_cols]
    colors = [GREEN if v >= 0 else RED for v in means.values]
    fig = _fig(title=f"{asset} Avg Reward Component per Trade", height=320)
    fig.add_trace(go.Bar(
        x=labels, y=means.values,
        marker_color=colors, name="Avg Reward",
        text=[f"{v:.4f}" for v in means.values], textposition="outside",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL, opacity=0.5)
    return fig


def make_rl_pnl_by_category(
    df_hist: pd.DataFrame, asset: str, category: str = "exit_reason"
) -> go.Figure:
    sub = (
        df_hist[df_hist["asset"] == asset].copy()
        if "asset" in df_hist.columns
        else df_hist.copy()
    )
    if sub.empty or "net_pnl" not in sub.columns or category not in sub.columns:
        return _empty(f"No data for {category}")
    grp = sub.groupby(category)["net_pnl"].agg(["mean", "count"]).reset_index()
    colors = [GREEN if v >= 0 else RED for v in grp["mean"]]
    fig = _fig(title=f"{asset} Avg Net PnL by {category}", height=300)
    fig.add_trace(go.Bar(
        x=grp[category], y=grp["mean"],
        marker_color=colors, name="Avg PnL",
        text=[f"n={int(n)}" for n in grp["count"]], textposition="outside",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL, opacity=0.5)
    fig.update_yaxes(title_text="Mean Net PnL ($)")
    return fig


def make_rl_winrate_by_category(
    df_hist: pd.DataFrame, asset: str, category: str = "side"
) -> go.Figure:
    sub = (
        df_hist[df_hist["asset"] == asset].copy()
        if "asset" in df_hist.columns
        else df_hist.copy()
    )
    if sub.empty or "net_pnl" not in sub.columns or category not in sub.columns:
        return _empty(f"No data for {category}")
    sub["win"] = (sub["net_pnl"] > 0).astype(float)
    grp = sub.groupby(category)["win"].agg(["mean", "count"]).reset_index()
    colors = [GREEN if v >= 0.5 else RED for v in grp["mean"]]
    fig = _fig(title=f"{asset} Win Rate by {category}", height=280)
    fig.add_trace(go.Bar(
        x=grp[category], y=grp["mean"] * 100,
        marker_color=colors, name="Win Rate",
        text=[f"{v:.1f}%  n={int(n)}" for v, n in zip(grp["mean"] * 100, grp["count"])],
        textposition="outside",
    ))
    fig.add_hline(y=50, line_dash="dot", line_color=NEUTRAL, opacity=0.5)
    fig.update_yaxes(title_text="Win Rate (%)", range=[0, 110])
    return fig


def make_rl_category_dist(
    df: pd.DataFrame, asset: str, col: str, title: str | None = None
) -> go.Figure:
    sub = (
        df[df["asset"] == asset].copy()
        if "asset" in df.columns
        else df.copy()
    )
    if sub.empty or col not in sub.columns:
        return _empty(f"No {col} data")
    counts = sub[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    fig = _fig(title=title or f"{asset} {col}", height=260)
    fig.add_trace(go.Bar(
        x=counts[col], y=counts["count"],
        marker_color=BLUE, name=col, opacity=0.85,
        text=counts["count"], textposition="outside",
    ))
    return fig


# ── Model Comparison ──────────────────────────────────────────────────────────

def make_grouped_model_bars(df: pd.DataFrame, metric: str) -> go.Figure:
    if df.empty or metric not in df.columns or "model" not in df.columns:
        return _empty(f"No {metric} data")
    sub = df.dropna(subset=[metric])
    fig = _fig(title=f"{metric} by Model and Asset", height=340, barmode="group")
    model_colors = {"Transformer": BLUE, "Ridge": ORANGE, "PPO_RL": GREEN, "XGBoost": PURPLE}
    for model in sub["model"].unique():
        m = sub[sub["model"] == model]
        fig.add_trace(go.Bar(
            x=m["asset"], y=m[metric],
            name=model, marker_color=model_colors.get(model, NEUTRAL), opacity=0.85,
        ))
    return fig


# ── Pipeline Health ───────────────────────────────────────────────────────────

def make_data_sparsity_chart(df: pd.DataFrame, asset: str) -> go.Figure:
    """Show news availability vs missing by day."""
    sub = df[df["asset"] == asset].copy() if "asset" in df.columns else df.copy()
    if sub.empty or "timestamp_utc" not in sub.columns:
        return _empty("No data")
    sub["day"] = sub["timestamp_utc"].dt.floor("D")
    daily = sub.groupby("day").size().rename("articles")
    fig = _fig(title=f"{asset} Daily Article Count", height=240)
    fig.add_trace(go.Bar(
        x=daily.index, y=daily.values,
        marker_color=BLUE, opacity=0.8, name="Articles",
    ))
    return fig


# ── Utilities ─────────────────────────────────────────────────────────────────

def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **_LAYOUT_BASE,
        annotations=[dict(
            text=f"⚠ {msg}", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color=NEUTRAL, size=14),
        )],
        height=200,
    )
    return fig
