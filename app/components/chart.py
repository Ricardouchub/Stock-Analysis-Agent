from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _prepare_dataframe(points: List[Dict[str, Any]]) -> pd.DataFrame:
    if not points:
        return pd.DataFrame()
    df = pd.DataFrame(points)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    numeric_cols = ["close", "ema50", "ema200", "support", "resistance"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp")


def render_price_chart(chart: Dict[str, Any]) -> None:
    points = chart.get("points") if chart else None
    df = _prepare_dataframe(points or [])
    if df.empty:
        st.info("No chart data available. Ingest prices and rebuild features first.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["close"],
            mode="lines",
            name="Close",
            line=dict(color="#1f77b4"),
            hovertemplate="Close %{y:.2f}<extra></extra>",
        )
    )
    if "ema50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["ema50"],
                mode="lines",
                name="EMA 50",
                line=dict(color="#ff7f0e", dash="dash"),
                hovertemplate="EMA50 %{y:.2f}<extra></extra>",
            )
        )
    if "ema200" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["ema200"],
                mode="lines",
                name="EMA 200",
                line=dict(color="#2ca02c", dash="dot"),
                hovertemplate="EMA200 %{y:.2f}<extra></extra>",
            )
        )

    if {"support", "resistance"}.issubset(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["resistance"],
                mode="lines",
                name="Resistance",
                line=dict(color="rgba(214, 39, 40, 0.4)", width=0),
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["support"],
                mode="lines",
                name="Support",
                line=dict(color="rgba(31, 119, 180, 0.4)", width=0),
                fill="tonexty",
                hoverinfo="skip",
                showlegend=True,
            )
        )

    if {"ema50", "ema200"}.issubset(df.columns):
        cross_up = (df["ema50"] > df["ema200"]) & (df["ema50"].shift(1) <= df["ema200"].shift(1))
        cross_down = (df["ema50"] < df["ema200"]) & (df["ema50"].shift(1) >= df["ema200"].shift(1))

        fig.add_trace(
            go.Scatter(
                x=df.loc[cross_up, "timestamp"],
                y=df.loc[cross_up, "close"],
                mode="markers",
                name="Bullish cross",
                marker=dict(color="#17becf", size=9, symbol="triangle-up"),
                hovertemplate="Bullish cross<br>%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.loc[cross_down, "timestamp"],
                y=df.loc[cross_down, "close"],
                mode="markers",
                name="Bearish cross",
                marker=dict(color="#d62728", size=9, symbol="triangle-down"),
                hovertemplate="Bearish cross<br>%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Date",
        yaxis_title="Price",
    )

    events = chart.get("events") or []
    if events:
        for event in events:
            ts = pd.to_datetime(event.get("timestamp"), utc=True, errors="coerce")
            price = event.get("price")
            if pd.isna(ts) or price is None:
                continue
            color = "#8c564b" if event.get("type") == "support_touch" else "#9467bd"
            name = "Support touch" if event.get("type") == "support_touch" else "Resistance touch"
            fig.add_trace(
                go.Scatter(
                    x=[ts],
                    y=[price],
                    mode="markers",
                    name=name,
                    marker=dict(color=color, size=8, symbol="diamond"),
                    hovertemplate=f"{name}<br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}<extra></extra>",
                    showlegend=False,
                )
            )

    st.plotly_chart(fig, use_container_width=True)
