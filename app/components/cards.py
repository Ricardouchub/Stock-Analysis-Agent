from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def render_metrics_card(snapshot: Dict[str, Any]) -> None:
    """Render core price metrics."""
    st.metric("Last Close", f"{snapshot.get('last_close', float('nan')):.2f}")
    cols = st.columns(3)
    with cols[0]:
        ema50 = snapshot.get("ema50")
        st.metric("EMA 50", f"{ema50:.2f}" if ema50 is not None else "NA")
    with cols[1]:
        ema200 = snapshot.get("ema200")
        st.metric("EMA 200", f"{ema200:.2f}" if ema200 is not None else "NA")
    with cols[2]:
        rsi = snapshot.get("rsi14")
        st.metric("RSI 14", f"{rsi:.1f}" if rsi is not None else "NA")
    dist = snapshot.get("dist_52w_high")
    if dist is not None:
        st.caption(f"Distance to 52W high: {dist * 100:.2f}%")

