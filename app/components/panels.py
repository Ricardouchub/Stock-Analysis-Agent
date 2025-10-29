from __future__ import annotations
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


def _format_percent(value: Optional[float]) -> str:
    return f"{value:.2%}" if isinstance(value, (int, float)) else "NA"


def _format_number(value: Optional[float], digits: int = 2) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "NA"


def _risk_badge(label: str, score: Optional[float], thresholds: Dict[str, float], invert: bool = False) -> str:
    if not isinstance(score, (int, float)):
        return f"{label}: <span style='background-color:#6c757d;color:white;padding:0.1rem 0.4rem;border-radius:0.4rem;'>NA</span>"
    value = -score if invert else score
    if value <= thresholds["green"]:
        color = "#2ca02c"
        text = "Low"
    elif value <= thresholds["amber"]:
        color = "#ff7f0e"
        text = "Medium"
    else:
        color = "#d62728"
        text = "High"
    return (
        f"{label}: <span style='background-color:{color};color:white;padding:0.1rem 0.4rem;"
        f"border-radius:0.4rem;'>{text}</span>"
    )


def render_risk_panel(risk: Dict[str, Any]) -> None:
    if not risk:
        return
    st.subheader("Risk snapshot")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Volatility (30d)", _format_percent(risk.get("volatility_30d")))
    with col2:
        st.metric("Beta", _format_number(risk.get("beta")))
    with col3:
        st.metric("Drawdown", _format_percent(risk.get("current_drawdown")))

    badges = [
        _risk_badge("Volatility", risk.get("volatility_30d"), {"green": 0.20, "amber": 0.35}),
        _risk_badge("Beta", risk.get("beta"), {"green": 0.9, "amber": 1.2}),
        _risk_badge("Drawdown", risk.get("current_drawdown"), {"green": 0.10, "amber": 0.20}, invert=True),
    ]
    st.markdown(" &nbsp; ".join(badges), unsafe_allow_html=True)

    percentile = risk.get("drawdown_percentile")
    if isinstance(percentile, (int, float)):
        st.caption(f"Current drawdown percentile: {percentile:.0%}")


def render_sector_comparison(sector: Dict[str, Any]) -> None:
    if not sector:
        return
    metrics = sector.get("metrics") or []
    if not metrics:
        return
    st.subheader("Sector comparison")
    sector_name = sector.get("sector")
    if sector_name:
        st.caption(f"Sector: {sector_name}")
    rows: List[Dict[str, Any]] = []
    for metric in metrics:
        rows.append(
            {
                "Metric": metric.get("name"),
                "Company": metric.get("company_value"),
                "Sector Median": metric.get("sector_median"),
                "Units": metric.get("units"),
            }
        )
    frame = pd.DataFrame(rows)
    st.dataframe(frame, use_container_width=True)
