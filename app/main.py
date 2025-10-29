from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

from app.agent_graph import agent_app, followup_chat
from app.components import (
    render_metrics_card,
    render_price_chart,
    render_risk_panel,
    render_sector_comparison,
)
from app.exporter import build_report_pdf
from app.tools import ToolExecutionError

LOG_DIR = Path(os.getenv("LOG_DIR", "./data/logs")).resolve()
LOG_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

st.set_page_config(page_title="Decision-Support Stock Agent", layout="wide")

if "last_artifacts" not in st.session_state:
    st.session_state["last_artifacts"] = None
if "followup_history" not in st.session_state:
    st.session_state["followup_history"] = []


def _render_errors(errors: list[Dict[str, Any]]) -> None:
    st.warning("One or more tools returned errors.")
    for item in errors:
        st.write(f"{item['tool']}: {item['message']}")


def _render_news(news_items: list[Dict[str, Any]]) -> None:
    st.subheader("Recent headlines")
    for item in news_items:
        published = item.get("published")
        if isinstance(published, str):
            ts = published
        elif isinstance(published, (datetime,)):
            ts = published.strftime("%Y-%m-%d %H:%M UTC")
        else:
            ts = "unknown time"
        st.markdown(f"- [{item.get('title')}]({item.get('url')}) — {ts} ({item.get('source')})")


def _render_backtest(backtest: Dict[str, Any]) -> None:
    st.subheader("Backtest: EMA 50 / EMA 200 crossover")
    st.json(backtest)


def _render_answer(answer: str) -> None:
    st.markdown("---")
    st.markdown(answer)


def _log_interaction(query: str, artifacts: Dict[str, Any]) -> None:
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "artifacts": artifacts,
    }
    log_path = LOG_DIR / "interactions.jsonl"
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, default=str) + "\n")


def main() -> None:
    st.sidebar.header("Session")
    ticker_input = st.sidebar.text_input(
        "Ticker (uppercase)",
        value="AAPL",
        help="The agent substitutes this symbol when your question uses {T}.",
    )
    costs_bps = st.sidebar.slider(
        "Costs (bps per trade)",
        min_value=0,
        max_value=50,
        value=5,
        help="Per-trade transaction cost expressed in basis points (1 bp = 0.01%). Applied to every entry and exit in backtests.",
    )
    st.sidebar.caption(
        "Set a realistic cost assumption so the EMA crossover backtest reflects commissions and slippage."
    )
    show_artifacts = st.sidebar.checkbox(
        "Show tool artifacts (debug)",
        value=False,
        help="Displays the raw JSON returned by each tool (snapshot, news, backtest). Useful for verifying sources, as_of timestamps, and cache paths.",
    )

    st.title("Decision-Support Stock Agent")
    st.caption("For personal research only. Not investment advice.")

    default_question = "What changed for {T} last week? Backtest the 50/200 cross."
    query = st.text_area("Ask a question", value=default_question, height=100)

    if st.button("Run analysis") and query:
        state = {
            "query": query.replace("{T}", ticker_input.strip().upper()),
            "ticker": None,
            "artifacts": {"costs_bps": costs_bps},
        }
        result_state = agent_app.invoke(state)
        artifacts = result_state.get("artifacts", {})
        _log_interaction(state["query"], artifacts)
        st.session_state["last_artifacts"] = artifacts
        st.session_state["followup_history"] = []

        st.rerun()

    artifacts = st.session_state.get("last_artifacts") or {}
    if artifacts:
        snapshot = artifacts.get("snapshot")
        if snapshot:
            render_metrics_card(snapshot)

        chart_data = artifacts.get("chart")
        if chart_data:
            render_price_chart(chart_data)

        news_items = artifacts.get("news") or []
        if news_items:
            _render_news(news_items)

        news_summary = artifacts.get("news_summary")
        if news_summary:
            st.subheader("News summary")
            st.markdown(news_summary)

        backtest = artifacts.get("backtest")
        if backtest:
            _render_backtest(backtest)

        risk = artifacts.get("risk")
        if risk:
            render_risk_panel(risk)

        sector = artifacts.get("sector")
        if sector:
            render_sector_comparison(sector)

        if artifacts.get("errors"):
            _render_errors(artifacts["errors"])

        answer = artifacts.get("answer")
        if answer:
            _render_answer(answer)

        llm_logs = artifacts.get("llm_logs") or []
        if llm_logs:
            st.subheader("LLM usage")
            st.json(llm_logs)

        pdf_bytes = build_report_pdf(artifacts, st.session_state.get("followup_history", []))
        file_label = f"{(snapshot or {}).get('ticker', ticker_input.strip().upper()) or 'report'}_analysis.pdf"
        st.download_button(
            "Export report (PDF)",
            data=pdf_bytes,
            file_name=file_label,
            mime="application/pdf",
        )

        if show_artifacts:
            st.subheader("Tool artifacts (debug)")
            st.caption("Raw JSON returned by the tools; useful for verifying sources, timestamps, and cache paths.")
            st.code(json.dumps(artifacts, indent=2, default=str))

        st.subheader("Follow-up chat")
        history: List[Dict[str, Any]] = st.session_state.get("followup_history", [])
        for entry in history:
            st.markdown(f"**You:** {entry['question']}")
            st.markdown(entry["answer"])
            meta_parts = []
            latency = entry.get("latency_sec")
            if latency is not None:
                meta_parts.append(f"latency {latency:.2f}s")
            usage = entry.get("token_usage") or {}
            in_tokens = usage.get("input_tokens")
            out_tokens = usage.get("output_tokens")
            token_parts = []
            if isinstance(in_tokens, int):
                token_parts.append(f"in {in_tokens}")
            if isinstance(out_tokens, int):
                token_parts.append(f"out {out_tokens}")
            if token_parts:
                meta_parts.append("tokens " + "/".join(token_parts))
            if meta_parts:
                st.caption(", ".join(meta_parts))
            st.markdown("---")

        with st.form("followup_form", clear_on_submit=True):
            followup_query = st.text_area(
                "Ask a follow-up about these results",
                height=100,
                placeholder="e.g., Explain how the drawdown percentile ties into the sector comparison.",
            )
            followup_submit = st.form_submit_button("Send follow-up")

        if followup_submit and followup_query.strip():
            try:
                result = followup_chat(followup_query.strip(), artifacts)
                entry = {
                    "question": followup_query.strip(),
                    "answer": result.get("content", ""),
                    "latency_sec": result.get("latency_sec"),
                    "token_usage": result.get("token_usage"),
                }
                st.session_state.setdefault("followup_history", []).append(entry)
                st.rerun()
            except ToolExecutionError as exc:
                st.error(f"Follow-up chat failed: {exc}")

if __name__ == "__main__":
    main()
