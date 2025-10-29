from __future__ import annotations

import datetime
import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.tools import (
    ToolExecutionError,
    backtest_sma_cross,
    fetch_ir_news,
    fetch_ir_rss,
    get_price_snapshot,
    get_risk_snapshot,
    get_sector_comparison,
    load_chart_data,
)


class AgentState(TypedDict):
    query: str
    ticker: str | None
    artifacts: Dict[str, Any]


TICKER_RE = re.compile(r"\b[A-Za-z]{1,5}\b")
STOP_TOKENS = {
    "NEWS",
    "LLM",
    "HTTP",
    "JSON",
    "ETF",
}
COMMON_WORDS = {
    "A",
    "ANY",
    "AN",
    "AND",
    "ARE",
    "ASK",
    "ABOUT",
    "BACKTEST",
    "CAN",
    "CROSS",
    "DO",
    "FOR",
    "FROM",
    "HOW",
    "IS",
    "IT",
    "ME",
    "OF",
    "ON",
    "PLEASE",
    "SHOW",
    "TELL",
    "THE",
    "TEAM",
    "THIS",
    "WHAT",
    "WEEK",
    "WITH",
    "LAST",
    "NEWS",
    "RISK",
    "RISKS",
    "STOCK",
    "LOOK",
    "NOTE",
}
DEFAULT_MODEL = os.getenv("OPENAI_MODEL") or os.getenv("DEEPSEEK_MODEL") or "gpt-4o-mini"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")


def _extract_ticker(text: str) -> str | None:
    candidates = TICKER_RE.findall(text)

    def _valid(symbol: str) -> bool:
        return (
            1 < len(symbol) <= 5
            and symbol.isalpha()
            and symbol not in STOP_TOKENS
            and symbol not in COMMON_WORDS
        )

    for token in candidates:
        symbol = token.upper()
        if not token.isupper():
            continue
        if _valid(symbol):
            return symbol

    for token in candidates:
        symbol = token.upper()
        if _valid(symbol):
            return symbol

    return None


def router_node(state: AgentState) -> AgentState:
    query = state["query"]
    lowered = query.lower()
    artifacts = state.setdefault("artifacts", {})
    if any(key in lowered for key in ("news", "what changed", "headline")):
        artifacts["needs_news"] = True
    if "backtest" in lowered or "back-test" in lowered:
        artifacts["needs_backtest"] = True
    if "risk" in lowered:
        artifacts["needs_risk"] = True
    state["ticker"] = state.get("ticker") or _extract_ticker(query)
    return state


def _record_error(artifacts: Dict[str, Any], tool: str, message: str) -> None:
    errors = artifacts.setdefault("errors", [])
    errors.append({"tool": tool, "message": message})


def tools_node(state: AgentState) -> AgentState:
    ticker = state.get("ticker")
    artifacts = state.setdefault("artifacts", {})
    if not ticker:
        return state

    try:
        snapshot = get_price_snapshot(ticker)
        snapshot_dict = snapshot.model_dump()
        artifacts["snapshot"] = snapshot_dict
        last_close_ts = snapshot.last_close_ts
        if last_close_ts.tzinfo is None:
            last_close_ts = last_close_ts.replace(tzinfo=datetime.timezone.utc)
        now_utc = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
        age_days = (now_utc - last_close_ts.astimezone(datetime.timezone.utc)).total_seconds() / 86400
        artifacts["needs_price_refresh"] = age_days > 2
    except ToolExecutionError as exc:
        _record_error(artifacts, "get_price_snapshot", str(exc))

    try:
        chart = load_chart_data(ticker)
        artifacts["chart"] = chart.model_dump()
    except ToolExecutionError as exc:
        _record_error(artifacts, "load_chart_data", str(exc))

    if artifacts.get("needs_news"):
        news_payload: List[Dict[str, Any]] = []
        try:
            news_items = fetch_ir_news(ticker)
            news_payload.extend(item.model_dump() for item in news_items)
        except ToolExecutionError as exc:
            _record_error(artifacts, "fetch_ir_news", str(exc))
        try:
            rss_items = fetch_ir_rss(ticker)
            news_payload.extend(item.model_dump() for item in rss_items)
        except ToolExecutionError as exc:
            _record_error(artifacts, "fetch_ir_rss", str(exc))

        deduped: Dict[str, Dict[str, Any]] = {}
        for item in news_payload:
            key = item.get("url")
            if key and key not in deduped:
                deduped[key] = item
        artifacts["news"] = list(deduped.values())[:10]

    if artifacts.get("needs_backtest"):
        try:
            bt = backtest_sma_cross(
                ticker=ticker,
                costs_bps=float(artifacts.get("costs_bps", 5)),
            )
            artifacts["backtest"] = bt
        except ToolExecutionError as exc:
            _record_error(artifacts, "backtest_sma_cross", str(exc))

    try:
        risk = get_risk_snapshot(ticker)
        artifacts["risk"] = risk.model_dump()
    except ToolExecutionError as exc:
        _record_error(artifacts, "get_risk_snapshot", str(exc))

    try:
        sector = get_sector_comparison(ticker)
        artifacts["sector"] = sector.model_dump()
    except ToolExecutionError as exc:
        _record_error(artifacts, "get_sector_comparison", str(exc))

    return state


def _load_system_prompt() -> str:
    prompt_path = Path(__file__).parent / "prompts" / "system.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return (
        "You are a cautious financial analysis assistant. "
        "Only reference numbers that appear in the tool outputs. "
        "Never provide investment advice. Offer pros, cons, and risks."
    )


@lru_cache(maxsize=1)
def _get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return ChatOpenAI(model=DEFAULT_MODEL, api_key=api_key, temperature=0.1)

    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key:
        return ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=deepseek_key,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.1,
        )

    raise RuntimeError(
        "No LLM credentials configured. Set OPENAI_API_KEY or DEEPSEEK_API_KEY."
    )


def _format_artifacts(artifacts: Dict[str, Any]) -> str:
    def _default(obj: Any) -> Any:
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return obj

    return json.dumps(artifacts, default=_default, indent=2, sort_keys=True)


def _fallback_reasoning(state: AgentState) -> str:
    artifacts = state.get("artifacts", {})
    ticker = state.get("ticker") or "the ticker"
    parts: List[str] = []
    snapshot = artifacts.get("snapshot")
    if snapshot:
        last_close = snapshot.get("last_close")
        as_of = snapshot.get("as_of")
        source = snapshot.get("source")
        parts.append(
            f"- Latest close for {ticker}: {last_close} (source: {source}, as_of: {as_of})"
        )
        ema50 = snapshot.get("ema50")
        ema200 = snapshot.get("ema200")
        rsi14 = snapshot.get("rsi14")
        if ema50 is not None and ema200 is not None:
            parts.append(
                f"- EMA trend: 50d {ema50} vs 200d {ema200} (source: {source})"
            )
        if rsi14 is not None:
            parts.append(f"- RSI(14): {rsi14} (source: {source})")
    news_items = artifacts.get("news") or []
    if news_items:
        parts.append("- Recent headlines:")
        for item in news_items[:5]:
            parts.append(f"  * {item.get('title')} ({item.get('source')})")
    news_summary = artifacts.get("news_summary")
    if news_summary:
        parts.append("- News summary:")
        for line in news_summary.splitlines():
            parts.append(f"  {line}")
    backtest = artifacts.get("backtest")
    if backtest:
        parts.append(
            "- Backtest summary: CAGR {cagr}, Sharpe {sharpe}, Max DD {max_drawdown}".format(
                **backtest
            )
        )
    risk = artifacts.get("risk")
    if risk:
        def _fmt_pct(value: Any, digits: int = 2) -> str:
            return f"{value:.{digits}%}" if isinstance(value, (int, float)) else "NA"

        def _fmt_num(value: Any, digits: int = 2) -> str:
            return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "NA"

        parts.append(
            "- Risk snapshot: vol30 {vol} beta {beta} drawdown {dd} (percentile {pct})".format(
                vol=_fmt_pct(risk.get("volatility_30d")),
                beta=_fmt_num(risk.get("beta")),
                dd=_fmt_pct(risk.get("current_drawdown")),
                pct=_fmt_pct(risk.get("drawdown_percentile"), digits=0),
            )
        )
    sector = artifacts.get("sector")
    if sector and sector.get("metrics"):
        sector_lines = []
        for metric in sector["metrics"][:3]:
            sector_lines.append(
                f"{metric['name']}: company {metric.get('company_value')} vs sector {metric.get('sector_median')}"
            )
        parts.append("- Sector check: " + "; ".join(sector_lines))
    summary_bits: List[str] = []
    if snapshot:
        close_clause = f"{ticker} last closed at {last_close}"
        trend_bits = []
        if ema50 is not None and ema200 is not None:
            trend_bits.append(f"EMA50 {ema50:.2f} vs EMA200 {ema200:.2f}")
        if rsi14 is not None:
            trend_bits.append(f"RSI14 {rsi14:.1f}")
        if trend_bits:
            close_clause += " (" + ", ".join(trend_bits) + ")"
        summary_bits.append(close_clause)
    if backtest:
        cagr = backtest.get("cagr")
        sharpe = backtest.get("sharpe")
        if isinstance(cagr, (int, float)) and isinstance(sharpe, (int, float)):
            summary_bits.append(
                f"EMA crossover backtest netted {cagr:.2%} CAGR with Sharpe {sharpe:.2f}"
            )
        else:
            summary_bits.append("EMA crossover backtest ran; review stats above.")
    if risk:
        vol = risk.get("volatility_30d")
        dd_val = risk.get("current_drawdown")
        if isinstance(vol, (int, float)) or isinstance(dd_val, (int, float)):
            vol_str = f"{vol:.2%}" if isinstance(vol, (int, float)) else "NA"
            dd_str = f"{dd_val:.2%}" if isinstance(dd_val, (int, float)) else "NA"
            summary_bits.append(
                f"Realized volatility sits near {vol_str} with current drawdown {dd_str}"
            )
    if summary_bits:
        parts.append("Summary: " + "; ".join(summary_bits))
    if not parts:
        parts.append("No cached data available. Consider running ingestion scripts first.")
    parts.append("This is not investment advice.")
    return "\n".join(parts)


def reason_node(state: AgentState) -> AgentState:
    artifacts = state.setdefault("artifacts", {})
    formatted_artifacts = _format_artifacts(artifacts)
    system_prompt = _load_system_prompt()

    try:
        llm = _get_llm()
    except RuntimeError:
        artifacts["answer"] = _fallback_reasoning(state)
        return state

    start = time.perf_counter()
    messages: List[BaseMessage | Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"User query:\n{state['query']}\n\n"
                f"Artifacts (as JSON):\n{formatted_artifacts}\n"
                "Respond with concise bullets listing pros, cons, and risks. "
                "If data is missing, state what should be fetched next. "
                "Only suggest refreshing price data if the artifacts include `\"needs_price_refresh\": true`. "
                "After the bullets, include a short paragraph (2-3 sentences) that interprets the metrics, risk snapshot, sector comparison, and backtests shown in the UI."
            ),
        },
    ]
    response = llm.invoke(messages)
    latency = time.perf_counter() - start
    usage = getattr(response, "response_metadata", {}).get("token_usage", {})
    artifacts["answer"] = response.content
    artifacts.setdefault("llm_logs", []).append(
        {
            "step": "reason",
            "latency_sec": latency,
            "token_usage": usage,
        }
    )
    return state


def summarize_news_node(state: AgentState) -> AgentState:
    artifacts = state.setdefault("artifacts", {})
    news_items = artifacts.get("news") or []
    if not news_items:
        artifacts["news_summary"] = "No new filings or IR announcements in the last 7 days."
        return state
    try:
        llm = _get_llm()
    except RuntimeError:
        return state
    snippets = []
    for item in news_items[:6]:
        snippets.append(f"{item.get('title')} ({item.get('source')}, {item.get('published')})")
    prompt = (
        "Summarize the following investor relations and news headlines into three factual bullet points. "
        "Mention dates and sources when possible.\n\n"
        + "\n".join(f"- {s}" for s in snippets)
    )
    start = time.perf_counter()
    response = llm.invoke(
        [
            {
                "role": "system",
                "content": "You write concise news summaries for investors. No speculation.",
            },
            {"role": "user", "content": prompt},
        ]
    )
    latency = time.perf_counter() - start
    usage = getattr(response, "response_metadata", {}).get("token_usage", {})
    artifacts["news_summary"] = response.content
    artifacts.setdefault("llm_logs", []).append(
        {
            "step": "news_summary",
            "latency_sec": latency,
            "token_usage": usage,
        }
    )
    return state


graph = StateGraph(AgentState)
graph.add_node("route", router_node)
graph.add_node("tools", tools_node)
graph.add_node("news_summary", summarize_news_node)
graph.add_node("reason", reason_node)
graph.add_edge(START, "route")
graph.add_edge("route", "tools")
graph.add_edge("tools", "news_summary")
graph.add_edge("news_summary", "reason")
graph.add_edge("reason", END)

agent_app = graph.compile()


def followup_chat(message: str, artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a follow-up LLM response referencing existing artifacts."""
    try:
        llm = _get_llm()
    except RuntimeError as exc:
        raise ToolExecutionError(str(exc)) from exc

    formatted_artifacts = _format_artifacts(artifacts)
    system_prompt = (
        "You are continuing a discussion about previously generated analysis. "
        "Reference only numbers that appear in the provided artifacts. "
        "Offer clarifications, deeper insight, or suggested next steps. "
        "Do not repeat the full report unless necessary."
    )
    start = time.perf_counter()
    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Follow-up question: {message}\n\n"
                    f"Artifacts:\n{formatted_artifacts}\n"
                    "Respond conversationally while grounding all claims in the artifacts."
                ),
            },
        ]
    )
    latency = time.perf_counter() - start
    usage = getattr(response, "response_metadata", {}).get("token_usage", {})
    return {
        "content": response.content,
        "latency_sec": latency,
        "token_usage": usage,
    }


__all__ = [
    "AgentState",
    "agent_app",
    "router_node",
    "tools_node",
    "reason_node",
    "followup_chat",
]
