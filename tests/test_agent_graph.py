from __future__ import annotations

import json

import pytest

from app.agent_graph import AgentState, _extract_ticker, router_node


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What changed for AAPL last week?", "AAPL"),
        ("Backtest the 50/200 cross for nflx please", "NFLX"),
        ("Is msft overbought?", "MSFT"),
        ("tell me about spy and qqq", "SPY"),
        ("Any risks for pricing team?", None),
    ],
)
def test_extract_ticker(query: str, expected: str | None) -> None:
    assert _extract_ticker(query) == expected


def test_router_sets_flags() -> None:
    state: AgentState = {"query": "Backtest and show news for AAPL", "ticker": None, "artifacts": {}}
    updated = router_node(state)
    assert updated["ticker"] == "AAPL"
    assert updated["artifacts"]["needs_backtest"] is True
    assert updated["artifacts"]["needs_news"] is True
