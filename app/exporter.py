from __future__ import annotations

from datetime import datetime
from io import BytesIO
from textwrap import wrap
from typing import Any, Dict, List

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def _draw_wrapped_text(c: canvas.Canvas, text: str, x: float, y: float, width: float, leading: float, font: str, size: int) -> float:
    c.setFont(font, size)
    lines = wrap(text, width=80)
    current_y = y
    for line in lines:
        if current_y < 60:
            c.showPage()
            current_y = letter[1] - 50
            c.setFont(font, size)
        c.drawString(x, current_y, line)
        current_y -= leading
    return current_y


def build_report_pdf(artifacts: Dict[str, Any], followups: List[Dict[str, Any]]) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 45
    y = height - margin

    def add_section(title: str, body: List[str]) -> None:
        nonlocal y
        if y < 100:
            c.showPage()
            y = height - margin
        c.setFont("Helvetica-Bold", 13)
        c.drawString(margin, y, title)
        y -= 18
        for line in body:
            y = _draw_wrapped_text(c, line, margin, y, width - 2 * margin, 14, "Helvetica", 11) - 6

    snapshot = artifacts.get("snapshot") or {}
    ticker = snapshot.get("ticker") or artifacts.get("ticker") or "N/A"
    generated = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, f"Decision-Support Report — {ticker}")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Generated: {generated}")
    y -= 25

    if snapshot:
        lines = [
            f"Last close: {snapshot.get('last_close')} (as_of {snapshot.get('as_of')}, source: {snapshot.get('source')})",
            f"EMA50: {snapshot.get('ema50')} | EMA200: {snapshot.get('ema200')} | RSI14: {snapshot.get('rsi14')}",
        ]
        add_section("Price snapshot", lines)

    backtest = artifacts.get("backtest") or {}
    if backtest:
        lines = [
            f"Window: {backtest.get('start')} → {backtest.get('end')} ({backtest.get('window_days')} trading days)",
            f"CAGR: {backtest.get('cagr'):.2%} | Sharpe: {backtest.get('sharpe'):.2f} | Max DD: {backtest.get('max_drawdown'):.2%}",
            f"Trades: {backtest.get('trades')} | Costs (bps): {backtest.get('costs_bps')}",
        ]
        add_section("EMA 50/200 backtest", lines)

    risk = artifacts.get("risk") or {}
    if risk:
        lines = [
            f"Volatility (30d): {risk.get('volatility_30d')}",
            f"Beta: {risk.get('beta')} | Drawdown: {risk.get('current_drawdown')} | Percentile: {risk.get('drawdown_percentile')}",
        ]
        add_section("Risk profile", lines)

    sector = artifacts.get("sector") or {}
    metrics = sector.get("metrics") or []
    if metrics:
        sector_lines = [f"Sector: {sector.get('sector')}"]
        for metric in metrics:
            sector_lines.append(
                f"{metric.get('name')}: company {metric.get('company_value')} vs sector {metric.get('sector_median')} ({metric.get('units') or 'units'})"
            )
        add_section("Sector comparison", sector_lines)

    news_summary = artifacts.get("news_summary")
    if news_summary:
        add_section("News summary", [news_summary])

    answer = artifacts.get("answer")
    if answer:
        add_section("Primary analysis", answer.splitlines())

    if followups:
        for idx, convo in enumerate(followups, start=1):
            blocks = [
                f"Question {idx}: {convo.get('question')}",
                f"Answer: {convo.get('answer')}",
            ]
            add_section(f"Follow-up {idx}", blocks)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()
