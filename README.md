# Decision-Support Stock Analysis Agent

Personal research environment for exploring equities with a guardrailed AI assistant. The agent reasons over locally cached data, computes technical indicators, highlights vetted headlines, and runs lightweight rule-based backtests. It never issues investment directives. Requires Python 3.12+.

## Features
- Streamlit front end backed by a LangGraph agent for tool orchestration (supports OpenAI or DeepSeek APIs).
- Local-first cache: prices, features, news (Finnhub + optional IR RSS), and DuckDB warehouse.
- Indicator library via `pandas-ta` with validation using Great Expectations.
- VectorBT-powered EMA crossover backtest with formatted date range and configurable trading costs.
- Risk dashboard (realized volatility, beta, drawdown percentile) plus sector-median valuation comparison.
- Plotly trend chart with EMA crossover markers and LLM summaries with token/latency logging panel.

## Quickstart
1. Copy the environment template and fill in credentials (supply either `OPENAI_API_KEY` or `DEEPSEEK_API_KEY`):
   ```bash
   cp .env.example .env
   ```
2. Install dependencies using `uv` (or `pip`):
   ```bash
   uv sync
   ```
3. Ingest price history and build features for a ticker:
   ```bash
   uv run python scripts/ingest_prices.py --ticker AAPL --start 2015-01-01 --end 2025-10-28
   uv run python scripts/build_features.py --ticker AAPL
   uv run python scripts/create_views.py
   ```
4. Launch the Streamlit UI:
   ```bash
   uv run streamlit run app/main.py
   ```

## Repository Layout
```
app/                 # Agent graph, tools, Streamlit UI, prompts, components
scripts/             # CLI helpers for ingesting data, building features, backtests
workflows/           # Prefect flows for scheduled refreshes
expectations/        # Great Expectations suites
data/                # Local cache (ignored by git)
.env.example         # Required API keys and configuration
pyproject.toml       # Dependency definitions
```

## Guardrails
- Numbers quoted by the assistant must originate from cache or API outputs and include `(source, as_of)`.
- Responses present pros, cons, and risks; the agent never says "buy" or "sell".
- Cache-first design: if data is missing or stale, the UI surfaces actionable follow-ups.

## Disclaimer
This project is for educational use only. Market data may be delayed or incorrect. Backtests are not predictive. Taxes, fees, and slippage matter. You are responsible for your decisions.
