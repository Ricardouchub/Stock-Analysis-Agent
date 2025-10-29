from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from prefect import flow, get_run_logger, task

from scripts.ingest_prices import fetch_and_persist_prices
from scripts.build_features import build_features_for_ticker


@task
def ingest_task(ticker: str, start: Optional[str], end: Optional[str]) -> None:
    logger = get_run_logger()
    logger.info("Ingesting prices for %s", ticker)
    fetch_and_persist_prices(
        ticker=ticker,
        start=start or (datetime.utcnow() - timedelta(days=365)).date().isoformat(),
        end=end or datetime.utcnow().date().isoformat(),
    )


@task
def feature_task(ticker: str) -> None:
    logger = get_run_logger()
    logger.info("Building features for %s", ticker)
    build_features_for_ticker(ticker)


@flow(name="daily-price-refresh")
def daily_price_refresh(ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> None:
    ingest_task(ticker, start, end)
    feature_task(ticker)


if __name__ == "__main__":
    daily_price_refresh(ticker="AAPL")
