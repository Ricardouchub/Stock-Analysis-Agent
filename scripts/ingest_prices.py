from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from polygon import RESTClient

load_dotenv()

CACHE_DIR = Path(os.getenv("CACHE_DIR", "./data")).resolve()


@dataclass
class PriceChunk:
    ticker: str
    df: pd.DataFrame
    source: Literal["polygon", "alphavantage", "yfinance"]
    as_of: datetime


def _normalize_bars(rows: list[dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame = frame.rename(
        columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame["adj_close"] = frame.get("adj_close", frame["close"])
    return frame[["timestamp", "open", "high", "low", "close", "adj_close", "volume"]]


def fetch_polygon(ticker: str, start: str, end: str) -> PriceChunk:
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY is not configured.")

    client = RESTClient(api_key)
    bars = client.get_aggs(
        ticker=ticker,
        multiplier=1,
        timespan="day",
        from_=start,
        to=end,
        adjusted=True,
        sort="asc",
        limit=50000,
    )
    rows = [bar.__dict__ for bar in bars]
    if not rows:
        raise RuntimeError("Polygon returned no data.")

    df = _normalize_bars(rows)
    return PriceChunk(ticker=ticker, df=df, source="polygon", as_of=datetime.utcnow())


def fetch_alphavantage(ticker: str) -> PriceChunk:
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY is not configured.")

    url = "https://www.alphavantage.co/query"
    params = {"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": ticker, "apikey": api_key, "outputsize": "full"}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("Time Series (Daily)")
    if not data:
        raise RuntimeError("Alpha Vantage returned no daily data.")

    records = []
    for ts, values in data.items():
        record = {
            "timestamp": pd.to_datetime(ts, utc=True),
            "open": float(values["1. open"]),
            "high": float(values["2. high"]),
            "low": float(values["3. low"]),
            "close": float(values["4. close"]),
            "adj_close": float(values["5. adjusted close"]),
            "volume": float(values["6. volume"]),
        }
        records.append(record)
    df = pd.DataFrame(records).sort_values("timestamp")
    return PriceChunk(ticker=ticker, df=df, source="alphavantage", as_of=datetime.utcnow())


def persist_parquet(chunk: PriceChunk, partition_year: Optional[int] = None) -> Path:
    ticker_dir = CACHE_DIR / "prices" / chunk.ticker.upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)
    year = partition_year or int(chunk.df["timestamp"].dt.year.min())
    path = ticker_dir / f"{year}.parquet"
    chunk.df.to_parquet(path, index=False)
    return path


def fetch_and_persist_prices(ticker: str, start: str, end: str) -> Path:
    ticker = ticker.upper()
    try:
        chunk = fetch_polygon(ticker, start, end)
    except Exception as polygon_exc:
        try:
            chunk = fetch_alphavantage(ticker)
        except Exception as alpha_exc:
            raise RuntimeError(
                f"Failed to download data from Polygon ({polygon_exc}) and Alpha Vantage ({alpha_exc})."
            ) from alpha_exc

    path = persist_parquet(chunk)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download price history into the local cache.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--start", required=False, default="2015-01-01")
    parser.add_argument("--end", required=False, default=datetime.utcnow().date().isoformat())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = fetch_and_persist_prices(args.ticker, args.start, args.end)
    print(f"Stored prices at {path}")


if __name__ == "__main__":
    main()
