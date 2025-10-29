from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = Path(os.getenv("CACHE_DIR", "./data")).resolve()


def _load_price_history(ticker: str) -> pd.DataFrame:
    price_dir = CACHE_DIR / "prices" / ticker.upper()
    if not price_dir.exists():
        raise FileNotFoundError(f"No cached prices for {ticker}. Run scripts/ingest_prices.py first.")
    files = sorted(price_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found for {ticker} in {price_dir}.")
    frames = [pd.read_parquet(path) for path in files]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame.set_index("timestamp", inplace=True)
    frame["ema50"] = ta.ema(frame["close"], length=50)
    frame["ema200"] = ta.ema(frame["close"], length=200)
    frame["rsi14"] = ta.rsi(frame["close"], length=14)
    rolling_max = frame["close"].rolling(window=252, min_periods=50).max()
    frame["dist_52w_high"] = (frame["close"] - rolling_max) / rolling_max
    frame = frame.reset_index()
    return frame


def persist_features(ticker: str, features: pd.DataFrame) -> Path:
    feature_dir = CACHE_DIR / "features" / ticker.upper()
    feature_dir.mkdir(parents=True, exist_ok=True)
    path = feature_dir / "daily.parquet"
    features.to_parquet(path, index=False)
    return path


def build_features_for_ticker(ticker: str) -> Path:
    df = _load_price_history(ticker)
    feats = compute_indicators(df)
    return persist_features(ticker, feats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute technical indicators for a cached ticker.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = build_features_for_ticker(args.ticker)
    print(f"Stored feature set at {path}")


if __name__ == "__main__":
    main()
