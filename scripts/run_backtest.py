from __future__ import annotations

import argparse
import json

from app.tools import backtest_sma_cross


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple EMA crossover backtest.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--costs_bps", type=float, default=5.0, help="Per-trade cost in basis points")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = backtest_sma_cross(args.ticker, args.costs_bps)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
