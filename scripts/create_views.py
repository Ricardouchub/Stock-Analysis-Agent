from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DATABASE = Path("data/duckdb/warehouse.duckdb")
DEFAULT_PRICES_GLOB = "data/prices/*/*.parquet"


def create_prices_view(database: Path, glob_pattern: str) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(database))
    try:
        sanitized = glob_pattern.replace("'", "''")
        conn.execute(
            f"CREATE OR REPLACE VIEW prices AS SELECT * FROM read_parquet('{sanitized}');"
        )
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or refresh DuckDB views.")
    parser.add_argument(
        "--database",
        default=str(DEFAULT_DATABASE),
        help="Path to the DuckDB database file.",
    )
    parser.add_argument(
        "--prices-glob",
        default=DEFAULT_PRICES_GLOB,
        help="Glob pattern of parquet files for the prices view.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_prices_view(Path(args.database), args.prices_glob)
    print(f"Created view 'prices' in {args.database} for {args.prices_glob}")


if __name__ == "__main__":
    main()
