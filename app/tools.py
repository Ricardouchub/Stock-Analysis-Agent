from __future__ import annotations

import os
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import pandas as pd
import requests
import vectorbt as vbt
from pandas import Timestamp
from pydantic import BaseModel, Field, ValidationError
import feedparser


class ToolExecutionError(RuntimeError):
    """Raised when a tool fails to execute as expected."""


CACHE_DIR = Path(os.getenv("CACHE_DIR", "./data")).resolve()
WAREHOUSE_DUCKDB = Path(
    os.getenv("WAREHOUSE_DUCKDB", "./data/duckdb/warehouse.duckdb")
).resolve()
DEFAULT_NEWS_DAYS = 7


class PriceSnapshot(BaseModel):
    ticker: str
    as_of: datetime
    source: str
    last_close: float = Field(..., description="Most recent close price")
    last_close_ts: datetime = Field(..., description="Timestamp of the last close")
    volume: Optional[float] = Field(None, description="Latest trading volume")
    ema50: Optional[float] = None
    ema200: Optional[float] = None
    rsi14: Optional[float] = None
    dist_52w_high: Optional[float] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class NewsItem(BaseModel):
    title: str
    url: str
    published: datetime
    source: str
    ticker: str


class RiskSnapshot(BaseModel):
    ticker: str
    as_of: datetime
    source: str
    volatility_30d: Optional[float] = Field(None, description="Annualized realized volatility based on 30d returns.")
    beta: Optional[float] = Field(None, description="Latest beta from fundamentals.")
    current_drawdown: Optional[float] = Field(None, description="Current drawdown from trailing peak.")
    drawdown_percentile: Optional[float] = Field(None, description="Percentile rank of current drawdown vs history.")


class SectorMetric(BaseModel):
    name: str
    company_value: Optional[float]
    sector_median: Optional[float]
    units: Optional[str] = None
    source: str = "finnhub"


class SectorComparison(BaseModel):
    ticker: str
    sector: Optional[str]
    as_of: datetime
    metrics: List[SectorMetric]


class ChartData(BaseModel):
    ticker: str
    as_of: datetime
    source: str
    points: List[Dict[str, Any]]
    events: List[Dict[str, Any]] = Field(default_factory=list)


def _select_latest_parquet(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise ToolExecutionError(f"Expected cache path does not exist: {path}")
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise ToolExecutionError(f"No parquet files found in {path}")
    return files[-1]


def _ensure_timestamp(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.tz_convert("UTC") if series.dt.tz is not None else series.dt.tz_localize("UTC")
    return pd.to_datetime(series, utc=True)


def _load_features_frame(ticker: str) -> pd.DataFrame:
    features_path = CACHE_DIR / "features" / ticker.upper() / "daily.parquet"
    if features_path.exists():
        df = pd.read_parquet(features_path)
        df["timestamp"] = _ensure_timestamp(df["timestamp"])
        return df.sort_values("timestamp")
    # fallback to raw prices
    price_dir = CACHE_DIR / "prices" / ticker.upper()
    latest_file = _select_latest_parquet(price_dir)
    df = pd.read_parquet(latest_file)
    df["timestamp"] = _ensure_timestamp(df["timestamp"])
    if "adj_close" not in df and "close" in df:
        df["adj_close"] = df["close"]
    return df.sort_values("timestamp")


def _load_chart_frame(ticker: str, periods: int = 260) -> pd.DataFrame:
    df = _load_features_frame(ticker)
    if df.empty:
        raise ToolExecutionError(f"No cached features for {ticker}")
    trimmed = df.tail(periods).copy()
    return trimmed


def get_price_snapshot(ticker: str) -> PriceSnapshot:
    ticker = ticker.upper()
    try:
        df = _load_features_frame(ticker)
    except Exception as exc:  # noqa: BLE001 - propagate as tool error
        raise ToolExecutionError(f"Unable to load cached data for {ticker}: {exc}") from exc

    if df.empty:
        raise ToolExecutionError(f"Cached dataframe is empty for {ticker}")

    record = df.iloc[-1].copy()
    last_close = float(record.get("close") or record.get("adj_close"))
    timestamp = record.get("timestamp")
    if isinstance(timestamp, Timestamp):
        last_ts = timestamp.to_pydatetime()
    else:
        last_ts = pd.to_datetime(timestamp, utc=True).to_pydatetime()

    snapshot = PriceSnapshot(
        ticker=ticker,
        as_of=datetime.utcnow(),
        source="parquet/features" if "ema50" in record else "parquet/prices",
        last_close=last_close,
        last_close_ts=last_ts,
        volume=float(record.get("volume")) if record.get("volume") is not None else None,
        ema50=float(record.get("ema50")) if record.get("ema50") is not None else None,
        ema200=float(record.get("ema200")) if record.get("ema200") is not None else None,
        rsi14=float(record.get("rsi14")) if record.get("rsi14") is not None else None,
        dist_52w_high=float(record.get("dist_52w_high"))
        if record.get("dist_52w_high") is not None
        else None,
        meta={
            "row_count": int(len(df)),
            "cache_path": str(CACHE_DIR),
            "warehouse_exists": WAREHOUSE_DUCKDB.exists(),
        },
    )
    return snapshot


def _is_domain_allowed(url: str) -> bool:
    whitelist = os.getenv("SOURCE_WHITELIST")
    if not whitelist:
        return True
    domains = {item.strip().lower() for item in whitelist.split(",") if item.strip()}
    host = urlparse(url).netloc.lower()
    return any(host.endswith(domain) for domain in domains)


def fetch_ir_news(ticker: str, days: int = DEFAULT_NEWS_DAYS) -> List[NewsItem]:
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise ToolExecutionError("FINNHUB_API_KEY is not configured.")

    ticker = ticker.upper()
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)

    try:
        response = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={
                "symbol": ticker,
                "from": start_date.isoformat(),
                "to": end_date.isoformat(),
                "token": api_key,
            },
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException as exc:  # noqa: PERF203 - fine for clarity
        raise ToolExecutionError(f"Finnhub request failed: {exc}") from exc

    payload = response.json()
    if not isinstance(payload, list):
        raise ToolExecutionError("Unexpected Finnhub response payload.")

    results: List[NewsItem] = []
    for entry in payload:
        try:
            url = entry.get("url", "")
            if not url or not _is_domain_allowed(url):
                continue
            published = datetime.utcfromtimestamp(int(entry.get("datetime", 0)))
            item = NewsItem(
                title=entry.get("headline", "").strip(),
                url=url,
                published=published,
                source=entry.get("source", "finnhub"),
                ticker=ticker,
            )
        except (ValueError, TypeError, ValidationError):
            continue
        results.append(item)
    return results


def _load_feature_series(df: pd.DataFrame, column: str) -> pd.Series:
    series = df[column] if column in df.columns else pd.Series(dtype=float)
    return series.astype(float, errors="ignore")


def backtest_sma_cross(ticker: str, costs_bps: float = 5.0) -> Dict[str, Any]:
    ticker = ticker.upper()
    try:
        df = _load_features_frame(ticker)
    except Exception as exc:
        raise ToolExecutionError(f"Unable to load features for {ticker}: {exc}") from exc

    if df.empty:
        raise ToolExecutionError(f"No data available to backtest {ticker}.")

    close = _load_feature_series(df, "close")

    def _ensure_ema(series: pd.Series, span: int) -> pd.Series:
        if series.empty:
            return series
        return series.ewm(span=span, adjust=False).mean()

    ema50 = _load_feature_series(df, "ema50")
    ema200 = _load_feature_series(df, "ema200")

    if ema50.empty or ema200.empty:
        ema50 = _ensure_ema(close, 50)
        ema200 = _ensure_ema(close, 200)

    if ema50.empty or ema200.empty:
        raise ToolExecutionError(
            "EMA columns are required for the SMA cross backtest. "
            "Run the feature builder first."
        )

    windows = [756, None]  # approx 3 years, then full history
    last_portfolio = None
    last_slice = close
    trades = 0

    for window in windows:
        if window and len(close) > window:
            close_slice = close.tail(window)
            ema50_slice = ema50.tail(window)
            ema200_slice = ema200.tail(window)
        else:
            close_slice = close
            ema50_slice = ema50
            ema200_slice = ema200

        entries = (ema50_slice > ema200_slice) & (ema50_slice.shift(1) <= ema200_slice.shift(1))
        exits = (ema50_slice < ema200_slice) & (ema50_slice.shift(1) >= ema200_slice.shift(1))

        portfolio = vbt.Portfolio.from_signals(
            close_slice,
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            fees=costs_bps / 10000.0,
            slippage=0.0005,
            freq="1D",
        )
        trades = int(portfolio.trades.count())
        last_portfolio = portfolio
        last_slice = close_slice
        if trades > 0 or window is None:
            break

    stats = {
        "ticker": ticker,
        "start": last_slice.index.min().date().isoformat() if hasattr(last_slice.index.min(), "date") else str(last_slice.index.min()),
        "end": last_slice.index.max().date().isoformat() if hasattr(last_slice.index.max(), "date") else str(last_slice.index.max()),
        "cagr": float(last_portfolio.annualized_return()) if last_portfolio else 0.0,
        "sharpe": float(last_portfolio.sharpe_ratio()) if last_portfolio else 0.0,
        "max_drawdown": float(last_portfolio.max_drawdown()) if last_portfolio else 0.0,
        "trades": trades,
        "costs_bps": float(costs_bps),
        "window_days": len(last_slice),
    }
    return stats


def fetch_ir_rss(ticker: str, max_items: int = 5) -> List[NewsItem]:
    ticker = ticker.upper()
    env_key = f"IR_RSS_{ticker}_FEED"
    url = os.getenv(env_key)
    if not url:
        return []
    try:
        feed = feedparser.parse(url)
    except Exception as exc:  # noqa: BLE001
        raise ToolExecutionError(f"RSS fetch failed for {ticker}: {exc}") from exc
    items: List[NewsItem] = []
    for entry in feed.entries[:max_items]:
        link = entry.get("link")
        if not link or not _is_domain_allowed(link):
            continue
        published = entry.get("published_parsed")
        if published:
            published_dt = datetime(*published[:6], tzinfo=None)
        else:
            published_dt = datetime.utcnow()
        items.append(
            NewsItem(
                title=entry.get("title", "").strip(),
                url=link,
                published=published_dt,
                source=urlparse(link).netloc or "rss",
                ticker=ticker,
            )
        )
    return items


def _get_returns(close: pd.Series) -> pd.Series:
    returns = close.pct_change().dropna()
    return returns.replace([float("inf"), float("-inf")], pd.NA).dropna()


def _fetch_finnhub_json(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise ToolExecutionError("FINNHUB_API_KEY is not configured.")
    response = requests.get(
        f"https://finnhub.io/api/v1/{endpoint}",
        params={**params, "token": api_key},
        timeout=20,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise ToolExecutionError(f"Finnhub request failed: {exc}") from exc
    return response.json()


def get_risk_snapshot(ticker: str) -> RiskSnapshot:
    ticker = ticker.upper()
    df = _load_features_frame(ticker)
    if df.empty:
        raise ToolExecutionError(f"No cached data to compute risk snapshot for {ticker}")
    close = _load_feature_series(df, "close")
    as_of = datetime.utcnow()

    volatility_30d: Optional[float] = None
    drawdown_percentile: Optional[float] = None
    current_drawdown: Optional[float] = None

    if len(close) >= 40:
        returns = _get_returns(close)[-60:]
        if not returns.empty:
            volatility_30d = float(returns[-30:].std(ddof=0) * math.sqrt(252)) if len(returns) >= 30 else None

    rolling_peak = close.cummax()
    drawdown_series = (close / rolling_peak) - 1
    if not drawdown_series.empty:
        current_drawdown = float(drawdown_series.iloc[-1])
        drawdown_percentile = float((drawdown_series <= current_drawdown).mean()) if len(drawdown_series) > 10 else None

    beta: Optional[float] = None
    try:
        metrics = _fetch_finnhub_json("stock/metric", {"symbol": ticker, "metric": "all"})
        beta_val = metrics.get("metric", {}).get("beta")
        beta = float(beta_val) if beta_val is not None else None
    except ToolExecutionError:
        beta = None

    return RiskSnapshot(
        ticker=ticker,
        as_of=as_of,
        source="parquet+finnhub",
        volatility_30d=volatility_30d,
        beta=beta,
        current_drawdown=current_drawdown,
        drawdown_percentile=drawdown_percentile,
    )


def get_sector_comparison(ticker: str) -> SectorComparison:
    ticker = ticker.upper()
    profile: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    try:
        profile = _fetch_finnhub_json("stock/profile2", {"symbol": ticker}) or {}
    except ToolExecutionError:
        profile = {}
    try:
        metrics = _fetch_finnhub_json("stock/metric", {"symbol": ticker, "metric": "all"}) or {}
    except ToolExecutionError:
        metrics = {}

    metric_data = metrics.get("metric", {}) if isinstance(metrics, dict) else {}

    def _safe_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    candidates = [
        (
            "P/E (TTM)",
            _safe_float(metric_data.get("peTTM")),
            _safe_float(metric_data.get("peNormalizedAnnual", metric_data.get("sectorMedianPe"))),
            "x",
        ),
        (
            "EV/EBITDA",
            _safe_float(metric_data.get("enterpriseValueOverEBITDA")),
            _safe_float(metric_data.get("sectorMedianEVToEBITDA")),
            "x",
        ),
        (
            "Net Margin",
            _safe_float(metric_data.get("netMargin")),
            _safe_float(metric_data.get("sectorMedianNetMargin")),
            "%",
        ),
        (
            "Free Cash Flow Yield",
            _safe_float(metric_data.get("freeCashFlowPerShareTTM")),
            _safe_float(metric_data.get("sectorMedianFreeCashFlowPerShareTTM")),
            None,
        ),
    ]

    metric_models = [
        SectorMetric(name=name, company_value=company, sector_median=sector, units=units)
        for name, company, sector, units in candidates
        if company is not None or sector is not None
    ]

    return SectorComparison(
        ticker=ticker,
        sector=profile.get("finnhubIndustry") or profile.get("sector"),
        as_of=datetime.utcnow(),
        metrics=metric_models,
    )


def load_chart_data(ticker: str, periods: int = 260) -> ChartData:
    frame = _load_chart_frame(ticker, periods=periods)
    frame = frame.copy()
    frame["support"] = frame["close"].rolling(window=20, min_periods=5).min()
    frame["resistance"] = frame["close"].rolling(window=20, min_periods=5).max()
    frame["support"] = frame["support"].where(frame["support"].notna())
    frame["resistance"] = frame["resistance"].where(frame["resistance"].notna())
    events: List[Dict[str, Any]] = []
    if {"support", "resistance"}.issubset(frame.columns):
        recent = frame.tail(120)
        recent_support = (recent["close"] <= recent["support"] * 1.001) & recent["support"].notna()
        recent_resistance = (recent["close"] >= recent["resistance"] * 0.999) & recent["resistance"].notna()
        for _, row in recent.loc[recent_support].iterrows():
            events.append(
                {
                    "type": "support_touch",
                    "timestamp": row["timestamp"].isoformat(),
                    "price": row["close"],
                }
            )
        for _, row in recent.loc[recent_resistance].iterrows():
            events.append(
                {
                    "type": "resistance_touch",
                    "timestamp": row["timestamp"].isoformat(),
                    "price": row["close"],
                }
            )

    rows = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "timestamp": row["timestamp"].isoformat() if hasattr(row["timestamp"], "isoformat") else row["timestamp"],
                "close": row.get("close"),
                "ema50": row.get("ema50"),
                "ema200": row.get("ema200"),
                "support": row.get("support"),
                "resistance": row.get("resistance"),
            }
        )
    return ChartData(
        ticker=ticker,
        as_of=datetime.utcnow(),
        source="parquet/features",
        points=rows,
        events=events,
    )


__all__ = [
    "PriceSnapshot",
    "NewsItem",
    "ToolExecutionError",
    "get_price_snapshot",
    "fetch_ir_news",
    "backtest_sma_cross",
    "fetch_ir_rss",
    "get_risk_snapshot",
    "RiskSnapshot",
    "SectorComparison",
    "get_sector_comparison",
    "load_chart_data",
    "ChartData",
]
