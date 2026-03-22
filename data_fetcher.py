"""
data_fetcher.py - Fetch real market data and compute technical indicators.

Usage:
    from data_fetcher import fetch_data
    df = fetch_data("RELIANCE.NS", period="5y")   # NSE
    df = fetch_data("AAPL", period="2y")           # NASDAQ
"""

import os
import hashlib
import pandas as pd
import numpy as np
import yfinance as yf

from config import Config


def _cache_path(ticker: str, period: str) -> str:
    """Deterministic cache file path based on ticker + period."""
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    key = hashlib.md5(f"{ticker}_{period}".encode()).hexdigest()[:12]
    return os.path.join(Config.CACHE_DIR, f"{ticker.replace('.', '_')}_{key}.csv")


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators used as model features.
    Returns a new DataFrame with NaN rows dropped.
    """
    df = df.copy()

    # Simple Moving Averages
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_30"] = df["Close"].rolling(30).mean()

    # Exponential Moving Averages
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # RSI (14-period)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.finfo(float).eps)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + bb_std * 2
    df["BB_Lower"] = df["BB_Middle"] - bb_std * 2

    # Price change & volatility
    df["Price_Change"] = df["Close"].pct_change()
    df["Volatility"] = df["Price_Change"].rolling(10).std()

    # Volume features
    df["Volume_SMA"] = df["Volume"].rolling(10).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"].replace(0, np.finfo(float).eps)

    return df.dropna()


def fetch_data(
    ticker: str | None = None,
    period: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance and add technical indicators.

    Parameters
    ----------
    ticker    : Yahoo Finance symbol (e.g. "RELIANCE.NS", "AAPL")
    period    : lookback period ("1y", "2y", "5y", "max")
    use_cache : if True, reuse a local CSV instead of re-downloading

    Returns
    -------
    pd.DataFrame with columns matching Config.FEATURE_COLS + extras
    """
    ticker = ticker or Config.DEFAULT_TICKER
    period = period or Config.DATA_PERIOD

    cache = _cache_path(ticker, period)

    if use_cache and os.path.exists(cache):
        print(f"[DATA] Loading cached data from {cache}")
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        print(f"[DATA] Downloading {ticker} ({period}) from Yahoo Finance ...")
        raw = yf.download(ticker, period=period, progress=False)
        if raw.empty:
            raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")

        # Handle multi-level columns from yfinance
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)

        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df = add_technical_indicators(df)
        df.to_csv(cache)
        print(f"[DATA] Cached {len(df)} rows to {cache}")

    print(f"[DATA] {ticker}: {len(df)} trading days  |  {df.index[0].date()} to {df.index[-1].date()}")
    return df


def fetch_fundamentals(ticker: str) -> dict:
    """
    Fetch fundamental data for a single ticker via yfinance.

    Returns a dict with safe defaults for missing fields:
        pe_ratio, roe, debt_to_equity, earnings_growth,
        revenue_growth, free_cash_flow, market_cap,
        fifty_two_week_high, fifty_two_week_low,
        sector, industry, name
    """
    t = yf.Ticker(ticker)
    try:
        info = t.info
    except Exception:
        info = {}

    def _safe(key, default=None):
        v = info.get(key, default)
        return v if v is not None else default

    return {
        "ticker": ticker,
        "name": _safe("shortName", ticker),
        "sector": _safe("sector", "N/A"),
        "industry": _safe("industry", "N/A"),
        "market_cap": _safe("marketCap", 0),
        "pe_ratio": _safe("trailingPE", 0.0),
        "forward_pe": _safe("forwardPE", 0.0),
        "roe": _safe("returnOnEquity", 0.0),
        "debt_to_equity": _safe("debtToEquity", 0.0),
        "earnings_growth": _safe("earningsGrowth", 0.0),
        "revenue_growth": _safe("revenueGrowth", 0.0),
        "free_cash_flow": _safe("freeCashflow", 0),
        "fifty_two_week_high": _safe("fiftyTwoWeekHigh", 0.0),
        "fifty_two_week_low": _safe("fiftyTwoWeekLow", 0.0),
        "current_price": _safe("currentPrice", _safe("regularMarketPrice", 0.0)),
        "dividend_yield": _safe("dividendYield", 0.0),
        "beta": _safe("beta", 1.0),
    }


if __name__ == "__main__":
    # Quick test
    df = fetch_data()
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nLast 3 rows:\n{df.tail(3)}")

    print("\n--- Fundamentals ---")
    fund = fetch_fundamentals(Config.DEFAULT_TICKER)
    for k, v in fund.items():
        print(f"  {k}: {v}")
