"""
Global Stock Exchange Data Module.

Provides market overview data for major stock exchanges worldwide,
including top gainers/losers, market indices, and exchange metadata.

Sources:
- Finnhub for real-time data (when API key is available)
- Yahoo Finance as universal fallback
- Alpha Vantage for US market movers
"""

from __future__ import annotations

import concurrent.futures
import os
import time

import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()

_EXCHANGE_CACHE: dict[str, tuple[float, object]] = {}
CACHE_TTL = 300  # 5 minutes

# ── Exchange Registry ────────────────────────────────────────────────────────
EXCHANGES = {
    "NSE": {
        "name": "National Stock Exchange",
        "country": "India",
        "currency": "INR",
        "flag": "IN",
        "timezone": "Asia/Kolkata",
        "index": "^NSEI",
        "index_name": "NIFTY 50",
        "suffix": ".NS",
        "representative_tickers": [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS",
            "AXISBANK.NS", "BAJFINANCE.NS", "MARUTI.NS", "TITAN.NS", "HCLTECH.NS",
            "SUNPHARMA.NS", "WIPRO.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "TECHM.NS",
        ],
    },
    "NASDAQ": {
        "name": "NASDAQ",
        "country": "United States",
        "currency": "USD",
        "flag": "US",
        "timezone": "America/New_York",
        "index": "^IXIC",
        "index_name": "NASDAQ Composite",
        "suffix": "",
        "representative_tickers": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
            "META", "TSLA", "AMD", "NFLX", "INTC",
            "AVGO", "COST", "PEP", "ADBE", "CSCO",
            "QCOM", "INTU", "TXN", "AMAT", "PYPL",
        ],
    },
    "NYSE": {
        "name": "New York Stock Exchange",
        "country": "United States",
        "currency": "USD",
        "flag": "US",
        "timezone": "America/New_York",
        "index": "^DJI",
        "index_name": "Dow Jones",
        "suffix": "",
        "representative_tickers": [
            "JPM", "V", "WMT", "JNJ", "UNH",
            "PG", "HD", "MA", "BAC", "DIS",
            "KO", "MRK", "CRM", "ABT", "CVX",
            "MCD", "NKE", "CAT", "GS", "AXP",
        ],
    },
    "LSE": {
        "name": "London Stock Exchange",
        "country": "United Kingdom",
        "currency": "GBP",
        "flag": "GB",
        "timezone": "Europe/London",
        "index": "^FTSE",
        "index_name": "FTSE 100",
        "suffix": ".L",
        "representative_tickers": [
            "SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "BP.L",
            "RIO.L", "GSK.L", "BATS.L", "DGE.L", "LSEG.L",
        ],
    },
    "TSE": {
        "name": "Tokyo Stock Exchange",
        "country": "Japan",
        "currency": "JPY",
        "flag": "JP",
        "timezone": "Asia/Tokyo",
        "index": "^N225",
        "index_name": "Nikkei 225",
        "suffix": ".T",
        "representative_tickers": [
            "7203.T", "6758.T", "9984.T", "8306.T", "6861.T",
            "9432.T", "6501.T", "7267.T", "4502.T", "8035.T",
        ],
    },
    "HKEX": {
        "name": "Hong Kong Stock Exchange",
        "country": "Hong Kong",
        "currency": "HKD",
        "flag": "HK",
        "timezone": "Asia/Hong_Kong",
        "index": "^HSI",
        "index_name": "Hang Seng",
        "suffix": ".HK",
        "representative_tickers": [
            "0700.HK", "9988.HK", "1299.HK", "0005.HK", "0941.HK",
            "2318.HK", "0388.HK", "0027.HK", "1810.HK", "3690.HK",
        ],
    },
    "SSE": {
        "name": "Shanghai Stock Exchange",
        "country": "China",
        "currency": "CNY",
        "flag": "CN",
        "timezone": "Asia/Shanghai",
        "index": "000001.SS",
        "index_name": "SSE Composite",
        "suffix": ".SS",
        "representative_tickers": [
            "601398.SS", "601288.SS", "600519.SS", "601857.SS", "600036.SS",
        ],
    },
    "ASX": {
        "name": "Australian Securities Exchange",
        "country": "Australia",
        "currency": "AUD",
        "flag": "AU",
        "timezone": "Australia/Sydney",
        "index": "^AXJO",
        "index_name": "ASX 200",
        "suffix": ".AX",
        "representative_tickers": [
            "BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX", "WBC.AX",
        ],
    },
    "BSE": {
        "name": "Bombay Stock Exchange",
        "country": "India",
        "currency": "INR",
        "flag": "IN",
        "timezone": "Asia/Kolkata",
        "index": "^BSESN",
        "index_name": "SENSEX",
        "suffix": ".BO",
        "representative_tickers": [
            "RELIANCE.BO", "TCS.BO", "HDFCBANK.BO", "INFY.BO", "ICICIBANK.BO",
        ],
    },
    "XETRA": {
        "name": "Frankfurt Stock Exchange",
        "country": "Germany",
        "currency": "EUR",
        "flag": "DE",
        "timezone": "Europe/Berlin",
        "index": "^GDAXI",
        "index_name": "DAX",
        "suffix": ".DE",
        "representative_tickers": [
            "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "BAS.DE",
        ],
    },
}


def _cached(key: str, ttl: int, factory):
    now = time.time()
    cached = _EXCHANGE_CACHE.get(key)
    if cached and (now - cached[0]) < ttl:
        return cached[1]
    value = factory()
    _EXCHANGE_CACHE[key] = (now, value)
    return value


def _safe_float(val, default=0.0):
    try:
        v = float(val)
        return v if not (v != v) else default  # NaN check
    except (TypeError, ValueError):
        return default


def _fetch_ticker_data(ticker: str) -> dict | None:
    """Fetch basic price/change data for a single ticker via yfinance."""
    try:
        obj = yf.Ticker(ticker)
        hist = obj.history(period="5d", interval="1d")
        if hist.empty or len(hist) < 2:
            return None
        latest = hist.iloc[-1]
        prev = hist.iloc[-2]
        price = float(latest["Close"])
        prev_close = float(prev["Close"])
        change = price - prev_close
        pct = (change / prev_close) * 100 if prev_close else 0

        info = obj.info or {}
        name = info.get("shortName") or info.get("longName") or ticker

        return {
            "ticker": ticker,
            "name": name,
            "price": round(price, 2),
            "change": round(change, 2),
            "percent_change": round(pct, 2),
            "volume": int(latest.get("Volume", 0)),
        }
    except Exception:
        return None


def get_exchange_overview(exchange_code: str) -> dict:
    """Get index value + top gainers/losers for a given exchange."""
    code = exchange_code.upper()
    if code not in EXCHANGES:
        return {"error": f"Unknown exchange: {code}", "exchanges": list(EXCHANGES.keys())}

    def factory():
        ex = EXCHANGES[code]
        tickers = ex["representative_tickers"]

        # Fetch index
        index_data = None
        try:
            idx = yf.Ticker(ex["index"])
            idx_hist = idx.history(period="5d", interval="1d")
            if not idx_hist.empty and len(idx_hist) >= 2:
                latest = idx_hist.iloc[-1]
                prev = idx_hist.iloc[-2]
                price = float(latest["Close"])
                prev_close = float(prev["Close"])
                index_data = {
                    "symbol": ex["index"],
                    "name": ex["index_name"],
                    "value": round(price, 2),
                    "change": round(price - prev_close, 2),
                    "percent_change": round(((price - prev_close) / prev_close) * 100, 2),
                }
        except Exception:
            pass

        # Fetch all tickers in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_fetch_ticker_data, t): t for t in tickers}
            for future in concurrent.futures.as_completed(futures):
                data = future.result()
                if data:
                    results.append(data)

        # Sort for gainers/losers
        results.sort(key=lambda x: x["percent_change"], reverse=True)
        gainers = results[:5]
        losers = list(reversed(results[-5:])) if len(results) >= 5 else []

        return {
            "exchange": code,
            "name": ex["name"],
            "country": ex["country"],
            "currency": ex["currency"],
            "flag": ex["flag"],
            "index": index_data,
            "gainers": gainers,
            "losers": losers,
            "total_tracked": len(results),
        }

    return _cached(f"exchange_overview:{code}", CACHE_TTL, factory)


def get_market_movers(exchange_code: str = "NSE") -> dict:
    """Get top gainers, top losers, and most active stocks."""
    overview = get_exchange_overview(exchange_code)
    return {
        "exchange": exchange_code,
        "gainers": overview.get("gainers", []),
        "losers": overview.get("losers", []),
        "index": overview.get("index"),
    }


def get_global_indices() -> list[dict]:
    """Fetch snapshot of all major global indices."""
    def factory():
        index_tickers = {
            code: ex["index"] for code, ex in EXCHANGES.items()
        }
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = {}
            for code, idx_ticker in index_tickers.items():
                futures[executor.submit(_fetch_ticker_data, idx_ticker)] = code

            for future in concurrent.futures.as_completed(futures):
                code = futures[future]
                data = future.result()
                ex = EXCHANGES[code]
                entry = {
                    "exchange": code,
                    "name": ex["index_name"],
                    "country": ex["country"],
                    "flag": ex["flag"],
                    "currency": ex["currency"],
                }
                if data:
                    entry["value"] = data["price"]
                    entry["change"] = data["change"]
                    entry["percent_change"] = data["percent_change"]
                else:
                    entry["value"] = None
                    entry["change"] = None
                    entry["percent_change"] = None
                results.append(entry)

        # Sort by exchange name
        results.sort(key=lambda x: x["name"])
        return results

    return _cached("global_indices", CACHE_TTL, factory)


def list_exchanges() -> list[dict]:
    """Return metadata for all supported exchanges."""
    return [
        {
            "code": code,
            "name": ex["name"],
            "country": ex["country"],
            "currency": ex["currency"],
            "flag": ex["flag"],
            "index_name": ex["index_name"],
            "stock_count": len(ex["representative_tickers"]),
        }
        for code, ex in EXCHANGES.items()
    ]
