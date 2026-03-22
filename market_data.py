"""
External market data integrations for the dashboard.

Primary sources:
- Groww for Indian real-time market data when configured
- Twelve Data for quote + symbol discovery
- Alpha Vantage for news sentiment
- yfinance fallback when API keys are absent
"""

from __future__ import annotations

import ast
import csv
import io
import os
from functools import lru_cache
from urllib.parse import quote_plus
from xml.etree import ElementTree

import requests
import yfinance as yf
from dotenv import load_dotenv


load_dotenv()

TWELVE_DATA_BASE = "https://api.twelvedata.com"
ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
GROWW_BASE = "https://api.groww.in/v1"
GROWW_INSTRUMENTS_URL = "https://growwapi-assets.groww.in/instruments/instrument.csv"
GROWW_API_KEY = os.getenv("GROWW_API_KEY", "").strip()
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()

LOCAL_SYMBOLS = [
    {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ", "country": "United States", "type": "Common Stock"},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ", "country": "United States", "type": "Common Stock"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "exchange": "NASDAQ", "country": "United States", "type": "Common Stock"},
    {"symbol": "AMZN", "name": "Amazon.com, Inc.", "exchange": "NASDAQ", "country": "United States", "type": "Common Stock"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "exchange": "NASDAQ", "country": "United States", "type": "Common Stock"},
    {"symbol": "META", "name": "Meta Platforms, Inc.", "exchange": "NASDAQ", "country": "United States", "type": "Common Stock"},
    {"symbol": "TSLA", "name": "Tesla, Inc.", "exchange": "NASDAQ", "country": "United States", "type": "Common Stock"},
    {"symbol": "AMD", "name": "Advanced Micro Devices, Inc.", "exchange": "NASDAQ", "country": "United States", "type": "Common Stock"},
    {"symbol": "NFLX", "name": "Netflix, Inc.", "exchange": "NASDAQ", "country": "United States", "type": "Common Stock"},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "exchange": "NYSE", "country": "United States", "type": "Common Stock"},
    {"symbol": "V", "name": "Visa Inc.", "exchange": "NYSE", "country": "United States", "type": "Common Stock"},
    {"symbol": "WMT", "name": "Walmart Inc.", "exchange": "NYSE", "country": "United States", "type": "Common Stock"},
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries Limited", "exchange": "NSE", "country": "India", "type": "Common Stock"},
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services Limited", "exchange": "NSE", "country": "India", "type": "Common Stock"},
    {"symbol": "INFY.NS", "name": "Infosys Limited", "exchange": "NSE", "country": "India", "type": "Common Stock"},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank Limited", "exchange": "NSE", "country": "India", "type": "Common Stock"},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank Limited", "exchange": "NSE", "country": "India", "type": "Common Stock"},
    {"symbol": "SBIN.NS", "name": "State Bank of India", "exchange": "NSE", "country": "India", "type": "Common Stock"},
    {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel Limited", "exchange": "NSE", "country": "India", "type": "Common Stock"},
    {"symbol": "ITC.NS", "name": "ITC Limited", "exchange": "NSE", "country": "India", "type": "Common Stock"},
    {"symbol": "LT.NS", "name": "Larsen & Toubro Limited", "exchange": "NSE", "country": "India", "type": "Common Stock"},
    {"symbol": "ASIANPAINT.NS", "name": "Asian Paints Limited", "exchange": "NSE", "country": "India", "type": "Common Stock"},
]


def _request_json(url: str, params: dict, headers: dict | None = None) -> dict | list:
    response = requests.get(url, params=params, headers=headers, timeout=20)
    response.raise_for_status()
    return response.json()


def _normalize_news_sentiment(item: dict) -> float | None:
    score = item.get("overall_sentiment_score")
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def _sentiment_label(score: float | None) -> str:
    if score is None:
        return "neutral"
    if score >= 0.2:
        return "bullish"
    if score <= -0.2:
        return "bearish"
    return "neutral"


def _symbol_name(item: dict) -> str:
    return item.get("instrument_name") or item.get("name") or item.get("symbol") or ""


def _text_value(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("displayName", "name", "title", "url", "value"):
            if value.get(key):
                return str(value[key])
        return None
    if isinstance(value, (list, tuple)):
        parts = [str(item) for item in value if item]
        return ", ".join(parts) if parts else None
    return str(value)


def _is_indian_cash_ticker(ticker: str) -> bool:
    return ticker.endswith(".NS") or ticker.endswith(".BO") or ticker.isupper()


def _groww_exchange_and_symbol(ticker: str) -> tuple[str, str]:
    if ticker.endswith(".NS"):
        return "NSE", ticker.removesuffix(".NS")
    if ticker.endswith(".BO"):
        return "BSE", ticker.removesuffix(".BO")
    return "NSE", ticker


def _format_exchange_symbol(exchange: str, trading_symbol: str) -> str:
    if exchange == "NSE":
        return f"{trading_symbol}.NS"
    if exchange == "BSE":
        return f"{trading_symbol}.BO"
    return trading_symbol


def _groww_headers() -> dict:
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {GROWW_API_KEY}",
        "X-API-VERSION": "1.0",
    }


@lru_cache(maxsize=1)
def _load_groww_instruments() -> list[dict]:
    response = requests.get(GROWW_INSTRUMENTS_URL, timeout=30)
    response.raise_for_status()

    reader = csv.DictReader(io.StringIO(response.text))
    instruments = []
    for row in reader:
        if row.get("segment") != "CASH":
            continue
        if row.get("instrument_type") not in {"EQ", "INDEX", "ETF"}:
            continue
        exchange = (row.get("exchange") or "").strip()
        trading_symbol = (row.get("trading_symbol") or "").strip()
        if exchange not in {"NSE", "BSE"} or not trading_symbol:
            continue
        instruments.append(
            {
                "symbol": _format_exchange_symbol(exchange, trading_symbol),
                "name": (row.get("name") or trading_symbol).strip(),
                "exchange": exchange,
                "country": "India",
                "type": row.get("instrument_type") or "Common Stock",
                "groww_symbol": row.get("groww_symbol"),
                "segment": row.get("segment"),
                "trading_symbol": trading_symbol,
            }
        )
    return instruments


def _parse_ohlc_payload(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            cleaned = cleaned[1:-1]
        pairs = []
        for item in cleaned.split(","):
            if ":" not in item:
                continue
            key, raw_value = item.split(":", 1)
            pairs.append((key.strip().strip("'\""), raw_value.strip()))
        parsed = {}
        for key, raw_value in pairs:
            try:
                parsed[key] = float(ast.literal_eval(raw_value))
            except Exception:
                try:
                    parsed[key] = float(raw_value)
                except Exception:
                    parsed[key] = raw_value
        return parsed
    return {}


def _groww_quote(ticker: str) -> dict:
    if not GROWW_API_KEY or not _is_indian_cash_ticker(ticker):
        raise ValueError("Groww quote unavailable")

    exchange, trading_symbol = _groww_exchange_and_symbol(ticker)
    data = _request_json(
        f"{GROWW_BASE}/live-data/quote",
        {"exchange": exchange, "segment": "CASH", "trading_symbol": trading_symbol},
        headers=_groww_headers(),
    )
    if not isinstance(data, dict) or data.get("status") != "SUCCESS":
        raise ValueError(f"Groww quote unavailable for {ticker}")

    payload = data.get("payload", {})
    ohlc = _parse_ohlc_payload(payload.get("ohlc", {}))
    last_price = float(payload.get("last_price"))
    change = float(payload.get("day_change") or 0.0)
    percent_change = float(payload.get("day_change_perc") or 0.0)

    return {
        "ticker": ticker,
        "name": trading_symbol,
        "price": last_price,
        "change": change,
        "percent_change": percent_change,
        "open": float(ohlc.get("open") or payload.get("open") or last_price),
        "high": float(ohlc.get("high") or payload.get("high") or last_price),
        "low": float(ohlc.get("low") or payload.get("low") or last_price),
        "previous_close": float(ohlc.get("close") or payload.get("close") or (last_price - change)),
        "volume": float(payload.get("volume") or 0),
        "exchange": exchange,
        "timestamp": payload.get("last_trade_time"),
        "source": "Groww",
        "is_market_open": None,
        "market_cap": payload.get("market_cap"),
        "week_52_high": payload.get("week_52_high"),
        "week_52_low": payload.get("week_52_low"),
    }


def _google_news_rss(query: str, limit: int = 8) -> list[dict]:
    url = (
        "https://news.google.com/rss/search"
        f"?q={quote_plus(query)}"
        "&hl=en-IN&gl=IN&ceid=IN:en"
    )
    response = requests.get(url, timeout=20)
    response.raise_for_status()

    root = ElementTree.fromstring(response.content)
    stories = []
    for item in root.findall("./channel/item")[:limit]:
        stories.append(
            {
                "title": item.findtext("title"),
                "summary": item.findtext("description"),
                "source": "Google News",
                "url": item.findtext("link"),
                "published_at": item.findtext("pubDate"),
                "sentiment_score": None,
                "sentiment_label": "neutral",
            }
        )
    return stories


@lru_cache(maxsize=64)
def search_symbols(query: str, limit: int = 12) -> list[dict]:
    query = query.strip()
    if not query:
        try:
            return _load_groww_instruments()[:limit]
        except Exception:
            return LOCAL_SYMBOLS[:limit]

    if TWELVE_DATA_API_KEY:
        try:
            params = {
                "symbol": query,
                "outputsize": max(limit, 12),
                "apikey": TWELVE_DATA_API_KEY,
            }
            data = _request_json(f"{TWELVE_DATA_BASE}/symbol_search", params)
            items = data.get("data", []) if isinstance(data, dict) else data
            return [
                {
                    "symbol": item.get("symbol"),
                    "name": _symbol_name(item),
                    "exchange": item.get("exchange"),
                    "country": item.get("country"),
                    "type": item.get("instrument_type") or item.get("type"),
                }
                for item in items[:limit]
                if item.get("symbol")
            ]
        except Exception:
            pass

    try:
        lowered = query.lower()
        groww_matches = [
            item
            for item in _load_groww_instruments()
            if lowered in item["symbol"].lower() or lowered in item["name"].lower()
        ]
        if groww_matches:
            return groww_matches[:limit]
    except Exception:
        pass

    lowered = query.lower()
    return [
        item for item in LOCAL_SYMBOLS
        if lowered in item["symbol"].lower() or lowered in item["name"].lower()
    ][:limit]


def get_realtime_quote(ticker: str) -> dict:
    if GROWW_API_KEY:
        try:
            return _groww_quote(ticker)
        except Exception:
            pass

    if TWELVE_DATA_API_KEY:
        try:
            data = _request_json(
                f"{TWELVE_DATA_BASE}/quote",
                {"symbol": ticker, "apikey": TWELVE_DATA_API_KEY},
            )
            if isinstance(data, dict) and data.get("code"):
                raise ValueError(data.get("message") or f"Quote unavailable for {ticker}")
            price = float(data["close"])
            previous_close = float(data.get("previous_close") or data.get("close"))
            change = float(data.get("change") or (price - previous_close))
            percent_change = float(data.get("percent_change") or ((change / previous_close) * 100 if previous_close else 0))
            return {
                "ticker": ticker,
                "name": data.get("name") or ticker,
                "price": price,
                "change": change,
                "percent_change": percent_change,
                "open": float(data.get("open") or price),
                "high": float(data.get("high") or price),
                "low": float(data.get("low") or price),
                "previous_close": previous_close,
                "volume": float(data.get("volume") or 0),
                "exchange": data.get("exchange"),
                "timestamp": data.get("datetime"),
                "source": "Twelve Data",
                "is_market_open": data.get("is_market_open"),
            }
        except Exception:
            pass

    ticker_obj = yf.Ticker(ticker)
    history = ticker_obj.history(period="5d", interval="1d")
    if history.empty:
        raise ValueError(f"Quote unavailable for {ticker}")
    latest = history.iloc[-1]
    previous = history.iloc[-2] if len(history) > 1 else latest
    price = float(latest["Close"])
    previous_close = float(previous["Close"])
    change = price - previous_close
    percent_change = (change / previous_close) * 100 if previous_close else 0.0
    return {
        "ticker": ticker,
        "name": ticker,
        "price": price,
        "change": change,
        "percent_change": percent_change,
        "open": float(latest["Open"]),
        "high": float(latest["High"]),
        "low": float(latest["Low"]),
        "previous_close": previous_close,
        "volume": float(latest["Volume"]),
        "exchange": None,
        "timestamp": str(history.index[-1]),
        "source": "Yahoo Finance fallback",
        "is_market_open": None,
    }


def get_stock_news(ticker: str, limit: int = 8) -> dict:
    if ALPHA_VANTAGE_API_KEY:
        try:
            data = _request_json(
                ALPHA_VANTAGE_BASE,
                {
                    "function": "NEWS_SENTIMENT",
                    "tickers": ticker,
                    "limit": limit,
                    "apikey": ALPHA_VANTAGE_API_KEY,
                },
            )
            feed = data.get("feed", []) if isinstance(data, dict) else []
            stories = []
            scores = []
            for item in feed[:limit]:
                score = _normalize_news_sentiment(item)
                if score is not None:
                    scores.append(score)
                stories.append(
                    {
                        "title": item.get("title"),
                        "summary": item.get("summary"),
                        "source": item.get("source"),
                        "url": item.get("url"),
                        "published_at": item.get("time_published"),
                        "sentiment_score": score,
                        "sentiment_label": _sentiment_label(score),
                    }
                )

            average = sum(scores) / len(scores) if scores else None
            return {
                "source": "Alpha Vantage",
                "average_sentiment": average,
                "average_label": _sentiment_label(average),
                "stories": stories,
            }
        except Exception:
            pass

    ticker_obj = yf.Ticker(ticker)
    items = getattr(ticker_obj, "news", []) or []
    stories = []
    for item in items[:limit]:
        content = item.get("content", {})
        provider = content.get("provider") or item.get("publisher")
        stories.append(
            {
                "title": _text_value(content.get("title") or item.get("title")),
                "summary": _text_value(content.get("summary") or item.get("summary")),
                "source": _text_value(provider),
                "url": _text_value(content.get("canonicalUrl", {}).get("url") or item.get("link")),
                "published_at": content.get("pubDate") or item.get("providerPublishTime"),
                "sentiment_score": None,
                "sentiment_label": "neutral",
            }
        )

    if len(stories) < limit:
        try:
            google_stories = _google_news_rss(f"{ticker} stock", limit=limit)
            seen = {story.get("url") for story in stories if story.get("url")}
            for story in google_stories:
                url = story.get("url")
                if url and url in seen:
                    continue
                stories.append(story)
                if url:
                    seen.add(url)
                if len(stories) >= limit:
                    break
        except Exception:
            pass

    return {
        "source": "Yahoo Finance + Google News fallback" if stories else "Yahoo Finance fallback",
        "average_sentiment": None,
        "average_label": "neutral",
        "stories": stories,
    }
