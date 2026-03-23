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
FINNHUB_BASE = "https://finnhub.io/api/v1"
GROWW_BASE = "https://api.groww.in/v1"
GROWW_INSTRUMENTS_URL = "https://growwapi-assets.groww.in/instruments/instrument.csv"
GROWW_API_KEY = os.getenv("GROWW_API_KEY", "").strip()
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()

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


def _google_news_rss(query: str, limit: int = 15) -> list[dict]:
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


def _economic_times_rss(query: str, limit: int = 10) -> list[dict]:
    """Fetch stock news from Economic Times RSS feeds (Indian market focus)."""
    feeds = [
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    ]
    stories = []
    query_lower = query.lower().replace(".ns", "").replace(".bo", "")
    for feed_url in feeds:
        try:
            resp = requests.get(feed_url, timeout=15)
            resp.raise_for_status()
            root = ElementTree.fromstring(resp.content)
            for item in root.findall("./channel/item"):
                title = item.findtext("title") or ""
                desc = item.findtext("description") or ""
                # Include if query term appears in title/desc, or just include all for broad coverage
                if query_lower in title.lower() or query_lower in desc.lower() or len(stories) < limit // 2:
                    stories.append({
                        "title": title,
                        "summary": desc[:300] if desc else "",
                        "source": "Economic Times",
                        "url": item.findtext("link"),
                        "published_at": item.findtext("pubDate"),
                        "sentiment_score": None,
                        "sentiment_label": "neutral",
                    })
                if len(stories) >= limit:
                    break
        except Exception:
            continue
        if len(stories) >= limit:
            break
    return stories[:limit]


def _moneycontrol_rss(limit: int = 8) -> list[dict]:
    """Fetch market news from Moneycontrol RSS."""
    feeds = [
        "https://www.moneycontrol.com/rss/marketreports.xml",
        "https://www.moneycontrol.com/rss/stocksinnews.xml",
    ]
    stories = []
    for feed_url in feeds:
        try:
            resp = requests.get(feed_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            root = ElementTree.fromstring(resp.content)
            for item in root.findall("./channel/item"):
                stories.append({
                    "title": item.findtext("title") or "",
                    "summary": (item.findtext("description") or "")[:300],
                    "source": "Moneycontrol",
                    "url": item.findtext("link"),
                    "published_at": item.findtext("pubDate"),
                    "sentiment_score": None,
                    "sentiment_label": "neutral",
                })
                if len(stories) >= limit:
                    break
        except Exception:
            continue
        if len(stories) >= limit:
            break
    return stories[:limit]


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


def _finnhub_ticker(ticker: str) -> str:
    """Convert yfinance ticker to Finnhub format."""
    if ticker.endswith(".NS"):
        return ticker.replace(".NS", ".NS")  # Finnhub uses same for NSE
    if ticker.endswith(".BO"):
        return ticker.replace(".BO", ".BO")
    return ticker


def _finnhub_news(ticker: str, limit: int = 8) -> list[dict]:
    """Fetch company news from Finnhub."""
    from datetime import datetime, timedelta
    today = datetime.now().strftime("%Y-%m-%d")
    week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    # For Indian stocks, search general market news with keyword
    symbol = ticker.replace(".NS", "").replace(".BO", "")

    try:
        data = requests.get(
            f"{FINNHUB_BASE}/company-news",
            params={"symbol": symbol, "from": week_ago, "to": today, "token": FINNHUB_API_KEY},
            timeout=15,
        ).json()

        if not isinstance(data, list) or not data:
            # Try general news search
            data = requests.get(
                f"{FINNHUB_BASE}/news",
                params={"category": "general", "token": FINNHUB_API_KEY},
                timeout=15,
            ).json()

        stories = []
        for item in (data or [])[:limit]:
            stories.append({
                "title": item.get("headline"),
                "summary": item.get("summary", ""),
                "source": item.get("source", "Finnhub"),
                "url": item.get("url"),
                "published_at": item.get("datetime"),
                "image": item.get("image"),
                "sentiment_score": None,
                "sentiment_label": "neutral",
            })
        return stories
    except Exception:
        return []


def get_stock_news(ticker: str, limit: int = 20) -> dict:
    """
    Aggregate news from 6+ sources for maximum coverage.
    Default limit raised from 8 → 20 for a richer news feed.
    Sources: Finnhub, Alpha Vantage, yfinance, Google News RSS,
             Economic Times RSS, Moneycontrol RSS.
    """
    stories = []
    seen_urls = set()
    seen_titles = set()

    def _dedup_add(new_stories: list[dict]):
        """Add stories with URL + fuzzy title deduplication."""
        for s in new_stories:
            url = s.get("url") or ""
            title = (s.get("title") or "").strip().lower()
            if not title:
                continue
            # Skip exact URL duplicates
            if url and url in seen_urls:
                continue
            # Skip near-duplicate titles (first 50 chars match)
            title_key = title[:50]
            if title_key in seen_titles:
                continue
            seen_urls.add(url)
            seen_titles.add(title_key)
            stories.append(s)

    # ── 1. Finnhub company news ──────────────────────────────────────
    if FINNHUB_API_KEY:
        _dedup_add(_finnhub_news(ticker, limit=15))

    # ── 2. Alpha Vantage sentiment ───────────────────────────────────
    if ALPHA_VANTAGE_API_KEY:
        try:
            data = _request_json(
                ALPHA_VANTAGE_BASE,
                {
                    "function": "NEWS_SENTIMENT",
                    "tickers": ticker,
                    "limit": 15,
                    "apikey": ALPHA_VANTAGE_API_KEY,
                },
            )
            feed = data.get("feed", []) if isinstance(data, dict) else []
            av_stories = []
            for item in feed[:15]:
                score = _normalize_news_sentiment(item)
                av_stories.append({
                    "title": item.get("title"),
                    "summary": item.get("summary"),
                    "source": item.get("source"),
                    "url": item.get("url"),
                    "published_at": item.get("time_published"),
                    "sentiment_score": score,
                    "sentiment_label": _sentiment_label(score),
                })
            _dedup_add(av_stories)
        except Exception:
            pass

    # ── 3. yfinance news ─────────────────────────────────────────────
    try:
        ticker_obj = yf.Ticker(ticker)
        items = getattr(ticker_obj, "news", []) or []
        yf_stories = []
        for item in items[:15]:
            content = item.get("content", {})
            provider = content.get("provider") or item.get("publisher")
            url = _text_value(content.get("canonicalUrl", {}).get("url") or item.get("link"))
            yf_stories.append({
                "title": _text_value(content.get("title") or item.get("title")),
                "summary": _text_value(content.get("summary") or item.get("summary")),
                "source": _text_value(provider),
                "url": url,
                "published_at": content.get("pubDate") or item.get("providerPublishTime"),
                "sentiment_score": None,
                "sentiment_label": "neutral",
            })
        _dedup_add(yf_stories)
    except Exception:
        pass

    # ── 4. Google News RSS (multiple queries for broader coverage) ───
    clean_symbol = ticker.replace(".NS", "").replace(".BO", "")
    queries = [
        f"{ticker} stock",
        f"{clean_symbol} share price",
        f"{clean_symbol} stock market news",
    ]
    for q in queries:
        if len(stories) >= limit:
            break
        try:
            _dedup_add(_google_news_rss(q, limit=10))
        except Exception:
            continue

    # ── 5. Economic Times RSS (Indian market) ────────────────────────
    if ".NS" in ticker or ".BO" in ticker:
        try:
            _dedup_add(_economic_times_rss(clean_symbol, limit=8))
        except Exception:
            pass

    # ── 6. Moneycontrol RSS (Indian market general) ──────────────────
    if ".NS" in ticker or ".BO" in ticker:
        try:
            _dedup_add(_moneycontrol_rss(limit=6))
        except Exception:
            pass

    primary = "multi-source"
    if FINNHUB_API_KEY:
        primary = "Finnhub + multi-source"

    return {
        "source": primary,
        "average_sentiment": None,
        "average_label": "neutral",
        "stories": stories[:limit],
    }


def get_finnhub_candles(ticker: str, resolution: str = "D", count: int = 365) -> list[dict]:
    """Fetch OHLCV candle data from Finnhub for charting."""
    if not FINNHUB_API_KEY:
        return []
    import time as _time
    now = int(_time.time())
    start = now - (count * 86400)
    symbol = ticker.replace(".NS", ".NS").replace(".BO", ".BO")
    try:
        data = requests.get(
            f"{FINNHUB_BASE}/stock/candle",
            params={"symbol": symbol, "resolution": resolution, "from": start, "to": now, "token": FINNHUB_API_KEY},
            timeout=15,
        ).json()
        if data.get("s") != "ok":
            return []
        candles = []
        for i in range(len(data.get("t", []))):
            candles.append({
                "time": data["t"][i],
                "open": data["o"][i],
                "high": data["h"][i],
                "low": data["l"][i],
                "close": data["c"][i],
                "volume": data["v"][i] if "v" in data else 0,
            })
        return candles
    except Exception:
        return []


def _is_nse_open_now() -> dict:
    """Smart NSE market-hours check using IST timezone.
    NSE pre-open: 09:00-09:15, normal trading: 09:15-15:30 IST, Mon-Fri.
    Also accounts for Indian public holidays (major ones).
    """
    from datetime import datetime, timezone, timedelta
    IST = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(IST)
    weekday = now.weekday()          # 0=Mon … 6=Sun

    # ---------- holidays (static list, covers major NSE closures) ----------
    HOLIDAYS_2025_2026 = {
        # 2025
        (1, 26), (2, 26), (3, 14), (3, 31), (4, 10), (4, 14), (4, 18),
        (5, 1), (6, 27), (8, 15), (8, 16), (8, 27), (10, 2), (10, 20),
        (10, 21), (10, 22), (10, 23), (11, 5), (11, 26), (12, 25),
        # 2026
        (1, 26), (2, 17), (3, 3), (3, 19), (3, 30), (4, 3), (4, 14),
        (5, 1), (5, 25), (6, 17), (7, 7), (8, 15), (9, 4), (10, 2),
        (10, 9), (10, 20), (11, 9), (11, 24), (12, 25),
    }
    today_md = (now.month, now.day)

    # Weekend
    if weekday >= 5:
        return {"available": True, "isOpen": False,
                "holiday": "Weekend", "exchange": "NSE",
                "t": now.isoformat()}

    # Holiday
    if today_md in HOLIDAYS_2025_2026:
        return {"available": True, "isOpen": False,
                "holiday": "Public Holiday", "exchange": "NSE",
                "t": now.isoformat()}

    # Trading hours: 09:15 – 15:30 IST
    market_open  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

    is_open = market_open <= now <= market_close

    return {
        "available": True,
        "isOpen": is_open,
        "exchange": "NSE",
        "t": now.isoformat(),
        "session": "pre-open" if (now.replace(hour=9, minute=0, second=0) <= now < market_open) else
                   "normal" if is_open else "closed",
    }


def get_market_status() -> dict:
    """Check if NSE / US markets are open.
    Priority: smart IST-based NSE check (always works, no API needed).
    Fallback Finnhub for US if key is available.
    """
    result = _is_nse_open_now()

    # Also try Finnhub for US market status if key is available
    if FINNHUB_API_KEY:
        try:
            us = requests.get(
                f"{FINNHUB_BASE}/stock/market-status",
                params={"exchange": "US", "token": FINNHUB_API_KEY},
                timeout=5,
            ).json()
            result["us_isOpen"] = us.get("isOpen", False)
        except Exception:
            result["us_isOpen"] = None

    return result
