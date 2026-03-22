"""
Application service layer for the stock dashboard API.

Wraps the existing training/backtest/advisory logic into JSON-friendly
functions the FastAPI server can return directly.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
import pickle
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
import torch
import yfinance as yf

from advisor import _classify
from backtest import _fetch_benchmark_returns, _price_to_signal, _simulate_trades
from config import Config
from data_fetcher import add_technical_indicators, fetch_data, fetch_fundamentals
from market_data import (
    ALPHA_VANTAGE_API_KEY,
    GROWW_API_KEY,
    TWELVE_DATA_API_KEY,
    get_realtime_quote,
    get_stock_news,
    search_symbols,
)
from metrics import (
    avg_win_loss_ratio,
    cagr,
    calmar_ratio,
    compute_alpha_beta,
    conditional_var,
    daily_volatility,
    expectancy,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
    win_rate,
)
from model import LSTMStockPredictor
from screener import fundamental_score, is_multibagger_candidate, momentum_score, technical_score
from train import train


_RUNTIME_CACHE: dict[str, tuple[float, object]] = {}
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cache_key(prefix: str, *parts) -> str:
    return "::".join([prefix, *[str(part) for part in parts]])


def _cached(ttl_seconds: int, key: str, factory):
    now = time.time()
    cached = _RUNTIME_CACHE.get(key)
    if cached and (now - cached[0]) < ttl_seconds:
        return cached[1]

    value = factory()
    _RUNTIME_CACHE[key] = (now, value)
    return value


def _clear_runtime_cache() -> None:
    _RUNTIME_CACHE.clear()


def _to_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if pd.isna(value):
        return None
    return float(value)


def _json_scalar(value):
    if value is None:
        return None
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, int):
        return int(value)
    return value


def _to_date(value) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _fetch_with_fallback(ticker: str, period: str, prefer_live: bool) -> pd.DataFrame:
    if not prefer_live:
        return fetch_data(ticker, period, use_cache=True)

    try:
        return fetch_data(ticker, period, use_cache=False)
    except Exception:
        return fetch_data(ticker, period, use_cache=True)


def _trained_tickers() -> list[str]:
    cfg = Config()
    tickers = []
    if os.path.isdir(cfg.METADATA_DIR):
        for name in os.listdir(cfg.METADATA_DIR):
            if not name.endswith(".json"):
                continue
            path = os.path.join(cfg.METADATA_DIR, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                ticker = payload.get("ticker")
                if ticker:
                    tickers.append(ticker)
            except Exception:
                continue
    if os.path.exists(cfg.MODEL_PATH) and os.path.exists(cfg.SCALER_PATH):
        tickers.append(cfg.DEFAULT_TICKER)
    return sorted(set(tickers))


@lru_cache(maxsize=32)
def _load_scalers(ticker: str):
    cfg = Config()
    scaler_path = cfg.resolve_scaler_path(ticker)
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scalers not found for {ticker}. Run train.py --ticker {ticker} first.")
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=32)
def _load_model(ticker: str):
    cfg = Config()
    model_path = cfg.resolve_model_path(ticker)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found for {ticker}. Run train.py --ticker {ticker} first.")

    model = LSTMStockPredictor(
        input_dim=cfg.INPUT_DIM,
        hidden_dim=cfg.HIDDEN_DIM,
        num_layers=cfg.NUM_LAYERS,
        output_dim=cfg.OUTPUT_DIM,
        dropout=cfg.DROPOUT,
    ).to(_device())
    model.load_state_dict(torch.load(model_path, map_location=_device()))
    model.eval()
    return model


def get_app_status() -> dict:
    cfg = Config()
    trained = _trained_tickers()
    return {
        "api": "ok",
        "model_exists": bool(trained),
        "scalers_exist": bool(trained),
        "cache_dir": cfg.CACHE_DIR,
        "results_dir": cfg.RESULTS_DIR,
        "default_ticker": cfg.DEFAULT_TICKER,
        "device": str(_device()),
        "groww_configured": bool(GROWW_API_KEY),
        "gemini_configured": bool(GEMINI_API_KEY),
        "twelve_data_configured": bool(TWELVE_DATA_API_KEY),
        "alpha_vantage_configured": bool(ALPHA_VANTAGE_API_KEY),
        "trained_tickers": trained,
    }


def get_ai_brief(ticker: str, period: str | None = None) -> dict:
    cfg = Config()
    if not GEMINI_API_KEY:
        return {
            "enabled": False,
            "ticker": ticker,
            "summary": "Gemini API key not configured.",
        }

    active_period = period or cfg.DATA_PERIOD

    def factory():
        try:
            quote = get_quote(ticker)
            news = get_news(ticker, limit=5)
            model_status = get_model_status(ticker)
            advisory = get_advisory(ticker, active_period) if model_status.get("ready") else None

            headlines = []
            for story in news.get("stories", [])[:5]:
                title = story.get("title") or "Untitled"
                source = story.get("source") or "Unknown source"
                headlines.append(f"- {title} ({source})")

            prompt = (
                f"You are a concise stock market analyst. Analyze {ticker} in 5 bullet points max. "
                f"Use plain English, avoid hype, mention risks, and end with one actionable watch item.\n\n"
                f"Live quote: price={quote.get('price')}, day_change_pct={quote.get('percent_change')}, "
                f"source={quote.get('source')}.\n"
                f"Model ready: {model_status.get('ready')}.\n"
                f"Model signal: {advisory.get('signal') if advisory else 'N/A'}.\n"
                f"Predicted 5d move: {round(advisory.get('direction_pct', 0), 2) if advisory else 'N/A'}.\n"
                f"News headlines:\n" + "\n".join(headlines)
            )

            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ]
            }
            response = requests.post(
                GEMINI_API_URL,
                headers={
                    "x-goog-api-key": GEMINI_API_KEY,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            summary = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )
            return {
                "enabled": True,
                "ticker": ticker,
                "summary": summary or "No summary returned.",
                "model": "gemini-2.5-flash",
            }
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code == 429:
                return {
                    "enabled": True,
                    "ticker": ticker,
                    "summary": "Gemini is rate-limited right now. Try the AI Brief button again in a minute.",
                    "model": "gemini-2.5-flash",
                }
            raise

    return _cached(300, _cache_key("ai-brief", ticker, active_period), factory)


def get_model_status(ticker: str) -> dict:
    cfg = Config()
    model_path = cfg.resolve_model_path(ticker)
    scaler_path = cfg.resolve_scaler_path(ticker)
    metadata_path = cfg.get_metadata_path(ticker)
    model_exists = os.path.exists(model_path)
    scaler_exists = os.path.exists(scaler_path)
    model_size_mb = round(os.path.getsize(model_path) / (1024 * 1024), 2) if model_exists else 0.0
    scaler_size_kb = round(os.path.getsize(scaler_path) / 1024, 2) if scaler_exists else 0.0
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}
    elif ticker == cfg.DEFAULT_TICKER and model_exists and scaler_exists:
        metadata = {"ticker": ticker, "legacy_default_artifact": True}

    return {
        "ready": model_exists and scaler_exists,
        "ticker": ticker,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "model_size_mb": model_size_mb,
        "scaler_size_kb": scaler_size_kb,
        "sequence_length": cfg.SEQ_LENGTH,
        "prediction_length": cfg.PRED_LENGTH,
        "features": cfg.FEATURE_COLS,
        "trained_tickers": _trained_tickers(),
        "metadata": metadata,
    }


def train_model_for_ticker(ticker: str, period: str | None = None, epochs: int | None = None) -> dict:
    cfg = Config()
    train(cfg, ticker=ticker, period=period or cfg.DATA_PERIOD, epochs=epochs)
    _load_model.cache_clear()
    _load_scalers.cache_clear()
    _clear_runtime_cache()
    return get_model_status(ticker)


def train_starter_pack(period: str | None = None, epochs: int | None = None) -> dict:
    cfg = Config()
    results = []
    for ticker in cfg.STARTER_MODEL_TICKERS:
        try:
            status = train_model_for_ticker(ticker=ticker, period=period or cfg.DATA_PERIOD, epochs=epochs)
            results.append({"ticker": ticker, "status": "trained", "ready": status["ready"]})
        except Exception as exc:
            results.append({"ticker": ticker, "status": "failed", "error": str(exc)})
    return {
        "requested": cfg.STARTER_MODEL_TICKERS,
        "results": results,
        "trained_tickers": _trained_tickers(),
    }


def get_symbol_search(query: str, limit: int = 12) -> dict:
    return {
        "query": query,
        "results": search_symbols(query, limit=limit),
    }


def get_quote(ticker: str) -> dict:
    return _cached(10, _cache_key("quote", ticker), lambda: get_realtime_quote(ticker))


def get_news(ticker: str, limit: int = 8) -> dict:
    def factory():
        payload = get_stock_news(ticker, limit=limit)
        stories = payload.get("stories", [])
        bullish = sum(1 for story in stories if story.get("sentiment_label") == "bullish")
        bearish = sum(1 for story in stories if story.get("sentiment_label") == "bearish")
        neutral = len(stories) - bullish - bearish

        payload["stats"] = {
            "count": len(stories),
            "bullish": bullish,
            "bearish": bearish,
            "neutral": neutral,
        }
        return payload

    return _cached(180, _cache_key("news", ticker, limit), factory)


def get_price_history(ticker: str, period: str) -> dict:
    def factory():
        df = _fetch_with_fallback(ticker, period, prefer_live=True)
        closes = [
            {"date": _to_date(idx.date()), "close": _to_float(row["Close"])}
            for idx, row in df.tail(180).iterrows()
        ]

        latest = df.iloc[-1]
        return {
            "ticker": ticker,
            "period": period,
            "rows": len(df),
            "history": closes,
            "latest": {
                "close": _to_float(latest["Close"]),
                "rsi": _to_float(latest.get("RSI")),
                "macd": _to_float(latest.get("MACD")),
                "macd_signal": _to_float(latest.get("MACD_Signal")),
                "sma_10": _to_float(latest.get("SMA_10")),
                "sma_30": _to_float(latest.get("SMA_30")),
                "volume_ratio": _to_float(latest.get("Volume_Ratio")),
            },
        }

    return _cached(180, _cache_key("history", ticker, period), factory)


def get_advisory(ticker: str, period: str | None = None) -> dict:
    cfg = Config()
    active_period = period or cfg.DATA_PERIOD

    def factory():
        df = _fetch_with_fallback(ticker, active_period, prefer_live=True)
        if len(df) < cfg.SEQ_LENGTH:
            raise ValueError(f"Need at least {cfg.SEQ_LENGTH} rows of data.")

        scalers = _load_scalers(ticker)
        model = _load_model(ticker)
        feature_scaler = scalers["feature"]
        close_scaler = scalers["close"]

        scaled = feature_scaler.transform(df[cfg.FEATURE_COLS])
        x = torch.FloatTensor(scaled[-cfg.SEQ_LENGTH :]).unsqueeze(0).to(_device())

        with torch.no_grad():
            preds_scaled, attn_weights = model(x)

        preds = close_scaler.inverse_transform(preds_scaled.cpu().numpy())[0]
        attention = attn_weights.cpu().numpy()[0]
        current_price = float(df["Close"].iloc[-1])
        pred_day5 = float(preds[-1])
        signal = _classify(current_price, pred_day5, cfg.SIGNAL_THRESHOLD)
        pct_change = ((pred_day5 - current_price) / current_price) * 100
        confidence = min(float(np.sort(attention)[-5:].sum() * 100), 99.0)
        top_steps = np.argsort(attention)[-5:][::-1]
        focus_days = [cfg.SEQ_LENGTH - int(t) for t in top_steps]

        return {
            "ticker": ticker,
            "as_of": _to_date(df.index[-1].date()),
            "current_price": current_price,
            "predictions": [
                {
                    "day": i,
                    "price": float(price),
                    "pct_change": float(((price - current_price) / current_price) * 100),
                }
                for i, price in enumerate(preds, start=1)
            ],
            "signal": signal,
            "direction_pct": float(pct_change),
            "confidence": confidence,
            "focus_days_ago": focus_days,
            "threshold_pct": cfg.SIGNAL_THRESHOLD * 100,
        }

    return _cached(300, _cache_key("advisory", ticker, active_period), factory)


def _metrics_payload(
    trades: list[dict],
    equity_curve: np.ndarray,
    daily_returns: np.ndarray,
    benchmark_returns: np.ndarray | None,
    initial_capital: float,
) -> dict:
    alpha = beta = None
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        alpha, beta = compute_alpha_beta(daily_returns, benchmark_returns)

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    return {
        "total_return_pct": _json_scalar(((equity_curve[-1] / equity_curve[0]) - 1) * 100 if len(equity_curve) else 0.0),
        "cagr_pct": _json_scalar(cagr(equity_curve) * 100),
        "final_capital": _json_scalar(float(equity_curve[-1]) if len(equity_curve) else float(initial_capital)),
        "sharpe": _json_scalar(sharpe_ratio(daily_returns)),
        "sortino": _json_scalar(sortino_ratio(daily_returns)),
        "max_drawdown_pct": _json_scalar(max_drawdown(equity_curve) * 100),
        "calmar": _json_scalar(calmar_ratio(equity_curve)),
        "alpha_pct": _json_scalar(alpha * 100 if alpha is not None else None),
        "beta": _json_scalar(beta),
        "total_trades": _json_scalar(len(trades)),
        "win_rate_pct": _json_scalar(win_rate(trades) * 100),
        "profit_factor": _json_scalar(profit_factor(trades)),
        "expectancy": _json_scalar(expectancy(trades)),
        "avg_win_loss": _json_scalar(avg_win_loss_ratio(trades)),
        "var_95_pct": _json_scalar(value_at_risk(daily_returns, 0.95) * 100),
        "cvar_95_pct": _json_scalar(conditional_var(daily_returns, 0.95) * 100),
        "volatility_pct": _json_scalar(daily_volatility(daily_returns) * 100),
        "winning_trades": _json_scalar(len(wins)),
        "losing_trades": _json_scalar(len(losses)),
    }


def get_backtest(ticker: str, period: str | None = None, threshold: float | None = None) -> dict:
    cfg = Config()
    active_period = period or cfg.DATA_PERIOD
    active_threshold = threshold if threshold is not None else cfg.SIGNAL_THRESHOLD

    def factory():
        df = fetch_data(ticker, active_period, use_cache=True)

        scalers = _load_scalers(ticker)
        model = _load_model(ticker)
        feature_scaler = scalers["feature"]
        close_scaler = scalers["close"]
        scaled = feature_scaler.transform(df[cfg.FEATURE_COLS])

        n_total = len(scaled) - cfg.SEQ_LENGTH - cfg.PRED_LENGTH
        test_start = int((cfg.TRAIN_RATIO + cfg.VAL_RATIO) * n_total)

        predictions = []
        for i in range(test_start, n_total):
            x = torch.FloatTensor(scaled[i : i + cfg.SEQ_LENGTH]).unsqueeze(0).to(_device())
            current_scaled = scaled[i + cfg.SEQ_LENGTH - 1, 3]
            current_price = close_scaler.inverse_transform([[current_scaled]])[0, 0]
            actual_scaled = scaled[i + cfg.SEQ_LENGTH + cfg.PRED_LENGTH - 1, 3]
            actual_price = close_scaler.inverse_transform([[actual_scaled]])[0, 0]

            with torch.no_grad():
                pred_scaled, _ = model(x)
            pred_last = pred_scaled[0, -1].item()
            pred_price = close_scaler.inverse_transform([[pred_last]])[0, 0]

            predictions.append(
                {
                    "date": df.index[i + cfg.SEQ_LENGTH - 1],
                    "current_price": float(current_price),
                    "actual_future": float(actual_price),
                    "pred_future": float(pred_price),
                    "actual_signal": _price_to_signal(current_price, actual_price, active_threshold),
                    "pred_signal": _price_to_signal(current_price, pred_price, active_threshold),
                }
            )

        trades, equity_curve = _simulate_trades(predictions, cfg.INITIAL_CAPITAL)
        eq = equity_curve[equity_curve > 0]
        daily_returns = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([0.0])
        bench_returns = _fetch_benchmark_returns(
            cfg.BENCHMARK_TICKER,
            pd.DatetimeIndex([p["date"] for p in predictions]),
        )

        trade_rows = []
        for trade in trades[-15:]:
            trade_rows.append(
                {
                    "entry_date": _to_date(pd.Timestamp(trade["entry_date"]).date()),
                    "exit_date": _to_date(pd.Timestamp(trade["exit_date"]).date()),
                    "entry_price": float(trade["entry_price"]),
                    "exit_price": float(trade["exit_price"]),
                    "pnl": float(trade["pnl"]),
                    "pnl_pct": float(trade["pnl_pct"] * 100),
                }
            )

        equity_points = []
        pred_dates = [p["date"] for p in predictions]
        for idx, value in enumerate(eq):
            if idx == 0:
                date = pred_dates[0] if pred_dates else pd.Timestamp.utcnow()
            else:
                date = pred_dates[min(idx - 1, len(pred_dates) - 1)]
            equity_points.append({"date": _to_date(pd.Timestamp(date).date()), "equity": float(value)})

        return {
            "ticker": ticker,
            "period": active_period,
            "threshold_pct": active_threshold * 100,
            "metrics": _metrics_payload(trades, eq, daily_returns, bench_returns, cfg.INITIAL_CAPITAL),
            "recent_trades": trade_rows,
            "equity_curve": equity_points,
        }

    return _cached(900, _cache_key("backtest", ticker, active_period, active_threshold), factory)


def _screen_one_ticker(ticker: str, cfg: Config) -> dict:
    info = fetch_fundamentals(ticker)
    raw = yf.download(ticker, period="1y", progress=False)
    if raw.empty:
        raise ValueError("no data")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = add_technical_indicators(df)

    f_score = fundamental_score(info)
    t_score = technical_score(df)
    m_score = momentum_score(info, df)
    composite = (
        cfg.FUND_WEIGHT * f_score
        + cfg.TECH_WEIGHT * t_score
        + cfg.MOM_WEIGHT * m_score
    )

    return {
        "ticker": ticker,
        "name": info.get("name", ticker),
        "score": float(round(composite, 3)),
        "fundamental": float(round(f_score, 3)),
        "technical": float(round(t_score, 3)),
        "momentum": float(round(m_score, 3)),
        "pe_ratio": _to_float(info.get("pe_ratio")) or 0.0,
        "roe_pct": (_to_float(info.get("roe")) or 0.0) * 100,
        "debt_to_equity": _to_float(info.get("debt_to_equity")) or 0.0,
        "earnings_growth_pct": (_to_float(info.get("earnings_growth")) or 0.0) * 100,
        "multibagger": is_multibagger_candidate(info, cfg),
    }


def get_screener(top_n: int = 10, tickers: list[str] | None = None) -> dict:
    cfg = Config()
    active_tickers = tickers or cfg.NIFTY_STOCKS

    def factory():
        results = []
        errors = []
        max_workers = min(8, max(1, len(active_tickers)))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_screen_one_ticker, ticker, cfg): ticker
                for ticker in active_tickers
            }
            for future in concurrent.futures.as_completed(future_map):
                ticker = future_map[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    errors.append(f"{ticker}: {exc}")

        results.sort(key=lambda item: item["score"], reverse=True)
        ranked = [{"rank": idx, **row} for idx, row in enumerate(results[:top_n], start=1)]

        return {
            "count": len(results),
            "top_n": top_n,
            "results": ranked,
            "errors": errors,
        }

    ticker_key = ",".join(active_tickers)
    return _cached(900, _cache_key("screener", top_n, ticker_key), factory)
