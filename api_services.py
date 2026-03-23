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
from backtest import _fetch_benchmark_returns, _price_to_signal, _simulate_trades, _simulate_fixed_horizon_trades, _adaptive_threshold
from confidence import compute_confidence
from config import Config
from data_fetcher import add_technical_indicators, fetch_data, fetch_fundamentals
from explainability import explain_signal
from market_data import (
    ALPHA_VANTAGE_API_KEY,
    FINNHUB_API_KEY,
    GROWW_API_KEY,
    TWELVE_DATA_API_KEY,
    get_finnhub_candles,
    get_market_status,
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
    directional_accuracy,
    expectancy,
    information_coefficient,
    max_drawdown,
    profit_factor,
    sample_size_warning,
    sharpe_ratio,
    signal_accuracy,
    sortino_ratio,
    statistical_significance,
    value_at_risk,
    win_rate,
)
from model import LSTMStockPredictor
from screener import fundamental_score, is_multibagger_candidate, momentum_score, technical_score
from sentiment import classify_headlines, aggregate_sentiment
from alpha_pulse import compute_pulse, PulseResult
from exchanges import get_exchange_overview, get_market_movers, get_global_indices, list_exchanges, EXCHANGES
from train import train


_RUNTIME_CACHE: dict[str, tuple[float, object]] = {}
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


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
        "openrouter_configured": bool(OPENROUTER_API_KEY),
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


def get_news(ticker: str, limit: int = 20) -> dict:
    def factory():
        payload = get_stock_news(ticker, limit=limit)
        stories = payload.get("stories", [])

        # NEW: Run FinBERT sentiment analysis on headlines
        headlines = [s.get("title", "") or "" for s in stories]
        sentiments = classify_headlines(headlines) if headlines else []

        # Update stories with FinBERT sentiment
        for i, story in enumerate(stories):
            if i < len(sentiments):
                sent = sentiments[i]
                story["sentiment_score"] = sent["score"]
                story["sentiment_label"] = sent["label"]
                story["sentiment_method"] = sent.get("method", "unknown")
                story["sentiment_positive"] = round(sent.get("positive", 0), 3)
                story["sentiment_negative"] = round(sent.get("negative", 0), 3)
                story["sentiment_neutral"] = round(sent.get("neutral", 0), 3)

        # Aggregate
        agg = aggregate_sentiment(sentiments) if sentiments else {}
        bullish = agg.get("bullish_count", 0)
        bearish = agg.get("bearish_count", 0)
        neutral_count = agg.get("neutral_count", len(stories))

        payload["stories"] = stories
        payload["average_sentiment"] = agg.get("average_score")
        payload["average_label"] = agg.get("average_label", "neutral")
        payload["sentiment_method"] = agg.get("method", "none")
        payload["stats"] = {
            "count": len(stories),
            "bullish": bullish,
            "bearish": bearish,
            "neutral": neutral_count,
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
        x = torch.FloatTensor(scaled[-cfg.SEQ_LENGTH:]).unsqueeze(0).to(_device())

        with torch.no_grad():
            preds_scaled, attn_weights = model(x)

        preds = close_scaler.inverse_transform(preds_scaled.cpu().numpy())[0]
        attention = attn_weights.cpu().numpy()[0]
        current_price = float(df["Close"].iloc[-1])   # historical close (model input context)

        # ── Live price as percentage reference ────────────────────────────────
        # The historical close can lag the live price significantly (cache,
        # adjusted prices, or data staleness), which causes extreme % values.
        # Use the live quote price for display percentages when available.
        reference_price = current_price
        try:
            live_q = get_realtime_quote(ticker)
            lp = float(live_q.get("price") or 0)
            if lp > 0:
                reference_price = lp
        except Exception:
            pass  # keep historical close as reference

        # Stale-model detection: flag when live price differs >10 % from training close
        price_drift = abs(reference_price - current_price) / max(reference_price, 1e-6)
        model_stale_warning: str | None = None
        if price_drift > 0.10:
            model_stale_warning = (
                f"Model trained on data up to {_to_date(df.index[-1].date())}. "
                f"Live price ({reference_price:.2f}) differs {price_drift * 100:.1f}% "
                f"from historical close ({current_price:.2f}). Retrain for accurate forecasts."
            )

        pred_day5 = float(preds[-1])
        signal = _classify(reference_price, pred_day5, cfg.SIGNAL_THRESHOLD)
        pct_change = ((pred_day5 - reference_price) / reference_price) * 100
        top_steps = np.argsort(attention)[-5:][::-1]
        focus_days = [cfg.SEQ_LENGTH - int(t) for t in top_steps]

        # NEW: Multi-factor confidence with uncertainty bands
        latest_row = df.iloc[-1].to_dict()
        try:
            conf = compute_confidence(
                model=model,
                x=x,
                close_scaler=close_scaler,
                signal=signal,
                latest_row=latest_row,
                df_close=df["Close"].values,
                attention_weights=attention,
                pred_pct_change=pct_change,
                cfg=cfg,
            )
            confidence = conf["confidence_pct"]
            uncertainty = conf["uncertainty_bands"]
            confidence_factors = {
                k: {"score": round(v["score"], 3), "weight": v["weight"]}
                for k, v in conf["factors"].items()
            }
            confidence_formula = conf["formula"]
        except Exception:
            confidence = min(float(np.sort(attention)[-5:].sum() * 100), 99.0)
            uncertainty = None
            confidence_factors = None
            confidence_formula = None

        # NEW: Signal explainability
        try:
            expl = explain_signal(
                signal=signal,
                current_price=reference_price,   # use live price for explanation text
                pred_price=pred_day5,
                pred_pct=pct_change,
                latest_row=latest_row,
                currency="Rs." if ".NS" in ticker else "$",
            )
            explanation = {
                "headline": expl["headline"],
                "confluence": expl["confluence"],
                "reasoning": expl["reasoning"][:3],
                "technicals": expl["technicals"][:4],
                "watch_items": expl["watch_items"][:3],
            }
        except Exception:
            explanation = None

        result = {
            "ticker": ticker,
            "as_of": _to_date(df.index[-1].date()),
            "current_price": current_price,       # historical close (model context)
            "live_price": reference_price,         # live price (used for % display)
            "model_stale_warning": model_stale_warning,
            "predictions": [
                {
                    "day": i,
                    "price": float(price),
                    # pct relative to live price — prevents extreme values from stale history
                    "pct_change": float(((price - reference_price) / reference_price) * 100),
                }
                for i, price in enumerate(preds, start=1)
            ],
            "signal": signal,
            "direction_pct": float(pct_change),
            "confidence": confidence,
            "confidence_factors": confidence_factors,
            "confidence_formula": confidence_formula,
            "focus_days_ago": focus_days,
            "threshold_pct": cfg.SIGNAL_THRESHOLD * 100,
            "explanation": explanation,
        }

        # Add uncertainty bands if available
        if uncertainty:
            result["uncertainty_bands"] = {
                "lower": [round(v, 2) for v in uncertainty["lower"]],
                "upper": [round(v, 2) for v in uncertainty["upper"]],
                "mean": [round(v, 2) for v in uncertainty["mean"]],
            }

        return result

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
    test_ratio = getattr(cfg, "BACKTEST_TEST_RATIO", 0.30)

    def factory():
        df = fetch_data(ticker, active_period, use_cache=True)

        # Adaptive threshold if not manually specified
        if threshold is not None:
            active_threshold = threshold
            threshold_type = "manual"
        else:
            active_threshold = _adaptive_threshold(df["Close"].values)
            threshold_type = "adaptive"

        scalers = _load_scalers(ticker)
        model = _load_model(ticker)
        feature_scaler = scalers["feature"]
        close_scaler = scalers["close"]
        scaled = feature_scaler.transform(df[cfg.FEATURE_COLS])

        n_total = len(scaled) - cfg.SEQ_LENGTH - cfg.PRED_LENGTH
        test_start = int((1.0 - test_ratio) * n_total)

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

        trades, equity_curve = _simulate_fixed_horizon_trades(predictions, cfg.INITIAL_CAPITAL, hold_days=cfg.PRED_LENGTH)
        eq = equity_curve[equity_curve > 0]
        daily_returns = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([0.0])
        bench_returns = _fetch_benchmark_returns(
            cfg.BENCHMARK_TICKER,
            pd.DatetimeIndex([p["date"] for p in predictions]),
        )

        trade_rows = []
        for trade in trades[-15:]:
            row = {
                "entry_date": _to_date(pd.Timestamp(trade["entry_date"]).date()),
                "exit_date": _to_date(pd.Timestamp(trade["exit_date"]).date()),
                "entry_price": float(trade["entry_price"]),
                "exit_price": float(trade["exit_price"]),
                "pnl": float(trade["pnl"]),
                "pnl_pct": float(trade["pnl_pct"] * 100),
            }
            if "direction" in trade:
                row["direction"] = trade["direction"]
            trade_rows.append(row)

        equity_points = []
        pred_dates = [p["date"] for p in predictions]
        for idx, value in enumerate(eq):
            if idx == 0:
                date = pred_dates[0] if pred_dates else pd.Timestamp.utcnow()
            else:
                date = pred_dates[min(idx - 1, len(pred_dates) - 1)]
            equity_points.append({"date": _to_date(pd.Timestamp(date).date()), "equity": float(value)})

        # Standard metrics
        metrics = _metrics_payload(trades, eq, daily_returns, bench_returns, cfg.INITIAL_CAPITAL)

        # NEW: Prediction quality metrics
        dir_acc = directional_accuracy(predictions)
        sig_acc = signal_accuracy(predictions)
        ic = information_coefficient(predictions)
        stat_sig = statistical_significance(trades)
        sw = sample_size_warning(len(trades))

        metrics["directional_accuracy_pct"] = _json_scalar(dir_acc * 100)
        metrics["information_coefficient"] = _json_scalar(ic)
        metrics["signal_accuracy_pct"] = _json_scalar(sig_acc["overall"] * 100)
        metrics["buy_precision_pct"] = _json_scalar(sig_acc["precision_buy"] * 100)
        metrics["buy_recall_pct"] = _json_scalar(sig_acc["recall_buy"] * 100)
        metrics["buy_f1_pct"] = _json_scalar(sig_acc["f1_buy"] * 100)
        metrics["n_predictions"] = len(predictions)
        metrics["stat_significant"] = stat_sig["significant"]
        metrics["p_value"] = _json_scalar(stat_sig["p_value"])
        metrics["sample_warning"] = sw

        return {
            "ticker": ticker,
            "period": active_period,
            "threshold_pct": active_threshold * 100,
            "threshold_type": threshold_type,
            "test_ratio_pct": test_ratio * 100,
            "metrics": metrics,
            "recent_trades": trade_rows,
            "equity_curve": equity_points,
            "stat_conclusion": stat_sig["conclusion"],
        }

    # Use adaptive threshold marker in cache key
    threshold_key = threshold if threshold is not None else "adaptive"
    return _cached(900, _cache_key("backtest", ticker, active_period, threshold_key), factory)


def _screen_one_ticker(ticker: str, cfg: Config, trained_set: set | None = None) -> dict:
    info = fetch_fundamentals(ticker)
    raw = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
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

    # NEW: Model coverage disclosure
    has_model = ticker in (trained_set or set())

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
        "has_lstm_model": has_model,
    }


def get_screener(top_n: int = 10, tickers: list[str] | None = None) -> dict:
    cfg = Config()
    active_tickers = tickers or getattr(cfg, "INDIAN_STOCKS", cfg.NIFTY_STOCKS)

    def factory():
        trained_set = set(_trained_tickers())
        results = []
        errors = []
        max_workers = min(8, max(1, len(active_tickers)))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_screen_one_ticker, ticker, cfg, trained_set): ticker
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

        # Model coverage disclosure
        with_model = sum(1 for r in results if r.get("has_lstm_model"))
        without_model = len(results) - with_model

        return {
            "count": len(results),
            "top_n": top_n,
            "results": ranked,
            "errors": errors,
            "model_coverage": {
                "with_model": with_model,
                "without_model": without_model,
                "total": len(results),
                "disclosure": (
                    f"{with_model}/{len(results)} stocks have trained LSTM models. "
                    f"Scores for the other {without_model} are based on fundamental/technical analysis only "
                    f"(no LSTM prediction backing)."
                ),
            },
        }

    ticker_key = ",".join(active_tickers)
    return _cached(900, _cache_key("screener", top_n, ticker_key), factory)


# ══════════════════════════════════════════════════════════════════════════════
# LLM-POWERED STOCK INTELLIGENCE (OpenRouter)
# ══════════════════════════════════════════════════════════════════════════════

def _llm_stock_assessment(
    ticker: str,
    pulse_data: dict,
    news_data: dict | None = None,
    fundamentals: dict | None = None,
    advisory_data: dict | None = None,
) -> dict | None:
    """
    Call OpenRouter LLM to produce an intelligent stock assessment.

    Gathers all available signals (technical pulse, news, fundamentals,
    LSTM forecast) and asks the LLM to synthesise a verdict.

    Returns dict with keys:
        llm_grade   : str   - STRONG BUY / BUY / HOLD / SELL / STRONG SELL
        llm_score   : float - 0-100 conviction score
        reasoning   : str   - concise explanation
        catalysts   : list  - upcoming catalysts / risks
        action      : str   - one-line actionable recommendation
    """
    if not OPENROUTER_API_KEY:
        return None

    try:
        # ── Build context string ────────────────────────────────────────
        sections = []

        # 1. Technical Pulse snapshot
        dims = pulse_data.get("dimensions", {})
        breakdown = pulse_data.get("breakdown", {})
        sections.append(
            f"ALPHA PULSE ENGINE SCORE: {pulse_data.get('score', 'N/A')}/100 "
            f"(Grade: {pulse_data.get('grade', 'N/A')})\n"
            f"  Regime: {pulse_data.get('regime', 'N/A')}, "
            f"Momentum: {pulse_data.get('momentum_direction', 'N/A')}, "
            f"Confluence: {pulse_data.get('confluence', 'N/A')}\n"
            f"  Dimensions:"
        )
        for dim_key, dim_val in dims.items():
            detail = breakdown.get(dim_key, "")
            sections.append(f"    {dim_key}: {dim_val:.3f} — {detail}")

        if pulse_data.get("alerts"):
            sections.append(f"  Alerts: {'; '.join(pulse_data['alerts'])}")

        # 2. News headlines + sentiment
        if news_data:
            stories = news_data.get("stories", [])[:8]
            avg_sent = news_data.get("average_sentiment")
            avg_label = news_data.get("average_label", "neutral")
            sections.append(
                f"\nNEWS SENTIMENT: avg={avg_sent}, label={avg_label}, "
                f"count={news_data.get('stats', {}).get('count', 0)}"
            )
            for s in stories:
                title = s.get("title", "Untitled")
                sent = s.get("sentiment_label", "neutral")
                sections.append(f"  [{sent.upper()}] {title}")

        # 3. Company fundamentals
        if fundamentals:
            sections.append(
                f"\nFUNDAMENTALS ({fundamentals.get('name', ticker)}):\n"
                f"  Sector: {fundamentals.get('sector', 'N/A')}, "
                f"Industry: {fundamentals.get('industry', 'N/A')}\n"
                f"  Market Cap: {fundamentals.get('market_cap', 0):,.0f}, "
                f"P/E: {fundamentals.get('pe_ratio', 0):.1f}, "
                f"Forward P/E: {fundamentals.get('forward_pe', 0):.1f}\n"
                f"  ROE: {(fundamentals.get('roe', 0) or 0) * 100:.1f}%, "
                f"Debt/Equity: {fundamentals.get('debt_to_equity', 0):.2f}\n"
                f"  Earnings Growth: {(fundamentals.get('earnings_growth', 0) or 0) * 100:.1f}%, "
                f"Revenue Growth: {(fundamentals.get('revenue_growth', 0) or 0) * 100:.1f}%\n"
                f"  52W High: {fundamentals.get('fifty_two_week_high', 0)}, "
                f"52W Low: {fundamentals.get('fifty_two_week_low', 0)}, "
                f"Beta: {fundamentals.get('beta', 1.0):.2f}"
            )

        # 4. LSTM model forecast
        if advisory_data:
            sections.append(
                f"\nLSTM MODEL FORECAST:\n"
                f"  Signal: {advisory_data.get('signal', 'N/A')}, "
                f"Confidence: {advisory_data.get('confidence', 'N/A')}%\n"
                f"  Current Price: {advisory_data.get('current_price', 'N/A')}, "
                f"5-day Predicted Move: {advisory_data.get('direction_pct', 0):.2f}%"
            )
            preds = advisory_data.get("predictions", [])
            if preds:
                pred_str = ", ".join(
                    f"D{p['day']}={p['price']:.2f}({p['pct_change']:+.1f}%)"
                    for p in preds
                )
                sections.append(f"  Day-by-day: {pred_str}")

        context = "\n".join(sections)

        # ── LLM prompt ─────────────────────────────────────────────────
        system_prompt = (
            "You are Alpha Pulse AI, an expert quantitative stock analyst. "
            "You receive multi-dimensional technical, fundamental, sentiment, "
            "and ML forecast data. Your job is to synthesise all signals into "
            "a single coherent verdict.\n\n"
            "RULES:\n"
            "- Be data-driven. Cite specific numbers from the data.\n"
            "- If signals conflict, explain the conflict and weight accordingly.\n"
            "- Identify catalysts (upcoming events, sector trends, macro risks).\n"
            "- Never give financial advice. This is for educational/research use.\n"
            "- Keep reasoning concise (3-5 bullet points).\n"
            "- Output EXACTLY this JSON format (no markdown, no code blocks):\n"
            '{"llm_grade":"STRONG BUY|BUY|HOLD|SELL|STRONG SELL",'
            '"llm_score":0-100,'
            '"reasoning":"3-5 bullet points separated by |",'
            '"catalysts":"2-3 catalysts/risks separated by |",'
            '"action":"one-line actionable recommendation"}'
        )

        user_prompt = (
            f"Analyze {ticker} using ALL the data below. "
            f"Produce your verdict as the specified JSON.\n\n"
            f"{context}"
        )

        payload = {
            "model": "google/gemini-2.0-flash-001",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 500,
            "temperature": 0.3,
        }

        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://stock-pulse-dashboard.local",
                "X-Title": "Stock Pulse Alpha Engine",
            },
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        raw_text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        # Parse JSON from response (strip markdown fences if any)
        clean = raw_text
        if "```" in clean:
            clean = clean.split("```")[1] if len(clean.split("```")) > 1 else clean
            if clean.startswith("json"):
                clean = clean[4:]
            clean = clean.strip()

        import json as _json
        result = _json.loads(clean)

        # Validate and normalise
        valid_grades = {"STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"}
        grade = result.get("llm_grade", "HOLD").upper()
        if grade not in valid_grades:
            grade = "HOLD"

        score = max(0, min(100, float(result.get("llm_score", 50))))

        return {
            "llm_grade": grade,
            "llm_score": round(score, 1),
            "reasoning": result.get("reasoning", ""),
            "catalysts": result.get("catalysts", ""),
            "action": result.get("action", ""),
            "model": data.get("model", "gemini-2.0-flash"),
        }

    except Exception as exc:
        print(f"[LLM] Assessment failed for {ticker}: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# ALPHA PULSE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def get_pulse(ticker: str, period: str = "1y") -> dict:
    """Compute the Alpha Pulse Engine score for a ticker, enhanced with LLM intelligence."""
    def factory():
        df = _fetch_with_fallback(ticker, period, prefer_live=True)

        # ── Gather intelligence from all sources ──────────────────────
        # 1. News + FinBERT sentiment
        try:
            news = get_news(ticker, limit=10)
            avg_sent = news.get("average_sentiment")
        except Exception:
            news = None
            avg_sent = None

        # 2. Company fundamentals (online)
        try:
            fundamentals = fetch_fundamentals(ticker)
        except Exception:
            fundamentals = None

        # 3. LSTM model forecast (if trained)
        advisory = None
        try:
            model_status = get_model_status(ticker)
            if model_status.get("ready"):
                advisory = get_advisory(ticker, period)
        except Exception:
            advisory = None

        # ── Core 7-dimension Pulse score ──────────────────────────────
        pulse = compute_pulse(df, sentiment_score=avg_sent)

        pulse_data = {
            "ticker": ticker,
            "score": pulse.score,
            "grade": pulse.grade,
            "dimensions": pulse.dimensions,
            "breakdown": pulse.breakdown,
            "confluence": pulse.confluence,
            "regime": pulse.regime,
            "momentum_direction": pulse.momentum_direction,
            "alerts": list(pulse.alerts),
        }

        # ── LLM-powered intelligent assessment ───────────────────────
        llm = _llm_stock_assessment(
            ticker=ticker,
            pulse_data=pulse_data,
            news_data=news,
            fundamentals=fundamentals,
            advisory_data=advisory,
        )

        if llm:
            pulse_data["ai_assessment"] = llm

            # ── Fuse LLM verdict with technical score ─────────────────
            # Weighted blend: 70% technical Pulse + 30% LLM score
            tech_score = pulse.score
            llm_score = llm["llm_score"]
            fused_score = round(0.70 * tech_score + 0.30 * llm_score, 1)

            # Grade from fused score
            if fused_score >= 80:
                fused_grade = "STRONG BUY"
            elif fused_score >= 60:
                fused_grade = "BUY"
            elif fused_score >= 40:
                fused_grade = "HOLD"
            elif fused_score >= 20:
                fused_grade = "SELL"
            else:
                fused_grade = "STRONG SELL"

            pulse_data["raw_technical_score"] = tech_score
            pulse_data["raw_technical_grade"] = pulse.grade
            pulse_data["score"] = fused_score
            pulse_data["grade"] = fused_grade

            # Add to alerts if LLM and technical disagree
            if llm["llm_grade"] != pulse.grade:
                pulse_data["alerts"].append(
                    f"AI divergence: Technical={pulse.grade}, LLM={llm['llm_grade']}"
                )
        else:
            pulse_data["ai_assessment"] = None

        # ── Add fundamentals summary ──────────────────────────────────
        if fundamentals:
            pulse_data["fundamentals"] = {
                "name": fundamentals.get("name", ticker),
                "sector": fundamentals.get("sector", "N/A"),
                "industry": fundamentals.get("industry", "N/A"),
                "pe_ratio": fundamentals.get("pe_ratio", 0),
                "roe_pct": round((fundamentals.get("roe") or 0) * 100, 1),
                "debt_to_equity": fundamentals.get("debt_to_equity", 0),
                "market_cap": fundamentals.get("market_cap", 0),
                "beta": fundamentals.get("beta", 1.0),
            }

        # ── Add LSTM forecast summary ─────────────────────────────────
        if advisory:
            pulse_data["lstm_forecast"] = {
                "signal": advisory.get("signal"),
                "confidence": advisory.get("confidence"),
                "direction_pct": advisory.get("direction_pct"),
                "current_price": advisory.get("current_price"),
            }

        return pulse_data

    return _cached(120, _cache_key("pulse", ticker, period), factory)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL EXCHANGES + MARKET MOVERS
# ══════════════════════════════════════════════════════════════════════════════

def get_exchange_data(exchange_code: str) -> dict:
    """Get exchange overview including gainers/losers."""
    return get_exchange_overview(exchange_code)


def get_movers(exchange_code: str = "NSE") -> dict:
    """Get top gainers and losers for an exchange."""
    return get_market_movers(exchange_code)


def get_indices() -> list[dict]:
    """Get all global indices snapshot."""
    return get_global_indices()


def get_exchanges_list() -> list[dict]:
    """Get list of supported exchanges."""
    return list_exchanges()


# ══════════════════════════════════════════════════════════════════════════════
# CANDLES + MARKET STATUS
# ══════════════════════════════════════════════════════════════════════════════

def get_candles(ticker: str, period: str = "1y") -> dict:
    """Get OHLCV candle data for TradingView-style charts."""
    def factory():
        # Try Finnhub first
        days = {"1m": 30, "3m": 90, "6m": 180, "1y": 365, "2y": 730, "5y": 1825}.get(period, 365)
        candles = get_finnhub_candles(ticker, resolution="D", count=days)

        if candles:
            return {"ticker": ticker, "source": "finnhub", "candles": candles, "count": len(candles)}

        # Fallback to yfinance
        df = _fetch_with_fallback(ticker, period, prefer_live=True)
        yf_candles = []
        for idx, row in df.iterrows():
            yf_candles.append({
                "time": int(idx.timestamp()),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row.get("Volume", 0)),
            })
        return {"ticker": ticker, "source": "yfinance", "candles": yf_candles, "count": len(yf_candles)}

    return _cached(180, _cache_key("candles", ticker, period), factory)


def get_mkt_status() -> dict:
    """Get market open/close status."""
    return get_market_status()
