from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api_services import (
    get_advisory,
    get_ai_brief,
    get_app_status,
    get_backtest,
    get_candles,
    get_exchange_data,
    get_exchanges_list,
    get_indices,
    get_mkt_status,
    get_model_status,
    get_movers,
    get_news,
    get_price_history,
    get_pulse,
    get_quote,
    get_screener,
    get_symbol_search,
    train_starter_pack,
    train_model_for_ticker,
)
from config import Config
from live_feed import get_live_engine


app = FastAPI(title="LSTM Stock Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIST = Path(__file__).resolve().parent / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"


@app.get("/api/health")
def health():
    payload = get_app_status()
    payload["frontend_ready"] = FRONTEND_INDEX.exists()
    payload["frontend_dir"] = str(FRONTEND_DIST)
    return payload


@app.get("/api/model-status")
def model_status(
    ticker: str = Query(default=Config.DEFAULT_TICKER),
):
    return get_model_status(ticker)


@app.get("/api/history")
def history(
    ticker: str = Query(default=Config.DEFAULT_TICKER),
    period: str = Query(default=Config.DATA_PERIOD),
):
    try:
        return get_price_history(ticker, period)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/quote")
def quote(
    ticker: str = Query(default=Config.DEFAULT_TICKER),
):
    try:
        return get_quote(ticker)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/news")
def news(
    ticker: str = Query(default=Config.DEFAULT_TICKER),
    limit: int = Query(default=20, ge=1, le=50),
):
    try:
        return get_news(ticker, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/symbols")
def symbols(
    query: str = Query(default=""),
    limit: int = Query(default=12, ge=1, le=40),
):
    try:
        return get_symbol_search(query=query, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/train")
def train_model(
    ticker: str = Query(default=Config.DEFAULT_TICKER),
    period: str = Query(default=Config.DATA_PERIOD),
    epochs: int | None = Query(default=30, ge=1, le=500),
):
    try:
        return train_model_for_ticker(ticker=ticker, period=period, epochs=epochs)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/train-starter-pack")
def train_defaults(
    period: str = Query(default=Config.DATA_PERIOD),
    epochs: int | None = Query(default=50, ge=1, le=500),
):
    try:
        return train_starter_pack(period=period, epochs=epochs)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/advisory")
def advisory(
    ticker: str = Query(default=Config.DEFAULT_TICKER),
    period: str = Query(default=Config.DATA_PERIOD),
):
    try:
        return get_advisory(ticker, period)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/ai-brief")
def ai_brief(
    ticker: str = Query(default=Config.DEFAULT_TICKER),
    period: str = Query(default=Config.DATA_PERIOD),
):
    try:
        return get_ai_brief(ticker, period)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/backtest")
def backtest(
    ticker: str = Query(default=Config.DEFAULT_TICKER),
    period: str = Query(default=Config.DATA_PERIOD),
    threshold: float | None = Query(default=None),
):
    try:
        return get_backtest(ticker, period, threshold)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/screener")
def screener(
    top_n: int = Query(default=10, ge=1, le=30),
    tickers: str | None = Query(default=None, description="Comma-separated ticker list"),
):
    try:
        ticker_list = [item.strip() for item in tickers.split(",")] if tickers else None
        return get_screener(top_n=top_n, tickers=ticker_list)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/pulse")
def pulse(
    ticker: str = Query(default=Config.DEFAULT_TICKER),
    period: str = Query(default="1y"),
):
    try:
        return get_pulse(ticker, period)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/exchanges")
def exchanges():
    return get_exchanges_list()


@app.get("/api/exchange/{code}")
def exchange_detail(code: str):
    try:
        return get_exchange_data(code)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/movers")
def movers(
    exchange: str = Query(default="NSE"),
):
    try:
        return get_movers(exchange)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/indices")
def indices():
    try:
        return get_indices()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/candles")
def candles(
    ticker: str = Query(default=Config.DEFAULT_TICKER),
    period: str = Query(default="1y"),
):
    try:
        return get_candles(ticker, period)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/market-status")
def market_status():
    return get_mkt_status()


# ── Live Feed Endpoints ──────────────────────────────────────────────────

@app.get("/api/live/snapshot")
def live_snapshot(
    ticker: str = Query(default=Config.DEFAULT_TICKER),
):
    """Get a single real-time inference snapshot for a ticker."""
    engine = get_live_engine()
    snapshot = engine.get_snapshot(ticker)
    if snapshot is None:
        return {"ticker": ticker, "status": "initializing", "model_ready": False}
    return {
        "ticker": snapshot.ticker,
        "timestamp": snapshot.timestamp,
        "price": snapshot.price,
        "change": snapshot.change,
        "change_pct": snapshot.change_pct,
        "signal": snapshot.signal,
        "direction_pct": snapshot.direction_pct,
        "confidence": snapshot.confidence,
        "predictions": snapshot.predictions,
        "model_ready": snapshot.model_ready,
        "feed_active": snapshot.feed_active,
        "candle_count": snapshot.candle_count,
        "source": snapshot.source,
    }


@app.get("/api/live/stream")
def live_stream(
    ticker: str = Query(default=Config.DEFAULT_TICKER),
    interval: int = Query(default=30, ge=5, le=300),
):
    """SSE stream for real-time LSTM predictions."""
    engine = get_live_engine()
    return StreamingResponse(
        engine.stream(ticker, interval=interval),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/live/status")
def live_status():
    """Get live feed engine status."""
    engine = get_live_engine()
    return {
        "running": engine.is_running,
        "tracked_tickers": engine.get_tracked_tickers(),
        "update_interval": engine.update_interval,
        "snapshots": {
            t: {
                "price": s.price,
                "signal": s.signal,
                "model_ready": s.model_ready,
            }
            for t, s in engine.get_all_snapshots().items()
        },
    }


@app.post("/api/live/start")
def live_start():
    """Start the live feed engine."""
    engine = get_live_engine()
    engine.start()
    return {"status": "started", "running": engine.is_running}


@app.post("/api/live/register")
def live_register(
    ticker: str = Query(default=Config.DEFAULT_TICKER),
):
    """Register a ticker for live tracking."""
    engine = get_live_engine()
    ok = engine.register(ticker)
    if not ok:
        raise HTTPException(status_code=400, detail="Max tickers reached")
    return {"status": "registered", "ticker": ticker}


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets")


@app.get("/")
def dashboard_index():
    if not FRONTEND_INDEX.exists():
        raise HTTPException(status_code=404, detail="Frontend build not found. Run the frontend build first.")
    return FileResponse(FRONTEND_INDEX)


@app.get("/{full_path:path}")
def dashboard_spa_fallback(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not Found")

    target = FRONTEND_DIST / full_path
    if target.exists() and target.is_file():
        return FileResponse(target)

    if not FRONTEND_INDEX.exists():
        raise HTTPException(status_code=404, detail="Frontend build not found. Run the frontend build first.")
    return FileResponse(FRONTEND_INDEX)
