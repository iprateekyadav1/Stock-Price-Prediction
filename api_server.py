from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api_services import (
    get_advisory,
    get_ai_brief,
    get_app_status,
    get_backtest,
    get_model_status,
    get_news,
    get_price_history,
    get_quote,
    get_screener,
    get_symbol_search,
    train_starter_pack,
    train_model_for_ticker,
)
from config import Config


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
    limit: int = Query(default=8, ge=1, le=20),
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
    epochs: int | None = Query(default=20, ge=1, le=500),
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
