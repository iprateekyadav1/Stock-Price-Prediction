"""
Live Feed Engine — Real-time data pipeline for LSTM inference.

Architecture:
  1. LiveFeedEngine polls real-time price data at configurable intervals
  2. Maintains a rolling window of SEQ_LENGTH candles per ticker
  3. Feeds through trained LSTM model for streaming predictions
  4. Exposes SSE (Server-Sent Events) endpoint for frontend consumption

Usage:
  # Programmatic
  engine = LiveFeedEngine()
  engine.register("RELIANCE.NS")
  snapshot = engine.get_snapshot("RELIANCE.NS")

  # Via API — SSE stream
  GET /api/live/stream?ticker=RELIANCE.NS

  # Via API — single snapshot
  GET /api/live/snapshot?ticker=RELIANCE.NS
"""

from __future__ import annotations

import json
import os
import pickle
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Generator

import numpy as np
import pandas as pd
import torch
import yfinance as yf

from config import Config
from data_fetcher import add_technical_indicators
from model import LSTMStockPredictor


@dataclass
class LiveSnapshot:
    """A single real-time inference result."""
    ticker: str
    timestamp: float
    price: float
    change: float
    change_pct: float
    predictions: list[dict]       # [{day: 1, price: X, pct: Y}, ...]
    signal: str                   # BUY / SELL / HOLD
    direction_pct: float          # predicted 5-day move %
    confidence: float | None
    model_ready: bool
    feed_active: bool
    candle_count: int             # how many candles in the rolling window
    update_interval: int          # seconds between updates
    source: str                   # data source used


@dataclass
class TickerFeed:
    """Internal state for a single tracked ticker."""
    ticker: str
    df: pd.DataFrame | None = None
    last_update: float = 0.0
    last_price: float = 0.0
    last_snapshot: LiveSnapshot | None = None
    error_count: int = 0
    active: bool = True


class LiveFeedEngine:
    """
    Real-time inference engine.

    - Maintains a rolling DataFrame per ticker (SEQ_LENGTH + buffer rows)
    - Polls yfinance/Groww for latest prices
    - Runs LSTM inference on each poll
    - Thread-safe for concurrent API access
    """

    def __init__(
        self,
        update_interval: int = 30,
        max_tickers: int = 50,
        cfg: Config | None = None,
    ):
        self.cfg = cfg or Config()
        self.update_interval = update_interval
        self.max_tickers = max_tickers
        self._feeds: dict[str, TickerFeed] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model cache
        self._models: dict[str, LSTMStockPredictor] = {}
        self._scalers: dict[str, dict] = {}

    # ── Model loading ────────────────────────────────────────────────

    def _load_model(self, ticker: str) -> LSTMStockPredictor | None:
        if ticker in self._models:
            return self._models[ticker]

        model_path = self.cfg.resolve_model_path(ticker)
        if not os.path.exists(model_path):
            return None

        model = LSTMStockPredictor(
            input_dim=self.cfg.INPUT_DIM,
            hidden_dim=self.cfg.HIDDEN_DIM,
            num_layers=self.cfg.NUM_LAYERS,
            output_dim=self.cfg.OUTPUT_DIM,
            dropout=self.cfg.DROPOUT,
        ).to(self._device)
        model.load_state_dict(torch.load(model_path, map_location=self._device))
        model.eval()
        self._models[ticker] = model
        return model

    def _load_scalers(self, ticker: str) -> dict | None:
        if ticker in self._scalers:
            return self._scalers[ticker]

        scaler_path = self.cfg.resolve_scaler_path(ticker)
        if not os.path.exists(scaler_path):
            return None

        with open(scaler_path, "rb") as f:
            scalers = pickle.load(f)
        self._scalers[ticker] = scalers
        return scalers

    # ── Data fetching ────────────────────────────────────────────────

    def _fetch_initial_data(self, ticker: str) -> pd.DataFrame | None:
        """Fetch enough historical data to fill the sequence window."""
        try:
            # Need at least SEQ_LENGTH + 30 rows for indicators to warm up
            raw = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True)
            if raw.empty:
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df = add_technical_indicators(df)
            return df
        except Exception:
            return None

    def _fetch_latest_candle(self, ticker: str) -> dict | None:
        """Fetch the most recent intraday candle."""
        try:
            obj = yf.Ticker(ticker)
            hist = obj.history(period="1d", interval="1m")
            if hist.empty:
                # Fallback to daily
                hist = obj.history(period="5d", interval="1d")
                if hist.empty:
                    return None

            latest = hist.iloc[-1]
            return {
                "Open": float(latest["Open"]),
                "High": float(latest["High"]),
                "Low": float(latest["Low"]),
                "Close": float(latest["Close"]),
                "Volume": int(latest.get("Volume", 0)),
                "timestamp": hist.index[-1],
            }
        except Exception:
            return None

    # ── Inference ────────────────────────────────────────────────────

    def _run_inference(self, ticker: str, df: pd.DataFrame) -> dict | None:
        """Run LSTM inference on the latest data window."""
        model = self._load_model(ticker)
        scalers = self._load_scalers(ticker)
        if model is None or scalers is None:
            return None

        if len(df) < self.cfg.SEQ_LENGTH:
            return None

        try:
            feature_scaler = scalers["feature"]
            close_scaler = scalers["close"]

            scaled = feature_scaler.transform(df[self.cfg.FEATURE_COLS].tail(self.cfg.SEQ_LENGTH))
            x = torch.FloatTensor(scaled).unsqueeze(0).to(self._device)

            with torch.no_grad():
                preds_scaled, attn = model(x)

            preds = close_scaler.inverse_transform(preds_scaled.cpu().numpy())[0]
            current_price = float(df["Close"].iloc[-1])
            pred_day5 = float(preds[-1])
            pct_change = ((pred_day5 - current_price) / current_price) * 100

            # Signal classification
            threshold = self.cfg.SIGNAL_THRESHOLD
            if pct_change > threshold * 100:
                signal = "BUY"
            elif pct_change < -threshold * 100:
                signal = "SELL"
            else:
                signal = "HOLD"

            predictions = []
            for i, price in enumerate(preds, start=1):
                predictions.append({
                    "day": i,
                    "price": round(float(price), 2),
                    "pct_change": round(((float(price) - current_price) / current_price) * 100, 2),
                })

            # Simple confidence from attention weights
            attention = attn.cpu().numpy()[0]
            confidence = min(float(np.sort(attention)[-5:].sum() * 100), 99.0)

            return {
                "predictions": predictions,
                "signal": signal,
                "direction_pct": round(pct_change, 2),
                "confidence": round(confidence, 1),
                "current_price": current_price,
            }
        except Exception:
            return None

    # ── Feed management ──────────────────────────────────────────────

    def register(self, ticker: str) -> bool:
        """Register a ticker for live tracking."""
        with self._lock:
            if ticker in self._feeds:
                self._feeds[ticker].active = True
                return True

            if len(self._feeds) >= self.max_tickers:
                return False

            feed = TickerFeed(ticker=ticker)

            # Fetch initial data
            df = self._fetch_initial_data(ticker)
            if df is not None and len(df) >= self.cfg.SEQ_LENGTH:
                feed.df = df
                feed.last_price = float(df["Close"].iloc[-1])
            else:
                feed.df = None

            self._feeds[ticker] = feed
            return True

    def unregister(self, ticker: str) -> None:
        """Stop tracking a ticker."""
        with self._lock:
            if ticker in self._feeds:
                self._feeds[ticker].active = False

    def _update_ticker(self, ticker: str) -> None:
        """Poll fresh data and run inference for a single ticker."""
        feed = self._feeds.get(ticker)
        if feed is None or not feed.active:
            return

        now = time.time()
        if (now - feed.last_update) < self.update_interval:
            return

        # Fetch latest candle
        candle = self._fetch_latest_candle(ticker)
        if candle is None:
            feed.error_count += 1
            return

        feed.error_count = 0
        feed.last_update = now
        feed.last_price = candle["Close"]

        # If we don't have initial data, try fetching it
        if feed.df is None:
            df = self._fetch_initial_data(ticker)
            if df is not None and len(df) >= self.cfg.SEQ_LENGTH:
                feed.df = df
            else:
                return

        # Check if model exists
        model_path = self.cfg.resolve_model_path(ticker)
        model_ready = os.path.exists(model_path)

        # Run inference
        inference = None
        if model_ready and feed.df is not None:
            inference = self._run_inference(ticker, feed.df)

        # Calculate change
        prev_close = float(feed.df["Close"].iloc[-2]) if feed.df is not None and len(feed.df) >= 2 else candle["Close"]
        change = candle["Close"] - prev_close
        change_pct = (change / prev_close) * 100 if prev_close else 0

        # Build snapshot
        feed.last_snapshot = LiveSnapshot(
            ticker=ticker,
            timestamp=now,
            price=candle["Close"],
            change=round(change, 2),
            change_pct=round(change_pct, 2),
            predictions=inference["predictions"] if inference else [],
            signal=inference["signal"] if inference else "N/A",
            direction_pct=inference["direction_pct"] if inference else 0.0,
            confidence=inference["confidence"] if inference else None,
            model_ready=model_ready,
            feed_active=True,
            candle_count=len(feed.df) if feed.df is not None else 0,
            update_interval=self.update_interval,
            source="yfinance",
        )

    def _poll_loop(self) -> None:
        """Background polling loop."""
        while self._running:
            with self._lock:
                tickers = [t for t, f in self._feeds.items() if f.active]

            for ticker in tickers:
                try:
                    self._update_ticker(ticker)
                except Exception:
                    pass

            time.sleep(max(5, self.update_interval // 2))

    # ── Control ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background polling thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background polling thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Snapshot access ──────────────────────────────────────────────

    def get_snapshot(self, ticker: str) -> LiveSnapshot | None:
        """Get the latest snapshot for a ticker."""
        with self._lock:
            feed = self._feeds.get(ticker)
            if feed is None:
                # Auto-register and do first update
                self.register(ticker)
                feed = self._feeds.get(ticker)
                if feed:
                    self._update_ticker(ticker)
                    return feed.last_snapshot
                return None

            # If stale, update now
            if (time.time() - feed.last_update) > self.update_interval:
                self._update_ticker(ticker)

            return feed.last_snapshot

    def get_all_snapshots(self) -> dict[str, LiveSnapshot]:
        """Get snapshots for all tracked tickers."""
        with self._lock:
            return {
                t: f.last_snapshot
                for t, f in self._feeds.items()
                if f.last_snapshot is not None
            }

    def get_tracked_tickers(self) -> list[str]:
        """List all currently tracked tickers."""
        with self._lock:
            return [t for t, f in self._feeds.items() if f.active]

    def stream(self, ticker: str, interval: int | None = None) -> Generator[str, None, None]:
        """
        SSE generator for real-time streaming.

        Yields JSON-encoded snapshots at the configured interval.
        Use with FastAPI's StreamingResponse.
        """
        poll = interval or self.update_interval
        self.register(ticker)

        while True:
            snapshot = self.get_snapshot(ticker)
            if snapshot:
                data = {
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
                    "source": snapshot.source,
                }
                yield f"data: {json.dumps(data)}\n\n"
            else:
                yield f"data: {json.dumps({'ticker': ticker, 'status': 'initializing'})}\n\n"

            time.sleep(poll)


# ── Singleton ─────────────────────────────────────────────────────────
_engine: LiveFeedEngine | None = None


def get_live_engine() -> LiveFeedEngine:
    """Get or create the singleton LiveFeedEngine."""
    global _engine
    if _engine is None:
        _engine = LiveFeedEngine(update_interval=30)
    return _engine
