"""
inference.py - Standalone inference script for LSTM Stock Predictor.

Supports:
  - Real market data via yfinance  (pip install yfinance)
  - Synthetic demo data            (no extra dependencies)

Usage:
    python inference.py --ticker AAPL          # real data
    python inference.py --ticker MSFT --days 3 # predict 3 days
    python inference.py --demo                 # synthetic data
"""

import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from config import Config


# ── Model definition (must match training architecture) ──────────────────────
class _Attention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, lstm_out: torch.Tensor):
        weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        return context, weights


class LSTMStockPredictor(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            cfg.INPUT_DIM,
            cfg.HIDDEN_DIM,
            cfg.NUM_LAYERS,
            batch_first=True,
            dropout=cfg.DROPOUT if cfg.NUM_LAYERS > 1 else 0,
        )
        self.attention = _Attention(cfg.HIDDEN_DIM)
        self.fc = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM, 64),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(64, cfg.OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor):
        out, _ = self.lstm(x)
        context, attn = self.attention(out)
        return self.fc(context), attn


# ── Feature engineering (mirrors the notebook) ───────────────────────────────
def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_30"] = df["Close"].rolling(30).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    df["BB_Middle"] = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + std * 2
    df["BB_Lower"] = df["BB_Middle"] - std * 2
    df["Price_Change"] = df["Close"].pct_change()
    df["Volatility"] = df["Price_Change"].rolling(10).std()
    df["Volume_SMA"] = df["Volume"].rolling(10).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"]
    return df.dropna()


# ── Data loaders ─────────────────────────────────────────────────────────────
def load_real_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance and compute indicators."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Install yfinance:  pip install yfinance")
    raw = yf.download(ticker, period=period, progress=False)
    raw = raw[["Open", "High", "Low", "Close", "Volume"]]
    return _add_indicators(raw)


def load_demo_data(days: int = 1500) -> pd.DataFrame:
    """Synthetic stock data — same generator as the training notebook."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, days)))
    df = pd.DataFrame(
        {
            "Open": prices * (1 + np.random.normal(0, 0.005, days)),
            "High": prices * (1 + np.abs(np.random.normal(0, 0.01, days))),
            "Low": prices * (1 - np.abs(np.random.normal(0, 0.01, days))),
            "Close": prices,
            "Volume": np.random.randint(1_000_000, 10_000_000, days),
        },
        index=dates,
    )
    return _add_indicators(df)


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(
    df: pd.DataFrame,
    model_path: str | None = None,
    cfg: Config | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on the last SEQ_LENGTH rows of `df`.

    Returns
    -------
    predictions : np.ndarray, shape (PRED_LENGTH,)   — price in $
    attention   : np.ndarray, shape (SEQ_LENGTH,)     — attention weights
    """
    if cfg is None:
        cfg = Config()
    if model_path is None:
        model_path = cfg.MODEL_PATH

    if len(df) < cfg.SEQ_LENGTH:
        raise ValueError(
            f"Need at least {cfg.SEQ_LENGTH} rows of data, got {len(df)}."
        )

    # Scale features
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[cfg.FEATURE_COLS])
    close_scaler = MinMaxScaler()
    close_scaler.fit(df[["Close"]])

    x = torch.FloatTensor(scaled[-cfg.SEQ_LENGTH :]).unsqueeze(0)  # (1, seq, feat)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMStockPredictor(cfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        preds_scaled, attn = model(x.to(device))

    predictions = close_scaler.inverse_transform(preds_scaled.cpu().numpy())[0]
    attention = attn.cpu().numpy()[0]
    return predictions, attention


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSTM Stock Predictor — inference")
    p.add_argument("--ticker", type=str, default=None, help="e.g. AAPL, MSFT, TSLA")
    p.add_argument("--demo", action="store_true", help="Use synthetic data")
    p.add_argument("--model", type=str, default=Config.MODEL_PATH)
    p.add_argument("--days", type=int, default=Config.PRED_LENGTH,
                   help="Override prediction horizon (must match training)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = Config()
    if args.days != cfg.PRED_LENGTH:
        cfg.OUTPUT_DIM = args.days
        cfg.PRED_LENGTH = args.days

    if args.demo or args.ticker is None:
        print("Using synthetic demo data ...")
        df = load_demo_data()
    else:
        print(f"Fetching real market data for {args.ticker} ...")
        df = load_real_data(args.ticker)

    predictions, attention = predict(df, model_path=args.model, cfg=cfg)

    print(f"\nLast known close : ${df['Close'].iloc[-1]:.2f}")
    print(f"Predicted next {cfg.PRED_LENGTH} closing price(s):")
    for i, price in enumerate(predictions, 1):
        print(f"  Day +{i}: ${price:.2f}")

    top_k = min(5, cfg.SEQ_LENGTH)
    top_steps = np.argsort(attention)[-top_k:][::-1]
    days_ago = [cfg.SEQ_LENGTH - int(t) for t in top_steps]
    print(f"\nTop {top_k} attended time steps (trading days before forecast): {days_ago}")


if __name__ == "__main__":
    main()
