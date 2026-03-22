"""
advisor.py - Interactive stock advisory system.

Fetches the latest market data, runs the LSTM model, and presents
a recommendation.  **Always asks for your permission** before
recording any simulated action.

Usage:
    python advisor.py                         # default ticker
    python advisor.py --ticker AAPL
    python advisor.py --ticker TCS.NS
"""

import argparse
import os
import pickle
from datetime import datetime

import numpy as np
import torch

from config import Config
from data_fetcher import fetch_data
from model import LSTMStockPredictor


SIGNAL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}
COLOUR = {"BUY": "\033[92m", "SELL": "\033[91m", "HOLD": "\033[93m", "END": "\033[0m"}


def _classify(current: float, predicted: float, threshold: float) -> str:
    pct = (predicted - current) / current
    if pct > threshold:
        return "BUY"
    elif pct < -threshold:
        return "SELL"
    return "HOLD"


def advise(cfg: Config, ticker: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load model + scalers ──────────────────────────────────────────
    if not os.path.exists(cfg.MODEL_PATH):
        print("[ERROR] Model not found. Run  python train.py  first.")
        return
    if not os.path.exists(cfg.SCALER_PATH):
        print("[ERROR] Scalers not found. Run  python train.py  first.")
        return

    model = LSTMStockPredictor(
        input_dim=cfg.INPUT_DIM,
        hidden_dim=cfg.HIDDEN_DIM,
        num_layers=cfg.NUM_LAYERS,
        output_dim=cfg.OUTPUT_DIM,
        dropout=cfg.DROPOUT,
    ).to(device)
    model.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=device))
    model.eval()

    with open(cfg.SCALER_PATH, "rb") as f:
        scalers = pickle.load(f)
    feature_scaler = scalers["feature"]
    close_scaler = scalers["close"]

    # ── 2. Fetch latest data ─────────────────────────────────────────────
    df = fetch_data(ticker, period=cfg.DATA_PERIOD, use_cache=False)

    if len(df) < cfg.SEQ_LENGTH:
        print(f"[ERROR] Not enough data ({len(df)} rows). Need >= {cfg.SEQ_LENGTH}.")
        return

    # Current price
    current_price = df["Close"].iloc[-1]
    current_date = df.index[-1].date()

    # ── 3. Predict ───────────────────────────────────────────────────────
    scaled = feature_scaler.transform(df[cfg.FEATURE_COLS])
    x = torch.FloatTensor(scaled[-cfg.SEQ_LENGTH :]).unsqueeze(0).to(device)

    with torch.no_grad():
        preds_scaled, attn_weights = model(x)

    preds = close_scaler.inverse_transform(preds_scaled.cpu().numpy())[0]
    attn = attn_weights.cpu().numpy()[0]

    # ── 4. Signal ────────────────────────────────────────────────────────
    pred_day5 = preds[-1]
    signal = _classify(current_price, pred_day5, cfg.SIGNAL_THRESHOLD)
    pct_change = ((pred_day5 - current_price) / current_price) * 100

    # Confidence heuristic: how concentrated is the attention?
    top5_attn = np.sort(attn)[-5:].sum()
    confidence = min(top5_attn * 100, 99.0)  # rough proxy

    # ── 5. Display ───────────────────────────────────────────────────────
    c = COLOUR.get(signal, "")
    e = COLOUR["END"]

    print(f"\n{'='*60}")
    print(f"  STOCK ADVISOR - {ticker}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")
    print(f"\n  Current price ({current_date})  : ${current_price:.2f}")
    print(f"\n  Predicted next {cfg.PRED_LENGTH} days:")
    for i, p in enumerate(preds, 1):
        chg = ((p - current_price) / current_price) * 100
        arrow = "+" if chg >= 0 else ""
        print(f"    Day +{i}: ${p:.2f}  ({arrow}{chg:.2f}%)")

    print(f"\n  {'-'*40}")
    print(f"  Signal       : {c}{signal}{e}")
    print(f"  Direction    : {'+' if pct_change >= 0 else ''}{pct_change:.2f}% over {cfg.PRED_LENGTH} days")
    print(f"  Confidence   : {confidence:.1f}% (attention concentration)")
    print(f"  {'-'*40}")

    # Attention insight
    top_k = 5
    top_steps = np.argsort(attn)[-top_k:][::-1]
    days_ago = [cfg.SEQ_LENGTH - int(t) for t in top_steps]
    print(f"\n  Model is most focused on data from {days_ago} trading days ago.")

    # ── 6. Advisory (with permission) ────────────────────────────────────
    print(f"\n  {'-'*40}")
    if signal == "BUY":
        print(f"  ADVISORY: The model suggests the price may rise ~{pct_change:.2f}%")
        print(f"            over the next {cfg.PRED_LENGTH} trading days.")
        print(f"            This is NOT financial advice - do your own research.")
        print(f"  {'-'*40}")
        response = input(f"\n  Would you like to record a simulated BUY? (yes/no): ").strip().lower()
        if response in ("yes", "y"):
            _log_action(ticker, "BUY", current_price, pred_day5, confidence, cfg)
            print("  Simulated BUY recorded.")
        else:
            print("  No action taken.")

    elif signal == "SELL":
        print(f"  ADVISORY: The model suggests the price may drop ~{abs(pct_change):.2f}%")
        print(f"            over the next {cfg.PRED_LENGTH} trading days.")
        print(f"            This is NOT financial advice - do your own research.")
        print(f"  {'-'*40}")
        response = input(f"\n  Would you like to record a simulated SELL? (yes/no): ").strip().lower()
        if response in ("yes", "y"):
            _log_action(ticker, "SELL", current_price, pred_day5, confidence, cfg)
            print("  Simulated SELL recorded.")
        else:
            print("  No action taken.")

    else:
        print(f"  ADVISORY: The model sees no strong directional move.")
        print(f"            Predicted change is within +/- {cfg.SIGNAL_THRESHOLD*100:.1f}%.")
        print(f"            Suggestion: HOLD / wait for a clearer signal.")
        print(f"  {'-'*40}")
        print("  No action required.")

    print()


def _log_action(ticker, action, price, target, confidence, cfg):
    """Append simulated action to a local log file."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(cfg.RESULTS_DIR, "advisor_log.csv")

    import csv
    header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(["timestamp", "ticker", "action", "price", "target_price", "confidence"])
        w.writerow([
            datetime.now().isoformat(),
            ticker, action,
            round(price, 2), round(target, 2),
            round(confidence, 1),
        ])


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stock Advisor (interactive)")
    p.add_argument("--ticker", type=str, default=Config.DEFAULT_TICKER)
    args = p.parse_args()

    cfg = Config()
    advise(cfg, ticker=args.ticker)
