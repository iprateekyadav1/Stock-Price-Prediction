"""
advisor.py - Interactive stock advisory system with full explainability.

Fetches the latest market data, runs the LSTM model, and presents
a recommendation with:
  - Multi-factor confidence score (transparent formula)
  - Uncertainty bands (prediction range, not just point forecast)
  - Technical explainability (why the signal, what agrees/disagrees)
  - Risk warnings

**Always asks for your permission** before recording any simulated action.

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
from confidence import compute_confidence
from data_fetcher import fetch_data
from explainability import explain_signal
from model import LSTMStockPredictor


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
    model_path = cfg.resolve_model_path(ticker)
    scaler_path = cfg.resolve_scaler_path(ticker)

    currency = "Rs." if ".NS" in ticker or ".BO" in ticker else "$"

    # ── 1. Load model + scalers ──────────────────────────────────────────
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found for {ticker}. Run python train.py --ticker {ticker} first.")
        return
    if not os.path.exists(scaler_path):
        print(f"[ERROR] Scalers not found for {ticker}. Run python train.py --ticker {ticker} first.")
        return

    model = LSTMStockPredictor(
        input_dim=cfg.INPUT_DIM,
        hidden_dim=cfg.HIDDEN_DIM,
        num_layers=cfg.NUM_LAYERS,
        output_dim=cfg.OUTPUT_DIM,
        dropout=cfg.DROPOUT,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)
    feature_scaler = scalers["feature"]
    close_scaler = scalers["close"]

    # ── 2. Fetch latest data ─────────────────────────────────────────────
    df = fetch_data(ticker, period=cfg.DATA_PERIOD, use_cache=False)

    if len(df) < cfg.SEQ_LENGTH:
        print(f"[ERROR] Not enough data ({len(df)} rows). Need >= {cfg.SEQ_LENGTH}.")
        return

    current_price = float(df["Close"].iloc[-1])
    current_date = df.index[-1].date()

    # ── 3. Predict ───────────────────────────────────────────────────────
    scaled = feature_scaler.transform(df[cfg.FEATURE_COLS])
    x = torch.FloatTensor(scaled[-cfg.SEQ_LENGTH:]).unsqueeze(0).to(device)

    with torch.no_grad():
        preds_scaled, attn_weights = model(x)

    preds = close_scaler.inverse_transform(preds_scaled.cpu().numpy())[0]
    attn = attn_weights.cpu().numpy()[0]

    # ── 4. Signal ────────────────────────────────────────────────────────
    pred_day5 = float(preds[-1])
    signal = _classify(current_price, pred_day5, cfg.SIGNAL_THRESHOLD)
    pct_change = ((pred_day5 - current_price) / current_price) * 100

    # ── 5. Multi-factor confidence ───────────────────────────────────────
    latest_row = df.iloc[-1].to_dict()
    conf = compute_confidence(
        model=model,
        x=x,
        close_scaler=close_scaler,
        signal=signal,
        latest_row=latest_row,
        df_close=df["Close"].values,
        attention_weights=attn,
        pred_pct_change=pct_change,
        cfg=cfg,
    )

    # ── 6. Explainability ────────────────────────────────────────────────
    explanation = explain_signal(
        signal=signal,
        current_price=current_price,
        pred_price=pred_day5,
        pred_pct=pct_change,
        latest_row=latest_row,
        confidence_factors=conf["factors"],
        currency=currency,
    )

    # ── 7. Display ───────────────────────────────────────────────────────
    c = COLOUR.get(signal, "")
    e = COLOUR["END"]
    bands = conf["uncertainty_bands"]

    print(f"\n{'='*65}")
    print(f"  STOCK ADVISOR - {ticker}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*65}")
    print(f"\n  Current price ({current_date})  : {currency}{current_price:,.2f}")

    # Prediction with uncertainty bands
    print(f"\n  Predicted next {cfg.PRED_LENGTH} days (with 80% confidence interval):")
    for i, p_price in enumerate(preds, 1):
        chg = ((p_price - current_price) / current_price) * 100
        lo = bands["lower"][i-1]
        hi = bands["upper"][i-1]
        print(
            f"    Day +{i}: {currency}{p_price:,.2f}  ({'+' if chg >= 0 else ''}{chg:.2f}%)  "
            f"  [{currency}{lo:,.2f} - {currency}{hi:,.2f}]"
        )

    print(f"\n  {'-'*55}")
    print(f"  Signal       : {c}{signal}{e}")
    print(f"  Direction    : {'+' if pct_change >= 0 else ''}{pct_change:.2f}% over {cfg.PRED_LENGTH} days")
    print(f"  Confidence   : {conf['confidence_pct']:.1f}%")
    print(f"  {'-'*55}")

    # Confidence breakdown
    print(f"\n  --- CONFIDENCE BREAKDOWN ---")
    print(f"  {conf['formula']}")
    factors = conf["factors"]
    for name, f_data in factors.items():
        label = name.replace("_", " ").title()
        extra = ""
        if name == "technical_alignment":
            agrees = ", ".join(f_data.get("agrees", []))
            disagrees = ", ".join(f_data.get("disagrees", []))
            extra = f"  Agrees: [{agrees}]  Conflicts: [{disagrees}]"
        elif name == "volatility_regime":
            extra = f"  Regime: {f_data.get('regime', 'unknown')}"
        print(f"    {label:<25s}: {f_data['score']:.2f} x {f_data['weight']:.0%} = {f_data['weighted']:.3f}")
        if extra:
            print(f"      {extra}")

    # Explainability
    print(f"\n  --- SIGNAL EXPLANATION ---")
    print(f"  {explanation['headline']}")
    print(f"  Confluence: {explanation['confluence'].upper()} ({explanation['supports']} agree, {explanation['conflicts']} conflict)")
    for reason in explanation["reasoning"]:
        print(f"    - {reason}")
    print(f"\n  Technical summary:")
    for tech in explanation["technicals"]:
        print(f"    - {tech}")

    # Watch items
    print(f"\n  --- WATCH ITEMS ---")
    for item in explanation["watch_items"]:
        print(f"    - {item}")

    # Attention insight
    top_k = 5
    top_steps = np.argsort(attn)[-top_k:][::-1]
    days_ago = [cfg.SEQ_LENGTH - int(t) for t in top_steps]
    print(f"\n  Model focus: data from {days_ago} trading days ago.")

    # ── 8. Advisory (with permission) ────────────────────────────────────
    print(f"\n  {'-'*55}")
    print(f"  {explanation['risk_warning']}")
    print(f"  {'-'*55}")

    if signal == "BUY":
        response = input(f"\n  Would you like to record a simulated BUY? (yes/no): ").strip().lower()
        if response in ("yes", "y"):
            _log_action(ticker, "BUY", current_price, pred_day5, conf["confidence_pct"], cfg)
            print("  Simulated BUY recorded.")
        else:
            print("  No action taken.")
    elif signal == "SELL":
        response = input(f"\n  Would you like to record a simulated SELL? (yes/no): ").strip().lower()
        if response in ("yes", "y"):
            _log_action(ticker, "SELL", current_price, pred_day5, conf["confidence_pct"], cfg)
            print("  Simulated SELL recorded.")
        else:
            print("  No action taken.")
    else:
        print(f"\n  HOLD -- no action required. Wait for a clearer signal.")

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
