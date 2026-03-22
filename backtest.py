"""
backtest.py - Backtest LSTM predictions using real-world trading metrics.

Simulates actual trades: BUY when model says BUY, SELL to exit.
Computes: Sharpe, Sortino, Max Drawdown, Calmar, Win Rate, Profit Factor,
Expectancy, VaR, Alpha, Beta -- the metrics hedge funds use.

Usage:
    python backtest.py                                # default ticker
    python backtest.py --ticker RELIANCE.NS
    python backtest.py --ticker AAPL --threshold 0.01
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
import yfinance as yf

from config import Config
from data_fetcher import fetch_data
from metrics import full_report
from model import LSTMStockPredictor


def _price_to_signal(current: float, future: float, threshold: float) -> str:
    pct = (future - current) / current
    if pct > threshold:
        return "BUY"
    elif pct < -threshold:
        return "SELL"
    return "HOLD"


def _simulate_trades(
    predictions: list[dict],
    initial_capital: float,
) -> tuple[list[dict], np.ndarray]:
    """
    Walk through predictions chronologically and simulate trades.

    Rules:
    - BUY signal  -> enter long position (if not already in one)
    - SELL signal -> exit position (if in one)
    - HOLD       -> do nothing

    Returns (trades, equity_curve).
    """
    capital = initial_capital
    position = None  # None = no position
    trades = []
    equity = [capital]

    for p in predictions:
        sig = p["pred_signal"]
        price = p["current_price"]

        if sig == "BUY" and position is None:
            # Enter long: buy shares with all available capital
            shares = capital / price
            position = {"entry_price": price, "shares": shares, "entry_date": p["date"]}

        elif sig == "SELL" and position is not None:
            # Exit long
            exit_value = position["shares"] * price
            pnl = exit_value - (position["shares"] * position["entry_price"])
            pnl_pct = (price - position["entry_price"]) / position["entry_price"]
            trades.append({
                "entry_date": position["entry_date"],
                "exit_date": p["date"],
                "entry_price": position["entry_price"],
                "exit_price": price,
                "shares": position["shares"],
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            })
            capital = exit_value
            position = None

        # Update equity (mark-to-market)
        if position is not None:
            current_value = position["shares"] * price
            equity.append(current_value)
        else:
            equity.append(capital)

    # Close any open position at last price
    if position is not None and predictions:
        last_price = predictions[-1]["current_price"]
        exit_value = position["shares"] * last_price
        pnl = exit_value - (position["shares"] * position["entry_price"])
        pnl_pct = (last_price - position["entry_price"]) / position["entry_price"]
        trades.append({
            "entry_date": position["entry_date"],
            "exit_date": predictions[-1]["date"],
            "entry_price": position["entry_price"],
            "exit_price": last_price,
            "shares": position["shares"],
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        })

    return trades, np.array(equity)


def _fetch_benchmark_returns(ticker: str, df_index: pd.DatetimeIndex) -> np.ndarray:
    """Fetch benchmark (Nifty 50) daily returns aligned to the same dates."""
    try:
        start = df_index[0].strftime("%Y-%m-%d")
        end = df_index[-1].strftime("%Y-%m-%d")
        bench = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(bench.columns, pd.MultiIndex):
            bench.columns = bench.columns.droplevel(1)
        bench_close = bench["Close"].reindex(df_index, method="ffill").dropna()
        return bench_close.pct_change().dropna().values
    except Exception:
        return np.array([])


def backtest(cfg: Config, ticker: str, period: str, threshold: float | None = None):
    threshold = threshold or cfg.SIGNAL_THRESHOLD
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    currency = "Rs." if ".NS" in ticker or ".BO" in ticker else "$"

    print(f"\n{'='*60}")
    print(f"  BACKTEST - {ticker}")
    print(f"  Signal threshold : +/- {threshold*100:.2f}%")
    print(f"  Initial capital  : {currency} {cfg.INITIAL_CAPITAL:,.0f}")
    print(f"{'='*60}")

    # 1. Load data
    df = fetch_data(ticker, period, use_cache=True)

    # 2. Load scalers
    if not os.path.exists(cfg.SCALER_PATH):
        raise FileNotFoundError(f"Scalers not found. Run train.py first.")
    with open(cfg.SCALER_PATH, "rb") as f:
        scalers = pickle.load(f)
    feature_scaler = scalers["feature"]
    close_scaler = scalers["close"]
    scaled = feature_scaler.transform(df[cfg.FEATURE_COLS])

    # 3. Load model
    model = LSTMStockPredictor(
        input_dim=cfg.INPUT_DIM,
        hidden_dim=cfg.HIDDEN_DIM,
        num_layers=cfg.NUM_LAYERS,
        output_dim=cfg.OUTPUT_DIM,
        dropout=cfg.DROPOUT,
    ).to(device)
    model.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=device))
    model.eval()
    print("[BACKTEST] Model loaded.\n")

    # 4. Generate predictions on the TEST portion (last 15%)
    n_total = len(scaled) - cfg.SEQ_LENGTH - cfg.PRED_LENGTH
    test_start = int((cfg.TRAIN_RATIO + cfg.VAL_RATIO) * n_total)

    predictions = []
    for i in range(test_start, n_total):
        x = torch.FloatTensor(scaled[i : i + cfg.SEQ_LENGTH]).unsqueeze(0).to(device)

        current_scaled = scaled[i + cfg.SEQ_LENGTH - 1, 3]
        current_price = close_scaler.inverse_transform([[current_scaled]])[0, 0]

        actual_scaled = scaled[i + cfg.SEQ_LENGTH + cfg.PRED_LENGTH - 1, 3]
        actual_price = close_scaler.inverse_transform([[actual_scaled]])[0, 0]

        with torch.no_grad():
            pred_scaled, _ = model(x)
        pred_last = pred_scaled[0, -1].item()
        pred_price = close_scaler.inverse_transform([[pred_last]])[0, 0]

        a_sig = _price_to_signal(current_price, actual_price, threshold)
        p_sig = _price_to_signal(current_price, pred_price, threshold)

        predictions.append({
            "date": df.index[i + cfg.SEQ_LENGTH - 1],
            "current_price": current_price,
            "actual_future": actual_price,
            "pred_future": pred_price,
            "actual_signal": a_sig,
            "pred_signal": p_sig,
        })

    # 5. Simulate trades
    trades, equity_curve = _simulate_trades(predictions, cfg.INITIAL_CAPITAL)

    # 6. Daily returns from equity curve
    eq = equity_curve[equity_curve > 0]
    daily_returns = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([0.0])

    # 7. Benchmark returns
    bench_returns = _fetch_benchmark_returns(
        cfg.BENCHMARK_TICKER,
        pd.DatetimeIndex([p["date"] for p in predictions]),
    )

    # 8. Full report with real financial metrics
    m = full_report(
        trades=trades,
        equity_curve=eq,
        daily_returns=daily_returns,
        benchmark_returns=bench_returns,
        initial_capital=cfg.INITIAL_CAPITAL,
        currency=currency,
    )

    # 9. Print trade log
    if trades:
        print("  --- TRADE LOG (last 10) ---")
        for t in trades[-10:]:
            sym = "+" if t["pnl"] > 0 else ""
            print(
                f"  {str(t['entry_date'])[:10]} -> {str(t['exit_date'])[:10]}  "
                f"| Entry: {currency}{t['entry_price']:.0f}  "
                f"| Exit: {currency}{t['exit_price']:.0f}  "
                f"| PnL: {sym}{currency}{t['pnl']:.0f} ({sym}{t['pnl_pct']*100:.1f}%)"
            )
        print()

    # 10. Save results
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    safe_ticker = ticker.replace(".", "_")

    trade_csv = os.path.join(cfg.RESULTS_DIR, f"trades_{safe_ticker}.csv")
    pd.DataFrame(trades).to_csv(trade_csv, index=False)

    equity_csv = os.path.join(cfg.RESULTS_DIR, f"equity_{safe_ticker}.csv")
    pd.DataFrame({"equity": eq}).to_csv(equity_csv, index=False)

    print(f"[BACKTEST] Trade log saved to {trade_csv}")
    print(f"[BACKTEST] Equity curve saved to {equity_csv}")

    return m


# -- CLI -------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Backtest LSTM Stock Predictor")
    p.add_argument("--ticker", type=str, default=Config.DEFAULT_TICKER)
    p.add_argument("--period", type=str, default=Config.DATA_PERIOD)
    p.add_argument("--threshold", type=float, default=None)
    args = p.parse_args()
    cfg = Config()
    backtest(cfg, ticker=args.ticker, period=args.period, threshold=args.threshold)
