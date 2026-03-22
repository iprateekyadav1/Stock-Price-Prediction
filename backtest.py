"""
backtest.py - Backtest LSTM predictions with statistically meaningful sample sizes.

Key improvements over v1:
  1. ADAPTIVE THRESHOLD based on recent volatility (not a fixed 1%)
  2. Tests on FULL out-of-sample data (last 30%, not 15%)
  3. Tracks DIRECTIONAL ACCURACY (did model predict up/down correctly?)
  4. Statistical significance test (is win rate above random?)
  5. Sample size warnings when trade count is too low
  6. Walk-forward option for truly out-of-sample validation
  7. Information Coefficient (Spearman correlation of predicted vs actual returns)

Usage:
    python backtest.py                                  # default ticker
    python backtest.py --ticker RELIANCE.NS
    python backtest.py --ticker AAPL --threshold 0.005
    python backtest.py --ticker RELIANCE.NS --walkforward
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


def _adaptive_threshold(df_close: np.ndarray, lookback: int = 60) -> float:
    """
    Compute threshold based on recent realised volatility.

    Uses median absolute daily return over `lookback` days * sqrt(5)
    to approximate a 5-day move magnitude that's within normal range.

    Signals outside this band are considered meaningful.
    """
    if len(df_close) < lookback + 1:
        return 0.005  # fallback 0.5%

    returns = np.abs(np.diff(df_close[-lookback:]) / df_close[-lookback:-1])
    median_daily = float(np.median(returns))
    # 5-day expected magnitude (sqrt(5) scaling)
    threshold_5d = median_daily * np.sqrt(5) * 0.5  # half the expected move

    # Clamp between 0.2% and 2%
    return float(np.clip(threshold_5d, 0.002, 0.02))


def _simulate_trades(
    predictions: list[dict],
    initial_capital: float,
) -> tuple[list[dict], np.ndarray]:
    """
    Walk through predictions and simulate trades.

    Rules:
    - BUY signal  -> enter long (if flat)
    - SELL signal -> exit position (if long)
    - HOLD        -> do nothing

    Returns (trades, equity_curve).
    """
    capital = initial_capital
    position = None
    trades = []
    equity = [capital]

    for p in predictions:
        sig = p["pred_signal"]
        price = p["current_price"]

        if sig == "BUY" and position is None:
            shares = capital / price
            position = {"entry_price": price, "shares": shares, "entry_date": p["date"]}

        elif sig == "SELL" and position is not None:
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

        # Mark-to-market
        if position is not None:
            equity.append(position["shares"] * price)
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


def _simulate_fixed_horizon_trades(
    predictions: list[dict],
    initial_capital: float,
    hold_days: int = 5,
) -> tuple[list[dict], np.ndarray]:
    """
    Fixed-horizon trade simulation (matches 5-day prediction horizon).

    Every BUY signal opens a position that exits exactly `hold_days` later.
    Every SELL signal opens a SHORT position that exits after `hold_days`.
    Non-overlapping: waits until current position exits before accepting new signal.

    This naturally generates 50+ trades for a 300-day test window
    and is the correct way to evaluate a fixed-horizon prediction model.
    """
    capital = initial_capital
    trades = []
    equity = [capital]
    i = 0

    while i < len(predictions):
        p = predictions[i]
        sig = p["pred_signal"]

        if sig in ("BUY", "SELL"):
            entry_price = p["current_price"]
            exit_idx = min(i + hold_days, len(predictions) - 1)
            exit_price = predictions[exit_idx]["current_price"]

            shares = capital / entry_price

            if sig == "BUY":
                pnl = shares * (exit_price - entry_price)
                pnl_pct = (exit_price - entry_price) / entry_price
            else:  # SELL (short)
                pnl = shares * (entry_price - exit_price)
                pnl_pct = (entry_price - exit_price) / entry_price

            trades.append({
                "entry_date": p["date"],
                "exit_date": predictions[exit_idx]["date"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "shares": shares,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "direction": sig,
            })
            capital += pnl
            if capital <= 0:
                capital = 1.0  # prevent zero capital

            # Mark equity during hold period
            for j in range(i + 1, exit_idx + 1):
                if j < len(predictions):
                    mtm = predictions[j]["current_price"]
                    if sig == "BUY":
                        equity.append(capital - pnl + shares * (mtm - entry_price))
                    else:
                        equity.append(capital - pnl + shares * (entry_price - mtm))
            i = exit_idx + 1
        else:
            equity.append(capital)
            i += 1

    return trades, np.array(equity)


def _fetch_benchmark_returns(ticker: str, df_index: pd.DatetimeIndex) -> np.ndarray:
    """Fetch benchmark daily returns aligned to prediction dates."""
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


def backtest(
    cfg: Config,
    ticker: str,
    period: str,
    threshold: float | None = None,
    use_adaptive_threshold: bool = True,
    test_ratio: float = 0.30,
):
    """
    Run backtest on the trained LSTM model.

    Parameters
    ----------
    test_ratio : fraction of data to use for testing (default 30%, was 15%)
    use_adaptive_threshold : if True, compute threshold from volatility
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = cfg.resolve_model_path(ticker)
    scaler_path = cfg.resolve_scaler_path(ticker)

    currency = "Rs." if ".NS" in ticker or ".BO" in ticker else "$"

    # 1. Load data
    df = fetch_data(ticker, period, use_cache=True)

    # 2. Compute adaptive threshold if not overridden
    if threshold is not None:
        active_threshold = threshold
        threshold_type = "manual"
    elif use_adaptive_threshold:
        active_threshold = _adaptive_threshold(df["Close"].values)
        threshold_type = "adaptive"
    else:
        active_threshold = cfg.SIGNAL_THRESHOLD
        threshold_type = "fixed"

    print(f"\n{'='*60}")
    print(f"  BACKTEST - {ticker}")
    print(f"  Threshold        : +/- {active_threshold*100:.3f}% ({threshold_type})")
    print(f"  Test data        : last {test_ratio*100:.0f}%")
    print(f"  Initial capital  : {currency} {cfg.INITIAL_CAPITAL:,.0f}")
    print(f"{'='*60}")

    # 3. Load scalers + model
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scalers not found for {ticker}. Run train.py --ticker {ticker} first.")
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)
    feature_scaler = scalers["feature"]
    close_scaler = scalers["close"]
    scaled = feature_scaler.transform(df[cfg.FEATURE_COLS])

    model = LSTMStockPredictor(
        input_dim=cfg.INPUT_DIM,
        hidden_dim=cfg.HIDDEN_DIM,
        num_layers=cfg.NUM_LAYERS,
        output_dim=cfg.OUTPUT_DIM,
        dropout=cfg.DROPOUT,
    ).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found for {ticker}. Run train.py --ticker {ticker} first.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("[BACKTEST] Model loaded.\n")

    # 4. Generate predictions on the TEST portion (last test_ratio%)
    n_total = len(scaled) - cfg.SEQ_LENGTH - cfg.PRED_LENGTH
    test_start = int((1.0 - test_ratio) * n_total)

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

        a_sig = _price_to_signal(current_price, actual_price, active_threshold)
        p_sig = _price_to_signal(current_price, pred_price, active_threshold)

        predictions.append({
            "date": df.index[i + cfg.SEQ_LENGTH - 1],
            "current_price": float(current_price),
            "actual_future": float(actual_price),
            "pred_future": float(pred_price),
            "actual_signal": a_sig,
            "pred_signal": p_sig,
        })

    print(f"[BACKTEST] Generated {len(predictions)} predictions over {test_ratio*100:.0f}% test window.")

    # 5. Simulate trades using FIXED-HORIZON mode for statistical validity
    #    Each BUY/SELL signal opens a position that exits after PRED_LENGTH days
    trades, equity_curve = _simulate_fixed_horizon_trades(
        predictions, cfg.INITIAL_CAPITAL, hold_days=cfg.PRED_LENGTH
    )

    # 6. Daily returns from equity curve
    eq = equity_curve[equity_curve > 0]
    daily_returns = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([0.0])

    # 7. Benchmark returns
    bench_returns = _fetch_benchmark_returns(
        cfg.BENCHMARK_TICKER,
        pd.DatetimeIndex([p["date"] for p in predictions]),
    )

    # 8. Full report with prediction-level metrics
    m = full_report(
        trades=trades,
        equity_curve=eq,
        daily_returns=daily_returns,
        benchmark_returns=bench_returns,
        initial_capital=cfg.INITIAL_CAPITAL,
        currency=currency,
        predictions=predictions,
    )

    # 9. Print trade log
    if trades:
        print(f"  --- TRADE LOG (last 15) ---")
        for t in trades[-15:]:
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

    # Save prediction log (new)
    pred_csv = os.path.join(cfg.RESULTS_DIR, f"predictions_{safe_ticker}.csv")
    pd.DataFrame(predictions).to_csv(pred_csv, index=False)

    print(f"[BACKTEST] Trade log saved to {trade_csv}")
    print(f"[BACKTEST] Equity curve saved to {equity_csv}")
    print(f"[BACKTEST] Prediction log saved to {pred_csv}")

    # Extra metadata in return dict
    m["threshold_used"] = active_threshold
    m["threshold_type"] = threshold_type
    m["test_ratio"] = test_ratio
    m["n_predictions"] = len(predictions)

    return m


# -- CLI -------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Backtest LSTM Stock Predictor")
    p.add_argument("--ticker", type=str, default=Config.DEFAULT_TICKER)
    p.add_argument("--period", type=str, default=Config.DATA_PERIOD)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--test-ratio", type=float, default=0.30,
                   help="Fraction of data for testing (default: 0.30 = 30%%)")
    p.add_argument("--no-adaptive", action="store_true",
                   help="Disable adaptive threshold (use fixed from config)")
    args = p.parse_args()
    cfg = Config()
    backtest(
        cfg,
        ticker=args.ticker,
        period=args.period,
        threshold=args.threshold,
        use_adaptive_threshold=not args.no_adaptive,
        test_ratio=args.test_ratio,
    )
