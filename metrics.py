"""
metrics.py - Real-world quantitative trading metrics.

All metrics used by professional algo traders and hedge funds.
Zero external dependencies beyond numpy/pandas/scipy.

Portfolio Metrics    : Sharpe, Sortino, Calmar, Max Drawdown, CAGR, Alpha, Beta
Trade Metrics        : Win Rate, Profit Factor, Expectancy, Avg Win/Loss Ratio
Risk Metrics         : Value at Risk (VaR), Conditional VaR, Daily Volatility
Prediction Metrics   : Directional Accuracy, Information Coefficient, Statistical Significance
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from config import Config


# ===========================================================================
#  PORTFOLIO / STRATEGY METRICS
# ===========================================================================

def sharpe_ratio(
    returns: np.ndarray,
    risk_free: float | None = None,
    trading_days: int = 252,
) -> float:
    """
    Sharpe Ratio = (mean excess return) / std(returns) * sqrt(252)

    Benchmark: >1.0 = acceptable, >2.0 = very good, >3.0 = excellent
    """
    rf = risk_free if risk_free is not None else Config.RISK_FREE_RATE
    excess = returns - rf
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(trading_days))


def sortino_ratio(
    returns: np.ndarray,
    risk_free: float | None = None,
    trading_days: int = 252,
) -> float:
    """
    Sortino Ratio = (mean excess return) / downside_std * sqrt(252)

    Like Sharpe but only penalises downside volatility.
    Benchmark: >2.0 = good
    """
    rf = risk_free if risk_free is not None else Config.RISK_FREE_RATE
    excess = returns - rf
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    down_std = np.std(downside, ddof=1)
    if down_std == 0:
        return 0.0
    return float(np.mean(excess) / down_std * np.sqrt(trading_days))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Maximum Drawdown = largest peak-to-trough decline (as positive fraction).

    Example: 0.15 means the portfolio dropped 15% from its peak.
    """
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / np.where(peak == 0, 1, peak)
    return float(np.max(dd))


def calmar_ratio(equity_curve: np.ndarray, trading_days: int = 252) -> float:
    """
    Calmar Ratio = CAGR / Max Drawdown

    Benchmark: >1.0 = acceptable, >3.0 = elite
    """
    mdd = max_drawdown(equity_curve)
    if mdd == 0:
        return 0.0
    annual_ret = cagr(equity_curve, trading_days)
    return float(annual_ret / mdd)


def cagr(equity_curve: np.ndarray, trading_days: int = 252) -> float:
    """
    Compound Annual Growth Rate.

    CAGR = (final / initial) ^ (252 / n_days) - 1
    """
    if len(equity_curve) < 2 or equity_curve[0] == 0:
        return 0.0
    n = len(equity_curve)
    total_return = equity_curve[-1] / equity_curve[0]
    if total_return <= 0:
        return -1.0
    years = n / trading_days
    return float(total_return ** (1 / years) - 1)


def compute_alpha_beta(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    risk_free: float | None = None,
    trading_days: int = 252,
) -> tuple[float, float]:
    """
    Alpha & Beta via linear regression (CAPM).

    Alpha: excess return independent of market (annualised).
    Beta:  sensitivity to market moves (1.0 = same as market).
    """
    rf = risk_free if risk_free is not None else Config.RISK_FREE_RATE
    n = min(len(strategy_returns), len(benchmark_returns))
    sr = strategy_returns[:n] - rf
    br = benchmark_returns[:n] - rf

    if len(br) < 2 or np.std(br) == 0:
        return 0.0, 1.0

    beta = float(np.cov(sr, br)[0, 1] / np.var(br))
    alpha_daily = float(np.mean(sr) - beta * np.mean(br))
    alpha_annual = alpha_daily * trading_days
    return alpha_annual, beta


# ===========================================================================
#  TRADE-LEVEL METRICS
# ===========================================================================

def win_rate(trades: list[dict]) -> float:
    """
    Win Rate = winning trades / total trades (as fraction).

    Each trade dict must have key 'pnl'.
    """
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t["pnl"] > 0)
    return wins / len(trades)


def profit_factor(trades: list[dict]) -> float:
    """
    Profit Factor = gross profit / gross loss.

    >1.0 = profitable, 1.5-2.5 = solid, >2.5 = excellent.
    """
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def expectancy(trades: list[dict]) -> float:
    """
    Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

    Expected Rs. earned per trade on average.
    Positive = strategy has edge.
    """
    if not trades:
        return 0.0
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [abs(t["pnl"]) for t in trades if t["pnl"] < 0]

    wr = len(wins) / len(trades)
    lr = 1 - wr
    avg_w = np.mean(wins) if wins else 0.0
    avg_l = np.mean(losses) if losses else 0.0

    return float(wr * avg_w - lr * avg_l)


def avg_win_loss_ratio(trades: list[dict]) -> float:
    """
    Average Win / Average Loss.

    >1.0 means winners are bigger than losers on average.
    """
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [abs(t["pnl"]) for t in trades if t["pnl"] < 0]
    if not losses:
        return float("inf") if wins else 0.0
    if not wins:
        return 0.0
    return float(np.mean(wins) / np.mean(losses))


# ===========================================================================
#  RISK METRICS
# ===========================================================================

def value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Value at Risk (Historical VaR).

    Returns the worst-case daily loss at the given confidence level.
    Example: VaR=0.02 at 95% means on 95% of days you lose less than 2%.
    """
    if len(returns) == 0:
        return 0.0
    return float(-np.percentile(returns, (1 - confidence) * 100))


def conditional_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Conditional VaR / Expected Shortfall.

    Average loss on days that exceed the VaR threshold.
    Shows tail-risk severity.
    """
    if len(returns) == 0:
        return 0.0
    var = value_at_risk(returns, confidence)
    tail = returns[returns < -var]
    if len(tail) == 0:
        return var
    return float(-np.mean(tail))


def daily_volatility(returns: np.ndarray, trading_days: int = 252) -> float:
    """Annualised volatility from daily returns."""
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(trading_days))


# ===========================================================================
#  PREDICTION QUALITY METRICS (the metrics that actually matter for ML)
# ===========================================================================

def directional_accuracy(predictions: list[dict]) -> float:
    """
    Did the model correctly predict UP or DOWN?

    This is THE most important metric for a stock prediction model.
    RMSE/MSE can be gamed by predicting yesterday's price.
    Directional accuracy cannot.

    Each prediction dict must have:
      - current_price: float
      - actual_future: float
      - pred_future: float

    Returns fraction correct (0.0 to 1.0).
    A random model scores ~50%.  >55% is meaningful.  >60% is excellent.
    """
    if not predictions:
        return 0.0
    correct = 0
    total = 0
    for p in predictions:
        actual_dir = 1 if p["actual_future"] > p["current_price"] else -1
        pred_dir = 1 if p["pred_future"] > p["current_price"] else -1
        if actual_dir == pred_dir:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def signal_accuracy(predictions: list[dict]) -> dict:
    """
    How often does each signal (BUY/SELL/HOLD) match the actual outcome?

    Returns {
        "overall": float,
        "buy_accuracy": float,
        "sell_accuracy": float,
        "hold_accuracy": float,
        "confusion": { "TP": int, "FP": int, "TN": int, "FN": int },
        "precision_buy": float,
        "recall_buy": float,
        "f1_buy": float,
    }
    """
    if not predictions:
        return {"overall": 0.0}

    correct = 0
    tp = fp = tn = fn = 0
    buy_correct = buy_total = 0
    sell_correct = sell_total = 0
    hold_correct = hold_total = 0

    for p in predictions:
        pred_sig = p.get("pred_signal", "HOLD")
        actual_sig = p.get("actual_signal", "HOLD")

        if pred_sig == actual_sig:
            correct += 1

        # BUY as positive class
        if pred_sig == "BUY" and actual_sig == "BUY":
            tp += 1
        elif pred_sig == "BUY" and actual_sig != "BUY":
            fp += 1
        elif pred_sig != "BUY" and actual_sig == "BUY":
            fn += 1
        else:
            tn += 1

        if pred_sig == "BUY":
            buy_total += 1
            if actual_sig == "BUY":
                buy_correct += 1
        elif pred_sig == "SELL":
            sell_total += 1
            if actual_sig == "SELL":
                sell_correct += 1
        else:
            hold_total += 1
            if actual_sig == "HOLD":
                hold_correct += 1

    precision_buy = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_buy = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_buy = 2 * precision_buy * recall_buy / (precision_buy + recall_buy) if (precision_buy + recall_buy) > 0 else 0.0

    return {
        "overall": correct / len(predictions),
        "buy_accuracy": buy_correct / buy_total if buy_total > 0 else 0.0,
        "sell_accuracy": sell_correct / sell_total if sell_total > 0 else 0.0,
        "hold_accuracy": hold_correct / hold_total if hold_total > 0 else 0.0,
        "buy_count": buy_total,
        "sell_count": sell_total,
        "hold_count": hold_total,
        "confusion": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "precision_buy": precision_buy,
        "recall_buy": recall_buy,
        "f1_buy": f1_buy,
    }


def information_coefficient(predictions: list[dict]) -> float:
    """
    IC = Spearman rank correlation between predicted returns and actual returns.

    Used by quant funds as the core alpha metric.
    IC > 0.05 is meaningful.  IC > 0.10 is strong.
    """
    if len(predictions) < 5:
        return 0.0

    pred_returns = []
    actual_returns = []
    for p in predictions:
        current = p["current_price"]
        if current == 0:
            continue
        pred_returns.append((p["pred_future"] - current) / current)
        actual_returns.append((p["actual_future"] - current) / current)

    if len(pred_returns) < 5:
        return 0.0

    corr, _ = sp_stats.spearmanr(pred_returns, actual_returns)
    return float(corr) if not np.isnan(corr) else 0.0


def statistical_significance(trades: list[dict], null_win_rate: float = 0.5) -> dict:
    """
    Is the win rate statistically significant or just luck?

    Uses binomial test: H0 = win rate is 50% (random).
    p < 0.05 means the result is unlikely due to chance alone.

    Also computes minimum trades needed for significance at current win rate.
    """
    n = len(trades)
    if n == 0:
        return {
            "n_trades": 0,
            "p_value": 1.0,
            "significant": False,
            "min_trades_needed": 30,
            "conclusion": "No trades to evaluate.",
        }

    wins = sum(1 for t in trades if t["pnl"] > 0)
    wr = wins / n

    # Binomial test
    result = sp_stats.binomtest(wins, n, null_win_rate, alternative="greater")
    p_value = float(result.pvalue)

    # Estimate min trades needed for significance at current win rate
    # Using normal approximation: n >= (z_alpha * sqrt(p0*q0) / (p - p0))^2
    if wr > null_win_rate:
        z = 1.645  # 95% one-tailed
        p0 = null_win_rate
        q0 = 1 - p0
        delta = wr - p0
        min_n = int(np.ceil((z * np.sqrt(p0 * q0) / delta) ** 2))
    else:
        min_n = 999  # can't achieve significance with wr <= 50%

    if p_value < 0.01:
        conclusion = f"STRONG evidence (p={p_value:.4f}). Win rate {wr:.1%} is very unlikely due to chance."
    elif p_value < 0.05:
        conclusion = f"Significant (p={p_value:.4f}). Win rate {wr:.1%} is unlikely due to chance."
    elif p_value < 0.10:
        conclusion = f"Weak evidence (p={p_value:.4f}). Win rate {wr:.1%} is suggestive but not conclusive."
    else:
        conclusion = f"NOT significant (p={p_value:.4f}). {n} trades too few or win rate {wr:.1%} too close to random."

    return {
        "n_trades": n,
        "wins": wins,
        "win_rate": wr,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "min_trades_needed": min_n,
        "conclusion": conclusion,
    }


def sample_size_warning(n_trades: int) -> str | None:
    """Return a warning if sample size is too small for reliable metrics."""
    if n_trades < 10:
        return (
            f"CRITICAL: Only {n_trades} trades. All metrics are statistically "
            f"meaningless. Need minimum 30 trades for any reliability."
        )
    elif n_trades < 30:
        return (
            f"WARNING: Only {n_trades} trades. Results have high uncertainty. "
            f"Need 30+ trades for basic statistical significance."
        )
    elif n_trades < 50:
        return (
            f"NOTE: {n_trades} trades provides moderate reliability. "
            f"50+ trades recommended for robust conclusions."
        )
    return None


# ===========================================================================
#  FULL REPORT
# ===========================================================================

def full_report(
    trades: list[dict],
    equity_curve: np.ndarray,
    daily_returns: np.ndarray,
    benchmark_returns: np.ndarray | None = None,
    initial_capital: float | None = None,
    currency: str = "Rs.",
    predictions: list[dict] | None = None,
) -> dict:
    """
    Print a comprehensive performance report and return all metrics as a dict.

    If predictions list is provided, also computes:
    - Directional accuracy
    - Signal accuracy (per-signal breakdown)
    - Information coefficient
    - Statistical significance
    """
    cap = initial_capital or Config.INITIAL_CAPITAL

    # Compute all metrics
    m = {}
    m["total_trades"] = len(trades)
    m["win_rate"] = win_rate(trades)
    m["profit_factor"] = profit_factor(trades)
    m["expectancy"] = expectancy(trades)
    m["avg_win_loss"] = avg_win_loss_ratio(trades)
    m["sharpe"] = sharpe_ratio(daily_returns)
    m["sortino"] = sortino_ratio(daily_returns)
    m["max_drawdown"] = max_drawdown(equity_curve)
    m["calmar"] = calmar_ratio(equity_curve)
    m["cagr"] = cagr(equity_curve)
    m["var_95"] = value_at_risk(daily_returns, 0.95)
    m["cvar_95"] = conditional_var(daily_returns, 0.95)
    m["volatility"] = daily_volatility(daily_returns)
    m["total_return"] = (equity_curve[-1] / equity_curve[0] - 1) if len(equity_curve) > 0 else 0
    m["final_capital"] = equity_curve[-1] if len(equity_curve) > 0 else cap

    if benchmark_returns is not None and len(benchmark_returns) > 0:
        alpha, beta = compute_alpha_beta(daily_returns, benchmark_returns)
        m["alpha"] = alpha
        m["beta"] = beta
    else:
        m["alpha"] = None
        m["beta"] = None

    # Winning / losing trades
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    m["winning_trades"] = len(wins)
    m["losing_trades"] = len(losses)
    m["avg_win"] = float(np.mean([t["pnl"] for t in wins])) if wins else 0.0
    m["avg_loss"] = float(np.mean([abs(t["pnl"]) for t in losses])) if losses else 0.0
    m["best_trade"] = max((t["pnl"] for t in trades), default=0)
    m["worst_trade"] = min((t["pnl"] for t in trades), default=0)

    # ── Prediction-level metrics (NEW) ─────────────────────────────────
    if predictions:
        m["directional_accuracy"] = directional_accuracy(predictions)
        m["signal_metrics"] = signal_accuracy(predictions)
        m["information_coefficient"] = information_coefficient(predictions)
        m["stat_significance"] = statistical_significance(trades)
        m["n_predictions"] = len(predictions)
    else:
        m["directional_accuracy"] = None
        m["signal_metrics"] = None
        m["information_coefficient"] = None
        m["stat_significance"] = None
        m["n_predictions"] = 0

    # Sample size warning
    m["sample_warning"] = sample_size_warning(len(trades))

    # --- Print report ---
    print(f"\n{'='*60}")
    print(f"  PERFORMANCE REPORT")
    print(f"{'='*60}")

    # Sample size warning first (most important)
    if m["sample_warning"]:
        print(f"\n  !! {m['sample_warning']}")

    print(f"\n  --- PORTFOLIO METRICS ---")
    print(f"  Total Return      : {m['total_return']*100:+.2f}%")
    print(f"  CAGR              : {m['cagr']*100:+.2f}%")
    print(f"  Final Capital     : {currency} {m['final_capital']:,.0f}  (started {currency} {cap:,.0f})")
    print(f"  Sharpe Ratio      : {m['sharpe']:.3f}")
    print(f"  Sortino Ratio     : {m['sortino']:.3f}")
    print(f"  Max Drawdown      : {m['max_drawdown']*100:.2f}%")
    print(f"  Calmar Ratio      : {m['calmar']:.3f}")
    if m["alpha"] is not None:
        print(f"  Alpha (annual)    : {m['alpha']*100:+.2f}%")
        print(f"  Beta              : {m['beta']:.3f}")

    print(f"\n  --- TRADE METRICS ---")
    print(f"  Total Trades      : {m['total_trades']}")
    print(f"  Win Rate          : {m['win_rate']*100:.1f}%  ({m['winning_trades']}W / {m['losing_trades']}L)")
    print(f"  Profit Factor     : {m['profit_factor']:.3f}")
    print(f"  Expectancy        : {currency} {m['expectancy']:+,.0f} per trade")
    print(f"  Avg Win/Loss      : {m['avg_win_loss']:.3f}")
    print(f"  Avg Win           : {currency} {m['avg_win']:+,.0f}")
    print(f"  Avg Loss          : {currency} {m['avg_loss']:,.0f}")
    print(f"  Best Trade        : {currency} {m['best_trade']:+,.0f}")
    print(f"  Worst Trade       : {currency} {m['worst_trade']:+,.0f}")

    # ── Prediction quality metrics (NEW) ───────────────────────────────
    if predictions:
        sig = m["signal_metrics"]
        stat = m["stat_significance"]
        print(f"\n  --- PREDICTION QUALITY (ML Rigor) ---")
        print(f"  Directional Acc.  : {m['directional_accuracy']*100:.1f}%  (>55% = meaningful, >60% = excellent)")
        print(f"  Info Coefficient  : {m['information_coefficient']:.4f}  (>0.05 = alpha signal)")
        print(f"  Signal Accuracy   : {sig['overall']*100:.1f}%  (BUY: {sig.get('buy_count',0)}, SELL: {sig.get('sell_count',0)}, HOLD: {sig.get('hold_count',0)})")
        print(f"  BUY Precision     : {sig['precision_buy']*100:.1f}%  | Recall: {sig['recall_buy']*100:.1f}%  | F1: {sig['f1_buy']*100:.1f}%")
        print(f"  Statistical Test  : {stat['conclusion']}")
        print(f"  Total Predictions : {m['n_predictions']}")

    print(f"\n  --- RISK METRICS ---")
    print(f"  VaR (95%)         : {m['var_95']*100:.2f}% daily")
    print(f"  CVaR (95%)        : {m['cvar_95']*100:.2f}% daily")
    print(f"  Volatility (ann.) : {m['volatility']*100:.2f}%")

    # Interpretation
    print(f"\n  --- INTERPRETATION ---")
    if m["sharpe"] > 2:
        print(f"  Sharpe > 2.0  :  Excellent risk-adjusted returns")
    elif m["sharpe"] > 1:
        print(f"  Sharpe > 1.0  :  Acceptable risk-adjusted returns")
    else:
        print(f"  Sharpe < 1.0  :  Below market-standard risk-adjusted returns")

    if m["profit_factor"] > 1.5:
        print(f"  PF > 1.5      :  Strategy generates more profit than loss")
    elif m["profit_factor"] > 1.0:
        print(f"  PF > 1.0      :  Marginally profitable")
    else:
        print(f"  PF < 1.0      :  Strategy is losing money")

    if m["expectancy"] > 0:
        print(f"  Expectancy +  :  Strategy has a statistical edge")
    else:
        print(f"  Expectancy -  :  Strategy does NOT have an edge")

    if m.get("directional_accuracy") is not None:
        da = m["directional_accuracy"]
        if da > 0.60:
            print(f"  DirAcc > 60%  :  Excellent directional prediction")
        elif da > 0.55:
            print(f"  DirAcc > 55%  :  Meaningful directional edge")
        elif da > 0.50:
            print(f"  DirAcc > 50%  :  Marginal edge (barely better than coin flip)")
        else:
            print(f"  DirAcc < 50%  :  WORSE than random -- model may be inverted")

    print(f"{'='*60}\n")

    return m
