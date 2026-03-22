"""
metrics.py - Real-world quantitative trading metrics.

All metrics used by professional algo traders and hedge funds.
Zero external dependencies beyond numpy/pandas.

Portfolio Metrics : Sharpe, Sortino, Calmar, Max Drawdown, CAGR, Alpha, Beta
Trade Metrics     : Win Rate, Profit Factor, Expectancy, Avg Win/Loss Ratio
Risk Metrics      : Value at Risk (VaR), Conditional VaR, Daily Volatility
"""

import numpy as np
import pandas as pd

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
#  FULL REPORT
# ===========================================================================

def full_report(
    trades: list[dict],
    equity_curve: np.ndarray,
    daily_returns: np.ndarray,
    benchmark_returns: np.ndarray | None = None,
    initial_capital: float | None = None,
    currency: str = "Rs.",
) -> dict:
    """
    Print a comprehensive performance report and return all metrics as a dict.
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

    # --- Print report ---
    print(f"\n{'='*60}")
    print(f"  PERFORMANCE REPORT")
    print(f"{'='*60}")

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

    print(f"{'='*60}\n")

    return m
