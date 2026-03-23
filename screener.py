"""
screener.py - AI-powered stock screener for finding multibagger candidates.

Scores every stock on three dimensions:
    1. Fundamental (40%) - P/E, ROE, Debt, Growth, Cash Flow
    2. Technical   (35%) - RSI, MACD, Price vs SMA, Bollinger position
    3. Momentum    (25%) - 52-week high proximity, volume trend, price momentum

Usage:
    python screener.py                            # scan NIFTY stocks
    python screener.py --tickers TCS.NS INFY.NS   # scan custom list
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd

from config import Config
from data_fetcher import add_technical_indicators, fetch_fundamentals

import yfinance as yf


# ===========================================================================
#  SCORING FUNCTIONS (each returns 0.0 to 1.0)
# ===========================================================================

def fundamental_score(info: dict) -> float:
    """
    Score based on company fundamentals (0-1).

    Factors (equally weighted within this category):
      - P/E ratio   : lower is better (relative to sector avg ~20)
      - ROE          : higher is better (>15% ideal)
      - Debt/Equity  : lower is better (<1.0 ideal)
      - Earnings growth : higher is better (>15% ideal)
      - Revenue growth  : higher is better
      - Free Cash Flow  : positive and large relative to market cap
    """
    scores = []

    # P/E: 5 = 1.0, 40+ = 0.0
    pe = info.get("pe_ratio", 0) or 0
    if 0 < pe <= 50:
        scores.append(max(0, 1 - (pe - 5) / 45))
    else:
        scores.append(0.3)  # unknown / negative PE gets neutral

    # ROE: 0% = 0.0, 30%+ = 1.0
    roe = info.get("roe", 0) or 0
    scores.append(min(1.0, max(0, roe / 0.30)))

    # Debt/Equity: 0 = 1.0, 2+ = 0.0
    de = info.get("debt_to_equity", 0) or 0
    de_ratio = de / 100 if de > 5 else de  # yfinance returns as %, normalise
    scores.append(max(0, 1 - de_ratio / 2))

    # Earnings growth: 0% = 0.3, 30%+ = 1.0
    eg = info.get("earnings_growth", 0) or 0
    scores.append(min(1.0, max(0, 0.3 + eg / 0.30 * 0.7)))

    # Revenue growth: 0% = 0.3, 25%+ = 1.0
    rg = info.get("revenue_growth", 0) or 0
    scores.append(min(1.0, max(0, 0.3 + rg / 0.25 * 0.7)))

    # FCF yield: positive = good, negative = bad
    fcf = info.get("free_cash_flow", 0) or 0
    mcap = info.get("market_cap", 1) or 1
    fcf_yield = fcf / mcap if mcap > 0 else 0
    scores.append(min(1.0, max(0, 0.3 + fcf_yield * 10)))

    return float(np.mean(scores))


def technical_score(df: pd.DataFrame) -> float:
    """
    Score based on current technical indicators (0-1).

    Uses the last row of the DataFrame (latest day).
    """
    if df.empty:
        return 0.5
    last = df.iloc[-1]
    scores = []

    # RSI: 30-50 = BUY zone (0.8-1.0), 50-70 = neutral, >70 = overbought
    rsi = last.get("RSI", 50)
    if rsi < 30:
        scores.append(1.0)  # oversold = buy opportunity
    elif rsi < 50:
        scores.append(0.8)
    elif rsi < 70:
        scores.append(0.5)
    else:
        scores.append(0.2)  # overbought

    # MACD above signal line = bullish
    macd = last.get("MACD", 0)
    macd_sig = last.get("MACD_Signal", 0)
    scores.append(0.8 if macd > macd_sig else 0.3)

    # Price above SMA_30 = uptrend
    close = last.get("Close", 0)
    sma30 = last.get("SMA_30", close)
    if sma30 > 0:
        ratio = close / sma30
        scores.append(min(1.0, max(0, 0.3 + (ratio - 0.95) * 5)))
    else:
        scores.append(0.5)

    # Bollinger Band position: near lower = buy opportunity
    bb_upper = last.get("BB_Upper", close)
    bb_lower = last.get("BB_Lower", close)
    bb_range = bb_upper - bb_lower
    if bb_range > 0:
        position = (close - bb_lower) / bb_range  # 0 = lower band, 1 = upper
        scores.append(max(0, 1 - position))  # lower = better buy signal
    else:
        scores.append(0.5)

    return float(np.mean(scores))


def momentum_score(info: dict, df: pd.DataFrame) -> float:
    """
    Score based on price momentum and relative strength (0-1).
    """
    scores = []

    # 52-week high proximity: closer to high = stronger momentum
    high52 = info.get("fifty_two_week_high", 0) or 0
    low52 = info.get("fifty_two_week_low", 0) or 0
    current = info.get("current_price", 0) or 0

    if high52 > low52 > 0 and current > 0:
        pos = (current - low52) / (high52 - low52)
        # Sweet spot: 0.6-0.8 (strong but not at peak)
        if 0.6 <= pos <= 0.85:
            scores.append(0.9)
        elif pos > 0.85:
            scores.append(0.6)  # near peak, less upside
        else:
            scores.append(max(0.2, pos))
    else:
        scores.append(0.5)

    # Volume trend: recent volume vs average
    if not df.empty and "Volume_Ratio" in df.columns:
        vol_ratio = df["Volume_Ratio"].iloc[-5:].mean()
        scores.append(min(1.0, max(0, vol_ratio / 1.5)))
    else:
        scores.append(0.5)

    # Recent price momentum (20-day return)
    if len(df) >= 20:
        ret_20d = (df["Close"].iloc[-1] / df["Close"].iloc[-20]) - 1
        scores.append(min(1.0, max(0, 0.5 + ret_20d * 5)))
    else:
        scores.append(0.5)

    return float(np.mean(scores))


def is_multibagger_candidate(info: dict, cfg: Config) -> bool:
    """
    Check if a stock meets multibagger criteria:
    - ROE > 15%
    - Debt/Equity < 1.0
    - Earnings growth > 15%
    - Positive free cash flow
    """
    roe = info.get("roe", 0) or 0
    de = info.get("debt_to_equity", 0) or 0
    de_ratio = de / 100 if de > 5 else de
    eg = info.get("earnings_growth", 0) or 0
    fcf = info.get("free_cash_flow", 0) or 0

    return (
        roe >= cfg.MIN_ROE
        and de_ratio <= cfg.MAX_DEBT_EQUITY
        and eg >= cfg.MIN_EARNINGS_GROWTH
        and fcf > cfg.MIN_FCF
    )


# ===========================================================================
#  MAIN SCANNER
# ===========================================================================

def scan_market(
    tickers: list[str] | None = None,
    cfg: Config | None = None,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Scan a list of tickers and rank them by composite score.

    Returns a DataFrame sorted by score (descending).
    """
    if cfg is None:
        cfg = Config()
    tickers = tickers or cfg.NIFTY_STOCKS

    print(f"\n{'='*72}")
    print(f"  AI STOCK SCREENER - Scanning {len(tickers)} stocks")
    print(f"{'='*72}\n")

    results = []
    errors = []

    for i, ticker in enumerate(tickers, 1):
        sys.stdout.write(f"\r  [{i}/{len(tickers)}] Scanning {ticker:<20s}")
        sys.stdout.flush()

        try:
            # Fetch fundamentals
            info = fetch_fundamentals(ticker)

            # Fetch price data (1 year for technical analysis)
            raw = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
            if raw.empty:
                errors.append(ticker)
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df = add_technical_indicators(df)

            # Compute scores
            f_score = fundamental_score(info)
            t_score = technical_score(df)
            m_score = momentum_score(info, df)

            composite = (
                cfg.FUND_WEIGHT * f_score
                + cfg.TECH_WEIGHT * t_score
                + cfg.MOM_WEIGHT * m_score
            )

            multi = is_multibagger_candidate(info, cfg)

            results.append({
                "Ticker": ticker,
                "Name": info.get("name", "")[:25],
                "Score": round(composite, 3),
                "Fund": round(f_score, 2),
                "Tech": round(t_score, 2),
                "Mom": round(m_score, 2),
                "P/E": round(info.get("pe_ratio", 0) or 0, 1),
                "ROE%": round((info.get("roe", 0) or 0) * 100, 1),
                "D/E": round((info.get("debt_to_equity", 0) or 0), 1),
                "EG%": round((info.get("earnings_growth", 0) or 0) * 100, 1),
                "Multibagger": "YES" if multi else "",
            })

            time.sleep(0.3)  # rate-limit Yahoo Finance

        except Exception as e:
            errors.append(f"{ticker}: {e}")

    print(f"\r  Scan complete. {len(results)} stocks analysed.              \n")

    if not results:
        print("  No results. Check your internet connection.")
        return pd.DataFrame()

    # Sort by composite score
    df_results = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
    df_results.index += 1  # rank starts at 1
    df_results.index.name = "Rank"

    # Display top N
    top = df_results.head(top_n)
    print(f"  TOP {min(top_n, len(top))} STOCKS BY COMPOSITE SCORE")
    print(f"  {'-'*70}")
    print(top.to_string())

    # Multibagger candidates
    multis = df_results[df_results["Multibagger"] == "YES"]
    if not multis.empty:
        print(f"\n  MULTIBAGGER CANDIDATES ({len(multis)} found)")
        print(f"  {'-'*70}")
        print(f"  Criteria: ROE>{cfg.MIN_ROE*100:.0f}%, D/E<{cfg.MAX_DEBT_EQUITY}, "
              f"EG>{cfg.MIN_EARNINGS_GROWTH*100:.0f}%, FCF>0")
        for _, row in multis.iterrows():
            print(f"    {row['Ticker']:<18s} | Score: {row['Score']:.3f} | "
                  f"ROE: {row['ROE%']:.1f}% | EG: {row['EG%']:.1f}%")
    else:
        print("\n  No multibagger candidates found in current scan.")

    if errors:
        print(f"\n  Skipped {len(errors)} ticker(s) due to errors.")

    print(f"\n{'='*72}\n")

    return df_results


# -- CLI -------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AI Stock Screener")
    p.add_argument("--tickers", nargs="+", default=None,
                   help="Custom ticker list (e.g. TCS.NS INFY.NS AAPL)")
    p.add_argument("--top", type=int, default=15, help="Show top N results")
    args = p.parse_args()
    cfg = Config()
    scan_market(tickers=args.tickers, cfg=cfg, top_n=args.top)
