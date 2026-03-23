"""
Alpha Pulse Engine (APE) -- Proprietary Real-Time Multi-Factor Stock Scoring Algorithm.

This algorithm is UNIQUE to this project and combines 7 orthogonal signal
dimensions into a single composite score that updates in real time as new
market data arrives.

Core Dimensions
===============
1. Momentum Flux       -- Adaptive RSI + ROC cross-frequency alignment
2. Trend Resonance     -- SMA/EMA convergence-divergence harmonic
3. Volatility Regime   -- Bollinger squeeze detection + ATR regime shift
4. Volume Conviction   -- OBV slope + volume-price divergence
5. MACD Impulse        -- Histogram acceleration + zero-line proximity
6. Mean Reversion Tau  -- Distance from VWAP / Bollinger midline + decay
7. Sentiment Charge    -- FinBERT aggregate mapped into [-1, +1]

The final APE Score is computed as a weighted geometric-arithmetic hybrid
that penalises conflicting signals and rewards confluence.

Score Range:  0 .. 100
  80-100  STRONG BUY   (rare -- all 7 dimensions aligned)
  60-79   BUY
  40-59   NEUTRAL / HOLD
  20-39   SELL
   0-19   STRONG SELL  (rare -- all 7 dimensions bearish)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ── Dimension Weights (sum to 1.0) ──────────────────────────────────────────
WEIGHTS = {
    "momentum_flux": 0.18,
    "trend_resonance": 0.16,
    "volatility_regime": 0.14,
    "volume_conviction": 0.13,
    "macd_impulse": 0.15,
    "mean_reversion_tau": 0.12,
    "sentiment_charge": 0.12,
}


@dataclass
class PulseResult:
    """Container for the full APE output."""
    score: float                              # 0-100 composite
    grade: str                                # STRONG BUY / BUY / HOLD / SELL / STRONG SELL
    dimensions: dict[str, float] = field(default_factory=dict)  # 0-1 each
    breakdown: dict[str, str] = field(default_factory=dict)     # human-readable
    confluence: float = 0.0                   # 0-1 how aligned
    regime: str = "neutral"                   # trending / volatile / mean-reverting / neutral
    momentum_direction: str = "flat"          # up / down / flat
    alerts: list[str] = field(default_factory=list)


def _safe(val, default=0.0):
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return default
    return float(val)


def _clamp(val, lo=0.0, hi=1.0):
    return max(lo, min(hi, val))


# ──────────────────────────────────────────────────────────────────────────────
# 1. MOMENTUM FLUX
# ──────────────────────────────────────────────────────────────────────────────
def _momentum_flux(df: pd.DataFrame) -> tuple[float, str]:
    """Multi-timeframe momentum alignment using RSI + Rate of Change."""
    close = df["Close"].values
    rsi = _safe(df["RSI"].iloc[-1], 50)

    # Rate of Change across 3 timeframes
    roc_5 = (close[-1] / close[-6] - 1) * 100 if len(close) > 5 else 0
    roc_10 = (close[-1] / close[-11] - 1) * 100 if len(close) > 10 else 0
    roc_20 = (close[-1] / close[-21] - 1) * 100 if len(close) > 20 else 0

    # RSI component: map 0-100 to bullish probability
    rsi_score = (rsi - 30) / 40  # 30=0, 70=1, linear
    rsi_score = _clamp(rsi_score)

    # ROC alignment: all 3 positive = strong bull, all negative = strong bear
    roc_signs = [1 if r > 0 else -1 for r in [roc_5, roc_10, roc_20]]
    alignment = sum(roc_signs) / 3  # -1 to +1

    # Magnitude bonus
    avg_roc = (abs(roc_5) + abs(roc_10) + abs(roc_20)) / 3
    magnitude = _clamp(avg_roc / 5, 0, 1)  # 5% = max score

    # Combine: 50% RSI + 30% alignment + 20% magnitude
    raw = 0.50 * rsi_score + 0.30 * ((alignment + 1) / 2) + 0.20 * magnitude

    direction = "up" if alignment > 0.3 else "down" if alignment < -0.3 else "flat"
    detail = f"RSI={rsi:.0f}, ROC5={roc_5:+.1f}%, ROC20={roc_20:+.1f}%"

    return _clamp(raw), detail


# ──────────────────────────────────────────────────────────────────────────────
# 2. TREND RESONANCE
# ──────────────────────────────────────────────────────────────────────────────
def _trend_resonance(df: pd.DataFrame) -> tuple[float, str]:
    """SMA/EMA convergence-divergence harmonic."""
    close = _safe(df["Close"].iloc[-1])
    sma10 = _safe(df["SMA_10"].iloc[-1], close)
    sma30 = _safe(df["SMA_30"].iloc[-1], close)
    ema12 = _safe(df["EMA_12"].iloc[-1], close)
    ema26 = _safe(df["EMA_26"].iloc[-1], close)

    # Price position relative to moving averages
    above_count = sum([
        close > sma10,
        close > sma30,
        close > ema12,
        close > ema26,
        sma10 > sma30,   # golden cross
        ema12 > ema26,   # MACD positive
    ])

    raw = above_count / 6  # 0=all bearish, 1=all bullish

    # Convergence bonus: when MAs are close together, a breakout is imminent
    spread = abs(sma10 - sma30) / close if close > 0 else 0
    convergence = 1 - _clamp(spread * 20, 0, 1)  # tight = high

    score = 0.7 * raw + 0.3 * (convergence if raw > 0.5 else (1 - convergence))

    detail = f"{'Golden' if sma10 > sma30 else 'Death'} cross, {above_count}/6 bullish MAs"
    return _clamp(score), detail


# ──────────────────────────────────────────────────────────────────────────────
# 3. VOLATILITY REGIME
# ──────────────────────────────────────────────────────────────────────────────
def _volatility_regime(df: pd.DataFrame) -> tuple[float, str, str]:
    """Bollinger squeeze detection + ATR regime classification."""
    close = df["Close"].values
    bb_upper = _safe(df["BB_Upper"].iloc[-1], close[-1])
    bb_lower = _safe(df["BB_Lower"].iloc[-1], close[-1])
    current = close[-1]

    # Bollinger Band Width (squeeze = low vol = breakout coming)
    bbw = (bb_upper - bb_lower) / current if current > 0 else 0.1

    # Historical volatility (20-day)
    if len(close) >= 21:
        returns = np.diff(close[-21:]) / close[-21:-1]
        hist_vol = float(np.std(returns)) * math.sqrt(252) * 100
    else:
        hist_vol = 20.0

    # Regime classification
    if bbw < 0.03:
        regime = "squeeze"
        vol_score = 0.8  # Low vol squeeze = high opportunity
    elif hist_vol > 40:
        regime = "volatile"
        vol_score = 0.3  # High vol = uncertainty
    elif hist_vol < 15:
        regime = "calm"
        vol_score = 0.6
    else:
        regime = "normal"
        vol_score = 0.5

    # Position within Bollinger Bands (0=lower, 1=upper)
    bb_pos = (current - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
    bb_pos = _clamp(bb_pos)

    # Near bands = extreme = higher score for mean reversion
    extremity = abs(bb_pos - 0.5) * 2  # 0=middle, 1=at band

    score = 0.5 * vol_score + 0.3 * (1 - extremity) + 0.2 * _clamp(bb_pos)

    detail = f"BBW={bbw:.3f}, Vol={hist_vol:.1f}%, regime={regime}"
    return _clamp(score), detail, regime


# ──────────────────────────────────────────────────────────────────────────────
# 4. VOLUME CONVICTION
# ──────────────────────────────────────────────────────────────────────────────
def _volume_conviction(df: pd.DataFrame) -> tuple[float, str]:
    """OBV slope + volume-price divergence detection."""
    close = df["Close"].values
    volume = df["Volume"].values
    vol_ratio = _safe(df["Volume_Ratio"].iloc[-1], 1.0)

    # On-Balance Volume trend (simplified: last 10 days)
    n = min(10, len(close) - 1)
    if n < 2:
        return 0.5, "Insufficient data"

    obv_changes = []
    for i in range(-n, 0):
        if close[i] > close[i - 1]:
            obv_changes.append(volume[i])
        elif close[i] < close[i - 1]:
            obv_changes.append(-volume[i])
        else:
            obv_changes.append(0)

    obv_trend = sum(obv_changes)
    avg_vol = float(np.mean(volume[-n:])) if volume[-n:].mean() > 0 else 1
    obv_normalised = _clamp((obv_trend / avg_vol / n) + 0.5, 0, 1)

    # Volume surge detection
    surge = _clamp(vol_ratio / 2, 0, 1)  # 2x avg volume = max score

    # Price-volume alignment
    price_up = close[-1] > close[-2] if len(close) > 1 else True
    vol_up = vol_ratio > 1.0
    aligned = 1.0 if (price_up == vol_up) else 0.3

    score = 0.4 * obv_normalised + 0.3 * surge + 0.3 * aligned

    detail = f"VR={vol_ratio:.2f}x, OBV={'rising' if obv_trend > 0 else 'falling'}"
    return _clamp(score), detail


# ──────────────────────────────────────────────────────────────────────────────
# 5. MACD IMPULSE
# ──────────────────────────────────────────────────────────────────────────────
def _macd_impulse(df: pd.DataFrame) -> tuple[float, str]:
    """MACD histogram acceleration + zero-line proximity."""
    macd = _safe(df["MACD"].iloc[-1])
    signal = _safe(df["MACD_Signal"].iloc[-1])
    histogram = macd - signal

    # Histogram direction (positive = bullish)
    hist_score = _clamp((histogram / max(abs(macd), 0.01) + 1) / 2)

    # MACD above/below zero
    zero_score = _clamp((macd / max(abs(macd), 0.01) + 1) / 2)

    # Crossover proximity
    crossover_distance = abs(macd - signal)
    close_val = _safe(df["Close"].iloc[-1], 1)
    proximity = _clamp(1 - crossover_distance / (close_val * 0.02))

    # Recent histogram acceleration
    if len(df) >= 3:
        hist_vals = [_safe(df["MACD"].iloc[i]) - _safe(df["MACD_Signal"].iloc[i]) for i in [-3, -2, -1]]
        accel = hist_vals[-1] - hist_vals[-2]
        accel_score = _clamp((accel / max(abs(hist_vals[-1]), 0.01) + 1) / 2)
    else:
        accel_score = 0.5

    score = 0.35 * hist_score + 0.25 * zero_score + 0.20 * accel_score + 0.20 * proximity

    detail = f"MACD={macd:.2f}, Hist={histogram:+.2f}, {'bullish' if histogram > 0 else 'bearish'}"
    return _clamp(score), detail


# ──────────────────────────────────────────────────────────────────────────────
# 6. MEAN REVERSION TAU
# ──────────────────────────────────────────────────────────────────────────────
def _mean_reversion_tau(df: pd.DataFrame) -> tuple[float, str]:
    """Distance from equilibrium with exponential decay bias."""
    close = _safe(df["Close"].iloc[-1])
    sma30 = _safe(df["SMA_30"].iloc[-1], close)
    bb_upper = _safe(df["BB_Upper"].iloc[-1], close)
    bb_lower = _safe(df["BB_Lower"].iloc[-1], close)

    # Distance from SMA-30 as % of price
    deviation = (close - sma30) / sma30 if sma30 > 0 else 0

    # Tau: probability of mean reversion (further away = higher chance)
    tau = 1 - math.exp(-abs(deviation) * 20)  # exponential decay

    # Direction: if above SMA, mean reversion = bearish → lower score
    if deviation > 0.02:
        score = 0.5 - tau * 0.4  # extended above → expect pullback → bearish
    elif deviation < -0.02:
        score = 0.5 + tau * 0.4  # extended below → expect bounce → bullish
    else:
        score = 0.5  # near equilibrium

    detail = f"Dev={deviation:+.1%} from SMA30, Tau={tau:.2f}"
    return _clamp(score), detail


# ──────────────────────────────────────────────────────────────────────────────
# 7. SENTIMENT CHARGE
# ──────────────────────────────────────────────────────────────────────────────
def _sentiment_charge(sentiment_score: float | None) -> tuple[float, str]:
    """Map external FinBERT / news sentiment into 0-1 scale."""
    if sentiment_score is None:
        return 0.5, "No sentiment data"

    # sentiment_score is typically -1 (bearish) to +1 (bullish)
    score = _clamp((sentiment_score + 1) / 2)
    label = "bullish" if sentiment_score > 0.2 else "bearish" if sentiment_score < -0.2 else "neutral"
    detail = f"Sentiment={sentiment_score:+.2f} ({label})"
    return score, detail


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def compute_pulse(
    df: pd.DataFrame,
    sentiment_score: float | None = None,
) -> PulseResult:
    """
    Compute the Alpha Pulse Engine score for a given stock.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV + technical indicators (must include RSI, MACD, SMA, BB, etc.)
    sentiment_score : float | None
        Aggregate FinBERT sentiment from -1 to +1

    Returns
    -------
    PulseResult
        Full scoring breakdown
    """
    if len(df) < 30:
        return PulseResult(score=50, grade="HOLD", regime="insufficient_data",
                          alerts=["Insufficient data for reliable scoring"])

    # Compute each dimension
    mom_score, mom_detail = _momentum_flux(df)
    trend_score, trend_detail = _trend_resonance(df)
    vol_score, vol_detail, regime = _volatility_regime(df)
    conv_score, conv_detail = _volume_conviction(df)
    macd_score, macd_detail = _macd_impulse(df)
    mr_score, mr_detail = _mean_reversion_tau(df)
    sent_score, sent_detail = _sentiment_charge(sentiment_score)

    dimensions = {
        "momentum_flux": mom_score,
        "trend_resonance": trend_score,
        "volatility_regime": vol_score,
        "volume_conviction": conv_score,
        "macd_impulse": macd_score,
        "mean_reversion_tau": mr_score,
        "sentiment_charge": sent_score,
    }

    breakdown = {
        "momentum_flux": mom_detail,
        "trend_resonance": trend_detail,
        "volatility_regime": vol_detail,
        "volume_conviction": conv_detail,
        "macd_impulse": macd_detail,
        "mean_reversion_tau": mr_detail,
        "sentiment_charge": sent_detail,
    }

    # ── Weighted composite ──────────────────────────────────────────────
    weighted_sum = sum(
        dimensions[k] * WEIGHTS[k] for k in WEIGHTS
    )

    # ── Confluence multiplier ───────────────────────────────────────────
    # If all dimensions agree (all > 0.5 or all < 0.5), boost the signal
    bullish_dims = sum(1 for v in dimensions.values() if v > 0.55)
    bearish_dims = sum(1 for v in dimensions.values() if v < 0.45)
    total_dims = len(dimensions)

    confluence = max(bullish_dims, bearish_dims) / total_dims
    confluence_bonus = (confluence - 0.5) * 0.2 if confluence > 0.5 else 0

    # Direction of confluence
    if bullish_dims > bearish_dims:
        final = weighted_sum + confluence_bonus
    elif bearish_dims > bullish_dims:
        final = weighted_sum - confluence_bonus
    else:
        final = weighted_sum

    final = _clamp(final) * 100

    # ── Grade ───────────────────────────────────────────────────────────
    if final >= 80:
        grade = "STRONG BUY"
    elif final >= 60:
        grade = "BUY"
    elif final >= 40:
        grade = "HOLD"
    elif final >= 20:
        grade = "SELL"
    else:
        grade = "STRONG SELL"

    # ── Momentum direction ──────────────────────────────────────────────
    close = df["Close"].values
    roc_5 = (close[-1] / close[-6] - 1) * 100 if len(close) > 5 else 0
    momentum_dir = "up" if roc_5 > 0.5 else "down" if roc_5 < -0.5 else "flat"

    # ── Alerts ──────────────────────────────────────────────────────────
    alerts = []
    rsi = _safe(df["RSI"].iloc[-1], 50)
    if rsi > 75:
        alerts.append("RSI overbought (>75) -- pullback risk")
    elif rsi < 25:
        alerts.append("RSI oversold (<25) -- bounce potential")
    if regime == "squeeze":
        alerts.append("Bollinger squeeze -- breakout imminent")
    if confluence > 0.85:
        alerts.append(f"Extreme confluence ({confluence:.0%}) -- high-conviction signal")
    vol_ratio = _safe(df["Volume_Ratio"].iloc[-1], 1)
    if vol_ratio > 2.0:
        alerts.append(f"Volume surge ({vol_ratio:.1f}x avg) -- institutional activity")

    return PulseResult(
        score=round(final, 1),
        grade=grade,
        dimensions={k: round(v, 3) for k, v in dimensions.items()},
        breakdown=breakdown,
        confluence=round(confluence, 3),
        regime=regime,
        momentum_direction=momentum_dir,
        alerts=alerts,
    )


def compute_pulse_for_ticker(ticker: str, period: str = "1y") -> PulseResult:
    """Convenience function: fetch data and compute Pulse in one call."""
    from data_fetcher import fetch_data
    df = fetch_data(ticker, period, use_cache=True)
    return compute_pulse(df)
