"""
explainability.py - Signal explanation engine.

Instead of just saying "BUY" or "SELL", explain WHY the model
thinks this, what the technicals show, where they agree or
disagree, and what the user should watch for.

Addresses: "RSI at 25.23 vs BUY Signal ... creates confusion.
The signal logic needs better explainability."
"""

from __future__ import annotations


def explain_signal(
    signal: str,
    current_price: float,
    pred_price: float,
    pred_pct: float,
    latest_row: dict,
    confidence_factors: dict | None = None,
    currency: str = "Rs.",
) -> dict:
    """
    Generate a structured explanation for the model's signal.

    Returns {
        "signal": str,
        "headline": str (one-line summary),
        "reasoning": list[str] (bullet points explaining the signal),
        "technicals_summary": str,
        "confluence": str ("strong" | "moderate" | "weak" | "conflicting"),
        "watch_items": list[str] (what to monitor),
        "risk_warning": str,
    }
    """
    rsi = latest_row.get("RSI", 50.0)
    macd = latest_row.get("MACD", 0.0)
    macd_signal = latest_row.get("MACD_Signal", 0.0)
    close = latest_row.get("Close", current_price)
    sma_30 = latest_row.get("SMA_30", close)
    bb_upper = latest_row.get("BB_Upper", close)
    bb_lower = latest_row.get("BB_Lower", close)
    volume_ratio = latest_row.get("Volume_Ratio", 1.0)

    reasoning = []
    technicals = []
    watch = []
    conflicts = 0
    supports = 0

    # ── LSTM Prediction reasoning ──────────────────────────────────────
    reasoning.append(
        f"LSTM model predicts {currency}{pred_price:,.2f} in 5 days "
        f"({pred_pct:+.2f}% from {currency}{current_price:,.2f})."
    )

    # ── RSI Analysis ───────────────────────────────────────────────────
    if rsi < 30:
        rsi_state = "oversold"
        rsi_outlook = "bullish"
        technicals.append(f"RSI={rsi:.1f}: Deep oversold -- historically signals a bounce.")
        if signal == "BUY":
            reasoning.append(
                f"RSI at {rsi:.1f} (oversold <30) supports this BUY -- "
                f"the model sees a mean-reversion bounce opportunity."
            )
            supports += 1
        elif signal == "SELL":
            reasoning.append(
                f"CAUTION: RSI at {rsi:.1f} is already oversold. "
                f"SELL signal may catch a falling knife, but bounce risk exists."
            )
            conflicts += 1
    elif rsi > 70:
        rsi_state = "overbought"
        rsi_outlook = "bearish"
        technicals.append(f"RSI={rsi:.1f}: Overbought -- pullback risk elevated.")
        if signal == "SELL":
            reasoning.append(
                f"RSI at {rsi:.1f} (overbought >70) supports this SELL -- "
                f"momentum exhaustion likely."
            )
            supports += 1
        elif signal == "BUY":
            reasoning.append(
                f"WARNING: RSI at {rsi:.1f} is overbought. "
                f"BUY signal at this level carries elevated risk of pullback."
            )
            conflicts += 1
    else:
        rsi_state = "neutral"
        rsi_outlook = "neutral"
        technicals.append(f"RSI={rsi:.1f}: Neutral zone -- no extreme signal.")

    # ── MACD Analysis ──────────────────────────────────────────────────
    macd_diff = macd - macd_signal
    if macd_diff > 0:
        macd_outlook = "bullish"
        technicals.append(f"MACD={macd:.2f} above signal={macd_signal:.2f}: Bullish momentum.")
        if signal == "BUY":
            reasoning.append("MACD is above signal line, confirming upward momentum.")
            supports += 1
        elif signal == "SELL":
            reasoning.append(
                f"Conflict: MACD is bullish ({macd:.2f} > {macd_signal:.2f}) "
                f"but model predicts downside."
            )
            conflicts += 1
    else:
        macd_outlook = "bearish"
        technicals.append(f"MACD={macd:.2f} below signal={macd_signal:.2f}: Bearish momentum.")
        if signal == "SELL":
            reasoning.append("MACD below signal line confirms bearish momentum.")
            supports += 1
        elif signal == "BUY":
            reasoning.append(
                f"Conflict: MACD is bearish ({macd:.2f} < {macd_signal:.2f}) "
                f"but model predicts upside. May be an early reversal signal."
            )
            conflicts += 1

    # ── SMA Trend ──────────────────────────────────────────────────────
    if sma_30 > 0:
        sma_pct = ((close - sma_30) / sma_30) * 100
        if sma_pct > 2:
            technicals.append(f"Price {sma_pct:+.1f}% above SMA-30: Uptrend intact.")
            if signal == "BUY":
                supports += 1
            elif signal == "SELL":
                conflicts += 1
        elif sma_pct < -2:
            technicals.append(f"Price {sma_pct:+.1f}% below SMA-30: Downtrend pressure.")
            if signal == "SELL":
                supports += 1
            elif signal == "BUY":
                reasoning.append(
                    f"Price is {abs(sma_pct):.1f}% below SMA-30 -- "
                    f"model may be predicting a reversal back to mean."
                )
                conflicts += 1
        else:
            technicals.append(f"Price near SMA-30 ({sma_pct:+.1f}%): Range-bound.")

    # ── Bollinger Band ─────────────────────────────────────────────────
    bb_range = bb_upper - bb_lower
    if bb_range > 0:
        bb_pos = (close - bb_lower) / bb_range
        if bb_pos < 0.2:
            technicals.append(f"Near lower Bollinger Band ({bb_pos:.0%}): Oversold territory.")
            if signal == "BUY":
                supports += 1
        elif bb_pos > 0.8:
            technicals.append(f"Near upper Bollinger Band ({bb_pos:.0%}): Overbought territory.")
            if signal == "SELL":
                supports += 1

    # ── Volume ─────────────────────────────────────────────────────────
    if volume_ratio > 1.5:
        technicals.append(f"Volume {volume_ratio:.1f}x average: High activity -- conviction in move.")
        watch.append("High volume confirms the current trend direction.")
    elif volume_ratio < 0.5:
        technicals.append(f"Volume {volume_ratio:.1f}x average: Low activity -- weak conviction.")
        watch.append("Low volume weakens signal reliability.")

    # ── Confluence Assessment ──────────────────────────────────────────
    total_indicators = supports + conflicts
    if total_indicators == 0:
        confluence = "neutral"
    elif conflicts == 0 and supports >= 3:
        confluence = "strong"
    elif supports > conflicts:
        confluence = "moderate"
    elif supports == conflicts:
        confluence = "conflicting"
    else:
        confluence = "weak"

    # ── Headline ──────────────────────────────────────────────────────
    if signal == "BUY":
        if confluence in ("strong", "moderate"):
            headline = (
                f"{signal} signal with {confluence} technical support -- "
                f"model predicts {pred_pct:+.2f}% in 5 days."
            )
        else:
            headline = (
                f"{signal} signal but technicals show {confluence} support -- "
                f"model predicts {pred_pct:+.2f}% in 5 days."
            )
    elif signal == "SELL":
        if confluence in ("strong", "moderate"):
            headline = (
                f"{signal} signal with {confluence} technical confirmation -- "
                f"model predicts {pred_pct:+.2f}% in 5 days."
            )
        else:
            headline = (
                f"{signal} signal but technicals show {confluence} confirmation -- "
                f"model predicts {pred_pct:+.2f}% in 5 days."
            )
    else:
        headline = (
            f"HOLD -- model sees no strong directional move "
            f"({pred_pct:+.2f}% predicted, within noise threshold)."
        )

    # ── Watch items ───────────────────────────────────────────────────
    if rsi < 35:
        watch.append(f"Watch RSI for bounce above 30 (currently {rsi:.1f}).")
    if rsi > 65:
        watch.append(f"Watch RSI for drop below 70 (currently {rsi:.1f}).")
    if abs(macd_diff) < 1.0:
        watch.append("MACD near signal line -- crossover could flip the signal.")
    if sma_30 > 0 and abs(close - sma_30) / sma_30 < 0.01:
        watch.append("Price at SMA-30 support/resistance -- breakout direction matters.")

    if not watch:
        watch.append("No immediate inflection points detected. Monitor regularly.")

    # ── Risk warning ──────────────────────────────────────────────────
    risk_warning = (
        "This is a model-generated signal, not financial advice. "
        "LSTM models have known limitations with financial data. "
        "Always do your own research and consider your risk tolerance."
    )

    return {
        "signal": signal,
        "headline": headline,
        "reasoning": reasoning,
        "technicals_summary": " | ".join(technicals),
        "technicals": technicals,
        "confluence": confluence,
        "supports": supports,
        "conflicts": conflicts,
        "watch_items": watch,
        "risk_warning": risk_warning,
    }
