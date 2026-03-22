"""
confidence.py - Multi-factor confidence scoring engine.

Replaces the old "attention concentration" hack with a transparent,
multi-signal confidence score that actually means something.

Confidence = weighted combination of four independent factors:

    1. Prediction Consistency (30%)
       Run N stochastic forward passes (MC Dropout).
       Low variance across passes = model is sure of its answer.

    2. Technical Alignment (25%)
       Do RSI, MACD, Bollinger, SMA all agree with the signal?
       Full agreement = high confidence. Contradiction = low.

    3. Volatility Regime (20%)
       Recent realised volatility vs historical average.
       Low-vol regimes are more predictable = higher confidence.
       High-vol = regime shift likely = lower confidence.

    4. Model Certainty (25%)
       Attention entropy (lower = model focuses on specific patterns)
       + prediction magnitude relative to recent price range.

Each factor is scaled to [0, 1] and the final score is a weighted
average expressed as a percentage.

Formula displayed in the UI so users know exactly what it means.
"""

from __future__ import annotations

import numpy as np
import torch

from config import Config


# ── Factor 1: Prediction Consistency (MC Dropout) ────────────────────────

def mc_dropout_predictions(
    model: torch.nn.Module,
    x: torch.Tensor,
    n_samples: int = 30,
    close_scaler=None,
) -> dict:
    """
    Run N stochastic forward passes with dropout enabled.

    Returns {
        "mean": array of mean predictions per day,
        "std": array of std per day,
        "lower": 10th percentile band,
        "upper": 90th percentile band,
        "all_preds": (n_samples, pred_length) raw predictions,
        "consistency_score": float 0-1 (low spread = high consistency),
    }
    """
    model.train()  # enable dropout
    device = x.device

    all_preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds_scaled, _ = model(x)
            pred_np = preds_scaled.cpu().numpy()[0]

            if close_scaler is not None:
                # inverse-transform each day individually
                pred_prices = np.array([
                    close_scaler.inverse_transform([[v]])[0, 0]
                    for v in pred_np
                ])
            else:
                pred_prices = pred_np

            all_preds.append(pred_prices)

    model.eval()  # restore eval mode

    all_preds = np.array(all_preds)  # (n_samples, pred_length)
    mean_pred = np.mean(all_preds, axis=0)
    std_pred = np.std(all_preds, axis=0)

    lower = np.percentile(all_preds, 10, axis=0)
    upper = np.percentile(all_preds, 90, axis=0)

    # Consistency = how tight the predictions are relative to the mean
    # coefficient of variation (CV): lower = more consistent
    cv = np.mean(std_pred / np.abs(mean_pred + 1e-8))
    # Map CV to 0-1 score: CV=0 -> 1.0, CV=0.05 -> 0.5, CV>=0.10 -> 0.0
    consistency_score = float(np.clip(1.0 - cv / 0.10, 0.0, 1.0))

    return {
        "mean": mean_pred,
        "std": std_pred,
        "lower": lower,
        "upper": upper,
        "all_preds": all_preds,
        "consistency_score": consistency_score,
    }


# ── Factor 2: Technical Alignment ─────────────────────────────────────────

def technical_alignment_score(signal: str, latest_row: dict) -> dict:
    """
    Check if technical indicators agree with the predicted signal.

    Returns {
        "score": float 0-1,
        "agrees": list of indicators that agree,
        "disagrees": list of indicators that disagree,
        "details": dict of individual indicator assessments,
    }
    """
    rsi = latest_row.get("RSI", 50.0)
    macd = latest_row.get("MACD", 0.0)
    macd_signal = latest_row.get("MACD_Signal", 0.0)
    close = latest_row.get("Close", 0.0)
    sma_30 = latest_row.get("SMA_30", close)
    bb_upper = latest_row.get("BB_Upper", close)
    bb_lower = latest_row.get("BB_Lower", close)

    indicators = {}
    agrees = []
    disagrees = []

    # RSI assessment
    if rsi < 30:
        rsi_bias = "BUY"
        rsi_reason = f"RSI {rsi:.1f} is oversold (<30) -- bounce likely"
    elif rsi > 70:
        rsi_bias = "SELL"
        rsi_reason = f"RSI {rsi:.1f} is overbought (>70) -- pullback likely"
    elif rsi < 45:
        rsi_bias = "BUY"
        rsi_reason = f"RSI {rsi:.1f} is in buy zone (30-45)"
    elif rsi > 55:
        rsi_bias = "SELL"
        rsi_reason = f"RSI {rsi:.1f} is in sell zone (55-70)"
    else:
        rsi_bias = "HOLD"
        rsi_reason = f"RSI {rsi:.1f} is neutral (45-55)"
    indicators["RSI"] = {"bias": rsi_bias, "reason": rsi_reason, "value": rsi}

    # MACD assessment
    macd_diff = macd - macd_signal
    if macd_diff > 0:
        macd_bias = "BUY"
        macd_reason = f"MACD ({macd:.2f}) above signal ({macd_signal:.2f}) -- bullish"
    else:
        macd_bias = "SELL"
        macd_reason = f"MACD ({macd:.2f}) below signal ({macd_signal:.2f}) -- bearish"
    indicators["MACD"] = {"bias": macd_bias, "reason": macd_reason, "value": macd}

    # SMA trend
    if sma_30 > 0:
        price_vs_sma = (close - sma_30) / sma_30
        if price_vs_sma > 0.02:
            sma_bias = "BUY"
            sma_reason = f"Price {price_vs_sma*100:+.1f}% above SMA-30 -- uptrend"
        elif price_vs_sma < -0.02:
            sma_bias = "SELL"
            sma_reason = f"Price {price_vs_sma*100:+.1f}% below SMA-30 -- downtrend"
        else:
            sma_bias = "HOLD"
            sma_reason = f"Price near SMA-30 ({price_vs_sma*100:+.1f}%) -- consolidating"
    else:
        sma_bias = "HOLD"
        sma_reason = "SMA-30 not available"
    indicators["SMA_Trend"] = {"bias": sma_bias, "reason": sma_reason, "value": close}

    # Bollinger position
    bb_range = bb_upper - bb_lower
    if bb_range > 0:
        bb_position = (close - bb_lower) / bb_range
        if bb_position < 0.2:
            bb_bias = "BUY"
            bb_reason = f"Near lower Bollinger Band ({bb_position:.0%}) -- oversold"
        elif bb_position > 0.8:
            bb_bias = "SELL"
            bb_reason = f"Near upper Bollinger Band ({bb_position:.0%}) -- overbought"
        else:
            bb_bias = "HOLD"
            bb_reason = f"Mid Bollinger Band ({bb_position:.0%}) -- neutral"
    else:
        bb_bias = "HOLD"
        bb_reason = "Bollinger Bands not available"
    indicators["Bollinger"] = {"bias": bb_bias, "reason": bb_reason}

    # Count agreements
    for name, ind in indicators.items():
        if signal == "HOLD":
            # HOLD signals get neutral treatment
            agrees.append(name) if ind["bias"] == "HOLD" else disagrees.append(name)
        elif ind["bias"] == signal:
            agrees.append(name)
        elif ind["bias"] == "HOLD":
            pass  # neutral indicators don't count against
        else:
            disagrees.append(name)

    total = len(indicators)
    agreement_count = len(agrees)
    # Score: all agree = 1.0, none agree = 0.1
    score = max(0.1, agreement_count / total)

    return {
        "score": float(score),
        "agrees": agrees,
        "disagrees": disagrees,
        "details": indicators,
        "signal_assessed": signal,
    }


# ── Factor 3: Volatility Regime ───────────────────────────────────────────

def volatility_regime_score(df_close: np.ndarray) -> dict:
    """
    Compare recent volatility (20-day) to historical average (252-day).

    Low-vol regime = more predictable = higher confidence.
    High-vol regime = regime shifts likely = lower confidence.

    Returns {
        "score": float 0-1,
        "recent_vol": float (annualised),
        "historical_vol": float (annualised),
        "regime": str ("low" | "normal" | "high"),
    }
    """
    if len(df_close) < 30:
        return {
            "score": 0.5,
            "recent_vol": 0.0,
            "historical_vol": 0.0,
            "regime": "unknown",
        }

    returns = np.diff(df_close) / df_close[:-1]

    recent = returns[-20:]
    recent_vol = float(np.std(recent, ddof=1) * np.sqrt(252))

    hist_window = min(252, len(returns))
    historical = returns[-hist_window:]
    hist_vol = float(np.std(historical, ddof=1) * np.sqrt(252))

    if hist_vol == 0:
        ratio = 1.0
    else:
        ratio = recent_vol / hist_vol

    # Score mapping: ratio=0.5 -> 1.0, ratio=1.0 -> 0.7, ratio=2.0 -> 0.1
    score = float(np.clip(1.0 - (ratio - 0.5) / 1.5, 0.1, 1.0))

    if ratio < 0.7:
        regime = "low"
    elif ratio < 1.3:
        regime = "normal"
    else:
        regime = "high"

    return {
        "score": score,
        "recent_vol": recent_vol,
        "historical_vol": hist_vol,
        "vol_ratio": float(ratio),
        "regime": regime,
    }


# ── Factor 4: Model Certainty (Attention Entropy) ─────────────────────────

def model_certainty_score(
    attention_weights: np.ndarray,
    pred_pct_change: float,
) -> dict:
    """
    Model certainty from attention pattern and prediction magnitude.

    - Low attention entropy = model focused on specific patterns = higher certainty
    - Extreme predictions (>5%) are penalised (less certain the model truly
      learned such a large move)

    Returns {
        "score": float 0-1,
        "attention_entropy": float,
        "prediction_penalty": float,
    }
    """
    # Attention entropy (normalised Shannon entropy)
    attn = attention_weights + 1e-10
    attn = attn / attn.sum()
    entropy = -np.sum(attn * np.log(attn))
    max_entropy = np.log(len(attn))
    normalised_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

    # Low entropy = focused attention = more certain
    # Map: entropy=0 -> 1.0, entropy=1.0 -> 0.0
    entropy_score = float(1.0 - normalised_entropy)

    # Prediction magnitude penalty
    # Small predictions (<2%) = full credit
    # Large predictions (>5%) = penalised (LSTM rarely predicts large moves correctly)
    abs_pct = abs(pred_pct_change)
    if abs_pct < 2.0:
        magnitude_penalty = 0.0
    elif abs_pct < 5.0:
        magnitude_penalty = (abs_pct - 2.0) / 3.0 * 0.3
    else:
        magnitude_penalty = 0.3 + min(0.4, (abs_pct - 5.0) / 10.0 * 0.4)

    score = float(np.clip(entropy_score - magnitude_penalty, 0.05, 1.0))

    return {
        "score": score,
        "attention_entropy": float(normalised_entropy),
        "prediction_penalty": float(magnitude_penalty),
    }


# ── Composite Confidence Score ─────────────────────────────────────────────

def compute_confidence(
    model: torch.nn.Module,
    x: torch.Tensor,
    close_scaler,
    signal: str,
    latest_row: dict,
    df_close: np.ndarray,
    attention_weights: np.ndarray,
    pred_pct_change: float,
    cfg: Config | None = None,
) -> dict:
    """
    Compute the full multi-factor confidence score.

    Returns {
        "confidence_pct": float (0-100, the headline number),
        "factors": {
            "prediction_consistency": { "score": ..., "weight": 0.30, ... },
            "technical_alignment": { "score": ..., "weight": 0.25, ... },
            "volatility_regime": { "score": ..., "weight": 0.20, ... },
            "model_certainty": { "score": ..., "weight": 0.25, ... },
        },
        "formula": str (human-readable formula),
        "uncertainty_bands": { "lower": [...], "upper": [...], "mean": [...] },
    }
    """
    if cfg is None:
        cfg = Config()

    weights = getattr(cfg, "CONFIDENCE_WEIGHTS", {
        "prediction_consistency": 0.30,
        "technical_alignment": 0.25,
        "volatility_regime": 0.20,
        "model_certainty": 0.25,
    })

    n_samples = getattr(cfg, "MC_DROPOUT_SAMPLES", 30)

    # Factor 1: MC Dropout
    mc = mc_dropout_predictions(model, x, n_samples=n_samples, close_scaler=close_scaler)
    f1 = mc["consistency_score"]

    # Factor 2: Technical alignment
    tech = technical_alignment_score(signal, latest_row)
    f2 = tech["score"]

    # Factor 3: Volatility regime
    vol = volatility_regime_score(df_close)
    f3 = vol["score"]

    # Factor 4: Model certainty
    cert = model_certainty_score(attention_weights, pred_pct_change)
    f4 = cert["score"]

    # Weighted composite
    composite = (
        weights["prediction_consistency"] * f1
        + weights["technical_alignment"] * f2
        + weights["volatility_regime"] * f3
        + weights["model_certainty"] * f4
    )
    confidence_pct = float(np.clip(composite * 100, 1.0, 95.0))

    formula = (
        f"Confidence = "
        f"{weights['prediction_consistency']:.0%} x Consistency({f1:.2f}) + "
        f"{weights['technical_alignment']:.0%} x TechAlign({f2:.2f}) + "
        f"{weights['volatility_regime']:.0%} x VolRegime({f3:.2f}) + "
        f"{weights['model_certainty']:.0%} x ModelCert({f4:.2f}) "
        f"= {confidence_pct:.1f}%"
    )

    return {
        "confidence_pct": confidence_pct,
        "factors": {
            "prediction_consistency": {
                "score": f1,
                "weight": weights["prediction_consistency"],
                "weighted": weights["prediction_consistency"] * f1,
                "detail": f"CV={np.mean(mc['std'] / (np.abs(mc['mean']) + 1e-8)):.4f}",
            },
            "technical_alignment": {
                "score": f2,
                "weight": weights["technical_alignment"],
                "weighted": weights["technical_alignment"] * f2,
                "agrees": tech["agrees"],
                "disagrees": tech["disagrees"],
                "details": tech["details"],
            },
            "volatility_regime": {
                "score": f3,
                "weight": weights["volatility_regime"],
                "weighted": weights["volatility_regime"] * f3,
                "regime": vol["regime"],
                "recent_vol_pct": vol["recent_vol"] * 100,
                "historical_vol_pct": vol["historical_vol"] * 100,
            },
            "model_certainty": {
                "score": f4,
                "weight": weights["model_certainty"],
                "weighted": weights["model_certainty"] * f4,
                "attention_entropy": cert["attention_entropy"],
                "prediction_penalty": cert["prediction_penalty"],
            },
        },
        "formula": formula,
        "uncertainty_bands": {
            "lower": mc["lower"].tolist(),
            "upper": mc["upper"].tolist(),
            "mean": mc["mean"].tolist(),
            "std": mc["std"].tolist(),
        },
    }
