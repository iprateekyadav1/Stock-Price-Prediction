"""
sentiment.py - Financial sentiment analysis using FinBERT.

Replaces the broken rule-based sentiment that marks everything as "neutral"
even when headlines like "Indian shares tumble" are clearly bearish.

Uses ProsusAI/finbert -- a BERT model fine-tuned on financial text
that outputs (positive, negative, neutral) probabilities.

Falls back to a keyword-based classifier if FinBERT can't be loaded
(e.g., no internet, low memory).
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

_finbert_pipeline = None
_finbert_load_attempted = False


def _load_finbert():
    """Lazy-load FinBERT pipeline. Only tries once."""
    global _finbert_pipeline, _finbert_load_attempted
    if _finbert_load_attempted:
        return _finbert_pipeline
    _finbert_load_attempted = True
    try:
        from transformers import pipeline
        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            top_k=None,
            device=-1,  # CPU (use 0 for GPU)
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT loaded successfully.")
    except Exception as exc:
        logger.warning(f"FinBERT not available: {exc}. Falling back to keyword-based sentiment.")
        _finbert_pipeline = None
    return _finbert_pipeline


# ── Keyword-based fallback ─────────────────────────────────────────────────

_BULLISH_KEYWORDS = {
    "surge", "surges", "surging", "rally", "rallies", "rallying",
    "gain", "gains", "gaining", "jump", "jumps", "jumping",
    "rise", "rises", "rising", "soar", "soars", "soaring",
    "climb", "climbs", "climbing", "up", "higher", "high",
    "record", "bull", "bullish", "outperform", "beat", "beats",
    "profit", "profits", "upgrade", "upgrades", "positive",
    "growth", "grows", "strong", "recover", "recovery", "rebound",
    "boost", "boosted", "optimism", "optimistic", "breakout",
}

_BEARISH_KEYWORDS = {
    "fall", "falls", "falling", "drop", "drops", "dropping",
    "decline", "declines", "declining", "tumble", "tumbles", "tumbling",
    "plunge", "plunges", "plunging", "crash", "crashes", "crashing",
    "sink", "sinks", "sinking", "down", "lower", "low",
    "sell", "selloff", "sell-off", "bear", "bearish", "underperform",
    "loss", "losses", "downgrade", "downgrades", "negative",
    "weak", "weakens", "fear", "recession", "slump", "slumps",
    "concern", "concerns", "warning", "warns", "risk", "volatile",
    "cut", "cuts", "slash", "miss", "misses", "deficit",
}


def _keyword_sentiment(text: str) -> dict:
    """Simple keyword-based sentiment as fallback."""
    words = set(text.lower().split())
    bull_count = len(words & _BULLISH_KEYWORDS)
    bear_count = len(words & _BEARISH_KEYWORDS)

    if bull_count > bear_count:
        score = min(0.5 + (bull_count - bear_count) * 0.1, 0.9)
        return {
            "label": "bullish",
            "score": score,
            "positive": score,
            "negative": 1.0 - score - 0.1,
            "neutral": 0.1,
            "method": "keyword",
        }
    elif bear_count > bull_count:
        score = min(0.5 + (bear_count - bull_count) * 0.1, 0.9)
        return {
            "label": "bearish",
            "score": -score,
            "positive": 1.0 - score - 0.1,
            "negative": score,
            "neutral": 0.1,
            "method": "keyword",
        }
    else:
        return {
            "label": "neutral",
            "score": 0.0,
            "positive": 0.2,
            "negative": 0.2,
            "neutral": 0.6,
            "method": "keyword",
        }


# ── FinBERT classification ────────────────────────────────────────────────

def classify_headline(text: str) -> dict:
    """
    Classify a single headline/text.

    Returns {
        "label": "bullish" | "bearish" | "neutral",
        "score": float (-1 to +1, negative=bearish, positive=bullish),
        "positive": float (probability),
        "negative": float (probability),
        "neutral": float (probability),
        "method": "finbert" | "keyword",
    }
    """
    if not text or not text.strip():
        return {
            "label": "neutral",
            "score": 0.0,
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 1.0,
            "method": "none",
        }

    pipe = _load_finbert()

    if pipe is not None:
        try:
            results = pipe(text[:512])[0]  # list of {label, score}
            probs = {}
            for item in results:
                probs[item["label"].lower()] = item["score"]

            pos = probs.get("positive", 0.0)
            neg = probs.get("negative", 0.0)
            neu = probs.get("neutral", 0.0)

            # Map to our labels
            if pos > neg and pos > neu:
                label = "bullish"
            elif neg > pos and neg > neu:
                label = "bearish"
            else:
                label = "neutral"

            # Composite score: [-1, +1]
            score = pos - neg

            return {
                "label": label,
                "score": float(score),
                "positive": float(pos),
                "negative": float(neg),
                "neutral": float(neu),
                "method": "finbert",
            }
        except Exception as exc:
            logger.warning(f"FinBERT inference failed: {exc}")

    return _keyword_sentiment(text)


def classify_headlines(texts: list[str]) -> list[dict]:
    """
    Classify multiple headlines efficiently.

    Uses batched inference with FinBERT when available.
    """
    if not texts:
        return []

    pipe = _load_finbert()

    if pipe is not None:
        try:
            truncated = [t[:512] for t in texts if t and t.strip()]
            if not truncated:
                return [classify_headline("") for _ in texts]

            batch_results = pipe(truncated)
            results = []
            trunc_idx = 0
            for original in texts:
                if not original or not original.strip():
                    results.append(classify_headline(""))
                    continue
                if trunc_idx < len(batch_results):
                    item_results = batch_results[trunc_idx]
                    trunc_idx += 1
                    probs = {}
                    for item in item_results:
                        probs[item["label"].lower()] = item["score"]

                    pos = probs.get("positive", 0.0)
                    neg = probs.get("negative", 0.0)
                    neu = probs.get("neutral", 0.0)

                    if pos > neg and pos > neu:
                        label = "bullish"
                    elif neg > pos and neg > neu:
                        label = "bearish"
                    else:
                        label = "neutral"

                    results.append({
                        "label": label,
                        "score": float(pos - neg),
                        "positive": float(pos),
                        "negative": float(neg),
                        "neutral": float(neu),
                        "method": "finbert",
                    })
                else:
                    results.append(classify_headline(original))

            return results

        except Exception as exc:
            logger.warning(f"FinBERT batch inference failed: {exc}")

    # Fallback to keyword-based
    return [_keyword_sentiment(t) if t else classify_headline("") for t in texts]


def aggregate_sentiment(sentiments: list[dict]) -> dict:
    """
    Aggregate multiple sentiment results into a summary.

    Returns {
        "average_score": float,
        "average_label": str,
        "bullish_count": int,
        "bearish_count": int,
        "neutral_count": int,
        "method": str,
    }
    """
    if not sentiments:
        return {
            "average_score": 0.0,
            "average_label": "neutral",
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "method": "none",
        }

    scores = [s["score"] for s in sentiments]
    avg_score = float(np.mean(scores))

    bullish = sum(1 for s in sentiments if s["label"] == "bullish")
    bearish = sum(1 for s in sentiments if s["label"] == "bearish")
    neutral = sum(1 for s in sentiments if s["label"] == "neutral")

    if avg_score > 0.15:
        avg_label = "bullish"
    elif avg_score < -0.15:
        avg_label = "bearish"
    else:
        avg_label = "neutral"

    methods = set(s.get("method", "unknown") for s in sentiments)

    return {
        "average_score": avg_score,
        "average_label": avg_label,
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": neutral,
        "method": "/".join(sorted(methods)),
    }
