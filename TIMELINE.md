# LSTM Stock Predictor - Timeline to Real-Time Usage

## Current State (What you have NOW)

| Component | File | Status |
|-----------|------|--------|
| Config | `config.py` | Ready |
| Model architecture | `model.py` | Ready |
| Real data pipeline | `data_fetcher.py` | Ready (yfinance) |
| Training script | `train.py` | Ready |
| Backtesting + metrics | `backtest.py` | Ready |
| Interactive advisor | `advisor.py` | Ready (asks permission) |

---

## Backtest Results (Actual runs on real data)

### RELIANCE.NS (5 years, 1208 trading days)

| Run | Epochs | Val Loss | Accuracy | Precision | Recall | F1 |
|-----|--------|----------|----------|-----------|--------|-----|
| 1 (0.5% threshold) | 60 | 0.0026 | 48.3% | 0.53 | 0.48 | 0.50 |
| 2 (0.5% threshold) | 88* | 0.0023 | 52.3% | 0.57 | 0.52 | 0.54 |
| 2 (1.0% threshold) | 88* | 0.0023 | **53.5%** | **0.59** | 0.53 | 0.54 |

*Early stopped

#### Best result (1% threshold) - per-signal breakdown:
| Signal | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| SELL | **0.72** | 0.42 | 0.53 |
| HOLD | 0.37 | 0.53 | 0.43 |
| BUY | 0.60 | 0.54 | 0.57 |

### TCS.NS (cross-validation)

| Metric | Value |
|--------|-------|
| Accuracy | 48.8% |
| SELL precision | 0.62 |
| BUY precision | 0.40 |

### Interpretation
- The model **outperforms random guessing** (33% for 3-class) by ~20 points
- **SELL signals are the most reliable** (72% precision on RELIANCE)
- **BUY signals are moderately reliable** (60% precision on RELIANCE)
- HOLD is the weakest signal (expected - it's the noise band)
- The 1% threshold works better than 0.5% (filters out noise)

---

## Timeline: From Here to Real-Time Usage

### Phase 1: Paper Trading (Weeks 1-4) - DO THIS FIRST

**Goal:** Validate the model in real-time WITHOUT risking money.

```
Week 1-2:  Run advisor.py daily for your chosen ticker
           Record every signal, track whether it was right or wrong
           Command:  python advisor.py --ticker RELIANCE.NS

Week 3-4:  Compare your paper-trade P&L against buy-and-hold
           If the model's signals would have lost money: STOP, retune
           If positive: proceed to Phase 2
```

**Daily routine:**
1. Open terminal in project folder
2. `python advisor.py --ticker RELIANCE.NS`
3. Read the advisory, decide whether to act
4. Advisor logs your decision to `results/advisor_log.csv`
5. After market close, check if prediction was correct

### Phase 2: Retrain + Multi-Ticker (Weeks 5-8)

**Goal:** Improve accuracy and test across stocks you care about.

```
- Retrain on multiple tickers you follow:
    python train.py --ticker RELIANCE.NS
    python backtest.py --ticker RELIANCE.NS --threshold 0.01

    python train.py --ticker TCS.NS
    python backtest.py --ticker TCS.NS --threshold 0.01

    python train.py --ticker INFY.NS
    python backtest.py --ticker INFY.NS --threshold 0.01

- Compare metrics across tickers
- Identify which stocks the model predicts best
- Tune threshold per ticker (some need 0.5%, others 2%)
```

### Phase 3: Small Real Trades (Weeks 9-12)

**Prerequisites before putting real money:**
1. Paper-trade accuracy > 55% sustained over 4+ weeks
2. SELL precision > 60% (to avoid false exit signals)
3. BUY precision > 55% (to avoid false entry signals)
4. You understand the model CAN be wrong 40-50% of the time

**Rules:**
- Start with amount you can afford to lose ENTIRELY
- Only act on HIGH-CONFIDENCE signals (confidence > 70%)
- Always use stop-losses (the model does NOT manage risk)
- The advisor ALWAYS asks your permission - never bypass this

### Phase 4: Automation + Monitoring (Month 4+)

**Only after Phase 3 is profitable:**

```
- Add a cron/scheduled task to retrain weekly:
    python train.py --ticker RELIANCE.NS  (every Sunday)

- Add Telegram/email alerts when signals appear
  (build on top of advisor.py)

- Connect to broker API (Zerodha Kite, Alpaca, etc.)
  for order placement - BUT keep the permission prompt

- Build a dashboard to track cumulative accuracy
```

---

## How to Use Right Now

```bash
# Navigate to project
cd portfolio-projects/project-2-lstm-stock-prediction

# Step 1: Train on your ticker
python train.py --ticker RELIANCE.NS

# Step 2: Check how good the model is
python backtest.py --ticker RELIANCE.NS --threshold 0.01

# Step 3: Get today's advisory (asks your permission)
python advisor.py --ticker RELIANCE.NS
```

---

## Important Disclaimers

1. This is a LEARNING/RESEARCH tool, not guaranteed income
2. Past backtest results do NOT guarantee future performance
3. The model is wrong ~47% of the time on average
4. Always do your own research before any trade
5. Never invest money you cannot afford to lose
6. The advisor is advisory only - final decision is always yours
