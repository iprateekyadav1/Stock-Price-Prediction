"""
config.py - Centralised hyperparameters for LSTM Stock Predictor.

All magic numbers live here so tuning, grid-search, or
environment-variable overrides touch one file.
"""

import os


class Config:
    # ── Random seed ───────────────────────────────────────────────────────
    SEED: int = 42

    # ── Data ─────────────────────────────────────────────────────────────
    DEFAULT_TICKER: str = "RELIANCE.NS"      # default ticker (Reliance on NSE)
    DATA_PERIOD: str = "5y"                  # how much history to fetch
    SEQ_LENGTH: int = 60                     # look-back window (trading days)
    PRED_LENGTH: int = 5                     # forecast horizon (trading days)
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    # test ratio is implicit: 1 - TRAIN_RATIO - VAL_RATIO = 0.15

    FEATURE_COLS = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_10", "SMA_30", "EMA_12", "EMA_26",
        "MACD", "MACD_Signal", "RSI",
        "BB_Upper", "BB_Lower", "Volume_Ratio",
    ]

    # ── Model ─────────────────────────────────────────────────────────────
    INPUT_DIM: int = len(FEATURE_COLS)       # derived automatically
    HIDDEN_DIM: int = 128
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.2
    OUTPUT_DIM: int = PRED_LENGTH            # one output per forecast day

    # ── Training ──────────────────────────────────────────────────────────
    BATCH_SIZE: int = 32
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-5
    NUM_EPOCHS: int = 100
    PATIENCE: int = 15                       # early-stopping patience
    GRAD_CLIP: float = 1.0
    LR_PATIENCE: int = 5                     # ReduceLROnPlateau patience
    LR_FACTOR: float = 0.5

    # ── Backtest / Signals ────────────────────────────────────────────────
    SIGNAL_THRESHOLD: float = 0.01           # 1% move = signal (best from testing)
    INITIAL_CAPITAL: float = 100_000.0       # Rs. 1 lakh starting capital
    RISK_FREE_RATE: float = 0.06 / 252       # India RBI rate ~6% annualised
    BENCHMARK_TICKER: str = "^NSEI"          # Nifty 50 index for alpha/beta

    # ── Screener ──────────────────────────────────────────────────────────
    FUND_WEIGHT: float = 0.40                # fundamental score weight
    TECH_WEIGHT: float = 0.35                # technical score weight
    MOM_WEIGHT: float = 0.25                 # momentum score weight

    # Multibagger detection thresholds
    MIN_ROE: float = 0.15
    MAX_DEBT_EQUITY: float = 1.0
    MIN_EARNINGS_GROWTH: float = 0.15
    MIN_FCF: float = 0                       # positive free cash flow

    # Stock universe for scanning (liquid NSE stocks)
    NIFTY_STOCKS = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
        "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
        "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "WIPRO.NS",
        "HCLTECH.NS", "TATACONSUM.NS", "NTPC.NS", "POWERGRID.NS", "TATASTEEL.NS",
        "JSWSTEEL.NS", "ADANIENT.NS", "TECHM.NS", "BAJAJ-AUTO.NS", "INDUSINDBK.NS",
    ]

    # ── Paths ─────────────────────────────────────────────────────────────
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH: str = os.path.join(BASE_DIR, "best_lstm_model.pth")
    SCALER_PATH: str = os.path.join(BASE_DIR, "scalers.pkl")
    CACHE_DIR: str = os.path.join(BASE_DIR, "cache")
    RESULTS_DIR: str = os.path.join(BASE_DIR, "results")
