"""
config.py - Centralised hyperparameters for LSTM Stock Predictor.

All magic numbers live here so tuning, grid-search, or
environment-variable overrides touch one file.
"""

import os
import re


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
    SIGNAL_THRESHOLD: float = 0.005          # 0.5% default (adaptive overrides this)
    INITIAL_CAPITAL: float = 100_000.0       # Rs. 1 lakh starting capital
    RISK_FREE_RATE: float = 0.06 / 252       # India RBI rate ~6% annualised
    BENCHMARK_TICKER: str = "^NSEI"          # Nifty 50 index for alpha/beta
    BACKTEST_TEST_RATIO: float = 0.30        # 30% of data for out-of-sample testing

    # ── Confidence Engine ────────────────────────────────────────────────
    MC_DROPOUT_SAMPLES: int = 30             # Monte Carlo Dropout forward passes
    CONFIDENCE_WEIGHTS: dict = {
        "prediction_consistency": 0.30,      # MC Dropout spread
        "technical_alignment": 0.25,         # RSI/MACD/SMA agreement
        "volatility_regime": 0.20,           # Recent vs historical vol
        "model_certainty": 0.25,             # Attention entropy + magnitude
    }

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
    # ── Original NIFTY 30 (backward compat) ──
    NIFTY_STOCKS = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
        "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
        "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "WIPRO.NS",
        "HCLTECH.NS", "TATACONSUM.NS", "NTPC.NS", "POWERGRID.NS", "TATASTEEL.NS",
        "JSWSTEEL.NS", "ADANIENT.NS", "TECHM.NS", "BAJAJ-AUTO.NS", "INDUSINDBK.NS",
    ]

    # ── Expanded Indian universe (Top 100 by market cap) ──
    # Source: BSE/NSE Top 200 + BS 1000 Index (parsed from PDF data)
    INDIAN_STOCKS = [
        # ── Mega Cap (₹5L+ Cr) ──
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
        # ── Large Cap I (₹1L-5L Cr) ──
        "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
        "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "WIPRO.NS",
        "HCLTECH.NS", "TATACONSUM.NS", "NTPC.NS", "POWERGRID.NS", "TATASTEEL.NS",
        "JSWSTEEL.NS", "ADANIENT.NS", "TECHM.NS", "BAJAJ-AUTO.NS", "INDUSINDBK.NS",
        "DMART.NS", "BAJAJFINSV.NS", "TATAMOTORS.NS", "HINDZINC.NS", "DIVISLAB.NS",
        "PIDILITIND.NS", "VEDL.NS", "SBILIFE.NS", "IOC.NS", "GRASIM.NS",
        # ── Large Cap II (₹50K-1L Cr) ──
        "DABUR.NS", "M&M.NS", "GODREJCP.NS", "HINDALCO.NS", "SHREECEM.NS",
        "COALINDIA.NS", "BPCL.NS", "DLF.NS", "BRITANNIA.NS", "HAVELLS.NS",
        "DRREDDY.NS", "AMBUJACEM.NS", "SIEMENS.NS", "CIPLA.NS", "INDIGO.NS",
        "EICHERMOT.NS", "MARICO.NS", "APOLLOHOSP.NS", "GAIL.NS", "ADANIPORTS.NS",
        "HDFCLIFE.NS", "ONGC.NS", "ADANIGREEN.NS", "BEL.NS", "HAL.NS",
        "TATAPOWER.NS", "HEROMOTOCO.NS", "IRCTC.NS", "ZOMATO.NS",
        # ── Mid Cap Elite (₹25K-50K Cr) ──
        "POLYCAB.NS", "TRENT.NS", "BHARATFORG.NS", "ABB.NS", "VBL.NS",
        "PFC.NS", "RECLTD.NS", "NHPC.NS", "TATAELXSI.NS", "PERSISTENT.NS",
        "DIXON.NS", "COFORGE.NS", "JUBLFOOD.NS", "SAIL.NS", "BANKBARODA.NS",
        "BOSCHLTD.NS", "JINDALSTEL.NS", "GODREJPROP.NS", "TORNTPHARM.NS",
        "MUTHOOTFIN.NS", "SRF.NS", "UPL.NS", "LUPIN.NS", "BIOCON.NS",
        "COLPAL.NS", "TVSMOTOR.NS", "CUMMINSIND.NS", "BATAINDIA.NS",
        "PIIND.NS", "PAGEIND.NS", "MRF.NS",
    ]

    STARTER_MODEL_TICKERS = [
        "AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    ]

    # ── Paths ─────────────────────────────────────────────────────────────
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH: str = os.path.join(BASE_DIR, "best_lstm_model.pth")
    SCALER_PATH: str = os.path.join(BASE_DIR, "scalers.pkl")
    CACHE_DIR: str = os.path.join(BASE_DIR, "cache")
    RESULTS_DIR: str = os.path.join(BASE_DIR, "results")
    MODELS_DIR: str = os.path.join(BASE_DIR, "artifacts", "models")
    SCALERS_DIR: str = os.path.join(BASE_DIR, "artifacts", "scalers")
    METADATA_DIR: str = os.path.join(BASE_DIR, "artifacts", "metadata")

    @staticmethod
    def safe_ticker(ticker: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_").upper()

    @classmethod
    def get_model_path(cls, ticker: str) -> str:
        return os.path.join(cls.MODELS_DIR, f"{cls.safe_ticker(ticker)}.pth")

    @classmethod
    def get_scaler_path(cls, ticker: str) -> str:
        return os.path.join(cls.SCALERS_DIR, f"{cls.safe_ticker(ticker)}.pkl")

    @classmethod
    def get_metadata_path(cls, ticker: str) -> str:
        return os.path.join(cls.METADATA_DIR, f"{cls.safe_ticker(ticker)}.json")

    @classmethod
    def resolve_model_path(cls, ticker: str) -> str:
        ticker_path = cls.get_model_path(ticker)
        if os.path.exists(ticker_path):
            return ticker_path
        if ticker == cls.DEFAULT_TICKER and os.path.exists(cls.MODEL_PATH):
            return cls.MODEL_PATH
        return ticker_path

    @classmethod
    def resolve_scaler_path(cls, ticker: str) -> str:
        ticker_path = cls.get_scaler_path(ticker)
        if os.path.exists(ticker_path):
            return ticker_path
        if ticker == cls.DEFAULT_TICKER and os.path.exists(cls.SCALER_PATH):
            return cls.SCALER_PATH
        return ticker_path
