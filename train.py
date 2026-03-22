"""
train.py - Train the LSTM stock predictor on real market data.

Usage:
    python train.py                            # default ticker (RELIANCE.NS)
    python train.py --ticker AAPL              # Apple
    python train.py --ticker TCS.NS --epochs 50
    python train.py --ticker INFY.NS --period 3y

The trained model + scalers are saved so that backtest.py and
advisor.py can load them without re-training.
"""

import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from config import Config
from data_fetcher import fetch_data
from model import LSTMStockPredictor


# -- Dataset ------------------------------------------------------------------
class StockDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_length: int, pred_length: int):
        self.data = data
        self.seq = seq_length
        self.pred = pred_length

    def __len__(self):
        return len(self.data) - self.seq - self.pred

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq]
        # target = Close price (index 3) for next pred_length days
        y = self.data[idx + self.seq : idx + self.seq + self.pred, 3]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# -- Training loop ------------------------------------------------------------
def train(cfg: Config, ticker: str, period: str, epochs: int | None = None):
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f" LSTM STOCK PREDICTOR — TRAINING")
    print(f" Ticker : {ticker}")
    print(f" Device : {device}")
    print(f"{'='*60}\n")

    # -- 1. Fetch data ----------------------------------------------------
    df = fetch_data(ticker, period)

    # -- 2. Scale ---------------------------------------------------------
    feature_scaler = MinMaxScaler()
    scaled = feature_scaler.fit_transform(df[cfg.FEATURE_COLS])

    close_scaler = MinMaxScaler()
    close_scaler.fit(df[["Close"]])

    # Save scalers for inference / backtest
    os.makedirs(os.path.dirname(cfg.SCALER_PATH) or ".", exist_ok=True)
    with open(cfg.SCALER_PATH, "wb") as f:
        pickle.dump({"feature": feature_scaler, "close": close_scaler}, f)
    print(f"[TRAIN] Scalers saved to {cfg.SCALER_PATH}")

    # -- 3. Datasets ------------------------------------------------------
    dataset = StockDataset(scaled, cfg.SEQ_LENGTH, cfg.PRED_LENGTH)
    n = len(dataset)
    train_n = int(cfg.TRAIN_RATIO * n)
    val_n = int(cfg.VAL_RATIO * n)
    test_n = n - train_n - val_n

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_n, val_n, test_n],
        generator=torch.Generator().manual_seed(cfg.SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    print(f"[TRAIN] Samples — train: {train_n}  val: {val_n}  test: {test_n}")

    # -- 4. Model ---------------------------------------------------------
    model = LSTMStockPredictor(
        input_dim=cfg.INPUT_DIM,
        hidden_dim=cfg.HIDDEN_DIM,
        num_layers=cfg.NUM_LAYERS,
        output_dim=cfg.OUTPUT_DIM,
        dropout=cfg.DROPOUT,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"[TRAIN] Model parameters: {params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg.LR_PATIENCE, factor=cfg.LR_FACTOR
    )

    # -- 5. Training ------------------------------------------------------
    num_epochs = epochs or cfg.NUM_EPOCHS
    best_val = float("inf")
    patience_ctr = 0
    history = {"train_loss": [], "val_loss": []}

    print(f"\n[TRAIN] Starting training — {num_epochs} epochs, patience={cfg.PATIENCE}\n")
    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        # --- train ---
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds, _ = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_loader)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds, _ = model(xb)
                val_loss += criterion(preds, yb).item()
        val_loss /= len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:3d}/{num_epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr_now:.2e}"
            )

        # --- early stopping ---
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), cfg.MODEL_PATH)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (patience={cfg.PATIENCE})")
                break

    elapsed = time.time() - t_start
    print(f"\n[TRAIN] Done in {elapsed:.1f}s — best val loss: {best_val:.6f}")
    print(f"[TRAIN] Model saved to {cfg.MODEL_PATH}")

    return history


# -- CLI -----------------------------------------------------------------------
def _parse():
    p = argparse.ArgumentParser(description="Train LSTM Stock Predictor")
    p.add_argument("--ticker", type=str, default=Config.DEFAULT_TICKER)
    p.add_argument("--period", type=str, default=Config.DATA_PERIOD)
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    cfg = Config()
    train(cfg, ticker=args.ticker, period=args.period, epochs=args.epochs)
