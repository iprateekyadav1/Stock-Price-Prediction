"""
model.py - LSTM + Attention architecture for stock price forecasting.

Shared by train.py, backtest.py, and advisor.py so the architecture
is defined in exactly one place.
"""

import torch
import torch.nn as nn


class Attention(nn.Module):
    """Additive attention over LSTM hidden states."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, lstm_output: torch.Tensor):
        # lstm_output: (batch, seq_len, hidden_dim)
        weights = torch.softmax(self.attn(lstm_output).squeeze(-1), dim=1)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)
        return context, weights


class LSTMStockPredictor(nn.Module):
    """
    LSTM encoder -> Attention pooling -> FC decoder.

    Parameters
    ----------
    input_dim  : number of input features per time-step
    hidden_dim : LSTM hidden size
    num_layers : stacked LSTM layers
    output_dim : number of future steps to predict
    dropout    : dropout rate (applied between LSTM layers + FC)
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = Attention(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor):
        out, _ = self.lstm(x)
        context, attn_weights = self.attention(out)
        preds = self.fc(context)
        return preds, attn_weights
