# 📈 Stock Price Prediction with LSTM + Attention

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview

Built an **LSTM neural network with Attention mechanism** for multi-step stock price forecasting. The model predicts the next 5 days of closing prices using technical indicators and historical price data.

**Key Highlights:**
- 🧠 **Attention Mechanism** for interpretable predictions
- 📊 **Multi-step forecasting** (5-day prediction horizon)
- 🔧 **Feature Engineering** with technical indicators (RSI, MACD, Bollinger Bands)
- 📉 **Strong performance** with low RMSE and MAPE

## 🎯 Business Applications

- **Algorithmic Trading:** Automated trading signal generation
- **Risk Management:** Portfolio risk assessment
- **Investment Analysis:** Price movement prediction for decision support

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| Framework | PyTorch |
| Architecture | LSTM + Attention Mechanism |
| Features | 15 technical indicators |
| Forecast | 5-day multi-step prediction |
| Evaluation | RMSE, MAE, MAPE, R² |

## 📊 Model Architecture

```
Input (batch, 60 days, 15 features)
    ↓
[Stacked LSTM] × 2 layers (hidden=128)
    ↓
[Attention Layer] - learns which timesteps matter most
    ↓
[Fully Connected] 128 → 64 → 5 (days)
    ↓
Predicted Prices (5 days)
```

## 📈 Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSE** | $3.45 | Average prediction error |
| **MAE** | $2.78 | Mean absolute error |
| **MAPE** | 2.8% | Percentage error |
| **R² Score** | 0.89 | 89% variance explained |

## 🔍 Explainability: Attention Visualization

The attention mechanism reveals which historical time steps the model focuses on:

![Attention Weights](attention_visualization.png)

This provides **interpretability** — crucial for financial applications where decisions need justification.

## 🚀 Usage

```bash
# Install dependencies
pip install torch numpy pandas matplotlib scikit-learn

# Run notebook
jupyter notebook lstm_stock_prediction.ipynb
```

## 💡 Key Learnings

1. **Attention is powerful** — enables the model to focus on relevant time periods
2. **Feature engineering matters** — technical indicators significantly improve performance
3. **Multi-step prediction** is harder but more practical than single-step
4. **Gradient clipping** prevents exploding gradients in RNNs

## 📚 Technical Indicators Used

| Indicator | Purpose |
|-----------|---------|
| SMA (10, 30) | Trend direction |
| EMA (12, 26) | Weighted trend |
| MACD | Momentum indicator |
| RSI | Overbought/oversold |
| Bollinger Bands | Volatility measurement |
| Volume Ratio | Trading activity |

## 🔗 Next Steps

- [ ] Backtest trading strategy using predictions
- [ ] Add news sentiment analysis as feature
- [ ] Deploy as real-time prediction API
- [ ] Ensemble with Transformer models

---

**Author:** Prateek  
**Date:** February 2025  
**Project Type:** Deep Learning / Time Series / Financial AI
