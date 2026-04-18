# 🚀 BLENNS ORIGINAL — Complete Trading Pipeline

**Advanced AI-Powered Financial Market Prediction using BFC Technology and Explainable AI**

---

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)

> *Revolutionizing financial forecasting with Blended Filtered Candles (BFC) and hybrid deep learning architectures.*

---

# 📌 Overview

**BLENNS (Blended Neural Networks)** is a full-stack deep learning framework for financial market prediction.  
It integrates:

- **Signal Processing** (BFC filtering)
- **Deep Learning** (CNN + LSTM + Attention)
- **Explainable AI** (SHAP + Expert Validation)
- **Statistical Testing** (robust evaluation)

The system is designed for **multi-asset prediction**, **interpretability**, and **research-grade validation**.

---

# ⭐ Key Features

| Feature | Description |
|--------|-------------|
| 🔹 **BFC Processing** | EMA → Heikin-Ashi → Kalman filtering |
| 🔹 **Deep Learning** | CNN + LSTM + Attention hybrid |
| 🔹 **Multi-Asset Support** | Stocks, Crypto, Forex, Commodities |
| 🔹 **Walk-Forward Validation** | Prevents look-ahead bias |
| 🔹 **SHAP Explainability** | Pixel-level interpretability |
| 🔹 **Expert Rule Validation** | Cohen’s Kappa agreement |
| 🔹 **Uncertainty Estimation** | Monte Carlo Dropout |
| 🔹 **Statistical Testing** | DM, SPA, ANOVA, Tukey |
| 🔹 **Visualization** | Candles, ROC, confusion matrix |

---

# 🌍 Supported Markets

| Category | Examples | Symbols |
|----------|----------|--------|
| **Stocks** | Apple, Tesla, NVIDIA | `AAPL`, `TSLA`, `NVDA` |
| **Indices** | S&P 500, NASDAQ | `^GSPC`, `^NDX` |
| **Crypto** | Bitcoin, Ethereum | `BTC-USD`, `ETH-USD` |
| **Forex** | EUR/USD, GBP/USD | `EURUSD=X`, `GBPUSD=X` |
| **Commodities** | Gold, Oil | `GC=F`, `CL=F` |

---

# ⚙️ Installation

## Requirements

- Python ≥ 3.8  
- 8GB+ RAM recommended  
- GPU (optional but recommended)

---

## 🔧 Local Installation

```bash
git clone https://github.com/NU-Academics/Blended-Neural-Networks-BLENNs-.git
cd Blended-Neural-Networks-BLENNs-
pip install -e .


 Google Colab Setup
!pip install yfinance tensorflow shap mplfinance pillow scikit-learn scipy statsmodels
!git clone https://github.com/NU-Academics/Blended-Neural-Networks-BLENNs-.git
%cd Blended-Neural-Networks-BLENNs-
!pip install -e .


Quick Start
🔹 Basic Prediction
from blenns_walk_forward import BLENNSWalkForward

trader = BLENNSWalkForward(symbol="AAPL")
result = trader.predict_next_day()

print(result['prediction']['direction'])
print(result['prediction']['confidence'])


🔹 Multi-Asset Prediction
symbols = ["AAPL", "BTC-USD", "EURUSD=X"]

for s in symbols:
    trader = BLENNSWalkForward(symbol=s)
    r = trader.predict_next_day()
    print(f"{s}: {r['prediction']['direction']} ({r['prediction']['confidence']:.2%})")


Model Architecture
BFC → CNN → LSTM → Attention → Fusion → Dense → Prediction
                    ↑
                 Volume

Pipeline Flow:

BFC Filtering
Candlestick Image Encoding
CNN Feature Extraction
LSTM Temporal Learning
Attention Weighting
Volume Fusion
Prediction Output


BFC Processing Pipeline
Raw OHLCV
   ↓
EMA Smoothing
   ↓
Heikin-Ashi Transformation
   ↓
Kalman Filtering
   ↓
BFC Candles


 Neural Network Details



Layer
Purpose




CNN
Spatial pattern detection


LSTM
Temporal sequence learning


Attention
Focus on important timesteps


Volume Input
Market activity signal


Dense Layers
Final prediction




 Explainability (RQ3)
BLENNS uses SHAP values to explain predictions and validates them using:
Expert Rules

RSI
MACD
Moving Average crossover
Volume confirmation
Support/Resistance


Cohen’s Kappa
[
\kappa = \frac{p_o - p_e}{1 - p_e}
]



Range
Meaning




0.61–0.80
Substantial


0.81–1.00
Almost Perfect




Statistical Validation

Diebold-Mariano Test
Hansen SPA Test
ANOVA & Tukey HSD
Cohen’s Kappa


 Performance Summary



Market
Accuracy




Equities
96.29%


Indices
90.91%


Forex
91.80%


Commodities
90.58%


Crypto
90.33%


Overall
91.98%




⚙️ Configuration
BFC Parameters
bfc_params = {
    'alpha': 0.2,
    'R': 0.01,
    'Q': 1e-5
}


Training Settings
params = {
    'epochs': 50,
    'batch_size': 32,
    'window_size': 5,
    'img_size': 64
}


Project Structure
blenns_walk_forward/
├── core.py
├── models.py
├── bfc.py
├── utils.py
├── stats.py
├── experts.py
├── cli.py


Troubleshooting



Issue
Fix




Import error
Add repo to path


Memory issue
Reduce dataset


Slow SHAP
Reduce samples


Low accuracy
Tune parameters




Disclaimer

Not financial advice
For research purposes only
Trading involves risk
Always validate before deployment


 Contributing
git clone https://github.com/NU-Academics/Blended-Neural-Networks-BLENNs-.git
pip install -e ".[dev]"
pytest


 License
MIT License

 Acknowledgments

Yahoo Finance
TensorFlow
SHAP
Academic supervisors and contributors


“The market is a device for transferring money from the impatient to the patient.” — Warren Buffett
