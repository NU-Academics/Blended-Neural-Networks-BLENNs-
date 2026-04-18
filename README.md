```markdown
# BLENNS ORIGINAL — Complete Trading Pipeline

**Advanced AI-Powered Financial Markets Prediction with BFC Technology and Explainable AI**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*Revolutionizing market analysis with Blended Filtered Candles (BFC) and hybrid deep learning architecture*

---

## Overview

BLENNS (Blended Neural Networks) is a complete deep learning pipeline for financial trading that combines advanced signal processing with state-of-the-art deep learning architectures. The system features proprietary **BFC (Blended Filtered Candles)** technology for superior noise reduction, a **CNN-LSTM-Attention** hybrid architecture, and comprehensive **explainability** through SHAP values with expert rule validation.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **BFC Processing** | Three-stage filtering (EMA → Heikin-Ashi → Kalman) for clean signals |
| **Deep Learning** | CNN + LSTM + Attention architecture for temporal pattern recognition |
| **Multi-Asset Support** | Stocks, Crypto, Forex, Commodities, Indices (30+ assets) |
| **Walk-Forward Validation** | Robust time-series training preventing look-ahead bias |
| **SHAP Explainability** | Model interpretability with pixel-region attribution (RQ3) |
| **Expert Rule Validation** | Cohen's Kappa agreement with 5 technical analysis rules |
| **Uncertainty Estimation** | Monte Carlo dropout for prediction confidence intervals |
| **Statistical Tests** | Diebold-Mariano, Hansen's SPA, ANOVA, Tukey HSD |
| **Visual Analytics** | Comprehensive candlestick visualization and performance metrics |

---

## Supported Markets

| Category | Examples | Yahoo Finance Symbol |
|----------|----------|---------------------|
| **Stocks** | Apple, Microsoft, Amazon, NVIDIA, Tesla, Meta | `AAPL`, `MSFT`, `AMZN`, `NVDA`, `TSLA`, `META` |
| **Indices** | S&P 500, NASDAQ, Dow Jones, Russell 2000, VIX | `^GSPC`, `^NDX`, `^DJI`, `^RUT`, `^VIX` |
| **Cryptocurrency** | Bitcoin, Ethereum, Solana, Ripple, Binance Coin | `BTC-USD`, `ETH-USD`, `SOL-USD`, `XRP-USD`, `BNB-USD` |
| **Forex** | EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CHF, USD/CAD | `EURUSD=X`, `GBPUSD=X`, `JPY=X`, `AUDUSD=X`, `CHF=X`, `CAD=X` |
| **Commodities** | Gold, Silver, Crude Oil, Natural Gas, Copper, Corn | `GC=F`, `SI=F`, `CL=F`, `NG=F`, `HG=F`, `ZC=F` |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM recommended for full dataset processing
- GPU support recommended for faster training

### Quick Install

```bash
# Clone the repository
git clone https://github.com/NU-Academics/Blended-Neural-Networks-BLENNs-.git
cd Blended-Neural-Networks-BLENNs-
pip install -e .
```

### Google Colab Installation

```python
# One-click Colab setup
!pip install yfinance tensorflow shap mplfinance pillow scikit-learn scipy statsmodels
!git clone https://github.com/NU-Academics/Blended-Neural-Networks-BLENNs-.git
%cd Blended-Neural-Networks-BLENNs-
!pip install -e .
```

---

## Quick Start

### Basic Prediction

```python
from blenns_walk_forward import BLENNSWalkForward

# Initialize with any financial instrument
trader = BLENNSWalkForward(symbol="AAPL")

# Get instant prediction
result = trader.predict_next_day()
print(f"Next day prediction: {result['prediction']['direction']}")
print(f"Confidence: {result['prediction']['confidence']:.2%}")
```

### Multi-Asset Analysis

```python
from blenns_walk_forward import BLENNSWalkForward

# Analyze multiple markets simultaneously
symbols = ["AAPL", "BTC-USD", "EURUSD=X", "GC=F", "^SPX"]

for symbol in symbols:
    trader = BLENNSWalkForward(symbol=symbol)
    result = trader.predict_next_day()
    print(f"{symbol}: {result['prediction']['direction']} ({result['prediction']['confidence']:.1%} conf)")
```

### SHAP Explainability with Expert Rule Validation (RQ3)

```python
from blenns_walk_forward import BLENNSWalkForward
from blenns_walk_forward.utils import (
    compute_expert_signals,
    calculate_cohens_kappa,
    interpret_kappa,
    plot_kappa_agreement
)

# Train model and get predictions
trader = BLENNSWalkForward(symbol="AAPL")
data = trader.get_data()
data = trader.create_target(data)
X_img, X_vol, y, dates = trader.prepare_inputs(data)
trader.train_model()

# Compute SHAP predictions
holdout = trader.evaluate_holdout()
y_pred_prob = holdout['y_pred']
y_pred_bin = (y_pred_prob > 0.5).astype(int)

# Compute expert signals (5 technical rules)
expert_signal, _ = compute_expert_signals(data)

# Align and calculate Cohen's Kappa
min_len = min(len(y_pred_bin), len(expert_signal))
kappa_results = calculate_cohens_kappa(expert_signal[-min_len:], y_pred_bin[:min_len])

print(f"Cohen's Kappa: {kappa_results['kappa']:.4f}")
print(f"Interpretation: {interpret_kappa(kappa_results['kappa'])}")
print(f"p-value: {kappa_results['p_value']:.6f}")
```

---

## BLENNS Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BLENNS ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                    │
│  │   NUMERICAL │    │     BFC     │    │   VISUAL    │                    │
│  │   MODALITY  │───▶│PREPROCESSING│◀───│  MODALITY   │                    │
│  │   OHLCV +   │    │ 3-Stage     │    │ Candlestick │                    │
│  │   Volume    │    │ Filtering   │    │   Charts    │                    │
│  └─────────────┘    └──────┬──────┘    └─────────────┘                    │
│                            │                                              │
│                    ┌───────┴───────┐                                      │
│                    │   5-Day Window │                                      │
│                    │    Sequences   │                                      │
│                    └───────┬───────┘                                      │
│                            │                                              │
│         ┌──────────────────┴──────────────────┐                           │
│         │                                     │                           │
│    ┌────▼─────┐                         ┌─────▼────┐                      │
│    │   CNN    │                         │   LSTM   │                      │
│    │  Encoder │                         │ Network  │                      │
│    │(32→128)  │                         │ (64 units)│                      │
│    └────┬─────┘                         └─────┬────┘                      │
│         │                                     │                           │
│         └───────────────┬─────────────────────┘                           │
│                         │                                                 │
│                    ┌────▼─────┐                                           │
│                    │ SELF-    │                                           │
│                    │ ATTENTION│                                           │
│                    └────┬─────┘                                           │
│                         │                                                 │
│                    ┌────▼─────┐          ┌─────────────┐                 │
│                    │  FUSION  │◄─────────│   VOLUME    │                 │
│                    │  LAYER   │          │   (1-dim)   │                 │
│                    └────┬─────┘          └─────────────┘                 │
│                         │                                                 │
│                    ┌────▼─────┐                                           │
│                    │  DENSE   │                                           │
│                    │  LAYERS  │                                           │
│                    └────┬─────┘                                           │
│                         │                                                 │
│                    ┌────▼─────┐                                           │
│                    │  OUTPUT  │                                           │
│                    │PREDICTION│                                           │
│                    └────┬─────┘                                           │
│                         │                                                 │
│         ┌───────────────┴────────────────┐                               │
│         │                                │                               │
│    ┌────▼─────┐                    ┌─────▼────┐                          │
│    │   SHAP   │                    │  MONTE   │                          │
│    │EXPLANATION│                    │  CARLO   │                          │
│    │ (RQ3)    │                    │DROPOUT   │                          │
│    └──────────┘                    └──────────┘                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## BFC Processing Pipeline

```
Raw OHLCV Data
      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 1: Exponential Smoothing                     │
│                         P_t^EMA = α·P_t + (1-α)·P_{t-1}^EMA                │
│                         α = 0.2 (theoretically optimal)                    │
└─────────────────────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 2: Enhanced Heikin-Ashi                       │
│                    HA_Close = (O+H+L+C)/4                                  │
│                    HA_Open = (HA_Open_prev + HA_Close_prev)/2              │
└─────────────────────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 3: Adaptive Kalman Filter                    │
│                    x̂ₖ = x̂ₖ₋₁ + Kₖ(zₖ - x̂ₖ₋₁)                              │
│                    Kₖ = Pₖ⁻ / (Pₖ⁻ + Rₖ)                                  │
│                    Rₖ = λ·Rₖ₋₁ + (1-λ)·(zₖ - x̂ₖ₋₁)²                      │
└─────────────────────────────────────────────────────────────────────────────┘
      ↓
                         BFC-Filtered OHLC
```

---

## Neural Network Architecture Details

| Component | Layer Type | Parameters | Output Shape | Purpose |
|-----------|------------|------------|--------------|---------|
| **Input** | Image Input | (1, 64, 64, 3) | (None, 1, 64, 64, 3) | 5-day candlestick sequence |
| **CNN Block 1** | TimeDistributed(Conv2D) | 32 filters, 3×3 | (None, 1, 32, 32, 32) | Spatial feature extraction |
| **CNN Block 2** | TimeDistributed(Conv2D) | 64 filters, 3×3 | (None, 1, 16, 16, 64) | Hierarchical pattern learning |
| **CNN Block 3** | TimeDistributed(Conv2D) | 128 filters, 3×3 | (None, 1, 8, 8, 128) | High-level feature detection |
| **Flatten** | TimeDistributed(Flatten) | — | (None, 1, 8192) | Prepare for LSTM |
| **LSTM** | LSTM | 64 units | (None, 1, 64) | Temporal dependency modeling |
| **Attention** | Self-Attention | — | (None, 64) | Focus on relevant time steps |
| **Volume Input** | Input | (1,) | (None, 1) | Trading volume feature |
| **Fusion** | Concatenate | — | (None, 65) | Multi-modal feature integration |
| **Dense 1** | Dense | 32 units, ReLU | (None, 32) | Feature compression |
| **Dropout** | Dropout | 0.2 | (None, 32) | Regularization |
| **Output** | Dense | 1 unit, Sigmoid | (None, 1) | Probability prediction |

**Total Trainable Parameters:** ~1.85 million

---

## Expert Rules for Explainability Validation (RQ3)

| Rule | Source | Definition |
|------|--------|------------|
| **RSI** | Wilder (1978) | Buy when RSI < 30 (oversold), Sell when RSI > 70 (overbought) |
| **MACD** | Appel (2005) | Buy when MACD line crosses above Signal line |
| **MA Crossover** | Murphy (1999) | Buy when SMA20 > SMA50 |
| **Volume Confirmation** | Karpoff (1987) | Buy when volume > 1.5×VMA20 and price up |
| **Support/Resistance** | Edwards & Magee (1948) | Buy near support, Sell near resistance |

**Inter-expert agreement among five rules:** κ = 0.74 (substantial agreement)

---

## Cohen's Kappa for Explainability (RQ3)

**Formula:**
```
κ = (p_o - p_e) / (1 - p_e)

where:
p_o = observed agreement proportion
p_e = expected agreement by chance
```

**Interpretation Scale (Landis & Koch, 1977):**

| κ Range | Agreement |
|---------|-----------|
| < 0.00 | Poor |
| 0.00 – 0.20 | Slight |
| 0.21 – 0.40 | Fair |
| 0.41 – 0.60 | Moderate |
| 0.61 – 0.80 | Substantial |
| 0.81 – 1.00 | Almost Perfect |

---

## Statistical Tests

| Test | Purpose | Formula |
|------|---------|---------|
| **Diebold-Mariano** | Compare forecast accuracy | DM = d̄ / √[V(d̄)] |
| **Hansen's SPA** | Multi-model comparison with data snooping control | T_SPA = max(max √n d̄_k/ω̂_kk, 0) |
| **Welch's t-test** | Compare SNR with unequal variances | t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂) |
| **ANOVA** | Compare multiple group means | F = MS_between / MS_within |
| **Tukey HSD** | Post-hoc pairwise comparisons | q = (x̄_i - x̄_j) / √(MS_error / n) |
| **Cohen's Kappa** | Chance-corrected agreement | κ = (p_o - p_e) / (1 - p_e) |

---

## Performance Results (30 Assets)

| Asset Class | Assets | Avg BLENNS Accuracy | Avg Benchmark Accuracy | Avg Improvement |
|-------------|--------|---------------------|------------------------|-----------------|
| Equities | 6 | 96.29% | 56.72% | +69.83% |
| Indices | 6 | 90.91% | 49.69% | +41.22% |
| Forex | 6 | 91.80% | 49.10% | +42.70% |
| Commodities | 6 | 90.58% | 47.81% | +42.77% |
| Cryptocurrencies | 6 | 90.33% | 46.61% | +43.72% |
| **OVERALL** | **30** | **91.98%** | **49.99%** | **+48.05%** |

**All improvements are statistically significant (Diebold-Mariano, p < 0.001)**

---

## Model Parameters Summary

| Component | Parameters | Value |
|-----------|------------|-------|
| **BFC Alpha** | α | 0.2 |
| **BFC R** | Measurement noise | 0.01 |
| **BFC Q** | Process noise | 1e-5 |
| **Window Size** | Candlestick lookback | 5 days |
| **Image Size** | CNN input resolution | 64×64 pixels |
| **CNN Filters** | Conv layers | 32 → 64 → 128 |
| **LSTM Units** | Temporal encoding | 64 |
| **Dropout Rates** | Regularization | CNN:0.3, LSTM:0.4, Dense:0.2 |
| **Batch Size** | Training | 32 |
| **Epochs** | Maximum training | 50 |
| **Walk-Forward Folds** | Cross-validation | 5 |

---

## Usage Examples

### Basic Trading System

```python
from blenns_walk_forward import BLENNSWalkForward
import warnings
warnings.filterwarnings("ignore")

# Initialize
trader = BLENNSWalkForward(symbol="AAPL")

# Complete workflow
data = trader.get_data()
data = trader.create_target(data)
X_img, X_vol, y, dates = trader.prepare_inputs(data)
metrics = trader.train_model(n_splits=5, epochs=50)
result = trader.predict_next_day()

print(f"Prediction: {result['prediction']['direction']}")
print(f"Confidence: {result['prediction']['confidence']:.2%}")
print(f"Model Accuracy: {np.mean(metrics['fold_accs'])*100:.2f}%")
```

### SHAP Explainability with Expert Rules

```python
from blenns_walk_forward.utils import (
    compute_expert_signals,
    calculate_cohens_kappa,
    interpret_kappa,
    plot_kappa_agreement
)

# Get predictions
holdout = trader.evaluate_holdout()
y_pred_bin = (holdout['y_pred'] > 0.5).astype(int)

# Compute expert signals
expert_signal, _ = compute_expert_signals(data)

# Calculate Cohen's Kappa
kappa_results = calculate_cohens_kappa(expert_signal, y_pred_bin)
print(f"Kappa: {kappa_results['kappa']:.4f}")
print(f"p-value: {kappa_results['p_value']:.6f}")
print(f"Interpretation: {interpret_kappa(kappa_results['kappa'])}")
```

### Multi-Asset Backtest

```python
assets = ["AAPL", "MSFT", "BTC-USD", "EURUSD=X", "GC=F"]

results = {}
for symbol in assets:
    trader = BLENNSWalkForward(symbol=symbol)
    data = trader.get_data()
    data = trader.create_target(data)
    X_img, X_vol, y, dates = trader.prepare_inputs(data)
    metrics = trader.train_model(n_splits=3, epochs=30)
    results[symbol] = {
        'accuracy': np.mean(metrics['fold_accs']),
        'auc': np.mean(metrics['fold_aucs'])
    }
    print(f"{symbol}: {results[symbol]['accuracy']*100:.2f}%")

# Summary
for symbol, metrics in results.items():
    print(f"{symbol}: Acc={metrics['accuracy']*100:.2f}%, AUC={metrics['auc']*100:.2f}%")
```

---

## Configuration Options

### BFC Parameters

```python
bfc_params = {
    'alpha': 0.2,      # EMA smoothing (0.1 = heavy, 0.3 = light)
    'R': 0.01,         # Kalman measurement noise variance
    'Q': 1e-5          # Kalman process noise variance
}

trader = BLENNSWalkForward(symbol="AAPL", bfc_params=bfc_params)
```

### Training Parameters

```python
# Configured in BLENNSWalkForward methods
training_params = {
    'n_splits': 5,      # Walk-forward validation splits
    'epochs': 50,       # Training epochs per split
    'batch_size': 32,   # Training batch size
    'window_size': 5,   # Candlestick lookback window
    'img_size': 64      # Image dimensions (64×64)
}
```

---

## File Structure

```
blenns_walk_forward/
├── __init__.py          # Package exports
├── core.py              # BLENNSWalkForward main class
├── utils.py             # Utility functions (visualization, SHAP, ATR, expert rules)
├── models.py            # Model architecture definitions
├── bfc.py               # BFC processing functions
├── experts.py           # Expert trading rules (RQ3)
├── stats.py             # Statistical tests (DM, SPA, ANOVA, Kappa)
├── cli.py               # Command-line interface
├── data/                # Sample data files
├── models/              # Pre-trained model weights
├── config/              # Configuration files
└── results/             # Output results
```

---

## Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **Import Error** | `import sys; sys.path.append('/path/to/BLENNs-')` |
| **Memory Error** | Reduce dataset: `trader.get_data(start_date="2023-01-01")` |
| **Training Instability** | Adjust BFC: `bfc_params = {'alpha': 0.1, 'R': 0.005, 'Q': 1e-6}` |
| **Low Accuracy** | Increase epochs or adjust walk-forward splits |
| **SHAP Computation Slow** | Reduce background samples or use perturbation method |

### Google Colab Specific

```python
# Ensure GPU runtime
# Runtime → Change runtime type → GPU

# Install dependencies
!pip install yfinance tensorflow shap mplfinance pillow scikit-learn scipy statsmodels

# Clone and install
!git clone https://github.com/NU-Academics/Blended-Neural-Networks-BLENNs-.git
%cd Blended-Neural-Networks-BLENNs-
!pip install -e .

# Restart runtime if needed
import IPython
IPython.Application.instance().kernel.do_shutdown(True)
```

---

## Important Disclaimers

### Risk Warning
- This is a **research and educational tool**, not financial advice
- Always conduct thorough backtesting before live deployment
- Past performance does not guarantee future results
- Financial trading involves substantial risk of loss

### Limitations
- Daily timeframe focus (intraday requires retraining)
- Works best on liquid, non-manipulated assets
- Performance varies across market regimes
- Requires continuous model retraining for adaptation
- SHAP explanations show slight agreement (κ = 0.0967) — human oversight essential

---

## Contributing

We welcome contributions!

```bash
git clone https://github.com/NU-Academics/Blended-Neural-Networks-BLENNs-.git
cd Blended-Neural-Networks-BLENNs-
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Yahoo Finance for market data access
- MetaTrader 5 & Binance API for live trading data
- TensorFlow team for deep learning framework
- SHAP developers for model interpretability tools
- Dr. Irene Tsapara for valuable review and guidance
- Dr. Hamzah Al-Najada for constant review and suggestions

---

<div align="center">

**"The market is a device for transferring money from the impatient to the patient."** — Warren Buffett

*Built for the quantitative finance research community*

[⬆ Back to Top](#blenns-original--complete-trading-pipeline)

</div>
```
