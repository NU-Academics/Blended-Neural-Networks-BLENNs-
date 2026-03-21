"""
BLENNS Walk Forward Trading System
Advanced AI-powered trading prediction with BFC integration

Original implementation from BLENNS Original Notebook:
- BFC (Blenns Filter Candles): 3-stage noise reduction (EMA → Heikin-Ashi → Kalman)
- Hybrid Architecture: CNN + LSTM + Self-Attention with volume fusion
- Walk-Forward Validation: TimeSeriesSplit with configurable folds
- Uncertainty Estimation: Monte Carlo Dropout for prediction confidence
- Model Interpretability: SHAP GradientExplainer for feature importance
"""

from .core import BLENNSWalkForward
from .bfc import compute_bfc, exponential_moving_average, kalman_filter
from .data import get_yfinance_data, encode_candle_chart, normalize_data
from .model import blenns_trading_model
from .utils import (
    visualize_candles,
    plot_training_curves,
    explain_model_with_shap,
    plot_roc_curve,
    plot_uncertainty_candle,
    monte_carlo_predict,
    compute_atr,
    atr_multipliers,
    plot_predicted_candle
)

__version__ = "2.0.0"
__author__ = "BLENNS Contributors"
__all__ = [
    # Core classes
    'BLENNSWalkForward',
    
    # BFC filter functions
    'compute_bfc',
    'exponential_moving_average',
    'kalman_filter',
    
    # Data handling
    'get_yfinance_data',
    'encode_candle_chart',
    'normalize_data',
    
    # Model architecture
    'blenns_trading_model',
    
    # Visualization utilities
    'visualize_candles',
    'plot_training_curves',
    'plot_roc_curve',
    'plot_uncertainty_candle',
    'plot_predicted_candle',
    
    # Analysis utilities
    'explain_model_with_shap',
    'monte_carlo_predict',
    'compute_atr',
    'atr_multipliers'
]
