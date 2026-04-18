"""
BLENNS ORIGINAL — Complete Trading Pipeline
================================================================================
Advanced AI-powered trading prediction with BFC integration

Original implementation from BLENNS Original Notebook:
- BFC (Blended Filtered Candles): 3-stage noise reduction (EMA → Heikin-Ashi → Kalman)
- Hybrid Architecture: CNN + LSTM + Self-Attention with volume fusion
- Walk-Forward Validation: TimeSeriesSplit with configurable folds
- Uncertainty Estimation: Monte Carlo Dropout for prediction confidence
- Model Interpretability: SHAP GradientExplainer with expert rule validation
- Expert Rules: RSI, MACD, MA Crossover, Volume Confirmation, Support/Resistance
- Statistical Validation: Cohen's Kappa, Diebold-Mariano, Hansen's SPA

COMPATIBLE SYMBOLS:
    Equities: AAPL, MSFT, AMZN, NVDA, GOOGL, META, TSLA, TLRY
    Indices: ^SPX, ^NDX, ^DJI, ^RUT, ^SOX, ^VIX
    Crypto: BTC-USD, ETH-USD, SOL-USD, XRP-USD, BNB-USD
    Forex: EURUSD=X, GBPUSD=X, JPY=X, AUDUSD=X, CHF=X, CAD=X
    Commodities: GC=F (Gold), SI=F (Silver), CL=F (Oil), NG=F, HG=F, ZC=F

REFERENCES:
    - Cohen (1960) - Statistical agreement measures
    - Landis & Koch (1977) - Kappa interpretation
    - Lundberg & Lee (2017) - SHAP values
    - Gal & Ghahramani (2016) - Monte Carlo Dropout
    - Diebold & Mariano (1995) - Forecast comparison
    - Hansen (2005) - Superior Predictive Ability test
================================================================================
"""

from .core import BLENNSWalkForward
from .utils import (
    # Visualization
    visualize_candles,
    plot_training_curves,
    plot_roc_curve,
    plot_uncertainty_candle,
    plot_predicted_candle,
    plot_confusion_matrix,
    plot_shap_importance,
    plot_kappa_agreement,
    
    # Data Processing
    normalize_data,
    
    # Analysis Utilities
    explain_model_with_shap,
    monte_carlo_predict,
    compute_atr,
    atr_multipliers,
    
    # Expert Rules (RQ3)
    compute_expert_signals,
    calculate_cohens_kappa,
    interpret_kappa,
    
    # Statistical Tests
    diebold_mariano_test,
    hansen_spa_test,
    
    # ATR Calculation
    compute_atr_vectorized,
    compute_atr_series,
    get_tp_sl_multipliers
)

__version__ = "2.0.0"
__author__ = "BLENNS Contributors"

__all__ = [
    # Core classes
    'BLENNSWalkForward',
    
    # Visualization
    'visualize_candles',
    'plot_training_curves',
    'plot_roc_curve',
    'plot_uncertainty_candle',
    'plot_predicted_candle',
    'plot_confusion_matrix',
    'plot_shap_importance',
    'plot_kappa_agreement',
    
    # Data Processing
    'normalize_data',
    
    # Analysis Utilities
    'explain_model_with_shap',
    'monte_carlo_predict',
    'compute_atr',
    'atr_multipliers',
    
    # Expert Rules (RQ3)
    'compute_expert_signals',
    'calculate_cohens_kappa',
    'interpret_kappa',
    
    # Statistical Tests
    'diebold_mariano_test',
    'hansen_spa_test',
    
    # ATR Calculation
    'compute_atr_vectorized',
    'compute_atr_series',
    'get_tp_sl_multipliers'
]
