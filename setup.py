#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
BLENNS ORIGINAL — Complete Trading Pipeline
Setup script for package installation
================================================================================
Author: BLENNS Framework Implementation
Date: April 2026

DESCRIPTION:
    BLENNS (Blended Neural Networks) is a complete deep learning pipeline for
    financial trading that combines:

    1. BFC (Blended Filtered Candles) - Three-stage noise reduction
       (Exponential Smoothing → Heikin-Ashi → Adaptive Kalman Filter)
    2. CNN + LSTM + Attention hybrid architecture
    3. Walk-forward validation with TimeSeriesSplit
    4. Monte Carlo Dropout uncertainty estimation
    5. SHAP explainability with expert rule validation
    6. Statistical tests (Diebold-Mariano, Hansen's SPA, Cohen's Kappa)

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

from setuptools import setup, find_packages
import os
import sys
import re

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def read_file(filename):
    """Read file content with UTF-8 encoding"""
    this_directory = os.path.abspath(os.path.dirname(__file__))
    filepath = os.path.join(this_directory, filename)
    try:
        with open(filepath, encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return None


def read_requirements(filename='requirements.txt'):
    """Read requirements from requirements.txt"""
    this_directory = os.path.abspath(os.path.dirname(__file__))
    req_file = os.path.join(this_directory, filename)
    
    if os.path.exists(req_file):
        with open(req_file, encoding='utf-8') as f:
            requirements = []
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    # Handle conditional requirements
                    if ';' in line:
                        # Check environment marker
                        condition = line.split(';')[1].strip()
                        # Simple environment check (expand as needed)
                        if 'sys_platform' in condition:
                            continue  # Skip for now, handle in extras
                        requirements.append(line)
                    else:
                        requirements.append(line)
            return requirements
    else:
        # Fallback dependencies (minimal set from BLENNS Original)
        return [
            "numpy>=1.21.0,<1.25.0",
            "pandas>=1.3.0,<2.0.0",
            "yfinance>=0.2.0,<0.3.0",
            "matplotlib>=3.5.0,<3.8.0",
            "mplfinance>=0.12.0,<0.13.0",
            "Pillow>=9.0.0,<10.0.0",
            "scikit-learn>=1.0.0,<1.3.0",
            "shap>=0.41.0,<0.43.0",
            "tensorflow>=2.10.0,<2.15.0",
            "scipy>=1.8.0,<1.10.0",
            "statsmodels>=0.13.0,<0.15.0",
        ]


def get_version():
    """Read version from VERSION file or from core module"""
    version_file = read_file('VERSION')
    if version_file:
        return version_file.strip()
    
    # Fallback version
    return "2.0.0"


def get_long_description():
    """Get long description from README.md"""
    readme = read_file('README.md')
    if readme:
        return readme
    else:
        return """
# BLENNS ORIGINAL — Complete Trading Pipeline

BLENNS (Blended Neural Networks) is a complete deep learning pipeline for financial trading.

## Features

- **BFC (Blended Filtered Candles)**: Three-stage noise reduction (EMA → Heikin-Ashi → Kalman)
- **Hybrid Architecture**: CNN + LSTM + Attention with volume fusion
- **Walk-Forward Validation**: TimeSeriesSplit with configurable folds
- **Uncertainty Estimation**: Monte Carlo Dropout for prediction confidence
- **Model Interpretability**: SHAP explainability with expert rule validation
- **Statistical Tests**: Diebold-Mariano, Hansen's SPA, Cohen's Kappa

## Quick Start

```python
from blenns_walk_forward import BLENNSWalkForward

# Initialize trader
trader = BLENNSWalkForward(symbol='AAPL')

# Get prediction for next day
result = trader.predict_next_day()
print(f"Prediction: {result['prediction']['direction']}")
print(f"Confidence: {result['prediction']['confidence']:.2%}")
