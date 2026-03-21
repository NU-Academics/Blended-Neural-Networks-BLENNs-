#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BLENNS Walk-Forward Trading System
Setup script for package installation

BLENNS (Blended Neural Networks) combines:
- BFC 3-stage filtering (EMA → Heikin-Ashi → Kalman)
- CNN + LSTM + Attention architecture
- Walk-forward validation
- Monte Carlo uncertainty estimation
- SHAP model interpretability
"""

from setuptools import setup, find_packages
import os
import sys

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "BLENNS Walk-Forward Trading System - Advanced AI-powered trading prediction with BFC integration"

# Read requirements from requirements.txt if available
def read_requirements():
    """Read requirements from requirements.txt"""
    req_file = os.path.join(this_directory, 'requirements.txt')
    if os.path.exists(req_file):
        with open(req_file, encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        # Fallback dependencies (minimal set from BLENNS Original)
        return [
            "numpy>=1.19.0",
            "pandas>=1.2.0",
            "yfinance>=0.1.70",
            "matplotlib>=3.3.0",
            "shap>=0.40.0",
            "tensorflow>=2.8.0",
            "mplfinance>=0.12.7",
            "Pillow>=8.0.0",
            "scikit-learn>=0.24.0",
        ]

setup(
    # Basic package information
    name="blenns_walk_forward",
    version="2.0.0",  # Major version bump for BLENNS Original implementation
    packages=find_packages(exclude=['tests', 'examples', 'docs']),
    
    # Metadata
    description="BLENNS Walk-Forward Trading System - Advanced AI-powered trading prediction with BFC integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Emmanuel A. Adeyemo",
    author_email="emmanuel.adeyemo@example.com",  # Update with actual email
    maintainer="BLENNS Contributors",
    maintainer_email="blenns@example.com",  # Update with actual email
    url="https://github.com/NU-Academics/Blended-Neural-Networks-BLENNs-",
    
    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Optional dependencies for extended functionality
    extras_require={
        'dev': [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
            "pre-commit>=2.15",
        ],
        'docs': [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "numpydoc>=1.1",
        ],
        'notebook': [
            "jupyter>=1.0",
            "ipywidgets>=7.6",
        ],
        'all': [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "sphinx>=4.0",
            "jupyter>=1.0",
        ],
    },
    
    # Package data
    include_package_data=True,
    package_data={
        'blenns_walk_forward': [
            '*.py',
            'data/*.csv',
            'models/*.h5',
            'models/*.weights',
        ],
    },
    
    # Entry points for command-line interface
    entry_points={
        'console_scripts': [
            'blenns-train=blenns_walk_forward.cli:train_model',
            'blenns-predict=blenns_walk_forward.cli:predict_next_day',
            'blenns-backtest=blenns_walk_forward.cli:backtest_strategy',
        ],
    },
    
    # Project URLs for documentation
    project_urls={
        'Bug Reports': 'https://github.com/NU-Academics/Blended-Neural-Networks-BLENNs-/issues',
        'Documentation': 'https://github.com/NU-Academics/Blended-Neural-Networks-BLENNs-/wiki',
        'Source Code': 'https://github.com/NU-Academics/Blended-Neural-Networks-BLENNs-',
        'Original Research': 'https://github.com/NU-Academics/Blended-Neural-Networks-BLENNs-',
    },
    
    # Keywords for PyPI search
    keywords="trading, ai, machine-learning, deep-learning, cnn, lstm, attention, finance, stocks, crypto, forex, bfc, blenns",
    
    # Zip safe flag
    zip_safe=False,
    
    # License
    license="MIT",
)

# Post-install message
if 'install' in sys.argv or 'develop' in sys.argv:
    print("\n" + "="*62)
    print("  BLENNS Walk-Forward Trading System v2.0.0")
    print("="*62)
    print("  Features:")
    print("  • BFC 3-stage filtering (EMA → Heikin-Ashi → Kalman)")
    print("  • CNN + LSTM + Attention hybrid architecture")
    print("  • Walk-forward validation with TimeSeriesSplit")
    print("  • Monte Carlo dropout uncertainty estimation")
    print("  • SHAP model interpretability")
    print("="*62)
    print("\n  Quick start:")
    print("    from blenns_walk_forward import BLENNSWalkForward")
    print("    trader = BLENNSWalkForward(symbol='AAPL')")
    print("    result = trader.predict_next_day()")
    print("\n  For documentation: https://github.com/NU-Academics/Blended-Neural-Networks-BLENNs-")
    print("="*62)
