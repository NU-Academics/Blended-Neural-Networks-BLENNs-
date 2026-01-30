from setuptools import setup, find_packages

setup(
    name="blenns_walk_forward",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "yfinance>=0.2.0",
        "matplotlib>=3.5.0",
        "shap>=0.41.0",
        "tensorflow>=2.10.0",
        "mplfinance>=0.12.0",
        "Pillow>=9.0.0",
        "scikit-learn>=1.0.0",
    ],
    description="BLENNS Walk-Forward Trading System with BFC Integration",
    author="Emmanuel A. Adeyemo",
    python_requires=">=3.8",
    keywords="trading, neural-networks, finance, ai, machine-learning",
    url="https://github.com/NU-Academics/Blended-Neural-Networks-BLENNs-",
)
