from setuptools import setup, find_packages

setup(
    name="blenns_walk_forward",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "yfinance",
        "matplotlib",
        "shap",
        "tensorflow",
        "mplfinance",
        "Pillow",
        "scikit-learn",
    ],
    description="BLENNS Walk-Forward Trading System",
    author="Emmanuel A.Adeyemo",
)