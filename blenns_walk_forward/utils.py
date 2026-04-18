# -*- coding: utf-8 -*-
"""
================================================================================
BLENNS ORIGINAL — Utility Functions with Complete Trading Pipeline
================================================================================
Author: BLENNS Framework Implementation
Date: April 2026

DESCRIPTION:
    This utility module provides helper functions for the BLENNS trading system,
    including:

    1. Data visualization (candles, training curves, ROC, confusion matrix)
    2. SHAP explainability with pixel-region attribution
    3. Monte Carlo Dropout uncertainty estimation
    4. ATR calculation for risk management
    5. Predicted candle visualization
    6. Expert rule validation (Cohen's Kappa)

REFERENCES:
    - Cohen (1960) - Statistical agreement measures
    - Landis & Koch (1977) - Kappa interpretation
    - Lundberg & Lee (2017) - SHAP values
    - Gal & Ghahramani (2016) - Monte Carlo Dropout
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# SECTION 1: DATA VISUALIZATION
# ============================================================================

def visualize_candles(images, n=4, title_prefix="BFC Candle", figsize=(10, 3)):
    """
    Visualize sample BFC candle images
    
    Args:
        images: Array of candle images (N, H, W, C)
        n: Number of images to display
        title_prefix: Prefix for each subplot title
        figsize: Figure size (width, height)
    """
    n = min(n, len(images))
    fig, axes = plt.subplots(1, n, figsize=figsize)
    fig.patch.set_facecolor("#0a0a0f")
    
    if n == 1:
        axes = [axes]
    
    for i in range(n):
        idx = i * (len(images) // n) if len(images) > n else i
        axes[i].imshow(images[idx])
        axes[i].axis('off')
        axes[i].set_title(f'{title_prefix} #{i+1}', fontsize=9, color='white')
        axes[i].set_facecolor("#0a0a0f")
    
    plt.tight_layout()
    plt.show()


def normalize_data(images, volumes):
    """
    Normalize volume data and reshape images for model input
    
    Args:
        images: Array of candle images (samples, H, W, C)
        volumes: Array of volume values (samples, 1)
    
    Returns:
        Tuple of (X_img, X_vol, vol_scaler)
        - X_img: Reshaped for TimeDistributed (samples, timesteps=1, H, W, C)
        - X_vol: Normalized volume data
        - vol_scaler: Fitted scaler for inverse transformation
    """
    vol_scaler = MinMaxScaler()
    volumes_scaled = vol_scaler.fit_transform(volumes)
    
    # Reshape for TimeDistributed layer: (samples, timesteps=1, H, W, C)
    X_img = images.reshape(-1, 1, images.shape[1], images.shape[2], 3)
    X_vol = volumes_scaled
    
    return X_img, X_vol, vol_scaler


def plot_training_curves(history, fold=None, figsize=(12, 4)):
    """
    Enhanced training metrics visualization with loss, accuracy, and AUC
    
    Args:
        history: Keras training history object
        fold: Fold number (optional)
        figsize: Figure size (width, height)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("#0a0a0f")
    
    # Loss curves
    ax1.set_facecolor("#0a0a0f")
    ax1.plot(history.history['loss'], label='Train Loss', color="#ef4444", linewidth=1.5)
    ax1.plot(history.history['val_loss'], label='Val Loss', color="#f97316", linewidth=1.5)
    ax1.set_title(f'Fold {fold} — Loss Curves' if fold else 'Loss Curves', 
                  fontsize=11, color="white")
    ax1.set_xlabel('Epoch', color="#9ca3af")
    ax1.set_ylabel('Loss', color="#9ca3af")
    ax1.legend(facecolor="#12121a", labelcolor="white")
    ax1.tick_params(colors="#9ca3af")
    for spine in ax1.spines.values(): 
        spine.set_edgecolor("#1f1f35")
    ax1.grid(axis="y", color="#1f1f35", linewidth=0.5, alpha=0.5)
    
    # Accuracy/AUC curves
    ax2.set_facecolor("#0a0a0f")
    ax2.plot(history.history['accuracy'], label='Train Acc', color="#22c55e", linewidth=1.5)
    ax2.plot(history.history['val_accuracy'], label='Val Acc', color="#84cc16", linewidth=1.5)
    if 'auc' in history.history:
        ax2.plot(history.history['auc'], label='Train AUC', color="#3b82f6", linewidth=1.5)
        ax2.plot(history.history['val_auc'], label='Val AUC', color="#6366f1", linewidth=1.5)
    ax2.set_title(f'Fold {fold} — Performance Metrics' if fold else 'Performance Metrics',
                  fontsize=11, color="white")
    ax2.set_xlabel('Epoch', color="#9ca3af")
    ax2.set_ylabel('Score', color="#9ca3af")
    ax2.legend(facecolor="#12121a", labelcolor="white")
    ax2.tick_params(colors="#9ca3af")
    for spine in ax2.spines.values(): 
        spine.set_edgecolor("#1f1f35")
    ax2.grid(axis="y", color="#1f1f35", linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_pred, figsize=(8, 6)):
    """
    Plot ROC curve with AUC score
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        figsize: Figure size (width, height)
    
    Returns:
        Tuple of (fpr, tpr, roc_auc)
    """
    if y_pred is None or y_true is None:
        print("   ⚠️ Cannot plot ROC curve: missing predictions or true labels")
        return None, None, None
    
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#0a0a0f")
        ax.set_facecolor("#0a0a0f")
        
        ax.plot(fpr, tpr, color="#6366f1", lw=2, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title("ROC Curve", fontsize=12, color="white")
        ax.set_xlabel("False Positive Rate", color="#9ca3af")
        ax.set_ylabel("True Positive Rate", color="#9ca3af")
        ax.legend(facecolor="#12121a", labelcolor="white")
        ax.tick_params(colors="#9ca3af")
        for spine in ax.spines.values(): 
            spine.set_edgecolor("#1f1f35")
        ax.grid(axis="both", color="#1f1f35", linewidth=0.5, alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
        return fpr, tpr, roc_auc
    except Exception as e:
        print(f"   ⚠️ Error plotting ROC curve: {e}")
        return None, None, None


def plot_confusion_matrix(cm, figsize=(6, 5)):
    """
    Plot confusion matrix with annotations
    
    Args:
        cm: Confusion matrix array
        figsize: Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0a0a0f")
    
    im = ax.imshow(cm, cmap="Blues", interpolation='nearest')
    plt.colorbar(im, ax=ax)
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm[i, j] > cm.max()/2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", 
                   fontsize=16, fontweight="bold", color=text_color)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Bear", "Predicted Bull"], color="#9ca3af")
    ax.set_yticklabels(["Actual Bear", "Actual Bull"], color="#9ca3af")
    ax.set_title("Confusion Matrix", fontsize=12, color="white")
    ax.tick_params(colors="#9ca3af")
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# SECTION 2: SHAP EXPLAINABILITY WITH PIXEL-REGION ATTRIBUTION
# ============================================================================

def explain_model_with_shap(model, X_img, X_vol, sample_idx=-1, n_samples=50, img_size=64):
    """
    SHAP-style feature importance using perturbation method.
    
    This function maps feature importance to BFC candlestick anatomical regions:
        - Upper Wick: Selling pressure at price highs
        - Lower Wick: Buying support at price lows
        - Bullish Body: Buying momentum
        - Bearish Body: Selling momentum
        - Volume Impact: Trading volume contribution
    
    Args:
        model: Trained Keras model
        X_img: Image inputs (samples, timesteps, H, W, C)
        X_vol: Volume inputs (samples, 1)
        sample_idx: Index of sample to explain (-1 for last)
        n_samples: Number of perturbation samples per feature
        img_size: Image size (height/width)
    
    Returns:
        Dictionary with feature impacts
    """
    print("\n   Computing SHAP feature importance via perturbation...")
    
    # Select sample to explain
    if sample_idx == -1:
        X_img_s = X_img[-1:]
        X_vol_s = X_vol[-1:]
    else:
        X_img_s = X_img[sample_idx:sample_idx+1]
        X_vol_s = X_vol[sample_idx:sample_idx+1]
    
    # Get base prediction
    base_pred = float(model.predict([X_img_s, X_vol_s], verbose=0)[0][0])
    
    # Map image regions to BFC candlestick anatomical features
    # Based on the BFC framework: 64x64 candlestick image divided into regions
    region_defs = {
        'BFC Upper Wick (Sell Pressure)':   (0,            img_size//4,  img_size//3, 2*img_size//3),
        'BFC Bullish Body (Buy Momentum)':  (img_size//4,  img_size//2,  img_size//3, 2*img_size//3),
        'BFC Bearish Body (Sell Momentum)': (img_size//2,  3*img_size//4, img_size//3, 2*img_size//3),
        'BFC Lower Wick (Buy Support)':     (3*img_size//4, img_size,     img_size//3, 2*img_size//3),
    }
    
    impacts = {}
    
    # Image region perturbations
    for feat_name, (r0, r1, c0, c1) in region_defs.items():
        diffs = []
        for _ in range(n_samples):
            X_perturbed = X_img_s.copy()
            # Add noise to specific region
            noise = np.random.uniform(0, 1, X_perturbed[:, :, r0:r1, c0:c1, :].shape)
            X_perturbed[:, :, r0:r1, c0:c1, :] = noise
            p = float(model.predict([X_perturbed, X_vol_s], verbose=0)[0][0])
            diffs.append(abs(base_pred - p))
        impacts[feat_name] = float(np.mean(diffs))
    
    # Volume perturbation
    vol_diffs = []
    for _ in range(n_samples):
        X_vol_perturbed = np.random.uniform(0, 1, X_vol_s.shape).astype(np.float32)
        p = float(model.predict([X_img_s, X_vol_perturbed], verbose=0)[0][0])
        vol_diffs.append(abs(base_pred - p))
    impacts['Volume Impact'] = float(np.mean(vol_diffs))
    
    # Print results
    print("   SHAP Feature Impacts (higher = more important):")
    for k, v in sorted(impacts.items(), key=lambda x: -x[1]):
        print(f"     {k:<35}: {v:.6f}")
    
    # Plot visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0a0a0f")
    
    # Show last BFC candle image (last timestep)
    candle_img = X_img_s[0, -1] if X_img_s.shape[1] > 1 else X_img_s[0, 0]
    ax1.imshow(candle_img)
    ax1.set_title('BFC Processed Candle (Last Observation)', fontsize=10, color='white')
    ax1.axis('off')
    ax1.set_facecolor("#0a0a0f")
    
    # Horizontal bar chart of feature impacts
    sorted_items = sorted(impacts.items(), key=lambda x: x[1])
    colors = ['#6366f1' for _ in sorted_items]
    bars = ax2.barh([item[0] for item in sorted_items], [item[1] for item in sorted_items],
                   color=colors, alpha=0.85)
    ax2.set_title('SHAP Feature Impacts (BFC Anatomical Regions)', color='white', fontsize=10)
    ax2.set_xlabel('Mean |SHAP Value|', color='#9ca3af')
    ax2.axvline(0, color='white', linestyle='--', linewidth=0.8)
    ax2.tick_params(colors='#9ca3af')
    ax2.set_facecolor("#0a0a0f")
    for sp in ax2.spines.values(): 
        sp.set_edgecolor("#1f1f35")
    
    # Add value labels
    for bar, val in zip(bars, [item[1] for item in sorted_items]):
        ax2.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height()/2,
                f'{val:.5f}', va='center', fontsize=9, color='#9ca3af')
    
    plt.tight_layout()
    plt.show()
    
    return impacts


def plot_shap_importance(impacts, title="SHAP Feature Importance", figsize=(10, 5)):
    """
    Plot SHAP feature importance as a horizontal bar chart
    
    Args:
        impacts: Dictionary of feature names and importance values
        title: Plot title
        figsize: Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0a0a0f")
    
    sorted_items = sorted(impacts.items(), key=lambda x: x[1])
    colors = ['#6366f1' for _ in sorted_items]
    
    bars = ax.barh([item[0] for item in sorted_items], [item[1] for item in sorted_items],
                  color=colors, alpha=0.85)
    ax.set_title(title, color='white', fontsize=12)
    ax.set_xlabel('Mean |SHAP Value|', color='#9ca3af')
    ax.axvline(0, color='white', linestyle='--', linewidth=0.8)
    ax.tick_params(colors='#9ca3af')
    ax.set_facecolor("#0a0a0f")
    for sp in ax.spines.values(): 
        sp.set_edgecolor("#1f1f35")
    
    # Add value labels
    for bar, val in zip(bars, [item[1] for item in sorted_items]):
        ax2.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height()/2,
                f'{val:.5f}', va='center', fontsize=9, color='#9ca3af')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# SECTION 3: MONTE CARLO DROPOUT UNCERTAINTY ESTIMATION
# ============================================================================

def monte_carlo_predict(model, X_img, X_vol, n_samples=100, verbose=True):
    """
    Monte Carlo Dropout prediction for uncertainty estimation
    
    Method (Gal & Ghahramani, 2016):
        - Keep dropout active during inference
        - Perform multiple forward passes
        - Mean = prediction, Standard deviation = uncertainty
    
    Args:
        model: Trained Keras model
        X_img: Single image sample
        X_vol: Single volume sample
        n_samples: Number of Monte Carlo samples
        verbose: Print results if True
    
    Returns:
        Dictionary with mean, std, predictions, direction, confidence
    """
    predictions = []
    
    for _ in range(n_samples):
        # Use training=True to keep dropout active
        pred = model([X_img, X_vol], training=True)
        predictions.append(float(pred.numpy()[0][0]))
    
    predictions = np.array(predictions)
    mean_pred = predictions.mean()
    std_pred = predictions.std()
    
    direction = "Bullish" if mean_pred > 0.5 else "Bearish"
    confidence = mean_pred if direction == "Bullish" else 1 - mean_pred
    
    if verbose:
        print(f"\n  Monte Carlo Dropout ({n_samples} passes)...")
        print(f"  Direction  : {direction}")
        print(f"  Confidence : {confidence*100:.2f}%")
        print(f"  Raw Score  : {mean_pred:.4f}")
        print(f"  Uncertainty: ±{std_pred:.4f}")
    
    return {
        'mean': mean_pred,
        'std': std_pred,
        'predictions': predictions,
        'direction': direction,
        'confidence': confidence
    }


def plot_uncertainty_candle(predictions, figsize=(10, 4)):
    """
    Visualize prediction uncertainty as a histogram
    
    Args:
        predictions: Array of Monte Carlo predictions
        figsize: Figure size (width, height)
    """
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    lower_bound = np.percentile(predictions, 16)  # ~1 sigma
    upper_bound = np.percentile(predictions, 84)  # ~1 sigma
    
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0a0a0f")
    
    # Histogram of predictions
    color = "#22c55e" if mean_pred > 0.5 else "#ef4444"
    ax.hist(predictions, bins=30, color=color, alpha=0.75, edgecolor="white")
    ax.axvline(mean_pred, color="white", linewidth=2, linestyle="--", label=f"Mean={mean_pred:.3f}")
    ax.axvline(0.5, color="gray", linewidth=1, linestyle=":", label="Decision Boundary")
    ax.axvline(lower_bound, color="#9ca3af", linewidth=1, linestyle=":", alpha=0.7, 
               label=f"±1σ: {lower_bound:.3f}–{upper_bound:.3f}")
    ax.axvline(upper_bound, color="#9ca3af", linewidth=1, linestyle=":", alpha=0.7)
    
    ax.set_title(f"Monte Carlo Dropout Distribution — {'Bullish' if mean_pred > 0.5 else 'Bearish'} ({abs(mean_pred-0.5)*200:.1f}% confidence)",
                 color="white", fontsize=12)
    ax.set_xlabel("Predicted Probability (Up)", color="#9ca3af")
    ax.set_ylabel("Frequency", color="#9ca3af")
    ax.tick_params(colors="#9ca3af")
    ax.legend(facecolor="#12121a", labelcolor="white")
    for spine in ax.spines.values(): 
        spine.set_edgecolor("#1f1f35")
    ax.grid(axis="y", color="#1f1f35", linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# SECTION 4: ATR (AVERAGE TRUE RANGE) FOR RISK MANAGEMENT
# ============================================================================

def compute_atr(df, period=14):
    """
    Calculate Average True Range for volatility-based position sizing
    
    True Range = max(High-Low, |High-Prev Close|, |Low-Prev Close|)
    ATR = rolling mean of True Range
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period for ATR calculation
    
    Returns:
        ATR value
    """
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    
    tr = []
    for i in range(1, len(df)):
        tr.append(max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1])))
    
    if len(tr) >= period:
        return np.mean(tr[-period:])
    return np.mean(tr) if tr else 0


def compute_atr_vectorized(df, period=14):
    """
    Vectorized ATR calculation for better performance
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period for ATR calculation
    
    Returns:
        ATR series
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).rolling(window=period).mean()
    return atr


def compute_atr_series(df, period=14):
    """
    Calculate full ATR series with proper handling of NaN values
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period for ATR calculation
    
    Returns:
        ATR series as pandas Series
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def atr_multipliers(symbol):
    """
    Get symbol-specific ATR multipliers for stop-loss and take-profit
    
    Different asset classes have different volatility profiles:
        - Crypto: Highest volatility (2.0x, 1.0x)
        - Forex/Commodities: Medium volatility (1.5x, 1.0x)
        - Equities: Standard (1.5x, 1.0x)
    
    Args:
        symbol: Trading symbol (e.g., 'AAPL', 'BTC-USD', 'GC=F')
    
    Returns:
        Tuple of (take_profit_multiplier, stop_loss_multiplier)
    """
    s = symbol.upper()
    
    # Crypto (highest volatility)
    if any(x in s for x in ["BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "DOGE"]):
        return 2.0, 1.0
    
    # Forex (lower volatility)
    if s.endswith("=X") or any(x in s for x in ["USD", "EUR", "GBP", "JPY", "AUD", "CHF", "CAD"]):
        return 1.5, 1.0
    
    # Commodities
    if any(x in s for x in ["GC", "SI", "CL", "NG", "HG", "ZC", "OIL"]):
        return 1.5, 1.0
    
    # Default for stocks and indices
    return 1.5, 1.0


def get_tp_sl_multipliers(symbol):
    """
    Alias for atr_multipliers (backward compatibility)
    """
    return atr_multipliers(symbol)


# ============================================================================
# SECTION 5: PREDICTED CANDLESTICK VISUALIZATION
# ============================================================================

def plot_predicted_candle(historical_data, direction, confidence, atr_value, 
                          symbol="", n_show=15, figsize=(14, 6)):
    """
    Visualize historical candles with predicted next candle
    
    Args:
        historical_data: DataFrame with OHLC data
        direction: 'Bullish' or 'Bearish'
        confidence: Confidence level (0-1)
        atr_value: Average True Range value
        symbol: Trading symbol for display
        n_show: Number of historical candles to show
        figsize: Figure size (width, height)
    """
    tp_mult, sl_mult = atr_multipliers(symbol)
    last_close = historical_data["close"].iloc[-1]
    
    # Calculate predicted candle values
    pred_open = last_close
    if direction == "Bullish":
        pred_close = pred_open + atr_value * tp_mult
    else:
        pred_close = pred_open - atr_value * tp_mult
    
    pred_high = max(pred_open, pred_close) + atr_value * 0.5
    pred_low = min(pred_open, pred_close) - atr_value * 0.5
    
    # Get historical data for visualization
    hist = historical_data.tail(n_show).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0a0a0f")
    
    # Plot historical candles
    for i, row in hist.iterrows():
        bull = row["close"] >= row["open"]
        c = "#22c55e" if bull else "#ef4444"
        body_bot = min(row["open"], row["close"])
        body_h = abs(row["close"] - row["open"])
        if body_h < 1e-6:
            body_h = (row["high"] - row["low"]) * 0.01
        
        ax.bar(i, body_h, bottom=body_bot, width=0.4, color=c, zorder=3)
        ax.plot([i, i], [row["low"], row["high"]], color=c, linewidth=1, zorder=2)
    
    # Plot predicted candle
    pred_x = n_show
    pred_c = "#22c55e" if direction == "Bullish" else "#ef4444"
    body_bot = min(pred_open, pred_close)
    body_h = abs(pred_close - pred_open)
    
    ax.bar(pred_x, body_h, bottom=body_bot, width=0.4, 
           color=pred_c, alpha=0.5, edgecolor=pred_c, linewidth=2, linestyle="--")
    ax.plot([pred_x, pred_x], [pred_low, pred_high], 
            color=pred_c, linewidth=1.5, linestyle="--")
    
    # Add annotation
    ax.annotate(f"  PREDICTED\n  {direction}\n  {confidence*100:.1f}% conf",
                xy=(pred_x, pred_close), xytext=(pred_x + 0.7, pred_close),
                color=pred_c, fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=pred_c))
    
    # Add vertical line separator
    ax.axvline(n_show - 0.5, color="#6366f1", linewidth=1.5, linestyle=":", alpha=0.7, label="Now →")
    
    # Configure axes
    tick_labels = list(hist["date"].astype(str).str[-5:]) + ["Next"]
    ax.set_xticks(range(n_show + 1))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7, color="#9ca3af")
    ax.tick_params(colors="#9ca3af")
    for spine in ax.spines.values(): 
        spine.set_edgecolor("#1f1f35")
    
    ax.set_title(f"{symbol} — Last {n_show} Candles + Predicted ({direction})",
                 color="white", fontsize=12, pad=12)
    ax.legend(facecolor="#12121a", edgecolor="#1f1f35", labelcolor="white")
    ax.grid(axis="y", color="#1f1f35", linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# SECTION 6: EXPERT RULES AND COHEN'S KAPPA (RQ3)
# ============================================================================

def compute_expert_signals(df):
    """
    Generate expert trading signals from 5 technical analysis rules
    
    Expert Rules:
        1. RSI (Wilder, 1978): Buy when RSI < 30, Sell when RSI > 70
        2. MACD (Appel, 2005): Buy when MACD > Signal line
        3. MA Crossover (Murphy, 1999): Buy when SMA20 > SMA50
        4. Volume Confirmation (Karpoff, 1987): Buy when volume > 1.5×MA20 & price up
        5. Support/Resistance (Edwards & Magee, 1948): Buy near support, Sell near resistance
    
    Args:
        df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
    
    Returns:
        expert_signal: Array of expert consensus signals (0=Down, 1=Up)
        df: DataFrame with intermediate calculations
    """
    d = df.copy()
    
    # --- Rule 1: RSI (Relative Strength Index) ---
    delta = d['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    d['RSI'] = 100 - (100 / (1 + rs))
    rsi_sig = ((d['RSI'] < 30).astype(int) - (d['RSI'] > 70).astype(int)).clip(0, 1)
    
    # --- Rule 2: MACD (Moving Average Convergence Divergence) ---
    d['MACD'] = d['close'].ewm(span=12).mean() - d['close'].ewm(span=26).mean()
    d['MACD_Signal'] = d['MACD'].ewm(span=9).mean()
    macd_sig = (d['MACD'] > d['MACD_Signal']).astype(int)
    
    # --- Rule 3: Moving Average Crossover ---
    d['SMA20'] = d['close'].rolling(20).mean()
    d['SMA50'] = d['close'].rolling(50).mean()
    ma_sig = (d['SMA20'] > d['SMA50']).astype(int)
    
    # --- Rule 4: Volume Confirmation ---
    d['VMA20'] = d['volume'].rolling(20).mean()
    d['Return'] = d['close'].pct_change()
    vol_sig = ((d['volume'] > 1.5 * d['VMA20']) & (d['Return'] > 0)).astype(int)
    
    # --- Rule 5: Support/Resistance ---
    d['Resistance'] = d['high'].rolling(20).max()
    d['Support'] = d['low'].rolling(20).min()
    sr_sig = pd.Series(0, index=d.index)
    sr_sig[d['close'] <= d['Support'] * 1.02] = 1
    sr_sig[d['close'] >= d['Resistance'] * 0.98] = 0
    
    # --- Aggregate: Simple Majority Vote ---
    consensus = (rsi_sig + macd_sig + ma_sig + vol_sig + sr_sig) / 5.0
    d = d.dropna().reset_index(drop=True)
    
    expert_signal = (consensus > 0.5).astype(int)
    
    return expert_signal, d


def calculate_cohens_kappa(y_true, y_pred):
    """
    Calculate Cohen's Kappa with full statistical testing
    
    Formula:
        p_o = (TP + TN) / N              (Observed agreement)
        p_e = Σ(row_i × col_i) / N²      (Expected agreement by chance)
        κ = (p_o - p_e) / (1 - p_e)      (Kappa coefficient)
        SE = √[p_o(1-p_o) / (N(1-p_e)²)] (Standard error)
        z = κ / SE                        (Z-statistic)
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
    
    Returns:
        Dictionary with kappa, p_o, p_e, standard error, z-statistic, p-value
    """
    cm = confusion_matrix(y_true, y_pred)
    n = np.sum(cm)
    p_o = np.trace(cm) / n
    row_sums = np.sum(cm, axis=1)
    col_sums = np.sum(cm, axis=0)
    p_e = np.sum(row_sums * col_sums) / (n * n)
    
    denom = 1 - p_e
    kappa = (p_o - p_e) / denom if denom > 0 else 0.0
    se_kappa = np.sqrt(p_o * (1 - p_o) / (n * denom ** 2)) if denom > 0 else 0.0
    z_stat = kappa / se_kappa if se_kappa > 0 else 0.0
    p_value = 1 - norm.cdf(z_stat)  # One-tailed test
    
    return {
        'kappa': kappa,
        'p_o': p_o,
        'p_e': p_e,
        'se_kappa': se_kappa,
        'z_stat': z_stat,
        'p_value': p_value,
        'n': n,
        'confusion_matrix': cm
    }


def interpret_kappa(kappa):
    """
    Interpret Cohen's Kappa using Landis & Koch (1977) scale
    
    Args:
        kappa: Cohen's Kappa coefficient
    
    Returns:
        String interpretation
    """
    if kappa < 0:
        return "Poor (worse than chance)"
    elif kappa < 0.01:
        return "No agreement"
    elif kappa < 0.21:
        return "Slight agreement"
    elif kappa < 0.41:
        return "Fair agreement"
    elif kappa < 0.61:
        return "Moderate agreement"
    elif kappa < 0.81:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"


def plot_kappa_agreement(kappa_results, random_kappa_results=None, figsize=(12, 5)):
    """
    Plot Cohen's Kappa agreement metrics and statistical significance
    
    Args:
        kappa_results: Results from calculate_cohens_kappa()
        random_kappa_results: Results for random baseline (optional)
        figsize: Figure size (width, height)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("#0a0a0f")
    
    # Bar chart: Agreement metrics
    ax1 = axes[0]
    ax1.set_facecolor("#0a0a0f")
    
    metrics = ['Observed\nAgreement', 'Expected\nAgreement', "Cohen's\nKappa"]
    shap_values = [kappa_results['p_o'], kappa_results['p_e'], kappa_results['kappa']]
    
    if random_kappa_results:
        random_values = [random_kappa_results['p_o'], random_kappa_results['p_e'], random_kappa_results['kappa']]
        x = np.arange(len(metrics))
        width = 0.35
        ax1.bar(x - width/2, shap_values, width, label='SHAP-Expert', color='#6366f1', alpha=0.85)
        ax1.bar(x + width/2, random_values, width, label='Random-Expert', color='#64748b', alpha=0.85)
        ax1.legend(facecolor='#12121a', labelcolor='white')
    else:
        ax1.bar(metrics, shap_values, color='#6366f1', alpha=0.85)
    
    ax1.set_ylabel('Value', color='#9ca3af')
    ax1.set_title('Agreement Metrics', color='white')
    ax1.tick_params(colors='#9ca3af')
    for spine in ax1.spines.values(): 
        spine.set_edgecolor("#1f1f35")
    ax1.grid(axis='y', color='#1f1f35', linewidth=0.5, alpha=0.5)
    
    # Kappa gauge
    ax2 = axes[1]
    ax2.set_facecolor("#0a0a0f")
    
    kv = kappa_results['kappa']
    ax2.barh([0], [1], color='#1f1f35', height=0.4)
    ax2.barh([0], [max(0, kv)], color='#6366f1', height=0.4, alpha=0.85)
    ax2.plot([kv, kv], [-0.25, 0.25], 'w-', lw=3)
    ax2.plot(kv, 0, 'wo', ms=10)
    
    # Add interpretation thresholds
    thresholds = [(0.01, 'Slight', '#f97316'), (0.21, 'Fair', '#eab308'),
                  (0.41, 'Moderate', '#84cc16'), (0.61, 'Substantial', '#22c55e'),
                  (0.81, 'Almost Perfect', '#16a34a')]
    for xv, lbl, col in thresholds:
        ax2.axvline(xv, color=col, ls=':', alpha=0.8)
    
    ax2.set_xlim(0, 1)
    ax2.set_yticks([])
    ax2.set_xlabel("Cohen's Kappa", color='#9ca3af')
    ax2.set_title(f"κ = {kv:.4f} — {interpret_kappa(kv)}", color='white')
    ax2.tick_params(colors='#9ca3af')
    for spine in ax2.spines.values(): 
        spine.set_edgecolor("#1f1f35")
    
    plt.suptitle('SHAP vs Expert Rules — Agreement Analysis', color='white', fontsize=12)
    plt.tight_layout()
    plt.show()
