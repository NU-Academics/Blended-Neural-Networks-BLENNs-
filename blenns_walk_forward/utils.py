# -*- coding: utf-8 -*-
"""
BLENNS Trading System - Utility Functions with BFC Integration
Complete implementation from BLENNS Original (2010-Present)

Provides:
- Candlestick visualization with BFC filtering
- Training curve plotting with loss and metrics
- SHAP model interpretability for BFC candle regions
- Monte Carlo dropout uncertainty estimation
- ROC curve and confusion matrix visualization
- ATR calculation for risk management
- Predicted candle visualization with confidence
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


def visualize_candles(images, n=4, title_prefix="BFC Candle", figsize=(10, 3)):
    """
    Visualize sample BFC candle images
    
    Args:
        images: Array of candle images
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
        fold: Fold number (optional, for title)
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
        y_true: True labels
        y_pred: Predicted probabilities
        figsize: Figure size (width, height)
    
    Returns:
        Tuple of (fpr, tpr, roc_auc)
    """
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


def plot_confusion_matrix(cm, figsize=(6, 5)):
    """
    Plot confusion matrix with annotations
    
    Args:
        cm: Confusion matrix array (2x2)
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
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", 
                   fontsize=16, fontweight="bold", color="white" if cm[i, j] > cm.max()/2 else "black")
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Bear", "Predicted Bull"], color="#9ca3af")
    ax.set_yticklabels(["Actual Bear", "Actual Bull"], color="#9ca3af")
    ax.set_title("Confusion Matrix", fontsize=12, color="white")
    ax.tick_params(colors="#9ca3af")
    
    plt.tight_layout()
    plt.show()


def explain_model_with_shap(model, X_img, X_vol, sample_idx=-1):
    """
    Enhanced SHAP explanation with BFC-specific feature analysis
    
    Args:
        model: Trained BLENNS model
        X_img: Image inputs (samples, timesteps, H, W, C)
        X_vol: Volume inputs (samples, 1)
        sample_idx: Index of sample to explain (default: last sample)
    
    Returns:
        Dictionary with feature impacts and SHAP values
    """
    print("\nComputing SHAP feature importance via GradientExplainer...")
    
    # Use last 50 samples as background for SHAP
    n_background = min(50, len(X_img))
    background = [X_img[-n_background:], X_vol[-n_background:]]
    sample = [X_img[sample_idx:sample_idx+1], X_vol[sample_idx:sample_idx+1]]
    
    # Create explainer and calculate SHAP values
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(sample)
    
    # Extract SHAP values
    img_shap = shap_values[0][0][0]   # shape: (H, W, C)
    vol_shap = float(shap_values[1][0][0])
    
    # Map pixel regions to BFC candle features
    # Regions based on typical candlestick anatomy
    impact_features = {
        'BFC Upper Wick':  np.mean(np.abs(img_shap[0:15,  25:40, 1])),   # Green channel
        'BFC Lower Wick':  np.mean(np.abs(img_shap[50:64, 25:40, 0])),   # Red channel
        'BFC Bullish Body': np.mean(np.abs(img_shap[25:40, 25:40, 1])),  # Green body
        'BFC Bearish Body': np.mean(np.abs(img_shap[25:40, 25:40, 0])),  # Red body
        'Volume Impact':   abs(vol_shap)
    }
    
    # Print feature impacts
    print("  SHAP Feature Impacts:")
    for k, v in sorted(impact_features.items(), key=lambda x: -x[1]):
        print(f"    {k:<22}: {v:.6f}")
    
    # Visualization 1: Candle with SHAP overlay
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0a0a0f")
    
    ax1.imshow(X_img[sample_idx][0])
    ax1.set_title('BFC Processed Candle (last)', fontsize=10, color="white")
    ax1.axis('off')
    ax1.set_facecolor("#0a0a0f")
    
    sorted_feats = dict(sorted(impact_features.items(), key=lambda x: x[1]))
    colors = ["#6366f1" if v >= 0 else "#ef4444" for v in sorted_feats.values()]
    ax2.barh(list(sorted_feats.keys()), list(sorted_feats.values()), color=colors, alpha=0.85)
    ax2.set_title('SHAP Feature Impacts (BFC Regions)', fontsize=10, color="white")
    ax2.axvline(0, color='white', linestyle='--', linewidth=0.8)
    ax2.set_facecolor("#0a0a0f")
    ax2.tick_params(colors="#9ca3af")
    for spine in ax2.spines.values(): 
        spine.set_edgecolor("#1f1f35")
    
    plt.tight_layout()
    plt.show()
    
    # Visualization 2: Pixel importance heatmap
    shap_img_display = np.mean(np.abs(img_shap), axis=-1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor("#0a0a0f")
    
    axes[0].imshow(X_img[sample_idx][0])
    axes[0].set_title('BFC Candle (RGB)', fontsize=10, color="white")
    axes[0].axis('off')
    axes[0].set_facecolor("#0a0a0f")
    
    im = axes[1].imshow(shap_img_display, cmap='hot')
    axes[1].set_title('SHAP Pixel Importance (|mean|)', fontsize=10, color="white")
    axes[1].axis('off')
    axes[1].set_facecolor("#0a0a0f")
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.show()
    
    return {
        'img_shap': img_shap,
        'vol_shap': vol_shap,
        'feature_impacts': impact_features
    }


def monte_carlo_predict(model, X_img, X_vol, n_samples=100, verbose=True):
    """
    Monte Carlo Dropout uncertainty estimation
    
    Args:
        model: Trained BLENNS model
        X_img: Single image sample (1, timesteps, H, W, C)
        X_vol: Single volume sample (1, 1)
        n_samples: Number of Monte Carlo samples
        verbose: Print prediction summary
    
    Returns:
        Dictionary with mean, std, all predictions, direction, and confidence
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
    Visualize prediction uncertainty as a candlestick chart
    
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
    ax.axvline(lower_bound, color="#9ca3af", linewidth=1, linestyle=":", alpha=0.7, label=f"±1σ: {lower_bound:.3f}–{upper_bound:.3f}")
    ax.axvline(upper_bound, color="#9ca3af", linewidth=1, linestyle=":", alpha=0.7)
    
    ax.set_title(f"Monte Carlo — {'Bullish' if mean_pred > 0.5 else 'Bearish'} ({abs(mean_pred-0.5)*200:.1f}% confidence)",
                 color="white", fontsize=12)
    ax.set_xlabel("Prediction Score", color="#9ca3af")
    ax.set_ylabel("Frequency", color="#9ca3af")
    ax.tick_params(colors="#9ca3af")
    ax.legend(facecolor="#12121a", labelcolor="white")
    for spine in ax.spines.values(): 
        spine.set_edgecolor("#1f1f35")
    ax.grid(axis="y", color="#1f1f35", linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    plt.show()


def compute_atr(df, period=14):
    """
    Calculate Average True Range for risk management
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR calculation period
    
    Returns:
        ATR value
    """
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    
    tr = []
    for i in range(1, len(df)):
        tr.append(max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1])))
    
    return np.mean(tr[-period:])


def atr_multipliers(symbol):
    """
    Get symbol-specific ATR multipliers for stop-loss and take-profit
    
    Args:
        symbol: Trading symbol (Yahoo Finance format)
    
    Returns:
        Tuple of (take_profit_multiplier, stop_loss_multiplier)
    """
    s = symbol.upper()
    
    # Crypto (higher volatility)
    if any(x in s for x in ["BTC", "ETH", "SOL", "DOGE", "ADA"]):
        return 2.0, 1.0
    
    # Forex (lower volatility)
    if s.endswith("=X") or any(x in s for x in ["USD", "EUR", "GBP", "JPY", "CHF"]):
        return 1.5, 1.0
    
    # Commodities (GC=F gold, CL=F oil)
    if any(x in s for x in ["GC", "OIL", "CL", "SI", "HG"]):
        return 1.5, 1.0
    
    # Default for stocks and indices
    return 1.5, 1.0


def plot_predicted_candle(historical_data, direction, confidence, atr_value, 
                          symbol="", n_show=15, figsize=(14, 6)):
    """
    Visualize historical candles with predicted next candle
    
    Args:
        historical_data: DataFrame with raw price data
        direction: 'Bullish' or 'Bearish'
        confidence: Confidence level (0-1)
        atr_value: ATR value for candle sizing
        symbol: Trading symbol for title
        n_show: Number of historical candles to show
        figsize: Figure size (width, height)
    """
    last_close = historical_data["close"].iloc[-1]
    tp_mult, sl_mult = atr_multipliers(symbol)
    
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
