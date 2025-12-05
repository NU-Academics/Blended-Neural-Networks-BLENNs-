"""
BLENNS Trading System - Utility Functions with BFC Integration
"""

import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler

def visualize_candles(images, n=2, figsize=(4, 2)):
    """Visualize sample candle images with BFC processing"""
    n = min(n, len(images))
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for i in range(n):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        axes[i].set_title(f'BFC Candle {i+1}')
    plt.tight_layout()
    plt.show()

def normalize_data(images, volumes):
    """Normalize volume data and reshape images"""
    vol_scaler = MinMaxScaler()
    volumes_scaled = vol_scaler.fit_transform(volumes)
    return images.reshape(-1, 1, images.shape[1], images.shape[2], 3), volumes_scaled

def plot_training_curves(history, fold):
    """Enhanced training metrics visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curves
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title(f'Fold {fold} - Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Accuracy/AUC curves
    ax2.plot(history.history['accuracy'], label='Train Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.plot(history.history['auc'], label='Train AUC')
    ax2.plot(history.history['val_auc'], label='Validation AUC')
    ax2.set_title(f'Fold {fold} - Performance Metrics')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def explain_model_with_shap(model, X_img, X_vol, sample_idx=0):
    """Enhanced SHAP explanation with BFC-specific features"""
    # Create explainer
    explainer = shap.GradientExplainer(model, [X_img[:1], X_vol[:1]])

    # Calculate SHAP values
    shap_values = explainer.shap_values([X_img[sample_idx:sample_idx+1], X_vol[sample_idx:sample_idx+1]])

    # Process SHAP values
    img_shap = shap_values[0][0][0]  # (H, W, C)
    vol_shap = shap_values[1][0][0]

    # Feature impact analysis
    impact_features = {
        'BFC Upper Wick': np.mean(img_shap[0:15, 25:40, 1]),  # Green channel
        'BFC Lower Wick': np.mean(img_shap[50:64, 25:40, 0]),  # Red channel
        'BFC Bullish Body': np.mean(img_shap[25:40, 25:40, 1]),
        'BFC Bearish Body': np.mean(img_shap[25:40, 25:40, 0]),
        'Volume Impact': float(vol_shap)
    }

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(X_img[sample_idx][0])
    ax1.set_title('BFC Processed Candle')
    ax1.axis('off')

    ax2.barh(list(impact_features.keys()), list(impact_features.values()))
    ax2.set_title('SHAP Feature Impacts')
    ax2.axvline(0, color='black', linestyle='--')
    plt.tight_layout()
    plt.show()

def monte_carlo_predict(model, X_img, X_vol, n_samples=100):
    """Monte Carlo Dropout uncertainty estimation"""
    predictions = []
    for _ in range(n_samples):
        predictions.append(model.predict([X_img, X_vol], verbose=0))
    return np.array(predictions)

def plot_uncertainty_candle(predictions):
    """Visualize prediction uncertainty"""
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)

    fig, ax = plt.subplots(figsize=(3, 3))
    color = 'g' if mean_pred > 0.5 else 'r'
    body_height = abs(mean_pred - 0.5) * 2
    upper_shadow = (1 - (mean_pred + std_pred)) * 0.2
    lower_shadow = (mean_pred - std_pred) * 0.2

    # Draw candle
    ax.plot([0.5, 0.5], [0.5 - lower_shadow, 0.5 + upper_shadow], color=color, linewidth=2)
    ax.add_patch(plt.Rectangle((0.4, 0.5 - body_height/2), 0.2, body_height,
                              color=color, alpha=0.6))

    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title(f"Prediction: {mean_pred:.2f} ± {std_pred:.2f}")
    plt.show()

def plot_roc_curve(y_true, y_pred):
    """Plot ROC curve with AUC"""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
