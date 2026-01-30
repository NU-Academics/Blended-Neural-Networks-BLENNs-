"""
BLENNS Trading System - Utility Functions
"""

import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_curve, auc, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def visualize_candles(images, n=5, figsize=(15, 3), title="BFC Processed Candles"):
    """
    Visualize sample candle images
    
    Parameters:
    -----------
    images : numpy.ndarray
        Array of candle images
    n : int
        Number of images to display
    figsize : tuple
        Figure size
    title : str
        Plot title
    """
    n = min(n, len(images))
    
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    
    for i in range(n):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        axes[i].set_title(f'Candle {i+1}')
    
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

def plot_training_curves(history, fold=1):
    """
    Plot training and validation metrics
    
    Parameters:
    -----------
    history : tf.keras.callbacks.History
        Training history
    fold : int
        Fold number for title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_title(f'Fold {fold} - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Acc', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
    axes[1].set_title(f'Fold {fold} - Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # AUC
    axes[2].plot(history.history['auc'], label='Train AUC', linewidth=2)
    axes[2].plot(history.history['val_auc'], label='Val AUC', linewidth=2)
    axes[2].set_title(f'Fold {fold} - AUC')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def explain_model_with_shap(model, X_img, X_vol, sample_idx=0, class_names=['Sell', 'Buy']):
    """
    Explain model predictions using SHAP
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained model
    X_img : numpy.ndarray
        Image data
    X_vol : numpy.ndarray
        Volume data
    sample_idx : int
        Index of sample to explain
    class_names : list
        Class names for interpretation
    """
    # Select sample
    sample_img = X_img[sample_idx:sample_idx+1]
    sample_vol = X_vol[sample_idx:sample_idx+1]
    
    # Create background dataset
    background_img = X_img[:50]
    background_vol = X_vol[:50]
    
    try:
        # Create explainer
        explainer = shap.GradientExplainer(model, [background_img, background_vol])
        
        # Calculate SHAP values
        shap_values = explainer.shap_values([sample_img, sample_vol])
        
        # Get prediction
        prediction = model.predict([sample_img, sample_vol], verbose=0)[0][0]
        predicted_class = class_names[int(prediction > 0.5)]
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(sample_img[0][0])
        axes[0].set_title(f'Original Candle\nPrediction: {predicted_class} ({prediction:.3f})')
        axes[0].axis('off')
        
        # SHAP values for image
        if isinstance(shap_values, list):
            img_shap = shap_values[0][0][0]
        else:
            img_shap = shap_values[0][0]
        
        axes[1].imshow(np.mean(np.abs(img_shap), axis=2), cmap='hot')
        axes[1].set_title('SHAP Values (Image)')
        axes[1].axis('off')
        
        # Feature importance
        feature_names = ['Upper Wick', 'Lower Wick', 'Bullish Body', 'Bearish Body', 'Volume']
        
        if isinstance(shap_values, list):
            vol_shap = shap_values[1][0][0] if len(shap_values) > 1 else 0
        else:
            vol_shap = shap_values[1][0] if shap_values.shape[0] > 1 else 0
        
        feature_importances = [
            np.mean(np.abs(img_shap[0:15, 25:40, 1])),  # Green upper wick
            np.mean(np.abs(img_shap[50:64, 25:40, 0])),  # Red lower wick
            np.mean(np.abs(img_shap[25:40, 25:40, 1])),  # Green body
            np.mean(np.abs(img_shap[25:40, 25:40, 0])),  # Red body
            np.abs(vol_shap) if isinstance(vol_shap, (int, float)) else np.abs(vol_shap[0])
        ]
        
        colors = ['green' if imp > 0 else 'red' for imp in feature_importances]
        axes[2].barh(feature_names, feature_importances, color=colors)
        axes[2].set_title('Feature Importance')
        axes[2].set_xlabel('Mean |SHAP|')
        axes[2].axvline(0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"⚠️ SHAP explanation failed: {e}")
        print("Using simpler visualization...")
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].imshow(sample_img[0][0])
        ax[0].set_title(f'Original Candle\nPrediction: {predicted_class} ({prediction:.3f})')
        ax[0].axis('off')
        ax[1].text(0.5, 0.5, f'SHAP not available\nPrediction: {prediction:.3f}',
                  ha='center', va='center', fontsize=12)
        ax[1].axis('off')
        plt.show()

def plot_roc_curve(y_true, y_pred, title="ROC Curve"):
    """
    Plot ROC curve with AUC score
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted probabilities
    title : str
        Plot title
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    plt.fill_between(fpr, tpr, alpha=0.3, color='darkorange')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Add threshold points
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10, 
             label=f'Optimal threshold: {optimal_threshold:.3f}')
    plt.legend(loc="lower right")
    
    plt.show()
    
    print(f"📊 AUC Score: {roc_auc:.3f}")
    print(f"🎯 Optimal Threshold: {optimal_threshold:.3f}")
    print(f"   TPR at optimal: {tpr[optimal_idx]:.3f}")
    print(f"   FPR at optimal: {fpr[optimal_idx]:.3f}")

def monte_carlo_predict(model, X_img, X_vol, n_samples=100, dropout=True):
    """
    Monte Carlo Dropout for uncertainty estimation
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained model
    X_img : numpy.ndarray
        Image data
    X_vol : numpy.ndarray
        Volume data
    n_samples : int
        Number of Monte Carlo samples
    dropout : bool
        Whether to use dropout at inference
        
    Returns:
    --------
    numpy.ndarray: Array of predictions
    """
    predictions = []
    
    print(f"🎲 Running Monte Carlo prediction with {n_samples} samples...")
    
    for i in range(n_samples):
        if dropout:
            # Enable dropout
            predictions.append(model([X_img, X_vol], training=True).numpy())
        else:
            predictions.append(model.predict([X_img, X_vol], verbose=0))
        
        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{n_samples} samples...")
    
    predictions = np.array(predictions).squeeze()
    
    print(f"✅ Monte Carlo complete:")
    print(f"   Mean prediction: {np.mean(predictions):.3f}")
    print(f"   Std prediction: {np.std(predictions):.3f}")
    print(f"   95% CI: [{np.percentile(predictions, 2.5):.3f}, {np.percentile(predictions, 97.5):.3f}]")
    
    return predictions

def plot_uncertainty_candle(predictions, current_price=100, figsize=(10, 6)):
    """
    Visualize prediction uncertainty as a candle
    
    Parameters:
    -----------
    predictions : numpy.ndarray
        Array of Monte Carlo predictions
    current_price : float
        Current price for reference
    figsize : tuple
        Figure size
    """
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    conf_95 = np.percentile(predictions, [2.5, 97.5])
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Uncertainty candle
    ax1 = axes[0]
    color = 'green' if mean_pred > 0.5 else 'red'
    
    # Create uncertainty candle
    body_height = abs(mean_pred - 0.5) * 2
    upper_wick = (1 - conf_95[1]) * 0.5
    lower_wick = (conf_95[0]) * 0.5
    
    # Plot candle
    ax1.plot([0.5, 0.5], [0.5 - lower_wick, 0.5 + upper_wick], 
             color=color, linewidth=3, alpha=0.7)
    ax1.add_patch(plt.Rectangle((0.4, 0.5 - body_height/2), 0.2, body_height,
                               color=color, alpha=0.5))
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title(f'Prediction Uncertainty\nMean: {mean_pred:.3f} ± {std_pred:.3f}')
    
    # Distribution plot
    ax2 = axes[1]
    ax2.hist(predictions, bins=30, alpha=0.7, color=color, edgecolor='black')
    ax2.axvline(mean_pred, color='black', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_pred:.3f}')
    ax2.axvline(0.5, color='gray', linestyle=':', linewidth=2, 
                label='Decision boundary')
    ax2.axvspan(conf_95[0], conf_95[1], alpha=0.2, color=color, 
                label=f'95% CI: [{conf_95[0]:.3f}, {conf_95[1]:.3f}]')
    
    ax2.set_xlabel('Prediction Probability', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Prediction Distribution', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"📊 Prediction Statistics:")
    print(f"   Mean: {mean_pred:.4f}")
    print(f"   Std: {std_pred:.4f}")
    print(f"   CV (Std/Mean): {std_pred/mean_pred if mean_pred != 0 else np.inf:.4f}")
    print(f"   95% Confidence Interval: [{conf_95[0]:.4f}, {conf_95[1]:.4f}]")
    print(f"   Buy Probability: {np.mean(predictions > 0.5):.1%}")
