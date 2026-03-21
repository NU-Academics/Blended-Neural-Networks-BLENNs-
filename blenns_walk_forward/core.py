# -*- coding: utf-8 -*-
"""
BLENNS Trading System - Core Module with BFC Integration
Complete implementation from BLENNS Original (2010-Present)

Features:
- BFC 3-stage filtering (EMA → Heikin-Ashi → Kalman)
- CNN + LSTM + Attention hybrid architecture
- Walk-forward validation with TimeSeriesSplit
- Monte Carlo dropout uncertainty estimation
- SHAP model interpretability
"""

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dropout,
    LSTM, Dense, TimeDistributed, concatenate, Attention
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings("ignore")


class BLENNSWalkForward:
    """
    BLENNS Walk-Forward Trading System with Universal BFC Integration
    
    Supports multiple asset classes:
    - Stocks: AAPL, TLRY, etc.
    - Indices: ^SPX, ^NDX
    - Crypto: BTC-USD
    - Forex: EURUSD=X
    - Futures: GC=F
    
    Key Features:
    - 3-stage BFC noise reduction
    - CNN + LSTM + Attention architecture
    - Walk-forward validation
    - Uncertainty estimation via Monte Carlo dropout
    """
    
    def __init__(self, symbol="AAPL", bfc_params=None):
        """
        Initialize BLENNS trading system
        
        Args:
            symbol: Trading symbol (Yahoo Finance format)
            bfc_params: Dictionary with BFC parameters (alpha, R, Q)
        """
        self.symbol = symbol
        self.model = None
        self.scaler = MinMaxScaler()
        self.vol_scaler = MinMaxScaler()
        self.bfc_params = bfc_params or {'alpha': 0.2, 'R': 0.01, 'Q': 1e-5}
        self.last_data = None
        self.images = None
        self.volumes = None
        self.targets = None
        self.metrics = {'fold_accs': [], 'fold_aucs': []}
        
    def exponential_moving_average(self, data, alpha):
        """
        Vectorized Exponential Moving Average
        
        Args:
            data: Input array
            alpha: Smoothing factor (0 < alpha < 1)
        
        Returns:
            EMA filtered array
        """
        result = np.zeros_like(data, dtype=np.float64)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    def kalman_filter(self, observations, R=0.01, Q=1e-5):
        """
        Kalman Filter with configurable noise parameters
        
        Args:
            observations: Input time series
            R: Measurement noise covariance
            Q: Process noise covariance
        
        Returns:
            Filtered time series
        """
        n = len(observations)
        filtered = np.zeros(n, dtype=np.float64)
        P = np.zeros(n, dtype=np.float64)
        
        filtered[0] = observations[0]
        P[0] = 1.0
        
        for i in range(1, n):
            # Prediction step
            filtered[i] = filtered[i-1]
            P[i] = P[i-1] + Q
            
            # Update step
            K_gain = P[i] / (P[i] + R)  # Kalman Gain
            filtered[i] += K_gain * (observations[i] - filtered[i])
            P[i] = (1 - K_gain) * P[i]
            
        return filtered
    
    def compute_bfc(self, df, alpha=0.2, R=0.01, Q=1e-5):
        """
        Enhanced Blenns Filter Candles: EMA → Heikin-Ashi → Kalman
        
        Args:
            df: DataFrame with 'open', 'high', 'low', 'close' columns
            alpha: EMA smoothing factor
            R: Kalman measurement noise
            Q: Kalman process noise
        
        Returns:
            DataFrame with BFC-filtered OHLC data
        """
        df = df.copy()
        
        # Stage 1: EMA Smoothing
        o_ema = self.exponential_moving_average(df['open'].values, alpha)
        h_ema = self.exponential_moving_average(df['high'].values, alpha)
        l_ema = self.exponential_moving_average(df['low'].values, alpha)
        c_ema = self.exponential_moving_average(df['close'].values, alpha)
        
        # Stage 2: Modified Heikin-Ashi
        ha_close = (o_ema + h_ema + l_ema + c_ema) / 4
        ha_open = np.zeros(len(df), dtype=np.float64)
        ha_open[0] = (o_ema[0] + c_ema[0]) / 2
        
        for i in range(1, len(df)):
            ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
            
        ha_high = np.maximum.reduce([h_ema, ha_open, ha_close])
        ha_low = np.minimum.reduce([l_ema, ha_open, ha_close])
        
        # Stage 3: Kalman Filter
        bfc_close = self.kalman_filter(ha_close, R=R, Q=Q)
        bfc_open = self.kalman_filter(ha_open, R=R, Q=Q)
        bfc_high = np.maximum.reduce([h_ema, bfc_open, bfc_close])
        bfc_low = np.minimum.reduce([l_ema, bfc_open, bfc_close])
        
        # Create BFC DataFrame
        bfc = df.copy()
        bfc['open'] = bfc_open
        bfc['high'] = bfc_high
        bfc['low'] = bfc_low
        bfc['close'] = bfc_close
        
        return bfc
    
    def get_data(self, start_date="2010-01-01", end_date=None, interval="1d"):
        """
        Fetch and process historical data with universal BFC application
        
        Args:
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (default: today)
            interval: Data interval (1d, 1h, etc.)
        
        Returns:
            DataFrame with processed price data
        """
        if end_date is None:
            end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
        
        try:
            # Download data
            data = yf.download(
                tickers=self.symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                progress=False
            ).reset_index()
            
            # Handle MultiIndex if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Standardize column names
            column_map = {
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            data = data.rename(columns={k: v for k, v in column_map.items() 
                                        if k in data.columns})
            
            # Validate required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            
            # Apply BFC filtering
            bfc_data = self.compute_bfc(data, **self.bfc_params)
            
            # Store for later use
            self.last_data = bfc_data
            self.raw_data = data
            
            print(f"    Fetched {len(data)} rows | Range: {data['date'].min().date()} → {data['date'].max().date()}")
            print(f"    Last raw close: {data['close'].iloc[-1]:.5f}")
            print(f"    BFC applied. Last BFC close: {bfc_data['close'].iloc[-1]:.6f}")
            
            return bfc_data
            
        except Exception as e:
            print(f"Data loading failed: {str(e)}")
            if 'data' in locals():
                print("Columns received:", data.columns.tolist())
            raise
    
    def create_target(self, data, lookahead=1):
        """
        Create prediction target with configurable lookahead
        
        Args:
            data: DataFrame with BFC-filtered data
            lookahead: Number of days to look ahead for target
        
        Returns:
            DataFrame with 'target' column (1 if next close > current close)
        """
        data = data.copy()
        data['target'] = (data['close'].shift(-lookahead) > data['close']).astype(int)
        data = data.dropna(subset=['target']).reset_index(drop=True)
        
        print(f"    Bullish targets: {data['target'].sum()} / {len(data)} ({data['target'].mean()*100:.1f}%)")
        
        return data
    
    def encode_candles(self, data, window_size=5, img_size=64, dpi=32):
        """
        Generate candlestick images from BFC-processed data
        
        Args:
            data: DataFrame with BFC-filtered OHLC data
            window_size: Number of candles per image
            img_size: Output image size (img_size x img_size)
            dpi: Resolution for rendering
        
        Returns:
            Tuple of (images array, volumes array)
        """
        from matplotlib.dates import date2num
        from mplfinance.original_flavor import candlestick_ohlc
        import matplotlib.pyplot as plt
        from PIL import Image
        import io
        
        encoded_images = []
        volumes = []
        
        print(f"    Encoding candles (window={window_size})...")
        
        for index in range(window_size, len(data)):
            subset = data.iloc[index - window_size:index + 1].copy()
            subset['date_num'] = subset['date'].apply(date2num)
            
            # Create figure with dark theme
            fig, ax = plt.subplots(figsize=(2, 2), dpi=dpi)
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            
            # Plot candlestick chart
            ohlc = subset[['date_num', 'open', 'high', 'low', 'close']].values
            candlestick_ohlc(ax, ohlc, width=0.6, colorup='lime', colordown='red', alpha=0.9)
            ax.axis('off')
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0,
                       facecolor='black')
            buf.seek(0)
            img = Image.open(buf).resize((img_size, img_size)).convert('RGB')
            plt.close(fig)
            
            encoded_images.append(np.array(img) / 255.0)
            volumes.append(float(data.iloc[index]['volume']))
        
        print(f"    Generated {len(encoded_images)} candle images")
        
        return np.array(encoded_images, dtype=np.float32), np.array(volumes, dtype=np.float32).reshape(-1, 1)
    
    def prepare_inputs(self, data, window_size=5, img_size=64):
        """
        Prepare model inputs from BFC data
        
        Args:
            data: DataFrame with BFC-filtered data and targets
            window_size: Number of candles per image
            img_size: Image size
        
        Returns:
            Tuple of (X_img, X_vol, y) ready for model training
        """
        # Encode candles
        images, volumes = self.encode_candles(data, window_size, img_size)
        
        # Normalize volume
        volumes_scaled = self.vol_scaler.fit_transform(volumes)
        
        # Reshape for TimeDistributed layer: (samples, timesteps=1, H, W, C)
        X_img = images.reshape(-1, 1, img_size, img_size, 3)
        X_vol = volumes_scaled
        
        # Get targets (align with encoded images)
        y = data['target'].iloc[window_size:].values[:len(X_img)]
        
        print(f"    X_img shape : {X_img.shape}")
        print(f"    X_vol shape : {X_vol.shape}")
        print(f"    y shape     : {y.shape}")
        
        # Store for later use
        self.images = X_img
        self.volumes = X_vol
        self.targets = y
        
        return X_img, X_vol, y
    
    def build_model(self, input_shape=(1, 64, 64, 3)):
        """
        Build BLENNS model architecture (CNN + LSTM + Attention)
        
        Architecture:
        - TimeDistributed CNN: Extracts spatial features from each candle image
        - LSTM: Captures temporal dependencies across the sequence
        - Self-Attention: Weights the most important time steps
        - Volume branch: Fused with visual features for final prediction
        
        Args:
            input_shape: Shape of input images (timesteps, height, width, channels)
        
        Returns:
            Compiled Keras model
        """
        # Image branch
        img_input = Input(shape=input_shape, name='img_input')
        
        # CNN feature extractor (applied to each timestep)
        x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(img_input)
        x = TimeDistributed(MaxPooling2D((2, 2)))(x)
        x = TimeDistributed(Dropout(0.3))(x)
        x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
        x = TimeDistributed(MaxPooling2D((2, 2)))(x)
        x = TimeDistributed(Dropout(0.3))(x)
        x = TimeDistributed(Flatten())(x)
        
        # LSTM temporal encoder
        x = LSTM(64, return_sequences=True)(x)
        x = Dropout(0.4)(x)
        
        # Self-Attention mechanism
        attn_out = Attention(name='self_attention')([x, x])
        
        # Volume branch
        vol_input = Input(shape=(1,), name='vol_input')
        
        # Feature fusion
        x = concatenate([Flatten()(attn_out), vol_input], name='feature_fusion')
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(1, activation='sigmoid', name='prediction')(x)
        
        # Create and compile model
        model = Model(inputs=[img_input, vol_input], outputs=output)
        model.compile(
            optimizer=Adam(0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def train_model(self, X_img=None, X_vol=None, y=None, n_splits=5, 
                   epochs=50, batch_size=32, verbose=1):
        """
        Walk-forward validation training with TimeSeriesSplit
        
        Args:
            X_img: Image inputs (if None, uses stored data)
            X_vol: Volume inputs (if None, uses stored data)
            y: Target values (if None, uses stored data)
            n_splits: Number of cross-validation folds
            epochs: Number of training epochs per fold
            batch_size: Batch size for training
            verbose: Verbosity level for training
        
        Returns:
            Dictionary with training metrics
        """
        # Use stored data if not provided
        if X_img is None:
            X_img = self.images
            X_vol = self.volumes
            y = self.targets
        
        if X_img is None:
            raise ValueError("No data available. Run prepare_inputs() first.")
        
        print(f"\n[6/8] Walk-forward training ({n_splits} folds, {epochs} epochs each)...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        self.metrics = {'fold_accs': [], 'fold_aucs': []}
        best_model = None
        best_auc_score = 0.0
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_img)):
            print(f"\n  ── Fold {fold+1}/{n_splits} | train={len(train_idx)}, val={len(val_idx)} ──")
            
            model = self.build_model(input_shape=(1, X_img.shape[2], X_img.shape[3], 3))
            
            history = model.fit(
                [X_img[train_idx], X_vol[train_idx]], y[train_idx],
                validation_data=([X_img[val_idx], X_vol[val_idx]], y[val_idx]),
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose
            )
            
            val_acc = history.history['val_accuracy'][-1]
            val_auc = history.history['val_auc'][-1]
            self.metrics['fold_accs'].append(val_acc)
            self.metrics['fold_aucs'].append(val_auc)
            
            print(f"  Fold {fold+1} → Val Acc: {val_acc*100:.2f}% | Val AUC: {val_auc*100:.2f}%")
            
            if val_auc > best_auc_score:
                best_auc_score = val_auc
                best_model = model
                self.training_history = history
        
        self.model = best_model
        
        print(f"\n  Average Accuracy : {np.mean(self.metrics['fold_accs'])*100:.2f}%")
        print(f"  Average AUC      : {np.mean(self.metrics['fold_aucs'])*100:.2f}%")
        print(f"  Best AUC Fold    : {best_auc_score*100:.2f}%")
        
        return self.metrics
    
    def evaluate_holdout(self, test_size=0.2):
        """
        Evaluate model on holdout data (last test_size portion)
        
        Args:
            test_size: Proportion of data to use for holdout evaluation
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        print("\n[7/8] Running holdout evaluation (last {:.0%} of data)...".format(test_size))
        
        eval_start = int(len(self.images) * (1 - test_size))
        y_pred_prob = self.model.predict([self.images[eval_start:], self.volumes[eval_start:]], 
                                         verbose=0).flatten()
        y_true = self.targets[eval_start:]
        
        final_acc = np.mean((y_pred_prob > 0.5).astype(int) == y_true)
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        final_auc = auc(fpr, tpr)
        cm = confusion_matrix(y_true, (y_pred_prob > 0.5).astype(int))
        
        print(f"  Holdout Accuracy : {final_acc*100:.2f}%")
        print(f"  Holdout AUC      : {final_auc*100:.2f}%")
        
        return {
            'accuracy': final_acc,
            'auc': final_auc,
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr
        }
    
    def monte_carlo_predict(self, X_img_sample=None, X_vol_sample=None, n_samples=100):
        """
        Monte Carlo Dropout prediction for uncertainty estimation
        
        Args:
            X_img_sample: Single image sample to predict
            X_vol_sample: Single volume sample to predict
            n_samples: Number of Monte Carlo samples
        
        Returns:
            Dictionary with mean prediction, standard deviation, and direction
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Use last sample if not provided
        if X_img_sample is None:
            X_img_sample = self.images[-1:]
            X_vol_sample = self.volumes[-1:]
        
        predictions = []
        for _ in range(n_samples):
            # Enable dropout during inference (training=True)
            pred = self.model([X_img_sample, X_vol_sample], training=True)
            predictions.append(float(pred.numpy()[0][0]))
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean()
        std_pred = predictions.std()
        
        direction = "Bullish" if mean_pred > 0.5 else "Bearish"
        confidence = mean_pred if direction == "Bullish" else 1 - mean_pred
        
        print(f"\n[8/8] Monte Carlo Dropout ({n_samples} passes)...")
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
    
    def predict_next_day(self, train_if_missing=True):
        """
        Generate prediction for the next trading day
        
        Args:
            train_if_missing: Automatically train if model is None
        
        Returns:
            Dictionary with prediction results
        """
        if self.last_data is None:
            self.last_data = self.get_data()
        
        # Prepare data
        data = self.create_target(self.last_data)
        self.prepare_inputs(data)
        
        # Train if needed
        if self.model is None and train_if_missing:
            print("Training model first...")
            self.train_model(n_splits=3, epochs=30)
        elif self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Run Monte Carlo prediction
        results = self.monte_carlo_predict(n_samples=100)
        
        return results
    
    def get_summary(self):
        """
        Print summary of model performance and configuration
        
        Returns:
            Dictionary with summary statistics
        """
        print("\n" + "="*62)
        print("  BLENNS ORIGINAL — PREDICTION SUMMARY")
        print("="*62)
        print(f"  Symbol        : {self.symbol}")
        
        if self.metrics.get('fold_accs'):
            print(f"  Avg Accuracy  : {np.mean(self.metrics['fold_accs'])*100:.2f}%")
            print(f"  Avg AUC       : {np.mean(self.metrics['fold_aucs'])*100:.2f}%")
        
        print("="*62)
        
        return {
            'symbol': self.symbol,
            'bfc_params': self.bfc_params,
            'metrics': self.metrics
        }
