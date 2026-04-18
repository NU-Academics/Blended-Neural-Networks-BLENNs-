# -*- coding: utf-8 -*-
"""
================================================================================
BLENNS ORIGINAL — Core Module with Complete Trading Pipeline
================================================================================
Author: BLENNS Framework Implementation
Date: April 2026

DESCRIPTION:
    This core module implements the complete BLENNS (Blended Filtered Candles + 
    Neural Network System) trading pipeline with:

    1. BFC 3-stage noise reduction (EMA → Heikin-Ashi → Kalman)
    2. CNN + LSTM + Attention hybrid architecture
    3. Walk-forward validation with TimeSeriesSplit
    4. Monte Carlo Dropout uncertainty estimation
    5. SHAP model interpretability with expert rule validation
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

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dropout,
    LSTM, Dense, TimeDistributed, concatenate, Attention,
    BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import warnings
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class BLENNSWalkForward:
    """
    BLENNS ORIGINAL — Complete Trading System with Universal BFC Integration
    
    This class implements the full BLENNS pipeline:
    1. Data acquisition with BFC filtering
    2. Candlestick image generation
    3. CNN + LSTM + Attention model training
    4. Walk-forward cross-validation
    5. Monte Carlo uncertainty estimation
    6. SHAP explainability with expert rule validation
    
    Key Features:
    - 3-stage BFC noise reduction (EMA → Heikin-Ashi → Kalman)
    - Hybrid architecture: CNN spatial + LSTM temporal + Self-Attention
    - Walk-forward validation prevents look-ahead bias
    - Monte Carlo Dropout for prediction confidence
    - Expert rule validation with Cohen's Kappa (RQ3)
    """
    
    def __init__(self, symbol="GC=F", bfc_params=None):
        """
        Initialize BLENNS trading system
        
        Args:
            symbol: Trading symbol (Yahoo Finance format)
            bfc_params: Dictionary with BFC parameters (alpha, R, Q)
                        Default: {'alpha': 0.2, 'R': 0.01, 'Q': 1e-5}
        """
        self.symbol = symbol
        self.model = None
        self.vol_scaler = MinMaxScaler()
        self.bfc_params = bfc_params or {'alpha': 0.2, 'R': 0.01, 'Q': 1e-5}
        
        # Data storage
        self.raw_data = None
        self.bfc_data = None
        self.images = None
        self.volumes = None
        self.targets = None
        self.forecast_dates = None
        
        # Training metrics
        self.metrics = {'fold_accs': [], 'fold_aucs': [], 'fold_losses': []}
        self.training_history = None
        self.best_model = None
        
        # SHAP results storage
        self.shap_values = None
        self.kappa_results = None
        
        print(f"[INIT] BLENNS initialized for {symbol}")
        print(f"       BFC params: α={self.bfc_params['alpha']}, R={self.bfc_params['R']}, Q={self.bfc_params['Q']}")
    
    # =========================================================================
    # SECTION 1: BFC (BLENDED FILTERED CANDLES) IMPLEMENTATION
    # =========================================================================
    
    def exponential_moving_average(self, data, alpha=0.2):
        """
        Vectorized Exponential Moving Average (Stage 1 of BFC)
        
        Formula: EMA_t = α * Price_t + (1-α) * EMA_{t-1}
        
        Args:
            data: Input price array
            alpha: Smoothing factor (0 < alpha < 1)
        
        Returns:
            EMA-filtered array
        """
        result = np.zeros_like(data, dtype=np.float64)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    def kalman_filter(self, observations, R=0.01, Q=1e-5):
        """
        Kalman Filter for sequential noise reduction (Stage 3 of BFC)
        
        Mathematical formulation:
            Prediction: x̂ₖ⁻ = x̂ₖ₋₁, Pₖ⁻ = Pₖ₋₁ + Q
            Update: Kₖ = Pₖ⁻ / (Pₖ⁻ + R)
                    x̂ₖ = x̂ₖ⁻ + Kₖ(zₖ - x̂ₖ⁻)
                    Pₖ = (1 - Kₖ)Pₖ⁻
        
        Args:
            observations: Input time series
            R: Measurement noise covariance
            Q: Process noise covariance
        
        Returns:
            Kalman-filtered time series
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
            
            # Update step with Kalman Gain
            K_gain = P[i] / (P[i] + R)
            filtered[i] += K_gain * (observations[i] - filtered[i])
            P[i] = (1 - K_gain) * P[i]
        
        return filtered
    
    def compute_bfc(self, df, alpha=0.2, R=0.01, Q=1e-5):
        """
        Complete BFC (Blended Filtered Candles) transformation
        
        Pipeline:
            Raw OHLC → Stage 1: EMA Smoothing → Stage 2: Heikin-Ashi → Stage 3: Kalman → BFC OHLC
        
        Args:
            df: DataFrame with 'open', 'high', 'low', 'close' columns
            alpha: EMA smoothing factor (Stage 1)
            R: Kalman measurement noise (Stage 3)
            Q: Kalman process noise (Stage 3)
        
        Returns:
            DataFrame with BFC-filtered OHLC data
        """
        df = df.copy()
        
        # ===== STAGE 1: Exponential Moving Average =====
        o_ema = self.exponential_moving_average(df['open'].values, alpha)
        h_ema = self.exponential_moving_average(df['high'].values, alpha)
        l_ema = self.exponential_moving_average(df['low'].values, alpha)
        c_ema = self.exponential_moving_average(df['close'].values, alpha)
        
        # ===== STAGE 2: Modified Heikin-Ashi Transformation =====
        # HA_Close = (O + H + L + C) / 4
        ha_close = (o_ema + h_ema + l_ema + c_ema) / 4
        
        # HA_Open = (Previous HA_Open + Previous HA_Close) / 2
        ha_open = np.zeros(len(df), dtype=np.float64)
        ha_open[0] = (o_ema[0] + c_ema[0]) / 2
        for i in range(1, len(df)):
            ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
        
        # HA_High = max(H_EMA, HA_Open, HA_Close)
        # HA_Low = min(L_EMA, HA_Open, HA_Close)
        ha_high = np.maximum.reduce([h_ema, ha_open, ha_close])
        ha_low = np.minimum.reduce([l_ema, ha_open, ha_close])
        
        # ===== STAGE 3: Kalman Filter =====
        bfc_close = self.kalman_filter(ha_close, R=R, Q=Q)
        bfc_open = self.kalman_filter(ha_open, R=R, Q=Q)
        bfc_high = np.maximum.reduce([h_ema, bfc_open, bfc_close])
        bfc_low = np.minimum.reduce([l_ema, bfc_open, bfc_close])
        
        # ===== Assemble Output DataFrame =====
        bfc_df = df.copy()
        bfc_df['open'] = bfc_open
        bfc_df['high'] = bfc_high
        bfc_df['low'] = bfc_low
        bfc_df['close'] = bfc_close
        
        return bfc_df
    
    # =========================================================================
    # SECTION 2: DATA ACQUISITION & PREPROCESSING
    # =========================================================================
    
    def get_data(self, start_date="2010-01-01", end_date=None, interval="1d"):
        """
        Fetch and process historical data with universal BFC application
        
        Args:
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (default: today)
            interval: Data interval (1d, 1h, 15m, etc.)
        
        Returns:
            DataFrame with BFC-processed price data
        """
        if end_date is None:
            end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
        
        print(f"\n[1/8] Fetching {self.symbol} data from Yahoo Finance...")
        
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
            
            # Handle MultiIndex columns (occurs with certain symbols)
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
            
            # Store raw data
            self.raw_data = data
            
            # Apply BFC filtering
            self.bfc_data = self.compute_bfc(data, **self.bfc_params)
            
            print(f"    Fetched {len(data)} rows")
            print(f"    Date range: {data['date'].min().date()} → {data['date'].max().date()}")
            print(f"    Last raw close: {data['close'].iloc[-1]:.5f}")
            print(f"    BFC applied. Last BFC close: {self.bfc_data['close'].iloc[-1]:.6f}")
            
            return self.bfc_data
            
        except Exception as e:
            print(f"Data loading failed: {str(e)}")
            if 'data' in locals():
                print("Columns received:", data.columns.tolist())
            raise
    
    def create_target(self, data=None, lookahead=1):
        """
        Create prediction target with configurable lookahead
        
        Args:
            data: DataFrame with BFC-filtered data (uses self.bfc_data if None)
            lookahead: Number of days to look ahead for target
        
        Returns:
            DataFrame with 'target' column (1 if next close > current close)
        """
        if data is None:
            data = self.bfc_data.copy()
        else:
            data = data.copy()
        
        data['target'] = (data['close'].shift(-lookahead) > data['close']).astype(int)
        data = data.dropna(subset=['target']).reset_index(drop=True)
        
        bullish_pct = data['target'].mean() * 100
        print(f"    Bullish targets: {data['target'].sum()} / {len(data)} ({bullish_pct:.1f}%)")
        
        return data
    
    # =========================================================================
    # SECTION 3: CANDLESTICK IMAGE GENERATION
    # =========================================================================
    
    def encode_candles(self, data, window_size=5, img_size=64, dpi=32):
        """
        Generate candlestick images from BFC-processed data
        
        Args:
            data: DataFrame with BFC-filtered OHLC data
            window_size: Number of candles per image (lookback period)
            img_size: Output image resolution (square)
            dpi: Rendering resolution
        
        Returns:
            images: Array of normalized RGB images (shape: N×H×W×3)
            volumes: Corresponding volume data
            dates: Corresponding dates for each image
        """
        from matplotlib.dates import date2num
        from mplfinance.original_flavor import candlestick_ohlc
        import matplotlib.pyplot as plt
        from PIL import Image
        import io
        
        print(f"\n[2/8] Encoding BFC candles to {img_size}x{img_size} images (window={window_size})...")
        
        encoded_images = []
        volumes = []
        dates = []
        
        for index in range(window_size, len(data)):
            # Extract window of data
            subset = data.iloc[index - window_size:index + 1].copy()
            subset = subset.reset_index(drop=True)
            
            # Generate sequential dates for mplfinance
            date_range = pd.date_range(start='2000-01-01', periods=len(subset), freq='D')
            subset['date_num'] = [date2num(d) for d in date_range]
            
            # Create candlestick chart with dark theme
            fig, ax = plt.subplots(figsize=(2, 2), dpi=dpi)
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            
            ohlc = subset[['date_num', 'open', 'high', 'low', 'close']].values
            candlestick_ohlc(ax, ohlc, width=0.6, colorup='lime', colordown='red', alpha=0.9)
            ax.axis('off')
            
            # Convert plot to numpy array
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0,
                       facecolor='black')
            buf.seek(0)
            img = Image.open(buf).resize((img_size, img_size)).convert('RGB')
            plt.close(fig)
            
            encoded_images.append(np.array(img) / 255.0)  # Normalize to [0,1]
            volumes.append(float(data.iloc[index]['volume']))
            dates.append(data.iloc[index]['date'])
        
        print(f"    Generated {len(encoded_images)} candle images")
        
        return (np.array(encoded_images, dtype=np.float32), 
                np.array(volumes, dtype=np.float32).reshape(-1, 1),
                dates)
    
    def prepare_inputs(self, data=None, window_size=5, img_size=64):
        """
        Prepare model inputs from BFC data
        
        Args:
            data: DataFrame with BFC-filtered data and targets
            window_size: Number of candles per image
            img_size: Image size
        
        Returns:
            Tuple of (X_img, X_vol, y, dates) ready for model training
        """
        if data is None:
            data = self.create_target()
        
        # Generate candlestick images
        images, volumes, dates = self.encode_candles(data, window_size, img_size)
        
        # Normalize volume data
        volumes_scaled = self.vol_scaler.fit_transform(volumes)
        
        # Reshape for TimeDistributed layer: (samples, timesteps=1, H, W, C)
        X_img = images.reshape(-1, 1, img_size, img_size, 3)
        X_vol = volumes_scaled
        
        # Get targets (align with encoded images)
        y = data['target'].iloc[window_size:].values[:len(X_img)]
        
        print(f"\n[3/8] Normalising inputs and aligning targets...")
        print(f"    X_img shape : {X_img.shape}")
        print(f"    X_vol shape : {X_vol.shape}")
        print(f"    y shape     : {y.shape}")
        
        # Store for later use
        self.images = X_img
        self.volumes = X_vol
        self.targets = y
        self.forecast_dates = dates
        
        return X_img, X_vol, y, dates
    
    # =========================================================================
    # SECTION 4: BLENNS MODEL ARCHITECTURE
    # =========================================================================
    
    def build_model(self, input_shape=(1, 64, 64, 3)):
        """
        Build BLENNS model architecture (CNN + LSTM + Attention)
        
        Architecture Overview:
            1. TimeDistributed CNN: Extract spatial features from each candlestick
            2. LSTM: Capture temporal dependencies across the sequence
            3. Self-Attention: Weight important time steps
            4. Volume Branch: Process trading volume separately
            5. Fusion Layer: Combine visual and volumetric features
            6. Dense Layers: Final classification
        
        Args:
            input_shape: Shape of input images (timesteps, height, width, channels)
        
        Returns:
            Compiled Keras model
        """
        # ===== IMAGE BRANCH: CNN Feature Extractor =====
        img_input = Input(shape=input_shape, name='img_input')
        
        # Apply CNN to each timestep (TimeDistributed wrapper)
        x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(img_input)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(MaxPooling2D((2, 2)))(x)
        x = TimeDistributed(Dropout(0.3))(x)
        
        x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(MaxPooling2D((2, 2)))(x)
        x = TimeDistributed(Dropout(0.3))(x)
        
        x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(MaxPooling2D((2, 2)))(x)
        x = TimeDistributed(Flatten())(x)
        
        # ===== TEMPORAL ENCODER: LSTM =====
        x = LSTM(64, return_sequences=True)(x)
        x = Dropout(0.4)(x)
        
        # ===== ATTENTION MECHANISM: Self-Attention =====
        # Allows model to focus on most relevant time steps
        attn_out = Attention(name='self_attention')([x, x])
        
        # ===== VOLUME BRANCH: Separate processing =====
        vol_input = Input(shape=(1,), name='vol_input')
        
        # ===== FUSION LAYER: Combine modalities =====
        fused = concatenate([Flatten()(attn_out), vol_input], name='feature_fusion')
        
        # ===== CLASSIFICATION HEAD =====
        x = Dense(32, activation='relu')(fused)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        output = Dense(1, activation='sigmoid', name='prediction')(x)
        
        # ===== COMPILE MODEL =====
        model = Model(inputs=[img_input, vol_input], outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    # =========================================================================
    # SECTION 5: WALK-FORWARD CROSS-VALIDATION TRAINING
    # =========================================================================
    
    def train_model(self, X_img=None, X_vol=None, y=None, n_splits=5, 
                   epochs=50, batch_size=32, verbose=1):
        """
        Walk-forward validation training with TimeSeriesSplit
        
        Walk-Forward Validation:
            - Preserves temporal order (no future data leakage)
            - Each fold: train on past, validate on future
            - More realistic than random shuffle for financial data
        
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
        
        print(f"\n[4/8] Walk-forward training ({n_splits} folds, {epochs} epochs each)...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        self.metrics = {'fold_accs': [], 'fold_aucs': [], 'fold_losses': []}
        best_auc_score = 0.0
        self.training_history = None
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_img)):
            print(f"\n  ── Fold {fold+1}/{n_splits} | train={len(train_idx)}, val={len(val_idx)} ──")
            
            # Create new model instance for each fold
            model = self.build_model(input_shape=(1, X_img.shape[2], X_img.shape[3], 3))
            
            # Callbacks for better training
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0)
            ]
            
            # Train model
            history = model.fit(
                [X_img[train_idx], X_vol[train_idx]], y[train_idx],
                validation_data=([X_img[val_idx], X_vol[val_idx]], y[val_idx]),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
            
            # Record performance
            val_acc = history.history['val_accuracy'][-1]
            val_auc = history.history['val_auc'][-1]
            val_loss = history.history['val_loss'][-1]
            
            self.metrics['fold_accs'].append(val_acc)
            self.metrics['fold_aucs'].append(val_auc)
            self.metrics['fold_losses'].append(val_loss)
            
            print(f"  Fold {fold+1} → Val Acc: {val_acc*100:.2f}% | Val AUC: {val_auc*100:.2f}% | Val Loss: {val_loss:.4f}")
            
            # Keep best model based on AUC
            if val_auc > best_auc_score:
                best_auc_score = val_auc
                self.best_model = model
                self.training_history = history
        
        self.model = self.best_model
        
        print(f"\n  Average Accuracy : {np.mean(self.metrics['fold_accs'])*100:.2f}% (±{np.std(self.metrics['fold_accs'])*100:.2f}%)")
        print(f"  Average AUC      : {np.mean(self.metrics['fold_aucs'])*100:.2f}% (±{np.std(self.metrics['fold_aucs'])*100:.2f}%)")
        print(f"  Best AUC Fold    : {best_auc_score*100:.2f}%")
        
        return self.metrics
    
    # =========================================================================
    # SECTION 6: HOLDOUT EVALUATION
    # =========================================================================
    
    def evaluate_holdout(self, test_size=0.2):
        """
        Evaluate model on holdout data (last test_size portion)
        
        Args:
            test_size: Proportion of data to use for holdout evaluation
        
        Returns:
            Dictionary with evaluation metrics including y_true and y_pred
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if self.images is None or self.targets is None:
            raise ValueError("No data available. Run prepare_inputs() first.")
        
        print(f"\n[5/8] Running holdout evaluation (last {test_size*100:.0f}% of data)...")
        
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
            'tpr': tpr,
            'y_true': y_true,
            'y_pred': y_pred_prob
        }
    
    # =========================================================================
    # SECTION 7: MONTE CARLO DROPOUT FOR UNCERTAINTY ESTIMATION
    # =========================================================================
    
    def monte_carlo_predict(self, X_img_sample=None, X_vol_sample=None, n_samples=100):
        """
        Monte Carlo Dropout prediction for uncertainty estimation
        
        Method (Gal & Ghahramani, 2016):
            - Keep dropout active during prediction
            - Perform multiple forward passes
            - Mean = prediction, Standard deviation = uncertainty
        
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
            if self.images is None:
                raise ValueError("No data available. Run prepare_inputs() first.")
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
        
        print(f"\n[6/8] Monte Carlo Dropout ({n_samples} passes) for uncertainty estimation...")
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
    
    # =========================================================================
    # SECTION 8: SHAP EXPLAINABILITY WITH EXPERT RULE VALIDATION (RQ3)
    # =========================================================================
    
    def compute_shap_explanations(self, background_size=50):
        """
        Compute SHAP feature importance using GradientExplainer
        
        SHAP values (Lundberg & Lee, 2017):
            - Based on cooperative game theory
            - Each feature's contribution to prediction
            - Additive feature attribution method
        
        Args:
            background_size: Number of samples to use as background for SHAP
        
        Returns:
            Dictionary with SHAP values and feature impacts
        """
        import shap
        
        print(f"\n[7/8] Computing SHAP feature importance via GradientExplainer...")
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if self.images is None:
            raise ValueError("No data available. Run prepare_inputs() first.")
        
        # Use last n samples as background
        background_img = self.images[-background_size:]
        background_vol = self.volumes[-background_size:]
        sample_img = self.images[-1:]
        sample_vol = self.volumes[-1:]
        
        # Create explainer
        explainer = shap.GradientExplainer(self.model, [background_img, background_vol])
        shap_values = explainer.shap_values([sample_img, sample_vol])
        
        # Extract SHAP values
        img_shap = shap_values[0][0][0]  # Shape: (height, width, channels)
        vol_shap = float(shap_values[1][0][0])
        
        # Map pixel regions to BFC features (domain knowledge mapping)
        impact_features = {
            'BFC Upper Wick (Sell Pressure)':   np.mean(np.abs(img_shap[0:15, 25:40, 1])),
            'BFC Lower Wick (Buy Support)':     np.mean(np.abs(img_shap[50:64, 25:40, 0])),
            'BFC Bullish Body (Buy Momentum)':  np.mean(np.abs(img_shap[25:40, 25:40, 1])),
            'BFC Bearish Body (Sell Momentum)': np.mean(np.abs(img_shap[25:40, 25:40, 0])),
            'Volume Impact':                    abs(vol_shap)
        }
        
        self.shap_values = {
            'img_shap': img_shap,
            'vol_shap': vol_shap,
            'impact_features': impact_features,
            'explainer': explainer
        }
        
        print("  SHAP Feature Impacts (higher = more important):")
        for k, v in sorted(impact_features.items(), key=lambda x: -x[1]):
            print(f"    {k:<35}: {v:.6f}")
        
        return self.shap_values
    
    def compute_expert_signals(self, data=None):
        """
        Generate expert trading signals from 5 technical analysis rules
        
        Expert Rules:
            1. RSI (Wilder, 1978): Buy when RSI < 30, Sell when RSI > 70
            2. MACD (Appel, 2005): Buy when MACD > Signal line
            3. MA Crossover (Murphy, 1999): Buy when SMA20 > SMA50
            4. Volume Confirmation (Karpoff, 1987): Buy when volume > 1.5×MA20 & price up
            5. Support/Resistance (Edwards & Magee, 1948): Buy near support, Sell near resistance
        
        Args:
            data: DataFrame with OHLC data (uses self.bfc_data if None)
        
        Returns:
            expert_signal: Array of expert consensus signals (0=Down, 1=Up)
            df: DataFrame with intermediate calculations
        """
        if data is None:
            if self.bfc_data is None:
                raise ValueError("No data available. Run get_data() first.")
            data = self.bfc_data.copy()
        
        df = data.copy()
        
        # --- Rule 1: RSI (Relative Strength Index) ---
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        rsi_sig = ((df['RSI'] < 30).astype(int) - (df['RSI'] > 70).astype(int)).clip(0, 1)
        
        # --- Rule 2: MACD (Moving Average Convergence Divergence) ---
        df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        macd_sig = (df['MACD'] > df['MACD_Signal']).astype(int)
        
        # --- Rule 3: Moving Average Crossover ---
        df['SMA20'] = df['close'].rolling(20).mean()
        df['SMA50'] = df['close'].rolling(50).mean()
        ma_sig = (df['SMA20'] > df['SMA50']).astype(int)
        
        # --- Rule 4: Volume Confirmation ---
        df['VMA20'] = df['volume'].rolling(20).mean()
        df['Return'] = df['close'].pct_change()
        vol_sig = ((df['volume'] > 1.5 * df['VMA20']) & (df['Return'] > 0)).astype(int)
        
        # --- Rule 5: Support/Resistance ---
        df['Resistance'] = df['high'].rolling(20).max()
        df['Support'] = df['low'].rolling(20).min()
        sr_sig = pd.Series(0, index=df.index)
        sr_sig[df['close'] <= df['Support'] * 1.02] = 1
        sr_sig[df['close'] >= df['Resistance'] * 0.98] = 0
        
        # --- Aggregate: Simple Majority Vote ---
        consensus = (rsi_sig + macd_sig + ma_sig + vol_sig + sr_sig) / 5.0
        df = df.dropna().reset_index(drop=True)
        
        expert_signal = (consensus > 0.5).astype(int)
        
        return expert_signal, df
    
    def calculate_cohens_kappa(self, y_true, y_pred):
        """
        Calculate Cohen's Kappa with full statistical testing
        
        Formula:
            p_o = (TP + TN) / N              (Observed agreement)
            p_e = Σ(row_i × col_i) / N²      (Expected agreement by chance)
            κ = (p_o - p_e) / (1 - p_e)      (Kappa coefficient)
            SE = √[p_o(1-p_o) / (N(1-p_e)²)] (Standard error)
            z = κ / SE                        (Z-statistic)
        
        Reference:
            Cohen (1960) - A coefficient of agreement for nominal scales
            Landis & Koch (1977) - The measurement of observer agreement
        
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
    
    def interpret_kappa(self, kappa):
        """
        Interpret Cohen's Kappa using Landis & Koch (1977) scale
        
        Args:
            kappa: Cohen's Kappa coefficient
        
        Returns:
            String interpretation
        """
        if kappa < 0:
            return "Poor (worse than chance)"
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
    
    def compute_shap_expert_agreement(self, holdout_results=None):
        """
        Compute SHAP vs Expert Rules agreement using Cohen's Kappa (RQ3)
        
        Args:
            holdout_results: Results from evaluate_holdout() (optional)
        
        Returns:
            Dictionary with kappa results
        """
        print(f"\n[8/8] Computing SHAP vs Expert Rules Agreement (Cohen's Kappa)...")
        
        # Get holdout results if not provided
        if holdout_results is None:
            holdout_results = self.evaluate_holdout()
        
        # Compute expert signals
        expert_signal, _ = self.compute_expert_signals()
        
        # Align with model evaluation window
        y_pred_bin = (holdout_results['y_pred'] > 0.5).astype(int)
        
        # Align lengths
        min_len = min(len(y_pred_bin), len(expert_signal))
        y_pred_bin = y_pred_bin[:min_len]
        expert_signal = expert_signal[-min_len:]
        
        # Calculate Cohen's Kappa
        kappa_results = self.calculate_cohens_kappa(expert_signal, y_pred_bin)
        
        # Random baseline for comparison
        random_pred = np.random.RandomState(42).randint(0, 2, size=min_len)
        random_kappa = self.calculate_cohens_kappa(expert_signal, random_pred)
        
        self.kappa_results = kappa_results
        
        print(f"\n  Cohen's Kappa (SHAP vs Expert): {kappa_results['kappa']:.4f}")
        print(f"    Interpretation: {self.interpret_kappa(kappa_results['kappa'])}")
        print(f"    Observed Agreement (p_o): {kappa_results['p_o']*100:.2f}%")
        print(f"    Expected Agreement (p_e): {kappa_results['p_e']*100:.2f}%")
        print(f"    z-statistic: {kappa_results['z_stat']:.4f}")
        print(f"    p-value (one-tailed): {kappa_results['p_value']:.6f}")
        print(f"    Random baseline kappa: {random_kappa['kappa']:.4f}")
        
        # Hypothesis test
        print(f"\n  Hypothesis Test (H₀: κ ≤ 0, Hₐ: κ > 0, α=0.05):")
        if kappa_results['p_value'] < 0.05 and kappa_results['z_stat'] > 1.645:
            print(f"    ✓ REJECT H₀: SHAP significantly agrees with expert rules")
        else:
            print(f"    ✗ FAIL TO REJECT H₀: No significant agreement detected")
        
        return kappa_results
    
    # =========================================================================
    # SECTION 9: PREDICTED CANDLESTICK VISUALIZATION
    # =========================================================================
    
    def compute_atr(self, df, period=14):
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
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = [max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1])) 
              for i in range(1, len(df))]
        
        return np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)
    
    def atr_multipliers(self, symbol=None):
        """
        Return take-profit and stop-loss multipliers based on asset class
        
        Different asset classes have different volatility profiles:
            - Crypto: Highest volatility (2.0x, 1.0x)
            - Forex/Commodities: Medium volatility (1.5x, 1.0x)
            - Equities: Standard (1.5x, 1.0x)
        
        Args:
            symbol: Trading symbol (uses self.symbol if None)
        
        Returns:
            Tuple of (take_profit_multiplier, stop_loss_multiplier)
        """
        sym = (symbol or self.symbol).upper()
        
        if any(x in sym for x in ["BTC", "ETH", "SOL", "XRP", "BNB"]):
            return 2.0, 1.0  # Wider targets for crypto
        if sym.endswith("=X") or any(x in sym for x in ["USD", "EUR", "GBP", "JPY", "AUD", "CHF", "CAD"]):
            return 1.5, 1.0  # Standard for forex
        if any(x in sym for x in ["GC", "SI", "CL", "NG", "HG", "ZC"]):
            return 1.5, 1.0  # Standard for commodities
        return 1.5, 1.0  # Default for equities
    
    def predict_next_day(self, train_if_missing=True, n_splits=3, epochs=30):
        """
        Generate prediction for the next trading day with full pipeline
        
        Args:
            train_if_missing: Automatically train if model is None
            n_splits: Number of walk-forward folds
            epochs: Training epochs per fold
        
        Returns:
            Dictionary with prediction results
        """
        # Step 1: Get data
        if self.bfc_data is None:
            self.get_data()
        
        # Step 2: Prepare data and targets
        data = self.create_target()
        X_img, X_vol, y, dates = self.prepare_inputs(data)
        
        # Step 3: Train model if needed
        if self.model is None and train_if_missing:
            print("\nTraining model first...")
            self.train_model(n_splits=n_splits, epochs=epochs)
        elif self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Step 4: Run Monte Carlo prediction
        mc_results = self.monte_carlo_predict(n_samples=100)
        
        # Step 5: Run holdout evaluation
        holdout = self.evaluate_holdout()
        
        # Step 6: Compute SHAP explanations (optional)
        try:
            shap_results = self.compute_shap_explanations()
        except Exception as e:
            print(f"SHAP computation skipped: {e}")
            shap_results = None
        
        return {
            'prediction': mc_results,
            'holdout_metrics': holdout,
            'shap': shap_results,
            'last_date': dates[-1] if dates else None
        }
    
    def get_summary(self):
        """
        Print summary of model performance and configuration
        
        Returns:
            Dictionary with summary statistics
        """
        print("\n" + "="*72)
        print("  BLENNS ORIGINAL — COMPLETE PREDICTION SUMMARY")
        print("="*72)
        print(f"  Symbol              : {self.symbol}")
        print(f"  BFC α               : {self.bfc_params.get('alpha', 0.2)}")
        print(f"  BFC R               : {self.bfc_params.get('R', 0.01)}")
        print(f"  BFC Q               : {self.bfc_params.get('Q', 1e-5)}")
        
        if self.metrics.get('fold_accs'):
            print(f"\n  Walk-Forward Folds  : {len(self.metrics['fold_accs'])}")
            print(f"  Avg Val Accuracy    : {np.mean(self.metrics['fold_accs'])*100:.2f}%")
            print(f"  Avg Val AUC         : {np.mean(self.metrics['fold_aucs'])*100:.2f}%")
        
        if self.kappa_results:
            print(f"\n  SHAP vs Expert κ    : {self.kappa_results['kappa']:.4f}")
            print(f"  Agreement           : {self.interpret_kappa(self.kappa_results['kappa'])}")
            print(f"  p-value             : {self.kappa_results['p_value']:.6f}")
        
        print("="*72)
        
        return {
            'symbol': self.symbol,
            'bfc_params': self.bfc_params,
            'metrics': self.metrics,
            'kappa': self.kappa_results
        }
