"""
BLENNS Trading System - Core Module with BFC Integration
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class BLENNSWalkForward:
    """BLENNS Walk Forward Trading System with BFC Integration"""
    
    def __init__(self, symbol="AAPL", bfc_params=None):
        """
        Initialize BLENNS Walk Forward model
        
        Parameters:
        -----------
        symbol : str
            Stock symbol (default: "AAPL")
        bfc_params : dict
            Parameters for Blenns Filter Candles
        """
        self.symbol = symbol
        self.model = None
        self.scaler = MinMaxScaler()
        self.bfc_params = bfc_params or {'alpha': 0.2, 'R': 0.1**2, 'Q': 1e-5}
        self.last_data = None
        self.history = None
        
    def compute_bfc(self, df, alpha=0.2, R=0.1**2, Q=1e-5):
        """
        Enhanced Blenns Filter Candles (BFC) processing
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with OHLC data
        alpha : float
            EMA smoothing factor
        R : float
            Kalman filter measurement noise
        Q : float
            Kalman filter process noise
            
        Returns:
        --------
        pandas.DataFrame with BFC processed OHLC
        """
        df = df.copy()

        # EMA Smoothing Stage
        df['O_ema'] = self._exponential_moving_average(df['open'].values, alpha)
        df['H_ema'] = self._exponential_moving_average(df['high'].values, alpha)
        df['L_ema'] = self._exponential_moving_average(df['low'].values, alpha)
        df['C_ema'] = self._exponential_moving_average(df['close'].values, alpha)

        # Modified Heikin-Ashi Stage
        bfc = pd.DataFrame(index=df.index)
        bfc['HA_Close'] = (df['O_ema'] + df['H_ema'] + df['L_ema'] + df['C_ema']) / 4

        # Vectorized Heikin-Ashi Open calculation
        ha_open = np.zeros(len(df))
        ha_open[0] = (df['O_ema'].iloc[0] + df['C_ema'].iloc[0]) / 2
        for i in range(1, len(df)):
            ha_open[i] = (ha_open[i-1] + bfc['HA_Close'].iloc[i-1]) / 2
        bfc['HA_Open'] = ha_open

        bfc['HA_High'] = np.maximum.reduce([
            df['H_ema'].values, 
            bfc['HA_Open'].values, 
            bfc['HA_Close'].values
        ])
        
        bfc['HA_Low'] = np.minimum.reduce([
            df['L_ema'].values, 
            bfc['HA_Open'].values, 
            bfc['HA_Close'].values
        ])

        # Kalman Filter Stage
        bfc['BFC_Close'] = self._kalman_filter(bfc['HA_Close'].values, R=R, Q=Q)
        bfc['BFC_Open'] = self._kalman_filter(bfc['HA_Open'].values, R=R, Q=Q)

        # Ensure High/Low consistency
        bfc['BFC_High'] = np.maximum.reduce([
            df['H_ema'].values, 
            bfc['BFC_Open'].values, 
            bfc['BFC_Close'].values
        ])
        
        bfc['BFC_Low'] = np.minimum.reduce([
            df['L_ema'].values, 
            bfc['BFC_Open'].values, 
            bfc['BFC_Close'].values
        ])

        return bfc[['BFC_Open', 'BFC_High', 'BFC_Low', 'BFC_Close']]
    
    def _exponential_moving_average(self, data, alpha):
        """Vectorized Exponential Moving Average"""
        result = np.zeros_like(data, dtype=np.float64)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    def _kalman_filter(self, observations, R=0.1**2, Q=1e-5):
        """Enhanced Kalman Filter with configurable noise parameters"""
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
            K = P[i] / (P[i] + R)  # Kalman Gain
            filtered[i] += K * (observations[i] - filtered[i])
            P[i] = (1 - K) * P[i]

        return filtered
    
    def get_data(self, start_date="2020-01-01", end_date=None, interval="1d"):
        """
        Fetch and process historical data with BFC application
        
        Parameters:
        -----------
        start_date : str
            Start date for data
        end_date : str
            End date for data (default: today)
        interval : str
            Data interval (1d, 1h, etc.)
            
        Returns:
        --------
        pandas.DataFrame with processed data
        """
        import pandas as pd
        from datetime import datetime
        
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')

        try:
            # Download data
            print(f"📥 Downloading {self.symbol} data from {start_date} to {end_date}...")
            data = yf.download(
                tickers=self.symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=True
            )

            # Reset index
            if isinstance(data.index, pd.MultiIndex):
                data = data.reset_index()
            
            # Standardize column names
            column_map = {
                'Date': 'date',
                'Datetime': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close'
            }
            
            data = data.rename(columns={k: v for k, v in column_map.items() 
                                      if k in data.columns})
            
            # Ensure date column is datetime
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            else:
                data = data.reset_index().rename(columns={'index': 'date'})
                data['date'] = pd.to_datetime(data['date'])

            # Validate required columns
            required_ohlc = ['open', 'high', 'low', 'close']
            if not all(col in data.columns for col in required_ohlc):
                raise ValueError(f"Missing required OHLC columns. Available: {list(data.columns)}")

            # Apply BFC
            print("🔄 Applying BFC processing...")
            bfc_data = self.compute_bfc(data, **self.bfc_params)
            
            # Update OHLC with BFC values
            data[['open', 'high', 'low', 'close']] = bfc_data.values
            
            # Add volume if missing
            if 'volume' not in data.columns:
                data['volume'] = 0

            self.last_data = data
            print(f"✅ Successfully loaded {len(data)} rows of data")
            return data

        except Exception as e:
            print(f"❌ Data loading failed: {str(e)}")
            raise
    
    def create_target(self, data, lookahead=1, threshold=0.0):
        """
        Create prediction target
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
        lookahead : int
            Number of periods to look ahead
        threshold : float
            Minimum percentage change to consider as signal
            
        Returns:
        --------
        pandas.DataFrame with target column
        """
        data = data.copy()

        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")

        # Calculate future returns
        future_close = data['close'].shift(-lookahead)
        returns = (future_close - data['close']) / data['close']
        
        # Create binary target
        data['target'] = (returns > threshold).astype(int)
        
        # Drop NaN values
        data = data.dropna(subset=['target'])
        
        print(f"✅ Created target: {data['target'].sum()} buy signals out of {len(data)} samples")
        return data
    
    def encode_candles(self, data, window_size=5, img_size=64, dpi=32):
        """
        Generate candlestick images from BFC-processed data
        
        Parameters:
        -----------
        data : pandas.DataFrame
            OHLC data
        window_size : int
            Number of candles per image
        img_size : int
            Output image size
        dpi : int
            Image resolution
            
        Returns:
        --------
        tuple: (images array, volumes array)
        """
        import matplotlib.pyplot as plt
        from matplotlib.dates import date2num
        from mplfinance.original_flavor import candlestick_ohlc
        from PIL import Image
        import io
        
        required_cols = ['date', 'open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in data.columns]
            raise ValueError(f"Missing columns: {missing}")

        encoded_images = []
        volumes = []

        print(f"🖼️ Encoding {len(data) - window_size} candle images...")
        
        for i in range(window_size, len(data)):
            subset = data.iloc[i-window_size:i].copy()
            subset = subset.reset_index(drop=True)
            
            # Convert dates to matplotlib format
            subset['date_num'] = np.arange(len(subset))
            
            # Create figure
            fig, ax = plt.subplots(figsize=(img_size/dpi, img_size/dpi), dpi=dpi)
            
            # Plot candlestick
            candlestick_ohlc(
                ax, 
                subset[['date_num', 'open', 'high', 'low', 'close']].values,
                width=0.6, 
                colorup='green', 
                colordown='red',
                alpha=1.0
            )
            
            # Format plot
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_facecolor('white')
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, 
                       facecolor='white', dpi=dpi)
            buf.seek(0)
            
            # Convert to array
            img = Image.open(buf).resize((img_size, img_size)).convert('RGB')
            encoded_images.append(np.array(img) / 255.0)
            
            # Get volume
            if 'volume' in data.columns:
                volumes.append(float(data.iloc[i]['volume']))
            else:
                volumes.append(0.0)
            
            plt.close(fig)
            
            # Progress indicator
            if i % 100 == 0:
                print(f"  Processed {i}/{len(data)} candles...")

        print(f"✅ Encoded {len(encoded_images)} images")
        return np.array(encoded_images, dtype=np.float32), np.array(volumes, dtype=np.float32).reshape(-1, 1)
    
    def build_model(self, input_shape=(1, 64, 64, 3)):
        """
        Build the BLENNS model architecture
        
        Parameters:
        -----------
        input_shape : tuple
            Input image shape
            
        Returns:
        --------
        tensorflow.keras.Model
        """
        # Image processing branch
        img_input = Input(shape=input_shape)
        x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(img_input)
        x = TimeDistributed(MaxPooling2D((2, 2)))(x)
        x = TimeDistributed(Dropout(0.3))(x)
        x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
        x = TimeDistributed(MaxPooling2D((2, 2)))(x)
        x = TimeDistributed(Flatten())(x)

        # Temporal processing
        x = LSTM(64, return_sequences=True)(x)
        x = Dropout(0.4)(x)
        
        # Attention mechanism
        attention = Attention()([x, x])
        x = tf.keras.layers.GlobalAveragePooling1D()(attention)

        # Volume input branch
        vol_input = Input(shape=(1,))
        vol_dense = Dense(16, activation='relu')(vol_input)

        # Feature fusion
        x = concatenate([x, vol_dense])

        # Prediction head
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[img_input, vol_input], outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        print("✅ Model built successfully")
        return model
    
    def train_model(self, X_img, X_vol, y, n_splits=5, epochs=20, batch_size=32, validation_split=0.2):
        """
        Train model with walk-forward validation
        
        Parameters:
        -----------
        X_img : numpy.ndarray
            Image data
        X_vol : numpy.ndarray
            Volume data
        y : numpy.ndarray
            Target labels
        n_splits : int
            Number of time series splits
        epochs : int
            Training epochs per fold
        batch_size : int
            Batch size
        validation_split : float
            Validation split ratio
            
        Returns:
        --------
        dict with training metrics
        """
        if len(X_img) != len(X_vol) or len(X_img) != len(y):
            raise ValueError(f"Data length mismatch: X_img={len(X_img)}, X_vol={len(X_vol)}, y={len(y)}")
        
        print(f"🏋️ Training model on {len(X_img)} samples...")
        
        # Reshape images if needed
        if len(X_img.shape) == 4:  # (n_samples, height, width, channels)
            X_img = X_img.reshape(-1, 1, X_img.shape[1], X_img.shape[2], X_img.shape[3])
        
        # Normalize volume
        X_vol = self.scaler.fit_transform(X_vol)
        
        # Build model
        self.model = self.build_model(input_shape=X_img.shape[1:])
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=min(n_splits, len(X_img)//10))
        metrics = {'val_acc': [], 'val_auc': [], 'val_loss': []}
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_img), 1):
            print(f"\n📊 Fold {fold}/{tscv.n_splits}")
            print(f"   Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
            
            # Reset model for each fold
            self.model = self.build_model(input_shape=X_img.shape[1:])
            
            # Train
            history = self.model.fit(
                [X_img[train_idx], X_vol[train_idx]], 
                y[train_idx],
                validation_data=([X_img[val_idx], X_vol[val_idx]], y[val_idx]),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=3,
                        min_lr=1e-6
                    )
                ]
            )
            
            # Store metrics
            metrics['val_acc'].append(history.history['val_accuracy'][-1])
            metrics['val_auc'].append(history.history['val_auc'][-1])
            metrics['val_loss'].append(history.history['val_loss'][-1])
            
            self.history = history
        
        print(f"\n✅ Training complete!")
        print(f"   Average Validation Accuracy: {np.mean(metrics['val_acc']):.3f}")
        print(f"   Average Validation AUC: {np.mean(metrics['val_auc']):.3f}")
        
        return metrics
    
    def predict_next_day(self, train_if_missing=True):
        """
        Generate prediction for the next trading day
        
        Parameters:
        -----------
        train_if_missing : bool
            Train model if not already trained
            
        Returns:
        --------
        str: 'Buy' or 'Sell' recommendation
        """
        if self.last_data is None:
            print("⚠️ No data available. Fetching recent data...")
            self.last_data = self.get_data(start_date="2023-01-01")
        
        # Prepare data
        data_with_target = self.create_target(self.last_data)
        images, volumes = self.encode_candles(data_with_target)
        
        # Reshape for model
        X_img = images.reshape(-1, 1, images.shape[1], images.shape[2], images.shape[3])
        X_vol = self.scaler.fit_transform(volumes)
        y = data_with_target['target'].iloc[5:].values
        
        # Train if needed
        if self.model is None and train_if_missing:
            print("🤖 Model not trained. Starting training...")
            self.train_model(X_img, X_vol, y, n_splits=3, epochs=10)
        elif self.model is None:
            raise ValueError("Model not trained. Call train_model() first or set train_if_missing=True")
        
        # Make prediction on latest data
        last_img = X_img[-1:]
        last_vol = X_vol[-1:]
        
        prediction_prob = self.model.predict([last_img, last_vol], verbose=0)[0][0]
        confidence = abs(prediction_prob - 0.5) * 2  # Convert to 0-1 scale
        
        print(f"\n📈 Prediction Results:")
        print(f"   Probability: {prediction_prob:.3f}")
        print(f"   Confidence: {confidence:.1%}")
        
        if prediction_prob > 0.5:
            return "Buy", prediction_prob, confidence
        else:
            return "Sell", prediction_prob, confidence
    
    def save_model(self, filepath='blenns_model.h5'):
        """Save trained model"""
        if self.model:
            self.model.save(filepath)
            print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='blenns_model.h5'):
        """Load trained model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"✅ Model loaded from {filepath}")
