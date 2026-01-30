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
        self.trained = False
        
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
        from datetime import datetime
        import pandas as pd
        
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

            # yfinance returns data with Date as index
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Check column names
            print(f"Raw columns from yfinance: {list(data.columns)}")
            
            # Handle different column naming
            date_column = None
            for col in data.columns:
                if isinstance(col, str) and 'date' in col.lower():
                    date_column = col
                    break
                elif col == 'Date' or col == 'Datetime':
                    date_column = col
            
            if date_column:
                data = data.rename(columns={date_column: 'date'})
            else:
                # If no date column found, check if index was reset
                if 'index' in data.columns:
                    data = data.rename(columns={'index': 'date'})
                else:
                    # Create date column from index
                    data = data.reset_index().rename(columns={'index': 'date'})
            
            # Convert date column to datetime
            data['date'] = pd.to_datetime(data['date'])
            
            # Standardize other column names
            column_mapping = {
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume'
            }
            
            # Rename columns
            for old_name, new_name in column_mapping.items():
                if old_name in data.columns:
                    data = data.rename(columns={old_name: new_name})
            
            # Validate required columns
            required_ohlc = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_ohlc if col not in data.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {list(data.columns)}")
            
            # Fill missing values if any
            data[required_ohlc] = data[required_ohlc].ffill().bfill()
            
            # Ensure volume column exists
            if 'volume' not in data.columns:
                data['volume'] = 0
            else:
                data['volume'] = data['volume'].fillna(0)
            
            # Sort by date
            data = data.sort_values('date').reset_index(drop=True)
            
            # Apply BFC
            print("🔄 Applying BFC processing...")
            bfc_data = self.compute_bfc(data, **self.bfc_params)
            
            # Update OHLC with BFC values
            data[['open', 'high', 'low', 'close']] = bfc_data.values
            
            # Store the data
            self.last_data = data
            
            print(f"✅ Successfully loaded {len(data)} rows of data")
            print(f"   Date range: {data['date'].min().date()} to {data['date'].max().date()}")
            print(f"   Columns: {list(data.columns)}")
            
            return data

        except Exception as e:
            print(f"❌ Data loading failed: {str(e)}")
            print("Trying alternative data source...")
            
            # Try alternative approach with different parameters
            try:
                data = yf.download(
                    tickers=self.symbol,
                    period="1y",  # Try with period instead of dates
                    interval=interval,
                    progress=False
                )
                
                data = data.reset_index()
                data = data.rename(columns={'Date': 'date'})
                data['date'] = pd.to_datetime(data['date'])
                
                # Standardize columns
                for old_name, new_name in column_mapping.items():
                    if old_name in data.columns:
                        data = data.rename(columns={old_name: new_name})
                
                # Apply BFC
                bfc_data = self.compute_bfc(data, **self.bfc_params)
                data[['open', 'high', 'low', 'close']] = bfc_data.values
                
                if 'volume' not in data.columns:
                    data['volume'] = 0
                
                self.last_data = data
                print(f"✅ Loaded {len(data)} rows using alternative method")
                return data
                
            except Exception as e2:
                print(f"❌ Alternative method also failed: {e2}")
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
        
        # Calculate class distribution
        buy_signals = data['target'].sum()
        total_samples = len(data)
        
        print(f"✅ Created target with lookahead={lookahead}")
        print(f"   Buy signals: {buy_signals}/{total_samples} ({buy_signals/total_samples:.1%})")
        print(f"   Sell signals: {total_samples - buy_signals}/{total_samples} ({(total_samples - buy_signals)/total_samples:.1%})")
        
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

        print(f"🖼️ Encoding candlestick images (window={window_size})...")
        
        # Ensure we have enough data
        if len(data) <= window_size:
            raise ValueError(f"Need at least {window_size + 1} data points, but only have {len(data)}")
        
        for i in range(window_size, len(data)):
            # Get window of data
            subset = data.iloc[i-window_size:i].copy()
            
            # Create sequential x-values for plotting
            subset = subset.reset_index(drop=True)
            subset['x_pos'] = np.arange(len(subset))
            
            # Create figure
            fig, ax = plt.subplots(figsize=(img_size/dpi, img_size/dpi), dpi=dpi)
            
            # Prepare data for candlestick
            candlestick_data = []
            for idx, row in subset.iterrows():
                candlestick_data.append([
                    row['x_pos'],
                    row['open'],
                    row['high'],
                    row['low'],
                    row['close']
                ])
            
            # Plot candlestick
            candlestick_ohlc(
                ax, 
                candlestick_data,
                width=0.6, 
                colorup='green', 
                colordown='red',
                alpha=1.0
            )
            
            # Format plot for clean image
            ax.set_xlim(-0.5, window_size - 0.5)
            
            # Get y-axis limits from data
            y_min = subset[['low']].min().min() * 0.99
            y_max = subset[['high']].max().max() * 1.01
            ax.set_ylim(y_min, y_max)
            
            # Remove axes and borders
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', 
                       bbox_inches='tight', 
                       pad_inches=0,
                       facecolor='white',
                       dpi=dpi)
            buf.seek(0)
            
            # Convert to array
            img = Image.open(buf).resize((img_size, img_size)).convert('RGB')
            encoded_images.append(np.array(img) / 255.0)
            
            # Get volume for current candle (not the window)
            if 'volume' in data.columns:
                volumes.append(float(data.iloc[i]['volume']))
            else:
                volumes.append(0.0)
            
            plt.close(fig)
            
            # Progress indicator
            if i % 50 == 0 and i > window_size:
                print(f"  Processed {i-window_size}/{len(data)-window_size} images...")

        print(f"✅ Encoded {len(encoded_images)} images of shape {encoded_images[0].shape}")
        print(f"   Volume array shape: {len(volumes)}")
        
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
        
        # First conv block
        x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(img_input)
        x = TimeDistributed(MaxPooling2D((2, 2)))(x)
        x = TimeDistributed(Dropout(0.2))(x)
        
        # Second conv block
        x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
        x = TimeDistributed(MaxPooling2D((2, 2)))(x)
        x = TimeDistributed(Dropout(0.3))(x)
        
        # Third conv block
        x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
        x = TimeDistributed(MaxPooling2D((2, 2)))(x)
        x = TimeDistributed(Flatten())(x)

        # Temporal processing with LSTM
        x = LSTM(64, return_sequences=True)(x)
        x = Dropout(0.4)(x)
        
        # Attention mechanism
        attention = Attention()([x, x])
        x = tf.keras.layers.GlobalAveragePooling1D()(attention)

        # Volume input branch
        vol_input = Input(shape=(1,))
        vol_dense = Dense(16, activation='relu')(vol_input)
        vol_dense = Dropout(0.2)(vol_dense)

        # Feature fusion
        x = concatenate([x, vol_dense])

        # Dense layers for prediction
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(1, activation='sigmoid')(x)

        # Create model
        model = Model(inputs=[img_input, vol_input], outputs=output)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        print("✅ Model built successfully")
        print(f"   Total parameters: {model.count_params():,}")
        
        return model
    
    def train_model(self, X_img, X_vol, y, n_splits=5, epochs=50, batch_size=32, validation_split=0.2):
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
        # Validate input shapes
        if len(X_img) != len(X_vol) or len(X_img) != len(y):
            raise ValueError(f"Data length mismatch: X_img={len(X_img)}, X_vol={len(X_vol)}, y={len(y)}")
        
        print(f"🏋️ Training model on {len(X_img)} samples...")
        print(f"   Image shape: {X_img.shape}")
        print(f"   Volume shape: {X_vol.shape}")
        print(f"   Target shape: {y.shape}")
        
        # Reshape images if needed (for single image per sample)
        if len(X_img.shape) == 4:  # (n_samples, height, width, channels)
            X_img = X_img.reshape(-1, 1, X_img.shape[1], X_img.shape[2], X_img.shape[3])
            print(f"   Reshaped images to: {X_img.shape}")
        
        # Normalize volume
        X_vol = self.scaler.fit_transform(X_vol)
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model(input_shape=X_img.shape[1:])
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=min(n_splits, len(X_img)//10))
        print(f"   Using {tscv.n_splits}-fold time series cross-validation")
        
        metrics = {
            'val_acc': [], 'val_auc': [], 'val_loss': [],
            'val_precision': [], 'val_recall': [],
            'train_acc': [], 'train_auc': [], 'train_loss': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_img), 1):
            print(f"\n📊 Fold {fold}/{tscv.n_splits}")
            print(f"   Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
            
            # Build fresh model for each fold
            self.model = self.build_model(input_shape=X_img.shape[1:])
            
            # Early stopping callback
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
            
            # Reduce learning rate on plateau
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
            
            # Train
            history = self.model.fit(
                [X_img[train_idx], X_vol[train_idx]], 
                y[train_idx],
                validation_data=([X_img[val_idx], X_vol[val_idx]], y[val_idx]),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                callbacks=[early_stopping, reduce_lr]
            )
            
            # Store metrics
            if 'val_accuracy' in history.history:
                metrics['val_acc'].append(history.history['val_accuracy'][-1])
            if 'val_auc' in history.history:
                metrics['val_auc'].append(history.history['val_auc'][-1])
            if 'val_loss' in history.history:
                metrics['val_loss'].append(history.history['val_loss'][-1])
            if 'val_precision' in history.history:
                metrics['val_precision'].append(history.history['val_precision'][-1])
            if 'val_recall' in history.history:
                metrics['val_recall'].append(history.history['val_recall'][-1])
            
            # Store training metrics
            if 'accuracy' in history.history:
                metrics['train_acc'].append(history.history['accuracy'][-1])
            if 'auc' in history.history:
                metrics['train_auc'].append(history.history['auc'][-1])
            if 'loss' in history.history:
                metrics['train_loss'].append(history.history['loss'][-1])
            
            self.history = history
        
        # Print summary
        print(f"\n✅ Training complete!")
        print(f"\n📈 Validation Metrics (Average):")
        if metrics['val_acc']:
            print(f"   Accuracy: {np.mean(metrics['val_acc']):.3f} (±{np.std(metrics['val_acc']):.3f})")
        if metrics['val_auc']:
            print(f"   AUC: {np.mean(metrics['val_auc']):.3f} (±{np.std(metrics['val_auc']):.3f})")
        if metrics['val_precision']:
            print(f"   Precision: {np.mean(metrics['val_precision']):.3f} (±{np.std(metrics['val_precision']):.3f})")
        if metrics['val_recall']:
            print(f"   Recall: {np.mean(metrics['val_recall']):.3f} (±{np.std(metrics['val_recall']):.3f})")
        
        self.trained = True
        return metrics
    
    def predict_next_day(self, train_if_missing=True, confidence_threshold=0.6):
        """
        Generate prediction for the next trading day
        
        Parameters:
        -----------
        train_if_missing : bool
            Train model if not already trained
        confidence_threshold : float
            Minimum confidence to trust prediction
            
        Returns:
        --------
        tuple: (signal, probability, confidence)
        """
        if self.last_data is None:
            print("⚠️ No data available. Fetching recent data...")
            self.last_data = self.get_data(start_date="2023-01-01")
        
        # Prepare data
        data_with_target = self.create_target(self.last_data, lookahead=1)
        
        # Check if we have enough data
        if len(data_with_target) < 10:
            raise ValueError(f"Not enough data for prediction. Need at least 10 samples, have {len(data_with_target)}")
        
        # Encode candles
        window_size = min(5, len(data_with_target) - 1)  # Adjust window size if needed
        images, volumes = self.encode_candles(data_with_target, window_size=window_size)
        
        # Reshape for model
        X_img = images.reshape(-1, 1, images.shape[1], images.shape[2], images.shape[3])
        X_vol = self.scaler.fit_transform(volumes)
        y = data_with_target['target'].iloc[window_size:].values
        
        # Ensure shapes match
        if len(X_img) != len(y):
            min_len = min(len(X_img), len(y))
            X_img = X_img[:min_len]
            X_vol = X_vol[:min_len]
            y = y[:min_len]
        
        # Train if needed
        if self.model is None and train_if_missing:
            print("🤖 Model not trained. Starting training...")
            self.train_model(X_img, X_vol, y, n_splits=min(3, len(X_img)//20), epochs=20)
        elif self.model is None:
            raise ValueError("Model not trained. Call train_model() first or set train_if_missing=True")
        
        # Make prediction on latest data
        last_img = X_img[-1:]
        last_vol = X_vol[-1:]
        
        prediction_prob = self.model.predict([last_img, last_vol], verbose=0)[0][0]
        confidence = abs(prediction_prob - 0.5) * 2  # Convert to 0-1 scale
        
        # Determine signal
        if prediction_prob > 0.5:
            signal = "Buy"
        else:
            signal = "Sell"
        
        print(f"\n📈 Prediction Results:")
        print(f"   Signal: {signal}")
        print(f"   Probability: {prediction_prob:.3f}")
        print(f"   Confidence: {confidence:.1%}")
        
        if confidence < confidence_threshold:
            print(f"   ⚠️ Low confidence prediction (below {confidence_threshold:.0%} threshold)")
            signal = f"{signal} (Low Confidence)"
        
        return signal, prediction_prob, confidence
    
    def predict_batch(self, X_img, X_vol):
        """
        Make predictions on a batch of data
        
        Parameters:
        -----------
        X_img : numpy.ndarray
            Batch of image data
        X_vol : numpy.ndarray
            Batch of volume data
            
        Returns:
        --------
        numpy.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first")
        
        # Reshape if needed
        if len(X_img.shape) == 4:
            X_img = X_img.reshape(-1, 1, X_img.shape[1], X_img.shape[2], X_img.shape[3])
        
        # Normalize volume
        X_vol = self.scaler.transform(X_vol)
        
        # Make predictions
        predictions = self.model.predict([X_img, X_vol], verbose=0)
        
        return predictions
    
    def evaluate_model(self, X_img, X_vol, y):
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X_img : numpy.ndarray
            Image data
        X_vol : numpy.ndarray
            Volume data
        y : numpy.ndarray
            True labels
            
        Returns:
        --------
        dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first")
        
        # Reshape if needed
        if len(X_img.shape) == 4:
            X_img = X_img.reshape(-1, 1, X_img.shape[1], X_img.shape[2], X_img.shape[3])
        
        # Normalize volume
        X_vol = self.scaler.transform(X_vol)
        
        # Evaluate
        evaluation = self.model.evaluate([X_img, X_vol], y, verbose=0)
        
        # Get metric names
        metric_names = self.model.metrics_names
        
        # Create results dictionary
        results = dict(zip(metric_names, evaluation))
        
        print("📊 Model Evaluation:")
        for metric, value in results.items():
            print(f"   {metric}: {value:.4f}")
        
        return results
    
    def save_model(self, filepath='blenns_model.h5'):
        """Save trained model"""
        if self.model:
            self.model.save(filepath)
            print(f"✅ Model saved to {filepath}")
            return True
        else:
            print("⚠️ No model to save")
            return False
    
    def load_model(self, filepath='blenns_model.h5'):
        """Load trained model"""
        try:
            self.model = tf.keras.models.load_model(filepath)
            print(f"✅ Model loaded from {filepath}")
            self.trained = True
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def get_feature_importance(self, X_img_sample, X_vol_sample):
        """
        Get feature importance using gradient-based method
        
        Parameters:
        -----------
        X_img_sample : numpy.ndarray
            Sample image data
        X_vol_sample : numpy.ndarray
            Sample volume data
            
        Returns:
        --------
        dict: Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Ensure correct shape
        if len(X_img_sample.shape) == 4:
            X_img_sample = X_img_sample.reshape(1, 1, X_img_sample.shape[1], 
                                               X_img_sample.shape[2], X_img_sample.shape[3])
        
        X_vol_sample = self.scaler.transform(X_vol_sample.reshape(-1, 1))
        
        # Get gradients
        with tf.GradientTape() as tape:
            tape.watch([X_img_sample, X_vol_sample])
            predictions = self.model([X_img_sample, X_vol_sample])
        
        gradients = tape.gradient(predictions, [X_img_sample, X_vol_sample])
        
        # Calculate importance scores
        img_grad = np.mean(np.abs(gradients[0].numpy()))
        vol_grad = np.mean(np.abs(gradients[1].numpy()))
        
        importance = {
            'image_features': float(img_grad),
            'volume': float(vol_grad),
            'image_to_volume_ratio': float(img_grad / vol_grad) if vol_grad != 0 else float('inf')
        }
        
        return importance
