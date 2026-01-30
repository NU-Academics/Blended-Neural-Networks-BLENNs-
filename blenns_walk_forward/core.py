# -*- coding: utf-8 -*-
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

class BLENNSWalkForward:
    def __init__(self, symbol="TLRY", bfc_params=None):
        self.symbol = symbol
        self.model = None
        self.scaler = MinMaxScaler()
        self.bfc_params = bfc_params or {'alpha': 0.2, 'R': 0.1**2, 'Q': 1e-5}
        self.last_data = None

    def compute_bfc(self, df, alpha=0.2, R=0.1**2, Q=1e-5):
        """Enhanced Blenns Filter Candles"""
        df = df.copy()

        # EMA Smoothing Stage
        df['O_ema'] = self.exponential_moving_average(df['open'].values, alpha)
        df['H_ema'] = self.exponential_moving_average(df['high'].values, alpha)
        df['L_ema'] = self.exponential_moving_average(df['low'].values, alpha)
        df['C_ema'] = self.exponential_moving_average(df['close'].values, alpha)

        # Modified Heikin-Ashi Stage
        bfc = pd.DataFrame(index=df.index)
        bfc['HA_Close'] = (df['O_ema'] + df['H_ema'] + df['L_ema'] + df['C_ema']) / 4

        # Vectorized Heikin-Ashi Open calculation
        ha_open = np.zeros(len(df))
        ha_open[0] = (df['O_ema'].iloc[0] + df['C_ema'].iloc[0]) / 2
        for i in range(1, len(df)):
            ha_open[i] = (ha_open[i-1] + bfc['HA_Close'].iloc[i-1]) / 2
        bfc['HA_Open'] = ha_open

        bfc['HA_High'] = np.maximum.reduce([df['H_ema'].values, bfc['HA_Open'].values, bfc['HA_Close'].values])
        bfc['HA_Low'] = np.minimum.reduce([df['L_ema'].values, bfc['HA_Open'].values, bfc['HA_Close'].values])

        # Kalman Filter Stage
        bfc['BFC_Close'] = self.kalman_filter(bfc['HA_Close'].values, R=R, Q=Q)
        bfc['BFC_Open'] = self.kalman_filter(bfc['HA_Open'].values, R=R, Q=Q)

        # Ensure High/Low consistency
        bfc['BFC_High'] = np.maximum.reduce([df['H_ema'].values, bfc['BFC_Open'].values, bfc['BFC_Close'].values])
        bfc['BFC_Low'] = np.minimum.reduce([df['L_ema'].values, bfc['BFC_Open'].values, bfc['BFC_Close'].values])

        return bfc[['BFC_Open', 'BFC_High', 'BFC_Low', 'BFC_Close']]

    def exponential_moving_average(self, data, alpha):
        """Vectorized Exponential Moving Average"""
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result

    def kalman_filter(self, observations, R=0.1**2, Q=1e-5):
        """Enhanced Kalman Filter with configurable noise parameters"""
        n = len(observations)
        filtered = np.zeros(n)
        P = np.zeros(n)

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

    def get_data(self, start_date="2010-01-01", end_date=None, interval="1d"):
        """Fetch and process historical data with universal BFC application"""
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
            data = data.rename(columns={k:v for k,v in column_map.items() if k in data.columns})

            # Validate required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_cols):
                missing = [col for col in required_cols if col not in data.columns]
                raise ValueError(f"Missing required columns: {missing}")

            # Apply BFC to all instrument types
            bfc_data = self.compute_bfc(data, **self.bfc_params)
            data[['open', 'high', 'low', 'close']] = bfc_data

            self.last_data = data
            return data

        except Exception as e:
            print(f"Data loading failed: {str(e)}")
            if 'data' in locals():
                print("Columns received:", data.columns.tolist())
            raise

    def create_target(self, data, lookahead=1):
        """Create prediction target with configurable lookahead"""
        data = data.copy()

        # Validate input
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column to create target")

        # Create target
        data['target'] = (data['close'].shift(-lookahead) > data['close']).astype(int)

        # Drop rows where target is NA (end of series)
        data = data.dropna(subset=['target'])

        return data

    def encode_candles(self, data, window_size=5, img_size=64, dpi=32):
        """Generate candlestick images from BFC-processed data"""
        from matplotlib.dates import date2num
        from mplfinance.original_flavor import candlestick_ohlc
        import matplotlib.pyplot as plt
        from PIL import Image
        import io

        # Validate input
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in data.columns]
            raise ValueError(f"Missing required columns: {missing}")

        encoded_images = []
        volumes = []

        for index in range(window_size, len(data)):
            subset = data.iloc[index-window_size:index+1].copy()
            subset['date_num'] = subset['date'].apply(date2num)

            fig, ax = plt.subplots(figsize=(2, 2), dpi=dpi)
            candlestick_ohlc(ax, subset[['date_num', 'open', 'high', 'low', 'close']].values,
                             width=0.6, colorup='g', colordown='r')
            plt.axis('off')

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            img = Image.open(buf).resize((img_size, img_size)).convert('RGB')
            plt.close(fig)

            encoded_images.append(np.array(img) / 255.0)
            volumes.append(float(data.iloc[index]['volume']))

        return np.array(encoded_images, dtype=np.float32), np.array(volumes, dtype=np.float32).reshape(-1, 1)

    def build_model(self, input_shape=(1, 64, 64, 3)):
        """Enhanced BLENNS model architecture"""
        # Image processing branch
        img_input = Input(shape=input_shape)
        x = TimeDistributed(Conv2D(32, (3,3), activation='relu', padding='same'))(img_input)
        x = TimeDistributed(MaxPooling2D(2,2))(x)
        x = TimeDistributed(Dropout(0.3))(x)
        x = TimeDistributed(Conv2D(64, (3,3), activation='relu', padding='same'))(x)
        x = TimeDistributed(MaxPooling2D(2,2))(x)
        x = TimeDistributed(Flatten())(x)

        # Temporal processing with attention
        x = LSTM(64, return_sequences=True)(x)
        x = Dropout(0.4)(x)
        attn = Attention()([x, x])

        # Volume input branch
        vol_input = Input(shape=(1,))

        # Feature fusion
        x = concatenate([Flatten()(attn), vol_input])

        # Prediction head
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[img_input, vol_input], outputs=output)
        model.compile(optimizer=Adam(0.001),
                     loss='binary_crossentropy',
                     metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
        return model

    def train_model(self, X_img, X_vol, y, n_splits=5, epochs=50, batch_size=32):
        """Enhanced walk-forward validation with metrics tracking"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics = {'val_acc': [], 'val_auc': [], 'val_loss': []}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_img)):
            print(f"\nTraining fold {fold+1}/{n_splits}...")
            self.model = self.build_model()

            history = self.model.fit(
                [X_img[train_idx], X_vol[train_idx]], y[train_idx],
                validation_data=([X_img[val_idx], X_vol[val_idx]], y[val_idx]),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )

            # Store metrics
            metrics['val_acc'].append(history.history['val_accuracy'][-1])
            metrics['val_auc'].append(history.history['val_auc'][-1])
            metrics['val_loss'].append(history.history['val_loss'][-1])

        return metrics

    def predict_next_day(self, train_if_missing=True):
        """Generate prediction for the next trading day"""
        if self.last_data is None:
            self.last_data = self.get_data()

        data = self.create_target(self.last_data)
        images, volumes = self.encode_candles(data)
        X_img = images.reshape(-1, 1, 64, 64, 3)
        X_vol = self.scaler.fit_transform(volumes)

        if self.model is None and train_if_missing:
            print("Training model first...")
            self.train_model(X_img, X_vol, data['target'].iloc[5:].values, n_splits=3, epochs=30)
        elif self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        last_img, last_vol = X_img[-1:], X_vol[-1:]
        prediction = self.model.predict([last_img, last_vol], verbose=0)[0][0]
        return "Buy" if prediction > 0.5 else "Sell"
