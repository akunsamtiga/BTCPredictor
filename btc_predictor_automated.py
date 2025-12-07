"""
Bitcoin Price Predictor - FIXED FOR ULTRA SHORT TIMEFRAMES
Special handling for 5-minute predictions
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
import pickle
import os
import logging
from functools import lru_cache
from typing import Dict, Optional, List

warnings.filterwarnings('ignore')

from config import DATA_CONFIG, API_CONFIG, MODEL_CONFIG, get_timeframe_category
from firebase_manager import FirebaseManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
    import tensorflow as tf
    from tensorflow import keras
    from keras.models import Sequential, load_model
    from keras.layers import LSTM, Dense, Dropout, Bidirectional
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from keras.optimizers import Adam
    ML_AVAILABLE = True
    logger.info("✅ Machine Learning libraries loaded")
except ImportError as e:
    ML_AVAILABLE = False
    logger.warning(f"⚠️ ML libraries not available: {e}")

# Technical Indicators
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, periods=20, std_dev=2):
    sma = data.rolling(window=periods).mean()
    std = data.rolling(window=periods).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def add_technical_indicators(df):
    try:
        df = df.copy()
        
        df['rsi'] = calculate_rsi(df['price'], 14)
        df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['price'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df['price'])
        
        df['ema_9'] = df['price'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['price'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['price'].ewm(span=50, adjust=False).mean()
        df['sma_20'] = df['price'].rolling(window=20).mean()
        df['sma_50'] = df['price'].rolling(window=50).mean()
        
        lowest_low = df['low'].rolling(window=14).min()
        highest_high = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['price'] - lowest_low) / (highest_high - lowest_low))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['price'].shift())
        low_close = np.abs(df['low'] - df['price'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        df['momentum'] = df['price'] - df['price'].shift(10)
        df['rate_of_change'] = ((df['price'] - df['price'].shift(10)) / df['price'].shift(10)) * 100
        
        for period in [1, 3, 5, 10]:
            df[f'price_change_{period}'] = df['price'].pct_change(period) * 100
            df[f'price_lag_{period}'] = df['price'].shift(period)
            df[f'volume_lag_{period}'] = df['volume'].shift(period)
        
        for window in [5, 10, 20, 50]:
            df[f'price_rolling_mean_{window}'] = df['price'].rolling(window).mean()
            df[f'price_rolling_std_{window}'] = df['price'].rolling(window).std()
            df[f'volume_rolling_mean_{window}'] = df['volume'].rolling(window).mean()
        
        df['volatility_5'] = df['price'].rolling(window=5).std()
        df['volatility_20'] = df['price'].rolling(window=20).std()
        
        df['price_above_sma20'] = (df['price'] > df['sma_20']).astype(int)
        df['price_above_sma50'] = (df['price'] > df['sma_50']).astype(int)
        df['ema_trend'] = (df['ema_9'] > df['ema_21']).astype(int)
        df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        logger.info("✅ Technical indicators calculated")
        return df
        
    except Exception as e:
        logger.error(f"❌ Error calculating indicators: {e}")
        return df

# Data Fetching
def get_bitcoin_data_realtime(days=7, interval='hour'):
    config = {
        'minute': {
            'url': 'https://min-api.cryptocompare.com/data/v2/histominute',
            'max_limit': 2000,
            'points_per_day': 1440
        },
        'hour': {
            'url': 'https://min-api.cryptocompare.com/data/v2/histohour',
            'max_limit': 2000,
            'points_per_day': 24
        },
        'day': {
            'url': 'https://min-api.cryptocompare.com/data/v2/histoday',
            'max_limit': 2000,
            'points_per_day': 1
        }
    }
    
    if interval not in config:
        logger.error(f"❌ Invalid interval: {interval}")
        return None
    
    cfg = config[interval]
    url = cfg['url']
    points_needed = int(days * cfg['points_per_day'])
    
    max_retries = API_CONFIG['max_retries']
    retry_delay = API_CONFIG['retry_delay']
    
    for attempt in range(max_retries):
        try:
            params = {
                'fsym': 'BTC',
                'tsym': 'USD',
                'limit': min(points_needed - 1, cfg['max_limit'] - 1)
            }
            
            if DATA_CONFIG['cryptocompare_api_key']:
                params['api_key'] = DATA_CONFIG['cryptocompare_api_key']
            
            response = requests.get(url, params=params, timeout=API_CONFIG['timeout'])
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('Response') != 'Success':
                logger.error(f"❌ API Error: {data.get('Message')}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                return None
            
            prices = data['Data']['Data']
            
            if not prices:
                logger.error("❌ No data returned")
                return None
            
            df = pd.DataFrame(prices)
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
            
            df = df.rename(columns={
                'high': 'high',
                'low': 'low',
                'open': 'open',
                'close': 'price',
                'volumefrom': 'volume'
            })
            
            df = df[['datetime', 'open', 'high', 'low', 'price', 'volume']]
            df = df.sort_values('datetime', ascending=False).reset_index(drop=True)
            
            if len(df) < DATA_CONFIG['min_data_points']:
                logger.warning(f"⚠️ Insufficient data: {len(df)} points")
                return None
            
            logger.info(f"✅ Fetched {len(df):,} data points ({interval})")
            return df
        
        except Exception as e:
            logger.warning(f"⚠️ Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
            else:
                logger.error(f"❌ Failed after {max_retries} attempts")
                return None
    
    return None

def get_current_btc_price():
    try:
        url = "https://min-api.cryptocompare.com/data/price"
        params = {'fsym': 'BTC', 'tsyms': 'USD'}
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        return data.get('USD')
        
    except Exception as e:
        logger.error(f"❌ Error getting current price: {e}")
        return None

# ML Predictor
class BitcoinMLPredictor:
    def __init__(self):
        self.lstm_model = None
        self.rf_model = None
        self.gb_model = None
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        self.sequence_length = MODEL_CONFIG['lstm']['sequence_length']
        self.feature_columns = []
        self.is_trained = False
        self.metrics = {}
        self.last_training = None
    
    def prepare_features(self, df):
        feature_cols = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_position', 'ema_9', 'ema_21', 'ema_50',
            'stoch_k', 'stoch_d', 'atr',
            'volume_ratio', 'momentum', 'rate_of_change',
            'volatility_5',
            'price_above_sma20', 'price_above_sma50', 'ema_trend'
        ]
        
        for lag in [1, 3, 5, 10]:
            feature_cols.extend([f'price_lag_{lag}', f'volume_lag_{lag}', f'price_change_{lag}'])
        
        for window in [5, 10, 20, 50]:
            feature_cols.extend([
                f'price_rolling_mean_{window}',
                f'price_rolling_std_{window}',
                f'volume_rolling_mean_{window}'
            ])
        
        available_features = [col for col in feature_cols if col in df.columns]
        self.feature_columns = available_features
        
        return df[available_features].copy()
    
    def create_sequences(self, data, target, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(target[i + sequence_length])
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape):
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True, dropout=0.2), input_shape=input_shape),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
            Dropout(0.3),
            Bidirectional(LSTM(32, dropout=0.2)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae']
        )
        
        return model
    
    def train_models(self, df, epochs=50, batch_size=32):
        if not ML_AVAILABLE:
            logger.error("❌ ML libraries not available")
            return False
        
        try:
            logger.info("\n🤖 TRAINING MODELS...")
            
            df_clean = df.dropna().copy()
            df_clean = df_clean.sort_values('datetime', ascending=True).reset_index(drop=True)
            
            if len(df_clean) < DATA_CONFIG['min_data_points']:
                logger.error(f"❌ Insufficient data: {len(df_clean)}")
                return False
            
            features = self.prepare_features(df_clean)
            target = df_clean['price'].values
            
            scaled_features = self.feature_scaler.fit_transform(features)
            scaled_target = self.price_scaler.fit_transform(target.reshape(-1, 1)).flatten()
            
            split_idx = int(len(df_clean) * 0.8)
            
            # LSTM
            logger.info("📈 Training LSTM...")
            X_lstm, y_lstm = self.create_sequences(scaled_features, scaled_target, self.sequence_length)
            
            X_train_lstm = X_lstm[:split_idx - self.sequence_length]
            X_test_lstm = X_lstm[split_idx - self.sequence_length:]
            y_train_lstm = y_lstm[:split_idx - self.sequence_length]
            y_test_lstm = y_lstm[split_idx - self.sequence_length:]
            
            self.lstm_model = self.build_lstm_model((self.sequence_length, len(self.feature_columns)))
            
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=MODEL_CONFIG['lstm']['patience'],
                restore_best_weights=True,
                verbose=0
            )
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=0
            )
            
            os.makedirs(MODEL_CONFIG['model_save_path'], exist_ok=True)
            checkpoint = ModelCheckpoint(
                f"{MODEL_CONFIG['model_save_path']}/lstm_best.keras",
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
            
            self.lstm_model.fit(
                X_train_lstm, y_train_lstm,
                validation_data=(X_test_lstm, y_test_lstm),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop, reduce_lr, checkpoint],
                verbose=0
            )
            
            lstm_pred = self.lstm_model.predict(X_test_lstm, verbose=0)
            y_test_original = self.price_scaler.inverse_transform(y_test_lstm.reshape(-1, 1)).flatten()
            lstm_pred_original = self.price_scaler.inverse_transform(lstm_pred).flatten()
            
            lstm_mae = mean_absolute_error(y_test_original, lstm_pred_original)
            lstm_rmse = np.sqrt(mean_squared_error(y_test_original, lstm_pred_original))
            
            self.metrics['lstm'] = {
                'mae': float(lstm_mae),
                'rmse': float(lstm_rmse)
            }
            
            logger.info(f"✅ LSTM trained - MAE: ${lstm_mae:,.2f}, RMSE: ${lstm_rmse:,.2f}")
            
            # Random Forest
            logger.info("🌲 Training Random Forest...")
            
            y_class = (df_clean['price'].shift(-1) > df_clean['price']).astype(int)
            y_class = y_class[:-1]
            features_rf = scaled_features[:-1]
            
            X_train_rf = features_rf[:split_idx]
            X_test_rf = features_rf[split_idx:]
            y_train_rf = y_class[:split_idx]
            y_test_rf = y_class[split_idx:]
            
            self.rf_model = RandomForestClassifier(
                n_estimators=MODEL_CONFIG['rf']['n_estimators'],
                max_depth=MODEL_CONFIG['rf']['max_depth'],
                min_samples_split=MODEL_CONFIG['rf']['min_samples_split'],
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            
            self.rf_model.fit(X_train_rf, y_train_rf)
            
            rf_pred = self.rf_model.predict(X_test_rf)
            rf_accuracy = accuracy_score(y_test_rf, rf_pred)
            
            self.metrics['rf'] = {'accuracy': float(rf_accuracy)}
            
            logger.info(f"✅ RF trained - Accuracy: {rf_accuracy:.4f}")
            
            # Gradient Boosting
            logger.info("🚀 Training Gradient Boosting...")
            
            X_train_gb = features_rf[:split_idx]
            X_test_gb = features_rf[split_idx:]
            y_train_gb = scaled_target[:-1][:split_idx]
            y_test_gb = scaled_target[:-1][split_idx:]
            
            self.gb_model = GradientBoostingRegressor(
                n_estimators=MODEL_CONFIG['gb']['n_estimators'],
                learning_rate=MODEL_CONFIG['gb']['learning_rate'],
                max_depth=MODEL_CONFIG['gb']['max_depth'],
                random_state=42,
                verbose=0
            )
            
            self.gb_model.fit(X_train_gb, y_train_gb)
            
            gb_pred_scaled = self.gb_model.predict(X_test_gb)
            gb_pred = self.price_scaler.inverse_transform(gb_pred_scaled.reshape(-1, 1)).flatten()
            y_test_gb_original = self.price_scaler.inverse_transform(y_test_gb.reshape(-1, 1)).flatten()
            
            gb_mae = mean_absolute_error(y_test_gb_original, gb_pred)
            gb_rmse = np.sqrt(mean_squared_error(y_test_gb_original, gb_pred))
            
            self.metrics['gb'] = {
                'mae': float(gb_mae),
                'rmse': float(gb_rmse)
            }
            
            logger.info(f"✅ GB trained - MAE: ${gb_mae:,.2f}, RMSE: ${gb_rmse:,.2f}")
            
            self.is_trained = True
            self.last_training = datetime.now()
            
            logger.info("✅ ALL MODELS TRAINED SUCCESSFULLY!\n")
            
            self.save_models()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, df, timeframe_minutes):
        """FIXED prediction for ultra short timeframes"""
        if not self.is_trained:
            logger.warning("⚠️ Models not trained")
            return None
        
        try:
            category = get_timeframe_category(timeframe_minutes)
            
            # CRITICAL FIX: Use appropriate sequence length
            if category == 'ultra_short':
                sequence_length = MODEL_CONFIG['lstm']['ultra_short_sequence']
            else:
                sequence_length = self.sequence_length
            
            df_clean = df.dropna().copy()
            df_clean = df_clean.sort_values('datetime', ascending=True).reset_index(drop=True)
            
            # CRITICAL FIX: Ensure enough data
            min_required = sequence_length + 10
            if len(df_clean) < min_required:
                logger.warning(f"⚠️ Insufficient data: {len(df_clean)} < {min_required}")
                return None
            
            features = self.prepare_features(df_clean)
            scaled_features = self.feature_scaler.transform(features)
            
            current_price = df_clean.iloc[-1]['price']
            
            # LSTM with adjusted sequence
            lstm_input = scaled_features[-sequence_length:].reshape(1, sequence_length, -1)
            lstm_pred_scaled = self.lstm_model.predict(lstm_input, verbose=0)[0][0]
            lstm_pred = self.price_scaler.inverse_transform([[lstm_pred_scaled]])[0][0]
            
            # RF
            rf_input = scaled_features[-1:].reshape(1, -1)
            rf_direction = self.rf_model.predict(rf_input)[0]
            rf_proba = self.rf_model.predict_proba(rf_input)[0]
            rf_confidence = max(rf_proba) * 100
            
            # GB
            gb_pred_scaled = self.gb_model.predict(rf_input)[0]
            gb_pred = self.price_scaler.inverse_transform([[gb_pred_scaled]])[0][0]
            
            # CRITICAL FIX: Adjusted time factor for ultra short
            if category == 'ultra_short':
                time_factor = min(timeframe_minutes / 30, 1.0)  # Reduced scaling
            else:
                time_factor = min(timeframe_minutes / 60, 2)
            
            predicted_change_lstm = (lstm_pred - current_price) * time_factor
            predicted_change_gb = (gb_pred - current_price) * time_factor
            
            # CRITICAL FIX: Different weights for ultra short
            if category == 'ultra_short':
                lstm_weight = 0.35
                gb_weight = 0.35
                rf_weight = 0.30  # Higher RF weight for short term
            else:
                lstm_weight = 0.4
                gb_weight = 0.4
                rf_weight = 0.2
            
            ensemble_change = (
                lstm_weight * predicted_change_lstm +
                gb_weight * predicted_change_gb
            ) * (1 + (rf_confidence - 50) / 100)
            
            predicted_price = current_price + ensemble_change
            
            # Confidence calculation
            model_agreement = 0
            if (predicted_change_lstm > 0) == (predicted_change_gb > 0) == (rf_direction == 1):
                model_agreement = 3
            elif (predicted_change_lstm > 0) == (predicted_change_gb > 0) or \
                 (predicted_change_lstm > 0) == (rf_direction == 1) or \
                 (predicted_change_gb > 0) == (rf_direction == 1):
                model_agreement = 2
            else:
                model_agreement = 1
            
            # CRITICAL FIX: Adjusted confidence for ultra short
            if category == 'ultra_short':
                base_confidence = 45
                confidence = min(base_confidence + (model_agreement * 12) + (rf_confidence - 50) * 0.4, 85)
            else:
                confidence = min(50 + (model_agreement * 15) + (rf_confidence - 50) * 0.3, 95)
            
            trend = "CALL (Bullish)" if ensemble_change > 0 else "PUT (Bearish)"
            
            # CRITICAL FIX: Tighter range for ultra short
            volatility = df_clean['price'].tail(20).std()
            if category == 'ultra_short':
                range_multiplier = 0.5 * time_factor
            else:
                range_multiplier = time_factor
            
            price_range_low = predicted_price - volatility * range_multiplier
            price_range_high = predicted_price + volatility * range_multiplier
            
            return {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change': ensemble_change,
                'price_change_pct': (ensemble_change / current_price) * 100,
                'price_range_low': price_range_low,
                'price_range_high': price_range_high,
                'trend': trend,
                'confidence': confidence,
                'lstm_prediction': lstm_pred,
                'gb_prediction': gb_pred,
                'rf_direction': 'UP' if rf_direction == 1 else 'DOWN',
                'rf_confidence': rf_confidence,
                'model_agreement': model_agreement,
                'timeframe_minutes': timeframe_minutes,
                'volatility': volatility,
                'method': f'ML Ensemble ({category})',
                'model_metrics': self.metrics,
                'category': category,
                'time_factor': time_factor
            }
            
        except Exception as e:
            logger.error(f"❌ Error predicting: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_models(self):
        try:
            path = MODEL_CONFIG['model_save_path']
            os.makedirs(path, exist_ok=True)
            
            if self.lstm_model:
                self.lstm_model.save(f'{path}/lstm_model.keras')
            
            if self.rf_model:
                with open(f'{path}/rf_model.pkl', 'wb') as f:
                    pickle.dump(self.rf_model, f)
            
            if self.gb_model:
                with open(f'{path}/gb_model.pkl', 'wb') as f:
                    pickle.dump(self.gb_model, f)
            
            with open(f'{path}/scalers.pkl', 'wb') as f:
                pickle.dump({
                    'price_scaler': self.price_scaler,
                    'feature_scaler': self.feature_scaler,
                    'feature_columns': self.feature_columns,
                    'metrics': self.metrics,
                    'last_training': self.last_training
                }, f)
            
            logger.info(f"✅ Models saved to {path}/")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving models: {e}")
            return False
    
    def load_models(self):
        try:
            path = MODEL_CONFIG['model_save_path']
            
            if not os.path.exists(f'{path}/lstm_model.keras'):
                logger.warning("⚠️ Models not found")
                return False
            
            self.lstm_model = load_model(f'{path}/lstm_model.keras')
            
            with open(f'{path}/rf_model.pkl', 'rb') as f:
                self.rf_model = pickle.load(f)
            
            with open(f'{path}/gb_model.pkl', 'rb') as f:
                self.gb_model = pickle.load(f)
            
            with open(f'{path}/scalers.pkl', 'rb') as f:
                data = pickle.load(f)
                self.price_scaler = data['price_scaler']
                self.feature_scaler = data['feature_scaler']
                self.feature_columns = data['feature_columns']
                self.metrics = data.get('metrics', {})
                self.last_training = data.get('last_training')
            
            self.is_trained = True
            logger.info(f"✅ Models loaded from {path}/")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False
    
    def needs_retraining(self):
        if not self.is_trained or not self.last_training:
            return True
        
        time_since_training = (datetime.now() - self.last_training).total_seconds()
        return time_since_training > MODEL_CONFIG['auto_retrain_interval']