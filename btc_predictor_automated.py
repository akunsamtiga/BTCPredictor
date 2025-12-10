"""
Bitcoin Price Predictor - FIXED: NO BIAS (Buy/Sell Balanced)
Fixed ensemble logic to prevent always-buy predictions
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import pickle
import os
import requests
import time

from config import MODEL_CONFIG, DATA_CONFIG, VPS_CONFIG, get_timeframe_category
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import RobustScaler, StandardScaler
    import tensorflow as tf
    from tensorflow import keras
    from keras.models import Sequential, Model, load_model
    from keras.layers import (LSTM, Dense, Dropout, Bidirectional, BatchNormalization,
                             Input, MultiHeadAttention, LayerNormalization)
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from keras.optimizers import Adam
    from keras.regularizers import l2
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    logger.error(f"❌ ML libraries not available: {e}")


# ============================================================================
# CONSTANTS
# ============================================================================

API_MAX_LIMIT = 2000
API_RATE_LIMIT_DELAY = 1


# ============================================================================
# DATA FETCHING (unchanged)
# ============================================================================

def get_current_btc_price() -> Optional[float]:
    """Get current Bitcoin price"""
    api_key = DATA_CONFIG.get('cryptocompare_api_key')
    
    if not api_key:
        logger.error("❌ API key not configured")
        return None
    
    url = "https://min-api.cryptocompare.com/data/price"
    params = {'fsym': 'BTC', 'tsyms': 'USD', 'api_key': api_key}
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'USD' in data:
                return float(data['USD'])
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"❌ Failed to get price")
                return None
    
    return None


def _fetch_single_batch(endpoint: str, limit: int, to_timestamp: Optional[int] = None) -> Optional[Dict]:
    """Fetch single batch from API"""
    api_key = DATA_CONFIG.get('cryptocompare_api_key')
    
    url = f"https://min-api.cryptocompare.com/data/v2/{endpoint}"
    params = {
        'fsym': 'BTC',
        'tsym': 'USD',
        'limit': min(limit, API_MAX_LIMIT),
        'api_key': api_key
    }
    
    if to_timestamp:
        params['toTs'] = to_timestamp
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get('Response') == 'Error':
            logger.error(f"❌ API Error: {data.get('Message')}")
            return None
        
        if 'Data' not in data or 'Data' not in data['Data']:
            return None
        
        return data
        
    except Exception as e:
        logger.error(f"❌ API request failed: {e}")
        return None


def get_bitcoin_data_realtime(days: int = 7, interval: str = 'hour') -> Optional[pd.DataFrame]:
    """Fetch historical Bitcoin data with automatic pagination"""
    api_key = DATA_CONFIG.get('cryptocompare_api_key')
    
    if not api_key:
        logger.error("❌ API key not configured")
        return None
    
    if interval == 'minute':
        endpoint = 'histominute'
        total_points = days * 1440
    elif interval == 'hour':
        endpoint = 'histohour'
        total_points = days * 24
    elif interval == 'day':
        endpoint = 'histoday'
        total_points = days
    else:
        logger.error(f"❌ Invalid interval: {interval}")
        return None
    
    if total_points <= API_MAX_LIMIT:
        return _fetch_single_request(endpoint, total_points)
    else:
        return _fetch_multiple_requests(endpoint, total_points)


def _fetch_single_request(endpoint: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetch with single request"""
    data = _fetch_single_batch(endpoint, limit)
    
    if not data:
        return None
    
    candles = data['Data']['Data']
    
    if not candles:
        return None
    
    return _parse_candles_to_dataframe(candles)


def _fetch_multiple_requests(endpoint: str, total_points: int) -> Optional[pd.DataFrame]:
    """Fetch with multiple requests"""
    all_candles = []
    points_fetched = 0
    current_to_timestamp = None
    
    num_batches = (total_points + API_MAX_LIMIT - 1) // API_MAX_LIMIT
    
    for batch_num in range(num_batches):
        remaining_points = total_points - points_fetched
        batch_size = min(remaining_points, API_MAX_LIMIT)
        
        data = _fetch_single_batch(endpoint, batch_size, current_to_timestamp)
        
        if not data:
            break
        
        candles = data['Data']['Data']
        
        if not candles:
            break
        
        all_candles.extend(candles)
        points_fetched += len(candles)
        
        oldest_candle = candles[0]
        current_to_timestamp = oldest_candle['time'] - 1
        
        if batch_num < num_batches - 1:
            time.sleep(API_RATE_LIMIT_DELAY)
        
        if points_fetched >= total_points:
            break
    
    if not all_candles:
        return None
    
    unique_candles = []
    seen_times = set()
    
    for candle in all_candles:
        if candle['time'] not in seen_times:
            unique_candles.append(candle)
            seen_times.add(candle['time'])
    
    return _parse_candles_to_dataframe(unique_candles)


def _parse_candles_to_dataframe(candles: list) -> pd.DataFrame:
    """Parse candles to DataFrame"""
    df = pd.DataFrame(candles)
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={'close': 'price', 'volumefrom': 'volume'})
    df = df[['datetime', 'price', 'open', 'high', 'low', 'volume']]
    df = df.sort_values('datetime', ascending=False).reset_index(drop=True)
    df = df[df['price'] > 0].dropna(subset=['price'])
    
    return df


# ============================================================================
# FEATURE ENGINEERING (same as before, unchanged)
# ============================================================================

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators with CONSISTENT FEATURES"""
    try:
        df = df.copy()
        df = df.sort_values('datetime', ascending=True).reset_index(drop=True)
        
        data_points = len(df)
        logger.debug(f"Adding indicators for {data_points} data points")
        
        # Basic features
        df['returns'] = df['price'].pct_change()
        df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
        
        # Moving averages
        base_ma_periods = [7, 14, 20]
        extended_ma_periods = [50]
        
        for window in base_ma_periods:
            if data_points >= window * 2:
                df[f'sma_{window}'] = df['price'].rolling(window=window, min_periods=window//2).mean()
                df[f'ema_{window}'] = df['price'].ewm(span=window, adjust=False, min_periods=window//2).mean()
                df[f'price_to_sma_{window}'] = (df['price'] - df[f'sma_{window}']) / df['price']
        
        if data_points >= 150:
            for window in extended_ma_periods:
                df[f'sma_{window}'] = df['price'].rolling(window=window, min_periods=window//2).mean()
                df[f'ema_{window}'] = df['price'].ewm(span=window, adjust=False, min_periods=window//2).mean()
                df[f'price_to_sma_{window}'] = (df['price'] - df[f'sma_{window}']) / df['price']
        
        # MA cross
        if 'sma_7' in df.columns and 'sma_20' in df.columns:
            df['ma_cross_7_20'] = (df['sma_7'] > df['sma_20']).astype(int)
            df['ma_diff_7_20'] = (df['sma_7'] - df['sma_20']) / df['price']
        
        # RSI
        rsi_periods = [14, 21]
        for period in rsi_periods:
            if data_points >= period * 2:
                delta = df['price'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period//2).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period//2).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
                df[f'rsi_{period}_norm'] = (df[f'rsi_{period}'] - 50) / 50
        
        # MACD
        if data_points >= 52:
            ema_12 = df['price'].ewm(span=12, adjust=False, min_periods=6).mean()
            ema_26 = df['price'].ewm(span=26, adjust=False, min_periods=13).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False, min_periods=4).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        if data_points >= bb_period * 2:
            df[f'bb_middle_{bb_period}'] = df['price'].rolling(window=bb_period, min_periods=bb_period//2).mean()
            bb_std = df['price'].rolling(window=bb_period, min_periods=bb_period//2).std()
            df[f'bb_upper_{bb_period}'] = df[f'bb_middle_{bb_period}'] + (bb_std * 2)
            df[f'bb_lower_{bb_period}'] = df[f'bb_middle_{bb_period}'] - (bb_std * 2)
            df[f'bb_position_{bb_period}'] = (df['price'] - df[f'bb_lower_{bb_period}']) / (df[f'bb_upper_{bb_period}'] - df[f'bb_lower_{bb_period}'])
            df[f'bb_width_{bb_period}'] = (df[f'bb_upper_{bb_period}'] - df[f'bb_lower_{bb_period}']) / df[f'bb_middle_{bb_period}']
        
        # Stochastic
        if data_points >= 28:
            k_period = 14
            low_k = df['low'].rolling(window=k_period, min_periods=k_period//2).min()
            high_k = df['high'].rolling(window=k_period, min_periods=k_period//2).max()
            df['stoch_k'] = 100 * ((df['price'] - low_k) / (high_k - low_k))
            df['stoch_d'] = df['stoch_k'].rolling(window=3, min_periods=2).mean()
        
        # ATR
        atr_period = 14
        if data_points >= atr_period * 2:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['price'].shift())
            low_close = np.abs(df['low'] - df['price'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df[f'atr_{atr_period}'] = true_range.rolling(window=atr_period, min_periods=atr_period//2).mean()
            df[f'atr_pct_{atr_period}'] = df[f'atr_{atr_period}'] / df['price']
        
        # ADX
        if data_points >= 28:
            period = 14
            high_diff = df['high'].diff()
            low_diff = -df['low'].diff()
            
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
            
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['price'].shift())
            low_close = np.abs(df['low'] - df['price'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=period, min_periods=period//2).mean()
            
            plus_di = 100 * (plus_dm.rolling(window=period, min_periods=period//2).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period, min_periods=period//2).mean() / atr)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            df['adx'] = dx.rolling(window=period, min_periods=period//2).mean()
            df['plus_di'] = plus_di
            df['minus_di'] = minus_di
        
        # Volume indicators
        vol_period = 20
        if data_points >= vol_period * 2:
            df[f'volume_sma_{vol_period}'] = df['volume'].rolling(window=vol_period, min_periods=vol_period//2).mean()
            df[f'volume_ratio_{vol_period}'] = df['volume'] / df[f'volume_sma_{vol_period}']
        
        if data_points >= 20:
            df['obv'] = (np.sign(df['price'].diff()) * df['volume']).fillna(0).cumsum()
            df['obv_ema'] = df['obv'].ewm(span=20, adjust=False, min_periods=10).mean()
        
        # Momentum
        mom_periods = [10, 20]
        for period in mom_periods:
            if data_points >= period * 2:
                df[f'momentum_{period}'] = df['price'] - df['price'].shift(period)
                df[f'roc_{period}'] = df['price'].pct_change(period) * 100
        
        # Rolling statistics
        stat_period = 20
        if data_points >= stat_period * 2:
            df[f'rolling_std_{stat_period}'] = df['price'].rolling(window=stat_period, min_periods=stat_period//2).std()
            df[f'rolling_max_{stat_period}'] = df['price'].rolling(window=stat_period, min_periods=stat_period//2).max()
            df[f'rolling_min_{stat_period}'] = df['price'].rolling(window=stat_period, min_periods=stat_period//2).min()
            df[f'dist_from_max_{stat_period}'] = (df[f'rolling_max_{stat_period}'] - df['price']) / df['price']
            df[f'dist_from_min_{stat_period}'] = (df['price'] - df[f'rolling_min_{stat_period}']) / df['price']
            df[f'volatility_{stat_period}'] = df['returns'].rolling(window=stat_period, min_periods=stat_period//2).std()
        
        # Price patterns
        df['high_low_ratio'] = (df['high'] - df['low']) / df['price']
        df['close_position'] = (df['price'] - df['low']) / (df['high'] - df['low'])
        df['body_size'] = np.abs(df['price'] - df['open']) / df['price']
        
        if data_points >= 20:
            df['trend_strength'] = np.abs(df['returns'].rolling(window=20, min_periods=10).mean()) * 100
        
        # Fill NaNs
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].ffill()
        df[numeric_columns] = df[numeric_columns].bfill()
        df = df.fillna(0)
        
        # Sort back to most recent first
        df = df.sort_values('datetime', ascending=False).reset_index(drop=True)
        
        logger.info(f"✅ Indicators added: {len(df)} points")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error adding indicators: {e}")
        import traceback
        traceback.print_exc()
        return df


# ============================================================================
# PREDICTOR CLASS - FIXED BIAS
# ============================================================================

class ImprovedBitcoinPredictor:
    """
    FIXED: NO BIAS - Balanced Buy/Sell Predictions
    """
    
    def __init__(self):
        self.lstm_model = None
        self.rf_model = None
        self.gb_model = None
        self.price_scaler = RobustScaler()
        self.feature_scaler = StandardScaler()
        self.sequence_length = MODEL_CONFIG['lstm']['sequence_length']
        self.feature_columns = []
        self.is_trained = False
        self.metrics = {}
        self.last_training = None
        
        logger.info("🤖 Predictor initialized (FIXED - No Bias)")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select consistent predictive features"""
        
        priority_features = [
            'rsi_14', 'rsi_14_norm', 'rsi_21',
            'macd', 'macd_signal', 'macd_hist',
            'bb_position_20', 'bb_width_20',
            'stoch_k', 'stoch_d',
            'atr_pct_14', 'volatility_20',
            'adx', 'plus_di', 'minus_di',
            'volume_ratio_20', 'obv_ema',
            'price_to_sma_7', 'price_to_sma_14', 'price_to_sma_20',
            'ma_diff_7_20',
            'price_to_sma_50',
            'roc_10', 'roc_20', 'momentum_10',
            'rolling_std_20', 'dist_from_max_20', 'dist_from_min_20',
            'high_low_ratio', 'close_position', 'body_size',
            'trend_strength',
        ]
        
        available = [col for col in priority_features if col in df.columns]
        self.feature_columns = available
        
        return df[available].copy()
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray, sequence_length: int):
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(target[i + sequence_length])
        return np.array(X), np.array(y)
    
    def build_improved_lstm(self, input_shape: tuple) -> Model:
        """Build optimized LSTM"""
        
        inputs = Input(shape=input_shape)
        
        x = Bidirectional(LSTM(
            128,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.1,
            kernel_regularizer=l2(0.0005)
        ))(inputs)
        x = LayerNormalization()(x)
        
        attention_output = MultiHeadAttention(
            num_heads=4,
            key_dim=32,
            dropout=0.1
        )(x, x)
        x = LayerNormalization()(attention_output + x)
        
        x = Bidirectional(LSTM(
            64,
            return_sequences=False,
            dropout=0.2,
            kernel_regularizer=l2(0.0005)
        ))(x)
        x = LayerNormalization()(x)
        
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.0005))(x)
        x = Dropout(0.3)(x)
        x = LayerNormalization()(x)
        
        x = Dense(32, activation='relu', kernel_regularizer=l2(0.0005))(x)
        x = Dropout(0.2)(x)
        
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse'])
        
        return model
    
    def train_models(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 64) -> bool:
        """Train all models"""
        
        if not ML_AVAILABLE:
            logger.error("❌ ML libraries not available")
            return False
        
        try:
            logger.info("\n🤖 TRAINING MODELS...")
            
            df_clean = df.dropna().copy()
            df_clean = df_clean.sort_values('datetime', ascending=True).reset_index(drop=True)
            
            if len(df_clean) < 1000:
                logger.error(f"❌ Insufficient data: {len(df_clean)}")
                return False
            
            features = self.prepare_features(df_clean)
            target = df_clean['price'].values
            
            logger.info(f"📊 Features: {len(features.columns)}")
            
            scaled_features = self.feature_scaler.fit_transform(features)
            scaled_target = self.price_scaler.fit_transform(target.reshape(-1, 1)).flatten()
            
            tscv = TimeSeriesSplit(n_splits=5)
            splits = list(tscv.split(scaled_features))
            train_idx, test_idx = splits[-1]
            
            # LSTM
            logger.info("\n🔵 Training LSTM...")
            X_lstm, y_lstm = self.create_sequences(scaled_features, scaled_target, self.sequence_length)
            
            train_idx_lstm = train_idx[train_idx < len(X_lstm)]
            test_idx_lstm = test_idx[test_idx < len(X_lstm)]
            
            X_train_lstm = X_lstm[train_idx_lstm]
            X_test_lstm = X_lstm[test_idx_lstm]
            y_train_lstm = y_lstm[train_idx_lstm]
            y_test_lstm = y_lstm[test_idx_lstm]
            
            self.lstm_model = self.build_improved_lstm((self.sequence_length, len(self.feature_columns)))
            
            early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001, verbose=1)
            
            os.makedirs(MODEL_CONFIG['model_save_path'], exist_ok=True)
            checkpoint = ModelCheckpoint(
                f"{MODEL_CONFIG['model_save_path']}/lstm_best.keras",
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
            
            history = self.lstm_model.fit(
                X_train_lstm, y_train_lstm,
                validation_data=(X_test_lstm, y_test_lstm),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop, reduce_lr, checkpoint],
                verbose=1
            )
            
            lstm_pred = self.lstm_model.predict(X_test_lstm, verbose=0)
            y_test_original = self.price_scaler.inverse_transform(y_test_lstm.reshape(-1, 1)).flatten()
            lstm_pred_original = self.price_scaler.inverse_transform(lstm_pred).flatten()
            
            lstm_mae = mean_absolute_error(y_test_original, lstm_pred_original)
            lstm_rmse = np.sqrt(mean_squared_error(y_test_original, lstm_pred_original))
            
            if len(lstm_pred_original) > 1:
                min_len = min(len(lstm_pred_original), len(y_test_original)) - 1
                lstm_direction = np.mean(
                    np.sign(lstm_pred_original[:min_len] - y_test_original[:min_len]) == 
                    np.sign(y_test_original[1:min_len+1] - y_test_original[:min_len])
                ) * 100
            else:
                lstm_direction = 0.0
            
            self.metrics['lstm'] = {
                'mae': float(lstm_mae),
                'rmse': float(lstm_rmse),
                'direction_accuracy': float(lstm_direction)
            }
            
            logger.info(f"✅ LSTM - Direction: {lstm_direction:.1f}%")
            
            # Random Forest
            logger.info("\n🌲 Training Random Forest...")
            y_direction = (df_clean['price'].shift(-1) > df_clean['price']).astype(int)
            features_rf = scaled_features[:-1]
            y_class = y_direction[:-1].values
            
            tscv_rf = TimeSeriesSplit(n_splits=5)
            splits_rf = list(tscv_rf.split(features_rf))
            train_idx_rf, test_idx_rf = splits_rf[-1]
            
            X_train_rf = features_rf[train_idx_rf]
            X_test_rf = features_rf[test_idx_rf]
            y_train_rf = y_class[train_idx_rf]
            y_test_rf = y_class[test_idx_rf]
            
            self.rf_model = RandomForestClassifier(
                n_estimators=300,
                max_depth=18,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',
                verbose=0
            )
            
            self.rf_model.fit(X_train_rf, y_train_rf)
            
            rf_pred = self.rf_model.predict(X_test_rf)
            rf_accuracy = accuracy_score(y_test_rf, rf_pred)
            
            self.metrics['rf'] = {'accuracy': float(rf_accuracy)}
            logger.info(f"✅ RF - Accuracy: {rf_accuracy:.2%}")
            
            # Gradient Boosting
            logger.info("\n🚀 Training Gradient Boosting...")
            y_gb = scaled_target[:-1]
            
            X_train_gb = features_rf[train_idx_rf]
            X_test_gb = features_rf[test_idx_rf]
            y_train_gb = y_gb[train_idx_rf]
            y_test_gb = y_gb[test_idx_rf]
            
            self.gb_model = GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.08,
                max_depth=7,
                subsample=0.8,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                verbose=0
            )
            
            self.gb_model.fit(X_train_gb, y_train_gb)
            
            gb_pred_scaled = self.gb_model.predict(X_test_gb)
            gb_pred = self.price_scaler.inverse_transform(gb_pred_scaled.reshape(-1, 1)).flatten()
            y_test_gb_original = self.price_scaler.inverse_transform(y_test_gb.reshape(-1, 1)).flatten()
            
            gb_mae = mean_absolute_error(y_test_gb_original, gb_pred)
            
            if len(gb_pred) > 1:
                min_len = min(len(gb_pred), len(y_test_gb_original)) - 1
                gb_direction = np.mean(
                    np.sign(gb_pred[:min_len] - y_test_gb_original[:min_len]) == 
                    np.sign(y_test_gb_original[1:min_len+1] - y_test_gb_original[:min_len])
                ) * 100
            else:
                gb_direction = 0.0
            
            self.metrics['gb'] = {
                'mae': float(gb_mae),
                'direction_accuracy': float(gb_direction)
            }
            
            logger.info(f"✅ GB - Direction: {gb_direction:.1f}%")
            
            avg_direction = (lstm_direction + rf_accuracy * 100 + gb_direction) / 3
            logger.info(f"\n📊 Ensemble Average: {avg_direction:.1f}%")
            
            self.is_trained = True
            self.last_training = datetime.now()
            
            self.save_models()
            
            logger.info("\n✅ ALL MODELS TRAINED!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, df: pd.DataFrame, timeframe_minutes: int, always_predict: bool = False) -> Optional[Dict]:
        """
        FIXED: UNBIASED PREDICTION
        Properly calculates both UP and DOWN predictions
        """
        
        if not self.is_trained:
            logger.warning("⚠️ Models not trained")
            return None
        
        try:
            category = get_timeframe_category(timeframe_minutes)
            sequence_length = self.sequence_length
            
            df_clean = df.dropna().copy()
            df_clean = df_clean.sort_values('datetime', ascending=True).reset_index(drop=True)
            
            min_required = sequence_length + 30
            if len(df_clean) < min_required:
                logger.warning(f"⚠️ Insufficient data: {len(df_clean)} < {min_required}")
                return None
            
            features = self.prepare_features(df_clean)
            
            if set(features.columns) != set(self.feature_columns):
                missing = set(self.feature_columns) - set(features.columns)
                if missing:
                    logger.error(f"❌ Missing features: {missing}")
                    return None
                features = features[self.feature_columns]
            
            scaled_features = self.feature_scaler.transform(features)
            current_price = df_clean.iloc[-1]['price']
            
            # === GET RAW MODEL PREDICTIONS ===
            
            # LSTM - predicts actual future price
            lstm_input = scaled_features[-sequence_length:].reshape(1, sequence_length, -1)
            lstm_pred_scaled = self.lstm_model.predict(lstm_input, verbose=0)[0][0]
            lstm_pred_price = self.price_scaler.inverse_transform([[lstm_pred_scaled]])[0][0]
            
            # GB - predicts actual future price
            gb_input = scaled_features[-1:].reshape(1, -1)
            gb_pred_scaled = self.gb_model.predict(gb_input)[0]
            gb_pred_price = self.price_scaler.inverse_transform([[gb_pred_scaled]])[0][0]
            
            # RF - predicts direction (0=DOWN, 1=UP)
            rf_direction = self.rf_model.predict(gb_input)[0]
            rf_proba = self.rf_model.predict_proba(gb_input)[0]
            rf_confidence = max(rf_proba) * 100
            
            # === CALCULATE PRICE CHANGES (absolute, not percentage yet) ===
            lstm_change = lstm_pred_price - current_price
            gb_change = gb_pred_price - current_price
            
            logger.debug(f"Raw predictions:")
            logger.debug(f"  LSTM: ${lstm_pred_price:,.2f} (change: ${lstm_change:+,.2f})")
            logger.debug(f"  GB:   ${gb_pred_price:,.2f} (change: ${gb_change:+,.2f})")
            logger.debug(f"  RF:   {'UP' if rf_direction == 1 else 'DOWN'} ({rf_confidence:.1f}%)")
            
            # === VOTING SYSTEM (FIXED) ===
            lstm_vote = 1 if lstm_change > 0 else -1
            gb_vote = 1 if gb_change > 0 else -1
            rf_vote = 1 if rf_direction == 1 else -1
            
            # Count votes
            votes = [lstm_vote, gb_vote, rf_vote]
            vote_sum = sum(votes)
            
            # Determine final direction by majority vote
            if vote_sum > 0:
                final_direction = 1  # Bullish
            elif vote_sum < 0:
                final_direction = -1  # Bearish
            else:
                # Tie - use RF confidence to break
                final_direction = rf_vote
            
            logger.debug(f"Voting: LSTM={lstm_vote}, GB={gb_vote}, RF={rf_vote} → Final={final_direction}")
            
            # === CALCULATE MAGNITUDE ===
            # Average the absolute changes from LSTM and GB
            avg_magnitude = (abs(lstm_change) + abs(gb_change)) / 2
            
            # Apply timeframe scaling
            time_factor = self._get_conservative_time_factor(timeframe_minutes, category)
            scaled_magnitude = avg_magnitude * time_factor
            
            # Apply confidence scaling
            confidence_factor = (rf_confidence / 100) ** 0.5  # Square root for less aggressive scaling
            scaled_magnitude *= confidence_factor
            
            # Apply direction
            predicted_change = scaled_magnitude * final_direction
            
            # === LIMIT MAXIMUM CHANGE ===
            max_change_pct = self._get_max_change_pct(category, timeframe_minutes)
            max_change = current_price * (max_change_pct / 100)
            
            if abs(predicted_change) > max_change:
                predicted_change = max_change if predicted_change > 0 else -max_change
                logger.debug(f"Capped change to {max_change_pct}%")
            
            predicted_price = current_price + predicted_change
            
            logger.debug(f"Final: ${predicted_price:,.2f} (change: ${predicted_change:+,.2f})")
            
            # === QUALITY AND CONFIDENCE ===
            quality_score = self._assess_quality_fixed(
                lstm_vote, gb_vote, rf_vote, rf_confidence,
                abs(lstm_change), abs(gb_change), df_clean
            )
            
            confidence = self._calculate_confidence_fixed(
                vote_sum, rf_confidence, quality_score, category
            )
            
            # === BUILD PREDICTION ===
            trend = "CALL (Bullish)" if predicted_change > 0 else "PUT (Bearish)"
            
            # Calculate price range using ATR
            volatility = df_clean['atr_14'].iloc[-1] if 'atr_14' in df_clean.columns else df_clean['price'].tail(20).std()
            range_mult = 0.5 * time_factor
            
            price_range_low = predicted_price - volatility * range_mult
            price_range_high = predicted_price + volatility * range_mult
            
            # Agreement metrics
            agreement_count = len([v for v in votes if v == final_direction])
            model_agreement = (agreement_count / 3.0) * 100
            all_agree = (agreement_count == 3)
            
            prediction = {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change': predicted_change,
                'price_change_pct': (predicted_change / current_price) * 100,
                'price_range_low': price_range_low,
                'price_range_high': price_range_high,
                'trend': trend,
                'confidence': confidence,
                'quality_score': quality_score,
                'lstm_prediction': lstm_pred_price,
                'gb_prediction': gb_pred_price,
                'rf_direction': 'UP' if rf_direction == 1 else 'DOWN',
                'rf_confidence': rf_confidence,
                'model_agreement': model_agreement,
                'all_models_agree': all_agree,
                'timeframe_minutes': timeframe_minutes,
                'volatility': volatility,
                'method': f'Fixed Ensemble ({category})',
                'model_metrics': self.metrics,
                'category': category,
                'votes': {'lstm': lstm_vote, 'gb': gb_vote, 'rf': rf_vote},
                'vote_result': vote_sum
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_conservative_time_factor(self, timeframe_minutes: int, category: str) -> float:
        """
        FIXED: Conservative time factors to prevent over-prediction
        """
        # Base factors are more conservative
        if category == 'ultra_short':
            return 0.3 * (timeframe_minutes / 15)  # Reduced from 0.6
        elif category == 'short':
            return 0.5 * (timeframe_minutes / 60)  # Reduced from 0.9
        elif category == 'medium':
            return 0.7 * (timeframe_minutes / 240)  # Reduced from 1.2
        else:  # long
            return 0.9 * (timeframe_minutes / 1440)  # Reduced from 1.5
    
    def _get_max_change_pct(self, category: str, timeframe_minutes: int) -> float:
        """
        FIXED: Realistic maximum change percentages
        """
        if category == 'ultra_short':
            return min(1.0 + (timeframe_minutes / 15) * 0.3, 2.5)
        elif category == 'short':
            return min(2.0 + (timeframe_minutes / 60) * 0.5, 4.0)
        elif category == 'medium':
            return min(3.0 + (timeframe_minutes / 240) * 0.7, 6.0)
        else:  # long
            return min(5.0 + (timeframe_minutes / 1440) * 1.0, 10.0)
    
    def _assess_quality_fixed(self, lstm_vote: int, gb_vote: int, rf_vote: int,
                              rf_confidence: float, lstm_mag: float, gb_mag: float,
                              df: pd.DataFrame) -> float:
        """FIXED: Quality assessment based on agreement and market conditions"""
        
        quality = 0
        
        # Model agreement (0-40 points)
        votes = [lstm_vote, gb_vote, rf_vote]
        agreement = len([v for v in votes if v == votes[0]])
        
        if agreement == 3:
            quality += 40
        elif agreement == 2:
            quality += 25
        else:
            quality += 10
        
        # RF confidence (0-20 points)
        if rf_confidence > 70:
            quality += 20
        elif rf_confidence > 60:
            quality += 15
        elif rf_confidence > 50:
            quality += 10
        else:
            quality += 5
        
        # Magnitude consistency (0-20 points)
        if lstm_mag > 0 and gb_mag > 0:
            ratio = min(lstm_mag, gb_mag) / max(lstm_mag, gb_mag)
            if ratio > 0.7:
                quality += 20
            elif ratio > 0.5:
                quality += 15
            elif ratio > 0.3:
                quality += 10
            else:
                quality += 5
        
        # Market conditions (0-20 points)
        try:
            recent = df.tail(20)
            
            # Trend strength (ADX)
            if 'adx' in recent.columns:
                adx = recent['adx'].iloc[-1]
                if adx > 30:
                    quality += 10
                elif adx > 20:
                    quality += 5
            
            # Volume confirmation
            if 'volume_ratio_20' in recent.columns:
                vol_ratio = recent['volume_ratio_20'].iloc[-1]
                if vol_ratio > 1.2:
                    quality += 10
                elif vol_ratio > 1.0:
                    quality += 5
        except:
            pass
        
        return min(quality, 100)
    
    def _calculate_confidence_fixed(self, vote_sum: int, rf_confidence: float,
                                    quality_score: float, category: str) -> float:
        """FIXED: Confidence based on voting strength"""
        
        # Base confidence from quality
        confidence = quality_score * 0.5
        
        # Voting strength bonus (0-25 points)
        if abs(vote_sum) == 3:  # Unanimous
            confidence += 25
        elif abs(vote_sum) == 1:  # 2-1 majority
            confidence += 15
        else:  # Tie broken by RF
            confidence += 5
        
        # RF confidence contribution (0-20 points)
        confidence += (rf_confidence - 50) * 0.4
        
        # Category bonus (longer timeframes more reliable)
        category_bonus = {
            'ultra_short': 0,
            'short': 3,
            'medium': 6,
            'long': 10
        }
        confidence += category_bonus.get(category, 0)
        
        # Maximum confidence limits (prevent overconfidence)
        max_confidence = {
            'ultra_short': 75,
            'short': 80,
            'medium': 85,
            'long': 90
        }
        
        return min(max(confidence, 0), max_confidence.get(category, 80))
    
    def save_models(self) -> bool:
        """Save models"""
        try:
            path = MODEL_CONFIG['model_save_path']
            os.makedirs(path, exist_ok=True)
            
            if self.lstm_model:
                self.lstm_model.save(f'{path}/lstm_model_optimized.keras')
            
            if self.rf_model:
                with open(f'{path}/rf_model_optimized.pkl', 'wb') as f:
                    pickle.dump(self.rf_model, f)
            
            if self.gb_model:
                with open(f'{path}/gb_model_optimized.pkl', 'wb') as f:
                    pickle.dump(self.gb_model, f)
            
            with open(f'{path}/scalers_optimized.pkl', 'wb') as f:
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
    
    def load_models(self) -> bool:
        """Load models"""
        try:
            path = MODEL_CONFIG['model_save_path']
            
            if os.path.exists(f'{path}/lstm_model_optimized.keras'):
                self.lstm_model = load_model(f'{path}/lstm_model_optimized.keras')
                
                with open(f'{path}/rf_model_optimized.pkl', 'rb') as f:
                    self.rf_model = pickle.load(f)
                
                with open(f'{path}/gb_model_optimized.pkl', 'rb') as f:
                    self.gb_model = pickle.load(f)
                
                with open(f'{path}/scalers_optimized.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.price_scaler = data['price_scaler']
                    self.feature_scaler = data['feature_scaler']
                    self.feature_columns = data['feature_columns']
                    self.metrics = data.get('metrics', {})
                    self.last_training = data.get('last_training')
                
                self.is_trained = True
                logger.info(f"✅ Fixed models loaded")
                return True
            
            elif os.path.exists(f'{path}/lstm_model.keras'):
                logger.warning("⚠️ Found old models - RETRAIN REQUIRED!")
                return False
            
            else:
                logger.warning("⚠️ No models found")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False
    
    def needs_retraining(self) -> bool:
        """Check if retraining needed"""
        if not self.is_trained or not self.last_training:
            return True
        
        time_since = (datetime.now() - self.last_training).total_seconds()
        return time_since > MODEL_CONFIG['auto_retrain_interval']


# Compatibility
BitcoinMLPredictor = ImprovedBitcoinPredictor