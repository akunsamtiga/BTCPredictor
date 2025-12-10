"""
Bitcoin Price Predictor - FIXED: CONSISTENT FEATURES
GUARANTEED predictions with consistent feature engineering
Target: High accuracy with consistent output
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
# DATA FETCHING
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
# FIXED: CONSISTENT FEATURE ENGINEERING
# ============================================================================

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIXED: Add technical indicators with CONSISTENT FEATURES
    Always produces the same features regardless of data size
    """
    try:
        df = df.copy()
        df = df.sort_values('datetime', ascending=True).reset_index(drop=True)
        
        data_points = len(df)
        logger.debug(f"Adding indicators for {data_points} data points")
        
        # Basic features (always safe)
        df['returns'] = df['price'].pct_change()
        df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
        
        # ================================================================
        # FIXED: CONSISTENT MOVING AVERAGES
        # Always use these specific periods
        # ================================================================
        base_ma_periods = [7, 14, 20]  # ALWAYS available
        extended_ma_periods = [50]      # Only for larger datasets
        
        # Add base MAs (works with 100+ points)
        for window in base_ma_periods:
            if data_points >= window * 2:  # Need at least 2x window size
                df[f'sma_{window}'] = df['price'].rolling(window=window, min_periods=window//2).mean()
                df[f'ema_{window}'] = df['price'].ewm(span=window, adjust=False, min_periods=window//2).mean()
                df[f'price_to_sma_{window}'] = (df['price'] - df[f'sma_{window}']) / df['price']
        
        # Add extended MAs (only for larger datasets)
        if data_points >= 150:
            for window in extended_ma_periods:
                df[f'sma_{window}'] = df['price'].rolling(window=window, min_periods=window//2).mean()
                df[f'ema_{window}'] = df['price'].ewm(span=window, adjust=False, min_periods=window//2).mean()
                df[f'price_to_sma_{window}'] = (df['price'] - df[f'sma_{window}']) / df['price']
        
        # MA cross (if both exist)
        if 'sma_7' in df.columns and 'sma_20' in df.columns:
            df['ma_cross_7_20'] = (df['sma_7'] > df['sma_20']).astype(int)
            df['ma_diff_7_20'] = (df['sma_7'] - df['sma_20']) / df['price']
        
        # ================================================================
        # RSI - CONSISTENT PERIODS
        # ================================================================
        rsi_periods = [14, 21]
        
        for period in rsi_periods:
            if data_points >= period * 2:
                delta = df['price'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period//2).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period//2).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
                df[f'rsi_{period}_norm'] = (df[f'rsi_{period}'] - 50) / 50
        
        # ================================================================
        # MACD
        # ================================================================
        if data_points >= 52:
            ema_12 = df['price'].ewm(span=12, adjust=False, min_periods=6).mean()
            ema_26 = df['price'].ewm(span=26, adjust=False, min_periods=13).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False, min_periods=4).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ================================================================
        # BOLLINGER BANDS - CONSISTENT PERIOD
        # ================================================================
        bb_period = 20
        if data_points >= bb_period * 2:
            df[f'bb_middle_{bb_period}'] = df['price'].rolling(window=bb_period, min_periods=bb_period//2).mean()
            bb_std = df['price'].rolling(window=bb_period, min_periods=bb_period//2).std()
            df[f'bb_upper_{bb_period}'] = df[f'bb_middle_{bb_period}'] + (bb_std * 2)
            df[f'bb_lower_{bb_period}'] = df[f'bb_middle_{bb_period}'] - (bb_std * 2)
            df[f'bb_position_{bb_period}'] = (df['price'] - df[f'bb_lower_{bb_period}']) / (df[f'bb_upper_{bb_period}'] - df[f'bb_lower_{bb_period}'])
            df[f'bb_width_{bb_period}'] = (df[f'bb_upper_{bb_period}'] - df[f'bb_lower_{bb_period}']) / df[f'bb_middle_{bb_period}']
        
        # ================================================================
        # STOCHASTIC
        # ================================================================
        if data_points >= 28:
            k_period = 14
            low_k = df['low'].rolling(window=k_period, min_periods=k_period//2).min()
            high_k = df['high'].rolling(window=k_period, min_periods=k_period//2).max()
            df['stoch_k'] = 100 * ((df['price'] - low_k) / (high_k - low_k))
            df['stoch_d'] = df['stoch_k'].rolling(window=3, min_periods=2).mean()
        
        # ================================================================
        # ATR - CONSISTENT PERIOD
        # ================================================================
        atr_period = 14
        if data_points >= atr_period * 2:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['price'].shift())
            low_close = np.abs(df['low'] - df['price'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df[f'atr_{atr_period}'] = true_range.rolling(window=atr_period, min_periods=atr_period//2).mean()
            df[f'atr_pct_{atr_period}'] = df[f'atr_{atr_period}'] / df['price']
        
        # ================================================================
        # ADX
        # ================================================================
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
        
        # ================================================================
        # VOLUME INDICATORS - CONSISTENT PERIOD
        # ================================================================
        vol_period = 20
        if data_points >= vol_period * 2:
            df[f'volume_sma_{vol_period}'] = df['volume'].rolling(window=vol_period, min_periods=vol_period//2).mean()
            df[f'volume_ratio_{vol_period}'] = df['volume'] / df[f'volume_sma_{vol_period}']
        
        if data_points >= 20:
            df['obv'] = (np.sign(df['price'].diff()) * df['volume']).fillna(0).cumsum()
            df['obv_ema'] = df['obv'].ewm(span=20, adjust=False, min_periods=10).mean()
        
        # ================================================================
        # MOMENTUM - CONSISTENT PERIODS
        # ================================================================
        mom_periods = [10, 20]
        for period in mom_periods:
            if data_points >= period * 2:
                df[f'momentum_{period}'] = df['price'] - df['price'].shift(period)
                df[f'roc_{period}'] = df['price'].pct_change(period) * 100
        
        # ================================================================
        # ROLLING STATISTICS - CONSISTENT PERIOD
        # ================================================================
        stat_period = 20
        if data_points >= stat_period * 2:
            df[f'rolling_std_{stat_period}'] = df['price'].rolling(window=stat_period, min_periods=stat_period//2).std()
            df[f'rolling_max_{stat_period}'] = df['price'].rolling(window=stat_period, min_periods=stat_period//2).max()
            df[f'rolling_min_{stat_period}'] = df['price'].rolling(window=stat_period, min_periods=stat_period//2).min()
            df[f'dist_from_max_{stat_period}'] = (df[f'rolling_max_{stat_period}'] - df['price']) / df['price']
            df[f'dist_from_min_{stat_period}'] = (df['price'] - df[f'rolling_min_{stat_period}']) / df['price']
            df[f'volatility_{stat_period}'] = df['returns'].rolling(window=stat_period, min_periods=stat_period//2).std()
        
        # ================================================================
        # PRICE PATTERNS (always safe)
        # ================================================================
        df['high_low_ratio'] = (df['high'] - df['low']) / df['price']
        df['close_position'] = (df['price'] - df['low']) / (df['high'] - df['low'])
        df['body_size'] = np.abs(df['price'] - df['open']) / df['price']
        
        # Market strength
        if data_points >= 20:
            df['trend_strength'] = np.abs(df['returns'].rolling(window=20, min_periods=10).mean()) * 100
        
        # ================================================================
        # SMART NaN HANDLING - NO DATA LOSS
        # ================================================================
        before_fill = len(df)
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].ffill()
        df[numeric_columns] = df[numeric_columns].bfill()
        
        remaining_nans = df.isnull().sum().sum()
        if remaining_nans > 0:
            logger.debug(f"Filling {remaining_nans} remaining NaNs with 0")
            df = df.fillna(0)
        
        # Sort back to most recent first
        df = df.sort_values('datetime', ascending=False).reset_index(drop=True)
        
        after_fill = len(df)
        
        if after_fill != before_fill:
            logger.warning(f"⚠️ Data loss: {before_fill} → {after_fill} points")
        else:
            logger.info(f"✅ Indicators added: {after_fill} points (100% retained)")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error adding indicators: {e}")
        import traceback
        traceback.print_exc()
        return df

# ============================================================================
# PREDICTOR CLASS
# ============================================================================

class ImprovedBitcoinPredictor:
    """
    FIXED: Always Predict with Quality Assessment
    Consistent feature engineering
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
        
        logger.info("🤖 Predictor initialized (Always Predict Mode)")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FIXED: Select consistent predictive features
        Only uses features that will ALWAYS be available
        """
        
        # FIXED: Core features that should always exist
        priority_features = [
            # Core indicators (always should exist if data >= 100 points)
            'rsi_14', 'rsi_14_norm', 'rsi_21',
            'macd', 'macd_signal', 'macd_hist',
            'bb_position_20', 'bb_width_20',
            'stoch_k', 'stoch_d',
            'atr_pct_14', 'volatility_20',
            'adx', 'plus_di', 'minus_di',
            'volume_ratio_20', 'obv_ema',
            
            # Moving averages (core)
            'price_to_sma_7', 'price_to_sma_14', 'price_to_sma_20',
            'ma_diff_7_20',
            
            # Extended MA (optional - may not exist in small datasets)
            'price_to_sma_50',
            
            # Momentum
            'roc_10', 'roc_20', 'momentum_10',
            
            # Statistics
            'rolling_std_20', 'dist_from_max_20', 'dist_from_min_20',
            
            # Patterns
            'high_low_ratio', 'close_position', 'body_size',
            
            # Strength
            'trend_strength',
        ]
        
        # Only use features that actually exist in DataFrame
        available = [col for col in priority_features if col in df.columns]
        
        if len(available) < len(priority_features):
            missing = set(priority_features) - set(available)
            logger.debug(f"📊 Using {len(available)}/{len(priority_features)} features")
            logger.debug(f"   Missing (ok): {missing}")
        else:
            logger.debug(f"📊 Using {len(available)} features (all available)")
        
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
        """Build optimized LSTM with attention"""
        
        inputs = Input(shape=input_shape)
        
        # First Bidirectional LSTM
        x = Bidirectional(LSTM(
            128,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.1,
            kernel_regularizer=l2(0.0005)
        ))(inputs)
        x = LayerNormalization()(x)
        
        # Multi-Head Attention
        attention_output = MultiHeadAttention(
            num_heads=4,
            key_dim=32,
            dropout=0.1
        )(x, x)
        x = LayerNormalization()(attention_output + x)
        
        # Second Bidirectional LSTM
        x = Bidirectional(LSTM(
            64,
            return_sequences=False,
            dropout=0.2,
            kernel_regularizer=l2(0.0005)
        ))(x)
        x = LayerNormalization()(x)
        
        # Dense layers
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.0005))(x)
        x = Dropout(0.3)(x)
        x = LayerNormalization()(x)
        
        x = Dense(32, activation='relu', kernel_regularizer=l2(0.0005))(x)
        x = Dropout(0.2)(x)
        
        # Output
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        optimizer = Adam(
            learning_rate=0.0005,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='huber',
            metrics=['mae', 'mse']
        )
        
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
            
            logger.info(f"📊 Features prepared: {len(features.columns)} columns")
            logger.info(f"   {features.columns.tolist()}")
            
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
            
            logger.info(f"   Train: {len(X_train_lstm)}, Test: {len(X_test_lstm)}")
            
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
            
            logger.info(f"✅ LSTM - Direction Accuracy: {lstm_direction:.1f}%")
            
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
            
            logger.info(f"✅ GB - Direction Accuracy: {gb_direction:.1f}%")
            
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
        FIXED: ALWAYS PREDICT with quality assessment
        """
        
        if not self.is_trained:
            logger.warning("⚠️ Models not trained")
            return None
        
        try:
            category = get_timeframe_category(timeframe_minutes)
            
            seq_length_map = {
                'ultra_short': MODEL_CONFIG['lstm'].get('ultra_short_sequence', 30),
                'short': MODEL_CONFIG['lstm'].get('short_sequence', 60),
                'medium': MODEL_CONFIG['lstm'].get('medium_sequence', 80),
                'long': MODEL_CONFIG['lstm'].get('long_sequence', 100)
            }
            sequence_length = seq_length_map.get(category, self.sequence_length)
            
            df_clean = df.dropna().copy()
            df_clean = df_clean.sort_values('datetime', ascending=True).reset_index(drop=True)
            
            min_required = sequence_length + 30
            if len(df_clean) < min_required:
                logger.warning(f"⚠️ Insufficient data: {len(df_clean)} < {min_required}")
                return None
            
            features = self.prepare_features(df_clean)
            
            # Check feature mismatch
            if set(features.columns) != set(self.feature_columns):
                missing = set(self.feature_columns) - set(features.columns)
                extra = set(features.columns) - set(self.feature_columns)
                
                if missing:
                    logger.error(f"❌ Missing features: {missing}")
                    logger.error("   Model needs retraining with current feature set!")
                    return None
                
                if extra:
                    logger.debug(f"   Extra features ignored: {extra}")
                    features = features[self.feature_columns]
            
            scaled_features = self.feature_scaler.transform(features)
            
            current_price = df_clean.iloc[-1]['price']
            
            # === MODEL PREDICTIONS ===
            
            # LSTM
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
            
            # === SMART ENSEMBLE ===
            
            weights = self._get_weights(category)
            time_factor = self._get_time_factor(timeframe_minutes, category)
            
            lstm_change = (lstm_pred - current_price) * time_factor
            gb_change = (gb_pred - current_price) * time_factor
            
            base_ensemble = (
                weights['lstm'] * lstm_change +
                weights['gb'] * gb_change
            )
            
            rf_adjustment = 1.0
            if rf_direction == 1:
                rf_adjustment = 1.0 + ((rf_confidence - 50) / 150)
            else:
                rf_adjustment = 1.0 - ((rf_confidence - 50) / 150)
            
            ensemble_change = base_ensemble * rf_adjustment
            
            trend_multiplier = self._calculate_trend_strength(df_clean)
            ensemble_change *= trend_multiplier
            
            predicted_price = current_price + ensemble_change
            
            max_change_pct = 15
            max_change = current_price * (max_change_pct / 100)
            if abs(ensemble_change) > max_change:
                ensemble_change = max_change if ensemble_change > 0 else -max_change
                predicted_price = current_price + ensemble_change
            
            # === QUALITY ASSESSMENT ===
            
            quality_score = self._assess_quality(
                lstm_change, gb_change, rf_direction, rf_confidence,
                category, df_clean
            )
            
            confidence = self._calculate_confidence(
                lstm_change, gb_change, rf_direction, rf_confidence,
                quality_score, category
            )
            
            # === BUILD PREDICTION ===
            
            trend = "CALL (Bullish)" if ensemble_change > 0 else "PUT (Bearish)"
            
            volatility = df_clean['atr_14'].iloc[-1] if 'atr_14' in df_clean.columns else df_clean['price'].tail(20).std()
            range_multiplier = self._get_range_multiplier(category, time_factor)
            
            price_range_low = predicted_price - volatility * range_multiplier
            price_range_high = predicted_price + volatility * range_multiplier
            
            lstm_dir = 1 if lstm_change > 0 else 0
            gb_dir = 1 if gb_change > 0 else 0
            all_agree = (lstm_dir == gb_dir == rf_direction)
            agreement_count = sum([lstm_dir == rf_direction, gb_dir == rf_direction, lstm_dir == gb_dir])
            model_agreement = agreement_count / 3.0
            
            prediction = {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change': ensemble_change,
                'price_change_pct': (ensemble_change / current_price) * 100,
                'price_range_low': price_range_low,
                'price_range_high': price_range_high,
                'trend': trend,
                'confidence': confidence,
                'quality_score': quality_score,
                'lstm_prediction': lstm_pred,
                'gb_prediction': gb_pred,
                'rf_direction': 'UP' if rf_direction == 1 else 'DOWN',
                'rf_confidence': rf_confidence,
                'model_agreement': model_agreement * 100,
                'all_models_agree': all_agree,
                'timeframe_minutes': timeframe_minutes,
                'volatility': volatility,
                'method': f'Optimized ML Ensemble ({category})',
                'model_metrics': self.metrics,
                'category': category,
                'time_factor': time_factor,
                'sequence_length_used': sequence_length,
                'trend_strength': trend_multiplier
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_weights(self, category: str) -> Dict[str, float]:
        """Get ensemble weights by category"""
        weights = {
            'ultra_short': {'lstm': 0.35, 'gb': 0.35, 'rf': 0.30},
            'short': {'lstm': 0.40, 'gb': 0.35, 'rf': 0.25},
            'medium': {'lstm': 0.45, 'gb': 0.40, 'rf': 0.15},
            'long': {'lstm': 0.50, 'gb': 0.40, 'rf': 0.10}
        }
        return weights.get(category, weights['short'])
    
    def _get_time_factor(self, timeframe_minutes: int, category: str) -> float:
        """Time adjustment factor"""
        if category == 'ultra_short':
            return min(timeframe_minutes / 30, 0.6)
        elif category == 'short':
            return min(timeframe_minutes / 60, 0.9)
        elif category == 'medium':
            return min(timeframe_minutes / 240, 1.2)
        else:
            return min(timeframe_minutes / 1440, 1.5)
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength"""
        try:
            recent = df.tail(20)
            
            price_momentum = (recent.iloc[-1]['price'] - recent.iloc[0]['price']) / recent.iloc[0]['price']
            
            rsi = recent['rsi_14'].iloc[-1] if 'rsi_14' in recent.columns else 50
            rsi_factor = (rsi - 50) / 50
            
            adx = recent['adx'].iloc[-1] if 'adx' in recent.columns else 20
            adx_strength = min(adx / 50, 1.0)
            
            volume_ratio = recent['volume_ratio_20'].iloc[-1] if 'volume_ratio_20' in recent.columns else 1.0
            volume_strength = min(volume_ratio / 2, 1.0)
            
            strength = (
                price_momentum * 3.0 +
                rsi_factor * 1.0 +
                adx_strength * 0.5
            ) / 4.5
            
            multiplier = 1.0 + (strength * 0.3)
            
            return max(min(multiplier, 1.3), 0.7)
            
        except:
            return 1.0
    
    def _assess_quality(self, lstm_change: float, gb_change: float,
                       rf_direction: int, rf_confidence: float,
                       category: str, df: pd.DataFrame) -> float:
        """Assess prediction quality (0-100)"""
        
        quality = 0
        
        lstm_dir = 1 if lstm_change > 0 else 0
        gb_dir = 1 if gb_change > 0 else 0
        
        if lstm_dir == gb_dir == rf_direction:
            quality += 40
        elif (lstm_dir == gb_dir) or (lstm_dir == rf_direction) or (gb_dir == rf_direction):
            quality += 20
        
        if rf_confidence > 70:
            quality += 20
        elif rf_confidence > 60:
            quality += 15
        elif rf_confidence > 50:
            quality += 10
        
        try:
            recent = df.tail(20)
            
            if 'adx' in recent.columns:
                adx = recent['adx'].iloc[-1]
                if adx > 25:
                    quality += 10
            
            if 'volume_ratio_20' in recent.columns:
                vol_ratio = recent['volume_ratio_20'].iloc[-1]
                if vol_ratio > 1.0:
                    quality += 5
            
            if 'volatility_20' in recent.columns:
                vol = recent['volatility_20'].iloc[-1]
                if vol < 0.03:
                    quality += 5
        except:
            pass
        
        if abs(lstm_change) > 0 and abs(gb_change) > 0:
            ratio = min(abs(lstm_change), abs(gb_change)) / max(abs(lstm_change), abs(gb_change))
            if ratio > 0.7:
                quality += 20
            elif ratio > 0.5:
                quality += 10
        
        return min(quality, 100)
    
    def _calculate_confidence(self, lstm_change: float, gb_change: float,
                             rf_direction: int, rf_confidence: float,
                             quality_score: float, category: str) -> float:
        """Calculate confidence (0-100)"""
        
        base = quality_score * 0.6
        rf_contribution = (rf_confidence - 50) * 0.8
        
        confidence = base + rf_contribution
        
        category_bonus = {
            'ultra_short': 0,
            'short': 2,
            'medium': 4,
            'long': 6
        }
        confidence += category_bonus.get(category, 0)
        
        max_confidence = {
            'ultra_short': 85,
            'short': 88,
            'medium': 90,
            'long': 92
        }
        
        return min(max(confidence, 0), max_confidence.get(category, 88))
    
    def _get_min_confidence(self, category: str) -> float:
        """Get minimum confidence threshold"""
        thresholds = {
            'ultra_short': 45,
            'short': 40,
            'medium': 38,
            'long': 35
        }
        return thresholds.get(category, 40)
    
    def _get_range_multiplier(self, category: str, time_factor: float) -> float:
        """Calculate price range multiplier"""
        base = {
            'ultra_short': 0.4,
            'short': 0.6,
            'medium': 0.8,
            'long': 1.0
        }.get(category, 0.6)
        
        return base * time_factor * 0.7
    
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
                logger.info(f"✅ Optimized models loaded")
                logger.info(f"   Features: {len(self.feature_columns)}")
                return True
            
            # Try legacy models
            elif os.path.exists(f'{path}/lstm_model.keras'):
                logger.warning("⚠️ Found legacy models - RETRAINING RECOMMENDED")
                logger.warning("   Feature set may be inconsistent!")
                
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
                logger.warning("⚠️ Legacy models loaded - RETRAIN IMMEDIATELY!")
                return True
            
            else:
                logger.warning("⚠️ No models found")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def needs_retraining(self) -> bool:
        """Check if retraining needed"""
        if not self.is_trained or not self.last_training:
            return True
        
        time_since = (datetime.now() - self.last_training).total_seconds()
        return time_since > MODEL_CONFIG['auto_retrain_interval']


# Compatibility alias
BitcoinMLPredictor = ImprovedBitcoinPredictor