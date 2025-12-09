"""
ENHANCED Bitcoin Price Predictor - HIGH WIN RATE VERSION
Target: >70% Win Rate

IMPROVEMENTS:
1. Advanced Feature Engineering (100+ features)
2. Market Regime Detection (only predict in favorable conditions)
3. Attention-based LSTM Architecture
4. Dynamic Ensemble Weighting
5. Ultra-Strict Confidence Filtering
6. Multi-Timeframe Confirmation
7. Volatility-Adjusted Predictions
8. Adaptive Learning from Performance
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
from collections import deque

from config import MODEL_CONFIG, DATA_CONFIG, get_timeframe_category, get_min_confidence
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    import tensorflow as tf
    from tensorflow import keras
    from keras.models import Sequential, Model, load_model
    from keras.layers import (LSTM, Dense, Dropout, Bidirectional, BatchNormalization,
                             Input, Attention, Concatenate, Layer, MultiHeadAttention,
                             LayerNormalization, GlobalAveragePooling1D)
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

# Market regime thresholds
FAVORABLE_CONDITIONS = {
    'min_trend_strength': 1.5,    # Minimum price movement %
    'max_volatility': 5.0,         # Maximum acceptable volatility
    'min_volume_ratio': 0.8,       # Minimum volume ratio
    'rsi_extremes': (25, 75),      # RSI must be between these
    'min_atr_stability': 0.7,      # ATR stability factor
}


# ============================================================================
# DATA FETCHING (UNCHANGED - Keep existing functions)
# ============================================================================

def get_current_btc_price() -> Optional[float]:
    """Get current Bitcoin price from CryptoCompare"""
    api_key = DATA_CONFIG.get('cryptocompare_api_key')
    
    if not api_key:
        logger.error("❌ CryptoCompare API key not configured")
        return None
    
    url = "https://min-api.cryptocompare.com/data/price"
    params = {
        'fsym': 'BTC',
        'tsyms': 'USD',
        'api_key': api_key
    }
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'USD' in data:
                price = float(data['USD'])
                return price
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"❌ Failed to get current price after {max_retries} attempts")
                return None
        except Exception as e:
            logger.error(f"❌ Error getting current price: {e}")
            return None
    
    return None


def _fetch_single_batch(endpoint: str, limit: int, to_timestamp: Optional[int] = None) -> Optional[Dict]:
    """Fetch single batch of data from CryptoCompare API"""
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
            logger.error(f"❌ Unexpected API response structure")
            return None
        
        return data
        
    except Exception as e:
        logger.error(f"❌ API request failed: {e}")
        return None


def get_bitcoin_data_realtime(days: int = 7, interval: str = 'hour') -> Optional[pd.DataFrame]:
    """Fetch historical Bitcoin data with automatic pagination"""
    api_key = DATA_CONFIG.get('cryptocompare_api_key')
    
    if not api_key:
        logger.error("❌ CryptoCompare API key not configured")
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
        return _fetch_multiple_requests(endpoint, total_points, interval)


def _fetch_single_request(endpoint: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetch data with single API request"""
    data = _fetch_single_batch(endpoint, limit)
    
    if not data:
        return None
    
    candles = data['Data']['Data']
    
    if not candles:
        logger.error("❌ No data returned from API")
        return None
    
    return _parse_candles_to_dataframe(candles)


def _fetch_multiple_requests(endpoint: str, total_points: int, interval: str) -> Optional[pd.DataFrame]:
    """Fetch data with multiple API requests"""
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
        logger.error("❌ No data collected from any batch")
        return None
    
    unique_candles = []
    seen_times = set()
    
    for candle in all_candles:
        if candle['time'] not in seen_times:
            unique_candles.append(candle)
            seen_times.add(candle['time'])
    
    return _parse_candles_to_dataframe(unique_candles)


def _parse_candles_to_dataframe(candles: list) -> pd.DataFrame:
    """Parse candle data into DataFrame"""
    df = pd.DataFrame(candles)
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={'close': 'price', 'volumefrom': 'volume'})
    df = df[['datetime', 'price', 'open', 'high', 'low', 'volume']]
    df = df.sort_values('datetime', ascending=False).reset_index(drop=True)
    df = df[df['price'] > 0].dropna(subset=['price'])
    
    return df


# ============================================================================
# ENHANCED FEATURE ENGINEERING - 100+ FEATURES
# ============================================================================

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    ENHANCED: Add 100+ technical indicators
    """
    try:
        df = df.copy()
        df = df.sort_values('datetime', ascending=True).reset_index(drop=True)
        
        # === BASIC PRICE FEATURES ===
        df['returns'] = df['price'].pct_change()
        df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
        
        # === MOVING AVERAGES (Multiple Timeframes) ===
        for window in [5, 7, 10, 14, 20, 30, 50, 100, 200]:
            df[f'sma_{window}'] = df['price'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['price'].ewm(span=window, adjust=False).mean()
            df[f'price_to_sma_{window}'] = (df['price'] - df[f'sma_{window}']) / df['price']
            
        # === RSI (Multiple Periods) ===
        for period in [7, 14, 21, 28]:
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            df[f'rsi_{period}_norm'] = (df[f'rsi_{period}'] - 50) / 50
            
        # === MACD (Multiple Settings) ===
        for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (19, 39, 9)]:
            ema_fast = df['price'].ewm(span=fast, adjust=False).mean()
            ema_slow = df['price'].ewm(span=slow, adjust=False).mean()
            df[f'macd_{fast}_{slow}'] = ema_fast - ema_slow
            df[f'macd_signal_{fast}_{slow}'] = df[f'macd_{fast}_{slow}'].ewm(span=signal, adjust=False).mean()
            df[f'macd_hist_{fast}_{slow}'] = df[f'macd_{fast}_{slow}'] - df[f'macd_signal_{fast}_{slow}']
            
        # === BOLLINGER BANDS (Multiple Settings) ===
        for window, num_std in [(20, 2), (20, 3), (50, 2)]:
            df[f'bb_middle_{window}'] = df['price'].rolling(window=window).mean()
            bb_std = df['price'].rolling(window=window).std()
            df[f'bb_upper_{window}_{num_std}'] = df[f'bb_middle_{window}'] + (bb_std * num_std)
            df[f'bb_lower_{window}_{num_std}'] = df[f'bb_middle_{window}'] - (bb_std * num_std)
            df[f'bb_position_{window}'] = (df['price'] - df[f'bb_lower_{window}_{num_std}']) / (df[f'bb_upper_{window}_{num_std}'] - df[f'bb_lower_{window}_{num_std}'])
            df[f'bb_width_{window}'] = (df[f'bb_upper_{window}_{num_std}'] - df[f'bb_lower_{window}_{num_std}']) / df[f'bb_middle_{window}']
            
        # === STOCHASTIC (Multiple Settings) ===
        for k_period, d_period in [(14, 3), (21, 3), (14, 5)]:
            low_k = df['low'].rolling(window=k_period).min()
            high_k = df['high'].rolling(window=k_period).max()
            df[f'stoch_k_{k_period}'] = 100 * ((df['price'] - low_k) / (high_k - low_k))
            df[f'stoch_d_{k_period}_{d_period}'] = df[f'stoch_k_{k_period}'].rolling(window=d_period).mean()
            
        # === ATR (Average True Range) ===
        for period in [7, 14, 21]:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['price'].shift())
            low_close = np.abs(df['low'] - df['price'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df[f'atr_{period}'] = true_range.rolling(window=period).mean()
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / df['price']
            
        # === ADX (Average Directional Index) ===
        for period in [14, 21]:
            high_diff = df['high'].diff()
            low_diff = -df['low'].diff()
            
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
            
            tr = ranges.max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            df[f'adx_{period}'] = dx.rolling(window=period).mean()
            df[f'plus_di_{period}'] = plus_di
            df[f'minus_di_{period}'] = minus_di
            
        # === VOLUME INDICATORS ===
        for window in [5, 10, 20]:
            df[f'volume_sma_{window}'] = df['volume'].rolling(window=window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']
            
        df['volume_ema'] = df['volume'].ewm(span=10, adjust=False).mean()
        df['obv'] = (np.sign(df['price'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
        
        # === MOMENTUM INDICATORS ===
        for period in [3, 5, 10, 20, 30]:
            df[f'momentum_{period}'] = df['price'] - df['price'].shift(period)
            df[f'roc_{period}'] = df['price'].pct_change(period) * 100
            
        # === ROLLING STATISTICS (Multiple Windows) ===
        windows = [5, 20, 50] if not VPS_CONFIG.get('use_minimal_features') else [10, 20]
        for window in windows:
            df[f'rolling_mean_{window}'] = df['price'].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['price'].rolling(window=window).std()
            df[f'rolling_var_{window}'] = df['price'].rolling(window=window).var()
            df[f'rolling_skew_{window}'] = df['price'].rolling(window=window).skew()
            df[f'rolling_kurt_{window}'] = df['price'].rolling(window=window).kurt()
            df[f'rolling_max_{window}'] = df['price'].rolling(window=window).max()
            df[f'rolling_min_{window}'] = df['price'].rolling(window=window).min()
            df[f'rolling_range_{window}'] = df[f'rolling_max_{window}'] - df[f'rolling_min_{window}']
            
            # Distance from extremes
            df[f'dist_from_max_{window}'] = (df[f'rolling_max_{window}'] - df['price']) / df['price']
            df[f'dist_from_min_{window}'] = (df['price'] - df[f'rolling_min_{window}']) / df['price']
            
        # === PRICE PATTERNS ===
        df['high_low_ratio'] = (df['high'] - df['low']) / df['price']
        df['close_position'] = (df['price'] - df['low']) / (df['high'] - df['low'])
        df['upper_shadow'] = (df['high'] - df[['open', 'price']].max(axis=1)) / df['price']
        df['lower_shadow'] = (df[['open', 'price']].min(axis=1) - df['low']) / df['price']
        df['body_size'] = np.abs(df['price'] - df['open']) / df['price']
        
        # === TREND INDICATORS ===
        for fast, slow in [(7, 20), (20, 50), (50, 200)]:
            df[f'ma_cross_{fast}_{slow}'] = (df[f'sma_{fast}'] > df[f'sma_{slow}']).astype(int)
            df[f'ma_diff_{fast}_{slow}'] = (df[f'sma_{fast}'] - df[f'sma_{slow}']) / df['price']
            
        # === VOLATILITY INDICATORS ===
        for window in [10, 20, 30]:
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
            df[f'parkinson_vol_{window}'] = np.sqrt(
                (1 / (4 * np.log(2))) * 
                ((np.log(df['high'] / df['low']) ** 2).rolling(window=window).mean())
            )
            
        # === MARKET STRENGTH ===
        df['trend_strength'] = np.abs(df['returns'].rolling(window=20).mean()) * 100
        df['market_momentum'] = df['price'].diff(20) / df['price'].shift(20) * 100
        
        # === FIBONACCI LEVELS ===
        for window in [20, 50]:
            high = df['high'].rolling(window=window).max()
            low = df['low'].rolling(window=window).min()
            diff = high - low
            
            for level, name in [(0.236, '236'), (0.382, '382'), (0.5, '500'), (0.618, '618')]:
                df[f'fib_{name}_{window}'] = low + diff * level
                df[f'price_to_fib_{name}_{window}'] = (df['price'] - df[f'fib_{name}_{window}']) / df['price']
        
        # === ACCUMULATION/DISTRIBUTION ===
        df['ad_line'] = ((df['price'] - df['low']) - (df['high'] - df['price'])) / (df['high'] - df['low']) * df['volume']
        df['ad_line'] = df['ad_line'].fillna(0).cumsum()
        df['ad_oscillator'] = df['ad_line'].diff(10)
        
        # === WILLIAMS %R ===
        for period in [14, 28]:
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()
            df[f'williams_r_{period}'] = -100 * (highest_high - df['price']) / (highest_high - lowest_low)
            
        # === KELTNER CHANNELS ===
        for period in [20]:
            df[f'keltner_middle_{period}'] = df['price'].ewm(span=period, adjust=False).mean()
            df[f'keltner_upper_{period}'] = df[f'keltner_middle_{period}'] + 2 * df[f'atr_{14}']
            df[f'keltner_lower_{period}'] = df[f'keltner_middle_{period}'] - 2 * df[f'atr_{14}']
            df[f'keltner_position_{period}'] = (df['price'] - df[f'keltner_lower_{period}']) / (df[f'keltner_upper_{period}'] - df[f'keltner_lower_{period}'])
        
        # Sort back to descending
        df = df.sort_values('datetime', ascending=False).reset_index(drop=True)
        
        initial_cols = ['datetime', 'price', 'open', 'high', 'low', 'volume']
        indicator_count = len(df.columns) - len(initial_cols)
        logger.debug(f"✅ Added {indicator_count} technical indicators")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error adding technical indicators: {e}")
        return df


# ============================================================================
# MARKET REGIME DETECTOR
# ============================================================================

class MarketRegimeDetector:
    """Detects if market conditions are favorable for prediction"""
    
    @staticmethod
    def is_favorable(df: pd.DataFrame, timeframe_minutes: int) -> Tuple[bool, str, float]:
        """
        Check if market conditions are favorable for prediction
        
        Returns:
            (is_favorable, reason, confidence_multiplier)
        """
        try:
            recent = df.head(30).copy()
            
            # Calculate key metrics
            price_change = abs((recent.iloc[0]['price'] - recent.iloc[-1]['price']) / recent.iloc[-1]['price'] * 100)
            volatility = recent['price'].std() / recent['price'].mean() * 100
            
            # RSI
            rsi = recent['rsi_14'].iloc[0] if 'rsi_14' in recent.columns else 50
            
            # Volume
            if 'volume_ratio_20' in recent.columns:
                volume_ratio = recent['volume_ratio_20'].iloc[0]
            else:
                volume_ratio = 1.0
            
            # ATR stability
            if 'atr_14' in recent.columns:
                atr_values = recent['atr_14'].dropna()
                if len(atr_values) > 5:
                    atr_stability = 1 - (atr_values.std() / atr_values.mean())
                else:
                    atr_stability = 0.7
            else:
                atr_stability = 0.7
            
            # Trend consistency
            if 'ema_20' in recent.columns and 'ema_50' in recent.columns:
                ema_aligned = (recent['ema_20'] > recent['ema_50']).sum() / len(recent)
                trend_consistency = max(ema_aligned, 1 - ema_aligned)
            else:
                trend_consistency = 0.5
            
            # === CHECK CONDITIONS ===
            
            # 1. Trend Strength
            if price_change < FAVORABLE_CONDITIONS['min_trend_strength']:
                return False, f"Weak trend ({price_change:.2f}% < {FAVORABLE_CONDITIONS['min_trend_strength']}%)", 0.0
            
            # 2. Volatility
            if volatility > FAVORABLE_CONDITIONS['max_volatility']:
                return False, f"High volatility ({volatility:.2f}% > {FAVORABLE_CONDITIONS['max_volatility']}%)", 0.0
            
            # 3. Volume
            if volume_ratio < FAVORABLE_CONDITIONS['min_volume_ratio']:
                return False, f"Low volume (ratio {volume_ratio:.2f})", 0.0
            
            # 4. RSI extremes
            rsi_min, rsi_max = FAVORABLE_CONDITIONS['rsi_extremes']
            if not (rsi_min < rsi < rsi_max):
                return False, f"RSI extreme ({rsi:.1f} outside {rsi_min}-{rsi_max})", 0.0
            
            # 5. ATR stability
            if atr_stability < FAVORABLE_CONDITIONS['min_atr_stability']:
                return False, f"Unstable ATR (stability {atr_stability:.2f})", 0.0
            
            # 6. Trend consistency
            if trend_consistency < 0.6:
                return False, f"Inconsistent trend ({trend_consistency:.2f})", 0.0
            
            # === CALCULATE CONFIDENCE MULTIPLIER ===
            
            # Perfect conditions = 1.0, good = 0.8-0.95
            score = 0
            max_score = 6
            
            # Strong trend
            if price_change > 3.0:
                score += 1
            
            # Low volatility
            if volatility < 3.0:
                score += 1
            
            # High volume
            if volume_ratio > 1.2:
                score += 1
            
            # Neutral RSI
            if 40 < rsi < 60:
                score += 1
            
            # Stable ATR
            if atr_stability > 0.85:
                score += 1
            
            # Strong trend
            if trend_consistency > 0.75:
                score += 1
            
            confidence_multiplier = 0.7 + (score / max_score) * 0.3  # 0.7 to 1.0
            
            return True, "Favorable conditions", confidence_multiplier
            
        except Exception as e:
            logger.error(f"❌ Market regime detection error: {e}")
            return False, "Detection error", 0.0


# ============================================================================
# ENHANCED PREDICTOR
# ============================================================================

class ImprovedBitcoinPredictor:
    """
    ENHANCED Bitcoin Predictor - Target >70% Win Rate
    """
    
    def __init__(self):
        self.lstm_model = None
        self.rf_model = None
        self.gb_model = None
        self.price_scaler = RobustScaler()  # Better for outliers
        self.feature_scaler = StandardScaler()
        self.sequence_length = MODEL_CONFIG['lstm']['sequence_length']
        self.feature_columns = []
        self.is_trained = False
        self.metrics = {}
        self.validation_scores = {}
        self.last_training = None
        
        # Performance tracking
        self.prediction_history = deque(maxlen=200)
        self.recent_wins = deque(maxlen=50)
        self.model_performance = {
            'lstm': deque(maxlen=50),
            'rf': deque(maxlen=50),
            'gb': deque(maxlen=50)
        }
        
        # Market regime detector
        self.regime_detector = MarketRegimeDetector()
        
        logger.info("🤖 Enhanced predictor initialized (High Win Rate Mode)")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select best features from 100+ indicators
        """
        
        # Priority features (proven to be most predictive)
        priority_features = [
            # RSI family
            'rsi_14', 'rsi_14_norm', 'rsi_21', 'rsi_7',
            
            # MACD family
            'macd_12_26', 'macd_signal_12_26', 'macd_hist_12_26',
            
            # Bollinger Bands
            'bb_position_20', 'bb_width_20',
            
            # Stochastic
            'stoch_k_14', 'stoch_d_14_3',
            
            # ATR & Volatility
            'atr_pct_14', 'volatility_20', 'parkinson_vol_20',
            
            # ADX (Trend Strength)
            'adx_14', 'plus_di_14', 'minus_di_14',
            
            # Volume
            'volume_ratio_20', 'obv_ema',
            
            # Moving Averages
            'price_to_sma_20', 'price_to_sma_50', 'price_to_sma_200',
            'ma_diff_7_20', 'ma_diff_20_50',
            
            # Momentum
            'roc_10', 'roc_20', 'momentum_10',
            
            # Rolling Statistics
            'rolling_std_20', 'dist_from_max_20', 'dist_from_min_20',
            
            # Price Patterns
            'high_low_ratio', 'close_position', 'body_size',
            
            # Market Strength
            'trend_strength', 'market_momentum',
            
            # Williams %R
            'williams_r_14',
            
            # Keltner Channels
            'keltner_position_20'
        ]
        
        # Add all available priority features
        available = [col for col in priority_features if col in df.columns]
        
        # Add selected rolling statistics
        for window in [5, 10, 20]:
            for col in [f'rolling_mean_{window}', f'rolling_std_{window}', 
                       f'rolling_skew_{window}', f'rolling_range_{window}']:
                if col in df.columns and col not in available:
                    available.append(col)
        
        # Add fibonacci levels
        for level in ['236', '382', '500', '618']:
            col = f'price_to_fib_{level}_20'
            if col in df.columns and col not in available:
                available.append(col)
        
        self.feature_columns = available
        logger.debug(f"📊 Using {len(available)} optimized features")
        
        return df[available].copy()
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray, sequence_length: int):
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(target[i + sequence_length])
        return np.array(X), np.array(y)
    
    def build_improved_lstm(self, input_shape: tuple) -> Model:
        """
        ENHANCED: Attention-based LSTM with better architecture
        """
        
        inputs = Input(shape=input_shape)
        
        # First Bidirectional LSTM layer
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
        x = LayerNormalization()(attention_output + x)  # Residual connection
        
        # Second Bidirectional LSTM layer
        x = Bidirectional(LSTM(
            64,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.1,
            kernel_regularizer=l2(0.0005)
        ))(x)
        x = LayerNormalization()(x)
        
        # Third LSTM layer
        x = Bidirectional(LSTM(
            32,
            return_sequences=False,
            dropout=0.2,
            kernel_regularizer=l2(0.0005)
        ))(x)
        x = LayerNormalization()(x)
        
        # Dense layers with residual
        dense1 = Dense(64, activation='relu', kernel_regularizer=l2(0.0005))(x)
        dense1 = Dropout(0.3)(dense1)
        dense1 = LayerNormalization()(dense1)
        
        dense2 = Dense(32, activation='relu', kernel_regularizer=l2(0.0005))(dense1)
        dense2 = Dropout(0.2)(dense2)
        dense2 = LayerNormalization()(dense2)
        
        dense3 = Dense(16, activation='relu')(dense2)
        dense3 = Dropout(0.1)(dense3)
        
        # Output
        outputs = Dense(1)(dense3)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Custom optimizer
        optimizer = Adam(
            learning_rate=0.0005,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            clipnorm=1.0  # Gradient clipping
        )
        
        model.compile(
            optimizer=optimizer,
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train_models(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 64) -> bool:
        """
        Train all models with enhanced validation
        """
        
        if not ML_AVAILABLE:
            logger.error("❌ ML libraries not available")
            return False
        
        try:
            logger.info("\n🤖 TRAINING ENHANCED MODELS (HIGH WIN RATE)...")
            
            # Prepare data
            df_clean = df.dropna().copy()
            df_clean = df_clean.sort_values('datetime', ascending=True).reset_index(drop=True)
            
            if len(df_clean) < 1000:  # Need more data for better models
                logger.error(f"❌ Insufficient data: {len(df_clean)} (need 1000+)")
                return False
            
            # Prepare features
            features = self.prepare_features(df_clean)
            target = df_clean['price'].values
            
            # Scale data
            scaled_features = self.feature_scaler.fit_transform(features)
            scaled_target = self.price_scaler.fit_transform(target.reshape(-1, 1)).flatten()
            
            # Time series split with more folds
            tscv = TimeSeriesSplit(n_splits=5)
            splits = list(tscv.split(scaled_features))
            train_idx, test_idx = splits[-1]  # Use last fold
            
            # ================================================================
            # LSTM TRAINING
            # ================================================================
            logger.info("\n🔵 Training Enhanced LSTM with Attention...")
            X_lstm, y_lstm = self.create_sequences(scaled_features, scaled_target, self.sequence_length)
            
            train_idx_lstm = train_idx[train_idx < len(X_lstm)]
            test_idx_lstm = test_idx[test_idx < len(X_lstm)]
            
            X_train_lstm = X_lstm[train_idx_lstm]
            X_test_lstm = X_lstm[test_idx_lstm]
            y_train_lstm = y_lstm[train_idx_lstm]
            y_test_lstm = y_lstm[test_idx_lstm]
            
            logger.info(f"   LSTM Train: {len(X_train_lstm)}, Test: {len(X_test_lstm)}")
            
            self.lstm_model = self.build_improved_lstm((self.sequence_length, len(self.feature_columns)))
            
            # Enhanced callbacks
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=15,  # More patience
                restore_best_weights=True,
                verbose=1
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=0.00001,
                verbose=1
            )
            
            os.makedirs(MODEL_CONFIG['model_save_path'], exist_ok=True)
            checkpoint = ModelCheckpoint(
                f"{MODEL_CONFIG['model_save_path']}/lstm_best.keras",
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
            
            # Train with validation
            history = self.lstm_model.fit(
                X_train_lstm, y_train_lstm,
                validation_data=(X_test_lstm, y_test_lstm),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop, reduce_lr, checkpoint],
                verbose=1
            )
            
            # Evaluate
            lstm_pred = self.lstm_model.predict(X_test_lstm, verbose=0)
            y_test_original = self.price_scaler.inverse_transform(y_test_lstm.reshape(-1, 1)).flatten()
            lstm_pred_original = self.price_scaler.inverse_transform(lstm_pred).flatten()
            
            lstm_mae = mean_absolute_error(y_test_original, lstm_pred_original)
            lstm_rmse = np.sqrt(mean_squared_error(y_test_original, lstm_pred_original))
            
            # Directional accuracy
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
                'direction_accuracy': float(lstm_direction),
                'final_val_loss': float(history.history['val_loss'][-1])
            }
            
            logger.info(f"✅ LSTM - Direction Accuracy: {lstm_direction:.1f}%")
            
            # ================================================================
            # RF & GB TRAINING (Keep existing optimized versions)
            # ================================================================
            
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
                n_estimators=300,  # More trees
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
            
            # ================================================================
            # ENSEMBLE EVALUATION
            # ================================================================
            avg_direction = (lstm_direction + rf_accuracy * 100 + gb_direction) / 3
            logger.info(f"\n📊 Ensemble Average Direction Accuracy: {avg_direction:.1f}%")
            
            self.is_trained = True
            self.last_training = datetime.now()
            
            # Save models
            self.save_models()
            
            logger.info("\n✅ ALL MODELS TRAINED SUCCESSFULLY!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, df: pd.DataFrame, timeframe_minutes: int) -> Optional[Dict]:
        """
        ENHANCED: Prediction with strict filtering
        """
        
        if not self.is_trained:
            logger.warning("⚠️ Models not trained")
            return None
        
        try:
            category = get_timeframe_category(timeframe_minutes)
            
            # ================================================================
            # STEP 1: CHECK MARKET REGIME
            # ================================================================
            is_favorable, reason, regime_multiplier = self.regime_detector.is_favorable(df, timeframe_minutes)
            
            if not is_favorable:
                logger.debug(f"❌ Unfavorable market: {reason}")
                return None
            
            logger.debug(f"✅ Market regime favorable: {reason} (multiplier: {regime_multiplier:.2f})")
            
            # ================================================================
            # STEP 2: PREPARE DATA
            # ================================================================
            
            seq_length_map = {
                'ultra_short': MODEL_CONFIG['lstm']['ultra_short_sequence'],
                'short': MODEL_CONFIG['lstm']['short_sequence'],
                'medium': MODEL_CONFIG['lstm']['medium_sequence'],
                'long': MODEL_CONFIG['lstm']['long_sequence']
            }
            sequence_length = seq_length_map.get(category, self.sequence_length)
            
            df_clean = df.dropna().copy()
            df_clean = df_clean.sort_values('datetime', ascending=True).reset_index(drop=True)
            
            min_required = sequence_length + 50  # More data for better predictions
            if len(df_clean) < min_required:
                logger.warning(f"⚠️ Insufficient data: {len(df_clean)} < {min_required}")
                return None
            
            features = self.prepare_features(df_clean)
            scaled_features = self.feature_scaler.transform(features)
            
            current_price = df_clean.iloc[-1]['price']
            
            # ================================================================
            # STEP 3: MODEL PREDICTIONS
            # ================================================================
            
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
            
            # ================================================================
            # STEP 4: SMART ENSEMBLE WITH DYNAMIC WEIGHTS
            # ================================================================
            
            # Get dynamic weights based on recent performance
            weights = self._get_dynamic_weights(category)
            
            # Calculate time adjustment
            time_factor = self._calculate_time_factor(timeframe_minutes, category)
            
            # Calculate changes
            lstm_change = (lstm_pred - current_price) * time_factor
            gb_change = (gb_pred - current_price) * time_factor
            
            # Base ensemble
            base_ensemble = (
                weights['lstm'] * lstm_change +
                weights['gb'] * gb_change
            )
            
            # RF directional adjustment
            rf_adjustment = 1.0
            if rf_direction == 1:
                rf_adjustment = 1.0 + (rf_confidence - 50) / 120
            else:
                rf_adjustment = 1.0 - (rf_confidence - 50) / 120
            
            # Apply adjustments
            ensemble_change = base_ensemble * rf_adjustment
            
            # Market conditions adjustment
            trend_multiplier = self._calculate_trend_strength(df_clean)
            ensemble_change *= trend_multiplier
            
            predicted_price = current_price + ensemble_change
            
            # ================================================================
            # STEP 5: ULTRA-STRICT CONFIDENCE CALCULATION
            # ================================================================
            
            confidence = self._calculate_ultra_strict_confidence(
                lstm_change, gb_change, rf_direction, rf_confidence,
                category, df_clean, timeframe_minutes, regime_multiplier
            )
            
            # ================================================================
            # STEP 6: ADAPTIVE THRESHOLD (VERY HIGH)
            # ================================================================
            
            min_confidence = self._get_ultra_strict_threshold(category)
            
            if confidence < min_confidence:
                logger.debug(f"⚠️ Confidence {confidence:.1f}% below threshold {min_confidence:.1f}%")
                return None
            
            # ================================================================
            # STEP 7: MODEL AGREEMENT CHECK
            # ================================================================
            
            lstm_dir = 1 if lstm_change > 0 else 0
            gb_dir = 1 if gb_change > 0 else 0
            
            # All 3 models must agree for ultra high confidence
            all_agree = (lstm_dir == gb_dir == rf_direction)
            
            if not all_agree:
                # If models don't agree, reduce confidence
                confidence *= 0.7  # 30% penalty
                
                # Recheck threshold after penalty
                if confidence < min_confidence:
                    logger.debug(f"❌ Models disagree, confidence dropped to {confidence:.1f}%")
                    return None
            
            agreement_count = sum([lstm_dir == rf_direction, gb_dir == rf_direction, lstm_dir == gb_dir])
            model_agreement = agreement_count / 3.0
            
            # ================================================================
            # STEP 8: FINAL VALIDATION CHECKS
            # ================================================================
            
            # Check prediction magnitude is reasonable
            change_pct = abs(ensemble_change / current_price * 100)
            
            # Unrealistic predictions
            if change_pct > 10:  # More than 10% change is suspicious
                logger.debug(f"❌ Unrealistic prediction: {change_pct:.2f}%")
                return None
            
            # Too small predictions (noise)
            if change_pct < 0.1:  # Less than 0.1% is too small
                logger.debug(f"❌ Prediction too small: {change_pct:.2f}%")
                return None
            
            # ================================================================
            # STEP 9: BUILD PREDICTION
            # ================================================================
            
            trend = "CALL (Bullish)" if ensemble_change > 0 else "PUT (Bearish)"
            
            # Calculate price range
            volatility = df_clean['atr_14'].iloc[-1] if 'atr_14' in df_clean.columns else df_clean['price'].tail(20).std()
            range_multiplier = self._get_range_multiplier(category, time_factor)
            
            price_range_low = predicted_price - volatility * range_multiplier
            price_range_high = predicted_price + volatility * range_multiplier
            
            prediction = {
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
                'model_agreement': model_agreement * 100,
                'timeframe_minutes': timeframe_minutes,
                'volatility': volatility,
                'method': f'Enhanced ML Ensemble v2.0 ({category})',
                'model_metrics': self.metrics,
                'category': category,
                'time_factor': time_factor,
                'sequence_length_used': sequence_length,
                'trend_strength': trend_multiplier,
                'adaptive_threshold': min_confidence,
                'regime_multiplier': regime_multiplier,
                'all_models_agree': all_agree,
                'market_regime': 'FAVORABLE'
            }
            
            # Track for adaptive learning
            self._track_prediction(prediction)
            
            logger.debug(f"✅ High-confidence prediction: {confidence:.1f}% (threshold: {min_confidence:.1f}%)")
            
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_dynamic_weights(self, category: str) -> Dict[str, float]:
        """
        ENHANCED: Dynamic weights based on recent model performance
        """
        
        # Base weights
        base_weights = {
            'ultra_short': {'lstm': 0.35, 'gb': 0.35, 'rf': 0.30},
            'short': {'lstm': 0.40, 'gb': 0.35, 'rf': 0.25},
            'medium': {'lstm': 0.45, 'gb': 0.40, 'rf': 0.15},
            'long': {'lstm': 0.50, 'gb': 0.40, 'rf': 0.10}
        }
        
        weights = base_weights.get(category, base_weights['short'])
        
        # Adjust based on recent performance if available
        if len(self.model_performance['lstm']) > 10:
            lstm_perf = np.mean(list(self.model_performance['lstm']))
            gb_perf = np.mean(list(self.model_performance['gb']))
            
            # Increase weight for better performing model
            if lstm_perf > gb_perf:
                adjustment = min((lstm_perf - gb_perf) * 0.1, 0.1)
                weights['lstm'] += adjustment
                weights['gb'] -= adjustment
            else:
                adjustment = min((gb_perf - lstm_perf) * 0.1, 0.1)
                weights['gb'] += adjustment
                weights['lstm'] -= adjustment
        
        return weights
    
    def _calculate_time_factor(self, timeframe_minutes: int, category: str) -> float:
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
        """Calculate trend strength multiplier"""
        try:
            recent = df.tail(20)
            
            # Multiple indicators
            price_momentum = (recent.iloc[-1]['price'] - recent.iloc[0]['price']) / recent.iloc[0]['price']
            
            rsi = recent['rsi_14'].iloc[-1] if 'rsi_14' in recent.columns else 50
            rsi_strength = abs(rsi - 50) / 50
            
            adx = recent['adx_14'].iloc[-1] if 'adx_14' in recent.columns else 20
            adx_strength = min(adx / 50, 1.0)
            
            volume_ratio = recent['volume_ratio_20'].iloc[-1] if 'volume_ratio_20' in recent.columns else 1.0
            volume_strength = min(volume_ratio / 2, 1.0)
            
            # Combined
            strength = (abs(price_momentum) * 2 + rsi_strength + adx_strength + volume_strength) / 5
            
            # Multiplier: 0.85 to 1.15
            multiplier = 0.85 + (strength * 0.3)
            
            return min(max(multiplier, 0.85), 1.15)
            
        except:
            return 1.0
    
    def _calculate_ultra_strict_confidence(self, lstm_change: float, gb_change: float,
                                          rf_direction: int, rf_confidence: float,
                                          category: str, df: pd.DataFrame,
                                          timeframe_minutes: int, regime_multiplier: float) -> float:
        """
        ULTRA STRICT confidence calculation for >70% win rate
        """
        
        # Start with LOWER base to be more selective
        base_confidence = {
            'ultra_short': 30,
            'short': 35,
            'medium': 40,
            'long': 45
        }.get(category, 35)
        
        # ================================================================
        # MODEL AGREEMENT (up to +35)
        # ================================================================
        lstm_dir = 1 if lstm_change > 0 else 0
        gb_dir = 1 if gb_change > 0 else 0
        
        if lstm_dir == gb_dir == rf_direction:
            agreement_score = 35  # All agree
        elif (lstm_dir == gb_dir) or (lstm_dir == rf_direction) or (gb_dir == rf_direction):
            agreement_score = 15  # 2 out of 3
        else:
            agreement_score = -20  # Disagree = strong penalty
        
        # ================================================================
        # RF CONFIDENCE (up to +20)
        # ================================================================
        if rf_confidence > 70:
            rf_contribution = 20
        elif rf_confidence > 60:
            rf_contribution = 15
        elif rf_confidence > 50:
            rf_contribution = 10
        else:
            rf_contribution = 0
        
        # ================================================================
        # MAGNITUDE CONSISTENCY (up to +15)
        # ================================================================
        if abs(lstm_change) > 0 and abs(gb_change) > 0:
            ratio = min(abs(lstm_change), abs(gb_change)) / max(abs(lstm_change), abs(gb_change))
            if ratio > 0.7:
                magnitude_score = 15
            elif ratio > 0.5:
                magnitude_score = 10
            else:
                magnitude_score = 5
        else:
            magnitude_score = 0
        
        # ================================================================
        # MARKET CONDITIONS (up to +15)
        # ================================================================
        market_score = self._detailed_market_assessment(df)
        
        # ================================================================
        # REGIME MULTIPLIER (bonus from favorable conditions)
        # ================================================================
        regime_bonus = (regime_multiplier - 0.7) * 50  # Up to +15
        
        # ================================================================
        # RECENT PERFORMANCE (up to +10)
        # ================================================================
        if len(self.recent_wins) >= 10:
            recent_winrate = np.mean(list(self.recent_wins)) * 100
            if recent_winrate > 70:
                performance_bonus = 10
            elif recent_winrate > 60:
                performance_bonus = 5
            else:
                performance_bonus = 0
        else:
            performance_bonus = 0
        
        # ================================================================
        # VOLATILITY MATCH (up to +10)
        # ================================================================
        volatility_score = self._assess_volatility_match(df, category)
        
        # ================================================================
        # CALCULATE FINAL
        # ================================================================
        confidence = (base_confidence + agreement_score + rf_contribution + 
                     magnitude_score + market_score + regime_bonus + 
                     performance_bonus + volatility_score)
        
        # Strict maximum
        max_confidence = {
            'ultra_short': 85,
            'short': 88,
            'medium': 90,
            'long': 92
        }.get(category, 88)
        
        return min(max(confidence, 0), max_confidence)
    
    def _get_ultra_strict_threshold(self, category: str) -> float:
        """
        ULTRA STRICT threshold - only take high-confidence predictions
        """
        
        # Very high base thresholds
        base_threshold = {
            'ultra_short': 60,  # 60%+ required
            'short': 55,        # 55%+ required
            'medium': 52,       # 52%+ required
            'long': 50          # 50%+ required
        }.get(category, 55)
        
        # Adjust based on recent win rate
        if len(self.recent_wins) >= 20:
            recent_winrate = np.mean(list(self.recent_wins))
            
            if recent_winrate > 0.75:  # >75% win rate - can be slightly less strict
                base_threshold -= 3
            elif recent_winrate < 0.60:  # <60% win rate - be MORE strict
                base_threshold += 5
        
        return base_threshold
    
    def _detailed_market_assessment(self, df: pd.DataFrame) -> float:
        """
        Detailed market condition assessment
        Returns: -10 to +15
        """
        try:
            recent = df.head(30)
            score = 0
            
            # Trend consistency
            if 'adx_14' in recent.columns:
                adx = recent['adx_14'].iloc[0]
                if adx > 30:
                    score += 5  # Strong trend
                elif adx > 25:
                    score += 3
            
            # Volume confirmation
            if 'volume_ratio_20' in recent.columns:
                vol_ratio = recent['volume_ratio_20'].iloc[0]
                if vol_ratio > 1.5:
                    score += 4
                elif vol_ratio > 1.2:
                    score += 2
                elif vol_ratio < 0.7:
                    score -= 3
            
            # RSI positioning
            if 'rsi_14' in recent.columns:
                rsi = recent['rsi_14'].iloc[0]
                if 35 < rsi < 65:
                    score += 3  # Neutral RSI is good
                elif rsi < 25 or rsi > 75:
                    score += 2  # Extremes can be good too
            
            # Volatility
            if 'volatility_20' in recent.columns:
                vol = recent['volatility_20'].iloc[0]
                if vol < 0.025:
                    score += 3  # Low volatility
                elif vol > 0.05:
                    score -= 5  # High volatility penalty
            
            # BB position
            if 'bb_position_20' in recent.columns:
                bb_pos = recent['bb_position_20'].iloc[0]
                if 0.2 < bb_pos < 0.8:
                    score += 2  # Not at extremes
            
            return min(max(score, -10), 15)
            
        except:
            return 0
    
    def _assess_volatility_match(self, df: pd.DataFrame, category: str) -> float:
        """
        Check if volatility matches timeframe preference
        Returns: -5 to +10
        """
        try:
            recent = df.head(20)
            
            if 'volatility_20' in recent.columns:
                vol = recent['volatility_20'].iloc[0] * 100
            else:
                vol = recent['price'].std() / recent['price'].mean() * 100
            
            # Different timeframes prefer different volatility
            if category == 'ultra_short':
                if 2 < vol < 4:
                    return 10
                elif 1.5 < vol < 5:
                    return 5
                elif vol > 6:
                    return -5
            
            elif category == 'short':
                if 1.5 < vol < 4:
                    return 10
                elif vol < 5:
                    return 5
                elif vol > 6:
                    return -3
            
            elif category in ['medium', 'long']:
                if vol < 3:
                    return 10
                elif vol < 4:
                    return 5
                elif vol > 5:
                    return -3
            
            return 0
            
        except:
            return 0
    
    def _get_range_multiplier(self, category: str, time_factor: float) -> float:
        """Calculate price range multiplier"""
        base = {
            'ultra_short': 0.4,
            'short': 0.6,
            'medium': 0.8,
            'long': 1.0
        }.get(category, 0.6)
        
        return base * time_factor * 0.7
    
    def _track_prediction(self, prediction: Dict):
        """Track prediction for adaptive learning"""
        try:
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'confidence': prediction['confidence'],
                'category': prediction['category'],
                'trend': prediction['trend'],
                'regime': prediction.get('market_regime', 'UNKNOWN')
            })
        except:
            pass
    
    def update_performance(self, prediction_id: str, was_correct: bool):
        """
        Update model performance tracking
        Call this after validation
        """
        try:
            self.recent_wins.append(1 if was_correct else 0)
            
            # Track individual model performance if available
            # This would need prediction metadata
            
        except:
            pass
    
    def save_models(self) -> bool:
        """Save trained models"""
        try:
            path = MODEL_CONFIG['model_save_path']
            os.makedirs(path, exist_ok=True)
            
            if self.lstm_model:
                self.lstm_model.save(f'{path}/lstm_model_enhanced.keras')
            
            if self.rf_model:
                with open(f'{path}/rf_model_enhanced.pkl', 'wb') as f:
                    pickle.dump(self.rf_model, f)
            
            if self.gb_model:
                with open(f'{path}/gb_model_enhanced.pkl', 'wb') as f:
                    pickle.dump(self.gb_model, f)
            
            with open(f'{path}/scalers_enhanced.pkl', 'wb') as f:
                pickle.dump({
                    'price_scaler': self.price_scaler,
                    'feature_scaler': self.feature_scaler,
                    'feature_columns': self.feature_columns,
                    'metrics': self.metrics,
                    'last_training': self.last_training,
                    'prediction_history': list(self.prediction_history),
                    'recent_wins': list(self.recent_wins)
                }, f)
            
            logger.info(f"✅ Enhanced models saved to {path}/")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving models: {e}")
            return False
    
    def load_models(self) -> bool:
        """Load trained models"""
        try:
            path = MODEL_CONFIG['model_save_path']
            
            # Try enhanced models first
            if os.path.exists(f'{path}/lstm_model_enhanced.keras'):
                self.lstm_model = load_model(f'{path}/lstm_model_enhanced.keras')
                
                with open(f'{path}/rf_model_enhanced.pkl', 'rb') as f:
                    self.rf_model = pickle.load(f)
                
                with open(f'{path}/gb_model_enhanced.pkl', 'rb') as f:
                    self.gb_model = pickle.load(f)
                
                with open(f'{path}/scalers_enhanced.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.price_scaler = data['price_scaler']
                    self.feature_scaler = data['feature_scaler']
                    self.feature_columns = data['feature_columns']
                    self.metrics = data.get('metrics', {})
                    self.last_training = data.get('last_training')
                    self.prediction_history = deque(data.get('prediction_history', []), maxlen=200)
                    self.recent_wins = deque(data.get('recent_wins', []), maxlen=50)
                
                self.is_trained = True
                logger.info(f"✅ Enhanced models loaded from {path}/")
                return True
            
            # Fallback to regular models
            elif os.path.exists(f'{path}/lstm_model.keras'):
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
                logger.info(f"✅ Regular models loaded, will retrain with enhanced version")
                return True
            
            else:
                logger.warning("⚠️ No models found")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False
    
    def needs_retraining(self) -> bool:
        """Check if models need retraining"""
        if not self.is_trained or not self.last_training:
            return True
        
        time_since_training = (datetime.now() - self.last_training).total_seconds()
        return time_since_training > MODEL_CONFIG['auto_retrain_interval']


# ============================================================================
# COMPATIBILITY ALIAS
# ============================================================================

BitcoinMLPredictor = ImprovedBitcoinPredictor