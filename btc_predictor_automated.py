"""
Improved Bitcoin Price Predictor with Validation
Better ML approach with proper validation and confidence calculation
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Optional
import pickle
import os

from config import MODEL_CONFIG, get_timeframe_category, get_min_confidence
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    import tensorflow as tf
    from tensorflow import keras
    from keras.models import Sequential, load_model
    from keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from keras.optimizers import Adam
    from keras.regularizers import l2
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    logger.error(f"❌ ML libraries not available: {e}")


class ImprovedBitcoinPredictor:
    """Improved ML predictor with validation"""
    
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
        self.validation_scores = {}
        self.last_training = None
        
        logger.info("🤖 Improved predictor initialized")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature set with better selection"""
        
        # Core features - most important
        core_features = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_position',
            'stoch_k', 'stoch_d',
            'atr',
            'volume_ratio',
            'momentum',
        ]
        
        # Price features
        price_features = [
            'ema_9', 'ema_21', 'ema_50',
            'price_above_sma20', 'price_above_sma50',
            'ema_trend'
        ]
        
        # Lag features - reduced to avoid data leakage
        lag_features = []
        for lag in [1, 3, 5]:  # Reduced from [1,3,5,10]
            lag_features.extend([
                f'price_change_{lag}',
                f'volume_lag_{lag}'
            ])
        
        # Rolling features - most relevant windows
        rolling_features = []
        for window in [5, 20]:  # Reduced from [5,10,20,50]
            rolling_features.extend([
                f'price_rolling_mean_{window}',
                f'price_rolling_std_{window}',
                f'volume_rolling_mean_{window}'
            ])
        
        # Combine all features
        all_features = core_features + price_features + lag_features + rolling_features
        
        # Filter to available features
        available = [col for col in all_features if col in df.columns]
        self.feature_columns = available
        
        return df[available].copy()
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray, 
                        sequence_length: int):
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(target[i + sequence_length])
        return np.array(X), np.array(y)
    
    def build_improved_lstm(self, input_shape: tuple) -> Sequential:
        """Build improved LSTM with better architecture"""
        
        model = Sequential([
            # First LSTM layer
            Bidirectional(LSTM(
                96, 
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.1,
                kernel_regularizer=l2(0.001)
            ), input_shape=input_shape),
            BatchNormalization(),
            
            # Second LSTM layer
            Bidirectional(LSTM(
                64,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.1,
                kernel_regularizer=l2(0.001)
            )),
            BatchNormalization(),
            
            # Third LSTM layer
            Bidirectional(LSTM(
                32,
                dropout=0.2,
                kernel_regularizer=l2(0.001)
            )),
            BatchNormalization(),
            
            # Dense layers
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.2),
            Dense(1)
        ])
        
        # Custom learning rate
        optimizer = Adam(
            learning_rate=0.0005,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train_models(self, df: pd.DataFrame, epochs: int = 50, 
                    batch_size: int = 64) -> bool:
        """Train models with proper validation"""
        
        if not ML_AVAILABLE:
            logger.error("❌ ML libraries not available")
            return False
        
        try:
            logger.info("\n🤖 TRAINING IMPROVED MODELS...")
            
            # Prepare data
            df_clean = df.dropna().copy()
            df_clean = df_clean.sort_values('datetime', ascending=True).reset_index(drop=True)
            
            if len(df_clean) < 500:  # Increased minimum
                logger.error(f"❌ Insufficient data: {len(df_clean)} (need 500+)")
                return False
            
            # Prepare features
            features = self.prepare_features(df_clean)
            target = df_clean['price'].values
            
            # Scale data
            scaled_features = self.feature_scaler.fit_transform(features)
            scaled_target = self.price_scaler.fit_transform(
                target.reshape(-1, 1)
            ).flatten()
            
            # Time series split for proper validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Store validation scores
            lstm_scores = []
            rf_scores = []
            gb_scores = []
            
            logger.info(f"📊 Using TimeSeriesSplit with 3 folds")
            
            # Train on last fold (most recent data)
            splits = list(tscv.split(scaled_features))
            train_idx, test_idx = splits[-1]
            
            # LSTM Training
            logger.info("\n🔵 Training LSTM...")
            X_lstm, y_lstm = self.create_sequences(
                scaled_features, 
                scaled_target, 
                self.sequence_length
            )
            
            X_train_lstm = X_lstm[train_idx[:-self.sequence_length]]
            X_test_lstm = X_lstm[test_idx[:-self.sequence_length]]
            y_train_lstm = y_lstm[train_idx[:-self.sequence_length]]
            y_test_lstm = y_lstm[test_idx[:-self.sequence_length]]
            
            self.lstm_model = self.build_improved_lstm(
                (self.sequence_length, len(self.feature_columns))
            )
            
            # Callbacks
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=MODEL_CONFIG['lstm']['patience'],
                restore_best_weights=True,
                verbose=1
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
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
            
            # Train LSTM
            history = self.lstm_model.fit(
                X_train_lstm, y_train_lstm,
                validation_data=(X_test_lstm, y_test_lstm),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop, reduce_lr, checkpoint],
                verbose=1
            )
            
            # Evaluate LSTM
            lstm_pred = self.lstm_model.predict(X_test_lstm, verbose=0)
            y_test_original = self.price_scaler.inverse_transform(
                y_test_lstm.reshape(-1, 1)
            ).flatten()
            lstm_pred_original = self.price_scaler.inverse_transform(lstm_pred).flatten()
            
            lstm_mae = mean_absolute_error(y_test_original, lstm_pred_original)
            lstm_rmse = np.sqrt(mean_squared_error(y_test_original, lstm_pred_original))
            lstm_mape = np.mean(np.abs((y_test_original - lstm_pred_original) / y_test_original)) * 100
            
            self.metrics['lstm'] = {
                'mae': float(lstm_mae),
                'rmse': float(lstm_rmse),
                'mape': float(lstm_mape),
                'final_val_loss': float(history.history['val_loss'][-1])
            }
            
            logger.info(f"✅ LSTM - MAE: ${lstm_mae:,.2f}, RMSE: ${lstm_rmse:,.2f}, MAPE: {lstm_mape:.2f}%")
            
            # Random Forest Training
            logger.info("\n🌲 Training Random Forest...")
            
            # Classification target (direction)
            y_class = (df_clean['price'].shift(-1) > df_clean['price']).astype(int)
            y_class = y_class[:-1]
            features_rf = scaled_features[:-1]
            
            X_train_rf = features_rf[train_idx]
            X_test_rf = features_rf[test_idx]
            y_train_rf = y_class[train_idx]
            y_test_rf = y_class[test_idx]
            
            self.rf_model = RandomForestClassifier(
                n_estimators=MODEL_CONFIG['rf']['n_estimators'],
                max_depth=MODEL_CONFIG['rf']['max_depth'],
                min_samples_split=MODEL_CONFIG['rf']['min_samples_split'],
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',  # Handle imbalanced data
                verbose=0
            )
            
            self.rf_model.fit(X_train_rf, y_train_rf)
            
            # Evaluate RF
            rf_pred = self.rf_model.predict(X_test_rf)
            rf_accuracy = accuracy_score(y_test_rf, rf_pred)
            rf_proba = self.rf_model.predict_proba(X_test_rf)
            
            self.metrics['rf'] = {
                'accuracy': float(rf_accuracy),
                'feature_importance_top': float(self.rf_model.feature_importances_.max())
            }
            
            logger.info(f"✅ RF - Accuracy: {rf_accuracy:.4f}")
            
            # Gradient Boosting Training
            logger.info("\n🚀 Training Gradient Boosting...")
            
            X_train_gb = features_rf[train_idx]
            X_test_gb = features_rf[test_idx]
            y_train_gb = scaled_target[:-1][train_idx]
            y_test_gb = scaled_target[:-1][test_idx]
            
            self.gb_model = GradientBoostingRegressor(
                n_estimators=MODEL_CONFIG['gb']['n_estimators'],
                learning_rate=MODEL_CONFIG['gb']['learning_rate'],
                max_depth=MODEL_CONFIG['gb']['max_depth'],
                subsample=0.8,
                random_state=42,
                verbose=0
            )
            
            self.gb_model.fit(X_train_gb, y_train_gb)
            
            # Evaluate GB
            gb_pred_scaled = self.gb_model.predict(X_test_gb)
            gb_pred = self.price_scaler.inverse_transform(
                gb_pred_scaled.reshape(-1, 1)
            ).flatten()
            y_test_gb_original = self.price_scaler.inverse_transform(
                y_test_gb.reshape(-1, 1)
            ).flatten()
            
            gb_mae = mean_absolute_error(y_test_gb_original, gb_pred)
            gb_rmse = np.sqrt(mean_squared_error(y_test_gb_original, gb_pred))
            
            self.metrics['gb'] = {
                'mae': float(gb_mae),
                'rmse': float(gb_rmse)
            }
            
            logger.info(f"✅ GB - MAE: ${gb_mae:,.2f}, RMSE: ${gb_rmse:,.2f}")
            
            # Mark as trained
            self.is_trained = True
            self.last_training = datetime.now()
            
            # Validate models
            if MODEL_CONFIG.get('enable_model_validation'):
                self._validate_models()
            
            # Save models
            self.save_models()
            
            logger.info("\n✅ ALL MODELS TRAINED SUCCESSFULLY!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _validate_models(self):
        """Validate model quality"""
        logger.info("\n🔍 Validating model quality...")
        
        min_score = MODEL_CONFIG.get('min_validation_score', 0.6)
        
        # Check LSTM
        if 'lstm' in self.metrics:
            lstm_valid = self.metrics['lstm']['mape'] < 5.0  # MAPE < 5%
            logger.info(f"{'✅' if lstm_valid else '⚠️'} LSTM validation: MAPE {self.metrics['lstm']['mape']:.2f}%")
        
        # Check RF
        if 'rf' in self.metrics:
            rf_valid = self.metrics['rf']['accuracy'] > min_score
            logger.info(f"{'✅' if rf_valid else '⚠️'} RF validation: Accuracy {self.metrics['rf']['accuracy']:.2%}")
        
        # Check GB
        if 'gb' in self.metrics:
            gb_valid = self.metrics['gb']['mae'] < 1000  # MAE < $1000
            logger.info(f"{'✅' if gb_valid else '⚠️'} GB validation: MAE ${self.metrics['gb']['mae']:.2f}")
    
    def predict(self, df: pd.DataFrame, timeframe_minutes: int) -> Optional[Dict]:
        """Make prediction with improved confidence calculation"""
        
        if not self.is_trained:
            logger.warning("⚠️ Models not trained")
            return None
        
        try:
            category = get_timeframe_category(timeframe_minutes)
            
            # Get appropriate sequence length
            seq_length_map = {
                'ultra_short': MODEL_CONFIG['lstm']['ultra_short_sequence'],
                'short': MODEL_CONFIG['lstm']['short_sequence'],
                'medium': MODEL_CONFIG['lstm']['medium_sequence'],
                'long': MODEL_CONFIG['lstm']['long_sequence']
            }
            sequence_length = seq_length_map.get(category, self.sequence_length)
            
            # Prepare data
            df_clean = df.dropna().copy()
            df_clean = df_clean.sort_values('datetime', ascending=True).reset_index(drop=True)
            
            min_required = sequence_length + 20
            if len(df_clean) < min_required:
                logger.warning(f"⚠️ Insufficient data: {len(df_clean)} < {min_required}")
                return None
            
            # Prepare features
            features = self.prepare_features(df_clean)
            scaled_features = self.feature_scaler.transform(features)
            
            current_price = df_clean.iloc[-1]['price']
            
            # LSTM prediction
            lstm_input = scaled_features[-sequence_length:].reshape(1, sequence_length, -1)
            lstm_pred_scaled = self.lstm_model.predict(lstm_input, verbose=0)[0][0]
            lstm_pred = self.price_scaler.inverse_transform([[lstm_pred_scaled]])[0][0]
            
            # RF prediction
            rf_input = scaled_features[-1:].reshape(1, -1)
            rf_direction = self.rf_model.predict(rf_input)[0]
            rf_proba = self.rf_model.predict_proba(rf_input)[0]
            rf_confidence = max(rf_proba) * 100
            
            # GB prediction
            gb_pred_scaled = self.gb_model.predict(rf_input)[0]
            gb_pred = self.price_scaler.inverse_transform([[gb_pred_scaled]])[0][0]
            
            # Calculate time adjustment factor
            time_factor = self._calculate_time_factor(timeframe_minutes, category)
            
            # Calculate predicted changes
            lstm_change = (lstm_pred - current_price) * time_factor
            gb_change = (gb_pred - current_price) * time_factor
            
            # Ensemble with improved weights
            weights = self._get_ensemble_weights(category)
            ensemble_change = (
                weights['lstm'] * lstm_change +
                weights['gb'] * gb_change
            ) * (1 + (rf_confidence - 50) / 200)  # Reduced RF impact
            
            predicted_price = current_price + ensemble_change
            
            # Calculate confidence (improved method)
            confidence = self._calculate_confidence(
                lstm_change, gb_change, rf_direction, rf_confidence,
                category, df_clean
            )
            
            # Check minimum confidence threshold
            min_confidence = get_min_confidence(timeframe_minutes)
            if confidence < min_confidence:
                logger.debug(f"⚠️ Confidence {confidence:.1f}% below threshold {min_confidence}%")
                return None
            
            # Determine trend
            trend = "CALL (Bullish)" if ensemble_change > 0 else "PUT (Bearish)"
            
            # Calculate price range with volatility
            volatility = df_clean['price'].tail(20).std()
            range_multiplier = self._get_range_multiplier(category, time_factor)
            
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
                'timeframe_minutes': timeframe_minutes,
                'volatility': volatility,
                'method': f'Improved ML Ensemble ({category})',
                'model_metrics': self.metrics,
                'category': category,
                'time_factor': time_factor,
                'sequence_length_used': sequence_length
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_time_factor(self, timeframe_minutes: int, category: str) -> float:
        """Calculate time adjustment factor"""
        if category == 'ultra_short':
            return min(timeframe_minutes / 45, 0.8)  # More conservative
        elif category == 'short':
            return min(timeframe_minutes / 60, 1.2)
        elif category == 'medium':
            return min(timeframe_minutes / 240, 1.5)
        else:  # long
            return min(timeframe_minutes / 1440, 2.0)
    
    def _get_ensemble_weights(self, category: str) -> Dict[str, float]:
        """Get ensemble weights based on category"""
        weights = {
            'ultra_short': {'lstm': 0.40, 'gb': 0.35, 'rf': 0.25},
            'short': {'lstm': 0.45, 'gb': 0.40, 'rf': 0.15},
            'medium': {'lstm': 0.50, 'gb': 0.40, 'rf': 0.10},
            'long': {'lstm': 0.55, 'gb': 0.35, 'rf': 0.10}
        }
        return weights.get(category, {'lstm': 0.45, 'gb': 0.40, 'rf': 0.15})
    
    def _calculate_confidence(self, lstm_change: float, gb_change: float,
                             rf_direction: int, rf_confidence: float,
                             category: str, df: pd.DataFrame) -> float:
        """Calculate prediction confidence (improved method)"""
        
        # Base confidence from category
        base_confidence = {
            'ultra_short': 40,
            'short': 45,
            'medium': 50,
            'long': 55
        }.get(category, 45)
        
        # Model agreement
        lstm_dir = 1 if lstm_change > 0 else 0
        gb_dir = 1 if gb_change > 0 else 0
        
        agreement_score = 0
        if lstm_dir == gb_dir == rf_direction:
            agreement_score = 25  # All agree
        elif (lstm_dir == gb_dir) or (lstm_dir == rf_direction) or (gb_dir == rf_direction):
            agreement_score = 15  # 2 out of 3 agree
        else:
            agreement_score = 0  # Disagree
        
        # RF confidence contribution (reduced weight)
        rf_contribution = (rf_confidence - 50) * 0.2
        
        # Recent performance (if available)
        performance_bonus = 0
        if hasattr(self, 'recent_accuracy') and self.recent_accuracy:
            performance_bonus = (self.recent_accuracy - 50) * 0.1
        
        # Market condition assessment
        market_bonus = self._assess_market_conditions(df)
        
        # Calculate final confidence
        confidence = base_confidence + agreement_score + rf_contribution + performance_bonus + market_bonus
        
        # Cap confidence
        max_confidence = {
            'ultra_short': 75,
            'short': 80,
            'medium': 85,
            'long': 90
        }.get(category, 80)
        
        return min(confidence, max_confidence)
    
    def _assess_market_conditions(self, df: pd.DataFrame) -> float:
        """Assess market conditions for confidence adjustment"""
        try:
            recent_data = df.tail(20)
            
            # Check trend strength
            price_change = (recent_data.iloc[-1]['price'] - recent_data.iloc[0]['price']) / recent_data.iloc[0]['price']
            trend_strength = abs(price_change) * 100
            
            # Check volatility
            volatility = recent_data['price'].std() / recent_data['price'].mean() * 100
            
            # Strong trend + low volatility = higher confidence
            if trend_strength > 2 and volatility < 2:
                return 5
            elif trend_strength > 1 and volatility < 3:
                return 3
            elif volatility > 5:  # High volatility = lower confidence
                return -5
            
            return 0
            
        except:
            return 0
    
    def _get_range_multiplier(self, category: str, time_factor: float) -> float:
        """Get price range multiplier"""
        base_multiplier = {
            'ultra_short': 0.4,
            'short': 0.6,
            'medium': 0.8,
            'long': 1.0
        }.get(category, 0.6)
        
        return base_multiplier * time_factor
    
    def save_models(self) -> bool:
        """Save trained models"""
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
                    'last_training': self.last_training,
                    'validation_scores': self.validation_scores
                }, f)
            
            logger.info(f"✅ Models saved to {path}/")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving models: {e}")
            return False
    
    def load_models(self) -> bool:
        """Load trained models"""
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
                self.validation_scores = data.get('validation_scores', {})
            
            self.is_trained = True
            logger.info(f"✅ Models loaded from {path}/")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False
    
    def needs_retraining(self) -> bool:
        """Check if models need retraining"""
        if not self.is_trained or not self.last_training:
            return True
        
        time_since_training = (datetime.now() - self.last_training).total_seconds()
        return time_since_training > MODEL_CONFIG['auto_retrain_interval']