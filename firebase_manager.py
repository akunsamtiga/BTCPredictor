"""
Firebase Manager for Bitcoin Predictor - FIXED VALIDATION
ALL TIMESTAMPS IN WIB (UTC+7)
"""

import firebase_admin
from firebase_admin import credentials, firestore, db
from datetime import datetime, timedelta
import logging
import json
import time
from typing import Dict, List, Optional
import pandas as pd
from config import FIREBASE_CONFIG, FIREBASE_COLLECTIONS
from timezone_utils import (
    get_local_now, 
    get_local_isoformat, 
    prepare_firebase_timestamp,
    add_minutes_local,
    format_firebase_timestamp,
    parse_iso_to_local,
    now_iso_wib
)

logger = logging.getLogger(__name__)

class FirebaseManager:
    """Manages all Firebase operations with auto-reconnect"""
    
    def __init__(self):
        """Initialize Firebase connection"""
        self.db = None
        self.firestore_db = None
        self.connected = False
        self.last_connection_attempt = None
        self.connection_failures = 0
        self._initialize_firebase()
    
    def _initialize_firebase(self, retry=True):
        """Initialize Firebase Admin SDK with retry logic"""
        max_retries = FIREBASE_CONFIG['max_retries']
        retry_delay = FIREBASE_CONFIG['retry_delay']
        
        for attempt in range(max_retries):
            try:
                if not firebase_admin._apps:
                    cred = credentials.Certificate(FIREBASE_CONFIG['credentials_path'])
                    firebase_admin.initialize_app(cred, {
                        'databaseURL': FIREBASE_CONFIG['database_url']
                    })
                
                self.db = db
                self.firestore_db = firestore.client()
                self.connected = True
                self.connection_failures = 0
                self.last_connection_attempt = get_local_now()
                
                logger.info("✅ Firebase initialized successfully")
                return True
                
            except FileNotFoundError:
                logger.error(f"❌ Firebase credentials file not found: {FIREBASE_CONFIG['credentials_path']}")
                raise
            except Exception as e:
                self.connection_failures += 1
                logger.warning(f"⚠️ Firebase connection attempt {attempt + 1}/{max_retries} failed: {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    logger.error(f"❌ Failed to initialize Firebase after {max_retries} attempts")
                    if not retry:
                        raise
                    return False
        
        return False
    
    def _ensure_connection(self):
        """Ensure Firebase connection is active"""
        if not self.connected:
            logger.warning("⚠️ Firebase not connected, attempting reconnection...")
            return self._initialize_firebase(retry=True)
        return True
    
    def _execute_with_retry(self, operation, *args, **kwargs):
        """Execute Firebase operation with retry logic"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                if not self._ensure_connection():
                    raise ConnectionError("Firebase not connected")
                
                result = operation(*args, **kwargs)
                return result, None
                
            except Exception as e:
                logger.warning(f"⚠️ Operation failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    self.connected = False
                    self._ensure_connection()
                else:
                    logger.error(f"❌ Operation failed after {max_retries} attempts: {e}")
                    return None, e
        
        return None, Exception("Max retries exceeded")
    
    def save_prediction(self, prediction: Dict) -> Optional[str]:
        """Save prediction to Firebase with retry - ALL TIMES IN WIB"""
        def _save():
            collection = self.firestore_db.collection(FIREBASE_COLLECTIONS['predictions'])
            
            # All times in WIB
            now_wib = get_local_now()
            target_time_wib = add_minutes_local(now_wib, prediction['timeframe_minutes'])
            
            doc_data = {
                'timestamp': now_iso_wib(),
                'prediction_time': prepare_firebase_timestamp(now_wib),
                'timeframe_minutes': prediction['timeframe_minutes'],
                'current_price': float(prediction['current_price']),
                'predicted_price': float(prediction['predicted_price']),
                'price_change': float(prediction['price_change']),
                'price_change_pct': float(prediction['price_change_pct']),
                'price_range_low': float(prediction['price_range_low']),
                'price_range_high': float(prediction['price_range_high']),
                'trend': prediction['trend'],
                'confidence': float(prediction['confidence']),
                'method': prediction['method'],
                'target_time': prepare_firebase_timestamp(target_time_wib),
                'validated': False,
                'validation_result': None,
                'actual_price': None,
            }
            
            if 'model_agreement' in prediction:
                doc_data.update({
                    'model_agreement': prediction['model_agreement'],
                    'lstm_prediction': float(prediction['lstm_prediction']),
                    'gb_prediction': float(prediction['gb_prediction']),
                    'rf_direction': prediction['rf_direction'],
                    'rf_confidence': float(prediction['rf_confidence']),
                })
            
            doc_ref = collection.add(doc_data)
            return doc_ref[1].id
        
        result, error = self._execute_with_retry(_save)
        
        if result:
            logger.info(f"✅ Prediction saved: {result} (Timeframe: {prediction['timeframe_minutes']}min)")
            return result
        else:
            logger.error(f"❌ Failed to save prediction: {error}")
            self._log_error("save_prediction", error)
            return None
    
    def get_unvalidated_predictions(self) -> List[Dict]:
        """Get all predictions that haven't been validated yet"""
        def _get():
            collection = self.firestore_db.collection(FIREBASE_COLLECTIONS['predictions'])
            now_wib = get_local_now()
            
            query = collection.where('validated', '==', False).limit(100)
            docs = query.stream()
            
            predictions = []
            for doc in docs:
                data = doc.to_dict()
                data['doc_id'] = doc.id
                
                # Parse target time (stored as WIB ISO)
                target_time_str = data.get('target_time')
                if target_time_str:
                    target_time_wib = parse_iso_to_local(target_time_str)
                    
                    # Check if time has passed
                    if target_time_wib <= now_wib:
                        predictions.append(data)
            
            return predictions
        
        result, error = self._execute_with_retry(_get)
        
        if result is not None:
            logger.info(f"📋 Found {len(result)} predictions ready for validation")
            return result
        else:
            logger.warning(f"⚠️ Failed to get unvalidated predictions: {error}")
            return []
    
    def validate_prediction(self, doc_id: str, actual_price: float, 
                          predicted_price: float, current_price: float, trend: str) -> bool:
        """
        FIXED: Validate a prediction correctly
        
        Logic:
        - CALL (UP) = prediksi harga naik dari current_price
        - WIN jika actual_price > current_price
        - PUT (DOWN) = prediksi harga turun dari current_price  
        - WIN jika actual_price < current_price
        """
        def _validate():
            # Tentukan prediksi direction
            predicted_direction = 'up' if 'CALL' in trend.upper() or 'UP' in trend.upper() else 'down'
            
            # FIXED: Bandingkan actual_price dengan CURRENT_PRICE (bukan predicted_price)
            # Ini adalah harga SAAT PREDIKSI dibuat vs harga ACTUAL
            if predicted_direction == 'up':
                # Prediksi NAIK - WIN jika actual > current
                is_win = actual_price > current_price
            else:
                # Prediksi TURUN - WIN jika actual < current
                is_win = actual_price < current_price
            
            # Calculate errors
            price_error = abs(actual_price - predicted_price)
            price_error_pct = (price_error / predicted_price) * 100
            
            # Calculate actual change
            actual_change = actual_price - current_price
            actual_change_pct = (actual_change / current_price) * 100
            
            doc_ref = self.firestore_db.collection(FIREBASE_COLLECTIONS['predictions']).document(doc_id)
            doc_ref.update({
                'validated': True,
                'validation_time': now_iso_wib(),
                'actual_price': float(actual_price),
                'validation_result': 'WIN' if is_win else 'LOSE',
                'price_error': float(price_error),
                'price_error_pct': float(price_error_pct),
                'direction_correct': is_win,
                'actual_change': float(actual_change),
                'actual_change_pct': float(actual_change_pct),
                'predicted_direction': predicted_direction
            })
            
            validation_data = {
                'timestamp': now_iso_wib(),
                'prediction_id': doc_id,
                'result': 'WIN' if is_win else 'LOSE',
                'current_price': float(current_price),
                'predicted_price': float(predicted_price),
                'actual_price': float(actual_price),
                'predicted_direction': predicted_direction,
                'actual_change': float(actual_change),
                'actual_change_pct': float(actual_change_pct),
                'error': float(price_error),
                'error_pct': float(price_error_pct)
            }
            
            self.firestore_db.collection(FIREBASE_COLLECTIONS['validation']).add(validation_data)
            
            return is_win
        
        result, error = self._execute_with_retry(_validate)
        
        if result is not None:
            result_emoji = "✅" if result else "❌"
            direction = "UP" if 'CALL' in trend.upper() or 'UP' in trend.upper() else "DOWN"
            logger.info(f"{result_emoji} {doc_id[:8]}: Predicted {direction} | "
                       f"Current: ${current_price:.0f} → Actual: ${actual_price:.0f} | "
                       f"{'WIN' if result else 'LOSE'}")
            return True
        else:
            logger.error(f"❌ Failed to validate prediction: {error}")
            return False
    
    def get_statistics(self, timeframe_minutes: Optional[int] = None, 
                      days: int = 7) -> Dict:
        """Get prediction statistics"""
        def _get_stats():
            collection = self.firestore_db.collection(FIREBASE_COLLECTIONS['predictions'])
            cutoff_date_wib = get_local_now() - timedelta(days=days)
            
            query = collection.where('validated', '==', True)
            if timeframe_minutes:
                query = query.where('timeframe_minutes', '==', timeframe_minutes)
            
            docs = query.stream()
            
            total = 0
            wins = 0
            losses = 0
            total_error = 0
            total_error_pct = 0
            
            for doc in docs:
                data = doc.to_dict()
                
                # Parse prediction time (WIB)
                if 'prediction_time' in data:
                    pred_time_wib = parse_iso_to_local(data['prediction_time'])
                    if pred_time_wib < cutoff_date_wib:
                        continue
                
                total += 1
                
                if data.get('validation_result') == 'WIN':
                    wins += 1
                else:
                    losses += 1
                
                if 'price_error' in data:
                    total_error += data['price_error']
                if 'price_error_pct' in data:
                    total_error_pct += data['price_error_pct']
            
            win_rate = (wins / total * 100) if total > 0 else 0
            avg_error = total_error / total if total > 0 else 0
            avg_error_pct = total_error_pct / total if total > 0 else 0
            
            return {
                'timeframe_minutes': timeframe_minutes,
                'period_days': days,
                'total_predictions': total,
                'wins': wins,
                'losses': losses,
                'win_rate': round(win_rate, 2),
                'avg_error': round(avg_error, 2),
                'avg_error_pct': round(avg_error_pct, 2),
                'last_updated': now_iso_wib()
            }
        
        result, error = self._execute_with_retry(_get_stats)
        
        if result:
            logger.info(f"📊 Statistics: Win rate {result['win_rate']:.1f}% ({result['wins']}/{result['total_predictions']})")
            return result
        else:
            logger.warning(f"⚠️ Failed to get statistics: {error}")
            return {}
    
    def save_statistics(self, stats: Dict) -> bool:
        """Save statistics to Firebase"""
        def _save():
            collection = self.firestore_db.collection(FIREBASE_COLLECTIONS['statistics'])
            doc_id = f"stats_{get_local_now().strftime('%Y%m%d_%H%M%S')}"
            collection.document(doc_id).set(stats)
            return doc_id
        
        result, error = self._execute_with_retry(_save)
        
        if result:
            logger.info(f"✅ Statistics saved: {result}")
            return True
        else:
            logger.warning(f"⚠️ Failed to save statistics: {error}")
            return False
    
    def save_model_performance(self, metrics: Dict) -> bool:
        """Save model performance metrics"""
        def _save():
            collection = self.firestore_db.collection(FIREBASE_COLLECTIONS['model_performance'])
            doc_data = {
                'timestamp': now_iso_wib(),
                'metrics': metrics
            }
            collection.add(doc_data)
            return True
        
        result, error = self._execute_with_retry(_save)
        
        if result:
            logger.info("✅ Model performance saved")
            return True
        else:
            logger.warning(f"⚠️ Failed to save model performance: {error}")
            return False
    
    def save_system_health(self, health_data: Dict) -> bool:
        """Save system health metrics"""
        def _save():
            collection = self.firestore_db.collection(FIREBASE_COLLECTIONS['system_health'])
            doc_data = {
                'timestamp': now_iso_wib(),
                **health_data
            }
            collection.add(doc_data)
            return True
        
        result, error = self._execute_with_retry(_save)
        return result is not None
    
    def _log_error(self, operation: str, error: Exception):
        """Log error to Firebase"""
        try:
            collection = self.firestore_db.collection(FIREBASE_COLLECTIONS['error_logs'])
            error_data = {
                'timestamp': now_iso_wib(),
                'operation': operation,
                'error_type': type(error).__name__,
                'error_message': str(error),
            }
            collection.add(error_data)
        except:
            pass
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old data from Firebase"""
        def _cleanup():
            cutoff_date_wib = get_local_now() - timedelta(days=days)
            cutoff_iso = prepare_firebase_timestamp(cutoff_date_wib)
            total_deleted = 0
            
            for collection_name in [FIREBASE_COLLECTIONS['raw_data'], 
                                   FIREBASE_COLLECTIONS['predictions']]:
                collection = self.firestore_db.collection(collection_name)
                old_docs = collection.where('timestamp', '<', cutoff_iso).limit(100).stream()
                
                batch = self.firestore_db.batch()
                count = 0
                
                for doc in old_docs:
                    batch.delete(doc.reference)
                    count += 1
                
                if count > 0:
                    batch.commit()
                    total_deleted += count
                    logger.info(f"🗑️ Cleaned up {count} old documents from {collection_name}")
            
            return total_deleted
        
        result, error = self._execute_with_retry(_cleanup)
        
        if result is not None:
            logger.info(f"✅ Cleanup completed: {result} documents deleted")
        else:
            logger.warning(f"⚠️ Cleanup failed: {error}")


def get_firebase_manager() -> FirebaseManager:
    """Get or create Firebase manager instance"""
    return FirebaseManager()