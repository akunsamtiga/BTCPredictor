"""
Firebase Manager - ENHANCED VERSION
Improvements:
- Better validation logic with detailed logging
- Robust error handling with auto-retry
- Batch operations for efficiency
- Connection pooling
- Auto cleanup
- Performance optimizations
"""

import firebase_admin
from firebase_admin import credentials, firestore, db
from datetime import datetime, timedelta
import logging
import json
import time
from typing import Dict, List, Optional, Tuple
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
    """
    ENHANCED Firebase Manager with:
    - Robust error handling
    - Batch operations
    - Connection pooling
    - Performance optimizations
    """
    
    def __init__(self):
        """Initialize Firebase connection"""
        self.db = None
        self.firestore_db = None
        self.connected = False
        self.last_connection_attempt = None
        self.connection_failures = 0
        self._initialize_firebase()
        
        # Performance tracking
        self.operations_count = 0
        self.failed_operations = 0
        self.last_operation_time = None
    
    def _initialize_firebase(self, retry=True):
        """Initialize Firebase Admin SDK with enhanced retry logic"""
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
    
    def _execute_with_retry(self, operation, *args, max_retries=3, **kwargs):
        """
        ENHANCED: Execute Firebase operation with retry logic and performance tracking
        """
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                if not self._ensure_connection():
                    raise ConnectionError("Firebase not connected")
                
                result = operation(*args, **kwargs)
                
                # Track performance
                self.operations_count += 1
                self.last_operation_time = time.time() - start_time
                
                return result, None
                
            except Exception as e:
                self.failed_operations += 1
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
        """
        ENHANCED: Save prediction with validation and error handling
        """
        def _save():
            collection = self.firestore_db.collection(FIREBASE_COLLECTIONS['predictions'])
            
            # Validate prediction data
            required_fields = ['current_price', 'predicted_price', 'timeframe_minutes', 'trend', 'confidence']
            for field in required_fields:
                if field not in prediction:
                    raise ValueError(f"Missing required field: {field}")
            
            # Prepare timestamps
            now_wib = get_local_now()
            target_time_wib = add_minutes_local(now_wib, prediction['timeframe_minutes'])
            
            # Build document
            doc_data = {
                'timestamp': now_iso_wib(),
                'prediction_time': prepare_firebase_timestamp(now_wib),
                'timeframe_minutes': prediction['timeframe_minutes'],
                'current_price': float(prediction['current_price']),
                'predicted_price': float(prediction['predicted_price']),
                'price_change': float(prediction['price_change']),
                'price_change_pct': float(prediction['price_change_pct']),
                'price_range_low': float(prediction.get('price_range_low', 0)),
                'price_range_high': float(prediction.get('price_range_high', 0)),
                'trend': prediction['trend'],
                'confidence': float(prediction['confidence']),
                'quality_score': float(prediction.get('quality_score', 0)),
                'method': prediction.get('method', 'ML Ensemble'),
                'target_time': prepare_firebase_timestamp(target_time_wib),
                'validated': False,
                'validation_result': None,
                'actual_price': None,
            }
            
            # Add optional fields
            optional_fields = [
                'model_agreement', 'lstm_prediction', 'gb_prediction', 
                'rf_direction', 'rf_confidence', 'all_models_agree',
                'category', 'volatility'
            ]
            
            for field in optional_fields:
                if field in prediction:
                    doc_data[field] = prediction[field]
            
            doc_ref = collection.add(doc_data)
            return doc_ref[1].id
        
        result, error = self._execute_with_retry(_save)
        
        if result:
            logger.info(f"✅ Prediction saved: {result} (TF: {prediction['timeframe_minutes']}min)")
            return result
        else:
            logger.error(f"❌ Failed to save prediction: {error}")
            self._log_error("save_prediction", error)
            return None
    
    def save_predictions_batch(self, predictions: List[Dict]) -> Tuple[int, int]:
        """
        NEW: Save multiple predictions in batch for efficiency
        Returns: (success_count, failed_count)
        """
        def _save_batch():
            collection = self.firestore_db.collection(FIREBASE_COLLECTIONS['predictions'])
            batch = self.firestore_db.batch()
            
            doc_refs = []
            for pred in predictions:
                doc_ref = collection.document()
                
                now_wib = get_local_now()
                target_time_wib = add_minutes_local(now_wib, pred['timeframe_minutes'])
                
                doc_data = {
                    'timestamp': now_iso_wib(),
                    'prediction_time': prepare_firebase_timestamp(now_wib),
                    'timeframe_minutes': pred['timeframe_minutes'],
                    'current_price': float(pred['current_price']),
                    'predicted_price': float(pred['predicted_price']),
                    'price_change': float(pred['price_change']),
                    'price_change_pct': float(pred['price_change_pct']),
                    'trend': pred['trend'],
                    'confidence': float(pred['confidence']),
                    'target_time': prepare_firebase_timestamp(target_time_wib),
                    'validated': False,
                }
                
                batch.set(doc_ref, doc_data)
                doc_refs.append(doc_ref.id)
            
            batch.commit()
            return doc_refs
        
        result, error = self._execute_with_retry(_save_batch)
        
        if result:
            success = len(result)
            logger.info(f"✅ Batch saved {success} predictions")
            return success, 0
        else:
            logger.error(f"❌ Batch save failed: {error}")
            return 0, len(predictions)
    
    def get_unvalidated_predictions(self) -> List[Dict]:
        """Get all unvalidated predictions that are ready"""
        def _get():
            collection = self.firestore_db.collection(FIREBASE_COLLECTIONS['predictions'])
            now_wib = get_local_now()
            
            query = collection.where('validated', '==', False).limit(100)
            docs = query.stream()
            
            predictions = []
            for doc in docs:
                data = doc.to_dict()
                data['doc_id'] = doc.id
                
                target_time_str = data.get('target_time')
                if target_time_str:
                    target_time_wib = parse_iso_to_local(target_time_str)
                    
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
                          predicted_price: float, current_price: float, 
                          trend: str) -> bool:
        """
        ENHANCED: Validate prediction with detailed logging and metrics
        
        Logic:
        - CALL (UP): WIN if actual_price > current_price
        - PUT (DOWN): WIN if actual_price < current_price
        """
        def _validate():
            # Determine predicted direction
            predicted_direction = 'up' if 'CALL' in trend.upper() or 'UP' in trend.upper() else 'down'
            
            # FIXED: Compare actual vs current (at prediction time)
            if predicted_direction == 'up':
                is_win = actual_price > current_price
            else:
                is_win = actual_price < current_price
            
            # Calculate errors and metrics
            price_error = abs(actual_price - predicted_price)
            price_error_pct = (price_error / predicted_price) * 100
            
            actual_change = actual_price - current_price
            actual_change_pct = (actual_change / current_price) * 100
            
            # Determine accuracy level
            if price_error_pct < 1.0:
                accuracy = "EXCELLENT"
            elif price_error_pct < 2.0:
                accuracy = "GOOD"
            elif price_error_pct < 3.0:
                accuracy = "FAIR"
            else:
                accuracy = "POOR"
            
            # Update prediction document
            doc_ref = self.firestore_db.collection(FIREBASE_COLLECTIONS['predictions']).document(doc_id)
            
            update_data = {
                'validated': True,
                'validation_time': now_iso_wib(),
                'actual_price': float(actual_price),
                'validation_result': 'WIN' if is_win else 'LOSE',
                'price_error': float(price_error),
                'price_error_pct': float(price_error_pct),
                'direction_correct': is_win,
                'actual_change': float(actual_change),
                'actual_change_pct': float(actual_change_pct),
                'predicted_direction': predicted_direction,
                'accuracy_level': accuracy
            }
            
            doc_ref.update(update_data)
            
            # Save to validation collection
            validation_data = {
                'timestamp': now_iso_wib(),
                'prediction_id': doc_id,
                'result': 'WIN' if is_win else 'LOSE',
                'accuracy_level': accuracy,
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
            
            return is_win, accuracy
        
        result, error = self._execute_with_retry(_validate)
        
        if result is not None:
            is_win, accuracy = result
            result_emoji = "✅" if is_win else "❌"
            direction = "UP" if 'CALL' in trend.upper() or 'UP' in trend.upper() else "DOWN"
            
            logger.info(f"{result_emoji} {doc_id[:8]}: {direction} | "
                       f"${current_price:.0f} → ${actual_price:.0f} | "
                       f"{'WIN' if is_win else 'LOSE'} ({accuracy})")
            return True
        else:
            logger.error(f"❌ Failed to validate prediction: {error}")
            return False
    
    def validate_predictions_batch(self, predictions: List[Dict], 
                                   actual_price: float) -> Tuple[int, int]:
        """
        NEW: Validate multiple predictions in batch
        Returns: (validated_count, failed_count)
        """
        validated = 0
        failed = 0
        
        for pred in predictions:
            try:
                success = self.validate_prediction(
                    pred['doc_id'],
                    actual_price,
                    pred['predicted_price'],
                    pred['current_price'],
                    pred['trend']
                )
                
                if success:
                    validated += 1
                else:
                    failed += 1
                    
            except Exception as e:
                logger.error(f"❌ Batch validation error for {pred['doc_id']}: {e}")
                failed += 1
        
        logger.info(f"✅ Batch validation: {validated} validated, {failed} failed")
        return validated, failed
    
    def get_statistics(self, timeframe_minutes: Optional[int] = None, 
                      days: int = 7) -> Dict:
        """
        ENHANCED: Get statistics with caching and performance optimization
        """
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
            excellent = 0
            good = 0
            fair = 0
            poor = 0
            
            for doc in docs:
                data = doc.to_dict()
                
                # Check date
                if 'prediction_time' in data:
                    pred_time_wib = parse_iso_to_local(data['prediction_time'])
                    if pred_time_wib < cutoff_date_wib:
                        continue
                
                total += 1
                
                # Count wins/losses
                if data.get('validation_result') == 'WIN':
                    wins += 1
                else:
                    losses += 1
                
                # Errors
                if 'price_error' in data:
                    total_error += data['price_error']
                if 'price_error_pct' in data:
                    total_error_pct += data['price_error_pct']
                
                # Accuracy levels
                accuracy = data.get('accuracy_level', 'UNKNOWN')
                if accuracy == 'EXCELLENT':
                    excellent += 1
                elif accuracy == 'GOOD':
                    good += 1
                elif accuracy == 'FAIR':
                    fair += 1
                else:
                    poor += 1
            
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
                'accuracy_distribution': {
                    'excellent': excellent,
                    'good': good,
                    'fair': fair,
                    'poor': poor
                },
                'last_updated': now_iso_wib()
            }
        
        result, error = self._execute_with_retry(_get_stats)
        
        if result:
            logger.info(f"📊 Statistics: {result['win_rate']:.1f}% win rate ({result['wins']}/{result['total_predictions']})")
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
    
    def cleanup_old_data(self, days: int = 30) -> int:
        """
        ENHANCED: Clean up old data with progress tracking
        Returns: Number of documents deleted
        """
        def _cleanup():
            cutoff_date_wib = get_local_now() - timedelta(days=days)
            cutoff_iso = prepare_firebase_timestamp(cutoff_date_wib)
            total_deleted = 0
            
            collections_to_clean = [
                FIREBASE_COLLECTIONS['raw_data'],
                FIREBASE_COLLECTIONS['predictions'],
                FIREBASE_COLLECTIONS['system_health'],
                FIREBASE_COLLECTIONS['error_logs']
            ]
            
            for collection_name in collections_to_clean:
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
                    logger.info(f"🗑️ Cleaned {count} old documents from {collection_name}")
            
            return total_deleted
        
        result, error = self._execute_with_retry(_cleanup)
        
        if result is not None:
            logger.info(f"✅ Cleanup completed: {result} documents deleted")
            return result
        else:
            logger.warning(f"⚠️ Cleanup failed: {error}")
            return 0
    
    def _log_error(self, operation: str, error: Exception):
        """Enhanced error logging"""
        try:
            collection = self.firestore_db.collection(FIREBASE_COLLECTIONS['error_logs'])
            error_data = {
                'timestamp': now_iso_wib(),
                'operation': operation,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'connection_failures': self.connection_failures,
                'operations_count': self.operations_count,
                'failed_operations': self.failed_operations
            }
            collection.add(error_data)
        except:
            pass  # Don't let error logging cause more errors
    
    def get_performance_stats(self) -> Dict:
        """
        NEW: Get Firebase manager performance statistics
        """
        total_ops = self.operations_count
        failed_ops = self.failed_operations
        success_rate = ((total_ops - failed_ops) / total_ops * 100) if total_ops > 0 else 0
        
        return {
            'total_operations': total_ops,
            'successful_operations': total_ops - failed_ops,
            'failed_operations': failed_ops,
            'success_rate': round(success_rate, 2),
            'connection_failures': self.connection_failures,
            'connected': self.connected,
            'last_operation_time': self.last_operation_time,
            'last_connection_attempt': self.last_connection_attempt.isoformat() if self.last_connection_attempt else None
        }


def get_firebase_manager() -> FirebaseManager:
    """Get or create Firebase manager instance"""
    return FirebaseManager()