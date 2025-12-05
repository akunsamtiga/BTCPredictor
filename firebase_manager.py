"""
Firebase Manager for Bitcoin Predictor
Handles all Firebase operations including saving predictions, validation, and statistics
"""

import firebase_admin
from firebase_admin import credentials, firestore, db
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Optional
import pandas as pd
from config import FIREBASE_CONFIG, FIREBASE_COLLECTIONS

logger = logging.getLogger(__name__)

"tes"
class FirebaseManager:
    """Manages all Firebase operations"""
    
    def __init__(self):
        """Initialize Firebase connection"""
        self.db = None
        self.firestore_db = None
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        try:
            # Check if already initialized
            if not firebase_admin._apps:
                cred = credentials.Certificate(FIREBASE_CONFIG['credentials_path'])
                firebase_admin.initialize_app(cred, {
                    'databaseURL': FIREBASE_CONFIG['database_url']
                })
            
            self.db = db
            self.firestore_db = firestore.client()
            logger.info("✅ Firebase initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Firebase: {e}")
            raise
    
    def save_prediction(self, prediction: Dict) -> Optional[str]:
        """
        Save prediction to Firebase
        
        Args:
            prediction: Dictionary containing prediction data
            
        Returns:
            Document ID if successful, None otherwise
        """
        try:
            collection = self.firestore_db.collection(FIREBASE_COLLECTIONS['predictions'])
            
            # Prepare data
            doc_data = {
                'timestamp': firestore.SERVER_TIMESTAMP,
                'prediction_time': datetime.now().isoformat(),
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
                'target_time': (datetime.now() + timedelta(minutes=prediction['timeframe_minutes'])).isoformat(),
                'validated': False,
                'validation_result': None,
                'actual_price': None,
            }
            
            # Add ML specific data if available
            if 'model_agreement' in prediction:
                doc_data.update({
                    'model_agreement': prediction['model_agreement'],
                    'lstm_prediction': float(prediction['lstm_prediction']),
                    'gb_prediction': float(prediction['gb_prediction']),
                    'rf_direction': prediction['rf_direction'],
                    'rf_confidence': float(prediction['rf_confidence']),
                })
            
            # Save to Firestore
            doc_ref = collection.add(doc_data)
            doc_id = doc_ref[1].id
            
            logger.info(f"✅ Prediction saved: {doc_id} (Timeframe: {prediction['timeframe_minutes']}min)")
            return doc_id
            
        except Exception as e:
            logger.error(f"❌ Error saving prediction: {e}")
            return None
    
    def save_raw_data(self, df, limit: int = 100):
        """
        Save raw Bitcoin data to Firebase (recent data only)
        
        Args:
            df: DataFrame containing Bitcoin data
            limit: Maximum number of recent records to save
        """
        try:
            collection = self.firestore_db.collection(FIREBASE_COLLECTIONS['raw_data'])
            
            # Get recent data
            recent_data = df.head(limit)
            
            # Batch write
            batch = self.firestore_db.batch()
            count = 0
            
            for idx, row in recent_data.iterrows():
                doc_ref = collection.document(f"btc_{row['datetime'].strftime('%Y%m%d_%H%M%S')}")
                
                data = {
                    'timestamp': firestore.SERVER_TIMESTAMP,
                    'datetime': row['datetime'].isoformat(),
                    'price': float(row['price']),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'volume': float(row['volume']),
                }
                
                # Add technical indicators if available
                if 'rsi' in row and not pd.isna(row['rsi']):
                    data['rsi'] = float(row['rsi'])
                if 'macd' in row and not pd.isna(row['macd']):
                    data['macd'] = float(row['macd'])
                
                batch.set(doc_ref, data, merge=True)
                count += 1
                
                # Commit batch every 500 operations
                if count % 500 == 0:
                    batch.commit()
                    batch = self.firestore_db.batch()
            
            # Commit remaining
            if count % 500 != 0:
                batch.commit()
            
            logger.info(f"✅ Saved {count} raw data points to Firebase")
            
        except Exception as e:
            logger.error(f"❌ Error saving raw data: {e}")
    
    def get_unvalidated_predictions(self) -> List[Dict]:
        """
        Get all predictions that haven't been validated yet
        
        Returns:
            List of prediction documents
        """
        try:
            collection = self.firestore_db.collection(FIREBASE_COLLECTIONS['predictions'])
            
            # Query unvalidated predictions where target time has passed
            now = datetime.now()
            
            query = collection.where('validated', '==', False).limit(100)
            docs = query.stream()
            
            predictions = []
            for doc in docs:
                data = doc.to_dict()
                data['doc_id'] = doc.id
                
                # Check if target time has passed
                target_time = datetime.fromisoformat(data['target_time'])
                if target_time <= now:
                    predictions.append(data)
            
            logger.info(f"📋 Found {len(predictions)} predictions ready for validation")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Error getting unvalidated predictions: {e}")
            return []
    
    def validate_prediction(self, doc_id: str, actual_price: float, 
                          predicted_price: float, trend: str) -> bool:
        """
        Validate a prediction and mark as win/lose
        
        Args:
            doc_id: Document ID of the prediction
            actual_price: Actual price at target time
            predicted_price: Predicted price
            trend: Predicted trend (CALL/PUT)
            
        Returns:
            True if validation successful
        """
        try:
            # Determine if prediction was correct
            predicted_direction = 'up' if 'CALL' in trend.upper() else 'down'
            actual_direction = 'up' if actual_price >= predicted_price else 'down'
            
            is_win = predicted_direction == actual_direction
            
            # Calculate error metrics
            price_error = abs(actual_price - predicted_price)
            price_error_pct = (price_error / predicted_price) * 100
            
            # Update prediction document
            doc_ref = self.firestore_db.collection(FIREBASE_COLLECTIONS['predictions']).document(doc_id)
            doc_ref.update({
                'validated': True,
                'validation_time': firestore.SERVER_TIMESTAMP,
                'actual_price': float(actual_price),
                'validation_result': 'WIN' if is_win else 'LOSE',
                'price_error': float(price_error),
                'price_error_pct': float(price_error_pct),
                'direction_correct': is_win
            })
            
            # Save to validation collection for analytics
            validation_data = {
                'timestamp': firestore.SERVER_TIMESTAMP,
                'prediction_id': doc_id,
                'result': 'WIN' if is_win else 'LOSE',
                'predicted_price': float(predicted_price),
                'actual_price': float(actual_price),
                'error': float(price_error),
                'error_pct': float(price_error_pct)
            }
            
            self.firestore_db.collection(FIREBASE_COLLECTIONS['validation']).add(validation_data)
            
            result_emoji = "✅" if is_win else "❌"
            logger.info(f"{result_emoji} Validation: {doc_id} - {validation_data['result']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating prediction: {e}")
            return False
    
    def get_statistics(self, timeframe_minutes: Optional[int] = None, 
                      days: int = 7) -> Dict:
        """
        Get prediction statistics
        
        Args:
            timeframe_minutes: Filter by specific timeframe
            days: Number of days to analyze
            
        Returns:
            Dictionary with statistics
        """
        try:
            collection = self.firestore_db.collection(FIREBASE_COLLECTIONS['predictions'])
            
            # Query validated predictions from last N days
            cutoff_date = datetime.now() - timedelta(days=days)
            
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
                
                # Check if within date range
                if 'prediction_time' in data:
                    pred_time = datetime.fromisoformat(data['prediction_time'])
                    if pred_time < cutoff_date:
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
            
            # Calculate statistics
            win_rate = (wins / total * 100) if total > 0 else 0
            avg_error = total_error / total if total > 0 else 0
            avg_error_pct = total_error_pct / total if total > 0 else 0
            
            stats = {
                'timeframe_minutes': timeframe_minutes,
                'period_days': days,
                'total_predictions': total,
                'wins': wins,
                'losses': losses,
                'win_rate': round(win_rate, 2),
                'avg_error': round(avg_error, 2),
                'avg_error_pct': round(avg_error_pct, 2),
                'last_updated': datetime.now().isoformat()
            }
            
            logger.info(f"📊 Statistics: Win rate {win_rate:.1f}% ({wins}/{total})")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting statistics: {e}")
            return {}
    
    def save_statistics(self, stats: Dict) -> bool:
        """Save statistics to Firebase"""
        try:
            collection = self.firestore_db.collection(FIREBASE_COLLECTIONS['statistics'])
            
            doc_id = f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            collection.document(doc_id).set(stats)
            
            logger.info(f"✅ Statistics saved: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving statistics: {e}")
            return False
    
    def save_model_performance(self, metrics: Dict) -> bool:
        """Save model performance metrics"""
        try:
            collection = self.firestore_db.collection(FIREBASE_COLLECTIONS['model_performance'])
            
            doc_data = {
                'timestamp': firestore.SERVER_TIMESTAMP,
                'datetime': datetime.now().isoformat(),
                'metrics': metrics
            }
            
            collection.add(doc_data)
            logger.info("✅ Model performance saved")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving model performance: {e}")
            return False
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old data from Firebase"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for collection_name in [FIREBASE_COLLECTIONS['raw_data'], 
                                   FIREBASE_COLLECTIONS['predictions']]:
                collection = self.firestore_db.collection(collection_name)
                
                # This is a simplified cleanup - in production, use batch deletes
                old_docs = collection.where('timestamp', '<', cutoff_date).limit(100).stream()
                
                batch = self.firestore_db.batch()
                count = 0
                
                for doc in old_docs:
                    batch.delete(doc.reference)
                    count += 1
                
                if count > 0:
                    batch.commit()
                    logger.info(f"🗑️  Cleaned up {count} old documents from {collection_name}")
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up old data: {e}")


# Convenience function
def get_firebase_manager() -> FirebaseManager:
    """Get or create Firebase manager instance"""
    return FirebaseManager()