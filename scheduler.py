"""
Automation Scheduler for Bitcoin Predictor
Handles continuous prediction and validation
"""

import time
import schedule
import threading
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd

from config import PREDICTION_CONFIG, MODEL_CONFIG
from firebase_manager import FirebaseManager
from btc_predictor_automated import (
    BitcoinMLPredictor,
    get_bitcoin_data_realtime,
    get_current_btc_price,
    add_technical_indicators
)

logger = logging.getLogger(__name__)


class PredictionScheduler:
    """Manages automated predictions and validations"""
    
    def __init__(self):
        self.predictor = BitcoinMLPredictor()
        self.firebase = FirebaseManager()
        self.is_running = False
        self.timeframes = PREDICTION_CONFIG['timeframes']
        self.last_prediction_time = {}
        self.data_cache = None
        self.data_cache_time = None
        
        logger.info("🚀 Prediction Scheduler initialized")
    
    def initialize_models(self):
        """Initialize or load ML models"""
        logger.info("🔧 Initializing models...")
        
        # Try to load existing models
        if self.predictor.load_models():
            logger.info("✅ Loaded existing models")
            
            # Check if retraining needed
            if self.predictor.needs_retraining():
                logger.info("⚠️ Models need retraining...")
                self.train_models()
        else:
            logger.info("📚 No existing models found, training new ones...")
            self.train_models()
    
    def train_models(self):
        """Train or retrain models"""
        try:
            logger.info("\n" + "="*80)
            logger.info("🤖 TRAINING MODELS")
            logger.info("="*80)
            
            # Fetch comprehensive data for training
            df = get_bitcoin_data_realtime(days=30, interval='hour')
            
            if df is None or len(df) < 200:
                logger.error("❌ Insufficient data for training")
                return False
            
            # Add indicators
            df = add_technical_indicators(df)
            
            # Train
            success = self.predictor.train_models(df, epochs=30, batch_size=32)
            
            if success:
                # Save model performance to Firebase
                self.firebase.save_model_performance(self.predictor.metrics)
                logger.info("✅ Training completed successfully\n")
                return True
            else:
                logger.error("❌ Training failed\n")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error training models: {e}")
            return False
    
    def get_fresh_data(self, force_refresh=False):
        """Get fresh Bitcoin data with caching"""
        try:
            # Use cache if available and recent (< 2 minutes old)
            if not force_refresh and self.data_cache is not None and self.data_cache_time:
                age = (datetime.now() - self.data_cache_time).total_seconds()
                if age < 120:  # 2 minutes
                    logger.debug("📦 Using cached data")
                    return self.data_cache
            
            # Fetch new data
            logger.info("📡 Fetching fresh data...")
            
            # Use minute data for short timeframes, hour data for longer ones
            max_timeframe = max(self.timeframes)
            
            if max_timeframe <= 240:  # Up to 4 hours
                df = get_bitcoin_data_realtime(days=3, interval='minute')
            elif max_timeframe <= 1440:  # Up to 24 hours
                df = get_bitcoin_data_realtime(days=7, interval='hour')
            else:
                df = get_bitcoin_data_realtime(days=14, interval='hour')
            
            if df is not None:
                # Add indicators
                df = add_technical_indicators(df)
                
                # Update cache
                self.data_cache = df
                self.data_cache_time = datetime.now()
                
                # Save to Firebase (limited to recent data)
                self.firebase.save_raw_data(df, limit=100)
                
                logger.info(f"✅ Data refreshed: {len(df)} points")
                return df
            else:
                logger.error("❌ Failed to fetch data")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting fresh data: {e}")
            return None
    
    def run_predictions(self):
        """Run predictions for all timeframes"""
        try:
            logger.info("\n" + "="*80)
            logger.info(f"🔮 RUNNING PREDICTIONS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("="*80)
            
            # Get fresh data
            df = self.get_fresh_data()
            
            if df is None:
                logger.error("❌ Cannot run predictions without data")
                return
            
            current_price = df.iloc[0]['price']
            logger.info(f"💰 Current BTC Price: ${current_price:,.2f}")
            
            # Run predictions for each timeframe
            for timeframe in self.timeframes:
                try:
                    # Check if we should predict for this timeframe
                    # (avoid predicting too frequently for the same timeframe)
                    if timeframe in self.last_prediction_time:
                        time_since_last = (datetime.now() - self.last_prediction_time[timeframe]).total_seconds()
                        min_interval = min(timeframe * 60 * 0.2, 300)  # 20% of timeframe or 5 min max
                        
                        if time_since_last < min_interval:
                            logger.debug(f"⏭️  Skipping {timeframe}min (too soon)")
                            continue
                    
                    logger.info(f"\n⏱️  Predicting for {timeframe} minutes...")
                    
                    # Make prediction
                    prediction = self.predictor.predict(df, timeframe)
                    
                    if prediction:
                        # Display prediction
                        self._display_prediction_summary(prediction)
                        
                        # Save to Firebase
                        doc_id = self.firebase.save_prediction(prediction)
                        
                        if doc_id:
                            self.last_prediction_time[timeframe] = datetime.now()
                            logger.info(f"✅ Prediction saved: {doc_id}")
                        else:
                            logger.warning("⚠️ Failed to save prediction")
                    else:
                        logger.warning(f"⚠️ Prediction failed for {timeframe}min")
                        
                except Exception as e:
                    logger.error(f"❌ Error predicting {timeframe}min: {e}")
                    continue
            
            logger.info("\n" + "="*80)
            logger.info("✅ Prediction cycle completed")
            logger.info("="*80 + "\n")
            
        except Exception as e:
            logger.error(f"❌ Error running predictions: {e}")
            import traceback
            traceback.print_exc()
    
    def validate_predictions(self):
        """Validate predictions that have reached their target time"""
        try:
            logger.info("\n🔍 VALIDATING PREDICTIONS...")
            
            # Get unvalidated predictions
            predictions = self.firebase.get_unvalidated_predictions()
            
            if not predictions:
                logger.info("✅ No predictions to validate")
                return
            
            # Get current price
            current_price = get_current_btc_price()
            
            if not current_price:
                logger.warning("⚠️ Cannot validate without current price")
                return
            
            logger.info(f"💰 Current price: ${current_price:,.2f}")
            
            # Validate each prediction
            validated_count = 0
            for pred in predictions:
                try:
                    doc_id = pred['doc_id']
                    predicted_price = pred['predicted_price']
                    trend = pred['trend']
                    timeframe = pred['timeframe_minutes']
                    
                    # Validate
                    success = self.firebase.validate_prediction(
                        doc_id, 
                        current_price, 
                        predicted_price, 
                        trend
                    )
                    
                    if success:
                        validated_count += 1
                        
                except Exception as e:
                    logger.error(f"❌ Error validating prediction {pred.get('doc_id')}: {e}")
                    continue
            
            logger.info(f"✅ Validated {validated_count}/{len(predictions)} predictions\n")
            
            # Update statistics
            self.update_statistics()
            
        except Exception as e:
            logger.error(f"❌ Error validating predictions: {e}")
    
    def update_statistics(self):
        """Calculate and save statistics"""
        try:
            logger.info("📊 Updating statistics...")
            
            # Overall statistics
            overall_stats = self.firebase.get_statistics(days=7)
            
            if overall_stats and overall_stats.get('total_predictions', 0) > 0:
                logger.info(f"   Overall: {overall_stats['win_rate']:.1f}% win rate ({overall_stats['wins']}/{overall_stats['total_predictions']})")
                self.firebase.save_statistics(overall_stats)
            
            # Per-timeframe statistics
            for timeframe in self.timeframes:
                stats = self.firebase.get_statistics(timeframe_minutes=timeframe, days=7)
                
                if stats and stats.get('total_predictions', 0) > 0:
                    logger.info(f"   {timeframe}min: {stats['win_rate']:.1f}% win rate ({stats['wins']}/{stats['total_predictions']})")
                    self.firebase.save_statistics(stats)
            
            logger.info("✅ Statistics updated\n")
            
        except Exception as e:
            logger.error(f"❌ Error updating statistics: {e}")
    
    def _display_prediction_summary(self, prediction):
        """Display concise prediction summary"""
        arrow = "🟢 ↗️" if prediction['price_change'] > 0 else "🔴 ↘️"
        
        logger.info(f"   {arrow} ${prediction['predicted_price']:,.2f} "
                   f"({prediction['price_change_pct']:+.2f}%) - "
                   f"Confidence: {prediction['confidence']:.1f}%")
    
    def cleanup_old_data(self):
        """Periodic cleanup of old data"""
        try:
            logger.info("🗑️  Running cleanup...")
            self.firebase.cleanup_old_data(days=30)
            logger.info("✅ Cleanup completed\n")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
    
    def start(self):
        """Start the automated scheduler"""
        logger.info("\n" + "="*80)
        logger.info("🚀 STARTING BITCOIN PREDICTOR AUTOMATION")
        logger.info("="*80)
        
        # Initialize models
        self.initialize_models()
        
        if not self.predictor.is_trained:
            logger.error("❌ Cannot start without trained models")
            return
        
        # Schedule tasks
        logger.info("\n📅 Setting up schedule:")
        logger.info(f"   • Predictions: Every {PREDICTION_CONFIG['prediction_interval']} seconds")
        logger.info(f"   • Validation: Every {PREDICTION_CONFIG['validation_check_interval']} seconds")
        logger.info(f"   • Model retraining: Every {MODEL_CONFIG['auto_retrain_interval'] // 3600} hours")
        logger.info(f"   • Data cleanup: Daily at 03:00")
        
        # Prediction schedule
        schedule.every(PREDICTION_CONFIG['prediction_interval']).seconds.do(self.run_predictions)
        
        # Validation schedule
        schedule.every(PREDICTION_CONFIG['validation_check_interval']).seconds.do(self.validate_predictions)
        
        # Retraining schedule (once per day)
        schedule.every().day.at("02:00").do(self.train_models)
        
        # Cleanup schedule (once per day)
        schedule.every().day.at("03:00").do(self.cleanup_old_data)
        
        # Run first prediction immediately
        logger.info("\n🎯 Running initial predictions...")
        self.run_predictions()
        
        self.is_running = True
        logger.info("\n✅ Automation started successfully!")
        logger.info("="*80 + "\n")
        
        # Main loop
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("\n⚠️ Shutdown requested...")
            self.stop()
    
    def stop(self):
        """Stop the scheduler"""
        logger.info("🛑 Stopping automation...")
        self.is_running = False
        logger.info("✅ Automation stopped\n")


# Main function for running as standalone script
def main():
    """Main entry point"""
    try:
        scheduler = PredictionScheduler()
        scheduler.start()
        
    except KeyboardInterrupt:
        logger.info("\n\n❌ Program interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()