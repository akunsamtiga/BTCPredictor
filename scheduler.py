"""
Enhanced Automation Scheduler with Heartbeat System
"""

import time
import schedule
import threading
import logging
import gc
import signal
import sys
from datetime import datetime, timedelta
from typing import List, Dict
import traceback

from config import PREDICTION_CONFIG, MODEL_CONFIG, VPS_CONFIG, HEALTH_CONFIG
from firebase_manager import FirebaseManager
from system_health import SystemHealthMonitor, monitor_health
from heartbeat import HeartbeatManager  # Import heartbeat manager
from btc_predictor_automated import (
    BitcoinMLPredictor,
    get_bitcoin_data_realtime,
    get_current_btc_price,
    add_technical_indicators
)

logger = logging.getLogger(__name__)


class EnhancedPredictionScheduler:
    """Enhanced scheduler with error handling, auto-recovery, and heartbeat"""
    
    def __init__(self):
        self.predictor = BitcoinMLPredictor()
        self.firebase = None
        self.health_monitor = SystemHealthMonitor()
        self.heartbeat = None  # Will be initialized after Firebase
        self.is_running = False
        self.timeframes = PREDICTION_CONFIG['timeframes']
        self.last_prediction_time = {}
        self.data_cache = None
        self.data_cache_time = None
        
        # Error tracking
        self.consecutive_failures = 0
        self.total_predictions = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        
        # Shutdown handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 Enhanced Prediction Scheduler initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"\n⚠️ Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def _initialize_firebase(self):
        """Initialize Firebase with retry"""
        max_retries = 5
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🔗 Attempting Firebase connection (attempt {attempt + 1}/{max_retries})...")
                self.firebase = FirebaseManager()
                
                if self.firebase.connected:
                    logger.info("✅ Firebase connected successfully")
                    
                    # Initialize heartbeat manager
                    self.heartbeat = HeartbeatManager(self.firebase, update_interval=30)
                    self.heartbeat.send_status_change('starting', 'System initializing')
                    
                    return True
                
            except Exception as e:
                logger.warning(f"⚠️ Firebase connection failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (2 ** attempt))
        
        logger.error("❌ Could not connect to Firebase after multiple attempts")
        return False
    
    def initialize_models(self):
        """Initialize or load ML models with error handling"""
        try:
            logger.info("🔧 Initializing models...")
            
            if self.heartbeat:
                self.heartbeat.send_status_change('loading_models', 'Loading ML models')
            
            # Try to load existing models
            if self.predictor.load_models():
                logger.info("✅ Loaded existing models")
                
                # Check if retraining needed
                if self.predictor.needs_retraining():
                    logger.info("⚠️ Models need retraining...")
                    return self.train_models()
                return True
            else:
                logger.info("📚 No existing models found, training new ones...")
                return self.train_models()
                
        except Exception as e:
            logger.error(f"❌ Error initializing models: {e}")
            traceback.print_exc()
            return False
    
    def train_models(self):
        """Train or retrain models with error handling"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                logger.info(f"\n{'='*80}")
                logger.info(f"🤖 TRAINING MODELS (Attempt {attempt + 1}/{max_retries})")
                logger.info(f"{'='*80}")
                
                if self.heartbeat:
                    self.heartbeat.send_status_change('training', 'Training ML models')
                
                # Fetch training data
                df = self._fetch_training_data()
                
                if df is None or len(df) < 200:
                    logger.error("❌ Insufficient data for training")
                    if attempt < max_retries - 1:
                        time.sleep(30)
                        continue
                    return False
                
                # Add indicators
                df = add_technical_indicators(df)
                
                # Train models
                success = self.predictor.train_models(df, epochs=50, batch_size=32)
                
                if success:
                    # Save performance to Firebase
                    if self.firebase and self.firebase.connected:
                        self.firebase.save_model_performance(self.predictor.metrics)
                    
                    logger.info("✅ Training completed successfully\n")
                    
                    if self.heartbeat:
                        self.heartbeat.send_status_change('trained', 'Models trained successfully')
                    
                    # Clear memory after training
                    if VPS_CONFIG['enable_memory_optimization']:
                        gc.collect()
                    
                    return True
                else:
                    logger.warning(f"⚠️ Training attempt {attempt + 1} failed")
                    if attempt < max_retries - 1:
                        time.sleep(60)
                
            except Exception as e:
                logger.error(f"❌ Training error: {e}")
                traceback.print_exc()
                
                if attempt < max_retries - 1:
                    time.sleep(60)
        
        logger.error("❌ Training failed after all attempts")
        return False
    
    def _fetch_training_data(self):
        """Fetch data for training with fallback options"""
        intervals = ['hour', 'day']
        days_options = [30, 60, 90]
        
        for days in days_options:
            for interval in intervals:
                try:
                    logger.info(f"📡 Trying to fetch {days} days of {interval} data...")
                    df = get_bitcoin_data_realtime(days=days, interval=interval)
                    
                    if df is not None and len(df) >= 200:
                        logger.info(f"✅ Successfully fetched {len(df)} data points")
                        return df
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to fetch {interval} data: {e}")
                    continue
        
        return None
    
    def get_fresh_data(self, force_refresh=False):
        """Get fresh Bitcoin data with caching and error handling"""
        try:
            # Use cache if available and recent
            if not force_refresh and self.data_cache is not None and self.data_cache_time:
                age = (datetime.now() - self.data_cache_time).total_seconds()
                if age < 120:
                    logger.debug("📦 Using cached data")
                    return self.data_cache
            
            # Fetch new data with retry
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    logger.info("📡 Fetching fresh data...")
                    
                    df = get_bitcoin_data_realtime(days=14, interval='hour')
                    
                    if df is None or len(df) < 200:
                        logger.warning(f"⚠️ Insufficient hour data, trying with more days...")
                        df = get_bitcoin_data_realtime(days=30, interval='hour')
                    
                    if df is None or len(df) < 200:
                        logger.warning("⚠️ Hour data failed, trying day data...")
                        df = get_bitcoin_data_realtime(days=90, interval='day')
                    
                    if df is not None and len(df) >= 200:
                        df = add_technical_indicators(df)
                        
                        self.data_cache = df
                        self.data_cache_time = datetime.now()
                        
                        if self.firebase and self.firebase.connected:
                            try:
                                self.firebase.save_raw_data(df, limit=100)
                            except:
                                pass
                        
                        logger.info(f"✅ Data refreshed: {len(df)} points")
                        return df
                    
                except Exception as e:
                    logger.warning(f"⚠️ Data fetch attempt {attempt + 1} failed: {e}")
                    
                    if attempt < max_retries - 1:
                        time.sleep(5 * (2 ** attempt))
            
            logger.error("❌ Failed to fetch data after all attempts")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting fresh data: {e}")
            traceback.print_exc()
            return None
    
    def run_predictions(self):
        """Run predictions with comprehensive error handling"""
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"🔮 RUNNING PREDICTIONS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*80}")
            
            # Send heartbeat
            if self.heartbeat:
                self.heartbeat.send_heartbeat({
                    'last_activity': 'running_predictions',
                    'total_predictions': self.total_predictions,
                    'successful_predictions': self.successful_predictions,
                    'failed_predictions': self.failed_predictions
                })
            
            # Check system health
            if HEALTH_CONFIG['enable_watchdog']:
                health = self.health_monitor.get_full_health_report()
                if health['overall_status'] == 'CRITICAL':
                    logger.error("❌ System health critical, skipping predictions")
                    return
            
            # Get fresh data
            df = self.get_fresh_data()
            
            if df is None:
                logger.error("❌ Cannot run predictions without data")
                self.consecutive_failures += 1
                self._handle_consecutive_failures()
                return
            
            current_price = df.iloc[0]['price']
            logger.info(f"💰 Current BTC Price: ${current_price:,.2f}")
            
            # Run predictions for each timeframe
            predictions_made = 0
            
            for timeframe in self.timeframes:
                try:
                    if self._should_skip_timeframe(timeframe):
                        continue
                    
                    logger.info(f"\n⏱️ Predicting for {timeframe} minutes...")
                    
                    prediction = self.predictor.predict(df, timeframe)
                    
                    if prediction:
                        self._display_prediction_summary(prediction)
                        
                        if self.firebase and self.firebase.connected:
                            doc_id = self.firebase.save_prediction(prediction)
                            
                            if doc_id:
                                self.last_prediction_time[timeframe] = datetime.now()
                                predictions_made += 1
                                self.successful_predictions += 1
                                logger.info(f"✅ Prediction saved: {doc_id}")
                            else:
                                logger.warning("⚠️ Failed to save prediction")
                                self.failed_predictions += 1
                        else:
                            logger.warning("⚠️ Firebase not connected, prediction not saved")
                            self.failed_predictions += 1
                    else:
                        logger.warning(f"⚠️ Prediction failed for {timeframe}min")
                        self.failed_predictions += 1
                        
                except Exception as e:
                    logger.error(f"❌ Error predicting {timeframe}min: {e}")
                    self.failed_predictions += 1
                    continue
            
            self.total_predictions += predictions_made
            
            # Reset consecutive failures on success
            if predictions_made > 0:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
                self._handle_consecutive_failures()
            
            logger.info(f"\n{'='*80}")
            logger.info(f"✅ Prediction cycle completed - {predictions_made} predictions made")
            logger.info(f"📊 Success rate: {self.successful_predictions}/{self.total_predictions}")
            logger.info(f"{'='*80}\n")
            
            # Memory optimization
            if VPS_CONFIG['enable_memory_optimization']:
                if self.total_predictions % 10 == 0:
                    gc.collect()
            
        except Exception as e:
            logger.error(f"❌ Critical error in prediction cycle: {e}")
            traceback.print_exc()
            self.consecutive_failures += 1
            self._handle_consecutive_failures()
    
    def _should_skip_timeframe(self, timeframe):
        """Check if should skip prediction for timeframe"""
        if timeframe in self.last_prediction_time:
            time_since_last = (datetime.now() - self.last_prediction_time[timeframe]).total_seconds()
            min_interval = min(timeframe * 60 * 0.2, 300)
            
            if time_since_last < min_interval:
                logger.debug(f"⏭️ Skipping {timeframe}min (too soon)")
                return True
        
        return False
    
    def _handle_consecutive_failures(self):
        """Handle consecutive failures"""
        max_failures = PREDICTION_CONFIG['max_consecutive_failures']
        
        if self.consecutive_failures >= max_failures:
            logger.error(f"❌ Too many consecutive failures ({self.consecutive_failures})")
            
            if self.heartbeat:
                self.heartbeat.send_status_change('error', f'Too many failures: {self.consecutive_failures}')
            
            if HEALTH_CONFIG['auto_restart_on_error']:
                logger.warning("🔄 Attempting auto-recovery...")
                
                self._initialize_firebase()
                
                if not self.predictor.is_trained:
                    self.initialize_models()
                
                self.data_cache = None
                self.data_cache_time = None
                
                self.consecutive_failures = 0
                
                logger.info("✅ Auto-recovery attempted")
            else:
                logger.error("❌ Auto-recovery disabled, manual intervention required")
    
    def validate_predictions(self):
        """Validate predictions with error handling"""
        try:
            logger.info("\n🔍 VALIDATING PREDICTIONS...")
            
            if not self.firebase or not self.firebase.connected:
                logger.warning("⚠️ Firebase not connected, skipping validation")
                return
            
            predictions = self.firebase.get_unvalidated_predictions()
            
            if not predictions:
                logger.info("✅ No predictions to validate")
                return
            
            current_price = get_current_btc_price()
            
            if not current_price:
                logger.warning("⚠️ Cannot validate without current price")
                return
            
            logger.info(f"💰 Current price: ${current_price:,.2f}")
            
            validated_count = 0
            for pred in predictions:
                try:
                    doc_id = pred['doc_id']
                    predicted_price = pred['predicted_price']
                    trend = pred['trend']
                    
                    success = self.firebase.validate_prediction(
                        doc_id, current_price, predicted_price, trend
                    )
                    
                    if success:
                        validated_count += 1
                        
                except Exception as e:
                    logger.error(f"❌ Error validating {pred.get('doc_id')}: {e}")
                    continue
            
            logger.info(f"✅ Validated {validated_count}/{len(predictions)} predictions\n")
            
            if validated_count > 0:
                self.update_statistics()
            
        except Exception as e:
            logger.error(f"❌ Error in validation cycle: {e}")
            traceback.print_exc()
    
    def update_statistics(self):
        """Update statistics with error handling"""
        try:
            if not self.firebase or not self.firebase.connected:
                return
            
            logger.info("📊 Updating statistics...")
            
            overall_stats = self.firebase.get_statistics(days=7)
            
            if overall_stats and overall_stats.get('total_predictions', 0) > 0:
                logger.info(f"   Overall: {overall_stats['win_rate']:.1f}% win rate")
                self.firebase.save_statistics(overall_stats)
            
            for timeframe in self.timeframes:
                stats = self.firebase.get_statistics(timeframe_minutes=timeframe, days=7)
                
                if stats and stats.get('total_predictions', 0) > 0:
                    logger.info(f"   {timeframe}min: {stats['win_rate']:.1f}% win rate")
                    self.firebase.save_statistics(stats)
            
            logger.info("✅ Statistics updated\n")
            
        except Exception as e:
            logger.error(f"❌ Error updating statistics: {e}")
    
    def _display_prediction_summary(self, prediction):
        """Display prediction summary"""
        arrow = "🟢 ↗️" if prediction['price_change'] > 0 else "🔴 ↘️"
        logger.info(f"   {arrow} ${prediction['predicted_price']:,.2f} "
                   f"({prediction['price_change_pct']:+.2f}%) - "
                   f"Confidence: {prediction['confidence']:.1f}%")
    
    def periodic_health_check(self):
        """Periodic system health check"""
        try:
            logger.info("\n🏥 SYSTEM HEALTH CHECK")
            
            report = monitor_health(self.firebase if self.firebase and self.firebase.connected else None)
            
            # Send heartbeat with health data
            if self.heartbeat:
                self.heartbeat.send_heartbeat({
                    'health_status': report['overall_status'],
                    'memory_mb': report['memory']['process_memory_mb'],
                    'cpu_percent': report['cpu']['cpu_percent']
                })
            
            if self.health_monitor.should_restart():
                logger.warning("⚠️ System requires restart")
                self.stop()
                sys.exit(1)
            
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
    
    def periodic_cleanup(self):
        """Periodic cleanup"""
        try:
            logger.info("🗑️ Running periodic cleanup...")
            
            if VPS_CONFIG['enable_memory_optimization']:
                gc.collect()
                logger.info("✅ Garbage collection completed")
            
            if self.firebase and self.firebase.connected:
                self.firebase.cleanup_old_data(days=30)
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    def start(self):
        """Start the automated scheduler"""
        logger.info(f"\n{'='*80}")
        logger.info("🚀 STARTING BITCOIN PREDICTOR AUTOMATION")
        logger.info(f"{'='*80}")
        
        if not self._initialize_firebase():
            logger.error("❌ Cannot start without Firebase connection")
            return
        
        if not self.initialize_models():
            logger.error("❌ Cannot start without trained models")
            return
        
        logger.info("\n📅 Setting up schedule:")
        logger.info(f"   • Predictions: Every {PREDICTION_CONFIG['prediction_interval']} seconds")
        logger.info(f"   • Validation: Every {PREDICTION_CONFIG['validation_check_interval']} seconds")
        logger.info(f"   • Health check: Every {HEALTH_CONFIG['health_check_interval']} seconds")
        logger.info(f"   • Heartbeat: Every 30 seconds")
        logger.info(f"   • Model retraining: Daily at 02:00")
        logger.info(f"   • Cleanup: Daily at 03:00")
        
        # Setup schedules
        schedule.every(PREDICTION_CONFIG['prediction_interval']).seconds.do(self.run_predictions)
        schedule.every(PREDICTION_CONFIG['validation_check_interval']).seconds.do(self.validate_predictions)
        schedule.every(HEALTH_CONFIG['health_check_interval']).seconds.do(self.periodic_health_check)
        
        # Heartbeat every 30 seconds
        if self.heartbeat:
            schedule.every(30).seconds.do(lambda: self.heartbeat.send_heartbeat({
                'last_activity': 'heartbeat',
                'predictions_count': self.total_predictions
            }))
        
        schedule.every().day.at("02:00").do(self.train_models)
        schedule.every().day.at("03:00").do(self.periodic_cleanup)
        
        if VPS_CONFIG['garbage_collection_interval']:
            schedule.every(VPS_CONFIG['garbage_collection_interval']).seconds.do(gc.collect)
        
        logger.info("\n🎯 Running initial predictions...")
        self.run_predictions()
        
        self.periodic_health_check()
        
        # Update status to running
        if self.heartbeat:
            self.heartbeat.send_status_change('running', 'System fully operational')
        
        self.is_running = True
        logger.info("\n✅ Automation started successfully!")
        logger.info(f"{'='*80}\n")
        
        # Main loop
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("\n⚠️ Shutdown requested...")
            self.stop()
        except Exception as e:
            logger.error(f"\n❌ Fatal error in main loop: {e}")
            traceback.print_exc()
            self.stop()
    
    def stop(self):
        """Stop the scheduler gracefully"""
        logger.info("🛑 Stopping automation...")
        self.is_running = False
        
        # Send shutdown signal
        if self.heartbeat:
            self.heartbeat.send_shutdown_signal()
        
        # Save final statistics
        if self.firebase and self.firebase.connected:
            try:
                logger.info("💾 Saving final statistics...")
                self.update_statistics()
            except:
                pass
        
        logger.info("✅ Automation stopped\n")


def main():
    """Main entry point"""
    try:
        scheduler = EnhancedPredictionScheduler()
        scheduler.start()
        
    except KeyboardInterrupt:
        logger.info("\n\n❌ Program interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()