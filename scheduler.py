"""
Enhanced Automation Scheduler - FIXED VERSION
Includes proper memory management, error handling, and auto-restart
"""

import time
import schedule
import threading
import logging
import gc
import signal
import sys
import os
import psutil
from datetime import datetime, timedelta
from typing import List, Dict
import traceback

from config import (
    PREDICTION_CONFIG, MODEL_CONFIG, VPS_CONFIG, HEALTH_CONFIG,
    STRATEGY_CONFIG, get_timeframe_category, get_timeframe_label,
    get_data_config_for_timeframe
)
from firebase_manager import FirebaseManager
from system_health import SystemHealthMonitor, monitor_health
from heartbeat import HeartbeatManager
from btc_predictor_automated import (
    BitcoinMLPredictor,
    get_bitcoin_data_realtime,
    get_current_btc_price,
    add_technical_indicators
)
from timezone_utils import get_local_now

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WatchdogTimer:
    """Watchdog timer to detect hanging processes"""
    
    def __init__(self, timeout=900):  # 15 minutes default
        self.timeout = timeout
        self.last_activity = time.time()
        self.is_running = False
        self.thread = None
    
    def reset(self):
        """Reset watchdog timer"""
        self.last_activity = time.time()
    
    def start(self):
        """Start watchdog monitoring"""
        self.is_running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        logger.info(f"🐕 Watchdog started (timeout: {self.timeout}s)")
    
    def stop(self):
        """Stop watchdog"""
        self.is_running = False
    
    def _monitor(self):
        """Monitor for hanging"""
        while self.is_running:
            time.sleep(30)
            elapsed = time.time() - self.last_activity
            
            if elapsed > self.timeout:
                logger.error(f"❌ WATCHDOG TIMEOUT! No activity for {elapsed:.0f}s")
                logger.error("🔄 Force restarting...")
                os._exit(1)  # Force exit, systemd will restart


class EnhancedMultiTimeframePredictionScheduler:
    """Enhanced scheduler with proper error handling and memory management"""
    
    def __init__(self):
        self.predictor = None
        self.firebase = None
        self.health_monitor = SystemHealthMonitor()
        self.heartbeat = None
        self.watchdog = WatchdogTimer(timeout=900)
        self.is_running = False
        
        # Timeframe management
        self.active_timeframes = PREDICTION_CONFIG['active_timeframes']
        self.priority_timeframes = PREDICTION_CONFIG['priority_timeframes']
        self.last_prediction_time = {}
        self.prediction_counters = {tf: 0 for tf in self.active_timeframes}
        
        # Data caching
        self.data_cache = {}
        self.data_cache_time = {}
        
        # Error tracking
        self.consecutive_failures = 0
        self.total_predictions = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        self.last_successful_prediction = None
        
        # Market state
        self.current_volatility = None
        self.current_volume_ratio = None
        self.current_trend = None
        
        # Memory tracking
        self.start_memory = None
        self.max_memory_seen = 0
        
        # Shutdown handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 Enhanced Scheduler initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"\n⚠️ Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def _check_memory(self):
        """Check and manage memory usage"""
        try:
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / 1024 / 1024
            
            self.max_memory_seen = max(self.max_memory_seen, mem_mb)
            
            if mem_mb > HEALTH_CONFIG['max_memory_mb']:
                logger.warning(f"⚠️ High memory: {mem_mb:.0f}MB (max: {HEALTH_CONFIG['max_memory_mb']}MB)")
                self._aggressive_memory_cleanup()
                
                # Check again
                mem_mb = process.memory_info().rss / 1024 / 1024
                if mem_mb > HEALTH_CONFIG['max_memory_mb'] * 1.2:
                    logger.error(f"❌ Memory still high after cleanup: {mem_mb:.0f}MB")
                    logger.error("🔄 Restarting to free memory...")
                    self.stop()
                    os._exit(1)
            
            return mem_mb
            
        except Exception as e:
            logger.error(f"❌ Error checking memory: {e}")
            return 0
    
    def _aggressive_memory_cleanup(self):
        """Aggressive memory cleanup"""
        try:
            logger.info("🧹 Running aggressive memory cleanup...")
            
            # Clear data cache
            self.data_cache.clear()
            self.data_cache_time.clear()
            
            # Clear TensorFlow/Keras session
            try:
                import tensorflow as tf
                from keras import backend as K
                K.clear_session()
                tf.compat.v1.reset_default_graph()
                logger.info("✅ TensorFlow session cleared")
            except Exception as e:
                logger.debug(f"TF clear failed: {e}")
            
            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()
            
            time.sleep(1)
            
            mem_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            logger.info(f"✅ Memory after cleanup: {mem_after:.0f}MB")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    def _initialize_firebase(self):
        """Initialize Firebase with retry"""
        max_retries = 5
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🔗 Firebase connection attempt {attempt + 1}/{max_retries}...")
                self.firebase = FirebaseManager()
                
                if self.firebase.connected:
                    logger.info("✅ Firebase connected")
                    self.heartbeat = HeartbeatManager(self.firebase, update_interval=30)
                    self.heartbeat.send_status_change('starting', 'System initializing')
                    return True
                
            except Exception as e:
                logger.warning(f"⚠️ Firebase failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (2 ** attempt))
        
        logger.error("❌ Firebase connection failed")
        return False
    
    def initialize_models(self):
        """Initialize ML models with proper error handling"""
        try:
            logger.info("🔧 Initializing models...")
            
            # Create predictor
            self.predictor = BitcoinMLPredictor()
            
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
                logger.info("📚 No models found, training new ones...")
                return self.train_models()
                
        except Exception as e:
            logger.error(f"❌ Model initialization error: {e}")
            traceback.print_exc()
            return False
    
    def train_models(self):
        """Train models with proper error handling"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                logger.info(f"\n{'='*80}")
                logger.info(f"🤖 TRAINING MODELS (Attempt {attempt + 1}/{max_retries})")
                logger.info(f"{'='*80}")
                
                if self.heartbeat:
                    self.heartbeat.send_status_change('training', 'Training ML models')
                
                # Fetch training data
                df = self._fetch_training_data_comprehensive()
                
                if df is None or len(df) < 200:
                    logger.error("❌ Insufficient training data")
                    if attempt < max_retries - 1:
                        time.sleep(30)
                        continue
                    return False
                
                # Add indicators
                df = add_technical_indicators(df)
                
                # Train with memory monitoring
                mem_before = self._check_memory()
                logger.info(f"Memory before training: {mem_before:.0f}MB")
                
                success = self.predictor.train_models(df, epochs=50, batch_size=32)
                
                mem_after = self._check_memory()
                logger.info(f"Memory after training: {mem_after:.0f}MB")
                
                if success:
                    # Save metrics to Firebase
                    if self.firebase and self.firebase.connected:
                        self.firebase.save_model_performance(self.predictor.metrics)
                    
                    logger.info("✅ Training completed\n")
                    
                    if self.heartbeat:
                        self.heartbeat.send_status_change('trained', 'Models trained')
                    
                    # Cleanup after training
                    self._aggressive_memory_cleanup()
                    
                    return True
                else:
                    logger.warning(f"⚠️ Training attempt {attempt + 1} failed")
                    if attempt < max_retries - 1:
                        self._aggressive_memory_cleanup()
                        time.sleep(60)
                
            except Exception as e:
                logger.error(f"❌ Training error: {e}")
                traceback.print_exc()
                self._aggressive_memory_cleanup()
                
                if attempt < max_retries - 1:
                    time.sleep(60)
        
        logger.error("❌ Training failed after all attempts")
        return False
    
    def _fetch_training_data_comprehensive(self):
        """Fetch training data with fallbacks"""
        strategies = [
            ('hour', 30),
            ('hour', 60),
            ('day', 90),
        ]
        
        for interval, days in strategies:
            try:
                logger.info(f"📡 Fetching {days} days of {interval} data...")
                df = get_bitcoin_data_realtime(days=days, interval=interval)
                
                if df is not None and len(df) >= 200:
                    logger.info(f"✅ Got {len(df)} data points")
                    return df
                
            except Exception as e:
                logger.warning(f"⚠️ Failed {interval} data: {e}")
                continue
        
        return None
    
    def get_data_for_category(self, category, force_refresh=False):
        """Get data with caching"""
        try:
            # Check cache
            if not force_refresh and category in self.data_cache:
                if category in self.data_cache_time:
                    age = (datetime.now() - self.data_cache_time[category]).total_seconds()
                    if age < 120:
                        return self.data_cache[category]
            
            # Get config
            data_req = PREDICTION_CONFIG['data_requirements'][category]
            
            # Fetch data
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    df = get_bitcoin_data_realtime(
                        days=data_req['days'],
                        interval=data_req['interval']
                    )
                    
                    if df is not None and len(df) >= data_req['min_points']:
                        df = add_technical_indicators(df)
                        
                        # Cache
                        self.data_cache[category] = df
                        self.data_cache_time[category] = datetime.now()
                        
                        return df
                    
                except Exception as e:
                    logger.warning(f"⚠️ Data fetch attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting data: {e}")
            return None
    
    def run_predictions_smart(self):
        """Run predictions with full error handling"""
        try:
            self.watchdog.reset()
            
            logger.info(f"\n{'='*80}")
            logger.info(f"🔮 PREDICTIONS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*80}")
            
            # Check memory first
            mem_mb = self._check_memory()
            logger.info(f"💾 Memory: {mem_mb:.0f}MB (Peak: {self.max_memory_seen:.0f}MB)")
            
            # Send heartbeat
            if self.heartbeat:
                try:
                    self.heartbeat.send_heartbeat({
                        'last_activity': 'running_predictions',
                        'total_predictions': self.total_predictions,
                        'memory_mb': mem_mb
                    })
                except Exception as e:
                    logger.warning(f"⚠️ Heartbeat failed: {e}")
            
            # Validate predictor
            if not self.predictor or not self.predictor.is_trained:
                logger.error("❌ Predictor not ready")
                self.consecutive_failures += 1
                
                if self.consecutive_failures >= 5:
                    logger.error("❌ Too many failures, reinitializing...")
                    if not self.initialize_models():
                        logger.error("❌ Reinitialization failed, restarting...")
                        os._exit(1)
                
                return
            
            # Get data
            categories_needed = set()
            for tf in self.active_timeframes:
                category = get_timeframe_category(tf)
                categories_needed.add(category)
            
            data_by_category = {}
            for category in categories_needed:
                df = self.get_data_for_category(category)
                if df is not None:
                    data_by_category[category] = df
            
            if not data_by_category:
                logger.error("❌ No data available")
                self.consecutive_failures += 1
                return
            
            # Get current price
            current_price = None
            for df in data_by_category.values():
                if df is not None:
                    current_price = df.iloc[0]['price']
                    break
            
            if current_price:
                logger.info(f"💰 BTC: ${current_price:,.2f}")
            
            # Run predictions
            predictions_made = 0
            predictions_by_category = {}
            
            for tf in self.active_timeframes:
                category = get_timeframe_category(tf)
                if category not in predictions_by_category:
                    predictions_by_category[category] = []
                predictions_by_category[category].append(tf)
            
            for category, timeframes in predictions_by_category.items():
                if category not in data_by_category:
                    continue
                
                df = data_by_category[category]
                
                for tf in timeframes:
                    try:
                        self.watchdog.reset()
                        
                        logger.info(f"\n⏱️  {get_timeframe_label(tf)}...")
                        
                        prediction = self.predictor.predict(df, tf)
                        
                        if prediction:
                            self._display_prediction_summary(prediction)
                            
                            if self.firebase and self.firebase.connected:
                                try:
                                    doc_id = self.firebase.save_prediction(prediction)
                                    
                                    if doc_id:
                                        self.last_prediction_time[tf] = datetime.now()
                                        self.prediction_counters[tf] += 1
                                        predictions_made += 1
                                        self.successful_predictions += 1
                                        self.last_successful_prediction = datetime.now()
                                        logger.info(f"✅ Saved: {doc_id}")
                                    else:
                                        self.failed_predictions += 1
                                except Exception as e:
                                    logger.error(f"❌ Firebase save failed: {e}")
                                    self.failed_predictions += 1
                        else:
                            logger.warning(f"⚠️ Prediction failed")
                            self.failed_predictions += 1
                            
                    except Exception as e:
                        logger.error(f"❌ Error predicting {get_timeframe_label(tf)}: {e}")
                        traceback.print_exc()
                        self.failed_predictions += 1
                        continue
            
            self.total_predictions += predictions_made
            
            if predictions_made > 0:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
            
            # Check if stuck
            if self.last_successful_prediction:
                time_since = (datetime.now() - self.last_successful_prediction).total_seconds()
                if time_since > 600:  # 10 minutes
                    logger.error(f"❌ No successful predictions for {time_since:.0f}s")
                    logger.error("🔄 Restarting...")
                    os._exit(1)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"✅ Cycle complete - {predictions_made} predictions")
            logger.info(f"📊 Success: {self.successful_predictions}/{self.total_predictions}")
            logger.info(f"{'='*80}\n")
            
            # Periodic cleanup
            if self.total_predictions % 10 == 0:
                self._aggressive_memory_cleanup()
            
            self.watchdog.reset()
            
        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR in prediction cycle: {e}")
            traceback.print_exc()
            self.consecutive_failures += 1
            
            if self.consecutive_failures >= 5:
                logger.error("❌ Too many consecutive failures, restarting...")
                os._exit(1)
    
    def _display_prediction_summary(self, prediction):
        """Display prediction summary"""
        arrow = "🟢 ↗️" if prediction['price_change'] > 0 else "🔴 ↘️"
        tf_label = get_timeframe_label(prediction['timeframe_minutes'])
        logger.info(f"   {arrow} ${prediction['predicted_price']:,.2f} "
                   f"({prediction['price_change_pct']:+.2f}%) - "
                   f"Confidence: {prediction['confidence']:.1f}%")
    
    def validate_predictions(self):
        """Validate predictions with error handling"""
        try:
            self.watchdog.reset()
            
            if not self.firebase or not self.firebase.connected:
                return
            
            predictions = self.firebase.get_unvalidated_predictions()
            
            if not predictions:
                return
            
            current_price = get_current_btc_price()
            
            if not current_price:
                return
            
            validated_count = 0
            for pred in predictions:
                try:
                    success = self.firebase.validate_prediction(
                        pred['doc_id'],
                        current_price,
                        pred['predicted_price'],
                        pred['trend']
                    )
                    
                    if success:
                        validated_count += 1
                        
                except Exception as e:
                    logger.error(f"❌ Validation error: {e}")
                    continue
            
            if validated_count > 0:
                logger.info(f"✅ Validated {validated_count} predictions")
                self.update_statistics()
            
            self.watchdog.reset()
            
        except Exception as e:
            logger.error(f"❌ Validation cycle error: {e}")
    
    def update_statistics(self):
        """Update statistics"""
        try:
            if not self.firebase or not self.firebase.connected:
                return
            
            overall_stats = self.firebase.get_statistics(days=7)
            
            if overall_stats and overall_stats.get('total_predictions', 0) > 0:
                self.firebase.save_statistics(overall_stats)
            
        except Exception as e:
            logger.error(f"❌ Stats update error: {e}")
    
    def periodic_health_check(self):
        """Health check"""
        try:
            self.watchdog.reset()
            
            logger.info("\n🏥 HEALTH CHECK")
            
            mem_mb = self._check_memory()
            
            report = monitor_health(self.firebase if self.firebase and self.firebase.connected else None)
            
            if self.heartbeat:
                self.heartbeat.send_heartbeat({
                    'health_status': report['overall_status'],
                    'memory_mb': mem_mb,
                    'total_predictions': self.total_predictions
                })
            
            # Check if critical
            if report['overall_status'] == 'CRITICAL':
                logger.error("❌ System critical, restarting...")
                os._exit(1)
            
            self.watchdog.reset()
            
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
    
    def periodic_cleanup(self):
        """Cleanup"""
        try:
            self.watchdog.reset()
            
            logger.info("🗑️ Cleanup...")
            
            self._aggressive_memory_cleanup()
            
            if self.firebase and self.firebase.connected:
                self.firebase.cleanup_old_data(days=30)
            
            self.watchdog.reset()
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    def start(self):
        """Start scheduler"""
        logger.info(f"\n{'='*80}")
        logger.info("🚀 STARTING BITCOIN PREDICTOR")
        logger.info(f"{'='*80}")
        
        # Record start memory
        self.start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        logger.info(f"💾 Starting memory: {self.start_memory:.0f}MB")
        
        # Start watchdog
        self.watchdog.start()
        
        # Initialize
        if not self._initialize_firebase():
            logger.error("❌ Firebase init failed")
            return
        
        if not self.initialize_models():
            logger.error("❌ Model init failed")
            return
        
        # Setup schedules
        logger.info("\n📅 Setting up schedule:")
        logger.info("   • Predictions: Every 5 minutes")
        logger.info("   • Validation: Every 60 seconds")
        logger.info("   • Health: Every 5 minutes")
        logger.info("   • Heartbeat: Every 30 seconds")
        logger.info("   • Cleanup: Every 30 minutes")
        
        schedule.every(300).seconds.do(self.run_predictions_smart)
        schedule.every(60).seconds.do(self.validate_predictions)
        schedule.every(300).seconds.do(self.periodic_health_check)
        schedule.every(1800).seconds.do(self.periodic_cleanup)
        
        if self.heartbeat:
            schedule.every(30).seconds.do(lambda: self.heartbeat.send_heartbeat({
                'status': 'running',
                'predictions': self.total_predictions
            }))
        
        # Initial run
        logger.info("\n🎯 Initial run...")
        self.run_predictions_smart()
        self.periodic_health_check()
        
        if self.heartbeat:
            self.heartbeat.send_status_change('running', 'System operational')
        
        self.is_running = True
        logger.info("\n✅ System started!\n")
        
        # Main loop with error recovery
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("\n⚠️ Shutdown requested...")
                break
            except Exception as e:
                logger.error(f"\n❌ Main loop error: {e}")
                traceback.print_exc()
                time.sleep(5)
        
        self.stop()
    
    def stop(self):
        """Stop scheduler"""
        logger.info("🛑 Stopping...")
        self.is_running = False
        
        if self.watchdog:
            self.watchdog.stop()
        
        if self.heartbeat:
            self.heartbeat.send_shutdown_signal()
        
        logger.info("✅ Stopped\n")


def main():
    """Main entry"""
    try:
        scheduler = EnhancedMultiTimeframePredictionScheduler()
        scheduler.start()
        
    except KeyboardInterrupt:
        logger.info("\n\n❌ Interrupted")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()