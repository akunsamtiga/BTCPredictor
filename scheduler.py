"""
Improved Scheduler with All Enhancements
Includes: Backtesting, Alerting, Caching, Better Error Handling
"""

import time
import schedule
import logging
import gc
import signal
import sys
import os
import psutil
from datetime import datetime, timedelta
from typing import Dict, List

from config import (
    PREDICTION_CONFIG, MODEL_CONFIG, HEALTH_CONFIG, BACKTEST_CONFIG, STRATEGY_CONFIG,
    get_timeframe_category, get_timeframe_label, is_paper_trading,
    is_production, get_config_summary
)
from firebase_manager import FirebaseManager
from system_health import SystemHealthMonitor
from heartbeat import HeartbeatManager
from alert_system import get_alert_manager, AlertSeverity
from backtest import BacktestEngine, run_comprehensive_backtest
from cache_manager import get_cache
from btc_predictor_automated import ImprovedBitcoinPredictor
from btc_predictor_automated import (
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
    
    def __init__(self, timeout=1200):  # 20 minutes
        self.timeout = timeout
        self.last_activity = time.time()
        self.is_running = False
        self.thread = None
    
    def reset(self):
        """Reset watchdog timer"""
        self.last_activity = time.time()
    
    def start(self):
        """Start watchdog monitoring"""
        import threading
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
                os._exit(1)


class ImprovedScheduler:
    """Improved scheduler with all enhancements"""
    
    def __init__(self):
        self.predictor = None
        self.firebase = None
        self.health_monitor = SystemHealthMonitor()
        self.heartbeat = None
        self.alert_manager = None
        self.backtest_engine = None
        self.cache = get_cache()
        self.watchdog = WatchdogTimer(timeout=HEALTH_CONFIG['watchdog_timeout'])
        self.is_running = False
        
        # Timeframe management
        self.active_timeframes = []
        self.enabled_timeframes = set()
        self.last_prediction_time = {}
        self.prediction_counters = {}
        
        # Performance tracking
        self.consecutive_failures = 0
        self.total_predictions = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        self.last_successful_prediction = None
        
        # Recent performance for confidence calculation
        self.recent_results = []
        self.recent_accuracy = None
        
        # Cooldown management
        self.loss_streak = 0
        self.cooldown_until = None
        
        # Memory tracking
        self.start_memory = None
        
        # Shutdown handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 Improved Scheduler initialized")
        logger.info(f"Configuration: {get_config_summary()}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"\n⚠️ Received signal {signum}, shutting down...")
        if self.alert_manager:
            self.alert_manager.alert_system_shutdown("signal received")
        self.stop()
        sys.exit(0)
    
    def _check_memory(self) -> float:
        """Check and manage memory usage"""
        try:
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / 1024 / 1024
            
            max_memory = HEALTH_CONFIG['max_memory_mb']
            
            if mem_mb > max_memory:
                logger.warning(f"⚠️ High memory: {mem_mb:.0f}MB (max: {max_memory}MB)")
                
                if self.alert_manager:
                    self.alert_manager.alert_high_memory(mem_mb, max_memory)
                
                self._aggressive_memory_cleanup()
                
                # Check again
                mem_mb = process.memory_info().rss / 1024 / 1024
                if mem_mb > max_memory * 1.2:
                    logger.error(f"❌ Memory still high: {mem_mb:.0f}MB")
                    if self.alert_manager:
                        self.alert_manager.send_alert(
                            "Critical Memory Usage",
                            f"Memory: {mem_mb:.0f}MB. System restarting.",
                            AlertSeverity.CRITICAL,
                            "critical_memory"
                        )
                    self.stop()
                    os._exit(1)
            
            return mem_mb
            
        except Exception as e:
            logger.error(f"❌ Error checking memory: {e}")
            return 0
    
    def _aggressive_memory_cleanup(self):
        """Aggressive memory cleanup"""
        try:
            logger.info("🧹 Running memory cleanup...")
            
            # Clear cache
            self.cache.clear()
            
            # Clear TensorFlow session
            try:
                import tensorflow as tf
                from keras import backend as K
                K.clear_session()
                tf.compat.v1.reset_default_graph()
            except:
                pass
            
            # Force GC
            for _ in range(3):
                gc.collect()
            
            time.sleep(1)
            
            mem_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            logger.info(f"✅ Memory after cleanup: {mem_after:.0f}MB")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    def initialize(self) -> bool:
        """Initialize all components"""
        try:
            logger.info("\n" + "="*80)
            logger.info("🚀 INITIALIZING SYSTEM")
            logger.info("="*80)
            
            # Initialize Firebase
            if not self._initialize_firebase():
                return False
            
            # Initialize alert system
            self.alert_manager = get_alert_manager(self.firebase)
            
            # Initialize models
            if not self.initialize_models():
                return False
            
            # Run backtesting if enabled
            if BACKTEST_CONFIG.get('backtest_on_startup'):
                if not self.run_backtest():
                    logger.warning("⚠️ Backtest failed, but continuing...")
            
            # Set active timeframes
            self._set_active_timeframes()
            
            # Send startup alert
            if self.alert_manager:
                self.alert_manager.alert_system_startup()
            
            logger.info("\n✅ System initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    def _initialize_firebase(self) -> bool:
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
        if self.alert_manager:
            self.alert_manager.alert_firebase_disconnected()
        return False
    
    def initialize_models(self) -> bool:
        """Initialize ML models"""
        try:
            logger.info("🔧 Initializing models...")
            
            self.predictor = ImprovedBitcoinPredictor()
            
            if self.heartbeat:
                self.heartbeat.send_status_change('loading_models', 'Loading ML models')
            
            # Try to load existing models
            if self.predictor.load_models():
                logger.info("✅ Loaded existing models")
                
                if self.predictor.needs_retraining():
                    logger.info("⚠️ Models need retraining...")
                    return self.train_models()
                
                return True
            else:
                logger.info("📚 No models found, training new ones...")
                return self.train_models()
                
        except Exception as e:
            logger.error(f"❌ Model initialization error: {e}")
            return False
    
    def train_models(self) -> bool:
        """Train models"""
        try:
            logger.info(f"\n{'='*80}")
            logger.info("🤖 TRAINING MODELS")
            logger.info(f"{'='*80}")
            
            if self.heartbeat:
                self.heartbeat.send_status_change('training', 'Training ML models')
            
            # Fetch training data
            df = get_bitcoin_data_realtime(days=30, interval='hour')
            
            if df is None or len(df) < 500:
                logger.error("❌ Insufficient training data")
                if self.alert_manager:
                    self.alert_manager.send_alert(
                        "Training Data Insufficient",
                        f"Only {len(df) if df is not None else 0} data points available",
                        AlertSeverity.CRITICAL,
                        "training_data"
                    )
                return False
            
            df = add_technical_indicators(df)
            
            mem_before = self._check_memory()
            logger.info(f"Memory before training: {mem_before:.0f}MB")
            
            success = self.predictor.train_models(df, epochs=40, batch_size=64)
            
            mem_after = self._check_memory()
            logger.info(f"Memory after training: {mem_after:.0f}MB")
            
            if success:
                # Save metrics
                if self.firebase and self.firebase.connected:
                    self.firebase.save_model_performance(self.predictor.metrics)
                
                # Send alert
                if self.alert_manager:
                    self.alert_manager.alert_model_retrain(True, self.predictor.metrics)
                
                logger.info("✅ Training completed\n")
                
                if self.heartbeat:
                    self.heartbeat.send_status_change('trained', 'Models trained')
                
                self._aggressive_memory_cleanup()
                return True
            else:
                if self.alert_manager:
                    self.alert_manager.alert_model_retrain(False)
                return False
                
        except Exception as e:
            logger.error(f"❌ Training error: {e}")
            if self.alert_manager:
                self.alert_manager.alert_model_retrain(False)
            return False
    
    def run_backtest(self) -> bool:
        """Run comprehensive backtest"""
        try:
            logger.info(f"\n{'='*80}")
            logger.info("🧪 RUNNING BACKTEST")
            logger.info(f"{'='*80}")
            
            timeframes_to_test = PREDICTION_CONFIG['active_timeframes']
            
            self.backtest_engine = run_comprehensive_backtest(
                self.predictor,
                get_bitcoin_data_realtime,
                timeframes_to_test,
                self.firebase
            )
            
            # Check which timeframes passed
            passed_timeframes = []
            failed_timeframes = []
            
            for tf in timeframes_to_test:
                if self.backtest_engine.should_enable_timeframe(tf):
                    passed_timeframes.append(tf)
                else:
                    failed_timeframes.append(tf)
                    label = get_timeframe_label(tf)
                    
                    # Alert on failed backtest
                    if self.alert_manager and tf in self.backtest_engine.results:
                        results = self.backtest_engine.results[tf]
                        avg_winrate = sum(r.win_rate for r in results.values()) / len(results)
                        min_winrate = BACKTEST_CONFIG.get('min_backtest_winrate', 52.0)
                        
                        self.alert_manager.alert_backtest_failed(label, avg_winrate, min_winrate)
            
            logger.info(f"\n✅ Backtest completed:")
            logger.info(f"   Passed: {len(passed_timeframes)} timeframes")
            logger.info(f"   Failed: {len(failed_timeframes)} timeframes")
            
            if len(passed_timeframes) == 0:
                logger.error("❌ No timeframes passed backtest!")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Backtest error: {e}")
            return False
    
    def _set_active_timeframes(self):
        """Set active timeframes based on backtest results"""
        if self.backtest_engine and BACKTEST_CONFIG.get('backtest_before_trading'):
            # Only enable timeframes that passed backtest
            self.active_timeframes = [
                tf for tf in PREDICTION_CONFIG['active_timeframes']
                if self.backtest_engine.should_enable_timeframe(tf)
            ]
        else:
            # Use all configured timeframes
            self.active_timeframes = PREDICTION_CONFIG['active_timeframes']
        
        self.enabled_timeframes = set(self.active_timeframes)
        
        for tf in self.active_timeframes:
            self.prediction_counters[tf] = 0
        
        logger.info(f"\n📋 Active timeframes: {[get_timeframe_label(tf) for tf in self.active_timeframes]}")
    
    def _check_cooldown(self) -> bool:
        """Check if system is in cooldown"""
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            remaining = (self.cooldown_until - datetime.now()).total_seconds() / 60
            logger.info(f"⏸️ In cooldown for {remaining:.1f} more minutes")
            return True
        return False
    
    def run_predictions(self):
        """Run predictions with all improvements"""
        try:
            self.watchdog.reset()
            
            # Check cooldown
            if self._check_cooldown():
                return
            
            logger.info(f"\n{'='*80}")
            logger.info(f"🔮 PREDICTIONS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*80}")
            
            # Check memory
            mem_mb = self._check_memory()
            logger.info(f"💾 Memory: {mem_mb:.0f}MB")
            
            # Send heartbeat
            if self.heartbeat:
                self.heartbeat.send_heartbeat({
                    'last_activity': 'running_predictions',
                    'total_predictions': self.total_predictions,
                    'memory_mb': mem_mb
                })
            
            # Validate predictor
            if not self.predictor or not self.predictor.is_trained:
                logger.error("❌ Predictor not ready")
                self.consecutive_failures += 1
                
                if self.consecutive_failures >= 3:
                    if self.alert_manager:
                        self.alert_manager.alert_consecutive_failures(self.consecutive_failures)
                    self.initialize_models()
                
                return
            
            # Get current price
            current_price = get_current_btc_price()
            if current_price:
                logger.info(f"💰 BTC: ${current_price:,.2f}")
            
            # Fetch data per category
            data_by_category = {}
            categories_needed = set()
            
            for tf in self.active_timeframes:
                category = get_timeframe_category(tf)
                categories_needed.add(category)
            
            for category in categories_needed:
                from config import get_data_config_for_timeframe
                # Use first timeframe in category to get config
                sample_tf = next((tf for tf in self.active_timeframes 
                                 if get_timeframe_category(tf) == category), None)
                
                if sample_tf:
                    data_config = get_data_config_for_timeframe(sample_tf)
                    
                    # Try cache first
                    cache_key = f"data:{category}:{data_config['days']}:{data_config['interval']}"
                    df = self.cache.get(cache_key)
                    
                    if df is None:
                        logger.info(f"📡 Fetching {category} data...")
                        df = get_bitcoin_data_realtime(
                            days=data_config['days'],
                            interval=data_config['interval']
                        )
                        
                        if df is not None:
                            df = add_technical_indicators(df)
                            self.cache.set(cache_key, df, ttl=300)  # 5 minutes
                            data_by_category[category] = df
                    else:
                        logger.info(f"📦 Using cached {category} data")
                        data_by_category[category] = df
            
            if not data_by_category:
                logger.error("❌ No data available")
                self.consecutive_failures += 1
                return
            
            # Run predictions
            predictions_made = 0
            
            for tf in self.active_timeframes:
                try:
                    self.watchdog.reset()
                    
                    category = get_timeframe_category(tf)
                    if category not in data_by_category:
                        continue
                    
                    df = data_by_category[category]
                    label = get_timeframe_label(tf)
                    
                    logger.info(f"\n⏱️ {label}...")
                    
                    # Update predictor with recent accuracy
                    if self.recent_accuracy is not None:
                        self.predictor.recent_accuracy = self.recent_accuracy
                    
                    prediction = self.predictor.predict(df, tf)
                    
                    if prediction:
                        self._display_prediction(prediction)
                        
                        # Save to Firebase if not paper trading or if allowed
                        if self.firebase and self.firebase.connected:
                            doc_id = self.firebase.save_prediction(prediction)
                            
                            if doc_id:
                                self.last_prediction_time[tf] = datetime.now()
                                self.prediction_counters[tf] += 1
                                predictions_made += 1
                                self.successful_predictions += 1
                                self.last_successful_prediction = datetime.now()
                                self.consecutive_failures = 0
                                logger.info(f"✅ Saved: {doc_id}")
                            else:
                                self.failed_predictions += 1
                    else:
                        logger.info(f"⏭️ Skipped (low confidence)")
                        
                except Exception as e:
                    logger.error(f"❌ Error predicting {get_timeframe_label(tf)}: {e}")
                    self.failed_predictions += 1
                    continue
            
            self.total_predictions += predictions_made
            
            # Check for stagnation
            if self.last_successful_prediction:
                time_since = (datetime.now() - self.last_successful_prediction).total_seconds()
                if time_since > 600:  # 10 minutes
                    logger.error(f"❌ No successful predictions for {time_since:.0f}s")
                    if self.alert_manager:
                        self.alert_manager.send_alert(
                            "System Stagnant",
                            f"No predictions for {time_since/60:.1f} minutes",
                            AlertSeverity.CRITICAL,
                            "stagnant"
                        )
            
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
            import traceback
            traceback.print_exc()
            self.consecutive_failures += 1
            
            if self.consecutive_failures >= 3:
                if self.alert_manager:
                    self.alert_manager.alert_consecutive_failures(self.consecutive_failures)
    
    def _display_prediction(self, prediction: Dict):
        """Display prediction summary"""
        arrow = "🟢 ↗️" if prediction['price_change'] > 0 else "🔴 ↘️"
        label = get_timeframe_label(prediction['timeframe_minutes'])
        logger.info(f"   {arrow} ${prediction['predicted_price']:,.2f} "
                   f"({prediction['price_change_pct']:+.2f}%) - "
                   f"Confidence: {prediction['confidence']:.1f}%")
    
    def validate_predictions(self):
        """Validate predictions and update statistics"""
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
            wins = 0
            losses = 0
            
            for pred in predictions:
                try:
                    # Validate
                    result = self.firebase.validate_prediction(
                        pred['doc_id'],
                        current_price,
                        pred['predicted_price'],
                        pred['trend']
                    )
                    
                    if result:
                        validated_count += 1
                        
                        # Track result
                        is_win = 'WIN' in str(result).upper()
                        self.recent_results.append(is_win)
                        
                        # Keep only last 100 results
                        if len(self.recent_results) > 100:
                            self.recent_results.pop(0)
                        
                        if is_win:
                            wins += 1
                            self.loss_streak = 0
                        else:
                            losses += 1
                            self.loss_streak += 1
                        
                except Exception as e:
                    logger.error(f"❌ Validation error: {e}")
                    continue
            
            if validated_count > 0:
                logger.info(f"✅ Validated {validated_count} predictions ({wins}W/{losses}L)")
                
                # Update recent accuracy
                if len(self.recent_results) >= 10:
                    self.recent_accuracy = sum(self.recent_results) / len(self.recent_results) * 100
                
                # Check for cooldown
                max_streak = STRATEGY_CONFIG['risk_management']['cooldown_after_loss_streak']
                if self.loss_streak >= max_streak:
                    cooldown_min = STRATEGY_CONFIG['risk_management']['cooldown_duration_minutes']
                    self.cooldown_until = datetime.now() + timedelta(minutes=cooldown_min)
                    logger.warning(f"⏸️ Entering cooldown for {cooldown_min} minutes after {self.loss_streak} losses")
                
                # Update statistics
                self.update_statistics()
            
            self.watchdog.reset()
            
        except Exception as e:
            logger.error(f"❌ Validation cycle error: {e}")
    
    def update_statistics(self):
        """Update and check statistics"""
        try:
            if not self.firebase or not self.firebase.connected:
                return
            
            # Get overall stats
            overall_stats = self.firebase.get_statistics(days=7)
            
            if overall_stats and overall_stats.get('total_predictions', 0) > 0:
                self.firebase.save_statistics(overall_stats)
                
                # Check win rate and alert if low
                win_rate = overall_stats.get('win_rate', 0)
                if self.alert_manager:
                    self.alert_manager.alert_low_winrate(win_rate, "Overall")
                
                # Check per timeframe
                for tf in self.active_timeframes:
                    tf_stats = self.firebase.get_statistics(timeframe_minutes=tf, days=7)
                    if tf_stats and tf_stats.get('total_predictions', 0) >= 10:
                        tf_winrate = tf_stats.get('win_rate', 0)
                        label = get_timeframe_label(tf)
                        
                        # Alert on low win rate
                        if self.alert_manager:
                            self.alert_manager.alert_low_winrate(tf_winrate, label)
                        
                        # Alert on good performance
                        if tf_winrate >= 65:
                            if self.alert_manager:
                                self.alert_manager.alert_good_performance(label, tf_winrate)
            
        except Exception as e:
            logger.error(f"❌ Stats update error: {e}")
    
    def periodic_health_check(self):
        """Periodic health check"""
        try:
            self.watchdog.reset()
            
            logger.info("\n🏥 HEALTH CHECK")
            
            mem_mb = self._check_memory()
            
            from system_health import monitor_health
            report = monitor_health(self.firebase if self.firebase and self.firebase.connected else None)
            
            if self.heartbeat:
                self.heartbeat.send_heartbeat({
                    'health_status': report['overall_status'],
                    'memory_mb': mem_mb,
                    'total_predictions': self.total_predictions
                })
            
            # Check disk space
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_gb = free / (1024**3)
            
            if free_gb < 1.0 and self.alert_manager:
                self.alert_manager.alert_disk_space_low(free_gb)
            
            # Check if critical
            if report['overall_status'] == 'CRITICAL':
                logger.error("❌ System critical")
                if self.alert_manager:
                    self.alert_manager.send_alert(
                        "System Health Critical",
                        "System health is critical. Check immediately.",
                        AlertSeverity.CRITICAL,
                        "health_critical"
                    )
            
            self.watchdog.reset()
            
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
    
    def start(self):
        """Start scheduler"""
        logger.info(f"\n{'='*80}")
        logger.info("🚀 STARTING BITCOIN PREDICTOR")
        logger.info(f"{'='*80}")
        logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
        logger.info(f"Trading Mode: {os.getenv('TRADING_MODE', 'paper')}")
        logger.info(f"{'='*80}\n")
        
        # Record start memory
        self.start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        logger.info(f"💾 Starting memory: {self.start_memory:.0f}MB")
        
        # Start watchdog
        self.watchdog.start()
        
        # Initialize
        if not self.initialize():
            logger.error("❌ Initialization failed")
            return
        
        # Setup schedules
        logger.info("\n📅 Setting up schedule:")
        logger.info("   • Predictions: Every 5 minutes")
        logger.info("   • Validation: Every 60 seconds")
        logger.info("   • Health: Every 5 minutes")
        logger.info("   • Heartbeat: Every 30 seconds")
        logger.info("   • Statistics: Every 10 minutes")
        
        schedule.every(300).seconds.do(self.run_predictions)
        schedule.every(60).seconds.do(self.validate_predictions)
        schedule.every(300).seconds.do(self.periodic_health_check)
        schedule.every(600).seconds.do(self.update_statistics)
        
        if self.heartbeat:
            schedule.every(30).seconds.do(lambda: self.heartbeat.send_heartbeat({
                'status': 'running',
                'predictions': self.total_predictions
            }))
        
        # Initial run
        logger.info("\n🎯 Initial run...")
        self.run_predictions()
        self.periodic_health_check()
        
        if self.heartbeat:
            self.heartbeat.send_status_change('running', 'System operational')
        
        self.is_running = True
        logger.info("\n✅ System started!\n")
        
        # Main loop
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("\n⚠️ Shutdown requested...")
                break
            except Exception as e:
                logger.error(f"\n❌ Main loop error: {e}")
                import traceback
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
    """Main entry point"""
    try:
        # Validate environment first
        from config import validate_environment
        validate_environment()
        
        scheduler = ImprovedScheduler()
        scheduler.start()
        
    except KeyboardInterrupt:
        logger.info("\n\n❌ Interrupted")
    except EnvironmentError as e:
        logger.error(f"\n❌ Environment error: {e}")
        logger.error("Please check your .env file and configuration.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()