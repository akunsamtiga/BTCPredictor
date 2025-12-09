"""
Smart Scheduler - Timeframe-Based Prediction
FIXED: Fetch more data for training (90 days minimum)
"""

import time
import logging
import gc
import signal
import sys
import os
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from config import (
    PREDICTION_CONFIG, MODEL_CONFIG, HEALTH_CONFIG, STRATEGY_CONFIG,
    get_timeframe_category, get_timeframe_label, get_data_config_for_timeframe
)
from firebase_manager import FirebaseManager
from system_health import SystemHealthMonitor
from heartbeat import HeartbeatManager
from alert_system import get_alert_manager, AlertSeverity
from cache_manager import get_cache
from btc_predictor_automated import ImprovedBitcoinPredictor
from btc_predictor_automated import (
    get_bitcoin_data_realtime, 
    get_current_btc_price,
    add_technical_indicators
)
from timezone_utils import get_local_now

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TimeframeScheduler:
    """
    Smart scheduler for each timeframe
    Each timeframe has its own schedule and data requirements
    """
    
    def __init__(self, timeframe_minutes: int):
        self.timeframe = timeframe_minutes
        self.last_prediction = None
        self.prediction_count = 0
        self.category = get_timeframe_category(timeframe_minutes)
        self.data_config = get_data_config_for_timeframe(timeframe_minutes)
        
    def should_predict_now(self) -> bool:
        """
        Check if this timeframe should make prediction now
        Based on current time alignment with timeframe
        """
        now = datetime.now()
        current_minute = now.hour * 60 + now.minute
        
        # Check if current time is aligned with timeframe interval
        if current_minute % self.timeframe == 0:
            # Check if we haven't predicted in this interval yet
            if self.last_prediction is None:
                return True
            
            # Check if enough time has passed since last prediction
            time_since_last = (now - self.last_prediction).total_seconds() / 60
            if time_since_last >= self.timeframe:
                return True
        
        return False
    
    def mark_prediction_made(self):
        """Mark that prediction was made"""
        self.last_prediction = datetime.now()
        self.prediction_count += 1
    
    def get_next_prediction_time(self) -> datetime:
        """Get next scheduled prediction time"""
        now = datetime.now()
        current_minute = now.hour * 60 + now.minute
        
        # Calculate next aligned time
        next_minute = ((current_minute // self.timeframe) + 1) * self.timeframe
        
        # Handle day rollover
        next_hour = next_minute // 60
        next_min = next_minute % 60
        
        if next_hour >= 24:
            next_day = now + timedelta(days=1)
            return next_day.replace(hour=next_hour % 24, minute=next_min, second=0, microsecond=0)
        
        return now.replace(hour=next_hour, minute=next_min, second=0, microsecond=0)
    
    def get_independent_data(self) -> Optional[object]:
        """
        Fetch data specifically for this timeframe
        Each timeframe gets its own optimized dataset
        """
        try:
            logger.info(f"📡 [{get_timeframe_label(self.timeframe)}] Fetching independent data...")
            logger.info(f"   Config: {self.data_config['days']} days, {self.data_config['interval']} interval")
            
            df = get_bitcoin_data_realtime(
                days=self.data_config['days'],
                interval=self.data_config['interval']
            )
            
            if df is None or len(df) < self.data_config['min_points']:
                logger.warning(f"⚠️ Insufficient data: {len(df) if df is not None else 0}")
                return None
            
            df = add_technical_indicators(df)
            
            logger.info(f"✅ Data ready: {len(df)} points")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching data: {e}")
            return None


class WatchdogTimer:
    """Watchdog timer to detect hanging processes"""
    
    def __init__(self, timeout=1200):
        self.timeout = timeout
        self.last_activity = time.time()
        self.is_running = False
        self.thread = None
    
    def reset(self):
        self.last_activity = time.time()
    
    def start(self):
        import threading
        self.is_running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        logger.info(f"🕐 Watchdog started (timeout: {self.timeout}s)")
    
    def stop(self):
        self.is_running = False
    
    def _monitor(self):
        while self.is_running:
            time.sleep(30)
            elapsed = time.time() - self.last_activity
            
            if elapsed > self.timeout:
                logger.error(f"❌ WATCHDOG TIMEOUT! No activity for {elapsed:.0f}s")
                logger.error("🔄 Force restarting...")
                os._exit(1)


class ImprovedScheduler:
    """
    Improved scheduler with smart timeframe-based prediction
    Each timeframe runs independently at optimal intervals
    """
    
    def __init__(self):
        self.predictor = None
        self.firebase = None
        self.health_monitor = SystemHealthMonitor()
        self.heartbeat = None
        self.alert_manager = None
        self.cache = get_cache()
        self.watchdog = WatchdogTimer(timeout=HEALTH_CONFIG['watchdog_timeout'])
        self.is_running = False
        
        # Create scheduler for each timeframe
        self.timeframe_schedulers: Dict[int, TimeframeScheduler] = {}
        for tf in PREDICTION_CONFIG['active_timeframes']:
            self.timeframe_schedulers[tf] = TimeframeScheduler(tf)
        
        # Performance tracking
        self.consecutive_failures = 0
        self.total_predictions = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        self.last_successful_prediction = None
        
        # Recent performance
        self.recent_results = []
        self.recent_accuracy = None
        
        # Cooldown
        self.loss_streak = 0
        self.cooldown_until = None
        
        # Memory tracking
        self.start_memory = None
        
        # Shutdown handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 Smart Scheduler initialized")
        logger.info(f"📊 Active timeframes: {[get_timeframe_label(tf) for tf in PREDICTION_CONFIG['active_timeframes']]}")
        self._display_schedule()
    
    def _display_schedule(self):
        """Display prediction schedule for each timeframe"""
        logger.info("\n" + "="*80)
        logger.info("📅 PREDICTION SCHEDULE")
        logger.info("="*80)
        
        now = datetime.now()
        
        for tf in sorted(self.timeframe_schedulers.keys()):
            scheduler = self.timeframe_schedulers[tf]
            next_time = scheduler.get_next_prediction_time()
            time_until = (next_time - now).total_seconds() / 60
            
            label = get_timeframe_label(tf)
            category = scheduler.category.upper()
            
            logger.info(f"{label:8} ({category:12}) | Every {tf:4} min | Next: {next_time.strftime('%H:%M')} (in {time_until:.0f}min)")
        
        logger.info("="*80 + "\n")
    
    def _signal_handler(self, signum, frame):
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
            
            self.cache.clear()
            
            try:
                import tensorflow as tf
                from keras import backend as K
                K.clear_session()
                tf.compat.v1.reset_default_graph()
            except:
                pass
            
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
        """
        FIXED: Train models with MORE DATA (90 days minimum)
        """
        try:
            logger.info(f"\n{'='*80}")
            logger.info("🤖 TRAINING MODELS")
            logger.info(f"{'='*80}")
            
            if self.heartbeat:
                self.heartbeat.send_status_change('training', 'Training ML models')
            
            # ================================================================
            # FIXED: Fetch MORE training data (90 days with hourly interval)
            # This gives ~2160 data points before dropna()
            # ================================================================
            logger.info("📡 Fetching training data (90 days, hourly)...")
            df = get_bitcoin_data_realtime(days=90, interval='hour')
            
            if df is None:
                logger.error("❌ Failed to fetch training data")
                return False
            
            initial_points = len(df)
            logger.info(f"📊 Retrieved {initial_points} raw data points")
            
            # Add technical indicators
            logger.info("🔧 Adding technical indicators...")
            df = add_technical_indicators(df)
            
            # Clean data
            df_clean = df.dropna()
            clean_points = len(df_clean)
            
            logger.info(f"🧹 After cleaning: {clean_points} data points")
            logger.info(f"   Dropped: {initial_points - clean_points} rows with NaN")
            
            # ================================================================
            # CHECK: Need at least 1000 points for training
            # ================================================================
            if clean_points < 1000:
                logger.error(f"❌ Still insufficient data: {clean_points} < 1000")
                logger.error("   Possible solutions:")
                logger.error("   1. Increase days (currently 90)")
                logger.error("   2. Use 'day' interval for more history")
                logger.error("   3. Check API rate limits")
                
                # Try fallback: 180 days daily data
                logger.info("\n🔄 FALLBACK: Trying 180 days with daily interval...")
                df = get_bitcoin_data_realtime(days=180, interval='day')
                
                if df is None:
                    logger.error("❌ Fallback also failed")
                    return False
                
                logger.info(f"📊 Fallback retrieved {len(df)} data points")
                df = add_technical_indicators(df)
                df_clean = df.dropna()
                clean_points = len(df_clean)
                
                logger.info(f"🧹 After cleaning: {clean_points} data points")
                
                if clean_points < 500:
                    logger.error(f"❌ Even fallback insufficient: {clean_points}")
                    
                    if self.alert_manager:
                        self.alert_manager.send_alert(
                            "Training Data Insufficient",
                            f"Only {clean_points} data points available after cleaning.\n"
                            f"Training aborted. Please check data source.",
                            AlertSeverity.CRITICAL,
                            "training_data"
                        )
                    return False
            
            mem_before = self._check_memory()
            logger.info(f"💾 Memory before training: {mem_before:.0f}MB")
            
            # Train with cleaned data
            success = self.predictor.train_models(df_clean, epochs=20, batch_size=32)
            
            mem_after = self._check_memory()
            logger.info(f"💾 Memory after training: {mem_after:.0f}MB")
            
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
            import traceback
            traceback.print_exc()
            
            if self.alert_manager:
                self.alert_manager.alert_model_retrain(False)
            return False
    
    def _check_cooldown(self) -> bool:
        """Check if system is in cooldown"""
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            remaining = (self.cooldown_until - datetime.now()).total_seconds() / 60
            logger.info(f"⏸️ In cooldown for {remaining:.1f} more minutes")
            return True
        return False
    
    def run_prediction_cycle(self):
        """
        Smart prediction cycle - only predict timeframes that are due
        Each timeframe runs at its optimal interval
        """
        try:
            self.watchdog.reset()
            
            # Check cooldown
            if self._check_cooldown():
                return
            
            now = datetime.now()
            current_time = now.strftime('%H:%M:%S')
            
            logger.info(f"\n{'='*80}")
            logger.info(f"🔮 PREDICTION CYCLE - {now.strftime('%Y-%m-%d %H:%M:%S')}")
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
            
            # Check which timeframes should predict now
            timeframes_to_predict = []
            
            for tf, scheduler in self.timeframe_schedulers.items():
                if scheduler.should_predict_now():
                    timeframes_to_predict.append(tf)
            
            if not timeframes_to_predict:
                logger.info("⏭️ No timeframes scheduled for this cycle")
                logger.info(f"{'='*80}\n")
                return
            
            logger.info(f"\n📋 Timeframes scheduled for prediction:")
            for tf in timeframes_to_predict:
                label = get_timeframe_label(tf)
                category = self.timeframe_schedulers[tf].category
                logger.info(f"   • {label:8} ({category})")
            
            # Run predictions for scheduled timeframes
            predictions_made = 0
            
            for tf in timeframes_to_predict:
                try:
                    self.watchdog.reset()
                    
                    scheduler = self.timeframe_schedulers[tf]
                    label = get_timeframe_label(tf)
                    category = scheduler.category
                    
                    logger.info(f"\n{'─'*80}")
                    logger.info(f"⏱️ {label} ({category.upper()})")
                    logger.info(f"{'─'*80}")
                    
                    # Get independent data for this timeframe
                    df = scheduler.get_independent_data()
                    
                    if df is None:
                        logger.warning(f"⚠️ Skipped - no data available")
                        continue
                    
                    # Update predictor with recent accuracy
                    if self.recent_accuracy is not None:
                        self.predictor.recent_accuracy = self.recent_accuracy
                    
                    # Make prediction
                    logger.info(f"🧠 Analyzing with {len(df)} data points...")
                    prediction = self.predictor.predict(df, tf)
                    
                    if prediction:
                        self._display_prediction(prediction)
                        
                        # Save to Firebase
                        if self.firebase and self.firebase.connected:
                            doc_id = self.firebase.save_prediction(prediction)
                            
                            if doc_id:
                                scheduler.mark_prediction_made()
                                predictions_made += 1
                                self.successful_predictions += 1
                                self.last_successful_prediction = datetime.now()
                                self.consecutive_failures = 0
                                logger.info(f"✅ Saved: {doc_id}")
                                
                                # Show next prediction time
                                next_time = scheduler.get_next_prediction_time()
                                logger.info(f"⏭️ Next prediction: {next_time.strftime('%H:%M')}")
                            else:
                                self.failed_predictions += 1
                    else:
                        logger.info(f"⛔ Skipped (low confidence)")
                        
                except Exception as e:
                    logger.error(f"❌ Error predicting {get_timeframe_label(tf)}: {e}")
                    self.failed_predictions += 1
                    continue
            
            self.total_predictions += predictions_made
            
            # Check for stagnation
            if self.last_successful_prediction:
                time_since = (datetime.now() - self.last_successful_prediction).total_seconds()
                if time_since > 600:
                    logger.error(f"❌ No successful predictions for {time_since:.0f}s")
                    if self.alert_manager:
                        self.alert_manager.send_alert(
                            "System Stagnant",
                            f"No predictions for {time_since/60:.1f} minutes",
                            AlertSeverity.CRITICAL,
                            "stagnant"
                        )
            
            logger.info(f"\n{'='*80}")
            logger.info(f"✅ Cycle complete - {predictions_made} predictions made")
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
        """Validate predictions"""
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
                    result = self.firebase.validate_prediction(
                        pred['doc_id'],
                        current_price,
                        pred['predicted_price'],
                        pred['current_price'],
                        pred['trend']
                    )
                    
                    if result:
                        validated_count += 1
                        
                        is_win = 'WIN' in str(pred.get('validation_result', '')).upper()
                        self.recent_results.append(is_win)
                        
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
                
                if len(self.recent_results) >= 10:
                    self.recent_accuracy = sum(self.recent_results) / len(self.recent_results) * 100
                
                max_streak = STRATEGY_CONFIG['risk_management']['cooldown_after_loss_streak']
                if self.loss_streak >= max_streak:
                    cooldown_min = STRATEGY_CONFIG['risk_management']['cooldown_duration_minutes']
                    self.cooldown_until = datetime.now() + timedelta(minutes=cooldown_min)
                    logger.warning(f"⏸️ Entering cooldown for {cooldown_min} minutes after {self.loss_streak} losses")
                
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
                
                win_rate = overall_stats.get('win_rate', 0)
                if self.alert_manager:
                    self.alert_manager.alert_low_winrate(win_rate, "Overall")
                
                for tf in PREDICTION_CONFIG['active_timeframes']:
                    tf_stats = self.firebase.get_statistics(timeframe_minutes=tf, days=7)
                    if tf_stats and tf_stats.get('total_predictions', 0) >= 10:
                        tf_winrate = tf_stats.get('win_rate', 0)
                        label = get_timeframe_label(tf)
                        
                        if self.alert_manager:
                            self.alert_manager.alert_low_winrate(tf_winrate, label)
                        
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
            
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_gb = free / (1024**3)
            
            if free_gb < 1.0 and self.alert_manager:
                self.alert_manager.alert_disk_space_low(free_gb)
            
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
        logger.info("🚀 STARTING BITCOIN PREDICTOR (SMART SCHEDULING)")
        logger.info(f"{'='*80}")
        logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
        logger.info(f"Trading Mode: {os.getenv('TRADING_MODE', 'paper')}")
        logger.info(f"{'='*80}\n")
        
        self.start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        logger.info(f"💾 Starting memory: {self.start_memory:.0f}MB")
        
        self.watchdog.start()
        
        if not self.initialize():
            logger.error("❌ Initialization failed")
            return
        
        logger.info("\n📅 System will check every minute for scheduled predictions")
        logger.info("   • Validation: Every 60 seconds")
        logger.info("   • Health: Every 5 minutes")
        logger.info("   • Heartbeat: Every 30 seconds")
        logger.info("   • Statistics: Every 10 minutes\n")
        
        # Initial run
        logger.info("🎯 Running initial prediction cycle...")
        self.run_prediction_cycle()
        self.periodic_health_check()
        
        if self.heartbeat:
            self.heartbeat.send_status_change('running', 'System operational')
        
        self.is_running = True
        logger.info("\n✅ System started!\n")
        
        # Main loop - check every minute
        last_validation = time.time()
        last_health = time.time()
        last_heartbeat = time.time()
        last_stats = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Heartbeat every 30 seconds
                if current_time - last_heartbeat >= 30:
                    if self.heartbeat:
                        self.heartbeat.send_heartbeat({
                            'status': 'running',
                            'predictions': self.total_predictions
                        })
                    last_heartbeat = current_time
                
                # Validation every 60 seconds
                if current_time - last_validation >= 60:
                    self.validate_predictions()
                    last_validation = current_time
                
                # Health check every 5 minutes
                if current_time - last_health >= 300:
                    self.periodic_health_check()
                    last_health = current_time
                
                # Statistics every 10 minutes
                if current_time - last_stats >= 600:
                    self.update_statistics()
                    last_stats = current_time
                
                # Check for predictions every minute
                self.run_prediction_cycle()
                
                # Sleep until next minute boundary
                now = datetime.now()
                sleep_seconds = 60 - now.second
                time.sleep(sleep_seconds)
                
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