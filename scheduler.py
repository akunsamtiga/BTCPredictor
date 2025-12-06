"""
Enhanced Automation Scheduler with Complex Multi-Timeframe Strategy
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

logger = logging.getLogger(__name__)


class EnhancedMultiTimeframePredictionScheduler:
    """Enhanced scheduler with complex multi-timeframe strategy"""
    
    def __init__(self):
        self.predictor = BitcoinMLPredictor()
        self.firebase = None
        self.health_monitor = SystemHealthMonitor()
        self.heartbeat = None
        self.is_running = False
        
        # Timeframe management
        self.active_timeframes = PREDICTION_CONFIG['active_timeframes']
        self.priority_timeframes = PREDICTION_CONFIG['priority_timeframes']
        self.last_prediction_time = {}
        self.prediction_counters = {tf: 0 for tf in self.active_timeframes}
        
        # Data caching per category
        self.data_cache = {}
        self.data_cache_time = {}
        
        # Performance tracking per timeframe
        self.timeframe_performance = {}
        
        # Error tracking
        self.consecutive_failures = 0
        self.total_predictions = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        
        # Market state
        self.current_volatility = None
        self.current_volume_ratio = None
        self.current_trend = None
        
        # Shutdown handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 Enhanced Multi-Timeframe Prediction Scheduler initialized")
        self._log_timeframe_config()
    
    def _log_timeframe_config(self):
        """Log timeframe configuration"""
        logger.info("\n" + "="*80)
        logger.info("⏱️  TIMEFRAME CONFIGURATION")
        logger.info("="*80)
        
        categories = {
            'Ultra Short (Scalping)': PREDICTION_CONFIG['ultra_short_timeframes'],
            'Short (Day Trading)': PREDICTION_CONFIG['short_timeframes'],
            'Medium (Swing Trading)': PREDICTION_CONFIG['medium_timeframes'],
            'Long (Position Trading)': PREDICTION_CONFIG['long_timeframes']
        }
        
        for name, timeframes in categories.items():
            labels = [get_timeframe_label(tf) for tf in timeframes]
            logger.info(f"{name:25}: {', '.join(labels)}")
        
        logger.info(f"\n{'Active Timeframes':25}: {len(self.active_timeframes)} timeframes")
        active_labels = [get_timeframe_label(tf) for tf in self.active_timeframes]
        logger.info(f"{'':25}  {', '.join(active_labels)}")
        
        logger.info(f"\n{'Priority Timeframes':25}: {len(self.priority_timeframes)} timeframes")
        priority_labels = [get_timeframe_label(tf) for tf in self.priority_timeframes]
        logger.info(f"{'':25}  {', '.join(priority_labels)}")
        logger.info("="*80 + "\n")
    
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
        """Initialize or load ML models"""
        try:
            logger.info("🔧 Initializing models...")
            
            if self.heartbeat:
                self.heartbeat.send_status_change('loading_models', 'Loading ML models')
            
            if self.predictor.load_models():
                logger.info("✅ Loaded existing models")
                
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
        """Train models with data for all timeframe categories"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                logger.info(f"\n{'='*80}")
                logger.info(f"🤖 TRAINING MODELS (Attempt {attempt + 1}/{max_retries})")
                logger.info(f"{'='*80}")
                
                if self.heartbeat:
                    self.heartbeat.send_status_change('training', 'Training ML models')
                
                # Fetch training data for different categories
                df = self._fetch_training_data_comprehensive()
                
                if df is None or len(df) < 200:
                    logger.error("❌ Insufficient data for training")
                    if attempt < max_retries - 1:
                        time.sleep(30)
                        continue
                    return False
                
                df = add_technical_indicators(df)
                
                success = self.predictor.train_models(df, epochs=50, batch_size=32)
                
                if success:
                    if self.firebase and self.firebase.connected:
                        self.firebase.save_model_performance(self.predictor.metrics)
                    
                    logger.info("✅ Training completed successfully\n")
                    
                    if self.heartbeat:
                        self.heartbeat.send_status_change('trained', 'Models trained successfully')
                    
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
    
    def _fetch_training_data_comprehensive(self):
        """Fetch comprehensive training data"""
        # Try to get the most data possible
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
                    logger.info(f"✅ Successfully fetched {len(df)} data points")
                    return df
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch {interval} data: {e}")
                continue
        
        return None
    
    def get_data_for_category(self, category, force_refresh=False):
        """Get data optimized for timeframe category"""
        try:
            # Check cache
            if not force_refresh and category in self.data_cache:
                if category in self.data_cache_time:
                    age = (datetime.now() - self.data_cache_time[category]).total_seconds()
                    if age < 120:
                        logger.debug(f"📦 Using cached data for {category}")
                        return self.data_cache[category]
            
            # Get data config for category
            data_req = PREDICTION_CONFIG['data_requirements'][category]
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"📡 Fetching {category} data: {data_req['days']} days, {data_req['interval']} interval")
                    
                    df = get_bitcoin_data_realtime(
                        days=data_req['days'],
                        interval=data_req['interval']
                    )
                    
                    if df is not None and len(df) >= data_req['min_points']:
                        df = add_technical_indicators(df)
                        
                        # Cache the data
                        self.data_cache[category] = df
                        self.data_cache_time[category] = datetime.now()
                        
                        logger.info(f"✅ Data for {category}: {len(df)} points")
                        return df
                    
                except Exception as e:
                    logger.warning(f"⚠️ Data fetch attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting data for {category}: {e}")
            return None
    
    def analyze_market_state(self, df):
        """Analyze current market state"""
        try:
            if df is None or len(df) < 20:
                return
            
            # Calculate volatility
            recent_prices = df.head(20)['price']
            volatility = (recent_prices.std() / recent_prices.mean()) * 100
            self.current_volatility = volatility
            
            # Calculate volume ratio
            if 'volume' in df.columns and 'volume_ma' in df.columns:
                current_volume = df.iloc[0]['volume']
                avg_volume = df.iloc[0]['volume_ma']
                self.current_volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Determine trend
            if 'ema_9' in df.columns and 'ema_21' in df.columns:
                ema_9 = df.iloc[0]['ema_9']
                ema_21 = df.iloc[0]['ema_21']
                self.current_trend = 'bullish' if ema_9 > ema_21 else 'bearish'
            
            logger.info(f"📊 Market State: Volatility={volatility:.2f}%, "
                       f"Volume Ratio={self.current_volume_ratio:.2f}, "
                       f"Trend={self.current_trend}")
            
        except Exception as e:
            logger.error(f"❌ Error analyzing market state: {e}")
    
    def get_active_timeframes_for_market(self):
        """Get active timeframes based on current market conditions"""
        if not PREDICTION_CONFIG['enable_smart_scheduling']:
            return self.active_timeframes
        
        active = []
        
        # Always include priority timeframes
        active.extend(self.priority_timeframes)
        
        # Add timeframes based on volatility
        if self.current_volatility is not None:
            if self.current_volatility > 3:  # High volatility
                volatility_tf = STRATEGY_CONFIG['volatility_adjustments']['high']['prefer_timeframes']
                active.extend(volatility_tf)
            elif self.current_volatility > 1:  # Medium volatility
                volatility_tf = STRATEGY_CONFIG['volatility_adjustments']['medium']['prefer_timeframes']
                active.extend(volatility_tf)
            else:  # Low volatility
                volatility_tf = STRATEGY_CONFIG['volatility_adjustments']['low']['prefer_timeframes']
                active.extend(volatility_tf)
        
        # Add timeframes based on time of day
        now = get_local_now()
        hour = now.hour
        day_of_week = now.weekday()
        
        if day_of_week >= 5:  # Weekend
            session_tf = STRATEGY_CONFIG['time_based_strategy']['weekend']['active_timeframes']
        elif 0 <= hour < 9:  # Asian session
            session_tf = STRATEGY_CONFIG['time_based_strategy']['asian_session']['active_timeframes']
        elif 14 <= hour < 23:  # European session
            session_tf = STRATEGY_CONFIG['time_based_strategy']['european_session']['active_timeframes']
        else:  # American session
            session_tf = STRATEGY_CONFIG['time_based_strategy']['american_session']['active_timeframes']
        
        active.extend(session_tf)
        
        # Remove duplicates and sort
        active = sorted(list(set(active)))
        
        return active
    
    def should_predict_timeframe(self, timeframe):
        """Check if should predict for this timeframe"""
        # Check if in active timeframes
        active_tf = self.get_active_timeframes_for_market()
        if timeframe not in active_tf:
            return False
        
        # Check last prediction time
        if timeframe in self.last_prediction_time:
            time_since = (datetime.now() - self.last_prediction_time[timeframe]).total_seconds()
            min_interval = min(timeframe * 60 * 0.1, 300)  # 10% of timeframe or 5 min max
            
            if time_since < min_interval:
                return False
        
        # Check volatility threshold
        if PREDICTION_CONFIG['skip_low_volatility'] and self.current_volatility is not None:
            if self.current_volatility < PREDICTION_CONFIG['min_volatility_threshold']:
                if timeframe not in self.priority_timeframes:
                    return False
        
        return True
    
    def run_predictions_smart(self):
        """Run predictions with smart timeframe selection"""
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"🔮 SMART PREDICTIONS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*80}")
            
            # Send heartbeat
            if self.heartbeat:
                self.heartbeat.send_heartbeat({
                    'last_activity': 'running_predictions',
                    'total_predictions': self.total_predictions,
                    'successful_predictions': self.successful_predictions
                })
            
            # Check system health
            if HEALTH_CONFIG['enable_watchdog']:
                health = self.health_monitor.get_full_health_report()
                if health['overall_status'] == 'CRITICAL':
                    logger.error("❌ System health critical, skipping predictions")
                    return
            
            # Get data and analyze market
            categories_to_fetch = set()
            for tf in self.active_timeframes:
                category = get_timeframe_category(tf)
                categories_to_fetch.add(category)
            
            # Fetch data for each category
            data_by_category = {}
            for category in categories_to_fetch:
                df = self.get_data_for_category(category)
                if df is not None:
                    data_by_category[category] = df
                    # Analyze market with short-term data
                    if category == 'short':
                        self.analyze_market_state(df)
            
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
                logger.info(f"💰 Current BTC Price: ${current_price:,.2f}")
            
            # Get active timeframes for current market
            active_tf = self.get_active_timeframes_for_market()
            logger.info(f"🎯 Active Timeframes: {len(active_tf)} timeframes")
            
            predictions_made = 0
            
            # Group predictions by category for efficiency
            predictions_by_category = {}
            for tf in active_tf:
                if self.should_predict_timeframe(tf):
                    category = get_timeframe_category(tf)
                    if category not in predictions_by_category:
                        predictions_by_category[category] = []
                    predictions_by_category[category].append(tf)
            
            # Run predictions per category
            for category, timeframes in predictions_by_category.items():
                if category not in data_by_category:
                    continue
                
                df = data_by_category[category]
                logger.info(f"\n📊 Processing {category.upper()} timeframes: {len(timeframes)} predictions")
                
                for tf in timeframes:
                    try:
                        logger.info(f"\n⏱️  Predicting for {get_timeframe_label(tf)}...")
                        
                        prediction = self.predictor.predict(df, tf)
                        
                        if prediction:
                            # Apply confidence adjustment based on market state
                            prediction = self._adjust_prediction_confidence(prediction, tf)
                            
                            self._display_prediction_summary(prediction)
                            
                            if self.firebase and self.firebase.connected:
                                doc_id = self.firebase.save_prediction(prediction)
                                
                                if doc_id:
                                    self.last_prediction_time[tf] = datetime.now()
                                    self.prediction_counters[tf] = self.prediction_counters.get(tf, 0) + 1
                                    predictions_made += 1
                                    self.successful_predictions += 1
                                    logger.info(f"✅ Prediction saved: {doc_id}")
                                else:
                                    self.failed_predictions += 1
                        else:
                            logger.warning(f"⚠️ Prediction failed for {get_timeframe_label(tf)}")
                            self.failed_predictions += 1
                            
                    except Exception as e:
                        logger.error(f"❌ Error predicting {get_timeframe_label(tf)}: {e}")
                        self.failed_predictions += 1
                        continue
            
            self.total_predictions += predictions_made
            
            if predictions_made > 0:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
            
            # Log summary
            logger.info(f"\n{'='*80}")
            logger.info(f"✅ Prediction cycle completed - {predictions_made} predictions made")
            logger.info(f"📊 Success rate: {self.successful_predictions}/{self.total_predictions}")
            logger.info(f"📈 Predictions per timeframe:")
            for tf in sorted(self.prediction_counters.keys()):
                count = self.prediction_counters[tf]
                if count > 0:
                    logger.info(f"   {get_timeframe_label(tf):6}: {count} predictions")
            logger.info(f"{'='*80}\n")
            
            # Memory optimization
            if VPS_CONFIG['enable_memory_optimization']:
                if self.total_predictions % 10 == 0:
                    gc.collect()
            
        except Exception as e:
            logger.error(f"❌ Critical error in prediction cycle: {e}")
            traceback.print_exc()
            self.consecutive_failures += 1
    
    def _adjust_prediction_confidence(self, prediction, timeframe):
        """Adjust prediction confidence based on market conditions"""
        try:
            category = get_timeframe_category(timeframe)
            original_confidence = prediction['confidence']
            
            # Volatility adjustment
            if self.current_volatility is not None:
                if self.current_volatility > 3:
                    multiplier = STRATEGY_CONFIG['volatility_adjustments']['high']['confidence_multiplier']
                elif self.current_volatility > 1:
                    multiplier = STRATEGY_CONFIG['volatility_adjustments']['medium']['confidence_multiplier']
                else:
                    multiplier = STRATEGY_CONFIG['volatility_adjustments']['low']['confidence_multiplier']
                
                prediction['confidence'] = min(95, prediction['confidence'] * multiplier)
            
            # Multi-timeframe confirmation
            if STRATEGY_CONFIG['enable_mtf_analysis']:
                correlation_tf = STRATEGY_CONFIG['correlation_timeframes'].get(timeframe, [])
                if correlation_tf:
                    # Check if correlated timeframes agree
                    # (This would require storing recent predictions - simplified here)
                    pass
            
            logger.debug(f"Confidence adjusted: {original_confidence:.1f}% → {prediction['confidence']:.1f}%")
            
        except Exception as e:
            logger.error(f"Error adjusting confidence: {e}")
        
        return prediction
    
    def _display_prediction_summary(self, prediction):
        """Display prediction summary"""
        arrow = "🟢 ↗️" if prediction['price_change'] > 0 else "🔴 ↘️"
        tf_label = get_timeframe_label(prediction['timeframe_minutes'])
        logger.info(f"   {arrow} ${prediction['predicted_price']:,.2f} "
                   f"({prediction['price_change_pct']:+.2f}%) - "
                   f"Confidence: {prediction['confidence']:.1f}%")
    
    def validate_predictions(self):
        """Validate predictions"""
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
        """Update statistics"""
        try:
            if not self.firebase or not self.firebase.connected:
                return
            
            logger.info("📊 Updating statistics...")
            
            overall_stats = self.firebase.get_statistics(days=7)
            
            if overall_stats and overall_stats.get('total_predictions', 0) > 0:
                logger.info(f"   Overall: {overall_stats['win_rate']:.1f}% win rate")
                self.firebase.save_statistics(overall_stats)
            
            # Update stats for each active timeframe
            for timeframe in self.active_timeframes:
                stats = self.firebase.get_statistics(timeframe_minutes=timeframe, days=7)
                
                if stats and stats.get('total_predictions', 0) > 0:
                    logger.info(f"   {get_timeframe_label(timeframe):6}: {stats['win_rate']:.1f}% win rate")
                    self.firebase.save_statistics(stats)
            
            logger.info("✅ Statistics updated\n")
            
        except Exception as e:
            logger.error(f"❌ Error updating statistics: {e}")
    
    def periodic_health_check(self):
        """Periodic system health check"""
        try:
            logger.info("\n🏥 SYSTEM HEALTH CHECK")
            
            report = monitor_health(self.firebase if self.firebase and self.firebase.connected else None)
            
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
        logger.info("🚀 STARTING MULTI-TIMEFRAME BITCOIN PREDICTOR")
        logger.info(f"{'='*80}")
        
        if not self._initialize_firebase():
            logger.error("❌ Cannot start without Firebase connection")
            return
        
        if not self.initialize_models():
            logger.error("❌ Cannot start without trained models")
            return
        
        logger.info("\n📅 Setting up schedule:")
        logger.info(f"   • Smart predictions: Every 5 minutes")
        logger.info(f"   • Validation: Every 60 seconds")
        logger.info(f"   • Health check: Every 5 minutes")
        logger.info(f"   • Heartbeat: Every 30 seconds")
        logger.info(f"   • Model retraining: Daily at 02:00")
        logger.info(f"   • Cleanup: Daily at 03:00")
        
        # Setup schedules
        schedule.every(300).seconds.do(self.run_predictions_smart)
        schedule.every(60).seconds.do(self.validate_predictions)
        schedule.every(300).seconds.do(self.periodic_health_check)
        
        if self.heartbeat:
            schedule.every(30).seconds.do(lambda: self.heartbeat.send_heartbeat({
                'last_activity': 'heartbeat',
                'predictions_count': self.total_predictions,
                'active_timeframes': len(self.get_active_timeframes_for_market())
            }))
        
        schedule.every().day.at("02:00").do(self.train_models)
        schedule.every().day.at("03:00").do(self.periodic_cleanup)
        
        if VPS_CONFIG['garbage_collection_interval']:
            schedule.every(VPS_CONFIG['garbage_collection_interval']).seconds.do(gc.collect)
        
        logger.info("\n🎯 Running initial predictions...")
        self.run_predictions_smart()
        self.periodic_health_check()
        
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
        
        if self.heartbeat:
            self.heartbeat.send_shutdown_signal()
        
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
        scheduler = EnhancedMultiTimeframePredictionScheduler()
        scheduler.start()
        
    except KeyboardInterrupt:
        logger.info("\n\n❌ Program interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()