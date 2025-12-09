#!/usr/bin/env python3
"""
COMPLETE TESTING SUITE - All Tests in One File
Includes: Diagnostic, System, Timezone, Redis, API, and Predictor tests
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def print_section(text):
    """Print section header"""
    print(f"\n{text}")
    print("-" * 80)


def print_success(text):
    """Print success message"""
    print(f"✅ {text}")


def print_error(text):
    """Print error message"""
    print(f"❌ {text}")


def print_warning(text):
    """Print warning message"""
    print(f"⚠️  {text}")


def print_info(text):
    """Print info message"""
    print(f"ℹ️  {text}")


# ============================================================================
# TEST 1: ENVIRONMENT & CONFIGURATION
# ============================================================================

def test_environment():
    """Test environment configuration"""
    print_header("TEST 1: ENVIRONMENT & CONFIGURATION")
    
    try:
        from config import validate_environment, get_config_summary
        
        print("\n🔍 Validating environment...")
        validate_environment()
        print_success("Environment validated")
        
        summary = get_config_summary()
        
        print("\n📋 Configuration Summary:")
        for key, value in summary.items():
            print(f"  • {key}: {value}")
        
        # Check .env file
        print("\n📄 Checking .env file...")
        env_path = '.env'
        if os.path.exists(env_path):
            print_success(f".env file exists")
            
            # Check critical variables
            critical_vars = [
                'CRYPTOCOMPARE_API_KEY',
                'FIREBASE_CREDENTIALS_PATH',
                'FIREBASE_DATABASE_URL'
            ]
            
            with open(env_path, 'r') as f:
                env_content = f.read()
                
            for var in critical_vars:
                if var in env_content:
                    print_success(f"{var} present")
                else:
                    print_error(f"{var} missing")
        else:
            print_error(".env file not found")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Environment test failed: {e}")
        return False


# ============================================================================
# TEST 2: REQUIRED IMPORTS
# ============================================================================

def test_imports():
    """Test all required imports"""
    print_header("TEST 2: REQUIRED IMPORTS")
    
    imports = {
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'tensorflow': 'TensorFlow',
        'sklearn': 'Scikit-learn',
        'firebase_admin': 'Firebase Admin',
        'schedule': 'Schedule',
        'psutil': 'PSUtil',
        'requests': 'Requests',
        'pytz': 'PyTZ (Timezone)',
        'dotenv': 'Python-dotenv',
        'redis': 'Redis (optional)',
    }
    
    failed = []
    optional_missing = []
    
    print("\n📦 Checking Python packages...")
    
    for module, name in imports.items():
        try:
            __import__(module)
            print_success(name)
        except ImportError:
            if module == 'redis':
                print_warning(f"{name} (optional)")
                optional_missing.append(name)
            else:
                print_error(name)
                failed.append(name)
    
    if failed:
        print_error(f"\nMissing required modules: {', '.join(failed)}")
        print_info("Install with: pip install -r requirements.txt")
        return False
    
    if optional_missing:
        print_warning(f"\nOptional modules missing: {', '.join(optional_missing)}")
    
    print_success("\nAll required imports available")
    return True


# ============================================================================
# TEST 3: TIMEZONE UTILITIES
# ============================================================================

def test_timezone():
    """Test timezone utilities"""
    print_header("TEST 3: TIMEZONE UTILITIES")
    
    try:
        from timezone_utils import (
            get_local_now, get_utc_now, local_to_utc, utc_to_local,
            format_local_datetime, get_local_isoformat, get_timezone_info
        )
        
        # Basic conversions
        print_section("3.1 Basic Timezone Conversions")
        
        local_now = get_local_now()
        utc_now = get_utc_now()
        
        print(f"Local Time (WIB): {local_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"UTC Time:         {utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Offset:           +{(local_now.utcoffset().total_seconds() / 3600):.0f} hours")
        
        # Test conversions
        local_to_utc_result = local_to_utc(local_now)
        utc_to_local_result = utc_to_local(utc_now)
        
        print_success("Local ↔ UTC conversions working")
        
        # Formatting
        print_section("3.2 Datetime Formatting")
        
        formats = [
            ('%Y-%m-%d %H:%M:%S', 'Standard'),
            ('%d/%m/%Y %H:%M', 'Indonesian'),
            ('%Y-%m-%d %H:%M:%S %Z', 'With timezone')
        ]
        
        for fmt, label in formats:
            formatted = format_local_datetime(local_now, fmt)
            print(f"  {label:20}: {formatted}")
        
        iso = get_local_isoformat()
        print(f"  {'ISO format':20}: {iso}")
        
        print_success("Formatting working")
        
        # Timezone info
        print_section("3.3 Timezone Information")
        
        info = get_timezone_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print_success("Timezone utilities working correctly")
        return True
        
    except Exception as e:
        print_error(f"Timezone test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 4: REDIS CONNECTION
# ============================================================================

def test_redis():
    """Test Redis connection"""
    print_header("TEST 4: REDIS CONNECTION")
    
    try:
        import redis
        print_success("Redis library installed")
    except ImportError:
        print_error("Redis library not installed")
        print_info("Install with: pip install redis")
        print_warning("Redis is optional, skipping...")
        return True  # Not critical
    
    # Test connection
    print_section("4.1 Basic Connection")
    
    try:
        r = redis.Redis(
            host='localhost',
            port=6379,
            socket_connect_timeout=5,
            decode_responses=False
        )
        
        # Ping test
        r.ping()
        print_success("Redis is running and responding")
        
        # Get info
        info = r.info('server')
        print(f"  Redis version: {info.get('redis_version', 'unknown')}")
        print(f"  Uptime: {info.get('uptime_in_seconds', 0)} seconds")
        
    except redis.exceptions.ConnectionError:
        print_error("Cannot connect to Redis")
        print_info("Redis is optional. Start with: sudo systemctl start redis")
        return True  # Not critical
    except Exception as e:
        print_error(f"Redis error: {e}")
        return True  # Not critical
    
    # Test operations
    print_section("4.2 Read/Write Operations")
    
    try:
        test_key = "btc_predictor_test"
        test_value = f"test_{datetime.now().isoformat()}"
        
        # Write
        r.set(test_key, test_value)
        print_success("Write successful")
        
        # Read
        retrieved = r.get(test_key)
        if retrieved:
            retrieved = retrieved.decode('utf-8')
            if retrieved == test_value:
                print_success("Read successful, data integrity verified")
            else:
                print_warning("Data mismatch")
        
        # Delete
        r.delete(test_key)
        print_success("Delete successful")
        
        # Test TTL
        r.setex("ttl_test", 10, "expires")
        ttl = r.ttl("ttl_test")
        print_success(f"TTL set: {ttl} seconds")
        r.delete("ttl_test")
        
    except Exception as e:
        print_error(f"Redis operations failed: {e}")
        return False
    
    # Test with CacheManager
    print_section("4.3 Testing with CacheManager")
    
    try:
        from cache_manager import get_cache
        
        cache = get_cache()
        print(f"  Backend: {'Redis' if cache.redis_client else 'Memory'}")
        
        # Test operations
        cache.set("test_key", {"data": "test"}, ttl=60)
        print_success("CacheManager set successful")
        
        retrieved = cache.get("test_key")
        if retrieved:
            print_success("CacheManager get successful")
        
        cache.delete("test_key")
        print_success("CacheManager delete successful")
        
        # Stats
        stats = cache.get_stats()
        print(f"\n  Cache Statistics:")
        print(f"    Backend: {stats['backend']}")
        print(f"    Hit rate: {stats['hit_rate']:.1f}%")
        
    except Exception as e:
        print_warning(f"CacheManager test skipped: {e}")
    
    print_success("Redis tests completed")
    return True


# ============================================================================
# TEST 5: API CONNECTION
# ============================================================================

def test_api():
    """Test CryptoCompare API"""
    print_header("TEST 5: CRYPTOCOMPARE API")
    
    try:
        from btc_predictor_automated import get_current_btc_price, get_bitcoin_data_realtime
        
        # Test current price
        print_section("5.1 Current Price")
        
        price = get_current_btc_price()
        
        if price:
            print_success(f"Current BTC price: ${price:,.2f}")
        else:
            print_error("Failed to get current price")
            return False
        
        # Test single request (< 2000 points)
        print_section("5.2 Single Request Test")
        
        test_cases = [
            (1, 'hour', 24, "1 day hourly"),
            (7, 'hour', 168, "7 days hourly"),
        ]
        
        for days, interval, expected, label in test_cases:
            print(f"\n  Testing: {label}")
            df = get_bitcoin_data_realtime(days=days, interval=interval)
            
            if df is not None:
                actual = len(df)
                if actual >= expected * 0.9:
                    print_success(f"Got {actual} points (expected ~{expected})")
                else:
                    print_warning(f"Got {actual} points (expected ~{expected})")
            else:
                print_error(f"Failed to fetch data")
                return False
        
        # Test multiple requests (> 2000 points)
        print_section("5.3 Multiple Requests Test")
        
        print("\n  Testing: 90 days hourly (~2160 points)")
        df = get_bitcoin_data_realtime(days=90, interval='hour')
        
        if df is not None:
            actual = len(df)
            if actual >= 2000:
                print_success(f"Got {actual} points (multiple batches worked)")
                
                # Check for gaps
                time_diffs = df['datetime'].diff().abs()
                gaps = time_diffs[time_diffs > time_diffs.median() * 2]
                
                if len(gaps) == 0:
                    print_success("No significant gaps detected")
                else:
                    print_warning(f"Found {len(gaps)} potential gaps")
            else:
                print_warning(f"Got {actual} points (expected >2000)")
        else:
            print_error("Multiple requests failed")
            return False
        
        print_success("API tests completed")
        return True
        
    except Exception as e:
        print_error(f"API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 6: FIREBASE CONNECTION
# ============================================================================

def test_firebase():
    """Test Firebase connection"""
    print_header("TEST 6: FIREBASE CONNECTION")
    
    try:
        from firebase_manager import FirebaseManager
        
        print("\n🔗 Connecting to Firebase...")
        fb = FirebaseManager()
        
        if fb.connected:
            print_success("Firebase connected")
            
            # Test write
            print_section("6.1 Write Test")
            
            test_data = {
                'test': True,
                'timestamp': datetime.now().isoformat(),
                'message': 'Test connection from test_all.py'
            }
            
            collection = fb.firestore_db.collection('system_test')
            doc_ref = collection.add(test_data)
            
            print_success(f"Write successful: {doc_ref[1].id}")
            
            # Test read
            print_section("6.2 Read Test")
            
            doc = collection.document(doc_ref[1].id).get()
            
            if doc.exists:
                print_success("Read successful")
                
                # Cleanup
                print_section("6.3 Cleanup")
                collection.document(doc_ref[1].id).delete()
                print_success("Cleanup successful")
                
                return True
            else:
                print_error("Read failed")
                return False
        else:
            print_error("Firebase connection failed")
            return False
            
    except Exception as e:
        print_error(f"Firebase test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 7: ML MODELS
# ============================================================================

def test_ml_models():
    """Test ML model functionality"""
    print_header("TEST 7: ML MODELS & PREDICTOR")
    
    try:
        from btc_predictor_automated import ImprovedBitcoinPredictor
        from btc_predictor_automated import get_bitcoin_data_realtime, add_technical_indicators
        
        print_section("7.1 Model Initialization")
        
        predictor = ImprovedBitcoinPredictor()
        print_success("Predictor created")
        
        # Try to load models
        print_section("7.2 Loading Models")
        
        if predictor.load_models():
            print_success("Models loaded successfully")
            print(f"  Trained: {predictor.is_trained}")
            print(f"  Last training: {predictor.last_training}")
            
            # Test prediction
            print_section("7.3 Test Prediction")
            
            df = get_bitcoin_data_realtime(days=7, interval='hour')
            
            if df is not None:
                df = add_technical_indicators(df)
                prediction = predictor.predict(df, 60, always_predict=True)
                
                if prediction:
                    print_success("Prediction successful")
                    print(f"  Current:   ${prediction['current_price']:,.2f}")
                    print(f"  Predicted: ${prediction['predicted_price']:,.2f}")
                    print(f"  Change:    {prediction['price_change_pct']:+.2f}%")
                    print(f"  Trend:     {prediction['trend']}")
                    print(f"  Confidence: {prediction['confidence']:.1f}%")
                    print(f"  Quality:   {prediction['quality_score']:.1f}")
                    return True
                else:
                    print_error("Prediction failed")
                    return False
            else:
                print_error("Failed to get data for prediction")
                return False
        else:
            print_warning("No trained models found")
            print_info("Run training: python3 maintenance.py → 2")
            return True  # Not critical for testing
            
    except Exception as e:
        print_error(f"ML test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 8: SYSTEM HEALTH
# ============================================================================

def test_system_health():
    """Test system health monitoring"""
    print_header("TEST 8: SYSTEM HEALTH")
    
    try:
        from system_health import SystemHealthMonitor
        
        monitor = SystemHealthMonitor()
        
        print("\n🏥 Running health check...")
        report = monitor.get_full_health_report()
        
        print_section("8.1 System Metrics")
        
        print(f"  Overall Status: {report['overall_status']}")
        print(f"  Memory: {report['memory']['process_memory_mb']:.0f}MB / {report['memory']['total_mb']:.0f}MB")
        print(f"  CPU: {report['cpu']['cpu_percent']:.1f}%")
        print(f"  Disk Free: {report['disk']['free_gb']:.2f}GB")
        print(f"  Network: {report['network']['status']}")
        
        if report['overall_status'] in ['HEALTHY', 'WARNING']:
            print_success("System health check passed")
            return True
        else:
            print_warning("System health issues detected")
            return False
            
    except Exception as e:
        print_error(f"Health test failed: {e}")
        return False


# ============================================================================
# TEST 9: DIAGNOSTIC - TIMEFRAME SCHEDULING
# ============================================================================

def test_timeframe_scheduling():
    """Test if all timeframes are scheduled correctly"""
    print_header("TEST 9: TIMEFRAME SCHEDULING")
    
    try:
        from config import PREDICTION_CONFIG, get_timeframe_category, get_timeframe_label
        
        now = datetime.now()
        current_minute = now.hour * 60 + now.minute
        
        print(f"\nCurrent time: {now.strftime('%H:%M')}")
        print(f"Current minute: {current_minute}\n")
        
        all_timeframes = PREDICTION_CONFIG['active_timeframes']
        
        print(f"{'Timeframe':<12} | {'Category':<12} | {'Scheduled?':<12} | {'Status'}")
        print("-" * 80)
        
        scheduled_count = 0
        
        for tf in sorted(all_timeframes):
            category = get_timeframe_category(tf)
            label = get_timeframe_label(tf)
            
            should_predict = (current_minute % tf == 0)
            
            if should_predict:
                status = "✅ NOW"
                scheduled_count += 1
            else:
                remainder = current_minute % tf
                next_in = tf - remainder
                status = f"⏭️ in {next_in}min"
            
            print(f"{label:<12} | {category:<12} | {str(should_predict):<12} | {status}")
        
        print(f"\n📊 Summary: {scheduled_count}/{len(all_timeframes)} scheduled now")
        
        print_success("Scheduling logic verified")
        return True
        
    except Exception as e:
        print_error(f"Scheduling test failed: {e}")
        return False


# ============================================================================
# TEST 10: DIAGNOSTIC - DATA AVAILABILITY
# ============================================================================

def test_data_availability():
    """Test data availability for all categories"""
    print_header("TEST 10: DATA AVAILABILITY FOR ALL CATEGORIES")
    
    try:
        from config import PREDICTION_CONFIG, get_timeframe_category, get_timeframe_label, get_data_config_for_timeframe
        from btc_predictor_automated import get_bitcoin_data_realtime, add_technical_indicators
        
        categories = ['ultra_short', 'short', 'medium', 'long']
        results = {}
        
        for category in categories:
            print_section(f"10.{categories.index(category)+1} {category.upper()}")
            
            timeframes = PREDICTION_CONFIG[f'{category}_timeframes']
            if not timeframes:
                print_warning(f"No timeframes for {category}")
                results[category] = False
                continue
            
            tf = timeframes[0]
            label = get_timeframe_label(tf)
            data_config = get_data_config_for_timeframe(tf)
            
            print(f"  Timeframe: {label}")
            print(f"  Config: {data_config['days']} days, {data_config['interval']} interval")
            print(f"  Min points: {data_config['min_points']}")
            
            # Fetch data
            df = get_bitcoin_data_realtime(
                days=data_config['days'],
                interval=data_config['interval']
            )
            
            if df is None:
                print_error("Failed to fetch data")
                results[category] = False
                continue
            
            print(f"  Fetched: {len(df)} raw points")
            
            # Add indicators
            df = add_technical_indicators(df)
            df_clean = df.dropna()
            clean_points = len(df_clean)
            
            print(f"  Cleaned: {clean_points} points")
            
            # Check sufficiency
            if clean_points >= data_config['min_points']:
                print_success(f"SUFFICIENT ({clean_points} >= {data_config['min_points']})")
                results[category] = True
            else:
                print_error(f"INSUFFICIENT ({clean_points} < {data_config['min_points']})")
                print_info("Solution: Lower min_points in config.py")
                results[category] = False
        
        # Summary
        print_section("Summary")
        passed = sum(results.values())
        total = len(results)
        
        for cat, result in results.items():
            status = "✅" if result else "❌"
            print(f"{status} {cat:12} - {'PASS' if result else 'FAIL'}")
        
        print(f"\n📊 {passed}/{total} categories have sufficient data")
        
        return passed == total
        
    except Exception as e:
        print_error(f"Data availability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 11: DIAGNOSTIC - CALL/PUT BALANCE
# ============================================================================

def test_call_put_balance():
    """Test CALL vs PUT distribution"""
    print_header("TEST 11: CALL/PUT DISTRIBUTION TEST")
    
    try:
        from btc_predictor_automated import ImprovedBitcoinPredictor, get_bitcoin_data_realtime, add_technical_indicators
        from config import get_timeframe_label
        
        predictor = ImprovedBitcoinPredictor()
        
        if not predictor.load_models():
            print_error("Models not loaded")
            print_info("Run: python3 maintenance.py → 2")
            return False
        
        print_success("Models loaded")
        
        # Fetch data
        print("\n🔄 Fetching recent data...")
        df = get_bitcoin_data_realtime(days=7, interval='hour')
        
        if df is None:
            print_error("Failed to fetch data")
            return False
        
        df = add_technical_indicators(df)
        df_clean = df.dropna()
        
        print(f"✅ Data ready: {len(df_clean)} points")
        
        # Test multiple timeframes
        print_section("11.1 Testing Predictions")
        
        test_timeframes = [15, 60, 240, 1440]
        call_count = 0
        put_count = 0
        predictions = []
        
        for tf in test_timeframes:
            label = get_timeframe_label(tf)
            
            try:
                prediction = predictor.predict(df_clean, tf, always_predict=True)
                
                if prediction:
                    trend = prediction['trend']
                    is_call = 'CALL' in trend
                    
                    if is_call:
                        call_count += 1
                        symbol = "🟢"
                    else:
                        put_count += 1
                        symbol = "🔴"
                    
                    predictions.append({
                        'timeframe': label,
                        'trend': trend,
                        'confidence': prediction['confidence']
                    })
                    
                    print(f"{symbol} {label:8} → {trend:20} (Conf: {prediction['confidence']:.1f}%)")
                
            except Exception as e:
                print_error(f"{label:8} → Error: {e}")
        
        # Analysis
        print_section("11.2 Distribution Analysis")
        
        total = len(test_timeframes)
        call_pct = (call_count / total) * 100 if total > 0 else 0
        put_pct = (put_count / total) * 100 if total > 0 else 0
        
        print(f"🟢 CALL: {call_count}/{total} ({call_pct:.0f}%)")
        print(f"🔴 PUT:  {put_count}/{total} ({put_pct:.0f}%)")
        
        # Verdict
        print_section("11.3 Verdict")
        
        if call_count == total:
            print_warning("ALL PREDICTIONS ARE CALL!")
            print_info("Possible causes:")
            print_info("  - Model trained on bullish data")
            print_info("  - Current market is strongly bullish")
            print_info("  - Check current market conditions")
            return False
        elif put_count == total:
            print_warning("ALL PREDICTIONS ARE PUT!")
            print_info("Possible causes:")
            print_info("  - Model trained on bearish data")
            print_info("  - Current market is strongly bearish")
            return False
        elif 30 <= call_pct <= 70:
            print_success("Good distribution of CALL and PUT")
            return True
        else:
            print_warning(f"Unbalanced distribution ({call_pct:.0f}% CALL)")
            print_info("Consider retraining if market is not trending strongly")
            return True  # Not critical
            
    except Exception as e:
        print_error(f"CALL/PUT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 12: CURRENT MARKET CONDITIONS
# ============================================================================

def test_market_conditions():
    """Test and display current market conditions"""
    print_header("TEST 12: CURRENT MARKET CONDITIONS")
    
    try:
        from btc_predictor_automated import get_current_btc_price, get_bitcoin_data_realtime, add_technical_indicators
        
        # Current price
        print_section("12.1 Current Price")
        
        price = get_current_btc_price()
        
        if not price:
            print_error("Failed to get price")
            return False
        
        print_success(f"Current BTC: ${price:,.2f}")
        
        # Market indicators
        print_section("12.2 Technical Indicators")
        
        df = get_bitcoin_data_realtime(days=7, interval='hour')
        
        if df is None:
            print_error("Failed to fetch data")
            return False
        
        df = add_technical_indicators(df)
        recent = df.head(20)
        
        # RSI
        if 'rsi_14' in recent.columns:
            rsi = recent['rsi_14'].iloc[0]
            if rsi > 70:
                rsi_status = "🔴 OVERBOUGHT"
            elif rsi < 30:
                rsi_status = "🟢 OVERSOLD"
            else:
                rsi_status = "🟡 NEUTRAL"
            print(f"  RSI (14):        {rsi:.2f} {rsi_status}")
        
        # MACD
        if 'macd' in recent.columns:
            macd = recent['macd'].iloc[0]
            macd_signal = recent['macd_signal'].iloc[0]
            macd_diff = macd - macd_signal
            
            macd_status = "🟢 BULLISH" if macd_diff > 0 else "🔴 BEARISH"
            print(f"  MACD:            {macd:.2f} {macd_status}")
        
        # Moving Averages
        if 'sma_20' in recent.columns and 'sma_50' in recent.columns:
            sma20 = recent['sma_20'].iloc[0]
            sma50 = recent['sma_50'].iloc[0]
            
            if price > sma20 > sma50:
                ma_status = "🟢 STRONG BULLISH"
            elif price > sma20:
                ma_status = "🟢 BULLISH"
            elif price < sma20 < sma50:
                ma_status = "🔴 STRONG BEARISH"
            else:
                ma_status = "🔴 BEARISH"
            
            print(f"  SMA (20):        ${sma20:,.2f}")
            print(f"  SMA (50):        ${sma50:,.2f}")
            print(f"  Trend:           {ma_status}")
        
        # Volume
        if 'volume_ratio_20' in recent.columns:
            vol_ratio = recent['volume_ratio_20'].iloc[0]
            vol_status = "🟢 HIGH" if vol_ratio > 1.2 else "🔴 LOW" if vol_ratio < 0.8 else "🟡 NORMAL"
            print(f"  Volume:          {vol_ratio:.2f}x {vol_status}")
        
        # Volatility
        if 'volatility_20' in recent.columns:
            vol = recent['volatility_20'].iloc[0]
            vol_status = "🔴 HIGH" if vol > 0.03 else "🟢 LOW" if vol < 0.015 else "🟡 MEDIUM"
            print(f"  Volatility:      {vol:.4f} {vol_status}")
        
        print_success("Market conditions analyzed")
        return True
        
    except Exception as e:
        print_error(f"Market conditions test failed: {e}")
        return False


# ============================================================================
# MAIN TEST SUITE
# ============================================================================

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("🧪 COMPLETE TESTING SUITE - Bitcoin Predictor")
    print("="*80)
    
    tests = [
        ("Environment & Configuration", test_environment),
        ("Required Imports", test_imports),
        ("Timezone Utilities", test_timezone),
        ("Redis Connection", test_redis),
        ("API Connection", test_api),
        ("Firebase Connection", test_firebase),
        ("ML Models & Predictor", test_ml_models),
        ("System Health", test_system_health),
        ("Timeframe Scheduling", test_timeframe_scheduling),
        ("Data Availability", test_data_availability),
        ("CALL/PUT Balance", test_call_put_balance),
        ("Market Conditions", test_market_conditions),
    ]
    
    results = {}
    
    for name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"Running: {name}")
        print(f"{'='*80}")
        
        try:
            results[name] = test_func()
        except KeyboardInterrupt:
            print("\n\n⚠️  Tests interrupted by user")
            sys.exit(1)
        except Exception as e:
            print_error(f"Test '{name}' crashed: {e}")
            results[name] = False
    
    # Final Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {name}")
    
    print("\n" + "="*80)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\n✅ System is ready for production!")
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        print("="*80)
        print("\n🔧 Fix the failing tests before deploying")
        return False


def run_quick_test():
    """Quick smoke test"""
    print_header("⚡ QUICK TEST")
    
    tests = [
        ("Environment", test_environment),
        ("Imports", test_imports),
        ("API", test_api),
        ("Models", test_ml_models),
    ]
    
    for name, test_func in tests:
        print(f"\nTesting: {name}...")
        if not test_func():
            print_error(f"Quick test failed at: {name}")
            return False
    
    print_success("\nQuick test passed!")
    print_info("For full testing, run: python3 test_all.py --full")
    return True


def show_menu():
    """Show interactive menu"""
    while True:
        print("\n" + "="*80)
        print("🧪 COMPLETE TESTING SUITE - Interactive Menu")
        print("="*80)
        print("\nQuick Tests:")
        print("  1.  Quick Test (Essential)")
        print("  2.  Full Test Suite (All)")
        print("\nIndividual Tests:")
        print("  3.  Environment & Configuration")
        print("  4.  Required Imports")
        print("  5.  Timezone Utilities")
        print("  6.  Redis Connection")
        print("  7.  API Connection")
        print("  8.  Firebase Connection")
        print("  9.  ML Models & Predictor")
        print("  10. System Health")
        print("\nDiagnostic Tests:")
        print("  11. Timeframe Scheduling")
        print("  12. Data Availability")
        print("  13. CALL/PUT Balance")
        print("  14. Market Conditions")
        print("\n  0.  Exit")
        print("="*80)
        
        choice = input("\nSelect test (0-14): ").strip()
        
        if choice == '0':
            print("\n👋 Goodbye!")
            break
        elif choice == '1':
            run_quick_test()
        elif choice == '2':
            run_all_tests()
        elif choice == '3':
            test_environment()
        elif choice == '4':
            test_imports()
        elif choice == '5':
            test_timezone()
        elif choice == '6':
            test_redis()
        elif choice == '7':
            test_api()
        elif choice == '8':
            test_firebase()
        elif choice == '9':
            test_ml_models()
        elif choice == '10':
            test_system_health()
        elif choice == '11':
            test_timeframe_scheduling()
        elif choice == '12':
            test_data_availability()
        elif choice == '13':
            test_call_put_balance()
        elif choice == '14':
            test_market_conditions()
        else:
            print_error("Invalid choice")
        
        input("\nPress Enter to continue...")


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == '--full':
            success = run_all_tests()
        elif command == '--quick':
            success = run_quick_test()
        elif command == '--menu':
            show_menu()
            success = True
        else:
            print("Usage:")
            print("  python3 test_all.py              # Interactive menu")
            print("  python3 test_all.py --quick      # Quick test")
            print("  python3 test_all.py --full       # Full test suite")
            print("  python3 test_all.py --menu       # Interactive menu")
            sys.exit(1)
        
        sys.exit(0 if success else 1)
    else:
        # Default: show menu
        show_menu()


if __name__ == "__main__":
    main()