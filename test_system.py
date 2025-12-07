"""
System Testing Utilities
Comprehensive tests for all components
"""

import sys
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def print_header(text):
    """Print test header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def test_environment():
    """Test environment configuration"""
    print_header("🧪 TEST 1: Environment Configuration")
    
    try:
        from config import validate_environment, get_config_summary
        
        validate_environment()
        summary = get_config_summary()
        
        print("\n✅ Environment validated")
        print(f"\nConfiguration:")
        for key, value in summary.items():
            print(f"  • {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Environment test failed: {e}")
        return False


def test_imports():
    """Test all required imports"""
    print_header("🧪 TEST 2: Required Imports")
    
    imports = {
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'tensorflow': 'TensorFlow',
        'sklearn': 'Scikit-learn',
        'firebase_admin': 'Firebase Admin',
        'schedule': 'Schedule',
        'psutil': 'PSUtil',
        'requests': 'Requests',
        'redis': 'Redis (optional)',
    }
    
    failed = []
    
    for module, name in imports.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            if module == 'redis':
                print(f"  ⚠️  {name} (optional, not required)")
            else:
                print(f"  ❌ {name}")
                failed.append(name)
    
    if failed:
        print(f"\n❌ Missing required modules: {', '.join(failed)}")
        return False
    
    print("\n✅ All required imports successful")
    return True


def test_firebase():
    """Test Firebase connection"""
    print_header("🧪 TEST 3: Firebase Connection")
    
    try:
        from firebase_manager import FirebaseManager
        
        print("\n🔗 Connecting to Firebase...")
        fb = FirebaseManager()
        
        if fb.connected:
            print("✅ Firebase connected")
            
            # Test write
            print("\n📝 Testing write...")
            test_data = {
                'test': True,
                'timestamp': datetime.now().isoformat(),
                'message': 'Test connection'
            }
            
            collection = fb.firestore_db.collection('system_test')
            doc_ref = collection.add(test_data)
            
            print(f"✅ Write successful: {doc_ref[1].id}")
            
            # Test read
            print("\n📖 Testing read...")
            doc = collection.document(doc_ref[1].id).get()
            
            if doc.exists:
                print("✅ Read successful")
                
                # Cleanup
                print("\n🧹 Cleaning up...")
                collection.document(doc_ref[1].id).delete()
                print("✅ Cleanup successful")
                
                return True
            else:
                print("❌ Read failed")
                return False
        else:
            print("❌ Firebase connection failed")
            return False
            
    except Exception as e:
        print(f"\n❌ Firebase test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api():
    """Test CryptoCompare API"""
    print_header("🧪 TEST 4: CryptoCompare API")
    
    try:
        from btc_predictor_automated import get_current_btc_price, get_bitcoin_data_realtime
        
        print("\n💰 Testing current price...")
        price = get_current_btc_price()
        
        if price:
            print(f"✅ Current BTC price: ${price:,.2f}")
        else:
            print("❌ Failed to get current price")
            return False
        
        print("\n📊 Testing historical data...")
        df = get_bitcoin_data_realtime(days=1, interval='hour')
        
        if df is not None and len(df) > 0:
            print(f"✅ Retrieved {len(df)} data points")
            print(f"   Latest price: ${df.iloc[0]['price']:,.2f}")
            print(f"   Date range: {df.iloc[-1]['datetime']} to {df.iloc[0]['datetime']}")
            return True
        else:
            print("❌ Failed to get historical data")
            return False
            
    except Exception as e:
        print(f"\n❌ API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ml_models():
    """Test ML model functionality"""
    print_header("🧪 TEST 5: ML Models")
    
    try:
        from btc_predictor_improved import ImprovedBitcoinPredictor
        from btc_predictor_automated import get_bitcoin_data_realtime, add_technical_indicators
        
        print("\n🤖 Creating predictor...")
        predictor = ImprovedBitcoinPredictor()
        print("✅ Predictor created")
        
        # Try to load existing models
        print("\n📦 Checking for existing models...")
        if predictor.load_models():
            print("✅ Models loaded successfully")
            
            # Test prediction
            print("\n🔮 Testing prediction...")
            df = get_bitcoin_data_realtime(days=7, interval='hour')
            
            if df is not None:
                df = add_technical_indicators(df)
                prediction = predictor.predict(df, 60)  # 1 hour prediction
                
                if prediction:
                    print("✅ Prediction successful")
                    print(f"   Current: ${prediction['current_price']:,.2f}")
                    print(f"   Predicted: ${prediction['predicted_price']:,.2f}")
                    print(f"   Change: {prediction['price_change_pct']:+.2f}%")
                    print(f"   Confidence: {prediction['confidence']:.1f}%")
                    return True
                else:
                    print("⚠️  Prediction returned None (confidence too low?)")
                    return True  # Not a failure
            else:
                print("❌ Failed to get data for prediction")
                return False
        else:
            print("⚠️  No trained models found")
            print("   Run training first: python3 scheduler_improved.py")
            return True  # Not a failure, just needs training
            
    except Exception as e:
        print(f"\n❌ ML test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache():
    """Test cache system"""
    print_header("🧪 TEST 6: Cache System")
    
    try:
        from cache_manager import get_cache
        
        cache = get_cache()
        
        print("\n📦 Testing cache operations...")
        
        # Test set
        test_key = "test_key"
        test_value = {"data": "test", "timestamp": datetime.now().isoformat()}
        
        cache.set(test_key, test_value, ttl=60)
        print("✅ Cache set")
        
        # Test get
        retrieved = cache.get(test_key)
        if retrieved and retrieved == test_value:
            print("✅ Cache get")
        else:
            print("❌ Cache get failed")
            return False
        
        # Test delete
        cache.delete(test_key)
        retrieved = cache.get(test_key)
        if retrieved is None:
            print("✅ Cache delete")
        else:
            print("❌ Cache delete failed")
            return False
        
        # Get stats
        stats = cache.get_stats()
        print(f"\n📊 Cache stats:")
        print(f"   Backend: {stats['backend']}")
        print(f"   Hit rate: {stats['hit_rate']:.1f}%")
        
        print("\n✅ Cache system working")
        return True
        
    except Exception as e:
        print(f"\n❌ Cache test failed: {e}")
        return False


def test_alerts():
    """Test alert system"""
    print_header("🧪 TEST 7: Alert System")
    
    try:
        from alert_system import get_alert_manager, AlertSeverity
        
        alert_mgr = get_alert_manager()
        
        print(f"\n📢 Alert system initialized")
        print(f"   Enabled: {alert_mgr.enabled}")
        
        if alert_mgr.enabled:
            print("\n🔔 Sending test alert...")
            success = alert_mgr.send_alert(
                "System Test",
                "This is a test alert from the Bitcoin Predictor system test suite.",
                AlertSeverity.INFO,
                "system_test"
            )
            
            if success:
                print("✅ Test alert sent")
            else:
                print("⚠️  Alert sending disabled or failed")
        else:
            print("⚠️  Alerts disabled in configuration")
        
        summary = alert_mgr.get_alert_summary()
        print(f"\n📊 Alert summary:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Alert test failed: {e}")
        return False


def test_system_health():
    """Test system health monitoring"""
    print_header("🧪 TEST 8: System Health")
    
    try:
        from system_health import SystemHealthMonitor
        
        monitor = SystemHealthMonitor()
        
        print("\n🏥 Running health check...")
        report = monitor.get_full_health_report()
        
        print(f"\n📊 Health Report:")
        print(f"   Overall Status: {report['overall_status']}")
        print(f"   Memory: {report['memory']['process_memory_mb']:.0f}MB")
        print(f"   CPU: {report['cpu']['cpu_percent']:.1f}%")
        print(f"   Disk Free: {report['disk']['free_gb']:.2f}GB")
        print(f"   Network: {report['network']['status']}")
        
        if report['overall_status'] in ['HEALTHY', 'WARNING']:
            print("\n✅ System health check passed")
            return True
        else:
            print("\n⚠️  System health issues detected")
            return False
            
    except Exception as e:
        print(f"\n❌ Health test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("🧪 BITCOIN PREDICTOR - SYSTEM TESTS")
    print("="*80)
    
    tests = [
        ("Environment", test_environment),
        ("Imports", test_imports),
        ("Firebase", test_firebase),
        ("API", test_api),
        ("ML Models", test_ml_models),
        ("Cache", test_cache),
        ("Alerts", test_alerts),
        ("System Health", test_system_health),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except KeyboardInterrupt:
            print("\n\n⚠️  Tests interrupted")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {name}")
    
    print("\n" + "="*80)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        print("="*80)
        return False


def quick_test():
    """Quick smoke test"""
    print_header("⚡ QUICK TEST")
    
    tests = [
        ("Environment", test_environment),
        ("Imports", test_imports),
        ("API", test_api),
    ]
    
    for name, test_func in tests:
        if not test_func():
            print(f"\n❌ Quick test failed at: {name}")
            return False
    
    print("\n✅ Quick test passed!")
    print("For full testing, run: python3 test_system.py --full")
    return True


def main():
    """Main test runner"""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--full':
            success = run_all_tests()
        elif sys.argv[1] == '--quick':
            success = quick_test()
        else:
            print("Usage:")
            print("  python3 test_system.py           # Quick test")
            print("  python3 test_system.py --quick   # Quick test")
            print("  python3 test_system.py --full    # Full test suite")
            sys.exit(1)
    else:
        success = quick_test()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()