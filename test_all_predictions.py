#!/usr/bin/env python3
"""
Test All Predictions - Force prediction for all timeframes
Tests if predictions can be successfully created and saved to Firebase
"""

import sys
import logging
from datetime import datetime
from firebase_manager import FirebaseManager
from btc_predictor_automated import (
    ImprovedBitcoinPredictor,
    get_bitcoin_data_realtime,
    add_technical_indicators,
    get_current_btc_price
)
from config import (
    PREDICTION_CONFIG,
    get_timeframe_category,
    get_timeframe_label,
    get_data_config_for_timeframe
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def print_section(text):
    """Print section"""
    print(f"\n{text}")
    print("-" * 80)


def test_single_timeframe(predictor, firebase, timeframe_minutes):
    """Test prediction for a single timeframe"""
    try:
        label = get_timeframe_label(timeframe_minutes)
        category = get_timeframe_category(timeframe_minutes)
        data_config = get_data_config_for_timeframe(timeframe_minutes)
        
        print(f"\n📊 Testing {label} ({category})...")
        print(f"   Config: {data_config['days']} days, {data_config['interval']} interval")
        
        # Fetch data
        print(f"   📡 Fetching data...")
        df = get_bitcoin_data_realtime(
            days=data_config['days'],
            interval=data_config['interval']
        )
        
        if df is None:
            print(f"   ❌ Failed to fetch data")
            return False
        
        print(f"   ✅ Fetched {len(df)} data points")
        
        # Add indicators
        df = add_technical_indicators(df)
        df_clean = df.dropna()
        
        print(f"   ✅ After cleaning: {len(df_clean)} points")
        
        if len(df_clean) < data_config['min_points']:
            print(f"   ⚠️ Insufficient data: {len(df_clean)} < {data_config['min_points']}")
            return False
        
        # Make prediction
        print(f"   🧠 Making prediction...")
        prediction = predictor.predict(df_clean, timeframe_minutes, always_predict=True)
        
        if not prediction:
            print(f"   ❌ Prediction failed")
            return False
        
        # Display prediction
        trend_icon = "🟢 ↗️" if prediction['price_change'] > 0 else "🔴 ↘️"
        print(f"   {trend_icon} ${prediction['predicted_price']:,.2f} "
              f"({prediction['price_change_pct']:+.2f}%)")
        print(f"   Confidence: {prediction['confidence']:.1f}%")
        print(f"   Quality: {prediction['quality_score']:.1f}")
        
        # Save to Firebase
        print(f"   💾 Saving to Firebase...")
        doc_id = firebase.save_prediction(prediction)
        
        if doc_id:
            print(f"   ✅ SAVED: {doc_id}")
            return True
        else:
            print(f"   ❌ FAILED TO SAVE")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_timeframes():
    """Test predictions for all active timeframes"""
    print_header("🧪 TESTING ALL TIMEFRAMES")
    
    # Initialize
    print_section("1. Initialization")
    
    print("🔧 Loading ML models...")
    predictor = ImprovedBitcoinPredictor()
    
    if not predictor.load_models():
        print("❌ Models not loaded!")
        print("   Run: python3 maintenance.py → 2 (Force Retrain)")
        return False
    
    print("✅ Models loaded")
    print(f"   Features: {len(predictor.feature_columns)}")
    
    print("\n🔗 Connecting to Firebase...")
    firebase = FirebaseManager()
    
    if not firebase.connected:
        print("❌ Firebase not connected!")
        return False
    
    print("✅ Firebase connected")
    
    # Get current price
    print("\n💰 Getting current BTC price...")
    current_price = get_current_btc_price()
    
    if current_price:
        print(f"✅ Current BTC: ${current_price:,.2f}")
    else:
        print("⚠️ Could not get current price (will continue anyway)")
    
    # Test all timeframes
    print_section("2. Testing All Timeframes")
    
    all_timeframes = PREDICTION_CONFIG['active_timeframes']
    
    # Group by category
    categories = {
        'ultra_short': [],
        'short': [],
        'medium': [],
        'long': []
    }
    
    for tf in all_timeframes:
        category = get_timeframe_category(tf)
        categories[category].append(tf)
    
    results = {}
    successful = 0
    failed = 0
    
    # Test each category
    for category_name in ['ultra_short', 'short', 'medium', 'long']:
        timeframes = categories[category_name]
        
        if not timeframes:
            continue
        
        print(f"\n{'='*80}")
        print(f"📈 {category_name.upper()} TIMEFRAMES")
        print(f"{'='*80}")
        
        for tf in sorted(timeframes):
            success = test_single_timeframe(predictor, firebase, tf)
            
            label = get_timeframe_label(tf)
            results[label] = success
            
            if success:
                successful += 1
            else:
                failed += 1
    
    # Summary
    print_section("3. Test Summary")
    
    total = len(all_timeframes)
    
    print(f"\n📊 Results:")
    print(f"   Total:      {total}")
    print(f"   ✅ Success: {successful}")
    print(f"   ❌ Failed:  {failed}")
    print(f"   Success Rate: {(successful/total*100):.1f}%")
    
    print(f"\n📋 Detailed Results:")
    
    for category_name in ['ultra_short', 'short', 'medium', 'long']:
        timeframes = categories[category_name]
        if not timeframes:
            continue
        
        print(f"\n{category_name.upper()}:")
        for tf in sorted(timeframes):
            label = get_timeframe_label(tf)
            status = "✅ PASS" if results.get(label, False) else "❌ FAIL"
            print(f"  {status}  {label}")
    
    # Firebase check
    if successful > 0:
        print_section("4. Verifying Firebase")
        
        print("\n📋 Getting unvalidated predictions...")
        predictions = firebase.get_unvalidated_predictions()
        
        print(f"✅ Found {len(predictions)} predictions in Firebase")
        
        if len(predictions) >= successful:
            print("✅ All predictions saved successfully!")
        else:
            print(f"⚠️ Some predictions may be missing ({len(predictions)}/{successful})")
    
    # Final verdict
    print_section("5. Final Verdict")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ System is working perfectly")
        print("✅ All timeframes can create and save predictions")
        return True
    elif successful > 0:
        print(f"\n⚠️ PARTIAL SUCCESS")
        print(f"✅ {successful} timeframes working")
        print(f"❌ {failed} timeframes failed")
        print("\n💡 Recommendations:")
        print("   1. Check failed timeframes' data requirements")
        print("   2. May need more training data for some timeframes")
        return False
    else:
        print("\n❌ ALL TESTS FAILED")
        print("🔧 System needs attention:")
        print("   1. Check model training")
        print("   2. Check data availability")
        print("   3. Check Firebase connection")
        return False


def test_quick():
    """Quick test - only test one timeframe per category"""
    print_header("⚡ QUICK TEST")
    
    # Initialize
    print("\n🔧 Loading models...")
    predictor = ImprovedBitcoinPredictor()
    
    if not predictor.load_models():
        print("❌ Models not loaded!")
        return False
    
    print("✅ Models loaded")
    
    print("\n🔗 Connecting to Firebase...")
    firebase = FirebaseManager()
    
    if not firebase.connected:
        print("❌ Firebase not connected!")
        return False
    
    print("✅ Firebase connected")
    
    # Test one from each category
    test_timeframes = {
        'ultra_short': 15,   # 15 minutes
        'short': 60,         # 1 hour
        'medium': 240,       # 4 hours
        'long': 1440         # 1 day
    }
    
    results = {}
    
    for category, tf in test_timeframes.items():
        print(f"\n{'='*80}")
        print(f"Testing {category.upper()}: {get_timeframe_label(tf)}")
        print(f"{'='*80}")
        
        success = test_single_timeframe(predictor, firebase, tf)
        results[category] = success
    
    # Summary
    print_section("Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    for category, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}  {category}")
    
    print(f"\n📊 {passed}/{total} categories passed")
    
    if passed == total:
        print("\n✅ Quick test passed!")
        print("   Run full test: python3 test_all_predictions.py --full")
        return True
    else:
        print(f"\n⚠️ {total - passed} categories failed")
        return False


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == '--full':
            success = test_all_timeframes()
        elif command == '--quick':
            success = test_quick()
        else:
            print("Usage:")
            print("  python3 test_all_predictions.py --full    # Test all timeframes")
            print("  python3 test_all_predictions.py --quick   # Quick test (one per category)")
            sys.exit(1)
    else:
        # Default: full test
        success = test_all_timeframes()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()