#!/usr/bin/env python3
"""
Test Multiple API Calls Feature
Verify that data fetching works correctly with pagination
"""

import sys
from datetime import datetime
from btc_predictor_automated import get_bitcoin_data_realtime, add_technical_indicators


def print_header(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def test_single_request():
    """Test with data that fits in single request (< 2000 points)"""
    print_header("TEST 1: Single Request (< 2000 points)")
    
    test_cases = [
        (1, 'hour', 24),      # 1 day hourly = 24 points
        (7, 'hour', 168),     # 7 days hourly = 168 points
        (30, 'hour', 720),    # 30 days hourly = 720 points
        (1, 'day', 1),        # 1 day daily = 1 point
        (60, 'day', 60),      # 60 days daily = 60 points
    ]
    
    for days, interval, expected_points in test_cases:
        print(f"\n📊 Testing: {days} days, {interval} interval (expect ~{expected_points} points)")
        
        df = get_bitcoin_data_realtime(days=days, interval=interval)
        
        if df is not None:
            actual_points = len(df)
            status = "✅" if actual_points >= expected_points * 0.9 else "⚠️"
            print(f"{status} Got {actual_points} points (expected ~{expected_points})")
            print(f"   Date range: {df.iloc[-1]['datetime']} to {df.iloc[0]['datetime']}")
            print(f"   Latest price: ${df.iloc[0]['price']:,.2f}")
        else:
            print("❌ Failed to fetch data")
            return False
    
    print("\n✅ Single request tests passed!")
    return True


def test_multiple_requests():
    """Test with data that requires multiple requests (> 2000 points)"""
    print_header("TEST 2: Multiple Requests (> 2000 points)")
    
    test_cases = [
        # (days, interval, expected_points)
        (90, 'hour', 2160),      # 90 days hourly = 2160 points (2 batches)
        (120, 'hour', 2880),     # 120 days hourly = 2880 points (2 batches)
        (2, 'minute', 2880),     # 2 days minutely = 2880 points (2 batches)
        (3, 'minute', 4320),     # 3 days minutely = 4320 points (3 batches)
        (6, 'year', 2190),       # 6 years daily = 2190 points (2 batches)
    ]
    
    for days, interval, expected_points in test_cases:
        # Handle year as special case
        if interval == 'year':
            days_calc = days * 365
            interval_calc = 'day'
        else:
            days_calc = days
            interval_calc = interval
        
        print(f"\n📦 Testing: {days} {interval}, {interval_calc} interval")
        print(f"   Expected ~{expected_points} points (needs multiple batches)")
        
        df = get_bitcoin_data_realtime(days=days_calc, interval=interval_calc)
        
        if df is not None:
            actual_points = len(df)
            
            # Check if we got close to expected
            min_expected = expected_points * 0.9
            status = "✅" if actual_points >= min_expected else "⚠️"
            
            print(f"{status} Got {actual_points} points (expected ~{expected_points})")
            print(f"   Date range: {df.iloc[-1]['datetime']} to {df.iloc[0]['datetime']}")
            print(f"   Latest price: ${df.iloc[0]['price']:,.2f}")
            
            # Check for gaps in data
            time_diffs = df['datetime'].diff().abs()
            gaps = time_diffs[time_diffs > time_diffs.median() * 2]
            
            if len(gaps) > 0:
                print(f"   ⚠️ Found {len(gaps)} potential gaps in data")
            else:
                print(f"   ✅ No significant gaps detected")
                
        else:
            print("❌ Failed to fetch data")
            return False
    
    print("\n✅ Multiple request tests passed!")
    return True


def test_data_integrity():
    """Test data integrity after multiple fetches"""
    print_header("TEST 3: Data Integrity Check")
    
    print("\n📊 Fetching large dataset (120 days hourly)...")
    df = get_bitcoin_data_realtime(days=120, interval='hour')
    
    if df is None:
        print("❌ Failed to fetch data")
        return False
    
    print(f"✅ Got {len(df)} data points")
    
    # Check 1: No null prices
    null_prices = df['price'].isnull().sum()
    print(f"\n1️⃣ Null prices: {null_prices} {'✅' if null_prices == 0 else '❌'}")
    
    # Check 2: No zero prices
    zero_prices = (df['price'] == 0).sum()
    print(f"2️⃣ Zero prices: {zero_prices} {'✅' if zero_prices == 0 else '❌'}")
    
    # Check 3: Prices in reasonable range
    min_price = df['price'].min()
    max_price = df['price'].max()
    print(f"3️⃣ Price range: ${min_price:,.2f} - ${max_price:,.2f} ✅")
    
    # Check 4: Chronological order
    is_sorted = df['datetime'].is_monotonic_decreasing
    print(f"4️⃣ Sorted (newest first): {is_sorted} {'✅' if is_sorted else '❌'}")
    
    # Check 5: No duplicate timestamps
    duplicates = df['datetime'].duplicated().sum()
    print(f"5️⃣ Duplicate timestamps: {duplicates} {'✅' if duplicates == 0 else '❌'}")
    
    # Check 6: Time gaps
    df_sorted = df.sort_values('datetime', ascending=True)
    time_diffs = df_sorted['datetime'].diff()
    avg_gap = time_diffs.mean()
    print(f"6️⃣ Average time gap: {avg_gap} ✅")
    
    # Check 7: Add technical indicators
    print(f"\n7️⃣ Testing technical indicators...")
    df_with_indicators = add_technical_indicators(df)
    
    if df_with_indicators is not None and len(df_with_indicators.columns) > len(df.columns):
        print(f"   ✅ Added {len(df_with_indicators.columns) - len(df.columns)} indicators")
        
        # Check for nulls in indicators
        null_indicators = df_with_indicators.isnull().sum().sum()
        print(f"   Null values in indicators: {null_indicators}")
    else:
        print(f"   ❌ Failed to add indicators")
        return False
    
    all_passed = (null_prices == 0 and zero_prices == 0 and 
                  is_sorted and duplicates == 0)
    
    if all_passed:
        print("\n✅ All integrity checks passed!")
    else:
        print("\n⚠️ Some integrity checks failed")
    
    return all_passed


def test_extreme_cases():
    """Test edge cases"""
    print_header("TEST 4: Edge Cases")
    
    # Test 1: Very small request
    print("\n1️⃣ Testing very small request (1 hour)...")
    df = get_bitcoin_data_realtime(days=0.04167, interval='hour')  # ~1 hour
    if df is not None and len(df) > 0:
        print(f"   ✅ Got {len(df)} points")
    else:
        print(f"   ❌ Failed")
        return False
    
    # Test 2: Exactly at limit
    print("\n2️⃣ Testing exactly at API limit (83 days hourly = ~2000 points)...")
    df = get_bitcoin_data_realtime(days=83, interval='hour')
    if df is not None and len(df) > 1900:
        print(f"   ✅ Got {len(df)} points")
    else:
        print(f"   ❌ Failed or insufficient data")
        return False
    
    # Test 3: Just over limit
    print("\n3️⃣ Testing just over limit (85 days hourly = ~2040 points)...")
    df = get_bitcoin_data_realtime(days=85, interval='hour')
    if df is not None and len(df) > 2000:
        print(f"   ✅ Got {len(df)} points (multiple batches worked)")
    else:
        print(f"   ❌ Failed or insufficient data")
        return False
    
    print("\n✅ Edge case tests passed!")
    return True


def compare_single_vs_multiple():
    """Compare results from single vs multiple requests"""
    print_header("TEST 5: Single vs Multiple Request Comparison")
    
    print("\n📊 Fetching 7 days hourly (single request)...")
    df_single = get_bitcoin_data_realtime(days=7, interval='hour')
    
    print(f"✅ Single request: {len(df_single)} points")
    
    print("\n📦 Fetching 90 days hourly (multiple requests)...")
    df_multiple = get_bitcoin_data_realtime(days=90, interval='hour')
    
    print(f"✅ Multiple requests: {len(df_multiple)} points")
    
    # Compare overlap period (first 7 days of 90-day data)
    df_multiple_subset = df_multiple.head(len(df_single))
    
    # Compare prices
    price_diff = abs(df_single['price'] - df_multiple_subset['price']).mean()
    
    print(f"\n📊 Comparison:")
    print(f"   Average price difference: ${price_diff:.2f}")
    print(f"   Status: {'✅ Consistent' if price_diff < 1 else '⚠️ Some differences'}")
    
    return price_diff < 1


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("🧪 MULTIPLE API CALLS - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    results = {
        'Single Request': test_single_request(),
        'Multiple Requests': test_multiple_requests(),
        'Data Integrity': test_data_integrity(),
        'Edge Cases': test_extreme_cases(),
        'Consistency Check': compare_single_vs_multiple(),
    }
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}  {test_name}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print("\n" + "="*80)
    print(f"Results: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        return True
    else:
        print(f"⚠️ {total_count - passed_count} test(s) failed")
        print("="*80)
        return False


def main():
    """Main test runner"""
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == 'single':
            success = test_single_request()
        elif test_type == 'multiple':
            success = test_multiple_requests()
        elif test_type == 'integrity':
            success = test_data_integrity()
        elif test_type == 'edge':
            success = test_extreme_cases()
        elif test_type == 'compare':
            success = compare_single_vs_multiple()
        elif test_type == 'all':
            success = run_all_tests()
        else:
            print("Usage:")
            print("  python3 test_multiple_fetch.py [single|multiple|integrity|edge|compare|all]")
            print("\nTests:")
            print("  single    - Test single API requests (< 2000 points)")
            print("  multiple  - Test multiple API requests (> 2000 points)")
            print("  integrity - Test data integrity")
            print("  edge      - Test edge cases")
            print("  compare   - Compare single vs multiple results")
            print("  all       - Run all tests (recommended)")
            sys.exit(1)
    else:
        success = run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()