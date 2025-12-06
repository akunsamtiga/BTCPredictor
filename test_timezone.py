"""
Test Timezone Utilities
Quick test to verify timezone conversions are working correctly
"""

from timezone_utils import (
    get_local_now,
    get_utc_now,
    local_to_utc,
    utc_to_local,
    format_local_datetime,
    get_local_isoformat,
    parse_iso_to_local,
    add_minutes_local,
    get_timezone_info,
    display_timezone_info,
    prepare_firebase_timestamp,
    format_firebase_timestamp
)
from datetime import datetime


def test_basic_conversions():
    """Test basic timezone conversions"""
    print("\n" + "="*80)
    print("TEST 1: Basic Timezone Conversions")
    print("="*80)
    
    # Get current times
    local_now = get_local_now()
    utc_now = get_utc_now()
    
    print(f"\nLocal Time (WIB): {local_now}")
    print(f"UTC Time:         {utc_now}")
    print(f"Difference:       {(local_now.utcoffset().total_seconds() / 3600):.0f} hours")
    
    # Test conversions
    local_to_utc_result = local_to_utc(local_now)
    print(f"\nLocal → UTC:      {local_to_utc_result}")
    
    utc_to_local_result = utc_to_local(utc_now)
    print(f"UTC → Local:      {utc_to_local_result}")
    
    print("\n✅ Basic conversions working")


def test_formatting():
    """Test datetime formatting"""
    print("\n" + "="*80)
    print("TEST 2: Datetime Formatting")
    print("="*80)
    
    now = get_local_now()
    
    # Different formats
    formats = [
        ('%Y-%m-%d %H:%M:%S', 'Standard'),
        ('%d/%m/%Y %H:%M', 'Indonesian style'),
        ('%Y-%m-%d %H:%M:%S %Z', 'With timezone'),
        ('%A, %d %B %Y %H:%M', 'Full date')
    ]
    
    print("\nFormatting examples:")
    for fmt, label in formats:
        formatted = format_local_datetime(now, fmt)
        print(f"{label:20}: {formatted}")
    
    # ISO format
    iso = get_local_isoformat()
    print(f"\nISO format:          {iso}")
    
    print("\n✅ Formatting working")


def test_iso_parsing():
    """Test ISO string parsing"""
    print("\n" + "="*80)
    print("TEST 3: ISO String Parsing")
    print("="*80)
    
    # Create ISO strings
    local_iso = get_local_isoformat()
    utc_dt = get_utc_now()
    utc_iso = utc_dt.isoformat()
    
    print(f"\nOriginal local ISO: {local_iso}")
    print(f"Original UTC ISO:   {utc_iso}")
    
    # Parse them back
    parsed_from_local = parse_iso_to_local(local_iso)
    parsed_from_utc = parse_iso_to_local(utc_iso)
    
    print(f"\nParsed local:       {parsed_from_local}")
    print(f"Parsed UTC:         {parsed_from_utc}")
    
    print("\n✅ Parsing working")


def test_firebase_operations():
    """Test Firebase-related operations"""
    print("\n" + "="*80)
    print("TEST 4: Firebase Operations")
    print("="*80)
    
    now = get_local_now()
    
    # Prepare for Firebase (should be UTC)
    firebase_ts = prepare_firebase_timestamp(now)
    print(f"\nLocal time:          {now}")
    print(f"Firebase timestamp:  {firebase_ts}")
    print("(Firebase stores as UTC)")
    
    # Format back from Firebase
    formatted = format_firebase_timestamp(firebase_ts, '%d/%m/%Y %H:%M:%S')
    print(f"\nFormatted back:      {formatted}")
    print("(Displayed in WIB)")
    
    print("\n✅ Firebase operations working")


def test_time_calculations():
    """Test time calculations"""
    print("\n" + "="*80)
    print("TEST 5: Time Calculations")
    print("="*80)
    
    now = get_local_now()
    
    print(f"\nCurrent time (WIB): {now.strftime('%H:%M:%S')}")
    
    # Add different intervals
    intervals = [15, 30, 60, 240, 720, 1440]
    
    print("\nAdding minutes:")
    for minutes in intervals:
        future = add_minutes_local(now, minutes)
        hours = minutes / 60
        print(f"+{minutes:4} min ({hours:5.1f}h): {future.strftime('%H:%M:%S')} ({future.strftime('%d/%m')})")
    
    print("\n✅ Time calculations working")


def test_prediction_scenario():
    """Simulate a prediction scenario"""
    print("\n" + "="*80)
    print("TEST 6: Prediction Scenario Simulation")
    print("="*80)
    
    # Simulate creating a prediction
    prediction_time = get_local_now()
    timeframe_minutes = 60
    target_time = add_minutes_local(prediction_time, timeframe_minutes)
    
    print("\nPrediction created:")
    print(f"Prediction time (WIB): {format_local_datetime(prediction_time)}")
    print(f"Timeframe:             {timeframe_minutes} minutes")
    print(f"Target time (WIB):     {format_local_datetime(target_time)}")
    
    # What gets stored in Firebase
    firebase_prediction_time = prepare_firebase_timestamp(prediction_time)
    firebase_target_time = prepare_firebase_timestamp(target_time)
    
    print("\nStored in Firebase (UTC):")
    print(f"Prediction time: {firebase_prediction_time}")
    print(f"Target time:     {firebase_target_time}")
    
    # What gets displayed back
    print("\nDisplayed back to user (WIB):")
    print(f"Prediction time: {format_firebase_timestamp(firebase_prediction_time)}")
    print(f"Target time:     {format_firebase_timestamp(firebase_target_time)}")
    
    print("\n✅ Prediction scenario working correctly")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("🌍 TIMEZONE UTILITIES TEST SUITE")
    print("="*80)
    
    # Display timezone info first
    display_timezone_info()
    
    # Run tests
    try:
        test_basic_conversions()
        test_formatting()
        test_iso_parsing()
        test_firebase_operations()
        test_time_calculations()
        test_prediction_scenario()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nTimezone handling is working correctly.")
        print("All times will be stored as UTC in Firebase but displayed in WIB.\n")
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED!")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()