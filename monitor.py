"""
Monitoring and Statistics Script for Bitcoin Predictor
Run this to check system status and performance
"""

import sys
import os
from datetime import datetime, timedelta
from firebase_manager import FirebaseManager
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_section(text):
    """Print section header"""
    print(f"\n{text}")
    print("-" * 80)


def check_firebase_connection():
    """Test Firebase connection"""
    try:
        fb = FirebaseManager()
        print_section("🔗 Firebase Connection")
        print("✅ Firebase connected successfully")
        return fb
    except Exception as e:
        print_section("🔗 Firebase Connection")
        print(f"❌ Firebase connection failed: {e}")
        return None


def show_overall_stats(fb, days=7):
    """Show overall prediction statistics"""
    print_section(f"📊 Overall Statistics (Last {days} Days)")
    
    try:
        stats = fb.get_statistics(days=days)
        
        if stats and stats.get('total_predictions', 0) > 0:
            print(f"Total Predictions:  {stats['total_predictions']}")
            print(f"Wins:               {stats['wins']} (✅)")
            print(f"Losses:             {stats['losses']} (❌)")
            print(f"Win Rate:           {stats['win_rate']:.2f}%")
            print(f"Average Error:      ${stats['avg_error']:.2f}")
            print(f"Average Error %:    {stats['avg_error_pct']:.2f}%")
            
            # Win rate indicator
            if stats['win_rate'] >= 70:
                print("\n🎉 Excellent performance!")
            elif stats['win_rate'] >= 60:
                print("\n✅ Good performance")
            elif stats['win_rate'] >= 50:
                print("\n⚠️  Average performance")
            else:
                print("\n❌ Below average performance")
        else:
            print("⚠️  No validated predictions yet")
            print("   Predictions need time to be validated.")
            
    except Exception as e:
        print(f"❌ Error getting overall stats: {e}")


def show_timeframe_stats(fb, days=7):
    """Show statistics per timeframe"""
    print_section(f"⏱️  Performance by Timeframe (Last {days} Days)")
    
    timeframes = [15, 30, 60, 240, 720, 1440]
    timeframe_labels = {
        15: "15 minutes",
        30: "30 minutes", 
        60: "1 hour",
        240: "4 hours",
        720: "12 hours",
        1440: "24 hours"
    }
    
    try:
        for tf in timeframes:
            stats = fb.get_statistics(timeframe_minutes=tf, days=days)
            
            if stats and stats.get('total_predictions', 0) > 0:
                label = timeframe_labels.get(tf, f"{tf}min")
                print(f"\n{label:12} | Win Rate: {stats['win_rate']:5.1f}% | "
                      f"Total: {stats['total_predictions']:3} | "
                      f"Wins: {stats['wins']:3} | "
                      f"Losses: {stats['losses']:3}")
                
                # Progress bar
                win_bars = int(stats['win_rate'] / 5)
                bar = "█" * win_bars + "░" * (20 - win_bars)
                print(f"             [{bar}]")
            else:
                label = timeframe_labels.get(tf, f"{tf}min")
                print(f"\n{label:12} | No validated predictions yet")
                
    except Exception as e:
        print(f"❌ Error getting timeframe stats: {e}")


def show_recent_predictions(fb, limit=10):
    """Show recent predictions"""
    print_section(f"🔮 Recent Predictions (Last {limit})")
    
    try:
        collection = fb.firestore_db.collection('bitcoin_predictions')
        docs = collection.order_by('timestamp', direction='DESCENDING').limit(limit).stream()
        
        count = 0
        for doc in docs:
            data = doc.to_dict()
            count += 1
            
            timestamp = data.get('prediction_time', 'N/A')
            if isinstance(timestamp, str):
                timestamp = timestamp[:19]  # Truncate to readable format
            
            timeframe = data.get('timeframe_minutes', 0)
            current = data.get('current_price', 0)
            predicted = data.get('predicted_price', 0)
            trend = data.get('trend', 'N/A')
            confidence = data.get('confidence', 0)
            validated = data.get('validated', False)
            result = data.get('validation_result', 'PENDING')
            
            # Format trend
            trend_short = "CALL" if "CALL" in trend else "PUT" if "PUT" in trend else "?"
            arrow = "🟢 ↗" if trend_short == "CALL" else "🔴 ↘" if trend_short == "PUT" else "⚪"
            
            # Format result
            if validated:
                result_icon = "✅" if result == "WIN" else "❌" if result == "LOSE" else "⏳"
                result_text = f"{result_icon} {result}"
            else:
                result_text = "⏳ PENDING"
            
            print(f"\n{count}. {timestamp} | {timeframe:4}min | "
                  f"{arrow} ${predicted:,.0f} | "
                  f"Conf: {confidence:.0f}% | {result_text}")
        
        if count == 0:
            print("⚠️  No predictions found")
            
    except Exception as e:
        print(f"❌ Error getting recent predictions: {e}")


def show_model_performance(fb):
    """Show model performance metrics"""
    print_section("🤖 Model Performance")
    
    try:
        collection = fb.firestore_db.collection('model_performance')
        docs = collection.order_by('timestamp', direction='DESCENDING').limit(1).stream()
        
        for doc in docs:
            data = doc.to_dict()
            metrics = data.get('metrics', {})
            timestamp = data.get('datetime', 'N/A')
            
            print(f"Last Training: {timestamp}")
            print("\nModel Metrics:")
            
            for model_name, model_metrics in metrics.items():
                print(f"\n  {model_name.upper()}:")
                for metric, value in model_metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"    {metric:10}: {value:.4f}")
            
            return
        
        print("⚠️  No model performance data available")
        
    except Exception as e:
        print(f"❌ Error getting model performance: {e}")


def show_pending_validations(fb):
    """Show predictions waiting for validation"""
    print_section("⏳ Pending Validations")
    
    try:
        predictions = fb.get_unvalidated_predictions()
        
        if predictions:
            print(f"Found {len(predictions)} predictions waiting for validation:\n")
            
            for i, pred in enumerate(predictions[:10], 1):  # Show max 10
                target_time = datetime.fromisoformat(pred['target_time'])
                timeframe = pred['timeframe_minutes']
                predicted = pred['predicted_price']
                trend = pred['trend']
                
                trend_short = "CALL" if "CALL" in trend else "PUT"
                arrow = "🟢 ↗" if trend_short == "CALL" else "🔴 ↘"
                
                time_until = (target_time - datetime.now()).total_seconds() / 60
                
                if time_until <= 0:
                    time_text = "Ready for validation"
                else:
                    time_text = f"in {int(time_until)} minutes"
                
                print(f"{i}. {timeframe:4}min | {arrow} ${predicted:,.0f} | {time_text}")
        else:
            print("✅ All predictions validated")
            
    except Exception as e:
        print(f"❌ Error getting pending validations: {e}")


def show_system_health(fb):
    """Show system health status"""
    print_section("🏥 System Health")
    
    try:
        # Check recent activity
        collection = fb.firestore_db.collection('bitcoin_predictions')
        recent_doc = collection.order_by('timestamp', direction='DESCENDING').limit(1).stream()
        
        for doc in recent_doc:
            data = doc.to_dict()
            last_prediction = data.get('prediction_time')
            
            if last_prediction:
                last_time = datetime.fromisoformat(last_prediction)
                time_diff = (datetime.now() - last_time).total_seconds() / 60
                
                print(f"Last Prediction: {last_prediction}")
                print(f"Time Since:      {int(time_diff)} minutes ago")
                
                if time_diff < 10:
                    print("Status:          ✅ HEALTHY - System running normally")
                elif time_diff < 30:
                    print("Status:          ⚠️  WARNING - No recent predictions")
                else:
                    print("Status:          ❌ ERROR - System may be down")
                
                return
        
        print("Status:          ❌ ERROR - No predictions found")
        
    except Exception as e:
        print(f"❌ Error checking system health: {e}")


def interactive_menu():
    """Interactive menu for monitoring"""
    print_header("🪙 Bitcoin Predictor - Monitoring Dashboard")
    
    # Check Firebase connection
    fb = check_firebase_connection()
    
    if not fb:
        print("\n❌ Cannot proceed without Firebase connection")
        return
    
    while True:
        print("\n" + "=" * 80)
        print("MENU:")
        print("  1. Show Overall Statistics")
        print("  2. Show Timeframe Statistics")
        print("  3. Show Recent Predictions")
        print("  4. Show Model Performance")
        print("  5. Show Pending Validations")
        print("  6. Show System Health")
        print("  7. Full Report (All)")
        print("  0. Exit")
        print("=" * 80)
        
        choice = input("\nSelect option (0-7): ").strip()
        
        if choice == '1':
            show_overall_stats(fb)
        elif choice == '2':
            show_timeframe_stats(fb)
        elif choice == '3':
            show_recent_predictions(fb)
        elif choice == '4':
            show_model_performance(fb)
        elif choice == '5':
            show_pending_validations(fb)
        elif choice == '6':
            show_system_health(fb)
        elif choice == '7':
            # Full report
            show_system_health(fb)
            show_overall_stats(fb)
            show_timeframe_stats(fb)
            show_recent_predictions(fb, limit=5)
            show_pending_validations(fb)
            show_model_performance(fb)
        elif choice == '0':
            print("\n👋 Goodbye!")
            break
        else:
            print("\n❌ Invalid choice")


def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Command line mode
        command = sys.argv[1].lower()
        
        fb = check_firebase_connection()
        if not fb:
            sys.exit(1)
        
        if command == 'stats':
            show_overall_stats(fb)
            show_timeframe_stats(fb)
        elif command == 'recent':
            show_recent_predictions(fb)
        elif command == 'health':
            show_system_health(fb)
        elif command == 'pending':
            show_pending_validations(fb)
        elif command == 'model':
            show_model_performance(fb)
        elif command == 'full':
            show_system_health(fb)
            show_overall_stats(fb)
            show_timeframe_stats(fb)
            show_recent_predictions(fb)
            show_pending_validations(fb)
            show_model_performance(fb)
        else:
            print("Usage: python monitor.py [stats|recent|health|pending|model|full]")
    else:
        # Interactive mode
        interactive_menu()


if __name__ == "__main__":
    main()