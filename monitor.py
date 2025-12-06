"""
Enhanced Monitoring and Statistics Script for Bitcoin Predictor
Multi-Timeframe Analysis with Category-Based Display
"""

import sys
import os
from datetime import datetime, timedelta
from firebase_manager import FirebaseManager
from config import (
    PREDICTION_CONFIG, get_timeframe_category, 
    get_timeframe_label, STRATEGY_CONFIG
)
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


def show_category_stats(fb, days=7):
    """Show statistics grouped by timeframe categories"""
    print_section(f"📈 Performance by Category (Last {days} Days)")
    
    categories = {
        'ultra_short': {
            'name': 'ULTRA SHORT (Scalping)',
            'icon': '⚡',
            'timeframes': PREDICTION_CONFIG['ultra_short_timeframes']
        },
        'short': {
            'name': 'SHORT TERM (Day Trading)',
            'icon': '📊',
            'timeframes': PREDICTION_CONFIG['short_timeframes']
        },
        'medium': {
            'name': 'MEDIUM TERM (Swing Trading)',
            'icon': '📈',
            'timeframes': PREDICTION_CONFIG['medium_timeframes']
        },
        'long': {
            'name': 'LONG TERM (Position Trading)',
            'icon': '🎯',
            'timeframes': PREDICTION_CONFIG['long_timeframes']
        }
    }
    
    try:
        for cat_key, cat_info in categories.items():
            print(f"\n{cat_info['icon']} {cat_info['name']}")
            print("   " + "-" * 74)
            
            category_stats = []
            total_predictions = 0
            total_wins = 0
            total_error = 0
            
            for tf in cat_info['timeframes']:
                stats = fb.get_statistics(timeframe_minutes=tf, days=days)
                
                if stats and stats.get('total_predictions', 0) > 0:
                    category_stats.append(stats)
                    total_predictions += stats['total_predictions']
                    total_wins += stats['wins']
                    total_error += stats['avg_error'] * stats['total_predictions']
                    
                    label = get_timeframe_label(tf)
                    win_rate = stats['win_rate']
                    
                    # Win rate color indicator
                    if win_rate >= 70:
                        indicator = "🟢"
                    elif win_rate >= 60:
                        indicator = "🟡"
                    elif win_rate >= 50:
                        indicator = "🟠"
                    else:
                        indicator = "🔴"
                    
                    # Progress bar
                    bar_length = int(win_rate / 5)
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    
                    print(f"   {indicator} {label:8} | Win: {win_rate:5.1f}% | "
                          f"Total: {stats['total_predictions']:4} | "
                          f"Wins: {stats['wins']:4} | "
                          f"Err: ${stats['avg_error']:6.0f}")
                    print(f"      [{bar}]")
            
            if category_stats:
                avg_win_rate = (total_wins / total_predictions * 100) if total_predictions > 0 else 0
                avg_error = total_error / total_predictions if total_predictions > 0 else 0
                
                print(f"\n   📌 Category Summary:")
                print(f"      Total Predictions: {total_predictions}")
                print(f"      Average Win Rate:  {avg_win_rate:.2f}%")
                print(f"      Average Error:     ${avg_error:.2f}")
            else:
                print(f"   ⚠️  No validated predictions for this category")
                
    except Exception as e:
        print(f"❌ Error getting category stats: {e}")


def show_timeframe_comparison(fb, days=7):
    """Show timeframe comparison table"""
    print_section(f"⚖️  Timeframe Comparison (Last {days} Days)")
    
    try:
        all_timeframes = PREDICTION_CONFIG['active_timeframes']
        
        # Header
        print(f"\n{'Timeframe':12} | {'Category':12} | {'Win%':6} | {'Total':6} | "
              f"{'Wins':5} | {'Loss':5} | {'AvgErr':8} | {'Status':8}")
        print("-" * 80)
        
        stats_list = []
        
        for tf in sorted(all_timeframes):
            stats = fb.get_statistics(timeframe_minutes=tf, days=days)
            
            if stats and stats.get('total_predictions', 0) > 0:
                stats_list.append((tf, stats))
        
        # Sort by win rate
        stats_list.sort(key=lambda x: x[1]['win_rate'], reverse=True)
        
        for tf, stats in stats_list:
            label = get_timeframe_label(tf)
            category = get_timeframe_category(tf)
            win_rate = stats['win_rate']
            
            # Status
            if win_rate >= 70:
                status = "🔥 HOT"
            elif win_rate >= 60:
                status = "✅ Good"
            elif win_rate >= 50:
                status = "⚠️  Fair"
            else:
                status = "❌ Poor"
            
            print(f"{label:12} | {category:12} | {win_rate:5.1f}% | "
                  f"{stats['total_predictions']:6} | "
                  f"{stats['wins']:5} | {stats['losses']:5} | "
                  f"${stats['avg_error']:7.0f} | {status}")
        
        if not stats_list:
            print("⚠️  No validated predictions available")
            
    except Exception as e:
        print(f"❌ Error showing comparison: {e}")


def show_best_worst_performers(fb, days=7):
    """Show best and worst performing timeframes"""
    print_section(f"🏆 Best & Worst Performers (Last {days} Days)")
    
    try:
        all_timeframes = PREDICTION_CONFIG['active_timeframes']
        
        stats_list = []
        for tf in all_timeframes:
            stats = fb.get_statistics(timeframe_minutes=tf, days=days)
            if stats and stats.get('total_predictions', 0) >= 10:  # Min 10 predictions
                stats_list.append((tf, stats))
        
        if not stats_list:
            print("⚠️  Not enough data for analysis")
            return
        
        # Sort by win rate
        stats_list.sort(key=lambda x: x[1]['win_rate'], reverse=True)
        
        # Top 5 performers
        print("\n🥇 TOP 5 PERFORMERS:")
        for i, (tf, stats) in enumerate(stats_list[:5], 1):
            label = get_timeframe_label(tf)
            category = get_timeframe_category(tf)
            
            medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i-1]
            print(f"{medal} {label:8} ({category:12}) | "
                  f"Win: {stats['win_rate']:5.1f}% | "
                  f"Total: {stats['total_predictions']:3} | "
                  f"Error: ${stats['avg_error']:6.0f}")
        
        # Bottom 5 performers
        print("\n⚠️  BOTTOM 5 PERFORMERS:")
        for i, (tf, stats) in enumerate(reversed(stats_list[-5:]), 1):
            label = get_timeframe_label(tf)
            category = get_timeframe_category(tf)
            
            print(f"{i}. {label:8} ({category:12}) | "
                  f"Win: {stats['win_rate']:5.1f}% | "
                  f"Total: {stats['total_predictions']:3} | "
                  f"Error: ${stats['avg_error']:6.0f}")
            
    except Exception as e:
        print(f"❌ Error showing best/worst: {e}")


def show_recent_predictions(fb, limit=10):
    """Show recent predictions with category grouping"""
    print_section(f"🔮 Recent Predictions (Last {limit})")
    
    try:
        collection = fb.firestore_db.collection('bitcoin_predictions')
        docs = collection.order_by('timestamp', direction='DESCENDING').limit(limit).stream()
        
        count = 0
        predictions_by_category = {}
        
        for doc in docs:
            data = doc.to_dict()
            count += 1
            
            timeframe = data.get('timeframe_minutes', 0)
            category = get_timeframe_category(timeframe)
            
            if category not in predictions_by_category:
                predictions_by_category[category] = []
            
            predictions_by_category[category].append(data)
        
        if count == 0:
            print("⚠️  No predictions found")
            return
        
        # Display by category
        for category in ['ultra_short', 'short', 'medium', 'long']:
            if category not in predictions_by_category:
                continue
            
            preds = predictions_by_category[category]
            
            category_names = {
                'ultra_short': '⚡ ULTRA SHORT',
                'short': '📊 SHORT',
                'medium': '📈 MEDIUM',
                'long': '🎯 LONG'
            }
            
            print(f"\n{category_names[category]}:")
            
            for pred in preds:
                timestamp = pred.get('prediction_time', 'N/A')
                if isinstance(timestamp, str):
                    timestamp = timestamp[:19]
                
                timeframe = pred.get('timeframe_minutes', 0)
                tf_label = get_timeframe_label(timeframe)
                current = pred.get('current_price', 0)
                predicted = pred.get('predicted_price', 0)
                trend = pred.get('trend', 'N/A')
                confidence = pred.get('confidence', 0)
                validated = pred.get('validated', False)
                result = pred.get('validation_result', 'PENDING')
                
                # Format trend
                trend_short = "CALL" if "CALL" in trend else "PUT" if "PUT" in trend else "?"
                arrow = "🟢 ↗" if trend_short == "CALL" else "🔴 ↘" if trend_short == "PUT" else "⚪"
                
                # Format result
                if validated:
                    result_icon = "✅" if result == "WIN" else "❌" if result == "LOSE" else "⏳"
                    result_text = f"{result_icon} {result}"
                else:
                    result_text = "⏳ PENDING"
                
                print(f"  • {timestamp[:16]} | {tf_label:6} | "
                      f"{arrow} ${predicted:,.0f} | "
                      f"Conf: {confidence:.0f}% | {result_text}")
        
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
            timestamp = data.get('timestamp', 'N/A')
            
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
    """Show predictions waiting for validation grouped by category"""
    print_section("⏳ Pending Validations")
    
    try:
        predictions = fb.get_unvalidated_predictions()
        
        if predictions:
            # Group by category
            pending_by_category = {}
            
            for pred in predictions:
                timeframe = pred['timeframe_minutes']
                category = get_timeframe_category(timeframe)
                
                if category not in pending_by_category:
                    pending_by_category[category] = []
                
                pending_by_category[category].append(pred)
            
            print(f"Found {len(predictions)} predictions waiting for validation:\n")
            
            for category in ['ultra_short', 'short', 'medium', 'long']:
                if category not in pending_by_category:
                    continue
                
                preds = pending_by_category[category]
                
                category_names = {
                    'ultra_short': '⚡ ULTRA SHORT',
                    'short': '📊 SHORT',
                    'medium': '📈 MEDIUM',
                    'long': '🎯 LONG'
                }
                
                print(f"{category_names[category]}: {len(preds)} pending")
                
                for pred in preds[:5]:  # Show max 5 per category
                    target_time = datetime.fromisoformat(pred['target_time'])
                    timeframe = pred['timeframe_minutes']
                    tf_label = get_timeframe_label(timeframe)
                    predicted = pred['predicted_price']
                    trend = pred['trend']
                    
                    trend_short = "CALL" if "CALL" in trend else "PUT"
                    arrow = "🟢 ↗" if trend_short == "CALL" else "🔴 ↘"
                    
                    time_until = (target_time - datetime.now()).total_seconds() / 60
                    
                    if time_until <= 0:
                        time_text = "⚡ Ready now"
                    elif time_until < 60:
                        time_text = f"⏰ in {int(time_until)}min"
                    else:
                        hours = time_until / 60
                        time_text = f"⏰ in {hours:.1f}h"
                    
                    print(f"  • {tf_label:6} | {arrow} ${predicted:,.0f} | {time_text}")
                
                if len(preds) > 5:
                    print(f"  ... and {len(preds) - 5} more")
                print()
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


def show_volatility_analysis(fb, days=7):
    """Show volatility-based performance analysis"""
    print_section(f"📊 Volatility Analysis (Last {days} Days)")
    
    try:
        print("\nPerformance by Market Condition:")
        print("(Based on strategy configuration)\n")
        
        volatility_ranges = {
            'High Volatility (>3%)': {
                'timeframes': STRATEGY_CONFIG['volatility_adjustments']['high']['prefer_timeframes'],
                'multiplier': STRATEGY_CONFIG['volatility_adjustments']['high']['confidence_multiplier']
            },
            'Medium Volatility (1-3%)': {
                'timeframes': STRATEGY_CONFIG['volatility_adjustments']['medium']['prefer_timeframes'],
                'multiplier': STRATEGY_CONFIG['volatility_adjustments']['medium']['confidence_multiplier']
            },
            'Low Volatility (<1%)': {
                'timeframes': STRATEGY_CONFIG['volatility_adjustments']['low']['prefer_timeframes'],
                'multiplier': STRATEGY_CONFIG['volatility_adjustments']['low']['confidence_multiplier']
            }
        }
        
        for condition, config in volatility_ranges.items():
            print(f"🔹 {condition}")
            print(f"   Preferred Timeframes: {[get_timeframe_label(tf) for tf in config['timeframes']]}")
            print(f"   Confidence Multiplier: {config['multiplier']}")
            
            # Show stats for preferred timeframes
            total_preds = 0
            total_wins = 0
            
            for tf in config['timeframes']:
                stats = fb.get_statistics(timeframe_minutes=tf, days=days)
                if stats and stats.get('total_predictions', 0) > 0:
                    total_preds += stats['total_predictions']
                    total_wins += stats['wins']
            
            if total_preds > 0:
                win_rate = (total_wins / total_preds) * 100
                print(f"   Combined Win Rate: {win_rate:.1f}% ({total_wins}/{total_preds})")
            else:
                print(f"   Combined Win Rate: No data")
            print()
            
    except Exception as e:
        print(f"❌ Error showing volatility analysis: {e}")


def export_detailed_report(fb, days=7, filename=None):
    """Export detailed report to file"""
    print_section(f"📄 Exporting Detailed Report")
    
    try:
        if filename is None:
            filename = f"btc_predictor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("BITCOIN PREDICTOR - DETAILED PERFORMANCE REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Period: Last {days} days\n")
            f.write("="*80 + "\n\n")
            
            # Overall stats
            stats = fb.get_statistics(days=days)
            if stats and stats.get('total_predictions', 0) > 0:
                f.write("OVERALL STATISTICS\n")
                f.write("-"*80 + "\n")
                f.write(f"Total Predictions: {stats['total_predictions']}\n")
                f.write(f"Wins: {stats['wins']}\n")
                f.write(f"Losses: {stats['losses']}\n")
                f.write(f"Win Rate: {stats['win_rate']:.2f}%\n")
                f.write(f"Average Error: ${stats['avg_error']:.2f}\n")
                f.write(f"Average Error %: {stats['avg_error_pct']:.2f}%\n\n")
            
            # Per category
            categories = {
                'ultra_short': 'ULTRA SHORT (Scalping)',
                'short': 'SHORT TERM (Day Trading)',
                'medium': 'MEDIUM TERM (Swing Trading)',
                'long': 'LONG TERM (Position Trading)'
            }
            
            for cat_key, cat_name in categories.items():
                f.write(f"\n{cat_name}\n")
                f.write("-"*80 + "\n")
                
                timeframes = PREDICTION_CONFIG[f'{cat_key}_timeframes']
                
                for tf in timeframes:
                    tf_stats = fb.get_statistics(timeframe_minutes=tf, days=days)
                    
                    if tf_stats and tf_stats.get('total_predictions', 0) > 0:
                        label = get_timeframe_label(tf)
                        f.write(f"\n{label}:\n")
                        f.write(f"  Total: {tf_stats['total_predictions']}\n")
                        f.write(f"  Wins: {tf_stats['wins']}\n")
                        f.write(f"  Win Rate: {tf_stats['win_rate']:.2f}%\n")
                        f.write(f"  Avg Error: ${tf_stats['avg_error']:.2f}\n")
                
                f.write("\n")
        
        print(f"✅ Report exported to: {filename}")
        print(f"   File size: {os.path.getsize(filename)} bytes")
        
    except Exception as e:
        print(f"❌ Error exporting report: {e}")


def interactive_menu():
    """Interactive menu for monitoring"""
    print_header("🪙 Bitcoin Predictor - Enhanced Multi-Timeframe Monitor")
    
    # Check Firebase connection
    fb = check_firebase_connection()
    
    if not fb:
        print("\n❌ Cannot proceed without Firebase connection")
        return
    
    while True:
        print("\n" + "=" * 80)
        print("MENU:")
        print("  1. Show Overall Statistics")
        print("  2. Show Category Statistics")
        print("  3. Show Timeframe Comparison")
        print("  4. Show Best/Worst Performers")
        print("  5. Show Recent Predictions")
        print("  6. Show Model Performance")
        print("  7. Show Pending Validations")
        print("  8. Show System Health")
        print("  9. Show Volatility Analysis")
        print(" 10. Export Detailed Report")
        print(" 11. Full Dashboard (All)")
        print("  0. Exit")
        print("=" * 80)
        
        choice = input("\nSelect option (0-11): ").strip()
        
        if choice == '1':
            show_overall_stats(fb)
        elif choice == '2':
            show_category_stats(fb)
        elif choice == '3':
            show_timeframe_comparison(fb)
        elif choice == '4':
            show_best_worst_performers(fb)
        elif choice == '5':
            show_recent_predictions(fb)
        elif choice == '6':
            show_model_performance(fb)
        elif choice == '7':
            show_pending_validations(fb)
        elif choice == '8':
            show_system_health(fb)
        elif choice == '9':
            show_volatility_analysis(fb)
        elif choice == '10':
            export_detailed_report(fb)
        elif choice == '11':
            # Full dashboard
            show_system_health(fb)
            show_overall_stats(fb)
            show_category_stats(fb)
            show_best_worst_performers(fb)
            show_recent_predictions(fb, limit=5)
            show_pending_validations(fb)
            show_volatility_analysis(fb)
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
            show_category_stats(fb)
        elif command == 'category':
            show_category_stats(fb)
        elif command == 'comparison':
            show_timeframe_comparison(fb)
        elif command == 'best':
            show_best_worst_performers(fb)
        elif command == 'recent':
            show_recent_predictions(fb)
        elif command == 'health':
            show_system_health(fb)
        elif command == 'pending':
            show_pending_validations(fb)
        elif command == 'model':
            show_model_performance(fb)
        elif command == 'volatility':
            show_volatility_analysis(fb)
        elif command == 'export':
            export_detailed_report(fb)
        elif command == 'full':
            show_system_health(fb)
            show_overall_stats(fb)
            show_category_stats(fb)
            show_timeframe_comparison(fb)
            show_best_worst_performers(fb)
            show_recent_predictions(fb)
            show_pending_validations(fb)
            show_volatility_analysis(fb)
        else:
            print("Usage: python monitor.py [stats|category|comparison|best|recent|health|pending|model|volatility|export|full]")
    else:
        # Interactive mode
        interactive_menu()


if __name__ == "__main__":
    main()