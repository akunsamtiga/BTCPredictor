"""
FIXED Maintenance Tools
FIXED: Use 90 days data for training
"""

import sys
import os
from datetime import datetime, timedelta
import logging
from firebase_manager import FirebaseManager
from btc_predictor_automated import (
    ImprovedBitcoinPredictor, get_bitcoin_data_realtime, 
    add_technical_indicators, get_current_btc_price
)
from config import (
    PREDICTION_CONFIG, get_timeframe_category, 
    get_timeframe_label, get_data_config_for_timeframe
)
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_section(text):
    """Print section"""
    print(f"\n{text}")


def cleanup_old_data(days=30):
    """Clean up data older than specified days"""
    print_header(f"🗑️ Cleaning Up Data Older Than {days} Days")
    
    try:
        fb = FirebaseManager()
        
        print(f"\nStarting cleanup...")
        fb.cleanup_old_data(days)
        
        print(f"\n✅ Cleanup completed successfully")
        
    except Exception as e:
        print(f"\n❌ Cleanup failed: {e}")


def force_retrain_models():
    """
    FIXED: Force retrain all ML models with 90 days data
    """
    print_header("🤖 Force Retraining Models (90 Days Data)")
    
    try:
        # ================================================================
        # FIXED: Fetch 90 days data for training
        # ================================================================
        print("\n📡 Fetching training data (90 days, hourly)...")
        df = get_bitcoin_data_realtime(days=90, interval='hour')
        
        if df is None:
            print("❌ Failed to fetch data")
            return
        
        initial_points = len(df)
        print(f"📊 Retrieved {initial_points} raw data points")
        
        print("🔧 Adding technical indicators...")
        df = add_technical_indicators(df)
        
        # Clean data
        df_clean = df.dropna()
        clean_points = len(df_clean)
        
        print(f"🧹 After cleaning: {clean_points} data points")
        print(f"   Dropped: {initial_points - clean_points} rows with NaN")
        
        # Check minimum requirement
        if clean_points < 1000:
            print(f"\n⚠️ Warning: Only {clean_points} data points (recommended: 1000+)")
            
            # Try fallback
            print("\n🔄 FALLBACK: Trying 180 days with daily interval...")
            df = get_bitcoin_data_realtime(days=180, interval='day')
            
            if df is None:
                print("❌ Fallback failed")
                return
            
            print(f"📊 Fallback retrieved {len(df)} data points")
            df = add_technical_indicators(df)
            df_clean = df.dropna()
            clean_points = len(df_clean)
            
            print(f"🧹 After cleaning: {clean_points} data points")
            
            if clean_points < 500:
                print(f"❌ Even fallback insufficient: {clean_points}")
                print("\n   Please check:")
                print("   1. API key is valid")
                print("   2. Internet connection")
                print("   3. CryptoCompare API status")
                return
        
        print("\n🤖 Initializing predictor...")
        predictor = ImprovedBitcoinPredictor()
        
        print("\n🚀 Starting training...")
        print("This may take 10-20 minutes...")
        
        success = predictor.train_models(df_clean, epochs=50, batch_size=32)
        
        if success:
            print("\n✅ Training completed successfully!")
            
            # Save performance to Firebase
            print("💾 Saving model performance to Firebase...")
            fb = FirebaseManager()
            fb.save_model_performance(predictor.metrics)
            
            print("\n📊 Model Metrics:")
            for model_name, metrics in predictor.metrics.items():
                print(f"\n{model_name.upper()}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
        else:
            print("\n❌ Training failed")
            
    except Exception as e:
        print(f"\n❌ Error during retraining: {e}")
        import traceback
        traceback.print_exc()


def validate_all_pending():
    """FIXED: Validate all pending predictions with correct logic"""
    print_header("✅ Validating All Pending Predictions")
    
    try:
        fb = FirebaseManager()
        
        print("\n📋 Getting unvalidated predictions...")
        predictions = fb.get_unvalidated_predictions()
        
        if not predictions:
            print("✅ No predictions to validate")
            return
        
        # Group by category
        predictions_by_category = {}
        for pred in predictions:
            tf = pred['timeframe_minutes']
            category = get_timeframe_category(tf)
            
            if category not in predictions_by_category:
                predictions_by_category[category] = []
            
            predictions_by_category[category].append(pred)
        
        print(f"Found {len(predictions)} predictions to validate")
        for cat, preds in predictions_by_category.items():
            print(f"  • {cat}: {len(preds)} predictions")
        
        print("\n💰 Getting current BTC price...")
        current_price = get_current_btc_price()
        
        if not current_price:
            print("❌ Cannot get current price")
            return
        
        print(f"Current price: ${current_price:,.2f}")
        
        print("\n🔍 Validating predictions...")
        validated_by_category = {cat: 0 for cat in predictions_by_category.keys()}
        
        for pred in predictions:
            doc_id = pred['doc_id']
            predicted_price = pred['predicted_price']
            original_price = pred['current_price']  # PENTING: Harga saat prediksi dibuat
            trend = pred['trend']
            timeframe = pred['timeframe_minutes']
            category = get_timeframe_category(timeframe)
            
            # FIXED: Pass both actual_price and original current_price
            success = fb.validate_prediction(
                doc_id, 
                current_price,      # actual_price (harga sekarang)
                predicted_price,    # predicted_price
                original_price,     # current_price (harga saat prediksi)
                trend
            )
            
            if success:
                validated_by_category[category] += 1
        
        print(f"\n✅ Validation completed:")
        for cat, count in validated_by_category.items():
            total = len(predictions_by_category[cat])
            print(f"  • {cat}: {count}/{total}")
        
        # Update statistics
        print("\n📊 Updating statistics...")
        fb.get_statistics(days=7)
        
        print("✅ Statistics updated")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()


def validate_category(category):
    """Validate pending predictions for specific category"""
    print_header(f"✅ Validating {category.upper()} Category")
    
    try:
        fb = FirebaseManager()
        
        timeframes = PREDICTION_CONFIG[f'{category}_timeframes']
        
        print(f"\nTimeframes: {[get_timeframe_label(tf) for tf in timeframes]}")
        print("Getting pending predictions...")
        
        all_predictions = fb.get_unvalidated_predictions()
        
        # Filter by category
        category_predictions = [
            p for p in all_predictions 
            if p['timeframe_minutes'] in timeframes
        ]
        
        if not category_predictions:
            print(f"✅ No pending predictions for {category}")
            return
        
        print(f"Found {len(category_predictions)} predictions for {category}")
        
        current_price = get_current_btc_price()
        if not current_price:
            print("❌ Cannot get current price")
            return
        
        print(f"Current price: ${current_price:,.2f}")
        print("\nValidating...")
        
        validated = 0
        for pred in category_predictions:
            success = fb.validate_prediction(
                pred['doc_id'],
                current_price,
                pred['predicted_price'],
                pred['current_price'],  # FIXED
                pred['trend']
            )
            if success:
                validated += 1
        
        print(f"\n✅ Validated {validated}/{len(category_predictions)} predictions")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")


def reset_statistics():
    """Recalculate and reset all statistics"""
    print_header("📊 Resetting Statistics")
    
    confirm = input("\n⚠️ This will recalculate all statistics. Continue? (yes/no): ")
    
    if confirm.lower() != 'yes':
        print("❌ Cancelled")
        return
    
    try:
        fb = FirebaseManager()
        
        print("\n🔄 Recalculating statistics...")
        
        # Overall stats
        overall = fb.get_statistics(days=30)
        if overall:
            fb.save_statistics(overall)
            print(f"✅ Overall stats: {overall['win_rate']:.1f}% win rate")
        
        # Per timeframe
        all_timeframes = PREDICTION_CONFIG['active_timeframes']
        
        categories = {}
        for tf in all_timeframes:
            category = get_timeframe_category(tf)
            if category not in categories:
                categories[category] = []
            categories[category].append(tf)
        
        for cat, timeframes in categories.items():
            print(f"\n{cat.upper()}:")
            for tf in timeframes:
                stats = fb.get_statistics(timeframe_minutes=tf, days=30)
                if stats and stats['total_predictions'] > 0:
                    fb.save_statistics(stats)
                    label = get_timeframe_label(tf)
                    print(f"  ✅ {label:6}: {stats['win_rate']:.1f}% win rate ({stats['total_predictions']} preds)")
        
        print("\n✅ Statistics reset completed")
        
    except Exception as e:
        print(f"\n❌ Reset failed: {e}")


def export_predictions_csv(days=7, filename=None, category=None):
    """Export predictions to CSV file"""
    if category:
        print_header(f"📤 Exporting {category.upper()} Predictions (Last {days} Days)")
    else:
        print_header(f"📤 Exporting All Predictions (Last {days} Days)")
    
    try:
        fb = FirebaseManager()
        
        if filename is None:
            if category:
                filename = f"predictions_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            else:
                filename = f"predictions_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        print(f"\n📋 Fetching predictions...")
        
        collection = fb.firestore_db.collection('bitcoin_predictions')
        cutoff_date = datetime.now() - timedelta(days=days)
        
        docs = collection.where('validated', '==', True).stream()
        
        predictions = []
        for doc in docs:
            data = doc.to_dict()
            
            # Check date
            if 'prediction_time' in data:
                pred_time = datetime.fromisoformat(data['prediction_time'])
                if pred_time < cutoff_date:
                    continue
            
            # Filter by category if specified
            if category:
                tf = data.get('timeframe_minutes', 0)
                if get_timeframe_category(tf) != category:
                    continue
            
            predictions.append({
                'timestamp': data.get('prediction_time'),
                'timeframe_minutes': data.get('timeframe_minutes'),
                'timeframe_label': get_timeframe_label(data.get('timeframe_minutes', 0)),
                'category': get_timeframe_category(data.get('timeframe_minutes', 0)),
                'current_price': data.get('current_price'),
                'predicted_price': data.get('predicted_price'),
                'price_change_pct': data.get('price_change_pct'),
                'trend': data.get('trend'),
                'confidence': data.get('confidence'),
                'actual_price': data.get('actual_price'),
                'validation_result': data.get('validation_result'),
                'price_error': data.get('price_error'),
                'price_error_pct': data.get('price_error_pct')
            })
        
        if not predictions:
            print("⚠️ No validated predictions found")
            return
        
        df = pd.DataFrame(predictions)
        df = df.sort_values('timestamp', ascending=False)
        df.to_csv(filename, index=False)
        
        print(f"\n✅ Exported {len(predictions)} predictions to {filename}")
        
        # Show summary by category
        category_counts = df['category'].value_counts()
        print("\nBreakdown by category:")
        for cat, count in category_counts.items():
            print(f"  • {cat}: {count} predictions")
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")


def show_disk_usage():
    """Show disk usage statistics"""
    print_header("💿 Disk Usage")
    
    try:
        import shutil
        
        directories = {
            'models': 'models',
            'logs': 'logs',
            'models_backup': 'models_backup'
        }
        
        print("\nDirectory sizes:")
        
        for name, path in directories.items():
            if os.path.exists(path):
                size = 0
                for dirpath, dirnames, filenames in os.walk(path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        size += os.path.getsize(fp)
                
                size_mb = size / (1024 * 1024)
                print(f"  {name:15}: {size_mb:10.2f} MB")
            else:
                print(f"  {name:15}: Not found")
        
        # Total disk
        total, used, free = shutil.disk_usage("/")
        print(f"\nTotal disk space:")
        print(f"  Total:  {total / (1024**3):.2f} GB")
        print(f"  Used:   {used / (1024**3):.2f} GB")
        print(f"  Free:   {free / (1024**3):.2f} GB")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


def main():
    """Main menu"""
    print_header("🔧 Bitcoin Predictor - Maintenance Tools (FIXED)")
    
    while True:
        print("\n" + "=" * 80)
        print("MENU:")
        print("  1. Cleanup Old Data")
        print("  2. Force Retrain Models (90 Days Data)")
        print("  3. Validate All Pending")
        print("  4. Validate by Category")
        print("  5. Reset Statistics")
        print("  6. Export Predictions CSV")
        print("  7. Show Disk Usage")
        print("  0. Exit")
        print("=" * 80)
        
        choice = input("\nSelect option (0-7): ").strip()
        
        if choice == '1':
            days = input("Delete data older than how many days? (default 30): ").strip()
            days = int(days) if days else 30
            cleanup_old_data(days)
            
        elif choice == '2':
            force_retrain_models()
            
        elif choice == '3':
            validate_all_pending()
            
        elif choice == '4':
            print("\nCategories:")
            print("  1. ultra_short")
            print("  2. short")
            print("  3. medium")
            print("  4. long")
            cat_choice = input("Select category (1-4): ").strip()
            categories = ['ultra_short', 'short', 'medium', 'long']
            if cat_choice in ['1', '2', '3', '4']:
                category = categories[int(cat_choice) - 1]
                validate_category(category)
                
        elif choice == '5':
            reset_statistics()
            
        elif choice == '6':
            days = input("Export last how many days? (default 7): ").strip()
            days = int(days) if days else 7
            export_predictions_csv(days)
            
        elif choice == '7':
            show_disk_usage()
            
        elif choice == '0':
            print("\n👋 Goodbye!")
            break
            
        else:
            print("\n❌ Invalid choice")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'cleanup':
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            cleanup_old_data(days)
        elif command == 'retrain':
            force_retrain_models()
        elif command == 'validate':
            validate_all_pending()
        elif command == 'validate-category':
            category = sys.argv[2] if len(sys.argv) > 2 else 'short'
            validate_category(category)
        elif command == 'export':
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
            export_predictions_csv(days)
        else:
            print("Usage: python maintenance.py [cleanup|retrain|validate|validate-category|export]")
    else:
        main()