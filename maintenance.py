"""
Enhanced Maintenance and Cleanup Utilities for Bitcoin Predictor
Multi-Timeframe Support with Category-Based Operations
"""

import sys
import os
from datetime import datetime, timedelta
import logging
from firebase_manager import FirebaseManager
from btc_predictor_automated import (
    BitcoinMLPredictor, get_bitcoin_data_realtime, 
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
    print_header(f"🗑️  Cleaning Up Data Older Than {days} Days")
    
    try:
        fb = FirebaseManager()
        
        print(f"\nStarting cleanup...")
        fb.cleanup_old_data(days)
        
        print(f"\n✅ Cleanup completed successfully")
        
    except Exception as e:
        print(f"\n❌ Cleanup failed: {e}")


def cleanup_by_category(category, days=30):
    """Clean up predictions for specific timeframe category"""
    print_header(f"🗑️  Cleaning Up {category.upper()} Category (>{days} days)")
    
    try:
        fb = FirebaseManager()
        
        # Get timeframes for this category
        timeframes = PREDICTION_CONFIG[f'{category}_timeframes']
        
        print(f"\nTimeframes in {category}: {[get_timeframe_label(tf) for tf in timeframes]}")
        print(f"Cleaning up predictions older than {days} days...")
        
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_iso = cutoff_date.isoformat()
        
        collection = fb.firestore_db.collection('bitcoin_predictions')
        total_deleted = 0
        
        for tf in timeframes:
            # Query old predictions for this timeframe
            old_docs = collection.where('timeframe_minutes', '==', tf) \
                                .where('timestamp', '<', cutoff_iso) \
                                .limit(100).stream()
            
            count = 0
            for doc in old_docs:
                doc.reference.delete()
                count += 1
            
            if count > 0:
                total_deleted += count
                print(f"  ✅ {get_timeframe_label(tf):6}: Deleted {count} predictions")
        
        print(f"\n✅ Cleanup completed: {total_deleted} predictions deleted")
        
    except Exception as e:
        print(f"\n❌ Cleanup failed: {e}")


def force_retrain_models():
    """Force retrain all ML models"""
    print_header("🤖 Force Retraining Models")
    
    try:
        print("\n📡 Fetching training data...")
        df = get_bitcoin_data_realtime(days=30, interval='hour')
        
        if df is None or len(df) < 200:
            print("❌ Insufficient data for training")
            return
        
        print("📊 Adding technical indicators...")
        df = add_technical_indicators(df)
        
        print("🤖 Initializing predictor...")
        predictor = BitcoinMLPredictor()
        
        print("\n🚀 Starting training...")
        print("This may take 10-20 minutes...")
        
        success = predictor.train_models(df, epochs=50, batch_size=32)
        
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
    """Validate all pending predictions manually"""
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
            trend = pred['trend']
            timeframe = pred['timeframe_minutes']
            category = get_timeframe_category(timeframe)
            
            success = fb.validate_prediction(doc_id, current_price, predicted_price, trend)
            
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
    
    confirm = input("\n⚠️  This will recalculate all statistics. Continue? (yes/no): ")
    
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


def reset_category_statistics(category):
    """Reset statistics for specific category"""
    print_header(f"📊 Resetting {category.upper()} Statistics")
    
    try:
        fb = FirebaseManager()
        
        timeframes = PREDICTION_CONFIG[f'{category}_timeframes']
        
        print(f"\nTimeframes: {[get_timeframe_label(tf) for tf in timeframes]}")
        print("Recalculating...")
        
        for tf in timeframes:
            stats = fb.get_statistics(timeframe_minutes=tf, days=30)
            if stats and stats['total_predictions'] > 0:
                fb.save_statistics(stats)
                label = get_timeframe_label(tf)
                print(f"  ✅ {label:6}: {stats['win_rate']:.1f}% ({stats['total_predictions']} preds)")
        
        print("\n✅ Category statistics reset completed")
        
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
            print("⚠️  No validated predictions found")
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


def export_category_report(category, days=7):
    """Export detailed report for specific category"""
    print_header(f"📄 Exporting {category.upper()} Detailed Report")
    
    try:
        fb = FirebaseManager()
        
        filename = f"report_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        timeframes = PREDICTION_CONFIG[f'{category}_timeframes']
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"BITCOIN PREDICTOR - {category.upper()} CATEGORY REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Period: Last {days} days\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"TIMEFRAMES IN CATEGORY:\n")
            f.write(f"{[get_timeframe_label(tf) for tf in timeframes]}\n\n")
            
            total_predictions = 0
            total_wins = 0
            total_error = 0
            
            for tf in timeframes:
                stats = fb.get_statistics(timeframe_minutes=tf, days=days)
                
                if stats and stats.get('total_predictions', 0) > 0:
                    label = get_timeframe_label(tf)
                    f.write(f"\n{label} STATISTICS\n")
                    f.write("-"*80 + "\n")
                    f.write(f"Total Predictions: {stats['total_predictions']}\n")
                    f.write(f"Wins: {stats['wins']}\n")
                    f.write(f"Losses: {stats['losses']}\n")
                    f.write(f"Win Rate: {stats['win_rate']:.2f}%\n")
                    f.write(f"Average Error: ${stats['avg_error']:.2f}\n")
                    f.write(f"Average Error %: {stats['avg_error_pct']:.2f}%\n")
                    
                    total_predictions += stats['total_predictions']
                    total_wins += stats['wins']
                    total_error += stats['avg_error'] * stats['total_predictions']
            
            # Category summary
            if total_predictions > 0:
                avg_win_rate = (total_wins / total_predictions) * 100
                avg_error = total_error / total_predictions
                
                f.write(f"\n\nCATEGORY SUMMARY\n")
                f.write("="*80 + "\n")
                f.write(f"Total Predictions: {total_predictions}\n")
                f.write(f"Total Wins: {total_wins}\n")
                f.write(f"Average Win Rate: {avg_win_rate:.2f}%\n")
                f.write(f"Average Error: ${avg_error:.2f}\n")
        
        print(f"✅ Report exported to: {filename}")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")


def backup_models(backup_name=None):
    """Backup trained models"""
    print_header("💾 Backing Up Models")
    
    try:
        import shutil
        
        if backup_name is None:
            backup_name = f"models_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        models_dir = "models"
        backup_dir = backup_name
        
        if not os.path.exists(models_dir):
            print("❌ Models directory not found")
            return
        
        print(f"\n📦 Creating backup: {backup_dir}")
        
        shutil.copytree(models_dir, backup_dir)
        
        print(f"✅ Backup created successfully")
        print(f"   Location: {os.path.abspath(backup_dir)}")
        
    except Exception as e:
        print(f"\n❌ Backup failed: {e}")


def restore_models(backup_dir):
    """Restore models from backup"""
    print_header(f"📥 Restoring Models from {backup_dir}")
    
    confirm = input("\n⚠️  This will overwrite current models. Continue? (yes/no): ")
    
    if confirm.lower() != 'yes':
        print("❌ Cancelled")
        return
    
    try:
        import shutil
        
        models_dir = "models"
        
        if not os.path.exists(backup_dir):
            print(f"❌ Backup directory not found: {backup_dir}")
            return
        
        # Remove current models
        if os.path.exists(models_dir):
            print("🗑️  Removing current models...")
            shutil.rmtree(models_dir)
        
        # Restore backup
        print(f"📦 Restoring from {backup_dir}...")
        shutil.copytree(backup_dir, models_dir)
        
        print("✅ Models restored successfully")
        
    except Exception as e:
        print(f"\n❌ Restore failed: {e}")


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
    print_header("🔧 Bitcoin Predictor - Enhanced Maintenance Tools")
    
    while True:
        print("\n" + "=" * 80)
        print("MENU:")
        print("  1. Cleanup Old Data (All)")
        print("  2. Cleanup by Category")
        print("  3. Force Retrain Models")
        print("  4. Validate All Pending")
        print("  5. Validate by Category")
        print("  6. Reset Statistics (All)")
        print("  7. Reset Statistics by Category")
        print("  8. Export Predictions CSV (All)")
        print("  9. Export Predictions by Category")
        print(" 10. Export Category Report")
        print(" 11. Backup Models")
        print(" 12. Restore Models")
        print(" 13. Show Disk Usage")
        print("  0. Exit")
        print("=" * 80)
        
        choice = input("\nSelect option (0-13): ").strip()
        
        if choice == '1':
            days = input("Delete data older than how many days? (default 30): ").strip()
            days = int(days) if days else 30
            cleanup_old_data(days)
            
        elif choice == '2':
            print("\nCategories:")
            print("  1. ultra_short")
            print("  2. short")
            print("  3. medium")
            print("  4. long")
            cat_choice = input("Select category (1-4): ").strip()
            categories = ['ultra_short', 'short', 'medium', 'long']
            if cat_choice in ['1', '2', '3', '4']:
                category = categories[int(cat_choice) - 1]
                days = input("Delete data older than how many days? (default 30): ").strip()
                days = int(days) if days else 30
                cleanup_by_category(category, days)
                
        elif choice == '3':
            force_retrain_models()
            
        elif choice == '4':
            validate_all_pending()
            
        elif choice == '5':
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
                
        elif choice == '6':
            reset_statistics()
            
        elif choice == '7':
            print("\nCategories:")
            print("  1. ultra_short")
            print("  2. short")
            print("  3. medium")
            print("  4. long")
            cat_choice = input("Select category (1-4): ").strip()
            categories = ['ultra_short', 'short', 'medium', 'long']
            if cat_choice in ['1', '2', '3', '4']:
                category = categories[int(cat_choice) - 1]
                reset_category_statistics(category)
                
        elif choice == '8':
            days = input("Export last how many days? (default 7): ").strip()
            days = int(days) if days else 7
            export_predictions_csv(days)
            
        elif choice == '9':
            print("\nCategories:")
            print("  1. ultra_short")
            print("  2. short")
            print("  3. medium")
            print("  4. long")
            cat_choice = input("Select category (1-4): ").strip()
            categories = ['ultra_short', 'short', 'medium', 'long']
            if cat_choice in ['1', '2', '3', '4']:
                category = categories[int(cat_choice) - 1]
                days = input("Export last how many days? (default 7): ").strip()
                days = int(days) if days else 7
                export_predictions_csv(days, category=category)
                
        elif choice == '10':
            print("\nCategories:")
            print("  1. ultra_short")
            print("  2. short")
            print("  3. medium")
            print("  4. long")
            cat_choice = input("Select category (1-4): ").strip()
            categories = ['ultra_short', 'short', 'medium', 'long']
            if cat_choice in ['1', '2', '3', '4']:
                category = categories[int(cat_choice) - 1]
                days = input("Report for last how many days? (default 7): ").strip()
                days = int(days) if days else 7
                export_category_report(category, days)
                
        elif choice == '11':
            backup_models()
            
        elif choice == '12':
            backup_dir = input("Enter backup directory name: ").strip()
            if backup_dir:
                restore_models(backup_dir)
                
        elif choice == '13':
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
        elif command == 'cleanup-category':
            category = sys.argv[2] if len(sys.argv) > 2 else 'short'
            days = int(sys.argv[3]) if len(sys.argv) > 3 else 30
            cleanup_by_category(category, days)
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
        elif command == 'export-category':
            category = sys.argv[2] if len(sys.argv) > 2 else 'short'
            days = int(sys.argv[3]) if len(sys.argv) > 3 else 7
            export_predictions_csv(days, category=category)
        elif command == 'backup':
            backup_models()
        else:
            print("Usage: python maintenance.py [cleanup|cleanup-category|retrain|validate|validate-category|export|export-category|backup]")
    else:
        main()