"""
Maintenance and Cleanup Utilities for Bitcoin Predictor
"""

import sys
import os
from datetime import datetime, timedelta
import logging
from firebase_manager import FirebaseManager
from btc_predictor_automated import BitcoinMLPredictor, get_bitcoin_data_realtime, add_technical_indicators

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def cleanup_old_data(days=30):
    """
    Clean up data older than specified days
    """
    print_header(f"🗑️  Cleaning Up Data Older Than {days} Days")
    
    try:
        fb = FirebaseManager()
        
        print(f"\nStarting cleanup...")
        fb.cleanup_old_data(days)
        
        print(f"\n✅ Cleanup completed successfully")
        
    except Exception as e:
        print(f"\n❌ Cleanup failed: {e}")


def force_retrain_models():
    """
    Force retrain all ML models
    """
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
    """
    Validate all pending predictions manually
    """
    print_header("✅ Validating All Pending Predictions")
    
    try:
        from btc_predictor_automated import get_current_btc_price
        
        fb = FirebaseManager()
        
        print("\n📋 Getting unvalidated predictions...")
        predictions = fb.get_unvalidated_predictions()
        
        if not predictions:
            print("✅ No predictions to validate")
            return
        
        print(f"Found {len(predictions)} predictions to validate")
        
        print("\n💰 Getting current BTC price...")
        current_price = get_current_btc_price()
        
        if not current_price:
            print("❌ Cannot get current price")
            return
        
        print(f"Current price: ${current_price:,.2f}")
        
        print("\n🔍 Validating predictions...")
        validated = 0
        
        for pred in predictions:
            doc_id = pred['doc_id']
            predicted_price = pred['predicted_price']
            trend = pred['trend']
            timeframe = pred['timeframe_minutes']
            
            success = fb.validate_prediction(doc_id, current_price, predicted_price, trend)
            
            if success:
                validated += 1
        
        print(f"\n✅ Validated {validated}/{len(predictions)} predictions")
        
        # Update statistics
        print("\n📊 Updating statistics...")
        fb.get_statistics(days=7)
        
        print("✅ Statistics updated")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")


def reset_statistics():
    """
    Recalculate and reset all statistics
    """
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
        for tf in [15, 30, 60, 240, 720, 1440]:
            stats = fb.get_statistics(timeframe_minutes=tf, days=30)
            if stats and stats['total_predictions'] > 0:
                fb.save_statistics(stats)
                print(f"✅ {tf}min stats: {stats['win_rate']:.1f}% win rate")
        
        print("\n✅ Statistics reset completed")
        
    except Exception as e:
        print(f"\n❌ Reset failed: {e}")


def export_predictions_csv(days=7, filename=None):
    """
    Export predictions to CSV file
    """
    print_header(f"📤 Exporting Predictions (Last {days} Days)")
    
    try:
        import pandas as pd
        fb = FirebaseManager()
        
        if filename is None:
            filename = f"predictions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
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
            
            predictions.append({
                'timestamp': data.get('prediction_time'),
                'timeframe_minutes': data.get('timeframe_minutes'),
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
        df.to_csv(filename, index=False)
        
        print(f"\n✅ Exported {len(predictions)} predictions to {filename}")
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")


def backup_models(backup_name=None):
    """
    Backup trained models
    """
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
    """
    Restore models from backup
    """
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
    """
    Show disk usage statistics
    """
    print_header("💿 Disk Usage")
    
    try:
        import shutil
        
        directories = {
            'models': 'models',
            'logs': 'logs'
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
    print_header("🔧 Bitcoin Predictor - Maintenance Tools")
    
    while True:
        print("\n" + "=" * 80)
        print("MENU:")
        print("  1. Cleanup Old Data")
        print("  2. Force Retrain Models")
        print("  3. Validate All Pending")
        print("  4. Reset Statistics")
        print("  5. Export Predictions to CSV")
        print("  6. Backup Models")
        print("  7. Restore Models")
        print("  8. Show Disk Usage")
        print("  0. Exit")
        print("=" * 80)
        
        choice = input("\nSelect option (0-8): ").strip()
        
        if choice == '1':
            days = input("Delete data older than how many days? (default 30): ").strip()
            days = int(days) if days else 30
            cleanup_old_data(days)
        elif choice == '2':
            force_retrain_models()
        elif choice == '3':
            validate_all_pending()
        elif choice == '4':
            reset_statistics()
        elif choice == '5':
            days = input("Export last how many days? (default 7): ").strip()
            days = int(days) if days else 7
            export_predictions_csv(days)
        elif choice == '6':
            backup_models()
        elif choice == '7':
            backup_dir = input("Enter backup directory name: ").strip()
            if backup_dir:
                restore_models(backup_dir)
        elif choice == '8':
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
        elif command == 'export':
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
            export_predictions_csv(days)
        elif command == 'backup':
            backup_models()
        else:
            print("Usage: python maintenance.py [cleanup|retrain|validate|export|backup]")
    else:
        main()