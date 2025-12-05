"""
Configuration file for Bitcoin Predictor Automation
"""

# Firebase Configuration
FIREBASE_CONFIG = {
    'credentials_path': 'service-account.json',  # Path to your Firebase credentials
    'database_url': 'https://stc-autotrade-18f67.firebaseio.com'
}

# Prediction Configuration
PREDICTION_CONFIG = {
    'timeframes': [15, 30, 60, 240, 720, 1440],  # Minutes: 15min, 30min, 1h, 4h, 12h, 24h
    'prediction_interval': 300,  # Run prediction every 5 minutes (300 seconds)
    'data_fetch_interval': 60,  # Fetch new data every 60 seconds
    'validation_check_interval': 60,  # Check prediction results every 60 seconds
}

# Data Configuration
DATA_CONFIG = {
    'cryptocompare_api_key': "ffb687da5df95e3406d379e05a57507512343439f68e01476dd6a97894818d3b",  # Optional: Add your API key for higher rate limits
    'data_retention_days': 30,  # Keep data for 30 days
    'min_data_points': 200,  # Minimum data points required for prediction
}

# Model Configuration
MODEL_CONFIG = {
    'lstm': {
        'epochs': 50,
        'batch_size': 32,
        'sequence_length': 60,
        'patience': 10
    },
    'rf': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5
    },
    'gb': {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 5
    },
    'auto_retrain_interval': 86400,  # Retrain models every 24 hours
    'model_save_path': 'models/'
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_file': 'btc_predictor_automation.log',
    'max_log_size': 10 * 1024 * 1024,  # 10 MB
    'backup_count': 5,
    'log_level': 'INFO'
}

# API Configuration
API_CONFIG = {
    'timeout': 15,
    'max_retries': 3,
    'retry_delay': 2
}

# Firebase Collections
FIREBASE_COLLECTIONS = {
    'predictions': 'bitcoin_predictions',
    'validation': 'prediction_validation',
    'statistics': 'prediction_statistics',
    'raw_data': 'bitcoin_data',
    'model_performance': 'model_performance'
}