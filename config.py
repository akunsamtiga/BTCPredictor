"""
Configuration file for Bitcoin Predictor Automation
Enhanced with error handling and retry mechanisms
"""

# Firebase Configuration
FIREBASE_CONFIG = {
    'credentials_path': 'service-account.json',
    'database_url': 'https://stc-autotrade-18f67.firebaseio.com',
    'max_retries': 5,
    'retry_delay': 5,  # seconds
    'connection_timeout': 30
}

# Prediction Configuration
PREDICTION_CONFIG = {
    'timeframes': [15, 30, 60, 240, 720, 1440],
    'prediction_interval': 300,  # 5 minutes
    'data_fetch_interval': 60,
    'validation_check_interval': 60,
    'health_check_interval': 300,  # 5 minutes - FIXED
    'max_consecutive_failures': 5,  # Stop after 5 consecutive failures
    'failure_backoff_multiplier': 2  # Exponential backoff
}

# Data Configuration
DATA_CONFIG = {
    'cryptocompare_api_key': "ffb687da5df95e3406d379e05a57507512343439f68e01476dd6a97894818d3b",
    'data_retention_days': 30,
    'min_data_points': 200,
    'cache_ttl': 120,  # 2 minutes
    'api_fallback_intervals': ['hour', 'day'],  # Fallback if minute fails
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
    'auto_retrain_interval': 86400,
    'model_save_path': 'models/',
    'backup_path': 'models_backup/'
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_file': 'btc_predictor_automation.log',
    'error_log_file': 'btc_predictor_errors.log',
    'max_log_size': 10 * 1024 * 1024,  # 10 MB
    'backup_count': 5,
    'log_level': 'INFO',
    'console_output': True
}

# API Configuration
API_CONFIG = {
    'timeout': 15,
    'max_retries': 5,
    'retry_delay': 2,
    'exponential_backoff': True,
    'rate_limit_delay': 1,  # seconds between requests
    'connection_pool_size': 10
}

# System Health Configuration
HEALTH_CONFIG = {
    'max_memory_mb': 2048,  # Alert if memory exceeds 2GB
    'max_cpu_percent': 80,
    'disk_space_min_gb': 1,
    'enable_watchdog': True,
    'watchdog_timeout': 600,  # 10 minutes
    'auto_restart_on_error': True,
    'max_auto_restarts': 3,
    'health_check_interval': 300  # 5 minutes - FIXED
}

# Firebase Collections
FIREBASE_COLLECTIONS = {
    'predictions': 'bitcoin_predictions',
    'validation': 'prediction_validation',
    'statistics': 'prediction_statistics',
    'raw_data': 'bitcoin_data',
    'model_performance': 'model_performance',
    'system_health': 'system_health',
    'error_logs': 'error_logs'
}

# VPS Optimization
VPS_CONFIG = {
    'enable_memory_optimization': True,
    'clear_tensorflow_session': True,
    'garbage_collection_interval': 3600,  # 1 hour
    'enable_swap_monitoring': True
}