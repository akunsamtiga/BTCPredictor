"""
Configuration for Bitcoin Predictor
OPTIMIZED for consistent predictions with quality
"""

import os
from typing import Dict
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

# Environment
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
TRADING_MODE = os.getenv('TRADING_MODE', 'paper')

def validate_environment():
    """Validate required environment variables"""
    required = ['CRYPTOCOMPARE_API_KEY', 'FIREBASE_CREDENTIALS_PATH', 'FIREBASE_DATABASE_URL']
    missing = [var for var in required if not os.getenv(var)]
    
    if missing:
        raise EnvironmentError(f"Missing: {', '.join(missing)}")
    
    logger.info(f"✅ Environment: {ENVIRONMENT}, Mode: {TRADING_MODE}")

# Firebase
FIREBASE_CONFIG = {
    'credentials_path': os.getenv('FIREBASE_CREDENTIALS_PATH'),
    'database_url': os.getenv('FIREBASE_DATABASE_URL'),
    'max_retries': 5,
    'retry_delay': 5,
    'connection_timeout': 30
}

# Prediction Configuration
PREDICTION_CONFIG = {
    # Timeframes by category (minutes)
    'ultra_short_timeframes': [5, 10, 15],
    'short_timeframes': [30, 60],
    'medium_timeframes': [120, 240, 480],
    'long_timeframes': [720, 1440],
    
    # Active timeframes - ALL WILL PREDICT
    'active_timeframes': [
        5, 10, 15,      # Ultra short
        30, 60,         # Short
        120, 240, 480,  # Medium
        720, 1440       # Long
    ],
    
    # OPTIMIZED: Lower thresholds for consistent predictions
    'min_confidence': {
        'ultra_short': int(os.getenv('MIN_CONFIDENCE_ULTRA_SHORT', 40)),
        'short': int(os.getenv('MIN_CONFIDENCE_SHORT', 38)),
        'medium': int(os.getenv('MIN_CONFIDENCE_MEDIUM', 35)),
        'long': int(os.getenv('MIN_CONFIDENCE_LONG', 32))
    },
    
    # Data requirements - OPTIMIZED for all timeframes
    'data_requirements': {
        'ultra_short': {
            'days': 2,
            'interval': 'minute',
            'min_points': 200  # Reduced from 400
        },
        'short': {
            'days': 3,
            'interval': 'hour',
            'min_points': 60   # Reduced from 100
        },
        'medium': {
            'days': 7,         # Reduced from 10
            'interval': 'hour',
            'min_points': 100  # Reduced from 150
        },
        'long': {
            'days': 30,        # Reduced from 45
            'interval': 'day',
            'min_points': 50   # Reduced from 90
        }
    },
    
    'validation_check_interval': 60,
    'health_check_interval': 300,
    'max_consecutive_failures': 3,
}

# Data Configuration
DATA_CONFIG = {
    'cryptocompare_api_key': os.getenv('CRYPTOCOMPARE_API_KEY'),
    'data_retention_days': 30,
    'min_data_points': 100,
    'cache_ttl': 300,
    'api_fallback_intervals': ['hour', 'day'],
    'enable_caching': True,
    'max_cache_size_mb': 100,
}

# Model Configuration
MODEL_CONFIG = {
    'lstm': {
        'epochs': 50,
        'batch_size': 64,
        'sequence_length': 60,
        'patience': 15,
        'ultra_short_sequence': 30,
        'short_sequence': 60,
        'medium_sequence': 80,
        'long_sequence': 100
    },
    'rf': {
        'n_estimators': 300,
        'max_depth': 18,
        'min_samples_split': 4,
        'min_samples_leaf': 2
    },
    'gb': {
        'n_estimators': 300,
        'learning_rate': 0.08,
        'max_depth': 7
    },
    'auto_retrain_interval': 86400,  # 24 hours
    'model_save_path': 'models/',
    'backup_path': 'models_backup/',
    'enable_model_validation': True,
    'min_validation_score': 0.55,
}

# Alert Configuration
ALERT_CONFIG = {
    'enable_alerts': os.getenv('ENABLE_ALERTS', 'false').lower() == 'true',
    'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
    'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID'),
    'alert_email': os.getenv('ALERT_EMAIL'),
    'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
    'smtp_port': int(os.getenv('SMTP_PORT', 587)),
    'smtp_username': os.getenv('SMTP_USERNAME'),
    'smtp_password': os.getenv('SMTP_PASSWORD'),
    
    'alert_on_low_winrate': True,
    'min_winrate_alert': 42.0,
    'alert_on_high_memory': True,
    'alert_on_consecutive_failures': True,
    'max_consecutive_failures': 3,
}

# System Health
HEALTH_CONFIG = {
    'max_memory_mb': int(os.getenv('MAX_MEMORY_MB', 2048)),
    'max_cpu_percent': int(os.getenv('MAX_CPU_PERCENT', 90)),
    'disk_space_min_gb': 1,
    'enable_watchdog': True,
    'watchdog_timeout': 1200,
    'auto_restart_on_error': True,
    'max_auto_restarts': 5,
    'health_check_interval': 300,
}

# Monitoring
MONITORING_CONFIG = {
    'enable_prometheus': os.getenv('ENABLE_PROMETHEUS', 'false').lower() == 'true',
    'prometheus_port': int(os.getenv('PROMETHEUS_PORT', 8000)),
    'enable_sentry': os.getenv('ENABLE_SENTRY', 'false').lower() == 'true',
    'sentry_dsn': os.getenv('SENTRY_DSN'),
    'log_level': os.getenv('LOG_LEVEL', 'INFO'),
    'enable_detailed_logging': ENVIRONMENT == 'development',
}

# Cache
CACHE_CONFIG = {
    'enable_redis': os.getenv('ENABLE_REDIS', 'false').lower() == 'true',
    'redis_host': os.getenv('REDIS_HOST', 'localhost'),
    'redis_port': int(os.getenv('REDIS_PORT', 6379)),
    'redis_password': os.getenv('REDIS_PASSWORD'),
    'redis_db': 0,
    'cache_prefix': 'btc_predictor:',
    'default_ttl': 300,
}

# API Configuration
API_CONFIG = {
    'timeout': 15,
    'max_retries': 5,
    'retry_delay': 2,
    'exponential_backoff': True,
    'rate_limit_delay': 1,
}

# VPS Optimization
VPS_CONFIG = {
    'enable_memory_optimization': True,
    'clear_tensorflow_session': True,
    'garbage_collection_interval': 1800,
    'tf_num_threads': 2,
    'tf_inter_threads': 1,
    'tf_intra_threads': 2,
}

# Firebase Collections
FIREBASE_COLLECTIONS = {
    'predictions': 'bitcoin_predictions',
    'validation': 'prediction_validation',
    'statistics': 'prediction_statistics',
    'raw_data': 'bitcoin_data',
    'model_performance': 'model_performance',
    'system_health': 'system_health',
    'error_logs': 'error_logs',
    'alerts': 'alerts',
}

# Strategy Configuration
STRATEGY_CONFIG = {
    'risk_management': {
        'max_daily_predictions': 200,
        'max_predictions_per_timeframe': 50,
        'cooldown_after_loss_streak': 5,
        'cooldown_duration_minutes': 30,
    },
}

# Helper Functions
def get_timeframe_category(minutes: int) -> str:
    """Get category for timeframe"""
    if minutes in PREDICTION_CONFIG['ultra_short_timeframes']:
        return 'ultra_short'
    elif minutes in PREDICTION_CONFIG['short_timeframes']:
        return 'short'
    elif minutes in PREDICTION_CONFIG['medium_timeframes']:
        return 'medium'
    elif minutes in PREDICTION_CONFIG['long_timeframes']:
        return 'long'
    return 'short'

def get_timeframe_label(minutes: int) -> str:
    """Get readable label"""
    if minutes < 60:
        return f"{minutes}min"
    elif minutes < 1440:
        hours = minutes / 60
        return f"{hours:.0f}h" if hours == int(hours) else f"{hours:.1f}h"
    else:
        days = minutes / 1440
        return f"{days:.0f}d" if days == int(days) else f"{days:.1f}d"

def get_data_config_for_timeframe(timeframe_minutes: int) -> Dict:
    """Get data config for timeframe"""
    category = get_timeframe_category(timeframe_minutes)
    return PREDICTION_CONFIG['data_requirements'].get(category, {
        'days': 5,
        'interval': 'hour',
        'min_points': 100
    })

def get_min_confidence(timeframe_minutes: int) -> float:
    """Get minimum confidence"""
    category = get_timeframe_category(timeframe_minutes)
    return PREDICTION_CONFIG['min_confidence'].get(category, 38.0)

def is_production() -> bool:
    """Check if production"""
    return ENVIRONMENT == 'production'

def get_config_summary() -> Dict:
    """Get config summary"""
    return {
        'environment': ENVIRONMENT,
        'trading_mode': TRADING_MODE,
        'active_timeframes': len(PREDICTION_CONFIG['active_timeframes']),
        'alerts_enabled': ALERT_CONFIG['enable_alerts'],
        'redis_enabled': CACHE_CONFIG['enable_redis'],
    }

# Validate on import
try:
    validate_environment()
    logger.info(f"✅ Configuration loaded")
except EnvironmentError as e:
    logger.error(f"❌ Configuration error: {e}")
    if is_production():
        raise