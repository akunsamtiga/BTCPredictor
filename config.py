"""
FIXED Configuration - NO BACKTEST
Simplified and focused configuration
"""

import os
from typing import Dict, Optional
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

# Environment
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
TRADING_MODE = os.getenv('TRADING_MODE', 'paper')

# Validate critical environment
def validate_environment():
    """Validate required environment variables"""
    required = ['CRYPTOCOMPARE_API_KEY', 'FIREBASE_CREDENTIALS_PATH', 'FIREBASE_DATABASE_URL']
    missing = [var for var in required if not os.getenv(var)]
    
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")
    
    logger.info(f"✅ Environment validated: {ENVIRONMENT}, Mode: {TRADING_MODE}")

# Firebase Configuration
FIREBASE_CONFIG = {
    'credentials_path': os.getenv('FIREBASE_CREDENTIALS_PATH'),
    'database_url': os.getenv('FIREBASE_DATABASE_URL'),
    'max_retries': 5,
    'retry_delay': 5,
    'connection_timeout': 30
}

# Prediction Configuration - SIMPLIFIED
PREDICTION_CONFIG = {
    # Timeframes (minutes) - REDUCED for better accuracy
    'ultra_short_timeframes': [5, 10],
    'short_timeframes': [15, 30, 60],
    'medium_timeframes': [120, 240, 720],
    'long_timeframes': [1440, 2880],
    
    # Active timeframes - FOCUSED on reliable ones
    'active_timeframes': [
        15, 30, 60,      # Short term
        240, 720,        # Medium term
        1440             # Long term
    ],
    
    'priority_timeframes': [60, 240, 1440],
    
    # Minimum confidence thresholds - MORE STRICT
    'min_confidence': {
        'ultra_short': int(os.getenv('MIN_CONFIDENCE_ULTRA_SHORT', 70)),
        'short': int(os.getenv('MIN_CONFIDENCE_SHORT', 60)),
        'medium': int(os.getenv('MIN_CONFIDENCE_MEDIUM', 55)),
        'long': int(os.getenv('MIN_CONFIDENCE_LONG', 50))
    },
    
    # Data requirements
    'data_requirements': {
        'ultra_short': {
            'days': 3,
            'interval': 'minute',
            'min_points': 500
        },
        'short': {
            'days': 7,
            'interval': 'hour',
            'min_points': 150
        },
        'medium': {
            'days': 14,
            'interval': 'hour',
            'min_points': 200
        },
        'long': {
            'days': 60,
            'interval': 'day',
            'min_points': 120
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
    'min_data_points': 150,
    'cache_ttl': 300,
    'api_fallback_intervals': ['hour', 'day'],
    'enable_caching': True,
    'max_cache_size_mb': 100,
}

# Model Configuration
MODEL_CONFIG = {
    'lstm': {
        'epochs': 40,
        'batch_size': 64,
        'sequence_length': 60,
        'patience': 10,
        'ultra_short_sequence': 30,
        'short_sequence': 60,
        'medium_sequence': 80,
        'long_sequence': 100
    },
    'rf': {
        'n_estimators': 150,
        'max_depth': 12,
        'min_samples_split': 8
    },
    'gb': {
        'n_estimators': 150,
        'learning_rate': 0.08,
        'max_depth': 5
    },
    'auto_retrain_interval': 86400,  # 24 hours
    'model_save_path': 'models/',
    'backup_path': 'models_backup/',
    'enable_model_validation': True,
    'min_validation_score': 0.6,
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
    'min_winrate_alert': 45.0,
    'alert_on_high_memory': True,
    'alert_on_consecutive_failures': True,
    'max_consecutive_failures': 3,
}

# System Health Configuration
HEALTH_CONFIG = {
    'max_memory_mb': int(os.getenv('MAX_MEMORY_MB', 2048)),
    'max_cpu_percent': int(os.getenv('MAX_CPU_PERCENT', 90)),
    'disk_space_min_gb': 1,
    'enable_watchdog': True,
    'watchdog_timeout': 1200,
    'auto_restart_on_error': True,
    'max_auto_restarts': 5,
    'health_check_interval': 300,
    'enable_performance_monitoring': True,
}

# Monitoring Configuration
MONITORING_CONFIG = {
    'enable_prometheus': os.getenv('ENABLE_PROMETHEUS', 'false').lower() == 'true',
    'prometheus_port': int(os.getenv('PROMETHEUS_PORT', 8000)),
    'enable_sentry': os.getenv('ENABLE_SENTRY', 'false').lower() == 'true',
    'sentry_dsn': os.getenv('SENTRY_DSN'),
    'log_level': os.getenv('LOG_LEVEL', 'INFO'),
    'enable_detailed_logging': ENVIRONMENT == 'development',
}

# Cache Configuration
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
    'connection_pool_size': 5,
    'max_requests_per_minute': 100,
}

# VPS Optimization
VPS_CONFIG = {
    'enable_memory_optimization': True,
    'clear_tensorflow_session': True,
    'garbage_collection_interval': 1800,
    'enable_swap_monitoring': True,
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

# Trading Strategy Configuration
STRATEGY_CONFIG = {
    'enable_mtf_analysis': True,
    'mtf_confirmation_required': 2,
    
    'correlation_timeframes': {
        5: [15, 30],
        15: [30, 60],
        30: [60, 240],
        60: [240, 720],
        240: [720, 1440],
        720: [1440, 2880],
        1440: [2880, 4320]
    },
    
    'volatility_adjustments': {
        'high': {
            'confidence_multiplier': 0.90,
            'prefer_timeframes': [5, 15, 30],
            'volatility_threshold': 3.0
        },
        'medium': {
            'confidence_multiplier': 1.0,
            'prefer_timeframes': [60, 240, 720],
            'volatility_threshold': 1.5
        },
        'low': {
            'confidence_multiplier': 1.1,
            'prefer_timeframes': [1440, 2880],
            'volatility_threshold': 1.0
        }
    },
    
    # Risk Management
    'risk_management': {
        'max_daily_predictions': 100,
        'max_predictions_per_timeframe': 20,
        'cooldown_after_loss_streak': 3,
        'cooldown_duration_minutes': 30,
    },
}

# Helper Functions
def get_timeframe_category(minutes: int) -> str:
    """Get category for a timeframe"""
    if minutes in PREDICTION_CONFIG.get('ultra_short_timeframes', []):
        return 'ultra_short'
    elif minutes in PREDICTION_CONFIG.get('short_timeframes', []):
        return 'short'
    elif minutes in PREDICTION_CONFIG.get('medium_timeframes', []):
        return 'medium'
    elif minutes in PREDICTION_CONFIG.get('long_timeframes', []):
        return 'long'
    return 'short'

def get_timeframe_label(minutes: int) -> str:
    """Get human-readable label for timeframe"""
    if minutes < 60:
        return f"{minutes}min"
    elif minutes < 1440:
        hours = minutes / 60
        return f"{hours:.0f}h" if hours == int(hours) else f"{hours:.1f}h"
    else:
        days = minutes / 1440
        return f"{days:.0f}d" if days == int(days) else f"{days:.1f}d"

def get_data_config_for_timeframe(timeframe_minutes: int) -> Dict:
    """Get recommended data configuration for a timeframe"""
    category = get_timeframe_category(timeframe_minutes)
    return PREDICTION_CONFIG['data_requirements'].get(category, {
        'days': 7,
        'interval': 'hour',
        'min_points': 150
    })

def get_min_confidence(timeframe_minutes: int) -> float:
    """Get minimum confidence threshold for timeframe"""
    category = get_timeframe_category(timeframe_minutes)
    return PREDICTION_CONFIG['min_confidence'].get(category, 60.0)

def is_production() -> bool:
    """Check if running in production"""
    return ENVIRONMENT == 'production'

def is_paper_trading() -> bool:
    """Check if in paper trading mode"""
    return TRADING_MODE == 'paper'

def get_config_summary() -> Dict:
    """Get configuration summary for logging"""
    return {
        'environment': ENVIRONMENT,
        'trading_mode': TRADING_MODE,
        'active_timeframes': PREDICTION_CONFIG['active_timeframes'],
        'alerts_enabled': ALERT_CONFIG['enable_alerts'],
        'redis_enabled': CACHE_CONFIG['enable_redis'],
        'prometheus_enabled': MONITORING_CONFIG['enable_prometheus'],
    }

# Validate on import
try:
    validate_environment()
    logger.info(f"✅ Configuration loaded: {get_config_summary()}")
except EnvironmentError as e:
    logger.error(f"❌ Configuration error: {e}")
    if is_production():
        raise