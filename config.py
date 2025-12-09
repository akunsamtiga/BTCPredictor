"""
IMPROVED Configuration
CHANGES:
1. Lower confidence thresholds for more predictions
2. Better timeframe distribution
3. Optimized data requirements
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

# IMPROVED: Prediction Configuration
PREDICTION_CONFIG = {
    # Timeframes (minutes) - OPTIMIZED
    'ultra_short_timeframes': [5, 10, 15],  # Added 15
    'short_timeframes': [30, 60],  # Simplified
    'medium_timeframes': [120, 240, 480],  # Added 480 (8h)
    'long_timeframes': [720, 1440],  # Removed 2880 (too long)
    
    # Active timeframes - MORE FOCUSED
    'active_timeframes': [
        5, 15, 30,       # Ultra short & short
        60, 120,         # Short & medium
        240, 720,        # Medium & long
        1440             # Daily
    ],
    
    'priority_timeframes': [15, 60, 240, 1440],  # Added 15min
    
    # LOWERED: Minimum confidence thresholds for more predictions
    'min_confidence': {
        'ultra_short': int(os.getenv('MIN_CONFIDENCE_ULTRA_SHORT', 45)),  # Was 60
        'short': int(os.getenv('MIN_CONFIDENCE_SHORT', 40)),              # Was 50
        'medium': int(os.getenv('MIN_CONFIDENCE_MEDIUM', 38)),            # Was 45
        'long': int(os.getenv('MIN_CONFIDENCE_LONG', 35))                 # Was 40
    },
    
    # OPTIMIZED: Data requirements
    'data_requirements': {
        'ultra_short': {
            'days': 2,              # Reduced from 3
            'interval': 'minute',
            'min_points': 400       # Reduced from 500
        },
        'short': {
            'days': 5,              # Reduced from 7
            'interval': 'hour',
            'min_points': 100       # Reduced from 150
        },
        'medium': {
            'days': 10,             # Reduced from 14
            'interval': 'hour',
            'min_points': 150       # Reduced from 200
        },
        'long': {
            'days': 45,             # Reduced from 60
            'interval': 'day',
            'min_points': 90        # Reduced from 120
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
    'min_data_points': 100,  # Reduced from 150
    'cache_ttl': 300,
    'api_fallback_intervals': ['hour', 'day'],
    'enable_caching': True,
    'max_cache_size_mb': 100,
}

# IMPROVED: Model Configuration
MODEL_CONFIG = {
    'lstm': {
        'epochs': 50,  # Increased from 40
        'batch_size': 64,
        'sequence_length': 60,
        'patience': 12,  # Increased from 10
        'ultra_short_sequence': 30,
        'short_sequence': 60,
        'medium_sequence': 80,
        'long_sequence': 100
    },
    'rf': {
        'n_estimators': 200,  # Increased from 150
        'max_depth': 15,      # Increased from 12
        'min_samples_split': 5  # Decreased from 8
    },
    'gb': {
        'n_estimators': 200,     # Increased from 150
        'learning_rate': 0.1,    # Increased from 0.08
        'max_depth': 6           # Increased from 5
    },
    'auto_retrain_interval': 86400,  # 24 hours
    'model_save_path': 'models/',
    'backup_path': 'models_backup/',
    'enable_model_validation': True,
    'min_validation_score': 0.55,  # Lowered from 0.6
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
    'min_winrate_alert': 42.0,  # Lowered from 45.0
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

# IMPROVED: Trading Strategy Configuration
STRATEGY_CONFIG = {
    'enable_mtf_analysis': True,
    'mtf_confirmation_required': 2,
    
    'correlation_timeframes': {
        5: [15, 30],
        15: [30, 60],
        30: [60, 120],
        60: [120, 240],
        120: [240, 480],
        240: [480, 720],
        720: [1440],
        1440: [720]  # Daily correlates with 12h
    },
    
    'volatility_adjustments': {
        'high': {
            'confidence_multiplier': 0.92,  # Less penalty
            'prefer_timeframes': [5, 15, 30],
            'volatility_threshold': 3.0
        },
        'medium': {
            'confidence_multiplier': 1.0,
            'prefer_timeframes': [60, 120, 240],
            'volatility_threshold': 1.5
        },
        'low': {
            'confidence_multiplier': 1.08,  # More bonus
            'prefer_timeframes': [720, 1440],
            'volatility_threshold': 1.0
        }
    },
    
    # Risk Management
    'risk_management': {
        'max_daily_predictions': 150,  # Increased from 100
        'max_predictions_per_timeframe': 30,  # Increased from 20
        'cooldown_after_loss_streak': 4,  # Increased from 3
        'cooldown_duration_minutes': 20,  # Reduced from 30
    },
}

FAVORABLE_CONDITIONS = {
    # Minimum price movement (%)
    'min_trend_strength': 1.5,    # Harus ada trend minimal 1.5%
    
    # Maximum acceptable volatility (%)
    'max_volatility': 5.0,         # Volatility tidak boleh >5%
    
    # Minimum volume ratio
    'min_volume_ratio': 0.8,       # Volume harus minimal 80% dari average
    
    # RSI range (must be between these values)
    'rsi_extremes': (25, 75),      # RSI antara 25-75 (avoid overbought/oversold)
    
    # ATR stability factor
    'min_atr_stability': 0.7,      # ATR harus stabil (tidak terlalu berubah)
}

# === ENHANCED MODEL CONFIG ===
MODEL_CONFIG_ENHANCED = {
    'lstm': {
        'epochs': 50,
        'batch_size': 64,
        'patience': 15,  # Lebih sabar untuk convergence
        
        # Sequence lengths per category
        'ultra_short_sequence': 30,
        'short_sequence': 60,
        'medium_sequence': 80,
        'long_sequence': 100,
        
        # Architecture
        'use_attention': True,
        'num_attention_heads': 4,
        'use_bidirectional': True,
        'gradient_clipping': 1.0,
    },
    
    'rf': {
        'n_estimators': 300,  # Lebih banyak trees
        'max_depth': 18,
        'min_samples_split': 4,
        'min_samples_leaf': 2,
        'class_weight': 'balanced'
    },
    
    'gb': {
        'n_estimators': 300,
        'learning_rate': 0.08,
        'max_depth': 7,
        'subsample': 0.8
    },
    
    # Minimum required training data
    'min_training_samples': 1000,  # Need more data for better models
    
    # Validation
    'time_series_splits': 5,  # Cross-validation folds
    'min_validation_score': 0.65,  # Minimum acceptable score
}

# === ULTRA-STRICT CONFIDENCE THRESHOLDS ===
ULTRA_STRICT_THRESHOLDS = {
    'ultra_short': 60,  # 60%+ confidence required
    'short': 55,        # 55%+ confidence required
    'medium': 52,       # 52%+ confidence required
    'long': 50,         # 50%+ confidence required
}

# === PREDICTION FILTERS ===
PREDICTION_FILTERS = {
    # Reject if prediction change too large
    'max_change_pct': 10.0,  # Reject >10% change
    
    # Reject if prediction change too small (noise)
    'min_change_pct': 0.1,   # Reject <0.1% change
    
    # Model agreement requirement
    'require_all_models_agree': False,  # Set True for maximum strictness
    
    # Minimum model agreement (0-1)
    'min_model_agreement': 0.67,  # At least 2 out of 3 must agree
}

# === ADAPTIVE LEARNING ===
ADAPTIVE_CONFIG = {
    # Track recent performance
    'recent_predictions_window': 50,
    'recent_wins_window': 50,
    
    # Threshold adjustment based on performance
    'adjust_threshold_on_poor_performance': True,
    'poor_performance_winrate': 0.60,  # <60% is poor
    'good_performance_winrate': 0.75,  # >75% is good
    
    # Threshold adjustments
    'threshold_increase_on_poor': 5,  # +5% threshold if performing poorly
    'threshold_decrease_on_good': 3,  # -3% threshold if performing well
}

# === ENHANCED FEATURES ===
FEATURE_CONFIG = {
    # Technical indicators to calculate
    'use_multiple_rsi_periods': True,
    'rsi_periods': [7, 14, 21, 28],
    
    'use_multiple_macd_settings': True,
    'macd_settings': [(12, 26, 9), (5, 35, 5), (19, 39, 9)],
    
    'use_multiple_bb_settings': True,
    'bb_settings': [(20, 2), (20, 3), (50, 2)],
    
    'use_adx': True,
    'adx_periods': [14, 21],
    
    'use_stochastic': True,
    'stochastic_settings': [(14, 3), (21, 3), (14, 5)],
    
    'use_fibonacci': True,
    'fibonacci_windows': [20, 50],
    
    'use_williams_r': True,
    'use_keltner_channels': True,
    'use_volume_indicators': True,
    
    # Total features: ~100+
    'expected_feature_count': 100,
}

# === DEPLOYMENT SETTINGS ===
DEPLOYMENT_CONFIG = {
    # Log all filtered predictions for analysis
    'log_rejected_predictions': True,
    'rejected_predictions_file': 'logs/rejected_predictions.log',
    
    # Performance monitoring
    'track_model_performance': True,
    'performance_update_interval': 10,  # Update every 10 validations
    
    # Auto-retraining
    'retrain_on_poor_performance': True,
    'retrain_threshold_winrate': 0.55,  # Retrain if <55%
    'min_predictions_before_retrain': 100,
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
        'days': 5,
        'interval': 'hour',
        'min_points': 100
    })

def get_min_confidence(timeframe_minutes: int) -> float:
    """Get minimum confidence threshold for timeframe"""
    category = get_timeframe_category(timeframe_minutes)
    return PREDICTION_CONFIG['min_confidence'].get(category, 40.0)

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