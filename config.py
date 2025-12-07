"""
Configuration file for Bitcoin Predictor - FIXED VERSION FOR ULTRA SHORT
Optimized for CPU-only VPS with special handling for 5-minute predictions
"""

# Firebase Configuration
FIREBASE_CONFIG = {
    'credentials_path': 'service-account.json',
    'database_url': 'https://stc-autotrade-18f67.firebaseio.com',
    'max_retries': 5,
    'retry_delay': 5,
    'connection_timeout': 30
}

# Prediction Configuration - FIXED for ultra short
PREDICTION_CONFIG = {
    # Ultra Short-term (Scalping) - 1-5 minutes
    'ultra_short_timeframes': [1, 2, 3, 5],
    
    # Short-term (Day Trading) - 10-60 minutes
    'short_timeframes': [10, 15, 20, 30, 45, 60],
    
    # Medium-term (Swing Trading) - 2-12 hours
    'medium_timeframes': [120, 180, 240, 360, 480, 720],
    
    # Long-term (Position Trading) - 1-7 days
    'long_timeframes': [1440, 2880, 4320, 5760, 7200, 10080],
    
    # Active timeframes - ADJUSTED
    'active_timeframes': [
        5, 15, 30, 60,           # Include 5 min for testing
        240, 720,
        1440
    ],
    
    'all_timeframes': [
        1, 2, 3, 5, 10, 15, 20, 30, 45, 60,
        120, 180, 240, 360, 480, 720,
        1440, 2880, 4320, 5760, 7200, 10080
    ],
    
    # Priority timeframes
    'priority_timeframes': [15, 60, 240, 1440],
    
    # Prediction intervals
    'prediction_intervals': {
        'ultra_short': 120,      # Every 2 minutes for ultra short
        'short': 180,
        'medium': 300,
        'long': 600,
        'all': 300
    },
    
    # Timeframe weights - ADJUSTED for ultra short
    'timeframe_weights': {
        1: 0.4,                  # Lower weight for very short
        2: 0.45,
        3: 0.5,
        5: 0.55,                 # Slightly increased for 5 min
        15: 0.85,
        30: 0.9,
        60: 0.95,
        240: 1.0,
        720: 1.0,
        1440: 0.95,
        2880: 0.9,
        4320: 0.85
    },
    
    # Minimum confidence - ADJUSTED
    'min_confidence': {
        'ultra_short': 55,       # Lowered from 60
        'short': 55,
        'medium': 50,
        'long': 45
    },
    
    # Data requirements - CRITICAL FIX
    'data_requirements': {
        'ultra_short': {
            'days': 1,           # REDUCED from 2 days
            'interval': 'minute',
            'min_points': 200    # REDUCED from 400
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
            'days': 30,
            'interval': 'day',
            'min_points': 100
        }
    },
    
    'validation_check_interval': 60,
    'health_check_interval': 300,
    'max_consecutive_failures': 5,
    'failure_backoff_multiplier': 2,
    
    'enable_smart_scheduling': True,
    'skip_low_volatility': False,  # Don't skip for ultra short
    'min_volatility_threshold': 0.5,
}

# Data Configuration
DATA_CONFIG = {
    'cryptocompare_api_key': "ffb687da5df95e3406d379e05a57507512343439f68e01476dd6a97894818d3b",
    'data_retention_days': 30,
    'min_data_points': 150,
    'cache_ttl': 60,               # REDUCED to 1 minute for ultra short
    'api_fallback_intervals': ['hour', 'day'],
    
    'fetch_strategies': {
        'ultra_short': {
            'interval': 'minute',
            'days': 1,             # FIXED: Only 1 day
            'priority': 1
        },
        'short': {
            'interval': 'hour',
            'days': 7,
            'priority': 2
        },
        'medium': {
            'interval': 'hour',
            'days': 14,
            'priority': 3
        },
        'long': {
            'interval': 'day',
            'days': 60,
            'priority': 4
        }
    }
}

# Model Configuration - CRITICAL FIX for ultra short
MODEL_CONFIG = {
    'lstm': {
        'epochs': 30,
        'batch_size': 64,
        'sequence_length': 40,
        'patience': 8,
        'ultra_short_sequence': 15,  # REDUCED from 20
        'short_sequence': 40,
        'medium_sequence': 60,
        'long_sequence': 80
    },
    'rf': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 10
    },
    'gb': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 4
    },
    'auto_retrain_interval': 86400,
    'model_save_path': 'models/',
    'backup_path': 'models_backup/',
    
    'use_specialized_models': False,
    'model_categories': ['ultra_short', 'short', 'medium', 'long']
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_file': 'btc_predictor_automation.log',
    'error_log_file': 'btc_predictor_errors.log',
    'max_log_size': 10 * 1024 * 1024,
    'backup_count': 3,
    'log_level': 'INFO',
    'console_output': True
}

# API Configuration
API_CONFIG = {
    'timeout': 15,
    'max_retries': 5,
    'retry_delay': 2,
    'exponential_backoff': True,
    'rate_limit_delay': 1,
    'connection_pool_size': 5
}

# System Health Configuration
HEALTH_CONFIG = {
    'max_memory_mb': 2048,
    'max_cpu_percent': 90,
    'disk_space_min_gb': 1,
    'enable_watchdog': True,
    'watchdog_timeout': 900,
    'auto_restart_on_error': True,
    'max_auto_restarts': 10,
    'health_check_interval': 300
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
    'timeframe_stats': 'timeframe_statistics'
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

# Trading Strategy Configuration - ADJUSTED for ultra short
STRATEGY_CONFIG = {
    'enable_mtf_analysis': True,
    'mtf_confirmation_required': 2,
    
    'correlation_timeframes': {
        5: [15, 30],             # 5 min looks at 15 and 30 min
        15: [30, 60],
        30: [60, 240],
        60: [240, 720],
        240: [720, 1440],
        720: [1440, 2880],
        1440: [2880, 4320]
    },
    
    'volatility_adjustments': {
        'high': {
            'confidence_multiplier': 0.95,  # INCREASED from 0.9
            'prefer_timeframes': [5, 15, 30]
        },
        'medium': {
            'confidence_multiplier': 1.0,
            'prefer_timeframes': [60, 240, 720]
        },
        'low': {
            'confidence_multiplier': 1.1,
            'prefer_timeframes': [1440, 2880]
        }
    },
    
    'time_based_strategy': {
        'asian_session': {
            'active_timeframes': [5, 15, 30, 60, 240],
            'volume_threshold': 0.7
        },
        'european_session': {
            'active_timeframes': [5, 15, 30, 60],
            'volume_threshold': 0.5
        },
        'american_session': {
            'active_timeframes': [5, 15, 30, 60, 240],
            'volume_threshold': 0.5
        },
        'weekend': {
            'active_timeframes': [60, 240, 720, 1440],
            'volume_threshold': 0.8
        }
    }
}

# Helper functions
def get_timeframe_category(minutes):
    """Get category for a timeframe"""
    if minutes in PREDICTION_CONFIG['ultra_short_timeframes']:
        return 'ultra_short'
    elif minutes in PREDICTION_CONFIG['short_timeframes']:
        return 'short'
    elif minutes in PREDICTION_CONFIG['medium_timeframes']:
        return 'medium'
    elif minutes in PREDICTION_CONFIG['long_timeframes']:
        return 'long'
    return 'short'

def get_timeframe_label(minutes):
    """Get human-readable label for timeframe"""
    if minutes < 60:
        return f"{minutes}min"
    elif minutes < 1440:
        hours = minutes / 60
        return f"{hours:.0f}h" if hours == int(hours) else f"{hours:.1f}h"
    else:
        days = minutes / 1440
        return f"{days:.0f}d" if days == int(days) else f"{days:.1f}d"

def get_data_config_for_timeframe(timeframe_minutes):
    """Get recommended data configuration for a timeframe"""
    category = get_timeframe_category(timeframe_minutes)
    return PREDICTION_CONFIG['data_requirements'].get(category, {
        'days': 7,
        'interval': 'hour',
        'min_points': 150
    })