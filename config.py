"""
Configuration file for Bitcoin Predictor Automation
Enhanced with complex timeframe strategies
"""

# Firebase Configuration
FIREBASE_CONFIG = {
    'credentials_path': 'service-account.json',
    'database_url': 'https://stc-autotrade-18f67.firebaseio.com',
    'max_retries': 5,
    'retry_delay': 5,
    'connection_timeout': 30
}

# Enhanced Prediction Configuration with Multiple Timeframe Strategies
PREDICTION_CONFIG = {
    # Ultra Short-term (Scalping) - 1-5 minutes
    'ultra_short_timeframes': [1, 2, 3, 5],
    
    # Short-term (Day Trading) - 10-60 minutes
    'short_timeframes': [10, 15, 20, 30, 45, 60],
    
    # Medium-term (Swing Trading) - 2-12 hours
    'medium_timeframes': [120, 180, 240, 360, 480, 720],
    
    # Long-term (Position Trading) - 1-7 days
    'long_timeframes': [1440, 2880, 4320, 5760, 7200, 10080],
    
    # Combined all timeframes
    'all_timeframes': [
        1, 2, 3, 5, 10, 15, 20, 30, 45, 60,  # Ultra & Short
        120, 180, 240, 360, 480, 720,        # Medium
        1440, 2880, 4320, 5760, 7200, 10080  # Long
    ],
    
    # Active timeframes (customize based on strategy)
    'active_timeframes': [
        5, 15, 30, 60,           # Short-term
        240, 480, 720,           # Medium-term
        1440, 2880, 4320         # Long-term
    ],
    
    # Timeframe intervals (when to predict each type)
    'prediction_intervals': {
        'ultra_short': 60,    # Every 1 minute
        'short': 180,         # Every 3 minutes
        'medium': 300,        # Every 5 minutes
        'long': 600,          # Every 10 minutes
        'all': 300            # Default: every 5 minutes
    },
    
    # Timeframe weights for ensemble predictions
    'timeframe_weights': {
        1: 0.5,      # Ultra short: lower weight (more noise)
        5: 0.7,      
        15: 0.85,    
        30: 0.9,     
        60: 0.95,    # Short: higher weight
        240: 1.0,    # Medium: full weight
        720: 1.0,    
        1440: 0.95,  # Long: slightly lower (less frequent updates)
        2880: 0.9,
        4320: 0.85
    },
    
    # Minimum confidence thresholds per timeframe
    'min_confidence': {
        'ultra_short': 60,  # Higher threshold for very short predictions
        'short': 55,        
        'medium': 50,       
        'long': 45          # Lower threshold for long predictions
    },
    
    # Data requirements per timeframe category
    'data_requirements': {
        'ultra_short': {
            'days': 3,
            'interval': 'minute',
            'min_points': 500
        },
        'short': {
            'days': 7,
            'interval': 'hour',
            'min_points': 300
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
    
    # Validation and health checks
    'validation_check_interval': 60,
    'health_check_interval': 300,
    'max_consecutive_failures': 5,
    'failure_backoff_multiplier': 2,
    
    # Smart scheduling
    'enable_smart_scheduling': True,
    'priority_timeframes': [5, 15, 60, 240, 1440],  # Always predict these
    'skip_low_volatility': True,     # Skip predictions during low volatility
    'min_volatility_threshold': 0.5,  # Minimum volatility % to predict
}

# Data Configuration
DATA_CONFIG = {
    'cryptocompare_api_key': "ffb687da5df95e3406d379e05a57507512343439f68e01476dd6a97894818d3b",
    'data_retention_days': 30,
    'min_data_points': 200,
    'cache_ttl': 120,
    'api_fallback_intervals': ['hour', 'day'],
    
    # Enhanced data fetching
    'fetch_strategies': {
        'ultra_short': {
            'interval': 'minute',
            'days': 2,
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

# Model Configuration
MODEL_CONFIG = {
    'lstm': {
        'epochs': 50,
        'batch_size': 32,
        'sequence_length': 60,
        'patience': 10,
        # Different configs for different timeframes
        'ultra_short_sequence': 30,
        'short_sequence': 60,
        'medium_sequence': 100,
        'long_sequence': 150
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
    'backup_path': 'models_backup/',
    
    # Separate models for different timeframe categories
    'use_specialized_models': True,
    'model_categories': ['ultra_short', 'short', 'medium', 'long']
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_file': 'btc_predictor_automation.log',
    'error_log_file': 'btc_predictor_errors.log',
    'max_log_size': 10 * 1024 * 1024,
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
    'rate_limit_delay': 1,
    'connection_pool_size': 10
}

# System Health Configuration
HEALTH_CONFIG = {
    'max_memory_mb': 2048,
    'max_cpu_percent': 80,
    'disk_space_min_gb': 1,
    'enable_watchdog': True,
    'watchdog_timeout': 600,
    'auto_restart_on_error': True,
    'max_auto_restarts': 3,
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
    'garbage_collection_interval': 3600,
    'enable_swap_monitoring': True
}

# Trading Strategy Configuration
STRATEGY_CONFIG = {
    # Multi-timeframe analysis
    'enable_mtf_analysis': True,
    'mtf_confirmation_required': 3,  # Need 3+ timeframes to agree
    
    # Timeframe correlations
    'correlation_timeframes': {
        5: [15, 30],          # 5min checks 15min and 30min
        15: [30, 60],         # 15min checks 30min and 60min
        30: [60, 240],        # 30min checks 1h and 4h
        60: [240, 720],       # 1h checks 4h and 12h
        240: [720, 1440],     # 4h checks 12h and 24h
        720: [1440, 2880],    # 12h checks 1d and 2d
        1440: [2880, 4320]    # 1d checks 2d and 3d
    },
    
    # Volatility-based adjustments
    'volatility_adjustments': {
        'high': {  # Volatility > 3%
            'confidence_multiplier': 0.9,
            'prefer_timeframes': [5, 15, 30]
        },
        'medium': {  # Volatility 1-3%
            'confidence_multiplier': 1.0,
            'prefer_timeframes': [60, 240, 720]
        },
        'low': {  # Volatility < 1%
            'confidence_multiplier': 1.1,
            'prefer_timeframes': [1440, 2880]
        }
    },
    
    # Time-of-day strategy
    'time_based_strategy': {
        'asian_session': {  # 00:00-09:00 WIB
            'active_timeframes': [15, 30, 60, 240],
            'volume_threshold': 0.7
        },
        'european_session': {  # 14:00-23:00 WIB
            'active_timeframes': [5, 15, 30, 60],
            'volume_threshold': 0.5
        },
        'american_session': {  # 20:00-05:00 WIB
            'active_timeframes': [5, 15, 30, 60, 240],
            'volume_threshold': 0.5
        },
        'weekend': {  # Saturday-Sunday
            'active_timeframes': [60, 240, 720, 1440],
            'volume_threshold': 0.8
        }
    }
}

# Helper function to get timeframe category
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
    return 'short'  # Default

# Helper function to get timeframe label
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

# Helper function to get recommended data
def get_data_config_for_timeframe(timeframe_minutes):
    """Get recommended data configuration for a timeframe"""
    category = get_timeframe_category(timeframe_minutes)
    return PREDICTION_CONFIG['data_requirements'].get(category, {
        'days': 7,
        'interval': 'hour',
        'min_points': 200
    })