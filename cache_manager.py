"""
Cache Manager for Bitcoin Predictor
Redis-based caching with fallback to memory cache
"""

import logging
import pickle
import json
from typing import Any, Optional
from datetime import timedelta
from functools import wraps

from config import CACHE_CONFIG

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching with Redis and memory fallback"""
    
    def __init__(self):
        self.redis_client = None
        self.memory_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        if CACHE_CONFIG.get('enable_redis'):
            self._initialize_redis()
        else:
            logger.info("ℹ️ Redis disabled, using memory cache")
    
    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            import redis
            
            self.redis_client = redis.Redis(
                host=CACHE_CONFIG.get('redis_host', 'localhost'),
                port=CACHE_CONFIG.get('redis_port', 6379),
                password=CACHE_CONFIG.get('redis_password'),
                db=CACHE_CONFIG.get('redis_db', 0),
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis connected")
            
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            logger.info("ℹ️ Falling back to memory cache")
            self.redis_client = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            # Try Redis first
            if self.redis_client:
                full_key = self._make_key(key)
                value = self.redis_client.get(full_key)
                
                if value is not None:
                    self.cache_hits += 1
                    return pickle.loads(value)
            
            # Fallback to memory cache
            if key in self.memory_cache:
                self.cache_hits += 1
                return self.memory_cache[key]
            
            self.cache_misses += 1
            return None
            
        except Exception as e:
            logger.debug(f"Cache get error: {e}")
            self.cache_misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            if ttl is None:
                ttl = CACHE_CONFIG.get('default_ttl', 300)
            
            # Try Redis first
            if self.redis_client:
                full_key = self._make_key(key)
                serialized = pickle.dumps(value)
                self.redis_client.setex(full_key, ttl, serialized)
                return True
            
            # Fallback to memory cache
            self.memory_cache[key] = value
            return True
            
        except Exception as e:
            logger.debug(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            # Try Redis
            if self.redis_client:
                full_key = self._make_key(key)
                self.redis_client.delete(full_key)
            
            # Also delete from memory cache
            if key in self.memory_cache:
                del self.memory_cache[key]
            
            return True
            
        except Exception as e:
            logger.debug(f"Cache delete error: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache"""
        try:
            if self.redis_client:
                # Delete all keys with our prefix
                pattern = f"{CACHE_CONFIG.get('cache_prefix', '')}*"
                for key in self.redis_client.scan_iter(pattern):
                    self.redis_client.delete(key)
            
            self.memory_cache.clear()
            logger.info("✅ Cache cleared")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cache clear error: {e}")
            return False
    
    def _make_key(self, key: str) -> str:
        """Make full cache key with prefix"""
        prefix = CACHE_CONFIG.get('cache_prefix', 'btc_predictor:')
        return f"{prefix}{key}"
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        stats = {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'backend': 'redis' if self.redis_client else 'memory',
            'memory_cache_size': len(self.memory_cache)
        }
        
        if self.redis_client:
            try:
                info = self.redis_client.info('memory')
                stats['redis_memory_used'] = info.get('used_memory_human')
            except:
                pass
        
        return stats


# Global cache instance
_cache_instance = None


def get_cache() -> CacheManager:
    """Get or create cache manager instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheManager()
    return _cache_instance


def cached(ttl: Optional[int] = None, key_prefix: str = ''):
    """
    Decorator for caching function results
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
    
    Example:
        @cached(ttl=300, key_prefix='btc_data')
        def get_bitcoin_data(days, interval):
            # Expensive operation
            return data
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Generate cache key from function name and arguments
            key_parts = [key_prefix or func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            cache_key = ':'.join(key_parts)
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_value
            
            # Execute function
            logger.debug(f"Cache miss: {cache_key}")
            result = func(*args, **kwargs)
            
            # Store in cache
            if result is not None:
                cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


# Example usage decorator for data fetching
def cache_btc_data(days: int, interval: str):
    """Decorator specifically for Bitcoin data caching"""
    ttl = 300  # 5 minutes
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            cache_key = f"btc_data:{days}:{interval}"
            
            # Try cache
            cached_data = cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"📦 Using cached data: {cache_key}")
                return cached_data
            
            # Fetch fresh data
            logger.debug(f"🔄 Fetching fresh data: {cache_key}")
            data = func(*args, **kwargs)
            
            # Cache it
            if data is not None:
                cache.set(cache_key, data, ttl)
            
            return data
        
        return wrapper
    return decorator