#!/usr/bin/env python3
"""
Test Redis Connection for Bitcoin Predictor
Quick test to verify Redis is working
"""

import sys
import os
from datetime import datetime

def test_redis_connection():
    """Test Redis connection"""
    print("\n" + "="*80)
    print("🔌 TESTING REDIS CONNECTION")
    print("="*80)
    
    try:
        import redis
        print("✅ Redis library installed")
    except ImportError:
        print("❌ Redis library not installed")
        print("\n   Install with: pip install redis")
        return False
    
    # Test 1: Basic Connection
    print("\n📡 Test 1: Basic Connection to Redis...")
    try:
        r = redis.Redis(
            host='localhost',
            port=6379,
            socket_connect_timeout=5,
            decode_responses=False
        )
        
        # Ping test
        r.ping()
        print("✅ Redis is running and responding to ping")
        
        # Get info
        info = r.info('server')
        print(f"   Redis version: {info.get('redis_version', 'unknown')}")
        print(f"   Uptime: {info.get('uptime_in_seconds', 0)} seconds")
        
    except redis.exceptions.ConnectionError as e:
        print(f"❌ Cannot connect to Redis: {e}")
        print("\n   Make sure Redis is running:")
        print("   - Check with: sudo systemctl status redis")
        print("   - Start with: sudo systemctl start redis")
        return False
    except Exception as e:
        print(f"❌ Redis error: {e}")
        return False
    
    # Test 2: Read/Write Operations
    print("\n📝 Test 2: Read/Write Operations...")
    try:
        # Write test
        test_key = "btc_predictor_test"
        test_value = f"test_{datetime.now().isoformat()}"
        
        r.set(test_key, test_value)
        print(f"✅ Write successful: {test_key} = {test_value}")
        
        # Read test
        retrieved = r.get(test_key)
        if retrieved:
            retrieved = retrieved.decode('utf-8')
            print(f"✅ Read successful: {retrieved}")
            
            if retrieved == test_value:
                print("✅ Data integrity verified")
            else:
                print("⚠️  Data mismatch!")
        else:
            print("❌ Failed to retrieve data")
            return False
        
        # Delete test
        r.delete(test_key)
        print("✅ Delete successful")
        
    except Exception as e:
        print(f"❌ Operation error: {e}")
        return False
    
    # Test 3: TTL (Time To Live)
    print("\n⏱️  Test 3: TTL Operations...")
    try:
        test_key = "btc_predictor_ttl_test"
        r.setex(test_key, 10, "expires in 10 seconds")
        
        ttl = r.ttl(test_key)
        print(f"✅ TTL set: {ttl} seconds")
        
        r.delete(test_key)
        
    except Exception as e:
        print(f"❌ TTL error: {e}")
        return False
    
    # Test 4: Memory Info
    print("\n💾 Test 4: Memory Info...")
    try:
        info = r.info('memory')
        used_memory = info.get('used_memory_human', 'unknown')
        max_memory = info.get('maxmemory_human', 'unlimited')
        
        print(f"   Used memory: {used_memory}")
        print(f"   Max memory:  {max_memory}")
        
    except Exception as e:
        print(f"⚠️  Memory info not available: {e}")
    
    # Test 5: Test with CacheManager
    print("\n🔧 Test 5: Testing with CacheManager...")
    try:
        from cache_manager import get_cache
        
        cache = get_cache()
        
        print(f"   Backend: {cache.redis_client if cache.redis_client else 'memory'}")
        
        # Test cache operations
        test_key = "btc_data_test"
        test_data = {
            'price': 43250.50,
            'timestamp': datetime.now().isoformat(),
            'data': [1, 2, 3, 4, 5]
        }
        
        cache.set(test_key, test_data, ttl=60)
        print(f"✅ CacheManager set successful")
        
        retrieved = cache.get(test_key)
        if retrieved:
            print(f"✅ CacheManager get successful")
            print(f"   Retrieved data: {retrieved}")
        else:
            print("❌ CacheManager get failed")
            return False
        
        cache.delete(test_key)
        print("✅ CacheManager delete successful")
        
        # Get stats
        stats = cache.get_stats()
        print(f"\n📊 Cache Statistics:")
        print(f"   Backend: {stats['backend']}")
        print(f"   Hit rate: {stats['hit_rate']:.1f}%")
        print(f"   Cache hits: {stats['cache_hits']}")
        print(f"   Cache misses: {stats['cache_misses']}")
        
    except ImportError:
        print("⚠️  CacheManager not available (not in btc-predictor directory?)")
    except Exception as e:
        print(f"❌ CacheManager error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "="*80)
    print("✅ ALL REDIS TESTS PASSED!")
    print("="*80)
    print("\nRedis is working correctly and ready to use.")
    print("\nTo enable Redis in Bitcoin Predictor:")
    print("1. Add to .env file:")
    print("   ENABLE_REDIS=true")
    print("   REDIS_HOST=localhost")
    print("   REDIS_PORT=6379")
    print("   REDIS_PASSWORD=  # Leave empty if no password")
    print("\n2. Restart the scheduler:")
    print("   sudo systemctl restart btc-predictor")
    print()
    
    return True


def quick_check():
    """Quick Redis availability check"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, socket_connect_timeout=2)
        r.ping()
        print("✅ Redis is running")
        return True
    except ImportError:
        print("❌ Redis library not installed")
        return False
    except:
        print("❌ Redis is not running")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        success = quick_check()
    else:
        success = test_redis_connection()
    
    sys.exit(0 if success else 1)