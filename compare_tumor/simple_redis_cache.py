# simple_redis_cache.py - Simple Redis cache using native Redis capabilities
import json
import pickle
import hashlib
import logging
import time
from functools import wraps
from typing import Any, Optional
import redis

logger = logging.getLogger(__name__)

# Bump to orphan all existing cache entries (e.g., after a data refresh).
CACHE_VERSION = "v2"

class SimpleRedisCache:
    """Simple Redis cache that uses Redis's native eviction policies"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.memory_cache = {}  # Fallback in-memory cache
        self.max_memory_cache_size = 100
        
    def _make_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Create a stable cache key from function name and arguments"""
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        
        key_string = json.dumps(key_data, default=str, sort_keys=True)
        return f"cache:{CACHE_VERSION}:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (Redis first, then memory fallback)"""
        # Try Redis first
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    logger.debug(f"Redis cache hit for key {key}")
                    return pickle.loads(data)
            except Exception as e:
                logger.warning(f"Redis get failed for key {key}: {e}")
        
        # Fallback to memory cache
        cache_entry = self.memory_cache.get(key)
        if cache_entry:
            value, expiry = cache_entry
            if time.time() < expiry:
                logger.debug(f"Memory cache hit for key {key}")
                return value
            else:
                del self.memory_cache[key]
        
        return None
    
    def set(self, key: str, value: Any) -> bool:
        """Set value in cache (let Redis handle eviction automatically)"""
        success = False
        
        # Store in Redis (no TTL - let Redis handle eviction)
        if self.redis_client:
            try:
                serialized = pickle.dumps(value)
                self.redis_client.set(key, serialized)
                success = True
                logger.debug(f"Redis cache set successful for key {key}")
            except Exception as e:
                logger.warning(f"Redis set failed for key {key}: {e}")
        
        # Also store in memory cache as backup
        try:
            if len(self.memory_cache) >= self.max_memory_cache_size:
                # Simple FIFO for memory cache
                oldest_key = next(iter(self.memory_cache))
                del self.memory_cache[oldest_key]
            
            expiry = time.time() + 3600  # 1 hour fallback
            self.memory_cache[key] = (value, expiry)
            success = True
        except Exception as e:
            logger.error(f"Memory cache set failed for key {key}: {e}")
        
        return success
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        success = False
        
        if self.redis_client:
            try:
                self.redis_client.delete(key)
                success = True
            except Exception as e:
                logger.warning(f"Redis delete failed for key {key}: {e}")
        
        if key in self.memory_cache:
            del self.memory_cache[key]
            success = True
        
        return success
    
    def clear_all(self) -> bool:
        """Clear all cache entries"""
        success = False
        
        if self.redis_client:
            try:
                keys = self.redis_client.keys("cache:*")
                if keys:
                    self.redis_client.delete(*keys)
                success = True
                logger.info("Cleared all Redis cache entries")
            except Exception as e:
                logger.warning(f"Redis clear failed: {e}")
        
        self.memory_cache.clear()
        return success
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        stats = {
            'memory_cache_size': len(self.memory_cache),
            'redis_available': self.redis_client is not None,
            'cache_type': 'Simple Redis (native eviction)'
        }
        
        if self.redis_client:
            try:
                info = self.redis_client.info('memory')
                cache_keys_count = len(self.redis_client.keys("cache:*"))
                
                stats.update({
                    'redis_memory_used': info.get('used_memory_human', 'unknown'),
                    'redis_total_keys': self.redis_client.dbsize(),
                    'redis_cache_keys': cache_keys_count,
                    'redis_maxmemory_policy': info.get('maxmemory_policy', 'noeviction')
                })
            except Exception as e:
                stats['redis_error'] = str(e)
        
        return stats


# Global cache instance
_simple_redis_cache = None

def get_simple_redis_cache(redis_client: Optional[redis.Redis] = None) -> SimpleRedisCache:
    """Get or create the global simple Redis cache instance"""
    global _simple_redis_cache
    if _simple_redis_cache is None:
        _simple_redis_cache = SimpleRedisCache(redis_client)
    return _simple_redis_cache


def simple_redis_cache(cache_instance: Optional[SimpleRedisCache] = None):
    """
    Simple Redis cache decorator that uses Redis's native eviction
    
    Usage:
        @simple_redis_cache()
        def expensive_function(arg1, arg2):
            return compute_expensive_result(arg1, arg2)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = cache_instance or get_simple_redis_cache()
            
            cache_key = cache._make_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Cache miss - execute function
            logger.debug(f"Cache miss for {func.__name__} - executing function")
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Store result in cache
            cache.set(cache_key, result)
            
            logger.info(f"Function {func.__name__} executed in {execution_time:.2f}s and cached (Redis)")
            return result
        
        # Add cache management methods
        wrapper.clear_cache = lambda: get_simple_redis_cache().clear_all()
        wrapper.cache_stats = lambda: get_simple_redis_cache().get_stats()
        
        return wrapper
    return decorator


def initialize_simple_redis_cache(redis_client: redis.Redis):
    """Initialize the global simple Redis cache"""
    global _simple_redis_cache
    _simple_redis_cache = SimpleRedisCache(redis_client)
    logger.info("Simple Redis cache initialized successfully")
    return _simple_redis_cache
