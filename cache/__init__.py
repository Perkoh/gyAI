"""
cache/
──────
Redis caching layer for ADIS.

Exports:
    cache         — singleton RedisClient instance
    RedisClient   — the client class (for type hints and testing)
    CacheStats    — dataclass returned by cache.get_stats()

Quick import:
    from cache import cache
    from cache.redis_client import cache, CacheStats
"""

from cache.redis_client import cache, RedisClient, CacheStats

__all__ = ["cache", "RedisClient", "CacheStats"]
