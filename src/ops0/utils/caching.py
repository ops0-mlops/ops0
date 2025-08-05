"""
ops0 Caching Utilities

Memory and disk caching with various eviction strategies.
"""

import time
import pickle
import json
import hashlib
import functools
import threading
from pathlib import Path
from typing import Any, Optional, Dict, Callable, Union, TypeVar, Generic
from dataclasses import dataclass, field
from collections import OrderedDict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live


@dataclass
class CacheEntry(Generic[T]):
    """Single cache entry"""
    key: str
    value: T
    size: int
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def access(self):
        """Record access to this entry"""
        self.last_accessed = time.time()
        self.access_count += 1


class MemoryCache(Generic[T]):
    """
    Thread-safe in-memory cache with configurable eviction.

    Example:
        cache = MemoryCache(max_size=100, strategy=CacheStrategy.LRU)

        # Basic usage
        cache.set("key", "value")
        value = cache.get("key")

        # With TTL
        cache.set("temp", "data", ttl=60)  # Expires in 60 seconds
    """

    def __init__(
            self,
            max_size: int = 1000,
            max_memory: Optional[int] = None,  # Bytes
            strategy: CacheStrategy = CacheStrategy.LRU,
            default_ttl: Optional[float] = None
    ):
        self.max_size = max_size
        self.max_memory = max_memory
        self.strategy = strategy
        self.default_ttl = default_ttl

        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._total_memory = 0
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired': 0
        }

    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """
        Get value from cache.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats['misses'] += 1
                return default

            # Check expiration
            if entry.is_expired():
                self._stats['expired'] += 1
                del self._cache[key]
                self._total_memory -= entry.size
                return default

            # Update access info
            entry.access()
            self._stats['hits'] += 1

            # Move to end for LRU
            if self.strategy == CacheStrategy.LRU:
                self._cache.move_to_end(key)

            return entry.value

    def set(
            self,
            key: str,
            value: T,
            ttl: Optional[float] = None,
            size: Optional[int] = None
    ) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            size: Size in bytes (auto-calculated if None)
        """
        with self._lock:
            # Calculate size if not provided
            if size is None:
                size = self._estimate_size(value)

            # Check if we need to evict
            self._evict_if_needed(size)

            # Remove old entry if exists
            if key in self._cache:
                old_entry = self._cache[key]
                self._total_memory -= old_entry.size

            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                size=size,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl=ttl or self.default_ttl
            )

            # Add to cache
            self._cache[key] = entry
            self._total_memory += size

            # Move to end for LRU
            if self.strategy == CacheStrategy.LRU:
                self._cache.move_to_end(key)

    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                del self._cache[key]
                self._total_memory -= entry.size
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._total_memory = 0

    def size(self) -> int:
        """Get number of entries in cache"""
        return len(self._cache)

    def memory_usage(self) -> int:
        """Get total memory usage in bytes"""
        return self._total_memory

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0

            return {
                **self._stats,
                'size': self.size(),
                'memory_usage': self._total_memory,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes"""
        try:
            # Try to serialize to get accurate size
            serialized = pickle.dumps(value)
            return len(serialized)
        except Exception:
            # Fallback to rough estimate
            return 128  # Default size

    def _evict_if_needed(self, new_size: int) -> None:
        """Evict entries if cache is full"""
        # Check size limit
        while self._cache and len(self._cache) >= self.max_size:
            self._evict_one()

        # Check memory limit
        if self.max_memory:
            while self._cache and self._total_memory + new_size > self.max_memory:
                self._evict_one()

    def _evict_one(self) -> None:
        """Evict one entry based on strategy"""
        if not self._cache:
            return

        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used (first item)
            key = next(iter(self._cache))
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            key = min(self._cache, key=lambda k: self._cache[k].access_count)
        elif self.strategy == CacheStrategy.FIFO:
            # Remove oldest (first item)
            key = next(iter(self._cache))
        elif self.strategy == CacheStrategy.TTL:
            # Remove expired or oldest
            for k, entry in self._cache.items():
                if entry.is_expired():
                    key = k
                    break
            else:
                key = next(iter(self._cache))
        else:
            key = next(iter(self._cache))

        # Remove entry
        entry = self._cache[key]
        del self._cache[key]
        self._total_memory -= entry.size
        self._stats['evictions'] += 1


class DiskCache:
    """
    Persistent disk-based cache.

    Example:
        cache = DiskCache("/tmp/cache", max_size_mb=100)

        cache.set("key", large_dataframe)
        df = cache.get("key")
    """

    def __init__(
            self,
            cache_dir: Union[str, Path],
            max_size_mb: int = 1000,
            serializer: str = 'pickle',
            compression: bool = True
    ):
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self.serializer = serializer
        self.compression = compression

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Metadata file
        self.metadata_file = self.cache_dir / '.cache_metadata.json'
        self.metadata = self._load_metadata()

        self._lock = threading.Lock()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass

        return {
            'entries': {},
            'total_size': 0
        }

    def _save_metadata(self) -> None:
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key"""
        # Hash key to avoid filesystem issues
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{key_hash}.cache"

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        with self._lock:
            cache_path = self._get_cache_path(key)

            if not cache_path.exists():
                return default

            # Check metadata
            if key not in self.metadata['entries']:
                return default

            entry = self.metadata['entries'][key]

            # Check expiration
            if entry.get('ttl'):
                if time.time() - entry['created_at'] > entry['ttl']:
                    self.delete(key)
                    return default

            try:
                # Load from disk
                with open(cache_path, 'rb') as f:
                    data = f.read()

                # Decompress if needed
                if self.compression:
                    import gzip
                    data = gzip.decompress(data)

                # Deserialize
                if self.serializer == 'pickle':
                    return pickle.loads(data)
                elif self.serializer == 'json':
                    return json.loads(data.decode('utf-8'))
                else:
                    raise ValueError(f"Unknown serializer: {self.serializer}")

            except Exception as e:
                logger.error(f"Failed to load cache entry {key}: {e}")
                return default

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache"""
        with self._lock:
            cache_path = self._get_cache_path(key)

            try:
                # Serialize
                if self.serializer == 'pickle':
                    data = pickle.dumps(value)
                elif self.serializer == 'json':
                    data = json.dumps(value).encode('utf-8')
                else:
                    raise ValueError(f"Unknown serializer: {self.serializer}")

                # Compress if needed
                if self.compression:
                    import gzip
                    data = gzip.compress(data)

                # Check size limit
                size_mb = len(data) / (1024 * 1024)
                if self.metadata['total_size'] + size_mb > self.max_size_mb:
                    self._evict_to_fit(size_mb)

                # Write to disk
                with open(cache_path, 'wb') as f:
                    f.write(data)

                # Update metadata
                self.metadata['entries'][key] = {
                    'size_mb': size_mb,
                    'created_at': time.time(),
                    'ttl': ttl
                }
                self.metadata['total_size'] += size_mb
                self._save_metadata()

            except Exception as e:
                logger.error(f"Failed to save cache entry {key}: {e}")
                raise

    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        with self._lock:
            cache_path = self._get_cache_path(key)

            if cache_path.exists():
                cache_path.unlink()

            if key in self.metadata['entries']:
                entry = self.metadata['entries'][key]
                self.metadata['total_size'] -= entry['size_mb']
                del self.metadata['entries'][key]
                self._save_metadata()
                return True

            return False

    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            # Remove all cache files
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()

            # Reset metadata
            self.metadata = {
                'entries': {},
                'total_size': 0
            }
            self._save_metadata()

    def _evict_to_fit(self, required_mb: float) -> None:
        """Evict entries to fit new data"""
        # Sort by creation time (FIFO)
        entries = sorted(
            self.metadata['entries'].items(),
            key=lambda x: x[1]['created_at']
        )

        for key, entry in entries:
            if self.metadata['total_size'] + required_mb <= self.max_size_mb:
                break

            self.delete(key)


# Cache decorators

def cache_decorator(
        cache: Optional[Union[MemoryCache, DiskCache]] = None,
        key_func: Optional[Callable] = None,
        ttl: Optional[float] = None
):
    """
    Decorator for caching function results.

    Args:
        cache: Cache instance (creates MemoryCache if None)
        key_func: Function to generate cache key from arguments
        ttl: Time to live for cached results

    Example:
        @cache_decorator(ttl=60)
        def expensive_function(x, y):
            return x ** y

        # Custom key function
        @cache_decorator(key_func=lambda x, y: f"{x}_{y}")
        def custom_key_func(x, y):
            return x + y
    """
    if cache is None:
        cache = MemoryCache()

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                key = "_".join(key_parts)

            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result

            # Compute result
            result = func(*args, **kwargs)

            # Store in cache
            cache.set(key, result, ttl=ttl)

            return result

        # Add cache control methods
        wrapper.cache = cache
        wrapper.clear_cache = cache.clear

        return wrapper

    return decorator


# Specialized cache implementations

class TTLCache(MemoryCache[T]):
    """Cache with automatic TTL-based expiration"""

    def __init__(self, default_ttl: float = 3600, **kwargs):
        super().__init__(
            strategy=CacheStrategy.TTL,
            default_ttl=default_ttl,
            **kwargs
        )
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """Start background thread for cleanup"""

        def cleanup():
            while True:
                time.sleep(60)  # Check every minute
                self._cleanup_expired()

        thread = threading.Thread(target=cleanup, daemon=True)
        thread.start()

    def _cleanup_expired(self):
        """Remove expired entries"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]

            for key in expired_keys:
                self.delete(key)


class LRUCache(MemoryCache[T]):
    """Least Recently Used cache"""

    def __init__(self, max_size: int = 128, **kwargs):
        super().__init__(
            max_size=max_size,
            strategy=CacheStrategy.LRU,
            **kwargs
        )


# Utility functions

def clear_cache(cache: Union[MemoryCache, DiskCache]) -> None:
    """Clear all entries from cache"""
    cache.clear()


def get_cache_size(cache: Union[MemoryCache, DiskCache]) -> int:
    """Get number of entries in cache"""
    if isinstance(cache, MemoryCache):
        return cache.size()
    elif isinstance(cache, DiskCache):
        return len(cache.metadata['entries'])
    else:
        raise ValueError(f"Unknown cache type: {type(cache)}")


def cache_key_generator(*args, **kwargs) -> str:
    """Generate cache key from arguments"""
    key_parts = []
    key_parts.extend(str(arg) for arg in args)
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))

    # Hash for consistent length
    key_str = "_".join(key_parts)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]