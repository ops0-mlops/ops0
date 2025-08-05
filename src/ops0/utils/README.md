# ops0 Utils Module

Comprehensive utilities package for the ops0 MLOps platform. This module provides shared functionality across all ops0 components.

## Overview

The utils module contains the following submodules:

### 🔄 Serialization (`serialization.py`)
Smart serialization with automatic format detection and optimization.

**Key Features:**
- Automatic format detection (Pickle, JSON, Parquet, NumPy, PyTorch)
- Compression support
- Custom serializers registration
- Content-aware serialization

**Example:**
```python
from ops0.utils import SmartSerializer, serialize_data

serializer = SmartSerializer()
# Automatically detects DataFrame → uses Parquet
result = serializer.serialize(dataframe)

# Force specific format
data = serialize_data(my_dict, format=SerializationFormat.JSON)
```

### ✅ Validation (`validation.py`)
Comprehensive validation for pipelines, configurations, and environments.

**Key Features:**
- Python version validation
- Step function validation
- Pipeline configuration validation
- Resource requirements validation
- Cloud credentials checking
- Docker environment validation

**Example:**
```python
from ops0.utils import validate_step_function, ValidationResult

@ops0.step
def my_step(data):
    return process(data)

result = validate_step_function(my_step)
if not result.valid:
    for issue in result.get_errors():
        print(f"❌ {issue.message}")
```

### 🌐 Network (`network.py`)
Network operations, HTTP clients, and connectivity helpers.

**Key Features:**
- Port availability checking
- Connection pooling
- HTTP client with retries
- File download with progress
- Rate limiting

**Example:**
```python
from ops0.utils import HTTPClient, download_with_progress

# High-level HTTP client
client = HTTPClient(base_url='https://api.example.com')
data = client.get('/users/123')

# Download with progress
download_with_progress(
    url='https://example.com/model.bin',
    destination=Path('/tmp/model.bin'),
    progress_callback=lambda c, t: print(f"{c/t*100:.1f}%")
)
```

### 📝 Formatting (`formatting.py`)
Human-readable formatting for various data types.

**Key Features:**
- Byte size formatting
- Duration formatting
- Timestamp formatting
- Progress bars
- Table formatting
- Colored output
- Diff formatting

**Example:**
```python
from ops0.utils import format_bytes, format_duration, create_progress_bar

print(format_bytes(1234567890))  # "1.15 GB"
print(format_duration(123.45))    # "2m 3s"

# Progress bar
progress = create_progress_bar(
    current=30,
    total=100,
    prefix='Processing: ',
    show_time=True
)
```

### 🔐 Hashing (`hashing.py`)
Secure hashing, checksums, and ID generation.

**Key Features:**
- Multiple hash algorithms (SHA256, SHA512, BLAKE2, etc.)
- File hashing
- HMAC signatures
- Unique ID generation
- Password hashing

**Example:**
```python
from ops0.utils import calculate_hash, generate_id, create_signature

# Hash data
hash_value = calculate_hash("hello world", HashAlgorithm.SHA256)

# Generate unique IDs
job_id = generate_id(prefix='job_', length=12)

# Create HMAC signature
signature = create_signature(message, secret_key)
```

### 💾 Caching (`caching.py`)
Memory and disk caching with various eviction strategies.

**Key Features:**
- In-memory caching with LRU/LFU/TTL strategies
- Disk-based persistent caching
- Cache decorators
- Thread-safe operations

**Example:**
```python
from ops0.utils import MemoryCache, cache_decorator

# Memory cache
cache = MemoryCache(max_size=1000, strategy=CacheStrategy.LRU)
cache.set("key", expensive_computation())
value = cache.get("key")

# Decorator
@cache_decorator(ttl=3600)
def expensive_function(x):
    return complex_calculation(x)
```

### 📁 Filesystem (`filesystem.py`)
Safe file operations, atomic writes, and filesystem helpers.

**Key Features:**
- Atomic file writes
- Safe file operations
- Directory watching
- File locking
- Temporary directories

**Example:**
```python
from ops0.utils import atomic_write, ensure_directory, FileLock

# Atomic write
with atomic_write('/tmp/config.json') as f:
    json.dump(config, f)

# File locking
with FileLock('/tmp/process.lock'):
    # Exclusive access
    process_data()
```

### 🔄 Concurrency (`concurrency.py`)
Thread pools, async helpers, and synchronization primitives.

**Key Features:**
- Managed thread pools
- Rate limiting
- Semaphores with statistics
- Async retry
- Parallel mapping

**Example:**
```python
from ops0.utils import ThreadPoolManager, RateLimiter, parallel_map

# Thread pool
with ThreadPoolManager(max_workers=4) as pool:
    future = pool.submit(process_item, item)
    result = future.result()

# Rate limiting
limiter = RateLimiter(rate=10, per=1.0)  # 10 per second
with limiter:
    api_call()
```

### 🔁 Retry (`retry.py`)
Retry decorators, backoff strategies, and circuit breakers.

**Key Features:**
- Configurable retry strategies
- Exponential/linear/fibonacci backoff
- Circuit breaker pattern
- Fallback mechanisms

**Example:**
```python
from ops0.utils import retry, CircuitBreaker, exponential_backoff

# Retry with exponential backoff
@exponential_backoff(initial=1.0, maximum=60.0)
def flaky_api_call():
    return requests.get("https://api.example.com")

# Circuit breaker
breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
@breaker
def external_service_call():
    return service.call()
```

### 💻 System (`system.py`)
System information, resource monitoring, and platform detection.

**Key Features:**
- System information gathering
- CPU/Memory/Disk monitoring
- GPU detection
- Process information
- Resource monitoring

**Example:**
```python
from ops0.utils import get_system_info, ResourceMonitor

# System info
info = get_system_info()
print(f"Platform: {info.platform}, CPUs: {info.cpu_count}")

# Resource monitoring
monitor = ResourceMonitor()
monitor.start()
# ... do work ...
stats = monitor.get_stats()
print(f"Peak memory: {stats['memory_peak_percent']:.1f}%")
```

### 🛠️ Helpers (`helpers.py`)
General-purpose decorators, utilities, and helper functions.

**Key Features:**
- Logging utilities
- Deprecation decorators
- Singleton pattern
- Lazy properties
- Timer decorator
- Dictionary utilities

**Example:**
```python
from ops0.utils import timer, singleton, lazy_property, merge_dicts

# Timer decorator
@timer(name="data processing")
def process_data(data):
    return transform(data)

# Singleton
@singleton
class DatabaseConnection:
    def __init__(self):
        self.conn = connect()

# Lazy property
class DataProcessor:
    @lazy_property
    def processed_data(self):
        return expensive_computation()
```

## Installation

The utils module is included with ops0:

```bash
pip install ops0
```

For specific functionality, install with extras:

```bash
# For ML serialization (parquet, numpy)
pip install ops0[ml]

# For cloud utilities
pip install ops0[cloud]

# For monitoring
pip install ops0[monitoring]
```

## Usage Patterns

### Error Handling

```python
from ops0.utils import retry, fallback

@retry(max_attempts=3, on_retry=lambda e, n: logger.warning(f"Retry {n}: {e}"))
@fallback(fallback_value={"status": "unavailable"})
def get_service_status():
    return external_service.get_status()
```

### Resource Management

```python
from ops0.utils import ResourceMonitor, get_available_resources

# Check available resources before heavy computation
resources = get_available_resources()
if resources['memory']['available_gb'] < 4:
    raise ResourceError("Insufficient memory")

# Monitor resource usage
with ResourceMonitor() as monitor:
    result = heavy_computation()
    stats = monitor.get_stats()
    logger.info(f"Peak memory usage: {stats['memory_peak_percent']:.1f}%")
```

### Performance Optimization

```python
from ops0.utils import cache_decorator, parallel_map, timer

# Cache expensive computations
@cache_decorator(ttl=3600)
@timer
def expensive_function(x):
    return complex_calculation(x)

# Parallel processing
results = parallel_map(
    process_item,
    items,
    max_workers=8,
    progress_callback=lambda c, t: print(f"Progress: {c}/{t}")
)
```

## Best Practices

1. **Use appropriate serialization formats**
   - JSON for configuration and small data
   - Parquet for DataFrames
   - Pickle for complex Python objects
   - NumPy format for arrays

2. **Handle errors gracefully**
   - Use retry decorators for network operations
   - Implement circuit breakers for external services
   - Provide fallbacks for non-critical operations

3. **Monitor resource usage**
   - Check available resources before heavy operations
   - Use ResourceMonitor for long-running processes
   - Set appropriate memory limits

4. **Optimize for performance**
   - Use caching for expensive computations
   - Leverage parallel processing for independent tasks
   - Profile with timer decorators

5. **Ensure thread safety**
   - Use provided thread-safe utilities
   - Leverage synchronization primitives
   - Avoid shared mutable state

## Contributing

When adding new utilities:

1. Follow the existing patterns and conventions
2. Add comprehensive docstrings with examples
3. Include type hints
4. Write unit tests
5. Update this documentation

## License

Part of the ops0 project. See LICENSE file in the root directory.