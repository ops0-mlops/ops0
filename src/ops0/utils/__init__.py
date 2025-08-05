"""
ops0 Utils Module

Common utilities and helpers for all ops0 components.
Provides shared functionality for serialization, validation, networking, and more.
"""

# Serialization utilities
from .serialization import (
    SmartSerializer,
    detect_serialization_format,
    serialize_data,
    deserialize_data,
    register_custom_serializer,
    get_serializer,
    SerializationFormat,
    compress_data,
    decompress_data,
)

# Validation utilities
from .validation import (
    validate_python_version,
    validate_step_function,
    validate_pipeline_config,
    validate_resource_requirements,
    validate_cloud_credentials,
    validate_docker_installed,
    ValidationResult,
    PipelineValidator,
    ConfigValidator,
)

# Network utilities
from .network import (
    get_free_port,
    wait_for_port,
    check_connectivity,
    download_with_progress,
    upload_with_retry,
    NetworkError,
    ConnectionPool,
    HTTPClient,
    get_external_ip,
    check_internet_connection,
)

# Formatting utilities
from .formatting import (
    format_bytes,
    format_duration,
    format_timestamp,
    format_progress,
    format_table,
    format_diff,
    truncate_string,
    humanize_number,
    colorize_text,
    create_progress_bar,
)

# Hashing and crypto utilities
from .hashing import (
    calculate_hash,
    calculate_file_hash,
    calculate_content_hash,
    verify_checksum,
    generate_id,
    generate_token,
    HashAlgorithm,
    create_signature,
    verify_signature,
)

# Caching utilities
from .caching import (
    MemoryCache,
    DiskCache,
    cache_decorator,
    clear_cache,
    get_cache_size,
    CacheStrategy,
    TTLCache,
    LRUCache,
    cache_key_generator,
)

# File system utilities
from .filesystem import (
    ensure_directory,
    safe_file_write,
    atomic_write,
    cleanup_old_files,
    find_files,
    copy_with_progress,
    get_file_info,
    watch_directory,
    FileLock,
    temporary_directory,
)

# Threading and async utilities
from .concurrency import (
    ThreadPoolManager,
    run_in_thread,
    run_async,
    synchronized,
    RateLimiter,
    Semaphore,
    async_retry,
    parallel_map,
    create_worker_pool,
)

# Retry and error handling
from .retry import (
    retry,
    exponential_backoff,
    linear_backoff,
    RetryConfig,
    RetryError,
    CircuitBreaker,
    fallback,
)

# System utilities
from .system import (
    get_system_info,
    get_memory_usage,
    get_cpu_usage,
    get_disk_usage,
    get_gpu_info,
    check_port_available,
    get_process_info,
    ResourceMonitor,
)

# Helpers and miscellaneous
from .helpers import (
    get_logger,
    setup_logging,
    deprecated,
    singleton,
    lazy_property,
    timer,
    memoize,
    flatten_dict,
    merge_dicts,
    chunk_list,
)

# Version
__version__ = "0.1.0"

__all__ = [
    # Serialization
    "SmartSerializer",
    "detect_serialization_format",
    "serialize_data",
    "deserialize_data",
    "register_custom_serializer",
    "get_serializer",
    "SerializationFormat",
    "compress_data",
    "decompress_data",

    # Validation
    "validate_python_version",
    "validate_step_function",
    "validate_pipeline_config",
    "validate_resource_requirements",
    "validate_cloud_credentials",
    "validate_docker_installed",
    "ValidationResult",
    "PipelineValidator",
    "ConfigValidator",

    # Network
    "get_free_port",
    "wait_for_port",
    "check_connectivity",
    "download_with_progress",
    "upload_with_retry",
    "NetworkError",
    "ConnectionPool",
    "HTTPClient",
    "get_external_ip",
    "check_internet_connection",

    # Formatting
    "format_bytes",
    "format_duration",
    "format_timestamp",
    "format_progress",
    "format_table",
    "format_diff",
    "truncate_string",
    "humanize_number",
    "colorize_text",
    "create_progress_bar",

    # Hashing
    "calculate_hash",
    "calculate_file_hash",
    "calculate_content_hash",
    "verify_checksum",
    "generate_id",
    "generate_token",
    "HashAlgorithm",
    "create_signature",
    "verify_signature",

    # Caching
    "MemoryCache",
    "DiskCache",
    "cache_decorator",
    "clear_cache",
    "get_cache_size",
    "CacheStrategy",
    "TTLCache",
    "LRUCache",
    "cache_key_generator",

    # Filesystem
    "ensure_directory",
    "safe_file_write",
    "atomic_write",
    "cleanup_old_files",
    "find_files",
    "copy_with_progress",
    "get_file_info",
    "watch_directory",
    "FileLock",
    "temporary_directory",

    # Concurrency
    "ThreadPoolManager",
    "run_in_thread",
    "run_async",
    "synchronized",
    "RateLimiter",
    "Semaphore",
    "async_retry",
    "parallel_map",
    "create_worker_pool",

    # Retry
    "retry",
    "exponential_backoff",
    "linear_backoff",
    "RetryConfig",
    "RetryError",
    "CircuitBreaker",
    "fallback",

    # System
    "get_system_info",
    "get_memory_usage",
    "get_cpu_usage",
    "get_disk_usage",
    "get_gpu_info",
    "check_port_available",
    "get_process_info",
    "ResourceMonitor",

    # Helpers
    "get_logger",
    "setup_logging",
    "deprecated",
    "singleton",
    "lazy_property",
    "timer",
    "memoize",
    "flatten_dict",
    "merge_dicts",
    "chunk_list",

    # Version
    "__version__",
]