"""
ops0 Helper Utilities

General-purpose decorators, utilities, and helper functions.
"""

import functools
import logging
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, Iterable
from collections import defaultdict
import threading
import inspect
from pathlib import Path
import sys

T = TypeVar('T')


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get configured logger instance.

    Args:
        name: Logger name (uses calling module if None)

    Returns:
        Logger instance

    Example:
        logger = get_logger()
        logger.info("Starting process")
    """
    if name is None:
        # Get calling module name
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'ops0')
        else:
            name = 'ops0'

    return logging.getLogger(name)


def setup_logging(
        level: Union[str, int] = logging.INFO,
        format: Optional[str] = None,
        handlers: Optional[List[logging.Handler]] = None,
        log_file: Optional[Union[str, Path]] = None
) -> None:
    """
    Setup logging configuration.

    Args:
        level: Logging level
        format: Log format string
        handlers: Custom handlers
        log_file: Optional log file path

    Example:
        setup_logging(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            log_file='/tmp/app.log'
        )
    """
    if format is None:
        format = '%(asctime)s - ops0.%(name)s - %(levelname)s - %(message)s'

    # Configure root logger
    root_logger = logging.getLogger('ops0')
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(format))
    root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format))
        root_logger.addHandler(file_handler)

    # Custom handlers
    if handlers:
        for handler in handlers:
            root_logger.addHandler(handler)


def deprecated(
        reason: Optional[str] = None,
        version: Optional[str] = None,
        alternative: Optional[str] = None
):
    """
    Decorator to mark functions as deprecated.

    Args:
        reason: Deprecation reason
        version: Version when deprecated
        alternative: Alternative function/method to use

    Example:
        @deprecated(
            reason="Use new_function instead",
            version="1.0.0",
            alternative="new_function"
        )
        def old_function():
            pass
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = f"{func.__name__} is deprecated"

            parts = []
            if version:
                parts.append(f"since version {version}")
            if reason:
                parts.append(f": {reason}")
            if alternative:
                parts.append(f". Use {alternative} instead")

            if parts:
                message += " " + "".join(parts)

            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        # Add deprecation to docstring
        if wrapper.__doc__:
            wrapper.__doc__ = f"**DEPRECATED** {message}\n\n{wrapper.__doc__}"
        else:
            wrapper.__doc__ = f"**DEPRECATED** {message}"

        return wrapper

    return decorator


def singleton(cls):
    """
    Decorator to make a class a singleton.

    Example:
        @singleton
        class DatabaseConnection:
            def __init__(self):
                self.connection = connect()

        # Always returns same instance
        db1 = DatabaseConnection()
        db2 = DatabaseConnection()
        assert db1 is db2
    """
    instances = {}
    lock = threading.Lock()

    def get_instance(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


def lazy_property(func):
    """
    Decorator for lazy evaluation of properties.

    The property is computed once on first access and cached.

    Example:
        class DataProcessor:
            def __init__(self, data):
                self.data = data

            @lazy_property
            def processed_data(self):
                # Expensive computation
                return process(self.data)

        proc = DataProcessor(data)
        # First access computes the value
        result1 = proc.processed_data
        # Second access returns cached value
        result2 = proc.processed_data
    """
    attr_name = f'_lazy_{func.__name__}'

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return wrapper


def timer(
        func: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        level: int = logging.INFO
):
    """
    Decorator to time function execution.

    Args:
        func: Function to time (when used without parentheses)
        name: Custom name for timing message
        logger: Logger to use
        level: Logging level

    Example:
        @timer
        def slow_function():
            time.sleep(1)

        @timer(name="data processing", level=logging.DEBUG)
        def process_data(data):
            return transform(data)
    """

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = f(*args, **kwargs)
                elapsed = time.time() - start_time

                # Log timing
                log_name = name or f.__name__
                message = f"{log_name} took {elapsed:.3f}s"

                if logger:
                    logger.log(level, message)
                else:
                    get_logger().log(level, message)

                return result

            except Exception:
                elapsed = time.time() - start_time
                log_name = name or f.__name__
                message = f"{log_name} failed after {elapsed:.3f}s"

                if logger:
                    logger.log(logging.ERROR, message)
                else:
                    get_logger().log(logging.ERROR, message)

                raise

        return wrapper

    # Handle @timer and @timer() syntax
    if func is None:
        return decorator
    else:
        return decorator(func)


def memoize(
        maxsize: Optional[int] = 128,
        typed: bool = False,
        key_func: Optional[Callable] = None
):
    """
    Decorator for memoization with custom key function.

    Args:
        maxsize: Maximum cache size (None for unlimited)
        typed: Consider argument types in cache
        key_func: Custom function to generate cache key

    Example:
        @memoize(maxsize=100)
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)

        # With custom key
        @memoize(key_func=lambda x, y: f"{x}_{y}")
        def compute(x, y):
            return expensive_operation(x, y)
    """

    def decorator(func):
        # Use functools.lru_cache if no custom key function
        if key_func is None:
            return functools.lru_cache(maxsize=maxsize, typed=typed)(func)

        # Custom implementation with key function
        cache = {}
        cache_lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = key_func(*args, **kwargs)

            # Check cache
            with cache_lock:
                if key in cache:
                    return cache[key]

            # Compute result
            result = func(*args, **kwargs)

            # Store in cache
            with cache_lock:
                cache[key] = result

                # Evict if needed (simple FIFO)
                if maxsize and len(cache) > maxsize:
                    # Remove oldest entry
                    oldest_key = next(iter(cache))
                    del cache[oldest_key]

            return result

        # Add cache control methods
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {'size': len(cache), 'maxsize': maxsize}

        return wrapper

    return decorator


def flatten_dict(
        d: Dict[str, Any],
        parent_key: str = '',
        separator: str = '.'
) -> Dict[str, Any]:
    """
    Flatten nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        separator: Key separator

    Returns:
        Flattened dictionary

    Example:
        data = {
            'user': {
                'name': 'Alice',
                'address': {
                    'city': 'NYC',
                    'zip': '10001'
                }
            }
        }
        flat = flatten_dict(data)
        # {'user.name': 'Alice', 'user.address.city': 'NYC', 'user.address.zip': '10001'}
    """
    items = []

    for key, value in d.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key

        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, separator).items())
        else:
            items.append((new_key, value))

    return dict(items)


def merge_dicts(
        *dicts: Dict[str, Any],
        deep: bool = True
) -> Dict[str, Any]:
    """
    Merge multiple dictionaries.

    Args:
        *dicts: Dictionaries to merge
        deep: Perform deep merge

    Returns:
        Merged dictionary

    Example:
        config1 = {'server': {'host': 'localhost', 'port': 8080}}
        config2 = {'server': {'port': 9090}, 'debug': True}
        merged = merge_dicts(config1, config2)
        # {'server': {'host': 'localhost', 'port': 9090}, 'debug': True}
    """
    result = {}

    for d in dicts:
        if deep:
            _deep_merge(result, d)
        else:
            result.update(d)

    return result


def _deep_merge(dest: Dict[str, Any], src: Dict[str, Any]) -> None:
    """Deep merge src into dest"""
    for key, value in src.items():
        if key in dest and isinstance(dest[key], dict) and isinstance(value, dict):
            _deep_merge(dest[key], value)
        else:
            dest[key] = value


def chunk_list(
        items: List[T],
        chunk_size: int
) -> List[List[T]]:
    """
    Split list into chunks.

    Args:
        items: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks

    Example:
        data = list(range(10))
        chunks = chunk_list(data, 3)
        # [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def retry_on_import_error(
        module_name: str,
        install_name: Optional[str] = None,
        install_command: Optional[str] = None
):
    """
    Decorator to handle missing module imports gracefully.

    Args:
        module_name: Module to import
        install_name: Package name for pip install
        install_command: Full install command

    Example:
        @retry_on_import_error('pandas', install_name='pandas')
        def process_dataframe():
            import pandas as pd
            return pd.DataFrame({'a': [1, 2, 3]})
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ImportError as e:
                if module_name in str(e):
                    message = f"Required module '{module_name}' not found. "

                    if install_command:
                        message += f"Install with: {install_command}"
                    elif install_name:
                        message += f"Install with: pip install {install_name}"
                    else:
                        message += f"Please install '{module_name}'"

                    raise ImportError(message) from e
                else:
                    raise

        return wrapper

    return decorator


class LazyImport:
    """
    Lazy import helper for optional dependencies.

    Example:
        # Instead of failing at import time:
        # import tensorflow as tf

        # Use lazy import:
        tf = LazyImport('tensorflow', install_name='tensorflow')

        # Module is imported on first use
        model = tf.keras.Sequential([...])
    """

    def __init__(
            self,
            module_name: str,
            install_name: Optional[str] = None,
            install_command: Optional[str] = None
    ):
        self.module_name = module_name
        self.install_name = install_name or module_name
        self.install_command = install_command
        self._module = None

    def _load(self):
        """Load the module"""
        if self._module is None:
            try:
                self._module = __import__(self.module_name)

                # Handle nested modules
                parts = self.module_name.split('.')
                for part in parts[1:]:
                    self._module = getattr(self._module, part)

            except ImportError as e:
                message = f"Required module '{self.module_name}' not found. "

                if self.install_command:
                    message += f"Install with: {self.install_command}"
                else:
                    message += f"Install with: pip install {self.install_name}"

                raise ImportError(message) from e

        return self._module

    def __getattr__(self, name):
        module = self._load()
        return getattr(module, name)

    def __dir__(self):
        module = self._load()
        return dir(module)


def get_caller_info(depth: int = 1) -> Dict[str, Any]:
    """
    Get information about the calling function.

    Args:
        depth: Stack depth (1 = immediate caller)

    Returns:
        Dictionary with caller information

    Example:
        def my_function():
            info = get_caller_info()
            print(f"Called from {info['function']} at line {info['line']}")
    """
    frame = inspect.currentframe()

    # Go up the stack
    for _ in range(depth + 1):
        if frame is None:
            break
        frame = frame.f_back

    if frame is None:
        return {
            'filename': '<unknown>',
            'line': 0,
            'function': '<unknown>',
            'module': '<unknown>'
        }

    return {
        'filename': frame.f_code.co_filename,
        'line': frame.f_lineno,
        'function': frame.f_code.co_name,
        'module': frame.f_globals.get('__name__', '<unknown>')
    }


def ensure_list(value: Union[T, List[T]]) -> List[T]:
    """
    Ensure value is a list.

    Args:
        value: Single value or list

    Returns:
        List containing the value(s)

    Example:
        ensure_list('hello')  # ['hello']
        ensure_list(['a', 'b'])  # ['a', 'b']
    """
    if isinstance(value, list):
        return value
    elif isinstance(value, (tuple, set)):
        return list(value)
    elif isinstance(value, str):
        # Don't split strings into characters
        return [value]
    elif hasattr(value, '__iter__'):
        return list(value)
    else:
        return [value]


def first_not_none(*values: Optional[T]) -> Optional[T]:
    """
    Return first non-None value.

    Args:
        *values: Values to check

    Returns:
        First non-None value or None

    Example:
        result = first_not_none(None, None, 'hello', 'world')
        # 'hello'
    """
    for value in values:
        if value is not None:
            return value
    return None


def safe_get(
        d: Dict[str, Any],
        path: str,
        default: Any = None,
        separator: str = '.'
) -> Any:
    """
    Safely get nested dictionary value.

    Args:
        d: Dictionary
        path: Path to value (e.g., 'user.address.city')
        default: Default value if not found
        separator: Path separator

    Returns:
        Value or default

    Example:
        data = {'user': {'name': 'Alice', 'address': {'city': 'NYC'}}}
        city = safe_get(data, 'user.address.city')  # 'NYC'
        country = safe_get(data, 'user.address.country', 'USA')  # 'USA'
    """
    keys = path.split(separator)
    current = d

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default

    return current