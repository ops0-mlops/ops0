"""
ops0 Core Utilities - Common helper functions and utilities.
"""

import hashlib
import json
import time
import functools
from typing import Any, Dict, Callable, Optional, Union, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def hash_object(obj: Any) -> str:
    """
    Create a consistent hash of any Python object.

    Args:
        obj: Object to hash

    Returns:
        Hexadecimal hash string
    """
    if isinstance(obj, dict):
        # Sort keys for consistent hashing
        obj = {k: obj[k] for k in sorted(obj.keys())}
    elif isinstance(obj, list):
        obj = tuple(obj)
    elif isinstance(obj, set):
        obj = tuple(sorted(obj))

    try:
        # Try JSON serialization first
        obj_str = json.dumps(obj, sort_keys=True, default=str)
    except (TypeError, ValueError):
        # Fallback to string representation
        obj_str = str(obj)

    return hashlib.sha256(obj_str.encode()).hexdigest()[:12]


def timeit(func: Callable = None, *, name: str = None):
    """
    Decorator to measure function execution time.

    Args:
        func: Function to time (when used without parentheses)
        name: Custom name for the timing log

    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                func_name = name or f.__name__
                logger.debug(f"⏱️ {func_name} executed in {execution_time:.4f}s")
        return wrapper

    if func is None:
        # Used with parentheses: @timeit(name="custom")
        return decorator
    else:
        # Used without parentheses: @timeit
        return decorator(func)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to retry function execution on failure.

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts (seconds)
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        # Last attempt failed
                        logger.error(f"❌ {func.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    else:
                        logger.warning(f"⚠️ {func.__name__} attempt {attempt + 1} failed: {e}")
                        logger.info(f"🔄 Retrying in {current_delay:.2f}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff

        return wrapper
    return decorator


def ensure_path(path: Union[str, Path]) -> Path:
    """
    Ensure a path exists, creating directories if necessary.

    Args:
        path: Path to ensure

    Returns:
        Path object
    """
    path_obj = Path(path)
    if path_obj.suffix:
        # It's a file, create parent directories
        path_obj.parent.mkdir(parents=True, exist_ok=True)
    else:
        # It's a directory
        path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def safe_import(module_name: str, package: str = None) -> Optional[Any]:
    """
    Safely import a module, returning None if import fails.

    Args:
        module_name: Name of module to import
        package: Package for relative imports

    Returns:
        Imported module or None
    """
    try:
        if package:
            from importlib import import_module
            return import_module(module_name, package)
        else:
            return __import__(module_name)
    except ImportError:
        logger.debug(f"Failed to import {module_name}")
        return None


def format_size(size_bytes: int) -> str:
    """
    Format byte size in human readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0

    while size_bytes >= 1024 and unit_index < len(units) - 1:
        size_bytes /= 1024
        unit_index += 1

    return f"{size_bytes:.1f} {units[unit_index]}"


def format_duration(seconds: float) -> str:
    """
    Format duration in human readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


class SingletonMeta(type):
    """Metaclass for creating singleton classes"""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


def get_function_info(func: Callable) -> Dict[str, Any]:
    """
    Extract comprehensive information about a function.

    Args:
        func: Function to analyze

    Returns:
        Dictionary with function information
    """
    import inspect

    try:
        signature = inspect.signature(func)
        source_file = inspect.getfile(func)
        source_lines = inspect.getsourcelines(func)

        return {
            "name": func.__name__,
            "module": func.__module__,
            "file": source_file,
            "line_number": source_lines[1],
            "parameters": {
                name: {
                    "annotation": param.annotation if param.annotation != inspect.Parameter.empty else None,
                    "default": param.default if param.default != inspect.Parameter.empty else None,
                    "kind": param.kind.name
                }
                for name, param in signature.parameters.items()
            },
            "return_annotation": signature.return_annotation if signature.return_annotation != inspect.Signature.empty else None,
            "docstring": func.__doc__,
            "source_hash": hash_object(source_lines[0])
        }
    except Exception as e:
        logger.warning(f"Could not extract function info for {func}: {e}")
        return {
            "name": getattr(func, "__name__", "unknown"),
            "module": getattr(func, "__module__", "unknown"),
            "error": str(e)
        }