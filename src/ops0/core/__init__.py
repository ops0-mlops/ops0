"""
ops0 Core Module

The heart of ops0 - Python-Native ML Pipeline Orchestration.
Provides decorators, storage, execution, and registry primitives.
"""

# Core decorators and pipeline management
from .decorators import step, pipeline, StepMetadata
from .graph import PipelineGraph, StepNode, get_current_pipeline

# Storage and data management
from .storage import storage, StorageLayer, LocalStorageBackend, with_namespace

# Execution engine
from .executor import run, deploy, run_local, PipelineExecutor, ExecutionResult, execution_context

# Function analysis
from .analyzer import FunctionAnalyzer, FunctionSignature, StorageDependency

# Registry for ML artifacts
from .registry import (
    registry,
    register_model,
    register_pipeline,
    get_latest_model,
    load_latest_model,
    search_registry,
    PipelineRegistryEntry,
    ModelRegistryEntry,
    ArtifactRegistryEntry,
    DeploymentRegistryEntry
)

# Configuration management
from .config import config, config_manager, get_config_value, set_config_value, configure_logging

# Exception hierarchy
from .exceptions import (
    Ops0Error,
    PipelineError,
    StepError,
    StorageError,
    ExecutionError,
    DependencyError,
    ValidationError,
    RegistryError,
    ConfigurationError,
    SerializationError
)

# Version compatibility
try:
    from ...__about__ import __version__
except ImportError:
    __version__ = "0.1.0-dev"

# Public API
__all__ = [
    # Core decorators
    "step",
    "pipeline",

    # Execution functions
    "run",
    "deploy",
    "run_local",

    # Storage
    "storage",
    "with_namespace",

    # Registry - ML artifact management
    "registry",
    "register_model",
    "register_pipeline",
    "get_latest_model",
    "load_latest_model",
    "search_registry",

    # Pipeline management
    "get_current_pipeline",
    "execution_context",

    # Configuration
    "config",
    "config_manager",
    "get_config_value",
    "set_config_value",

    # Advanced classes (for library users)
    "PipelineGraph",
    "StepNode",
    "StepMetadata",
    "PipelineExecutor",
    "ExecutionResult",
    "StorageLayer",
    "LocalStorageBackend",
    "FunctionAnalyzer",
    "FunctionSignature",
    "StorageDependency",

    # Registry entries
    "PipelineRegistryEntry",
    "ModelRegistryEntry",
    "ArtifactRegistryEntry",
    "DeploymentRegistryEntry",

    # Exceptions
    "Ops0Error",
    "PipelineError",
    "StepError",
    "StorageError",
    "ExecutionError",
    "DependencyError",
    "ValidationError",
    "RegistryError",
    "ConfigurationError",
    "SerializationError",

    # Metadata
    "__version__",
]

# Module-level configuration
import logging
import os

# Configure ops0 logging
_log_level = os.getenv("OPS0_LOG_LEVEL", "INFO").upper()
_logger = logging.getLogger("ops0")
_logger.setLevel(getattr(logging, _log_level, logging.INFO))

# Add console handler if none exists
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(asctime)s - ops0.%(name)s - %(levelname)s - %(message)s"
    )
    _handler.setFormatter(_formatter)
    _logger.addHandler(_handler)

# Development mode indicators
_dev_mode = os.getenv("OPS0_ENV", "").lower() == "development"
if _dev_mode:
    _logger.info("🐍 ops0 core loaded in development mode")


# Module validation
def _validate_core_installation():
    """Validate that core components are properly installed"""
    try:
        # Test basic imports
        from .decorators import step, pipeline
        from .storage import storage
        from .executor import run, deploy
        from .graph import PipelineGraph
        from .registry import registry

        # Test basic functionality
        with PipelineGraph("validation-test") as test_pipeline:
            @step
            def test_step():
                return "validation successful"

        # Test storage
        storage.save("test_key", "test_value")
        assert storage.load("test_key") == "test_value"
        storage.delete("test_key")

        # Test registry
        stats = registry.get_stats()
        assert isinstance(stats, dict)

        return True

    except Exception as e:
        if _dev_mode:
            _logger.warning(f"Core validation failed: {e}")
        return False


# Run validation in development mode
if _dev_mode and not _validate_core_installation():
    _logger.warning("⚠️  Core validation failed - some features may not work")


# Expose commonly used patterns
def quick_pipeline(name: str = "quick-pipeline"):
    """
    Quick pipeline context for rapid prototyping.

    Args:
        name: Pipeline name

    Returns:
        Pipeline context manager

    Example:
        with ops0.core.quick_pipeline("test"):
            @ops0.step
            def my_step():
                return "hello world"
    """
    return PipelineGraph(name)


def quick_run(pipeline_func: callable = None, **kwargs):
    """
    Quick execution of a pipeline function.

    Args:
        pipeline_func: Function containing @ops0.step decorated functions
        **kwargs: Additional arguments for execution

    Returns:
        Execution result

    Example:
        def my_pipeline():
            @ops0.step
            def step1():
                return "result"

        result = ops0.core.quick_run(my_pipeline)
    """
    if pipeline_func:
        pipeline_name = pipeline_func.__name__
        with PipelineGraph(pipeline_name) as pipeline:
            pipeline_func()
        return run_local(pipeline, **kwargs)
    else:
        # Use current pipeline context
        current = get_current_pipeline()
        if current:
            return run_local(current, **kwargs)
        else:
            raise PipelineError("No pipeline found - provide pipeline_func or use pipeline context")


def get_or_create_pipeline(name: str) -> PipelineGraph:
    """
    Get existing pipeline or create new one.

    Args:
        name: Pipeline name

    Returns:
        Pipeline graph instance
    """
    current = get_current_pipeline()
    if current and current.name == name:
        return current
    return PipelineGraph(name)


# Convenience functions for common ML workflows
def ml_pipeline(name: str):
    """
    Decorator for ML pipeline functions.

    Args:
        name: Pipeline name

    Example:
        @ops0.core.ml_pipeline("training")
        def my_training_pipeline():
            @ops0.step
            def load_data():
                pass

            @ops0.step
            def train_model():
                pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with PipelineGraph(name) as pipeline:
                result = func(*args, **kwargs)
                # Auto-register pipeline in registry
                try:
                    import inspect
                    source = inspect.getsource(func)
                    register_pipeline(
                        name=name,
                        pipeline_source=source,
                        steps=list(pipeline.steps.keys()),
                        description=func.__doc__ or f"ML pipeline: {name}"
                    )
                except Exception as e:
                    _logger.debug(f"Could not auto-register pipeline {name}: {e}")
                return result
        return wrapper
    return decorator


# Auto-configuration on import
try:
    # Configure logging based on config
    configure_logging()

    # Initialize storage backend if needed
    storage_path = config.storage.storage_path
    if not os.path.exists(storage_path):
        os.makedirs(storage_path, exist_ok=True)
        if _dev_mode:
            _logger.debug(f"Created storage directory: {storage_path}")

except Exception as e:
    if _dev_mode:
        _logger.warning(f"Auto-configuration warning: {e}")


# Development helpers
if _dev_mode:
    def debug_info():
        """Print debug information about ops0 core"""
        print("🔍 ops0 Core Debug Information")
        print("=" * 40)
        print(f"Version: {__version__}")
        print(f"Config file: {getattr(config_manager, 'config_file_path', 'None')}")
        print(f"Storage path: {config.storage.storage_path}")
        print(f"Environment: {config.environment}")
        print(f"Debug mode: {config.development.debug_mode}")
        print(f"Registry stats: {registry.get_stats()}")
        print()

        # Test basic functionality
        try:
            with quick_pipeline("debug-test"):
                @step
                def debug_step():
                    storage.save("debug", "working")
                    return "debug complete"
            print("✅ Basic functionality test passed")
        except Exception as e:
            print(f"❌ Basic functionality test failed: {e}")

    # Add to __all__ in dev mode
    __all__.append("debug_info")


# Module initialization message
if _dev_mode:
    _logger.info(f"✨ ops0 core v{__version__} ready - {len(__all__)} components available")
else:
    _logger.debug(f"ops0 core v{__version__} initialized")


# Export version for compatibility
def get_version():
    """Get ops0 core version"""
    return __version__


# Validate API completeness
def _check_api_completeness():
    """Ensure all promised features are available"""
    required_features = [
        "step", "pipeline", "run", "storage", "registry",
        "register_model", "get_latest_model"
    ]

    missing = []
    for feature in required_features:
        if feature not in globals():
            missing.append(feature)

    if missing and _dev_mode:
        _logger.warning(f"Missing API features: {missing}")

    return len(missing) == 0


# Run API check in development
if _dev_mode:
    _check_api_completeness()