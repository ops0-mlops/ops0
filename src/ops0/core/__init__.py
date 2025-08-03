"""
ops0 Core Module

The heart of ops0 - Python-Native ML Pipeline Orchestration.
Provides decorators, storage, and execution primitives.
"""

from .decorators import step, pipeline, StepMetadata
from .storage import storage, StorageLayer, LocalStorageBackend
from .executor import run, deploy, PipelineExecutor
from .graph import PipelineGraph, StepNode, get_current_pipeline
from .analyzer import FunctionAnalyzer
from .exceptions import (
    Ops0Error,
    PipelineError,
    StepError,
    StorageError,
    ExecutionError,
    DependencyError,
    ValidationError
)

# Version compatibility
try:
    from ...__about__ import __version__
except ImportError:
    __version__ = "0.1.0-dev"

__all__ = [
    # Core decorators
    "step",
    "pipeline",

    # Execution
    "run",
    "deploy",

    # Storage
    "storage",

    # Pipeline management
    "get_current_pipeline",

    # Advanced classes (for library users)
    "PipelineGraph",
    "StepNode",
    "StepMetadata",
    "PipelineExecutor",
    "StorageLayer",
    "LocalStorageBackend",
    "FunctionAnalyzer",

    # Exceptions
    "Ops0Error",
    "PipelineError",
    "StepError",
    "StorageError",
    "ExecutionError",
    "DependencyError",
    "ValidationError",

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

        # Test basic functionality
        with PipelineGraph("validation-test") as test_pipeline:
            @step
            def test_step():
                return "validation successful"

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

    Example:
        with ops0.quick_pipeline():
            @ops0.step
            def my_step():
                return "Hello ops0!"

            result = ops0.run()
    """
    return PipelineGraph(name)


# Add to __all__
__all__.append("quick_pipeline")

# Module docstring for help()
__doc__ = f"""
ops0 Core Module v{__version__}

🐍⚡ Python-Native ML Pipeline Orchestration

Key Components:
  @step       - Transform functions into pipeline steps
  @pipeline   - Define pipeline contexts  
  storage     - Transparent data sharing between steps
  run()       - Execute pipelines locally or distributed
  deploy()    - Deploy to production with zero config

Quick Start:
  import ops0

  @ops0.step
  def my_step():
      return "Hello ops0!"

  with ops0.pipeline("my-first-pipeline"):
      my_step()
      result = ops0.run()

Documentation: https://docs.ops0.xyz
"""