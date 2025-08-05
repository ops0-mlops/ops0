"""
ops0 - Python-Native ML Pipeline Orchestration

Write Python. Ship Production. Forget the Infrastructure.
"""
from .runtime.containers import container_orchestrator

# Version import
try:
    from ops0.__about__ import __version__
except ImportError:
    from .__about__ import __version__

# Core imports - handle both development and installed package
try:
    # When running as installed package
    from ..core.decorators import step, pipeline
    from ..core.storage import storage
    from ..core.executor import run, deploy
    from ..core.graph import get_current_pipeline
except ImportError:
    # When running in development mode
    from .core.decorators import step, pipeline
    from .core.storage import storage
    from .core.executor import run, deploy
    from .core.graph import get_current_pipeline

# Runtime imports (lazy loading for performance)
def _import_runtime():
    """Lazy import runtime components to avoid heavy startup"""
    try:
        try:
            from ..runtime.containers import container_orchestrator
        except ImportError:
            from .runtime.containers import container_orchestrator
        return container_orchestrator
    except ImportError:
        return None

# Public API
__all__ = [
    # Core API
    "step",
    "pipeline",
    "storage",
    "run",
    "deploy",
    "get_current_pipeline",

    # Metadata
    "__version__",

    # Advanced (lazy loaded)
    "container_orchestrator",
]

# Lazy attribute access for heavy imports
def __getattr__(name: str):
    if name == "container_orchestrator":
        runtime = _import_runtime()
        if runtime is None:
            raise ImportError(
                "Container orchestrator not available. "
                "Install with: pip install 'ops0[runtime]'"
            )
        return runtime

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Package-level configuration
import logging
import os

# Configure logging
log_level = os.getenv("OPS0_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - ops0 - %(levelname)s - %(message)s"
)

# Package initialization message
if os.getenv("OPS0_ENV") == "development":
    print("🐍 ops0 development mode activated")

# Validate Python version
import sys
if sys.version_info < (3, 9):
    raise RuntimeError(
        f"ops0 requires Python 3.9+, you have {sys.version_info.major}.{sys.version_info.minor}"
    )