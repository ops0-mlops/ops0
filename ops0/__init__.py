"""
ops0 - Write Python, Ship Production 🚀

Zero-configuration MLOps platform that transforms Python functions into 
production-ready ML pipelines automatically.
"""
import sys
import os

__version__ = "0.1.0"
__author__ = "ops0 Contributors"
__license__ = "MIT"

# Core decorators
from ops0.decorators import step, pipeline, monitor

# Storage functions
from ops0.storage import (
    save,
    load,
    save_model,
    load_model,
    save_dataframe,
    load_dataframe,
    list_models,
    delete_model
)

# CLI functions (when used programmatically)
from ops0.cli import main as cli

# Execution utilities
from ops0.executor import ExecutionContext, ExecutionMode

# Storage backends (for advanced usage)
from ops0.storage import StorageBackend, LocalStorage, S3Storage, get_storage, set_storage

# Parser utilities (for advanced usage)
from ops0.parser import analyze_function, build_dag

# All public exports
__all__ = [
    # Core decorators
    'step',
    'pipeline',
    'monitor',

    # Storage functions
    'save',
    'load',
    'save_model',
    'load_model',
    'save_dataframe',
    'load_dataframe',
    'list_models',
    'delete_model',

    # Storage configuration
    'get_storage',
    'set_storage',
    'StorageBackend',
    'LocalStorage',
    'S3Storage',

    # Execution
    'ExecutionContext',
    'ExecutionMode',

    # Analysis
    'analyze_function',
    'build_dag',

    # CLI
    'cli',

    # Version
    '__version__'
]


# Convenience namespace for notifications (future feature)
class notify:
    """Notification integrations (placeholder for future implementation)"""

    @staticmethod
    def slack(message: str, channel: str = None):
        """Send Slack notification"""
        print(f"[Slack notification]: {message}")

    @staticmethod
    def email(message: str, to: str = None):
        """Send email notification"""
        print(f"[Email notification]: {message}")

    @staticmethod
    def pagerduty(message: str, severity: str = "info"):
        """Send PagerDuty alert"""
        print(f"[PagerDuty {severity}]: {message}")


# IMPORTANT: Ne PAS exécuter deploy() au moment de l'import!
# Créer une fonction deploy au lieu d'une instance
def deploy(stage: str = "prod", region: str = "us-east-1"):
    """
    Deploy the current pipeline to cloud infrastructure.

    Args:
        stage: Deployment stage (dev/staging/prod)
        region: AWS region

    Example:
        ops0.deploy()  # Deploy to prod
        ops0.deploy(stage="dev")  # Deploy to dev
    """
    import subprocess
    cmd = [sys.executable, "-m", "ops0.cli", "deploy", "--stage", stage, "--region", region]
    subprocess.run(cmd)


# Logging utilities
class logging:
    """Structured logging for ops0 pipelines"""

    @staticmethod
    def info(message: str, **kwargs):
        """Log info message with optional structured data"""
        import json
        from datetime import datetime

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "INFO",
            "message": message,
            **kwargs
        }
        print(json.dumps(log_entry))

    @staticmethod
    def error(message: str, **kwargs):
        """Log error message with optional structured data"""
        import json
        from datetime import datetime

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "ERROR",
            "message": message,
            **kwargs
        }
        print(json.dumps(log_entry), file=sys.stderr)

    @staticmethod
    def warning(message: str, **kwargs):
        """Log warning message with optional structured data"""
        import json
        from datetime import datetime

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "WARNING",
            "message": message,
            **kwargs
        }
        print(json.dumps(log_entry))


# Environment utilities
def get_execution_context() -> ExecutionContext:
    """Get current execution context"""
    ctx = ExecutionContext.current()
    if not ctx:
        # Create default context
        from ops0.executor import ExecutionContext, ExecutionMode
        ctx = ExecutionContext(mode=ExecutionMode.LOCAL)
    return ctx


def is_production() -> bool:
    """Check if running in production environment"""
    return os.environ.get("OPS0_ENV", "").lower() in ["prod", "production"]


def is_local() -> bool:
    """Check if running locally"""
    ctx = get_execution_context()
    return ctx.mode == ExecutionMode.LOCAL


# Version check - MAIS seulement si on n'est pas en train d'importer
def check_version():
    """Check for ops0 updates"""
    try:
        import requests
        response = requests.get("https://pypi.org/pypi/ops0/json", timeout=2)
        latest = response.json()["info"]["version"]

        if latest != __version__:
            print(f"📦 New version available: {latest} (current: {__version__})")
            print(f"   Run: pip install --upgrade ops0")
    except:
        # Silently fail if can't check
        pass


# Auto-import pour notebooks - seulement si dans un notebook
def _setup_notebook_environment():
    """Setup Jupyter notebook environment"""
    try:
        get_ipython()  # Only available in IPython/Jupyter

        # Lazy imports pour notebooks
        print("🚀 ops0 notebook environment ready!")
        print("   Available: ops0, pd (pandas), np (numpy)")
        print("   Note: pandas and numpy will be imported when first used")

    except NameError:
        # Not in a notebook
        pass


# Ne PAS exécuter automatiquement au moment de l'import
# Ces actions devraient être déclenchées explicitement
if __name__ == "__main__":
    # Seulement si le module est exécuté directement
    _setup_notebook_environment()

    # Check for updates seulement si demandé
    if not os.environ.get("OPS0_NO_UPDATE_CHECK"):
        import threading
        threading.Thread(target=check_version, daemon=True).start()