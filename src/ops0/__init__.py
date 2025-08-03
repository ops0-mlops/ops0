"""
ops0 - Python-Native ML Pipeline Orchestration

Write Python. Ship Production. Forget the Infrastructure.
"""

from core.decorators import step, pipeline
from core.storage import storage
from core.executor import run, deploy
from core.graph import get_current_pipeline
from runtime.containers import container_orchestrator

from __about__ import __version__

__all__ = [
    "step",
    "pipeline",
    "storage",
    "run",
    "deploy",
    "get_current_pipeline",
    "container_orchestrator",
    "__version__",
]