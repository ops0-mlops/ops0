"""
ops0 - Python-Native ML Pipeline Orchestration

Write Python. Ship Production. Forget the Infrastructure.

Example:
    import ops0

    @ops0.step
    def preprocess(data):
        return data.dropna()

    @ops0.step
    def train(clean_data):
        model = RandomForestClassifier()
        model.fit(clean_data.X, clean_data.y)
        return model

    # Deploy pipeline
    ops0.deploy()
"""

from core.decorators import step, pipeline
from core.storage import storage
from core.executor import run, deploy
from core.graph import get_current_pipeline
from runtime.containers import container_orchestrator

__version__ = "0.1.0"
__author__ = "ops0 Team"

# Expose main API
__all__ = [
    "step",
    "pipeline",
    "storage",
    "run",
    "deploy",
    "get_current_pipeline",
    "container_orchestrator"
]