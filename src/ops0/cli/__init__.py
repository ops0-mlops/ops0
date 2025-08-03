"""
ops0 CLI Module

Command Line Interface for ops0 - Python-Native ML Pipeline Orchestration
"""

from .main import app
from .config import config
from .utils import (
    console,
    print_success,
    print_error,
    print_info,
    print_warning,
    confirm_action,
    get_current_pipeline_info,
    format_duration,
    format_memory,
    show_pipeline_tree,
    ProgressTracker,
)
from .commands import (
    create_project,
    show_pipeline_status,
    containerize_current_pipeline,
    debug_pipeline_execution,
    PROJECT_TEMPLATES,
)

__all__ = [
    "app",
    "config",
    "console",
    "print_success",
    "print_error",
    "print_info",
    "print_warning",
    "confirm_action",
    "get_current_pipeline_info",
    "format_duration",
    "format_memory",
    "show_pipeline_tree",
    "ProgressTracker",
    "create_project",
    "show_pipeline_status",
    "containerize_current_pipeline",
    "debug_pipeline_execution",
    "PROJECT_TEMPLATES",
]