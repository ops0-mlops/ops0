"""List command for ops0 - list pipelines, steps, and resources."""

import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from ..core import Pipeline
from ..core.config import Config
from ..registry import Registry

from .utils import (
    ensure_project_initialized,
    print_info,
    format_time_ago,
    format_duration,
    format_bytes
)

console = Console()


def list_cmd(
        resource: str = typer.Argument("pipelines", help="Resource to list (pipelines, steps, models, deployments)"),
        all: bool = typer.Option(False, "--all", "-a", help="Show all resources across environments"),
        env: str = typer.Option(None, "--env", "-e", help="Filter by environment"),
        format: str = typer.Option("table", "--format", "-f", help="Output format (table, json, yaml)"),
        filter: Optional[str] = typer.Option(None, "--filter", help="Filter results by name pattern"),
        sort: str = typer.Option("name", "--sort", "-s", help="Sort by field (name, created, updated, size)"),
        reverse: bool = typer.Option(False, "--reverse", "-r", help="Reverse sort order"),
        limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Limit number of results"),
):
    """
    List ops0 resources.

    Examples:
        ops0 list                    # List all pipelines
        ops0 list steps             # List all steps
        ops0 list models            # List saved models
        ops0 list deployments       # List deployments
        ops0 list --all             # Show all environments
        ops0 list --format json     # Output as JSON
    """
    # Ensure we're in an ops0 project
    project_root = ensure_project_initialized()

    # Get registry
    registry = Registry(project_root)

    # List appropriate resource
    if resource in ["pipelines", "pipeline"]:
        items = list_pipelines(registry, env, all)
    elif resource in ["steps", "step"]:
        items = list_steps(registry)
    elif resource in ["models", "model"]:
        items = list_models(registry)
    elif resource in ["deployments", "deployment"]:
        items = list_deployments(registry, env, all)
    else:
        console.print(f"[red]Unknown resource type: {resource}[/red]")
        console.print("Available resources: pipelines, steps, models, deployments")
        raise typer.Exit(1)

    # Apply filters
    if filter:
        items = [item for item in items if filter.lower() in item.get('name', '').lower()]

    # Sort items
    if sort in ["created", "updated", "size"]:
        items.sort(key=lambda x: x.get(sort, 0), reverse=not reverse)
    else:
        items.sort(key=lambda x: x.get('name', ''), reverse=reverse)

    # Limit results
    if limit:
        items = items[:limit]

    # Display results
    if not items:
        print_info(f"No {resource} found")
        return

    if format == "json":
        console.print(json.dumps(items, indent=2))
    elif format == "yaml":
        import yaml
        console.print(yaml.dump(items, default_flow_style=False))
    else:
        display_items_table(resource, items)


def list_pipelines(registry: Registry, env: Optional[str], show_all: bool) -> List[Dict[str, Any]]:
    """List all pipelines."""
    # Mock data for demonstration
    pipelines = [
        {
            "name": "fraud-detector",
            "version": "0.2.0",
            "environment": "production",
            "status": "running",
            "created": time.time() - 30 * 86400,
            "updated": time.time() - 3600,
            "steps": 4,
            "schedule": "*/30 * * * *"
        },
        {
            "name": "customer-churn",
            "version": "0.1.5",
            "environment": "production",
            "status": "idle",
            "created": time.time() - 45 * 86400,
            "updated": time.time() - 7200,
            "steps": 6,
            "schedule": "0 0 * * *"
        },
        {
            "name": "recommendation-engine",
            "version": "1.0.0",
            "environment": "staging",
            "status": "stopped",
            "created": time.time() - 10 * 86400,
            "updated": time.time() - 86400,
            "steps": 8,
            "schedule": None
        }
    ]

    # Filter by environment
    if env and not show_all:
        pipelines = [p for p in pipelines if p['environment'] == env]

    return pipelines


def list_steps(registry: Registry) -> List[Dict[str, Any]]:
    """List all available steps."""
    # Mock data for demonstration
    steps = [
        {
            "name": "load_data",
            "pipeline": "fraud-detector",
            "type": "data_loader",
            "inputs": ["file_path: str"],
            "outputs": ["pd.DataFrame"],
            "cached": True,
            "avg_duration": 5.2
        },
        {
            "name": "preprocess",
            "pipeline": "fraud-detector",
            "type": "transformer",
            "inputs": ["data: pd.DataFrame"],
            "outputs": ["pd.DataFrame"],
            "cached": True,
            "avg_duration": 12.3
        },
        {
            "name": "train_model",
            "pipeline": "fraud-detector",
            "type": "trainer",
            "inputs": ["data: pd.DataFrame", "model_name: str"],
            "outputs": ["dict"],
            "cached": False,
            "avg_duration": 145.7
        },
        {
            "name": "evaluate",
            "pipeline": "fraud-detector",
            "type": "evaluator",
            "inputs": ["metrics: dict"],
            "outputs": ["dict"],
            "cached": False,
            "avg_duration": 3.1
        },
        {
            "name": "feature_engineering",
            "pipeline": "customer-churn",
            "type": "transformer",
            "inputs": ["data: pd.DataFrame"],
            "outputs": ["pd.DataFrame"],
            "cached": True,
            "avg_duration": 23.5
        }
    ]

    return steps


def list_models(registry: Registry) -> List[Dict[str, Any]]:
    """List all saved models."""
    # Mock data for demonstration
    models = [
        {
            "name": "fraud-detector-rf",
            "version": "0.2.0",
            "type": "RandomForestClassifier",
            "size": 15 * 1024 * 1024,  # 15 MB
            "created": time.time() - 3600,
            "accuracy": 0.94,
            "tags": ["production", "fraud"]
        },
        {
            "name": "fraud-detector-xgb",
            "version": "0.1.8",
            "type": "XGBClassifier",
            "size": 8 * 1024 * 1024,  # 8 MB
            "created": time.time() - 86400,
            "accuracy": 0.92,
            "tags": ["staging", "fraud"]
        },
        {
            "name": "churn-predictor",
            "version": "1.0.0",
            "type": "LogisticRegression",
            "size": 2 * 1024 * 1024,  # 2 MB
            "created": time.time() - 7 * 86400,
            "accuracy": 0.87,
            "tags": ["production", "churn"]
        },
        {
            "name": "recommendation-als",
            "version": "0.5.0",
            "type": "ALSModel",
            "size": 125 * 1024 * 1024,  # 125 MB
            "created": time.time() - 14 * 86400,
            "accuracy": None,
            "tags": ["experimental"]
        }
    ]

    return models


def list_deployments(registry: Registry, env: Optional[str], show_all: bool) -> List[Dict[str, Any]]:
    """List all deployments."""
    # Mock data for demonstration
    deployments = [
        {
            "name": "fraud-detector-prod",
            "pipeline": "fraud-detector",
            "version": "0.2.0",
            "environment": "production",
            "status": "healthy",
            "replicas": "3/3",
            "cpu": "45%",
            "memory": "512Mi",
            "created": time.time() - 3600,
            "endpoint": "https://fraud-detector.ops0.xyz"
        },
        {
            "name": "customer-churn-prod",
            "pipeline": "customer-churn",
            "version": "0.1.5",
            "environment": "production",
            "status": "healthy",
            "replicas": "2/2",
            "cpu": "12%",
            "memory": "256Mi",
            "created": time.time() - 86400,
            "endpoint": "https://customer-churn.ops0.xyz"
        },
        {
            "name": "fraud-detector-staging",
            "pipeline": "fraud-detector",
            "version": "0.2.1-beta",
            "environment": "staging",
            "status": "deploying",
            "replicas": "1/3",
            "cpu": "78%",
            "memory": "768Mi",
            "created": time.time() - 300,
            "endpoint": "https://fraud-detector-staging.ops0.xyz"
        }
    ]

    # Filter by environment
    if env and not show_all:
        deployments = [d for d in deployments if d['environment'] == env]

    return deployments


def display_items_table(resource_type: str, items: List[Dict[str, Any]]):
    """Display items in a table format."""
    if resource_type in ["pipelines", "pipeline"]:
        display_pipelines_table(items)
    elif resource_type in ["steps", "step"]:
        display_steps_table(items)
    elif resource_type in ["models", "model"]:
        display_models_table(items)
    elif resource_type in ["deployments", "deployment"]:
        display_deployments_table(items)


def display_pipelines_table(pipelines: List[Dict[str, Any]]):
    """Display pipelines in a table."""
    table = Table(title="Pipelines", box=box.ROUNDED)
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Environment")
    table.add_column("Status", justify="center")
    table.add_column("Steps")
    table.add_column("Schedule")
    table.add_column("Last Updated")

    for p in pipelines:
        status_style = "green" if p['status'] == "running" else "yellow"

        table.add_row(
            p['name'],
            p['version'],
            p['environment'],
            f"[{status_style}]{p['status']}[/{status_style}]",
            str(p['steps']),
            p['schedule'] or "manual",
            format_time_ago(p['updated'])
        )

    console.print(table)


def display_steps_table(steps: List[Dict[str, Any]]):
    """Display steps in a table."""
    table = Table(title="Steps", box=box.ROUNDED)
    table.add_column("Name", style="cyan")
    table.add_column("Pipeline")
    table.add_column("Type")
    table.add_column("Inputs")
    table.add_column("Outputs")
    table.add_column("Cached")
    table.add_column("Avg Duration")

    for s in steps:
        table.add_row(
            s['name'],
            s['pipeline'],
            s['type'],
            ", ".join(s['inputs']),
            ", ".join(s['outputs']),
            "✓" if s['cached'] else "✗",
            format_duration(s['avg_duration'])
        )

    console.print(table)


def display_models_table(models: List[Dict[str, Any]]):
    """Display models in a table."""
    table = Table(title="Models", box=box.ROUNDED)
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Type")
    table.add_column("Size")
    table.add_column("Accuracy")
    table.add_column("Tags")
    table.add_column("Created")

    for m in models:
        accuracy = f"{m['accuracy']:.2%}" if m['accuracy'] else "N/A"

        table.add_row(
            m['name'],
            m['version'],
            m['type'],
            format_bytes(m['size']),
            accuracy,
            ", ".join(m['tags']),
            format_time_ago(m['created'])
        )

    console.print(table)


def display_deployments_table(deployments: List[Dict[str, Any]]):
    """Display deployments in a table."""
    table = Table(title="Deployments", box=box.ROUNDED)
    table.add_column("Name", style="cyan")
    table.add_column("Pipeline")
    table.add_column("Version")
    table.add_column("Environment")
    table.add_column("Status", justify="center")
    table.add_column("Replicas")
    table.add_column("CPU")
    table.add_column("Memory")
    table.add_column("Endpoint")

    for d in deployments:
        status_style = "green" if d['status'] == "healthy" else "yellow"

        table.add_row(
            d['name'],
            d['pipeline'],
            d['version'],
            d['environment'],
            f"[{status_style}]{d['status']}[/{status_style}]",
            d['replicas'],
            d['cpu'],
            d['memory'],
            d['endpoint']
        )

    console.print(table)


class Registry:
    """Mock registry for demonstration."""

    def __init__(self, project_root: Path):
        self.project_root = project_root