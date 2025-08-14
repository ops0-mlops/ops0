"""Logs command for ops0 - view pipeline logs."""

import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from rich.panel import Panel
from rich import box

from ..core.config import Config
from ..runtime.monitoring import LogCollector

from .utils import (
    ensure_project_initialized,
    print_error,
    print_info,
    format_time_ago
)

console = Console()


def logs(
        pipeline: Optional[str] = typer.Argument(None, help="Pipeline name"),
        step: Optional[str] = typer.Option(None, "--step", "-s", help="Filter by step name"),
        level: Optional[str] = typer.Option(None, "--level", "-l",
                                            help="Filter by log level (DEBUG, INFO, WARNING, ERROR)"),
        since: Optional[str] = typer.Option(None, "--since", help="Show logs since (e.g., 1h, 30m, 24h)"),
        tail: Optional[int] = typer.Option(None, "--tail", "-n", help="Number of lines to show"),
        follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
        json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
        grep: Optional[str] = typer.Option(None, "--grep", "-g", help="Filter logs by pattern"),
        no_color: bool = typer.Option(False, "--no-color", help="Disable color output"),
):
    """
    View logs from ops0 pipelines.

    Examples:
        ops0 logs                        # Show all recent logs
        ops0 logs my-pipeline           # Show logs for specific pipeline
        ops0 logs --step preprocess    # Show logs for specific step
        ops0 logs --level ERROR        # Show only errors
        ops0 logs --since 1h           # Show logs from last hour
        ops0 logs --follow             # Follow log output
        ops0 logs --grep "error"       # Search for pattern
    """
    # Ensure we're in an ops0 project
    project_root = ensure_project_initialized()

    # Parse time filter
    since_timestamp = parse_since(since) if since else None

    # Get log collector
    collector = LogCollector(project_root)

    # Get logs
    log_entries = collector.get_logs(
        pipeline=pipeline,
        step=step,
        level=level,
        since=since_timestamp,
        limit=tail,
        pattern=grep
    )

    if not log_entries and not follow:
        print_info("No logs found matching criteria")
        return

    # JSON output
    if json_output:
        for entry in log_entries:
            console.print(json.dumps(entry.to_dict()))
        if follow:
            follow_logs_json(collector, pipeline, step, level, grep)
        return

    # Display logs
    if follow:
        # Show existing logs first
        display_logs(log_entries, no_color)
        # Then follow new logs
        follow_logs(collector, pipeline, step, level, grep, no_color)
    else:
        display_logs(log_entries, no_color)


def parse_since(since_str: str) -> float:
    """Parse time duration string to timestamp."""
    now = time.time()

    # Parse duration
    if since_str.endswith('s'):
        seconds = int(since_str[:-1])
    elif since_str.endswith('m'):
        seconds = int(since_str[:-1]) * 60
    elif since_str.endswith('h'):
        seconds = int(since_str[:-1]) * 3600
    elif since_str.endswith('d'):
        seconds = int(since_str[:-1]) * 86400
    else:
        # Try to parse as number of minutes
        seconds = int(since_str) * 60

    return now - seconds


def display_logs(entries: List['LogEntry'], no_color: bool):
    """Display log entries."""
    if not entries:
        return

    # Group by pipeline for better display
    by_pipeline = {}
    for entry in entries:
        if entry.pipeline not in by_pipeline:
            by_pipeline[entry.pipeline] = []
        by_pipeline[entry.pipeline].append(entry)

    for pipeline_name, pipeline_entries in by_pipeline.items():
        if len(by_pipeline) > 1:
            console.print(f"\n[bold]{pipeline_name}[/bold]")
            console.print("─" * 50)

        for entry in pipeline_entries:
            display_log_entry(entry, no_color)


def display_log_entry(entry: 'LogEntry', no_color: bool):
    """Display a single log entry."""
    # Format timestamp
    timestamp = datetime.fromtimestamp(entry.timestamp).strftime("%H:%M:%S")

    # Get level style
    level_styles = {
        "DEBUG": "dim",
        "INFO": "blue",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold red"
    }
    level_style = level_styles.get(entry.level, "white")

    # Format log line
    if no_color:
        line = f"{timestamp} [{entry.level:>8}] {entry.step:>15} | {entry.message}"
        console.print(line)
    else:
        # Colored output
        console.print(
            f"[dim]{timestamp}[/dim] "
            f"[{level_style}]{entry.level:>8}[/{level_style}] "
            f"[cyan]{entry.step:>15}[/cyan] | "
            f"{entry.message}"
        )

    # Show details if present
    if entry.details:
        if isinstance(entry.details, dict):
            for key, value in entry.details.items():
                console.print(f"    {key}: {value}", style="dim")
        else:
            console.print(f"    {entry.details}", style="dim")

    # Show traceback for errors
    if entry.traceback:
        if no_color:
            console.print(entry.traceback)
        else:
            syntax = Syntax(entry.traceback, "python", theme="monokai", line_numbers=True)
            console.print(syntax)


def follow_logs(collector: LogCollector, pipeline: Optional[str],
                step: Optional[str], level: Optional[str],
                pattern: Optional[str], no_color: bool):
    """Follow log output in real-time."""
    console.print("\n[blue]Following logs... (Press Ctrl+C to stop)[/blue]\n")

    last_timestamp = time.time()

    try:
        while True:
            # Get new logs
            new_entries = collector.get_logs(
                pipeline=pipeline,
                step=step,
                level=level,
                since=last_timestamp,
                pattern=pattern
            )

            # Display new entries
            for entry in new_entries:
                display_log_entry(entry, no_color)
                last_timestamp = max(last_timestamp, entry.timestamp)

            time.sleep(0.5)  # Poll interval

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped following logs[/yellow]")


def follow_logs_json(collector: LogCollector, pipeline: Optional[str],
                     step: Optional[str], level: Optional[str],
                     pattern: Optional[str]):
    """Follow logs in JSON format."""
    last_timestamp = time.time()

    try:
        while True:
            new_entries = collector.get_logs(
                pipeline=pipeline,
                step=step,
                level=level,
                since=last_timestamp,
                pattern=pattern
            )

            for entry in new_entries:
                console.print(json.dumps(entry.to_dict()))
                last_timestamp = max(last_timestamp, entry.timestamp)

            time.sleep(0.5)

    except KeyboardInterrupt:
        pass


class LogCollector:
    """Mock log collector for demonstration."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.log_dir = project_root / ".ops0" / "logs"

    def get_logs(self, pipeline: Optional[str] = None,
                 step: Optional[str] = None,
                 level: Optional[str] = None,
                 since: Optional[float] = None,
                 limit: Optional[int] = None,
                 pattern: Optional[str] = None) -> List['LogEntry']:
        """Get log entries matching criteria."""
        # Mock implementation - return sample logs
        entries = [
            LogEntry(
                timestamp=time.time() - 300,
                level="INFO",
                pipeline=pipeline or "fraud-detector",
                step="load_data",
                message="Loading data from s3://ops0-data/fraud/train.csv"
            ),
            LogEntry(
                timestamp=time.time() - 295,
                level="INFO",
                pipeline=pipeline or "fraud-detector",
                step="load_data",
                message="Loaded 50000 rows, 25 columns"
            ),
            LogEntry(
                timestamp=time.time() - 290,
                level="DEBUG",
                pipeline=pipeline or "fraud-detector",
                step="preprocess",
                message="Starting data preprocessing",
                details={"null_values": 234, "duplicates": 12}
            ),
            LogEntry(
                timestamp=time.time() - 280,
                level="WARNING",
                pipeline=pipeline or "fraud-detector",
                step="preprocess",
                message="Found 234 missing values, will be imputed"
            ),
            LogEntry(
                timestamp=time.time() - 270,
                level="INFO",
                pipeline=pipeline or "fraud-detector",
                step="train_model",
                message="Training RandomForest model",
                details={"n_estimators": 100, "max_depth": 10}
            ),
            LogEntry(
                timestamp=time.time() - 200,
                level="ERROR",
                pipeline=pipeline or "fraud-detector",
                step="train_model",
                message="Model training failed: insufficient memory",
                traceback="""Traceback (most recent call last):
  File "/app/pipeline.py", line 45, in train_model
    model.fit(X_train, y_train)
  File "/usr/local/lib/python3.9/site-packages/sklearn/ensemble/_forest.py", line 456, in fit
    trees = Parallel(n_jobs=self.n_jobs, **joblib_parallel_args)(
MemoryError: Unable to allocate array with shape (40000, 25) and data type float64"""
            ),
            LogEntry(
                timestamp=time.time() - 180,
                level="INFO",
                pipeline=pipeline or "fraud-detector",
                step="train_model",
                message="Retrying with reduced batch size"
            ),
            LogEntry(
                timestamp=time.time() - 150,
                level="INFO",
                pipeline=pipeline or "fraud-detector",
                step="train_model",
                message="Model training completed successfully",
                details={"accuracy": 0.94, "precision": 0.92, "recall": 0.88}
            ),
        ]

        # Filter entries
        filtered = entries

        if step:
            filtered = [e for e in filtered if e.step == step]

        if level:
            filtered = [e for e in filtered if e.level == level.upper()]

        if since:
            filtered = [e for e in filtered if e.timestamp >= since]

        if pattern:
            filtered = [e for e in filtered if pattern.lower() in e.message.lower()]

        if limit:
            filtered = filtered[-limit:]

        return filtered


class LogEntry:
    """Log entry data."""

    def __init__(self, timestamp: float, level: str, pipeline: str,
                 step: str, message: str, details: Optional[Dict[str, Any]] = None,
                 traceback: Optional[str] = None):
        self.timestamp = timestamp
        self.level = level
        self.pipeline = pipeline
        self.step = step
        self.message = message
        self.details = details
        self.traceback = traceback

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "timestamp": self.timestamp,
            "level": self.level,
            "pipeline": self.pipeline,
            "step": self.step,
            "message": self.message
        }

        if self.details:
            data["details"] = self.details

        if self.traceback:
            data["traceback"] = self.traceback

        return data