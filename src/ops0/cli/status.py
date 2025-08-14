"""Status command for ops0 - show pipeline status and metrics."""

import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich import box
from datetime import datetime, timedelta

from ..core import Pipeline
from ..core.config import Config
from ..runtime.monitoring import MetricsCollector

from .utils import (
    ensure_project_initialized,
    print_error,
    print_info,
    format_duration,
    format_time_ago,
    format_bytes
)

console = Console()


def status(
        pipeline: Optional[str] = typer.Argument(None, help="Pipeline name"),
        watch: bool = typer.Option(False, "--watch", "-w", help="Watch status in real-time"),
        interval: int = typer.Option(5, "--interval", "-i", help="Watch interval in seconds"),
        detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed metrics"),
        json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    Show status of ops0 pipelines.

    Examples:
        ops0 status                  # Show all pipelines
        ops0 status my-pipeline     # Show specific pipeline
        ops0 status --watch        # Watch status in real-time
        ops0 status --detailed     # Show detailed metrics
    """
    # Ensure we're in an ops0 project
    project_root = ensure_project_initialized()

    # Get pipeline status
    if pipeline:
        status_data = get_pipeline_status(pipeline)
        if not status_data:
            print_error(f"Pipeline '{pipeline}' not found")
            raise typer.Exit(1)
        pipelines = [status_data]
    else:
        pipelines = get_all_pipeline_status()
        if not pipelines:
            print_info("No pipelines found. Deploy a pipeline first with 'ops0 deploy'")
            return

    # JSON output
    if json_output:
        import json
        console.print(json.dumps([p.to_dict() for p in pipelines], indent=2))
        return

    # Display status
    if watch:
        watch_status(pipelines, interval, detailed)
    else:
        show_status(pipelines, detailed)


def get_pipeline_status(name: str) -> Optional['PipelineStatus']:
    """Get status of a specific pipeline."""
    # In a real implementation, this would query the runtime
    # For now, we'll return mock data
    return PipelineStatus(
        name=name,
        status="running",
        version="0.1.0",
        environment="production",
        last_run=time.time() - 300,  # 5 minutes ago
        next_run=time.time() + 3300,  # 55 minutes from now
        success_rate=0.98,
        avg_duration=45.5,
        total_runs=1234,
        failed_runs=25,
        resources={
            "cpu": "45%",
            "memory": "512Mi / 1Gi",
            "disk": "2.3Gi"
        },
        steps=[
            StepStatus("load_data", "completed", 5.2, time.time() - 280),
            StepStatus("preprocess", "completed", 12.3, time.time() - 275),
            StepStatus("train_model", "running", 28.0, time.time() - 262),
            StepStatus("evaluate", "pending", 0, 0),
        ]
    )


def get_all_pipeline_status() -> List['PipelineStatus']:
    """Get status of all pipelines."""
    # Mock data for demonstration
    return [
        get_pipeline_status("fraud-detector"),
        PipelineStatus(
            name="customer-churn",
            status="idle",
            version="0.2.1",
            environment="production",
            last_run=time.time() - 7200,
            next_run=time.time() + 600,
            success_rate=0.95,
            avg_duration=120.5,
            total_runs=856,
            failed_runs=43,
            resources={"cpu": "0%", "memory": "0Mi / 1Gi", "disk": "1.2Gi"},
            steps=[]
        )
    ]


def show_status(pipelines: List['PipelineStatus'], detailed: bool):
    """Display pipeline status."""
    console.print("\n[bold]Pipeline Status[/bold]\n")

    # Summary table
    table = Table(box=box.ROUNDED)
    table.add_column("Pipeline", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Version")
    table.add_column("Environment")
    table.add_column("Last Run")
    table.add_column("Success Rate")

    if detailed:
        table.add_column("Resources")
        table.add_column("Avg Duration")

    for p in pipelines:
        status_style = get_status_style(p.status)
        status_icon = get_status_icon(p.status)

        row = [
            p.name,
            f"[{status_style}]{status_icon} {p.status}[/{status_style}]",
            p.version,
            p.environment,
            format_time_ago(p.last_run),
            f"{p.success_rate:.1%}"
        ]

        if detailed:
            row.extend([
                p.resources.get('cpu', 'N/A'),
                format_duration(p.avg_duration)
            ])

        table.add_row(*row)

    console.print(table)

    # Detailed view for each pipeline
    if detailed:
        for p in pipelines:
            show_pipeline_details(p)


def show_pipeline_details(pipeline: 'PipelineStatus'):
    """Show detailed pipeline information."""
    # Pipeline info panel
    info_lines = [
        f"[bold]Total Runs:[/bold] {pipeline.total_runs}",
        f"[bold]Failed Runs:[/bold] {pipeline.failed_runs}",
        f"[bold]Next Run:[/bold] {format_time_until(pipeline.next_run)}",
        f"[bold]Resources:[/bold]",
        f"  • CPU: {pipeline.resources.get('cpu', 'N/A')}",
        f"  • Memory: {pipeline.resources.get('memory', 'N/A')}",
        f"  • Disk: {pipeline.resources.get('disk', 'N/A')}"
    ]

    console.print(f"\n[bold]{pipeline.name}[/bold]")
    console.print(Panel("\n".join(info_lines), box=box.SIMPLE))

    # Steps table if running
    if pipeline.steps:
        step_table = Table(box=box.SIMPLE, show_header=True)
        step_table.add_column("Step", style="cyan")
        step_table.add_column("Status", justify="center")
        step_table.add_column("Duration")
        step_table.add_column("Started")

        for step in pipeline.steps:
            status_style = get_status_style(step.status)
            status_icon = get_status_icon(step.status)

            step_table.add_row(
                step.name,
                f"[{status_style}]{status_icon} {step.status}[/{status_style}]",
                format_duration(step.duration) if step.duration > 0 else "-",
                format_time_ago(step.started) if step.started > 0 else "-"
            )

        console.print("\n[bold]Current Steps:[/bold]")
        console.print(step_table)


def watch_status(pipelines: List['PipelineStatus'], interval: int, detailed: bool):
    """Watch pipeline status in real-time."""
    console.print(f"[blue]Watching pipeline status (refresh every {interval}s)...[/blue]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                # Refresh pipeline status
                if len(pipelines) == 1:
                    pipelines = [get_pipeline_status(pipelines[0].name)]
                else:
                    pipelines = get_all_pipeline_status()

                # Create display
                display = create_watch_display(pipelines, detailed)
                live.update(display)

                time.sleep(interval)

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped watching[/yellow]")


def create_watch_display(pipelines: List['PipelineStatus'], detailed: bool) -> Table:
    """Create display for watch mode."""
    table = Table(
        title=f"Pipeline Status - {datetime.now().strftime('%H:%M:%S')}",
        box=box.ROUNDED
    )

    table.add_column("Pipeline", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Current Step")
    table.add_column("Progress")
    table.add_column("CPU")
    table.add_column("Memory")

    for p in pipelines:
        status_style = get_status_style(p.status)
        status_icon = get_status_icon(p.status)

        # Find current step
        current_step = "idle"
        progress = ""
        if p.steps:
            for i, step in enumerate(p.steps):
                if step.status == "running":
                    current_step = step.name
                    progress = f"{i + 1}/{len(p.steps)}"
                    break

        table.add_row(
            p.name,
            f"[{status_style}]{status_icon} {p.status}[/{status_style}]",
            current_step,
            progress,
            p.resources.get('cpu', '0%'),
            p.resources.get('memory', '0Mi')
        )

    return table


def get_status_style(status: str) -> str:
    """Get color style for status."""
    styles = {
        "running": "green",
        "completed": "green",
        "failed": "red",
        "error": "red",
        "pending": "yellow",
        "idle": "dim",
        "stopped": "dim"
    }
    return styles.get(status, "white")


def get_status_icon(status: str) -> str:
    """Get icon for status."""
    icons = {
        "running": "▶",
        "completed": "✓",
        "failed": "✗",
        "error": "⚠",
        "pending": "⏸",
        "idle": "◯",
        "stopped": "■"
    }
    return icons.get(status, "?")


def format_time_until(timestamp: float) -> str:
    """Format time until a future timestamp."""
    if timestamp <= 0:
        return "N/A"

    now = time.time()
    if timestamp <= now:
        return "now"

    diff = timestamp - now
    if diff < 60:
        return f"in {int(diff)}s"
    elif diff < 3600:
        return f"in {int(diff / 60)}m"
    elif diff < 86400:
        return f"in {int(diff / 3600)}h"
    else:
        return f"in {int(diff / 86400)}d"


class PipelineStatus:
    """Pipeline status data."""

    def __init__(self, name: str, status: str, version: str, environment: str,
                 last_run: float, next_run: float, success_rate: float,
                 avg_duration: float, total_runs: int, failed_runs: int,
                 resources: Dict[str, str], steps: List['StepStatus']):
        self.name = name
        self.status = status
        self.version = version
        self.environment = environment
        self.last_run = last_run
        self.next_run = next_run
        self.success_rate = success_rate
        self.avg_duration = avg_duration
        self.total_runs = total_runs
        self.failed_runs = failed_runs
        self.resources = resources
        self.steps = steps

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status,
            "version": self.version,
            "environment": self.environment,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "success_rate": self.success_rate,
            "avg_duration": self.avg_duration,
            "total_runs": self.total_runs,
            "failed_runs": self.failed_runs,
            "resources": self.resources,
            "steps": [s.to_dict() for s in self.steps]
        }


class StepStatus:
    """Step status data."""

    def __init__(self, name: str, status: str, duration: float, started: float):
        self.name = name
        self.status = status
        self.duration = duration
        self.started = started

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status,
            "duration": self.duration,
            "started": self.started
        }