"""Run command for ops0 - execute pipelines locally."""

import os
import sys
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.live import Live
from rich import box

from ..core import Pipeline
from ..core.config import Config
from ..runtime.local import LocalRuntime
from ..runtime.docker import DockerRuntime

from .utils import (
    ensure_project_initialized,
    print_success,
    print_error,
    print_warning,
    print_info,
    format_duration,
    load_pipeline_config,
    find_step_files
)

console = Console()


def run(
        pipeline: Optional[str] = typer.Argument(None, help="Pipeline name or file"),
        local: bool = typer.Option(True, "--local", "-l", help="Run locally (default)"),
        docker: bool = typer.Option(False, "--docker", "-d", help="Run in Docker containers"),
        steps: Optional[str] = typer.Option(None, "--steps", "-s", help="Run specific steps (comma-separated)"),
        params: Optional[str] = typer.Option(None, "--params", "-p", help="Pipeline parameters (JSON format)"),
        watch: bool = typer.Option(False, "--watch", "-w", help="Watch for file changes and rerun"),
        debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
        env_file: Optional[Path] = typer.Option(None, "--env-file", "-e", help="Environment file"),
        no_cache: bool = typer.Option(False, "--no-cache", help="Disable caching"),
        profile: bool = typer.Option(False, "--profile", help="Enable performance profiling"),
):
    """
    Run an ops0 pipeline locally or in containers.

    Examples:
        ops0 run                    # Run pipeline.py locally
        ops0 run --docker          # Run in Docker containers
        ops0 run --steps preprocess,train  # Run specific steps
        ops0 run --watch           # Watch for changes and rerun
    """
    # Ensure we're in an ops0 project
    project_root = ensure_project_initialized()

    # Load environment variables
    if env_file and env_file.exists():
        load_env_file(env_file)

    # Find and load pipeline
    pipeline_obj = load_pipeline(project_root, pipeline)

    if not pipeline_obj:
        print_error("No pipeline found. Create a pipeline.py file or specify a pipeline file.")
        raise typer.Exit(1)

    # Parse parameters
    params_dict = {}
    if params:
        try:
            params_dict = json.loads(params)
        except json.JSONDecodeError:
            print_error("Invalid JSON parameters")
            raise typer.Exit(1)

    # Filter steps if specified
    if steps:
        step_names = [s.strip() for s in steps.split(",")]
        pipeline_obj = filter_pipeline_steps(pipeline_obj, step_names)

    # Configure runtime
    if docker:
        runtime = DockerRuntime(debug=debug)
        runtime_name = "Docker"
    else:
        runtime = LocalRuntime(debug=debug, no_cache=no_cache)
        runtime_name = "Local"

    console.print(f"\n[bold blue]🚀 Running pipeline: {pipeline_obj.name}[/bold blue]")
    console.print(f"[dim]Runtime: {runtime_name} | Steps: {len(pipeline_obj.steps)}[/dim]\n")

    # Run with watch mode
    if watch:
        run_with_watch(project_root, pipeline_obj, runtime, params_dict, profile)
    else:
        run_once(pipeline_obj, runtime, params_dict, profile)


def load_pipeline(project_root: Path, pipeline_path: Optional[str]) -> Optional[Pipeline]:
    """Load pipeline from file or find default pipeline."""
    if pipeline_path:
        # Load specific pipeline file
        path = Path(pipeline_path)
        if not path.exists():
            path = project_root / pipeline_path

        if path.exists():
            return load_pipeline_from_file(path)
    else:
        # Try to find default pipeline
        default_paths = [
            project_root / "pipeline.py",
            project_root / "main.py",
            project_root / "src" / "pipeline.py"
        ]

        for path in default_paths:
            if path.exists():
                return load_pipeline_from_file(path)

    return None


def load_pipeline_from_file(file_path: Path) -> Pipeline:
    """Load pipeline from Python file."""
    import importlib.util

    # Load module
    spec = importlib.util.spec_from_file_location("pipeline_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find pipeline object
    if hasattr(module, 'pipeline') and isinstance(module.pipeline, Pipeline):
        return module.pipeline

    # Otherwise, create pipeline from decorated functions
    from ..core import create_pipeline_from_module
    return create_pipeline_from_module(module)


def filter_pipeline_steps(pipeline: Pipeline, step_names: List[str]) -> Pipeline:
    """Filter pipeline to only include specified steps."""
    filtered_steps = []

    for step in pipeline.steps:
        if step.name in step_names:
            filtered_steps.append(step)

    if not filtered_steps:
        print_error(f"No matching steps found: {', '.join(step_names)}")
        raise typer.Exit(1)

    pipeline.steps = filtered_steps
    return pipeline


def run_once(pipeline: Pipeline, runtime: Any, params: Dict[str, Any], profile: bool):
    """Run pipeline once."""
    start_time = time.time()

    # Create progress display
    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
            transient=True
    ) as progress:

        task = progress.add_task("Running pipeline...", total=len(pipeline.steps))

        try:
            # Run each step
            results = {}
            for i, step in enumerate(pipeline.steps):
                progress.update(task, description=f"Running {step.name}...")

                # Run step
                step_start = time.time()
                result = runtime.run_step(step, params, results)
                step_duration = time.time() - step_start

                results[step.name] = result

                # Update progress
                progress.update(task, advance=1)
                console.print(
                    f"  [green]✓[/green] {step.name} "
                    f"[dim]({format_duration(step_duration)})[/dim]"
                )

                # Show profiling info if enabled
                if profile:
                    show_step_profile(step.name, step_duration, result)

            # Pipeline completed
            total_duration = time.time() - start_time

            console.print("\n" + Panel(
                f"[green]✅ Pipeline completed successfully![/green]\n\n"
                f"Total time: {format_duration(total_duration)}\n"
                f"Steps run: {len(pipeline.steps)}",
                title="[bold green]Success[/bold green]",
                border_style="green"
            ))

            # Show results summary
            if results:
                show_results_summary(results)

        except Exception as e:
            progress.stop()
            console.print(f"\n[red]❌ Pipeline failed at step: {pipeline.steps[i].name}[/red]")
            console.print(f"[red]Error:[/red] {str(e)}")

            if runtime.debug:
                import traceback
                console.print("\n[dim]Traceback:[/dim]")
                console.print(traceback.format_exc())

            raise typer.Exit(1)


def run_with_watch(project_root: Path, pipeline: Pipeline, runtime: Any,
                   params: Dict[str, Any], profile: bool):
    """Run pipeline with file watching."""
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    class PipelineWatcher(FileSystemEventHandler):
        def __init__(self):
            self.last_run = 0
            self.debounce_seconds = 1

        def on_modified(self, event):
            if event.src_path.endswith('.py'):
                current_time = time.time()
                if current_time - self.last_run > self.debounce_seconds:
                    self.last_run = current_time
                    console.print(f"\n[yellow]File changed: {event.src_path}[/yellow]")
                    console.print("[blue]Rerunning pipeline...[/blue]\n")
                    run_once(pipeline, runtime, params, profile)

    # Initial run
    run_once(pipeline, runtime, params, profile)

    # Set up file watcher
    event_handler = PipelineWatcher()
    observer = Observer()
    observer.schedule(event_handler, project_root, recursive=True)
    observer.start()

    console.print("\n[blue]👁  Watching for file changes... (Press Ctrl+C to stop)[/blue]")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        console.print("\n[yellow]Stopped watching[/yellow]")

    observer.join()


def load_env_file(env_file: Path):
    """Load environment variables from file."""
    from .utils import parse_env_file

    env_vars = parse_env_file(env_file)
    for key, value in env_vars.items():
        os.environ[key] = value

    print_info(f"Loaded {len(env_vars)} environment variables from {env_file}")


def show_step_profile(step_name: str, duration: float, result: Any):
    """Show profiling information for a step."""
    profile_data = {
        "Duration": format_duration(duration),
        "Memory": "Not implemented",  # TODO: Add memory profiling
        "Result Type": type(result).__name__,
    }

    if isinstance(result, (list, dict, str)):
        profile_data["Result Size"] = len(result)

    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")

    for key, value in profile_data.items():
        table.add_row(key, str(value))

    console.print(f"\n  [dim]Profile for {step_name}:[/dim]")
    console.print(table)


def show_results_summary(results: Dict[str, Any]):
    """Show summary of pipeline results."""
    console.print("\n[bold]Pipeline Results:[/bold]")

    table = Table(box=box.ROUNDED)
    table.add_column("Step", style="cyan")
    table.add_column("Result Type")
    table.add_column("Summary")

    for step_name, result in results.items():
        result_type = type(result).__name__

        # Create summary based on result type
        if isinstance(result, dict):
            summary = f"{len(result)} keys"
            if len(result) <= 3:
                summary = ", ".join(f"{k}={v}" for k, v in result.items())
        elif isinstance(result, list):
            summary = f"{len(result)} items"
        elif isinstance(result, str):
            summary = result[:50] + "..." if len(result) > 50 else result
        elif hasattr(result, 'shape'):  # DataFrame or numpy array
            summary = f"Shape: {result.shape}"
        else:
            summary = str(result)[:50]

        table.add_row(step_name, result_type, summary)

    console.print(table)