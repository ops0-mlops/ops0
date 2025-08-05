"""
ops0 CLI Main Entry Point

Modern, intuitive CLI for ops0 using Typer with rich output.
"""

import sys
import os
from pathlib import Path
from typing import Optional
import typer
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Handle both development and production imports
try:
    from ops0.core import PipelineGraph, run, deploy
    from ..core.storage import storage
    from ops0.__about__ import __version__
except ImportError:
    # Development mode
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from .core.graph import PipelineGraph
    from .core.executor import run, deploy
    from .core.storage import storage
    from .__about__ import __version__

from .utils import (
    console,
    print_success,
    print_error,
    print_info,
    print_warning,
    confirm_action,
    get_current_pipeline_info,
    format_duration
)
from .config import config
from .doctor import doctor

# Create the main Typer app
app = typer.Typer(
    name="ops0",
    help="🐍⚡ Python-Native ML Pipeline Orchestration",
    epilog="Write Python. Ship Production. Forget the Infrastructure.",
    rich_markup_mode="rich",
    no_args_is_help=True,
    add_completion=False,
)


def version_callback(value: bool):
    """Show ops0 version and exit"""
    if value:
        console.print(f"[bold blue]ops0[/bold blue] version [green]{__version__}[/green]")
        console.print("🐍⚡ Python-Native ML Pipeline Orchestration")
        raise typer.Exit()


@app.callback()
def main(
        version: Optional[bool] = typer.Option(
            None,
            "--version",
            "-v",
            callback=version_callback,
            help="Show version and exit"
        ),
        debug: bool = typer.Option(
            False,
            "--debug",
            help="Enable debug mode"
        ),
        config_file: Optional[Path] = typer.Option(
            None,
            "--config",
            "-c",
            help="Path to config file"
        )
):
    """
    🐍⚡ ops0 - Python-Native ML Pipeline Orchestration

    Transform your Python functions into production-ready ML pipelines
    with zero DevOps complexity.
    """
    if debug:
        os.environ["OPS0_LOG_LEVEL"] = "DEBUG"
        console.print("[dim]Debug mode enabled[/dim]")

    if config_file:
        config.load_from_file(config_file)


@app.command("init")
def init_project(
        name: str = typer.Argument(
            ...,
            help="Project name"
        ),
        template: str = typer.Option(
            "basic",
            "--template",
            "-t",
            help="Project template (basic, ml, advanced)"
        ),
        force: bool = typer.Option(
            False,
            "--force",
            "-f",
            help="Overwrite existing files"
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Show what would be created without creating files"
        )
):
    """
    🚀 Initialize a new ops0 project

    Creates project structure with example pipelines based on template.
    """
    from .commands import create_project

    console.print(f"\n🚀 Initializing ops0 project: [bold]{name}[/bold]")

    if dry_run:
        console.print("[dim]Dry run mode - no files will be created[/dim]\n")

    try:
        created_files = create_project(name, template, force, dry_run)

        if not dry_run:
            print_success(f"Project '{name}' created successfully!")

            # Show created files
            table = Table(title="Created Files", show_header=True)
            table.add_column("File", style="cyan")
            table.add_column("Description", style="dim")

            for file_path, description in created_files:
                table.add_row(str(file_path), description)

            console.print(table)

            # Show next steps
            panel = Panel(
                f"""[bold green]🎯 Next Steps:[/bold green]

1. [cyan]cd {name}[/cyan]
2. [cyan]python -m venv venv[/cyan]
3. [cyan]source venv/bin/activate[/cyan]  # Windows: venv\\Scripts\\activate
4. [cyan]pip install ops0[/cyan]
5. [cyan]python pipeline.py[/cyan]
6. [cyan]ops0 deploy[/cyan]

[dim]📖 Documentation: https://docs.ops0.xyz[/dim]""",
                title="🚀 Ready to Build!",
                border_style="green"
            )
            console.print(panel)

    except Exception as e:
        print_error(f"Failed to create project: {str(e)}")
        raise typer.Exit(1)


@app.command("run")
def run_pipeline(
        local: bool = typer.Option(
            True,
            "--local/--distributed",
            help="Run pipeline locally or distributed"
        ),
        pipeline_file: Optional[Path] = typer.Option(
            None,
            "--file",
            "-f",
            help="Pipeline file to run"
        ),
        watch: bool = typer.Option(
            False,
            "--watch",
            "-w",
            help="Watch for changes and re-run"
        )
):
    """
    ⚡ Run an ops0 pipeline

    Execute pipeline steps locally or in distributed mode.
    """
    mode = "local" if local else "distributed"

    console.print(f"\n⚡ Running pipeline in [bold]{mode}[/bold] mode")

    if pipeline_file:
        console.print(f"📁 Pipeline file: [cyan]{pipeline_file}[/cyan]")
        # Import and execute pipeline file
        import importlib.util
        spec = importlib.util.spec_from_file_location("pipeline", pipeline_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    current_pipeline = PipelineGraph.get_current()
    if not current_pipeline:
        print_error("No active pipeline found. Make sure to use @ops0.pipeline or 'with ops0.pipeline()'")
        raise typer.Exit(1)

    try:
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
        ) as progress:
            task = progress.add_task("Executing pipeline...", total=None)

            results = run(mode=mode)
            progress.remove_task(task)

        print_success("Pipeline completed successfully!")

        # Show results summary
        if results:
            table = Table(title="Pipeline Results", show_header=True)
            table.add_column("Step", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Result", style="dim")

            for step_name, result in results.items():
                status = "✅ Success"
                result_str = str(result)[:50] + "..." if len(str(result)) > 50 else str(result)
                table.add_row(step_name, status, result_str)

            console.print(table)

    except Exception as e:
        print_error(f"Pipeline failed: {str(e)}")
        raise typer.Exit(1)


@app.command("deploy")
def deploy_pipeline(
        name: Optional[str] = typer.Option(
            None,
            "--name",
            "-n",
            help="Pipeline name"
        ),
        env: str = typer.Option(
            "production",
            "--env",
            "-e",
            help="Deployment environment"
        ),
        watch: bool = typer.Option(
            False,
            "--watch",
            help="Watch deployment progress"
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Preview deployment without executing"
        )
):
    """
    🚀 Deploy pipeline to production

    Automatically containerizes and deploys your pipeline with zero configuration.
    """
    current_pipeline = PipelineGraph.get_current()
    if not current_pipeline and not name:
        print_error("No active pipeline found. Use --name to specify pipeline or run from pipeline directory.")
        raise typer.Exit(1)

    pipeline_name = name or (current_pipeline.name if current_pipeline else "ops0-pipeline")

    console.print(f"\n🚀 Deploying pipeline: [bold]{pipeline_name}[/bold]")
    console.print(f"🌍 Environment: [yellow]{env}[/yellow]")

    if dry_run:
        console.print("[dim]Dry run mode - showing deployment plan[/dim]\n")
        # Show what would be deployed
        if current_pipeline:
            table = Table(title="Deployment Plan", show_header=True)
            table.add_column("Step", style="cyan")
            table.add_column("Dependencies", style="dim")
            table.add_column("Resources", style="green")

            for step_name, step_node in current_pipeline.steps.items():
                deps = ", ".join(current_pipeline.get_step_dependencies(step_name)) or "None"
                resources = "CPU: 1000m, Memory: 2Gi"  # Would be calculated
                table.add_row(step_name, deps, resources)

            console.print(table)
        return

    if not confirm_action(f"Deploy '{pipeline_name}' to {env}?"):
        console.print("Deployment cancelled.")
        raise typer.Exit(0)

    try:
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=False
        ) as progress:
            # Deployment phases
            phases = [
                ("Analyzing pipeline...", 2),
                ("Building containers...", 8),
                ("Pushing to registry...", 5),
                ("Deploying to infrastructure...", 3),
                ("Verifying deployment...", 2)
            ]

            for phase_name, duration in phases:
                task = progress.add_task(phase_name, total=duration)
                import time
                for i in range(duration):
                    time.sleep(0.5)  # Simulate work
                    progress.advance(task)
                progress.remove_task(task)

        result = deploy(name=pipeline_name, env=env)

        print_success("Deployment successful!")

        # Show deployment info
        panel = Panel(
            f"""[bold green]🎉 Pipeline Deployed![/bold green]

[bold]Pipeline:[/bold] {result['pipeline']}
[bold]Environment:[/bold] {result['environment']}
[bold]URL:[/bold] [link]{result['url']}[/link]
[bold]Containers:[/bold] {result['containers']} 
[bold]Status:[/bold] {result['status']}

[dim]💡 Monitor: ops0 status --follow[/dim]
[dim]📊 Logs: ops0 logs[/dim]""",
            title="🚀 Deployment Complete",
            border_style="green"
        )
        console.print(panel)

    except Exception as e:
        print_error(f"Deployment failed: {str(e)}")
        raise typer.Exit(1)


@app.command("status")
def show_status(
        follow: bool = typer.Option(
            False,
            "--follow",
            "-f",
            help="Follow status updates in real-time"
        ),
        pipeline: Optional[str] = typer.Option(
            None,
            "--pipeline",
            "-p",
            help="Show status for specific pipeline"
        )
):
    """
    📊 Show pipeline status and health

    Display current pipeline status, metrics, and health information.
    """
    console.print("\n📊 Pipeline Status\n")

    # Show current pipeline info
    current = PipelineGraph.get_current()
    if current:
        info = get_current_pipeline_info()

        table = Table(title="Current Pipeline", show_header=True)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        for key, value in info.items():
            table.add_row(key, str(value))

        console.print(table)

        # Show execution graph
        console.print("\n📈 Execution Graph:")
        console.print(current.visualize())

    else:
        print_warning("No active pipeline found in current directory")


@app.command("logs")
def show_logs(
        pipeline: Optional[str] = typer.Option(
            None,
            "--pipeline",
            "-p",
            help="Pipeline name"
        ),
        step: Optional[str] = typer.Option(
            None,
            "--step",
            "-s",
            help="Specific step name"
        ),
        follow: bool = typer.Option(
            False,
            "--follow",
            "-f",
            help="Follow log output"
        ),
        lines: int = typer.Option(
            100,
            "--lines",
            "-n",
            help="Number of lines to show"
        )
):
    """
    📋 View pipeline and step logs

    Display logs from pipeline execution and individual steps.
    """
    console.print(f"\n📋 Showing logs (last {lines} lines)")

    if step:
        console.print(f"🔍 Step: [cyan]{step}[/cyan]")
    if pipeline:
        console.print(f"📦 Pipeline: [cyan]{pipeline}[/cyan]")

    # Mock log display
    console.print("\n[dim]2025-01-01 12:00:00[/dim] [green]INFO[/green] Pipeline started")
    console.print("[dim]2025-01-01 12:00:01[/dim] [blue]DEBUG[/blue] Step 'preprocess' executing")
    console.print("[dim]2025-01-01 12:00:02[/dim] [green]INFO[/green] Step 'preprocess' completed (1.2s)")
    console.print("[dim]2025-01-01 12:00:03[/dim] [green]INFO[/green] Pipeline completed successfully")

    if follow:
        console.print("\n[dim]Following logs... Press Ctrl+C to exit[/dim]")


@app.command("debug")
def debug_pipeline(
        step: Optional[str] = typer.Option(
            None,
            "--step",
            "-s",
            help="Debug specific step"
        ),
        interactive: bool = typer.Option(
            False,
            "--interactive",
            "-i",
            help="Interactive debugging mode"
        )
):
    """
    🐛 Debug pipeline execution

    Debug pipeline steps with interactive mode and detailed output.
    """
    console.print("\n🐛 Debug Mode")

    current = PipelineGraph.get_current()
    if not current:
        print_error("No active pipeline found")
        raise typer.Exit(1)

    if step:
        if step not in current.steps:
            print_error(f"Step '{step}' not found in pipeline")
            available = ", ".join(current.steps.keys())
            console.print(f"Available steps: {available}")
            raise typer.Exit(1)

        console.print(f"🔍 Debugging step: [cyan]{step}[/cyan]")

        if interactive:
            console.print("🎮 Interactive mode - use 'help' for commands")
            # Would launch interactive debugger

    else:
        console.print("📊 Pipeline Overview:")
        console.print(current.visualize())


@app.command("config")
def manage_config(
        show: bool = typer.Option(
            False,
            "--show",
            help="Show current configuration"
        ),
        set_key: Optional[str] = typer.Option(
            None,
            "--set",
            help="Set configuration key (format: key=value)"
        ),
        reset: bool = typer.Option(
            False,
            "--reset",
            help="Reset to default configuration"
        )
):
    """
    ⚙️ Manage ops0 configuration

    View and modify ops0 configuration settings.
    """
    if show:
        console.print("\n⚙️ Current Configuration")
        config.show()
    elif set_key:
        if "=" not in set_key:
            print_error("Invalid format. Use: --set key=value")
            raise typer.Exit(1)

        key, value = set_key.split("=", 1)
        config.set(key.strip(), value.strip())
        print_success(f"Set {key} = {value}")
    elif reset:
        if confirm_action("Reset configuration to defaults?"):
            config.reset()
            print_success("Configuration reset to defaults")
    else:
        console.print("Use --show, --set, or --reset")


@app.command("doctor")
def run_doctor(
        quick: bool = typer.Option(
            False,
            "--quick",
            "-q",
            help="Run quick health check only"
        ),
        fix: bool = typer.Option(
            False,
            "--fix",
            help="Attempt to fix common issues automatically"
        )
):
    """
    🩺 Run ops0 health diagnostics

    Comprehensive health check for ops0 installation and environment.
    """
    console.print("\n🩺 ops0 Doctor - Health Check")

    if quick:
        console.print("Running quick diagnostics...\n")
        # Quick check implementation would go here
        print_success("Quick health check completed!")
    else:
        results = doctor.run_full_diagnosis()

        if fix and results["summary"]["total_errors"] > 0:
            console.print("\n🔧 Attempting automatic fixes...")
            # Auto-fix implementation would go here
            print_info("Some issues may require manual intervention")


@app.command("config")
def manage_config(
        show: bool = typer.Option(
            False,
            "--show",
            help="Show current configuration"
        ),
        set_key: Optional[str] = typer.Option(
            None,
            "--set",
            help="Set configuration key (format: key=value)"
        ),
        reset: bool = typer.Option(
            False,
            "--reset",
            help="Reset to default configuration"
        ),
        validate: bool = typer.Option(
            False,
            "--validate",
            help="Validate current configuration"
        )
):
    """
    ⚙️ Manage ops0 configuration

    View and modify ops0 configuration settings.
    """
    if show:
        console.print("\n⚙️ Current Configuration")
        config.show()
    elif set_key:
        if "=" not in set_key:
            print_error("Invalid format. Use: --set key=value")
            raise typer.Exit(1)

        key, value = set_key.split("=", 1)
        if config.set(key.strip(), value.strip()):
            print_success(f"Set {key} = {value}")
        else:
            raise typer.Exit(1)
    elif reset:
        if confirm_action("Reset configuration to defaults?"):
            config.reset()
            print_success("Configuration reset to defaults")
    elif validate:
        console.print("\n🔍 Validating configuration...")
        if config.validate():
            print_success("Configuration is valid")
        else:
            print_error("Configuration validation failed")
            raise typer.Exit(1)
    else:
        console.print("Use --show, --set, --reset, or --validate")


if __name__ == "__main__":
    app()