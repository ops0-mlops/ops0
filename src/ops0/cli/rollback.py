"""Rollback command for ops0 - rollback pipeline deployments."""

import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

from ..core.config import Config
from ..deployment import DeploymentManager

from .utils import (
    ensure_project_initialized,
    print_success,
    print_error,
    print_warning,
    print_info,
    confirm,
    format_time_ago
)

console = Console()


def rollback(
        pipeline: Optional[str] = typer.Argument(None, help="Pipeline name"),
        version: Optional[str] = typer.Option(None, "--version", "-v", help="Specific version to rollback to"),
        steps: Optional[int] = typer.Option(1, "--steps", "-n", help="Number of versions to rollback"),
        env: str = typer.Option("production", "--env", "-e", help="Environment"),
        force: bool = typer.Option(False, "--force", "-f", help="Force rollback without confirmation"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be rolled back"),
):
    """
    Rollback an ops0 pipeline to a previous version.

    Examples:
        ops0 rollback                    # Rollback to previous version
        ops0 rollback --steps 2         # Rollback 2 versions
        ops0 rollback --version 0.1.0   # Rollback to specific version
        ops0 rollback --dry-run         # Preview rollback
    """
    # Ensure we're in an ops0 project
    project_root = ensure_project_initialized()

    # Get deployment manager
    manager = DeploymentManager(project_root, env)

    # Get pipeline name if not specified
    if not pipeline:
        pipelines = manager.list_pipelines()
        if not pipelines:
            print_error("No deployed pipelines found")
            raise typer.Exit(1)
        elif len(pipelines) == 1:
            pipeline = pipelines[0]
        else:
            # Ask user to select
            pipeline = select_pipeline(pipelines)

    # Get deployment history
    history = manager.get_deployment_history(pipeline)
    if not history:
        print_error(f"No deployment history found for '{pipeline}'")
        raise typer.Exit(1)

    # Show current deployment
    current = history[0]
    console.print(f"\n[bold]Current deployment:[/bold]")
    show_deployment_info(current)

    # Determine target version
    if version:
        # Find specific version
        target = find_version_in_history(version, history)
        if not target:
            print_error(f"Version '{version}' not found in deployment history")
            show_available_versions(history)
            raise typer.Exit(1)
    else:
        # Rollback by steps
        if steps >= len(history):
            print_error(f"Cannot rollback {steps} versions. Only {len(history) - 1} previous versions available.")
            raise typer.Exit(1)
        target = history[steps]

    # Show target deployment
    console.print(f"\n[bold]Target deployment:[/bold]")
    show_deployment_info(target)

    # Show changes
    show_rollback_changes(current, target)

    # Dry run mode
    if dry_run:
        console.print("\n[yellow]This is a dry run. No changes will be made.[/yellow]")
        return

    # Confirm rollback
    if not force:
        if not confirm(f"\nRollback '{pipeline}' from v{current.version} to v{target.version}?"):
            print_info("Rollback cancelled")
            return

    # Perform rollback
    try:
        perform_rollback(manager, pipeline, target)
    except Exception as e:
        print_error(f"Rollback failed: {e}")
        raise typer.Exit(1)


def select_pipeline(pipelines: List[str]) -> str:
    """Let user select a pipeline."""
    console.print("\n[bold]Select pipeline to rollback:[/bold]")

    for i, name in enumerate(pipelines, 1):
        console.print(f"  {i}. {name}")

    while True:
        try:
            choice = int(console.input("\nEnter number: "))
            if 1 <= choice <= len(pipelines):
                return pipelines[choice - 1]
        except ValueError:
            pass

        print_error("Invalid choice. Please enter a number.")


def find_version_in_history(version: str, history: List['Deployment']) -> Optional['Deployment']:
    """Find specific version in deployment history."""
    for deployment in history:
        if deployment.version == version:
            return deployment
    return None


def show_deployment_info(deployment: 'Deployment'):
    """Show deployment information."""
    info = [
        f"Version: {deployment.version}",
        f"Deployed: {format_time_ago(deployment.timestamp)}",
        f"Deployed by: {deployment.deployed_by}",
        f"Image: {deployment.image}",
        f"Replicas: {deployment.replicas}"
    ]

    console.print(Panel("\n".join(info), box=box.SIMPLE))


def show_available_versions(history: List['Deployment']):
    """Show available versions for rollback."""
    console.print("\n[bold]Available versions:[/bold]")

    table = Table(box=box.SIMPLE)
    table.add_column("Version", style="cyan")
    table.add_column("Deployed")
    table.add_column("Status")

    for deployment in history:
        status = "current" if deployment == history[0] else "available"
        table.add_row(
            deployment.version,
            format_time_ago(deployment.timestamp),
            status
        )

    console.print(table)


def show_rollback_changes(current: 'Deployment', target: 'Deployment'):
    """Show what will change in the rollback."""
    console.print("\n[bold]Changes:[/bold]")

    changes = []

    if current.version != target.version:
        changes.append(f"• Version: {current.version} → {target.version}")

    if current.image != target.image:
        changes.append(f"• Image: {current.image} → {target.image}")

    if current.replicas != target.replicas:
        changes.append(f"• Replicas: {current.replicas} → {target.replicas}")

    if current.config != target.config:
        changes.append("• Configuration changes detected")

    for change in changes:
        console.print(change)

    # Show warnings
    time_diff = current.timestamp - target.timestamp
    if time_diff > 7 * 24 * 3600:  # More than 7 days old
        print_warning(
            f"\nTarget version is {int(time_diff / 86400)} days old. "
            "Make sure it's compatible with current dependencies."
        )


def perform_rollback(manager: DeploymentManager, pipeline: str, target: 'Deployment'):
    """Perform the rollback."""
    steps = [
        ("Validating target version", lambda: manager.validate_deployment(target)),
        ("Creating rollback checkpoint", lambda: manager.create_checkpoint()),
        ("Switching to target version", lambda: manager.switch_version(pipeline, target)),
        ("Updating configuration", lambda: manager.update_config(target.config)),
        ("Restarting services", lambda: manager.restart_services(pipeline)),
        ("Verifying deployment", lambda: manager.verify_deployment(pipeline)),
    ]

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
    ) as progress:

        main_task = progress.add_task("Rolling back...", total=len(steps))

        for step_name, step_func in steps:
            step_task = progress.add_task(step_name, total=None)

            try:
                step_func()
                progress.update(step_task, completed=True)
                print_success(step_name)

            except Exception as e:
                progress.update(step_task, completed=True)
                print_error(f"{step_name}: {e}")

                # Attempt recovery
                print_warning("Attempting to recover from failed rollback...")
                try:
                    manager.restore_checkpoint()
                    print_success("Recovered from checkpoint")
                except:
                    print_error("Recovery failed. Manual intervention may be required.")

                raise

            finally:
                progress.remove_task(step_task)
                progress.update(main_task, advance=1)

    # Success message
    console.print("\n" + Panel(
        f"[green]✅ Rollback completed successfully![/green]\n\n"
        f"Pipeline '{pipeline}' is now running version {target.version}\n\n"
        f"[bold]Next steps:[/bold]\n"
        f"• Check status: ops0 status {pipeline}\n"
        f"• View logs: ops0 logs {pipeline}\n"
        f"• Monitor metrics: ops0 dashboard",
        title="[bold green]Rollback Complete[/bold green]",
        border_style="green"
    ))


class DeploymentManager:
    """Mock deployment manager for demonstration."""

    def __init__(self, project_root: Path, environment: str):
        self.project_root = project_root
        self.environment = environment

    def list_pipelines(self) -> List[str]:
        """List deployed pipelines."""
        return ["fraud-detector", "customer-churn"]

    def get_deployment_history(self, pipeline: str) -> List['Deployment']:
        """Get deployment history for a pipeline."""
        # Mock data
        return [
            Deployment(
                version="0.2.0",
                timestamp=time.time() - 3600,  # 1 hour ago
                deployed_by="user@example.com",
                image=f"ops0/{pipeline}:0.2.0",
                replicas=3,
                config={"env": "production", "debug": False}
            ),
            Deployment(
                version="0.1.9",
                timestamp=time.time() - 86400,  # 1 day ago
                deployed_by="user@example.com",
                image=f"ops0/{pipeline}:0.1.9",
                replicas=3,
                config={"env": "production", "debug": False}
            ),
            Deployment(
                version="0.1.8",
                timestamp=time.time() - 3 * 86400,  # 3 days ago
                deployed_by="ci@example.com",
                image=f"ops0/{pipeline}:0.1.8",
                replicas=2,
                config={"env": "production", "debug": True}
            ),
            Deployment(
                version="0.1.0",
                timestamp=time.time() - 30 * 86400,  # 30 days ago
                deployed_by="admin@example.com",
                image=f"ops0/{pipeline}:0.1.0",
                replicas=1,
                config={"env": "production", "debug": False}
            ),
        ]

    def validate_deployment(self, deployment: 'Deployment'):
        """Validate deployment configuration."""
        time.sleep(1)  # Simulate validation

    def create_checkpoint(self):
        """Create backup checkpoint."""
        time.sleep(1)  # Simulate checkpoint

    def switch_version(self, pipeline: str, deployment: 'Deployment'):
        """Switch to target version."""
        time.sleep(2)  # Simulate version switch

    def update_config(self, config: Dict[str, Any]):
        """Update configuration."""
        time.sleep(1)  # Simulate config update

    def restart_services(self, pipeline: str):
        """Restart pipeline services."""
        time.sleep(2)  # Simulate restart

    def verify_deployment(self, pipeline: str):
        """Verify deployment is working."""
        time.sleep(1)  # Simulate verification

    def restore_checkpoint(self):
        """Restore from checkpoint."""
        time.sleep(2)  # Simulate restore


class Deployment:
    """Deployment information."""

    def __init__(self, version: str, timestamp: float, deployed_by: str,
                 image: str, replicas: int, config: Dict[str, Any]):
        self.version = version
        self.timestamp = timestamp
        self.deployed_by = deployed_by
        self.image = image
        self.replicas = replicas
        self.config = config