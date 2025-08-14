# ops0/cli/utils.py
"""Utility functions for ops0 CLI."""

import os
import sys
import importlib.util
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def check_dependencies() -> Tuple[bool, List[str]]:
    """
    Check if all required dependencies are installed.

    Returns:
        Tuple of (all_installed, missing_packages)
    """
    required_packages = [
        'docker',
        'kubernetes',
        'pydantic',
        'typer',
        'rich',
        'httpx',
        'prometheus-client',
        'psutil',
        'aiofiles',
        'watchdog'
    ]

    missing = []
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing.append(package)

    return len(missing) == 0, missing


def check_docker() -> Tuple[bool, str]:
    """
    Check if Docker is installed and running.

    Returns:
        Tuple of (is_running, message)
    """
    try:
        result = subprocess.run(
            ['docker', 'version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True, "Docker is installed and running"
        else:
            return False, "Docker is installed but not running"
    except FileNotFoundError:
        return False, "Docker is not installed"
    except subprocess.TimeoutExpired:
        return False, "Docker command timed out"
    except Exception as e:
        return False, f"Error checking Docker: {str(e)}"


def check_kubernetes() -> Tuple[bool, str]:
    """
    Check if kubectl is installed and configured.

    Returns:
        Tuple of (is_configured, message)
    """
    try:
        result = subprocess.run(
            ['kubectl', 'version', '--client'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Try to check if connected to a cluster
            cluster_result = subprocess.run(
                ['kubectl', 'cluster-info'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if cluster_result.returncode == 0:
                return True, "kubectl is installed and connected to a cluster"
            else:
                return True, "kubectl is installed but not connected to a cluster"
        else:
            return False, "kubectl is installed but not configured"
    except FileNotFoundError:
        return False, "kubectl is not installed"
    except subprocess.TimeoutExpired:
        return False, "kubectl command timed out"
    except Exception as e:
        return False, f"Error checking kubectl: {str(e)}"


def format_status_table(statuses: Dict[str, Tuple[bool, str]]) -> Table:
    """
    Format status checks as a Rich table.

    Args:
        statuses: Dictionary of component -> (is_ok, message)

    Returns:
        Formatted Rich Table
    """
    table = Table(
        title="ops0 System Status",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )

    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Details", style="dim")

    for component, (is_ok, message) in statuses.items():
        status_emoji = "✅" if is_ok else "❌"
        status_color = "green" if is_ok else "red"
        table.add_row(
            component,
            f"[{status_color}]{status_emoji}[/{status_color}]",
            message
        )

    return table


def print_error(message: str, exit_code: int = 1) -> None:
    """Print error message and exit."""
    console.print(f"[red]Error:[/red] {message}")
    if exit_code > 0:
        sys.exit(exit_code)


def print_success(message: str) -> None:
    """Print success message."""
    console.print(f"[green]✓[/green] {message}")


def print_warning(message: str) -> None:
    """Print warning message."""
    console.print(f"[yellow]Warning:[/yellow] {message}")


def print_info(message: str) -> None:
    """Print info message."""
    console.print(f"[blue]Info:[/blue] {message}")


def confirm(message: str, default: bool = False) -> bool:
    """Ask for user confirmation."""
    suffix = " [Y/n]" if default else " [y/N]"
    response = console.input(f"{message}{suffix}: ").lower().strip()

    if not response:
        return default

    return response in ['y', 'yes']


def get_project_root() -> Optional[Path]:
    """Find the project root directory (contains .ops0 folder)."""
    current = Path.cwd()

    while current != current.parent:
        if (current / '.ops0').exists():
            return current
        current = current.parent

    # Check current directory as last resort
    if (Path.cwd() / '.ops0').exists():
        return Path.cwd()

    return None


def ensure_project_initialized() -> Path:
    """Ensure we're in an ops0 project directory."""
    project_root = get_project_root()

    if not project_root:
        print_error(
            "Not in an ops0 project directory. Run 'ops0 init' first.",
            exit_code=1
        )

    return project_root


def format_pipeline_info(pipeline_data: Dict[str, Any]) -> Panel:
    """Format pipeline information as a Rich panel."""
    content = []

    if 'name' in pipeline_data:
        content.append(f"[bold]Name:[/bold] {pipeline_data['name']}")

    if 'version' in pipeline_data:
        content.append(f"[bold]Version:[/bold] {pipeline_data['version']}")

    if 'steps' in pipeline_data:
        content.append(f"[bold]Steps:[/bold] {len(pipeline_data['steps'])}")
        for step in pipeline_data['steps']:
            content.append(f"  • {step.get('name', 'unknown')}")

    if 'status' in pipeline_data:
        status = pipeline_data['status']
        color = "green" if status == "running" else "yellow"
        content.append(f"[bold]Status:[/bold] [{color}]{status}[/{color}]")

    return Panel(
        "\n".join(content),
        title="Pipeline Information",
        border_style="blue"
    )


def validate_step_name(name: str) -> bool:
    """Validate that a step name is valid."""
    import re
    # Step names should be valid Python identifiers
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name))


def parse_env_file(env_file: Path) -> Dict[str, str]:
    """Parse environment variables from a .env file."""
    env_vars = {}

    if not env_file.exists():
        return env_vars

    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip().strip('"\'')

    return env_vars