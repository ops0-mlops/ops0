# ops0/cli/doctor.py
"""Doctor command for ops0 - system diagnostics and health checks."""

import sys
import platform
import os
from pathlib import Path
from typing import Dict, Tuple, List
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from ..core import get_version
from ..core.config import Config
from ..core.storage import LocalStorage
from ..runtime.docker import DockerRuntime
from ..runtime.kubernetes import KubernetesRuntime

from .utils import (
    check_dependencies,
    check_docker,
    check_kubernetes,
    format_status_table,
    print_error,
    print_success,
    print_warning,
    print_info,
    get_project_root
)

console = Console()
doctor_app = typer.Typer(help="Run system diagnostics")


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible."""
    version = sys.version_info
    min_version = (3, 8)

    if version >= min_version:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} (minimum required: 3.8)"


def check_ops0_installation() -> Tuple[bool, str]:
    """Check if ops0 is properly installed."""
    try:
        version = get_version()
        return True, f"ops0 v{version}"
    except Exception as e:
        return False, f"Error getting ops0 version: {str(e)}"


def check_storage() -> Tuple[bool, str]:
    """Check if storage is accessible."""
    try:
        storage = LocalStorage()
        test_key = "_doctor_test"
        test_value = {"test": "data"}

        # Try write and read
        storage.save(test_key, test_value)
        retrieved = storage.load(test_key)
        storage.delete(test_key)

        if retrieved == test_value:
            return True, "Storage is working correctly"
        else:
            return False, "Storage read/write mismatch"
    except Exception as e:
        return False, f"Storage error: {str(e)}"


def check_project_structure() -> Tuple[bool, str]:
    """Check if current directory is an ops0 project."""
    project_root = get_project_root()

    if project_root:
        required_dirs = ['.ops0', '.ops0/storage', '.ops0/logs']
        missing = []

        for dir_name in required_dirs:
            if not (project_root / dir_name).exists():
                missing.append(dir_name)

        if missing:
            return False, f"Missing directories: {', '.join(missing)}"
        else:
            return True, f"Project structure OK at {project_root}"
    else:
        return False, "Not in an ops0 project directory"


def check_environment() -> Tuple[bool, str]:
    """Check environment variables and configuration."""
    try:
        config = Config()

        # Check for any critical environment variables
        env_vars = {
            'OPS0_ENV': os.getenv('OPS0_ENV', 'development'),
            'OPS0_LOG_LEVEL': os.getenv('OPS0_LOG_LEVEL', 'INFO'),
        }

        return True, f"Environment: {env_vars['OPS0_ENV']}"
    except Exception as e:
        return False, f"Configuration error: {str(e)}"


def run_all_checks() -> Dict[str, Tuple[bool, str]]:
    """Run all system checks."""
    checks = {
        "Python Version": check_python_version(),
        "ops0 Installation": check_ops0_installation(),
        "Dependencies": check_dependencies(),
        "Project Structure": check_project_structure(),
        "Storage": check_storage(),
        "Environment": check_environment(),
        "Docker": check_docker(),
        "Kubernetes": check_kubernetes(),
    }

    return checks


@doctor_app.command()
def doctor(
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
        fix: bool = typer.Option(False, "--fix", help="Attempt to fix issues automatically")
):
    """
    Run system diagnostics and check ops0 health.

    This command checks:
    - Python version compatibility
    - ops0 installation
    - Required dependencies
    - Docker and Kubernetes availability
    - Project structure
    - Storage accessibility
    """
    console.print("\n[bold blue]🔍 Running ops0 system diagnostics...[/bold blue]\n")

    # Run all checks
    results = run_all_checks()

    # Display results
    table = format_status_table(results)
    console.print(table)
    console.print()

    # Count issues
    issues = [(name, result) for name, result in results.items() if not result[0]]

    if not issues:
        console.print(Panel(
            "[green]✅ All systems operational![/green]\n\n"
            "Your ops0 installation is healthy and ready to use.",
            title="[bold green]Success[/bold green]",
            border_style="green"
        ))
        return

    # Show issues
    console.print(Panel(
        f"[yellow]⚠️  Found {len(issues)} issue(s)[/yellow]",
        title="[bold yellow]Issues Detected[/bold yellow]",
        border_style="yellow"
    ))

    # Detailed issue information
    if verbose or fix:
        console.print("\n[bold]Issue Details:[/bold]\n")

        for name, (is_ok, message) in issues:
            console.print(f"[red]✗[/red] {name}: {message}")

            if fix:
                fix_result = attempt_fix(name, message)
                if fix_result:
                    print_success(f"  Fixed: {fix_result}")
                else:
                    print_warning(f"  Could not automatically fix this issue")

            console.print()

    # Provide recommendations
    show_recommendations(issues)

    # Exit with error code if issues found
    if issues and not fix:
        sys.exit(1)


def attempt_fix(issue_name: str, message: str) -> str:
    """Attempt to fix common issues automatically."""
    fixes = {
        "Dependencies": fix_dependencies,
        "Project Structure": fix_project_structure,
        "Storage": fix_storage,
    }

    fix_func = fixes.get(issue_name)
    if fix_func:
        return fix_func(message)

    return ""


def fix_dependencies(message: str) -> str:
    """Try to install missing dependencies."""
    _, missing = check_dependencies()
    if missing:
        import subprocess
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install"] + missing,
                check=True
            )
            return f"Installed missing packages: {', '.join(missing)}"
        except subprocess.CalledProcessError:
            return ""
    return ""


def fix_project_structure(message: str) -> str:
    """Create missing project directories."""
    project_root = Path.cwd()
    created = []

    for dir_name in ['.ops0', '.ops0/storage', '.ops0/logs', '.ops0/registry']:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created.append(dir_name)

    if created:
        return f"Created directories: {', '.join(created)}"
    return ""


def fix_storage(message: str) -> str:
    """Try to fix storage issues."""
    try:
        # Ensure storage directory exists
        storage_dir = Path.cwd() / '.ops0' / 'storage'
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Test write permissions
        test_file = storage_dir / '.test'
        test_file.write_text("test")
        test_file.unlink()

        return "Fixed storage permissions"
    except Exception:
        return ""


def show_recommendations(issues: List[Tuple[str, Tuple[bool, str]]]):
    """Show recommendations for fixing issues."""
    recommendations = []

    for name, (_, message) in issues:
        if name == "Docker" and "not installed" in message:
            recommendations.append(
                "• Install Docker: https://docs.docker.com/get-docker/"
            )
        elif name == "Docker" and "not running" in message:
            recommendations.append(
                "• Start Docker Desktop or run: sudo systemctl start docker"
            )
        elif name == "Kubernetes" and "not installed" in message:
            recommendations.append(
                "• Install kubectl: https://kubernetes.io/docs/tasks/tools/"
            )
        elif name == "Dependencies":
            recommendations.append(
                "• Run: pip install -r requirements.txt"
            )
        elif name == "Project Structure":
            recommendations.append(
                "• Run: ops0 init"
            )

    if recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for rec in recommendations:
            console.print(rec)


if __name__ == "__main__":
    doctor_app()