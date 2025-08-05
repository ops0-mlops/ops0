"""
CLI utilities and helpers
"""
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Union

from rich.console import Console
from rich.theme import Theme
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
import typer

# Create console with ops0 theme
console = Console(
    theme=Theme({
        "info": "dim cyan",
        "warning": "magenta",
        "error": "bold red",
        "success": "bold green",
        "highlight": "bold yellow",
        "ops0.primary": "bold blue",
        "ops0.secondary": "cyan",
        "ops0.accent": "yellow",
    })
)

# Default ops0 directory
OPS0_DIR = Path.home() / ".ops0"
OPS0_CONFIG_FILE = OPS0_DIR / "config.json"


def ensure_ops0_dir():
    """Ensure ops0 directory exists"""
    OPS0_DIR.mkdir(exist_ok=True)
    (OPS0_DIR / "pipelines").mkdir(exist_ok=True)
    (OPS0_DIR / "logs").mkdir(exist_ok=True)


def load_config() -> Dict[str, Any]:
    """Load ops0 configuration"""
    ensure_ops0_dir()
    if OPS0_CONFIG_FILE.exists():
        with open(OPS0_CONFIG_FILE) as f:
            return json.load(f)
    return {}


def save_config(config: Dict[str, Any]):
    """Save ops0 configuration"""
    ensure_ops0_dir()
    with open(OPS0_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def print_banner():
    """Print ops0 banner"""
    banner = """
    ╔═══════════════════════════════════════╗
    ║            [bold blue]ops0[/bold blue] v0.1.0-dev            ║
    ║   [dim]Python-Native MLOps Framework[/dim]     ║
    ╚═══════════════════════════════════════╝
    """
    console.print(banner)


def print_success(message: str):
    """Print success message"""
    console.print(f"[success]✓[/success] {message}")


def print_error(message: str):
    """Print error message"""
    console.print(f"[error]✗[/error] {message}")


def print_warning(message: str):
    """Print warning message"""
    console.print(f"[warning]![/warning] {message}")


def print_info(message: str):
    """Print info message"""
    console.print(f"[info]ℹ[/info] {message}")


def confirm_action(message: str, default: bool = False) -> bool:
    """Confirm an action with the user"""
    return Confirm.ask(message, default=default)


def create_progress_bar():
    """Create a progress bar for long operations"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    )


def format_memory(bytes_value: Union[int, str]) -> str:
    """Format memory size in human readable format"""
    if isinstance(bytes_value, str):
        # Parse string like "512Mi" or "2Gi"
        if bytes_value.endswith("Ki"):
            return bytes_value[:-2] + " KB"
        elif bytes_value.endswith("Mi"):
            return bytes_value[:-2] + " MB"
        elif bytes_value.endswith("Gi"):
            return bytes_value[:-2] + " GB"
        return bytes_value

    # Format integer bytes
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in human readable format"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def format_timestamp(timestamp: Optional[float] = None) -> str:
    """Format timestamp in human-readable way"""
    if timestamp is None:
        timestamp = time.time()

    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def create_pipeline_table(pipelines: List[Dict[str, Any]]) -> Table:
    """Create a formatted table of pipelines"""
    table = Table(title="ops0 Pipelines", show_header=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Steps", justify="right")
    table.add_column("Last Run", style="dim")
    table.add_column("Duration", justify="right")

    for pipeline in pipelines:
        status_color = {
            "running": "yellow",
            "success": "green",
            "failed": "red",
            "pending": "dim"
        }.get(pipeline.get("status", "pending"), "white")

        status = f"[{status_color}]{pipeline.get('status', 'pending')}[/{status_color}]"

        table.add_row(
            pipeline["name"],
            status,
            str(pipeline.get("steps", 0)),
            pipeline.get("last_run", "Never"),
            format_duration(pipeline.get("duration", 0))
        )

    return table


def create_step_tree(pipeline_name: str, steps: Dict[str, Any]) -> Tree:
    """Create a tree visualization of pipeline steps"""
    tree = Tree(f"[bold blue]{pipeline_name}[/bold blue]")

    # Group steps by dependencies
    root_steps = []
    dependent_steps = {}

    for name, info in steps.items():
        deps = info.get("dependencies", [])
        if not deps:
            root_steps.append(name)
        else:
            for dep in deps:
                if dep not in dependent_steps:
                    dependent_steps[dep] = []
                dependent_steps[dep].append(name)

    # Build tree recursively
    def add_steps(parent, step_name):
        step_info = steps.get(step_name, {})
        status = step_info.get("status", "pending")

        status_icon = {
            "success": "✓",
            "failed": "✗",
            "running": "⟳",
            "pending": "○"
        }.get(status, "?")

        status_color = {
            "success": "green",
            "failed": "red",
            "running": "yellow",
            "pending": "dim"
        }.get(status, "white")

        node_text = f"[{status_color}]{status_icon}[/{status_color}] {step_name}"

        if step_info.get("duration"):
            node_text += f" [dim]({format_duration(step_info['duration'])})[/dim]"

        branch = parent.add(node_text)

        # Add dependent steps
        for dep in dependent_steps.get(step_name, []):
            add_steps(branch, dep)

    # Add root steps
    for step in root_steps:
        add_steps(tree, step)

    return tree


def format_code(code: str, language: str = "python") -> Syntax:
    """Format code with syntax highlighting"""
    return Syntax(code, language, theme="monokai", line_numbers=True)


def create_deployment_panel(deployment_info: Dict[str, Any]) -> Panel:
    """Create a panel showing deployment information"""
    content = []

    content.append(f"[bold]Pipeline:[/bold] {deployment_info.get('name', 'Unknown')}")
    content.append(f"[bold]Target:[/bold] {deployment_info.get('target', 'Unknown')}")
    content.append(f"[bold]Status:[/bold] {deployment_info.get('status', 'Unknown')}")

    if deployment_info.get("endpoint"):
        content.append(f"[bold]Endpoint:[/bold] {deployment_info['endpoint']}")

    if deployment_info.get("resources"):
        content.append("\n[bold]Resources:[/bold]")
        for resource, value in deployment_info["resources"].items():
            content.append(f"  • {resource}: {value}")

    return Panel(
        "\n".join(content),
        title="[bold blue]Deployment Info[/bold blue]",
        border_style="blue"
    )


def get_project_root() -> Optional[Path]:
    """Find the project root directory (contains ops0.yaml or .ops0/)"""
    current = Path.cwd()

    while current != current.parent:
        if (current / "ops0.yaml").exists() or (current / ".ops0").exists():
            return current
        current = current.parent

    return None


def is_ops0_project() -> bool:
    """Check if current directory is an ops0 project"""
    return get_project_root() is not None


def load_project_config() -> Optional[Dict[str, Any]]:
    """Load project configuration from ops0.yaml"""
    root = get_project_root()
    if not root:
        return None

    config_file = root / "ops0.yaml"
    if not config_file.exists():
        return None

    try:
        import yaml
        with open(config_file) as f:
            return yaml.safe_load(f)
    except ImportError:
        print_warning("PyYAML not installed. Using JSON config instead.")
        return None
    except Exception as e:
        print_error(f"Failed to load config: {e}")
        return None


def get_current_pipeline_info() -> Dict[str, Any]:
    """Get information about the current pipeline"""
    try:
        from ops0.core.graph import PipelineGraph
    except ImportError:
        # Try development import
        from ..core.graph import PipelineGraph

    current = PipelineGraph.get_current()
    if not current:
        return {"status": "No active pipeline"}

    info = {
        "Name": current.name,
        "Steps": len(current.steps),
        "Status": "Active",
        "Created": format_timestamp(),
    }

    # Add step details
    if current.steps:
        step_names = list(current.steps.keys())
        info["Step Names"] = ", ".join(step_names)

        # Build execution order if possible
        try:
            execution_order = current.build_execution_order()
            info["Execution Levels"] = len(execution_order)
            info["Max Parallelism"] = max(len(level) for level in execution_order) if execution_order else 0
        except Exception:
            info["Execution Order"] = "Not available"

    return info


def show_pipeline_tree(pipeline_name: str, steps: Dict[str, Any]) -> None:
    """Display pipeline steps as a tree (helper function)"""
    tree = create_step_tree(pipeline_name, steps)
    console.print(tree)


class ProgressTracker:
    """Simple progress tracker for long operations"""

    def __init__(self, description: str = "Processing..."):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        )
        self.description = description
        self.task = None

    def __enter__(self):
        self.progress.start()
        self.task = self.progress.add_task(self.description, total=None)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.task:
            self.progress.remove_task(self.task)
        self.progress.stop()

    def update(self, description: str):
        """Update progress description"""
        if self.task:
            self.progress.update(self.task, description=description)


# Export all utilities
__all__ = [
    "console",
    "print_banner",
    "print_success",
    "print_error",
    "print_warning",
    "print_info",
    "confirm_action",
    "create_progress_bar",
    "format_memory",
    "format_duration",
    "format_timestamp",
    "create_pipeline_table",
    "create_step_tree",
    "format_code",
    "create_deployment_panel",
    "get_project_root",
    "is_ops0_project",
    "load_project_config",
    "load_config",
    "save_config",
    "ensure_ops0_dir",
    "get_current_pipeline_info",
    "show_pipeline_tree",
    "ProgressTracker",
    "OPS0_DIR",
    "OPS0_CONFIG_FILE"
]