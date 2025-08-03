"""
ops0 CLI Utilities

Common utilities and helpers for the ops0 command line interface.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Confirm, Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.tree import Tree
from rich.syntax import Syntax

# Create global console instance with ops0 styling
console = Console(
    theme={
        "info": "dim cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "accent": "bold blue",
        "code": "bold magenta",
    }
)

# Handle imports for both development and production
try:
    from ops0.core.graph import PipelineGraph
    from ops0.core.storage import storage
except ImportError:
    # Development mode
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    try:
        from src.ops0.core.graph import PipelineGraph
        from src.ops0.core.storage import storage
    except ImportError:
        # Graceful fallback if core modules not available
        PipelineGraph = None
        storage = None


def print_success(message: str):
    """Print success message with styling"""
    console.print(f"✅ {message}", style="success")


def print_error(message: str):
    """Print error message with styling"""
    console.print(f"❌ {message}", style="error")


def print_warning(message: str):
    """Print warning message with styling"""
    console.print(f"⚠️  {message}", style="warning")


def print_info(message: str):
    """Print info message with styling"""
    console.print(f"ℹ️  {message}", style="info")


def print_header(title: str, subtitle: Optional[str] = None):
    """Print a formatted header"""
    text = Text(title, style="bold accent")
    if subtitle:
        text.append(f"\n{subtitle}", style="dim")

    panel = Panel(text, border_style="accent")
    console.print(panel)


def confirm_action(message: str, default: bool = False) -> bool:
    """Ask for user confirmation"""
    return Confirm.ask(message, default=default)


def prompt_input(message: str, default: Optional[str] = None) -> str:
    """Prompt for user input"""
    return Prompt.ask(message, default=default)


def format_duration(seconds: float) -> str:
    """Format duration in a human-readable way"""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_memory(bytes_value: Union[int, str]) -> str:
    """Format memory size in human-readable way"""
    if isinstance(bytes_value, str):
        # Handle Kubernetes-style memory specs (e.g., "2Gi", "512Mi")
        if bytes_value.endswith(('Ki', 'Mi', 'Gi', 'Ti')):
            return bytes_value
        try:
            bytes_value = int(bytes_value)
        except ValueError:
            return bytes_value

    if bytes_value < 1024:
        return f"{bytes_value}B"
    elif bytes_value < 1024 ** 2:
        return f"{bytes_value / 1024:.1f}KB"
    elif bytes_value < 1024 ** 3:
        return f"{bytes_value / (1024 ** 2):.1f}MB"
    elif bytes_value < 1024 ** 4:
        return f"{bytes_value / (1024 ** 3):.1f}GB"
    else:
        return f"{bytes_value / (1024 ** 4):.1f}TB"


def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """Format timestamp in human-readable way"""
    if timestamp is None:
        timestamp = datetime.now()

    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def get_current_pipeline_info() -> Dict[str, Any]:
    """Get information about the current pipeline"""
    if not PipelineGraph:
        return {"error": "Pipeline module not available"}

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

        # Build execution order
        try:
            execution_order = current.build_execution_order()
            info["Execution Levels"] = len(execution_order)
            info["Max Parallelism"] = max(len(level) for level in execution_order)
        except Exception:
            info["Execution Order"] = "Error building order"

    return info


def show_pipeline_tree(pipeline_graph) -> None:
    """Display pipeline as a tree structure"""
    if not pipeline_graph or not pipeline_graph.steps:
        console.print("No pipeline steps found")
        return

    tree = Tree(f"🚀 Pipeline: [bold]{pipeline_graph.name}[/bold]")

    try:
        execution_order = pipeline_graph.build_execution_order()

        for level_idx, level_steps in enumerate(execution_order):
            level_branch = tree.add(f"📋 Level {level_idx + 1}")

            for step_name in level_steps:
                step_node = pipeline_graph.steps[step_name]
                deps = pipeline_graph.get_step_dependencies(step_name)

                if deps:
                    dep_str = f" (depends on: {', '.join(deps)})"
                else:
                    dep_str = ""

                step_status = "✅" if getattr(step_node, 'executed', False) else "⏸️"
                level_branch.add(f"{step_status} {step_name}{dep_str}")

    except Exception as e:
        tree.add(f"❌ Error building execution order: {e}")

        # Fallback: show all steps
        for step_name in pipeline_graph.steps.keys():
            tree.add(f"📦 {step_name}")

    console.print(tree)


def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are available"""
    dependencies = {
        "docker": False,
        "git": False,
        "python": True,  # Always true if we're running Python
    }

    # Check Docker
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            timeout=5
        )
        dependencies["docker"] = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Check Git
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            timeout=5
        )
        dependencies["git"] = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return dependencies


def validate_project_structure() -> Dict[str, bool]:
    """Validate ops0 project structure"""
    checks = {
        "ops0_directory": Path(".ops0").exists(),
        "storage_directory": Path(".ops0/storage").exists(),
        "python_files": len(list(Path.cwd().glob("*.py"))) > 0,
        "pipeline_files": len(list(Path.cwd().glob("*pipeline*.py"))) > 0,
    }

    return checks


def get_system_info() -> Dict[str, str]:
    """Get system information for debugging"""
    import platform

    info = {
        "Platform": platform.platform(),
        "Python Version": platform.python_version(),
        "Architecture": platform.architecture()[0],
        "Machine": platform.machine(),
        "Processor": platform.processor() or "Unknown",
    }

    # Add ops0-specific info
    try:
        if "OPS0_ENV" in os.environ:
            info["ops0 Environment"] = os.environ["OPS0_ENV"]

        # Try to get ops0 version
        try:
            from ops0.__about__ import __version__
            info["ops0 Version"] = __version__
        except ImportError:
            info["ops0 Version"] = "Development"

    except Exception:
        pass

    return info


def create_progress_context(description: str = "Processing..."):
    """Create a progress context for long-running operations"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True
    )


def display_error_details(error: Exception, show_traceback: bool = False):
    """Display detailed error information"""
    error_panel = Panel(
        f"[bold red]Error:[/bold red] {str(error)}\n\n"
        f"[dim]Type:[/dim] {type(error).__name__}\n"
        f"[dim]Time:[/dim] {format_timestamp()}",
        title="❌ Error Details",
        border_style="red"
    )
    console.print(error_panel)

    if show_traceback:
        import traceback
        console.print("\n[dim]Traceback:[/dim]")
        console.print(traceback.format_exc())


def show_code_snippet(code: str, language: str = "python", title: Optional[str] = None):
    """Display a syntax-highlighted code snippet"""
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)

    if title:
        panel = Panel(syntax, title=title, border_style="code")
        console.print(panel)
    else:
        console.print(syntax)


def create_comparison_table(
        items: List[Dict[str, Any]],
        title: str = "Comparison",
        key_column: str = "Item"
) -> Table:
    """Create a comparison table for displaying differences"""
    if not items:
        return Table(title=title)

    table = Table(title=title, show_header=True)

    # Get all possible columns
    all_keys = set()
    for item in items:
        all_keys.update(item.keys())

    # Add columns
    table.add_column(key_column, style="cyan")
    for key in sorted(all_keys):
        if key != key_column:
            table.add_column(key.replace("_", " ").title(), style="white")

    # Add rows
    for item in items:
        row_data = [str(item.get(key_column, ""))]
        for key in sorted(all_keys):
            if key != key_column:
                value = item.get(key, "")
                row_data.append(str(value))
        table.add_row(*row_data)

    return table


def watch_file_changes(file_path: Path, callback):
    """Watch file for changes and execute callback"""
    try:
        import watchdog
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class ChangeHandler(FileSystemEventHandler):
            def on_modified(self, event):
                if event.src_path == str(file_path):
                    callback()

        observer = Observer()
        observer.schedule(ChangeHandler(), str(file_path.parent), recursive=False)
        observer.start()

        console.print(f"👀 Watching {file_path} for changes... Press Ctrl+C to stop")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

    except ImportError:
        print_warning("File watching requires 'watchdog' package: pip install watchdog")


def safe_import(module_name: str, fallback=None):
    """Safely import a module with fallback"""
    try:
        return __import__(module_name)
    except ImportError:
        return fallback


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if needed"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_ansi_codes(text: str) -> str:
    """Remove ANSI color codes from text"""
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def truncate_text(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def highlight_diff(old_text: str, new_text: str) -> str:
    """Highlight differences between two texts"""
    if old_text == new_text:
        return new_text

    # Simple highlighting - in production would use difflib
    return f"[red]-{old_text}[/red] [green]+{new_text}[/green]"


class ProgressTracker:
    """Track progress of multi-step operations"""

    def __init__(self, steps: List[str], title: str = "Progress"):
        self.steps = steps
        self.title = title
        self.current_step = 0
        self.progress = None
        self.task = None

    def __enter__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        )
        self.progress.start()
        self.task = self.progress.add_task(
            f"{self.title} - {self.steps[0]}",
            total=len(self.steps)
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress:
            self.progress.stop()

    def next_step(self, message: Optional[str] = None):
        """Move to the next step"""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            step_name = self.steps[self.current_step]
            description = message or f"{self.title} - {step_name}"
            self.progress.update(self.task, advance=1, description=description)

    def complete(self, message: Optional[str] = None):
        """Mark as complete"""
        if self.progress and self.task:
            final_message = message or f"{self.title} - Complete"
            self.progress.update(
                self.task,
                completed=len(self.steps),
                description=final_message
            )