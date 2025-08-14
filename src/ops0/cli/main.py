"""Main CLI entry point for ops0."""

import sys
import os
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Add src to path for development
src_path = Path(__file__).parent.parent.parent
if src_path.exists():
    sys.path.insert(0, str(src_path))

from ops0 import __version__
from ops0.core import Pipeline, Step
from ops0.core.config import Config
from ops0.core.storage import LocalStorage
from ops0.runtime.local import LocalRuntime
from ops0.runtime.docker import DockerRuntime
from ops0.runtime.kubernetes import KubernetesRuntime

# Import CLI commands
from .init import init
from .run import run
from .deploy import deploy
from .status import status
from .logs import logs
from .rollback import rollback
from .list import list_cmd
from .describe import describe
from .doctor import doctor
from .config import config_cmd

# Initialize console
console = Console()

# Create main app
app = typer.Typer(
    name="ops0",
    help="ops0 - Where Python meets Production 🚀\n\nTransform Python code into production ML pipelines.",
    add_completion=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)

# Add subcommands
app.command()(init)
app.command()(run)
app.command()(deploy)
app.command()(status)
app.command()(logs)
app.command()(rollback)
app.command(name="list")(list_cmd)
app.command()(describe)
app.command()(doctor)
app.command(name="config")(config_cmd)


@app.callback()
def main_callback(
        version: bool = typer.Option(
            None, "--version", "-v",
            help="Show ops0 version",
            is_eager=True
        ),
        debug: bool = typer.Option(
            False, "--debug", "-d",
            help="Enable debug logging"
        )
):
    """
    ops0 - Transform Python code into production ML pipelines.

    Write Python, Ship Production 🚀
    """
    if version:
        console.print(f"ops0 version {__version__}")
        raise typer.Exit()

    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        os.environ["OPS0_LOG_LEVEL"] = "DEBUG"


@app.command()
def quickstart():
    """
    Show quickstart guide for ops0.

    This command displays a quick tutorial on how to get started with ops0.
    """
    quickstart_content = """
[bold cyan]Welcome to ops0! 🚀[/bold cyan]

[bold]1. Initialize your project:[/bold]
   $ ops0 init

[bold]2. Write your pipeline:[/bold]
   ```python
   import ops0

   @ops0.step
   def preprocess(data):
       return data.dropna()

   @ops0.step  
   def train(data):
       model = train_model(data)
       ops0.save_model(model, "my-model")
       return {"accuracy": 0.95}
   ```

[bold]3. Test locally:[/bold]
   $ ops0 run --local

[bold]4. Deploy to production:[/bold]
   $ ops0 deploy

[bold]5. Monitor your pipeline:[/bold]
   $ ops0 status
   $ ops0 logs

[dim]For more information: https://docs.ops0.xyz[/dim]
"""

    console.print(Panel(
        quickstart_content,
        title="[bold green]ops0 Quickstart Guide[/bold green]",
        border_style="green",
        padding=(1, 2)
    ))


@app.command()
def playground():
    """
    Open ops0 playground in your browser.

    The playground provides an interactive environment to experiment with ops0.
    """
    import webbrowser

    url = "https://try.ops0.xyz"
    console.print(f"[green]Opening ops0 playground at {url}...[/green]")

    try:
        webbrowser.open(url)
    except Exception as e:
        console.print(f"[red]Could not open browser:[/red] {e}")
        console.print(f"Please visit {url} manually.")


@app.command()
def upgrade():
    """
    Upgrade ops0 to the latest version.
    """
    import subprocess

    console.print("[blue]Checking for updates...[/blue]")

    try:
        # Get current version
        current = __version__

        # Check PyPI for latest version
        result = subprocess.run(
            [sys.executable, "-m", "pip", "index", "versions", "ops0"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            # Parse latest version from output
            lines = result.stdout.strip().split('\n')
            latest = None
            for line in lines:
                if "Available versions:" in line:
                    versions = line.split(":")[1].strip().split(", ")
                    latest = versions[0]
                    break

            if latest and latest != current:
                console.print(f"[yellow]New version available: {latest} (current: {current})[/yellow]")

                # Confirm upgrade
                if typer.confirm("Do you want to upgrade?"):
                    console.print("[blue]Upgrading ops0...[/blue]")
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "--upgrade", "ops0"],
                        check=True
                    )
                    console.print("[green]✓ Successfully upgraded ops0![/green]")
            else:
                console.print("[green]✓ You're already on the latest version![/green]")
        else:
            console.print("[red]Could not check for updates[/red]")

    except Exception as e:
        console.print(f"[red]Error during upgrade:[/red] {e}")
        raise typer.Exit(1)


def main():
    """Main entry point."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        if os.getenv("OPS0_LOG_LEVEL") == "DEBUG":
            raise
        else:
            console.print(f"[red]Error:[/red] {e}")
            console.print("[dim]Run with --debug for more details[/dim]")
            raise typer.Exit(1)


if __name__ == "__main__":
    main()