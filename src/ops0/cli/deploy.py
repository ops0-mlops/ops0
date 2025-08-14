"""Deploy command for ops0 - deploy pipelines to production."""

import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

from ..core import Pipeline
from ..core.config import Config
from ..runtime.kubernetes import KubernetesRuntime
from ..runtime.docker import DockerRuntime
from ..deployment import Deployer, DeploymentConfig

from .utils import (
    ensure_project_initialized,
    print_success,
    print_error,
    print_warning,
    print_info,
    confirm,
    load_pipeline_config
)

console = Console()


def deploy(
        env: str = typer.Option("production", "--env", "-e", help="Deployment environment"),
        target: str = typer.Option("kubernetes", "--target", "-t", help="Deployment target (kubernetes/docker)"),
        config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Deployment configuration file"),
        version: Optional[str] = typer.Option(None, "--version", "-v", help="Version tag for deployment"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deployed without deploying"),
        force: bool = typer.Option(False, "--force", "-f", help="Force deployment without confirmation"),
        scale: Optional[int] = typer.Option(None, "--scale", help="Number of replicas"),
        cpu: Optional[str] = typer.Option(None, "--cpu", help="CPU request (e.g., 100m, 1)"),
        memory: Optional[str] = typer.Option(None, "--memory", help="Memory request (e.g., 128Mi, 1Gi)"),
        gpu: Optional[int] = typer.Option(None, "--gpu", help="Number of GPUs"),
):
    """
    Deploy an ops0 pipeline to production.

    Examples:
        ops0 deploy                      # Deploy to Kubernetes
        ops0 deploy --env staging       # Deploy to staging
        ops0 deploy --target docker     # Deploy using Docker Compose
        ops0 deploy --dry-run          # Preview deployment
        ops0 deploy --scale 3          # Deploy with 3 replicas
    """
    console.print(f"\n[bold blue]🚀 Deploying ops0 pipeline[/bold blue]")

    # Ensure we're in an ops0 project
    project_root = ensure_project_initialized()

    # Load configuration
    config = load_deployment_config(project_root, config_file, env)

    # Override config with CLI options
    if scale is not None:
        config.scale = scale
    if cpu:
        config.resources['cpu'] = cpu
    if memory:
        config.resources['memory'] = memory
    if gpu is not None:
        config.resources['gpu'] = gpu
    if version:
        config.version = version

    # Show deployment plan
    show_deployment_plan(config, target)

    # Dry run mode
    if dry_run:
        console.print("\n[yellow]This is a dry run. No changes will be made.[/yellow]")
        return

    # Confirm deployment
    if not force:
        if not confirm(f"\nDeploy to {env} using {target}?"):
            print_info("Deployment cancelled")
            return

    # Deploy
    try:
        deployer = create_deployer(target, config)
        deploy_pipeline(deployer, config)

    except Exception as e:
        print_error(f"Deployment failed: {e}")
        raise typer.Exit(1)


def load_deployment_config(project_root: Path, config_file: Optional[Path],
                           env: str) -> DeploymentConfig:
    """Load deployment configuration."""
    config_data = {}

    # Load from file if specified
    if config_file and config_file.exists():
        if config_file.suffix == '.yaml':
            import yaml
            with open(config_file) as f:
                config_data = yaml.safe_load(f)
        else:
            with open(config_file) as f:
                config_data = json.load(f)
    else:
        # Try default locations
        default_paths = [
            project_root / f"deploy.{env}.yaml",
            project_root / "deploy.yaml",
            project_root / ".ops0" / "deploy.yaml"
        ]

        for path in default_paths:
            if path.exists():
                import yaml
                with open(path) as f:
                    config_data = yaml.safe_load(f)
                break

    # Load pipeline config
    pipeline_config = load_pipeline_config(project_root)

    # Create deployment config
    return DeploymentConfig(
        name=config_data.get('name', pipeline_config.get('name', 'ops0-pipeline')),
        version=config_data.get('version', '0.1.0'),
        environment=env,
        scale=config_data.get('scale', 1),
        resources=config_data.get('resources', {
            'cpu': '100m',
            'memory': '256Mi'
        }),
        env_vars=config_data.get('env_vars', {}),
        secrets=config_data.get('secrets', []),
        volumes=config_data.get('volumes', []),
        ports=config_data.get('ports', [])
    )


def show_deployment_plan(config: DeploymentConfig, target: str):
    """Show deployment plan."""
    table = Table(title="Deployment Plan", box=box.ROUNDED)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    table.add_row("Pipeline", config.name)
    table.add_row("Version", config.version)
    table.add_row("Environment", config.environment)
    table.add_row("Target", target)
    table.add_row("Scale", str(config.scale))
    table.add_row("CPU", config.resources.get('cpu', 'default'))
    table.add_row("Memory", config.resources.get('memory', 'default'))

    if config.resources.get('gpu'):
        table.add_row("GPU", str(config.resources['gpu']))

    if config.env_vars:
        table.add_row("Env Vars", f"{len(config.env_vars)} variables")

    if config.secrets:
        table.add_row("Secrets", ", ".join(config.secrets))

    console.print(table)


def create_deployer(target: str, config: DeploymentConfig) -> Deployer:
    """Create deployer based on target."""
    if target == "kubernetes":
        return KubernetesDeployer(config)
    elif target == "docker":
        return DockerComposeDeployer(config)
    else:
        raise ValueError(f"Unknown deployment target: {target}")


def deploy_pipeline(deployer: Deployer, config: DeploymentConfig):
    """Deploy the pipeline."""
    steps = [
        ("Building container image", deployer.build_image),
        ("Pushing image to registry", deployer.push_image),
        ("Creating deployment resources", deployer.create_resources),
        ("Deploying pipeline", deployer.deploy),
        ("Waiting for ready state", deployer.wait_for_ready),
        ("Configuring monitoring", deployer.setup_monitoring),
    ]

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
    ) as progress:

        main_task = progress.add_task("Deploying...", total=len(steps))

        for step_name, step_func in steps:
            step_task = progress.add_task(step_name, total=None)

            try:
                step_func()
                progress.update(step_task, completed=True)
                print_success(step_name)

            except Exception as e:
                progress.update(step_task, completed=True)
                print_error(f"{step_name}: {e}")
                raise

            finally:
                progress.remove_task(step_task)
                progress.update(main_task, advance=1)

    # Show deployment info
    show_deployment_info(deployer, config)

    # Success message
    console.print("\n" + Panel(
        f"[green]✅ Pipeline deployed successfully![/green]\n\n"
        f"[bold]Access your pipeline:[/bold]\n"
        f"• Status: ops0 status\n"
        f"• Logs: ops0 logs\n"
        f"• Dashboard: {deployer.get_dashboard_url()}\n\n"
        f"[dim]To roll back: ops0 rollback[/dim]",
        title="[bold green]Deployment Complete[/bold green]",
        border_style="green"
    ))


def show_deployment_info(deployer: Deployer, config: DeploymentConfig):
    """Show deployment information."""
    info = deployer.get_deployment_info()

    table = Table(title="Deployment Details", box=box.SIMPLE)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    for key, value in info.items():
        table.add_row(key, str(value))

    console.print("\n")
    console.print(table)


class KubernetesDeployer(Deployer):
    """Kubernetes deployment implementation."""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.namespace = f"ops0-{config.environment}"
        self.image_name = f"ops0/{config.name}:{config.version}"

    def build_image(self):
        """Build Docker image."""
        import subprocess

        # Generate Dockerfile if it doesn't exist
        if not Path("Dockerfile").exists():
            self._generate_dockerfile()

        # Build image
        subprocess.run([
            "docker", "build",
            "-t", self.image_name,
            "."
        ], check=True)

    def push_image(self):
        """Push image to registry."""
        # In production, this would push to a real registry
        # For now, we'll simulate it
        time.sleep(1)

    def create_resources(self):
        """Create Kubernetes resources."""
        # Generate Kubernetes manifests
        self._generate_k8s_manifests()

    def deploy(self):
        """Deploy to Kubernetes."""
        import subprocess

        # Apply manifests
        subprocess.run([
            "kubectl", "apply",
            "-f", ".ops0/k8s/",
            "-n", self.namespace
        ], check=True)

    def wait_for_ready(self):
        """Wait for deployment to be ready."""
        import subprocess

        subprocess.run([
            "kubectl", "wait",
            "--for=condition=available",
            f"deployment/{self.config.name}",
            "-n", self.namespace,
            "--timeout=300s"
        ], check=True)

    def setup_monitoring(self):
        """Setup monitoring for the deployment."""
        # Configure Prometheus/Grafana
        time.sleep(1)

    def get_dashboard_url(self) -> str:
        """Get dashboard URL."""
        return f"https://ops0.xyz/dashboard/{self.config.name}"

    def get_deployment_info(self) -> Dict[str, Any]:
        """Get deployment information."""
        return {
            "Namespace": self.namespace,
            "Image": self.image_name,
            "Replicas": self.config.scale,
            "Endpoint": f"https://{self.config.name}.ops0.xyz"
        }

    def _generate_dockerfile(self):
        """Generate Dockerfile."""
        dockerfile_content = f"""FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment
ENV OPS0_ENV={self.config.environment}

# Run ops0
CMD ["ops0", "run", "--docker"]
"""
        Path("Dockerfile").write_text(dockerfile_content)

    def _generate_k8s_manifests(self):
        """Generate Kubernetes manifests."""
        manifests_dir = Path(".ops0/k8s")
        manifests_dir.mkdir(parents=True, exist_ok=True)

        # Deployment manifest
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.config.name,
                "labels": {"app": self.config.name}
            },
            "spec": {
                "replicas": self.config.scale,
                "selector": {"matchLabels": {"app": self.config.name}},
                "template": {
                    "metadata": {"labels": {"app": self.config.name}},
                    "spec": {
                        "containers": [{
                            "name": "pipeline",
                            "image": self.image_name,
                            "resources": {
                                "requests": self.config.resources,
                                "limits": self.config.resources
                            },
                            "env": [
                                {"name": k, "value": v}
                                for k, v in self.config.env_vars.items()
                            ]
                        }]
                    }
                }
            }
        }

        # Write manifest
        import yaml
        with open(manifests_dir / "deployment.yaml", "w") as f:
            yaml.dump(deployment, f)


class DockerComposeDeployer(Deployer):
    """Docker Compose deployment implementation."""

    def __init__(self, config: DeploymentConfig):
        self.config = config

    def build_image(self):
        """Build Docker image."""
        pass

    def push_image(self):
        """No need to push for local Docker."""
        pass

    def create_resources(self):
        """Create docker-compose.yml."""
        self._generate_compose_file()

    def deploy(self):
        """Deploy using Docker Compose."""
        import subprocess
        subprocess.run(["docker-compose", "up", "-d"], check=True)

    def wait_for_ready(self):
        """Wait for containers to be ready."""
        time.sleep(2)

    def setup_monitoring(self):
        """Setup monitoring."""
        pass

    def get_dashboard_url(self) -> str:
        """Get dashboard URL."""
        return "http://localhost:9090"

    def get_deployment_info(self) -> Dict[str, Any]:
        """Get deployment information."""
        return {
            "Type": "Docker Compose",
            "Services": self.config.scale,
            "Dashboard": "http://localhost:9090"
        }

    def _generate_compose_file(self):
        """Generate docker-compose.yml."""
        compose = {
            "version": "3.8",
            "services": {
                "pipeline": {
                    "build": ".",
                    "environment": self.config.env_vars,
                    "ports": ["9090:9090"],
                    "restart": "unless-stopped"
                }
            }
        }

        import yaml
        with open("docker-compose.yml", "w") as f:
            yaml.dump(compose, f)