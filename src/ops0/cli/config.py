"""Config command for ops0 - manage configuration settings."""

import os
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich import box

from .core.config import Config, ConfigManager

from .utils import (
    ensure_project_initialized,
    print_success,
    print_error,
    print_warning,
    print_info,
    confirm
)

console = Console()


def config_cmd(
        action: str = typer.Argument("list", help="Action to perform (list, get, set, unset, edit, validate)"),
        key: Optional[str] = typer.Argument(None, help="Configuration key"),
        value: Optional[str] = typer.Argument(None, help="Configuration value"),
        env: Optional[str] = typer.Option(None, "--env", "-e", help="Environment (dev, staging, prod)"),
        global_config: bool = typer.Option(False, "--global", "-g", help="Use global configuration"),
        format: str = typer.Option("yaml", "--format", "-f", help="Output format (yaml, json, env)"),
        file: Optional[Path] = typer.Option(None, "--file", help="Configuration file to use"),
        validate_only: bool = typer.Option(False, "--validate", help="Validate configuration only"),
):
    """
    Manage ops0 configuration.

    Examples:
        ops0 config                          # List all settings
        ops0 config get api.key             # Get specific value
        ops0 config set api.key abc123      # Set value
        ops0 config unset api.key           # Remove value
        ops0 config edit                    # Edit in editor
        ops0 config validate                # Validate config
        ops0 config --env prod              # Use prod config
        ops0 config --global                # Use global config
    """
    # Get project root or use global config
    if global_config:
        config_dir = Path.home() / ".ops0"
        config_dir.mkdir(exist_ok=True)
    else:
        project_root = ensure_project_initialized()
        config_dir = project_root / ".ops0"

    # Get config manager
    manager = ConfigManager(config_dir, environment=env)

    # Handle actions
    if action == "list":
        list_config(manager, format)
    elif action == "get":
        if not key:
            print_error("Key required for 'get' action")
            raise typer.Exit(1)
        get_config_value(manager, key, format)
    elif action == "set":
        if not key:
            print_error("Key required for 'set' action")
            raise typer.Exit(1)
        if value is None:
            value = Prompt.ask(f"Value for {key}")
        set_config_value(manager, key, value)
    elif action == "unset":
        if not key:
            print_error("Key required for 'unset' action")
            raise typer.Exit(1)
        unset_config_value(manager, key)
    elif action == "edit":
        edit_config(manager, file)
    elif action == "validate":
        validate_config(manager, file)
    elif action == "init":
        init_config(manager)
    elif action == "show-env":
        show_environment_config(manager)
    elif action == "migrate":
        migrate_config(manager)
    else:
        print_error(f"Unknown action: {action}")
        console.print("Valid actions: list, get, set, unset, edit, validate, init, show-env, migrate")
        raise typer.Exit(1)


def list_config(manager: ConfigManager, output_format: str):
    """List all configuration settings."""
    config = manager.get_all()

    if not config:
        print_info("No configuration found")
        return

    if output_format == "json":
        console.print(json.dumps(config, indent=2))
    elif output_format == "env":
        # Export as environment variables
        for key, value in flatten_config(config).items():
            env_key = f"OPS0_{key.upper().replace('.', '_')}"
            console.print(f"{env_key}={value}")
    else:  # yaml or table
        # Show as table for better readability
        table = Table(title="ops0 Configuration", box=box.ROUNDED)
        table.add_column("Key", style="cyan")
        table.add_column("Value")
        table.add_column("Source")

        # Flatten nested config
        for key, value in flatten_config(config).items():
            source = manager.get_source(key)
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value)
            else:
                value_str = str(value)

            # Mask sensitive values
            if any(sensitive in key.lower() for sensitive in ['key', 'secret', 'password', 'token']):
                if value_str and len(value_str) > 4:
                    value_str = value_str[:4] + "*" * (len(value_str) - 4)

            table.add_row(key, value_str, source)

        console.print(table)

        # Show active environment
        console.print(f"\n[dim]Active environment: {manager.environment or 'default'}[/dim]")


def get_config_value(manager: ConfigManager, key: str, output_format: str):
    """Get a specific configuration value."""
    value = manager.get(key)

    if value is None:
        print_error(f"Configuration key '{key}' not found")

        # Suggest similar keys
        all_keys = list(flatten_config(manager.get_all()).keys())
        similar = [k for k in all_keys if key.lower() in k.lower()]
        if similar:
            console.print("\nDid you mean one of these?")
            for k in similar[:5]:
                console.print(f"  • {k}")

        raise typer.Exit(1)

    # Output value
    if output_format == "json":
        console.print(json.dumps({key: value}, indent=2))
    elif output_format == "env":
        env_key = f"OPS0_{key.upper().replace('.', '_')}"
        console.print(f"{env_key}={value}")
    else:
        if isinstance(value, (dict, list)):
            console.print(Syntax(yaml.dump({key: value}), "yaml", theme="monokai"))
        else:
            console.print(value)


def set_config_value(manager: ConfigManager, key: str, value: str):
    """Set a configuration value."""
    # Parse value if it looks like JSON
    try:
        if value.startswith('{') or value.startswith('['):
            parsed_value = json.loads(value)
        elif value.lower() in ['true', 'false']:
            parsed_value = value.lower() == 'true'
        elif value.isdigit():
            parsed_value = int(value)
        elif '.' in value and all(part.isdigit() for part in value.split('.')):
            parsed_value = float(value)
        else:
            parsed_value = value
    except:
        parsed_value = value

    # Validate the key
    if not validate_config_key(key):
        print_error(f"Invalid configuration key: {key}")
        console.print("Keys should be in format: section.subsection.name")
        raise typer.Exit(1)

    # Set the value
    old_value = manager.get(key)
    manager.set(key, parsed_value)

    if old_value is None:
        print_success(f"Set {key} = {parsed_value}")
    else:
        print_success(f"Updated {key} = {parsed_value} (was: {old_value})")

    # Save configuration
    manager.save()


def unset_config_value(manager: ConfigManager, key: str):
    """Remove a configuration value."""
    if not manager.exists(key):
        print_error(f"Configuration key '{key}' not found")
        raise typer.Exit(1)

    # Confirm deletion
    value = manager.get(key)
    if not confirm(f"Remove {key} = {value}?"):
        print_info("Cancelled")
        return

    manager.unset(key)
    manager.save()
    print_success(f"Removed {key}")


def edit_config(manager: ConfigManager, file: Optional[Path]):
    """Edit configuration in default editor."""
    import tempfile
    import subprocess

    # Get current config
    if file and file.exists():
        with open(file) as f:
            if file.suffix == '.json':
                config = json.load(f)
            else:
                config = yaml.safe_load(f)
    else:
        config = manager.get_all()

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
        yaml.dump(config, tmp, default_flow_style=False)
        tmp_path = tmp.name

    # Get editor
    editor = os.environ.get('EDITOR', 'nano' if os.name != 'nt' else 'notepad')

    try:
        # Open in editor
        subprocess.run([editor, tmp_path], check=True)

        # Read back
        with open(tmp_path) as f:
            new_config = yaml.safe_load(f)

        # Validate
        validation_errors = validate_config_data(new_config)
        if validation_errors:
            print_error("Configuration validation failed:")
            for error in validation_errors:
                console.print(f"  • {error}")
            raise typer.Exit(1)

        # Apply changes
        manager.update(new_config)
        manager.save()
        print_success("Configuration updated")

    finally:
        # Clean up
        Path(tmp_path).unlink(missing_ok=True)


def validate_config(manager: ConfigManager, file: Optional[Path]):
    """Validate configuration."""
    if file and file.exists():
        # Validate specific file
        try:
            with open(file) as f:
                if file.suffix == '.json':
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)

            errors = validate_config_data(config)

        except Exception as e:
            print_error(f"Failed to load config file: {e}")
            raise typer.Exit(1)
    else:
        # Validate current config
        config = manager.get_all()
        errors = validate_config_data(config)

    if errors:
        console.print("[red]❌ Configuration validation failed[/red]\n")
        for error in errors:
            console.print(f"  • {error}")
        raise typer.Exit(1)
    else:
        console.print("[green]✅ Configuration is valid[/green]")

        # Show config summary
        summary = get_config_summary(config)
        if summary:
            console.print("\n[bold]Configuration Summary:[/bold]")
            for item in summary:
                console.print(f"  • {item}")


def init_config(manager: ConfigManager):
    """Initialize configuration with defaults."""
    console.print("[bold]Initializing ops0 configuration...[/bold]\n")

    # Check if config already exists
    if manager.exists_any():
        if not confirm("Configuration already exists. Overwrite?"):
            print_info("Cancelled")
            return

    # Gather basic configuration
    config = {}

    # Project settings
    console.print("[bold]Project Settings:[/bold]")
    config['project'] = {
        'name': Prompt.ask("Project name", default="my-ops0-project"),
        'version': Prompt.ask("Version", default="0.1.0"),
        'description': Prompt.ask("Description", default=""),
    }

    # Runtime settings
    console.print("\n[bold]Runtime Settings:[/bold]")
    runtime_type = Prompt.ask(
        "Default runtime",
        choices=["local", "docker", "kubernetes"],
        default="local"
    )
    config['runtime'] = {
        'type': runtime_type,
        'default_resources': {
            'cpu': Prompt.ask("Default CPU request", default="100m"),
            'memory': Prompt.ask("Default memory request", default="256Mi")
        }
    }

    # Storage settings
    console.print("\n[bold]Storage Settings:[/bold]")
    storage_type = Prompt.ask(
        "Storage backend",
        choices=["local", "s3", "gcs", "azure"],
        default="local"
    )
    config['storage'] = {'type': storage_type}

    if storage_type != "local":
        config['storage']['bucket'] = Prompt.ask(f"{storage_type.upper()} bucket name")

    # API Keys (optional)
    console.print("\n[bold]API Keys (optional, press Enter to skip):[/bold]")
    api_keys = {}

    openai_key = Prompt.ask("OpenAI API Key", password=True, default="")
    if openai_key:
        api_keys['openai'] = openai_key

    aws_key = Prompt.ask("AWS Access Key ID", password=True, default="")
    if aws_key:
        api_keys['aws_access_key_id'] = aws_key
        api_keys['aws_secret_access_key'] = Prompt.ask("AWS Secret Access Key", password=True)

    if api_keys:
        config['api_keys'] = api_keys

    # Monitoring settings
    console.print("\n[bold]Monitoring Settings:[/bold]")
    config['monitoring'] = {
        'enabled': Prompt.ask("Enable monitoring?", choices=["y", "n"], default="y") == "y",
        'port': int(Prompt.ask("Monitoring port", default="9090")),
    }

    # Save configuration
    manager.update(config)
    manager.save()

    print_success("Configuration initialized successfully!")

    # Show next steps
    console.print("\n[bold]Next steps:[/bold]")
    console.print("1. Review configuration: ops0 config list")
    console.print("2. Set environment-specific values: ops0 config set --env prod key value")
    console.print("3. Validate configuration: ops0 config validate")


def show_environment_config(manager: ConfigManager):
    """Show environment-specific configuration."""
    envs = ['development', 'staging', 'production']

    table = Table(title="Environment Configuration", box=box.ROUNDED)
    table.add_column("Setting", style="cyan")

    for env in envs:
        table.add_column(env.capitalize())

    # Get config for each environment
    configs = {}
    for env in envs:
        env_manager = ConfigManager(manager.config_dir, environment=env)
        configs[env] = env_manager.get_all()

    # Find all keys
    all_keys = set()
    for config in configs.values():
        all_keys.update(flatten_config(config).keys())

    # Show differences
    for key in sorted(all_keys):
        row = [key]
        for env in envs:
            value = get_nested_value(configs[env], key)
            if value is None:
                row.append("[dim]not set[/dim]")
            else:
                # Mask sensitive values
                if any(s in key.lower() for s in ['key', 'secret', 'password', 'token']):
                    if isinstance(value, str) and len(value) > 4:
                        value = value[:4] + "*" * (len(value) - 4)
                row.append(str(value))

        table.add_row(*row)

    console.print(table)


def migrate_config(manager: ConfigManager):
    """Migrate configuration to new format."""
    console.print("[bold]Migrating configuration...[/bold]\n")

    # Backup current config
    backup_path = manager.config_dir / "config.backup.yaml"
    current_config = manager.get_all()

    with open(backup_path, 'w') as f:
        yaml.dump(current_config, f)

    print_success(f"Created backup at {backup_path}")

    # Apply migrations
    migrations = [
        migrate_v1_to_v2,
        migrate_api_keys,
        migrate_runtime_config,
    ]

    migrated = current_config.copy()
    changes = []

    for migration in migrations:
        result, migration_changes = migration(migrated)
        if migration_changes:
            migrated = result
            changes.extend(migration_changes)

    if not changes:
        print_info("No migrations needed")
        return

    # Show changes
    console.print("[bold]Migration changes:[/bold]")
    for change in changes:
        console.print(f"  • {change}")

    # Confirm
    if not confirm("\nApply migrations?"):
        print_info("Migration cancelled")
        return

    # Apply migrations
    manager.update(migrated)
    manager.save()

    print_success("Configuration migrated successfully!")


# Helper functions

def flatten_config(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested configuration."""
    items = {}

    for key, value in config.items():
        new_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict) and not key.endswith('_raw'):
            items.update(flatten_config(value, new_key))
        else:
            items[new_key] = value

    return items


def get_nested_value(config: Dict[str, Any], key: str) -> Any:
    """Get value from nested dict using dot notation."""
    parts = key.split('.')
    value = config

    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None

    return value


def validate_config_key(key: str) -> bool:
    """Validate configuration key format."""
    import re
    return bool(re.match(r'^[a-z_]+(\.[a-z_]+)*$', key))


def validate_config_data(config: Dict[str, Any]) -> List[str]:
    """Validate configuration data."""
    errors = []

    # Required fields
    required = ['project.name', 'runtime.type']
    for field in required:
        if get_nested_value(config, field) is None:
            errors.append(f"Missing required field: {field}")

    # Validate runtime type
    runtime_type = get_nested_value(config, 'runtime.type')
    if runtime_type and runtime_type not in ['local', 'docker', 'kubernetes']:
        errors.append(f"Invalid runtime type: {runtime_type}")

    # Validate storage type
    storage_type = get_nested_value(config, 'storage.type')
    if storage_type and storage_type not in ['local', 's3', 'gcs', 'azure']:
        errors.append(f"Invalid storage type: {storage_type}")

    # Validate resources
    cpu = get_nested_value(config, 'runtime.default_resources.cpu')
    if cpu and not validate_cpu_value(cpu):
        errors.append(f"Invalid CPU value: {cpu}")

    memory = get_nested_value(config, 'runtime.default_resources.memory')
    if memory and not validate_memory_value(memory):
        errors.append(f"Invalid memory value: {memory}")

    # Validate port numbers
    port = get_nested_value(config, 'monitoring.port')
    if port and (not isinstance(port, int) or port < 1 or port > 65535):
        errors.append(f"Invalid port number: {port}")

    return errors


def validate_cpu_value(cpu: str) -> bool:
    """Validate CPU resource value."""
    import re
    return bool(re.match(r'^\d+(\.\d+)?m?$', str(cpu)))


def validate_memory_value(memory: str) -> bool:
    """Validate memory resource value."""
    import re
    return bool(re.match(r'^\d+(\.\d+)?[KMGT]i?$', str(memory)))


def get_config_summary(config: Dict[str, Any]) -> List[str]:
    """Get configuration summary."""
    summary = []

    # Project info
    project_name = get_nested_value(config, 'project.name')
    if project_name:
        summary.append(f"Project: {project_name}")

    # Runtime
    runtime = get_nested_value(config, 'runtime.type')
    if runtime:
        summary.append(f"Runtime: {runtime}")

    # Storage
    storage = get_nested_value(config, 'storage.type')
    if storage:
        summary.append(f"Storage: {storage}")

    # API keys
    api_keys = config.get('api_keys', {})
    if api_keys:
        summary.append(f"API Keys configured: {', '.join(api_keys.keys())}")

    # Monitoring
    monitoring = get_nested_value(config, 'monitoring.enabled')
    if monitoring:
        port = get_nested_value(config, 'monitoring.port')
        summary.append(f"Monitoring: enabled (port {port})")

    return summary


def migrate_v1_to_v2(config: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
    """Migrate v1 config to v2 format."""
    changes = []
    new_config = config.copy()

    # Migrate old fields
    if 'pipeline_name' in config:
        new_config['project'] = {'name': config['pipeline_name']}
        del new_config['pipeline_name']
        changes.append("Migrated pipeline_name to project.name")

    return new_config, changes


def migrate_api_keys(config: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
    """Migrate API keys to secure format."""
    changes = []
    new_config = config.copy()

    # Move top-level API keys
    keys_to_move = ['openai_api_key', 'aws_access_key_id', 'aws_secret_access_key']
    api_keys = new_config.get('api_keys', {})

    for key in keys_to_move:
        if key in config:
            api_keys[key.replace('_api_key', '')] = config[key]
            del new_config[key]
            changes.append(f"Moved {key} to api_keys section")

    if api_keys:
        new_config['api_keys'] = api_keys

    return new_config, changes


def migrate_runtime_config(config: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
    """Migrate runtime configuration."""
    changes = []
    new_config = config.copy()

    # Migrate deployment settings
    if 'deployment' in config:
        if 'runtime' not in new_config:
            new_config['runtime'] = {}

        new_config['runtime'].update(config['deployment'])
        del new_config['deployment']
        changes.append("Migrated deployment settings to runtime section")

    return new_config, changes


class ConfigManager:
    """Configuration manager for ops0."""

    def __init__(self, config_dir: Path, environment: Optional[str] = None):
        self.config_dir = config_dir
        self.environment = environment
        self._config = {}
        self._load()

    def _load(self):
        """Load configuration from files."""
        # Load base config
        base_config_file = self.config_dir / "config.yaml"
        if base_config_file.exists():
            with open(base_config_file) as f:
                self._config = yaml.safe_load(f) or {}

        # Load environment-specific config
        if self.environment:
            env_config_file = self.config_dir / f"config.{self.environment}.yaml"
            if env_config_file.exists():
                with open(env_config_file) as f:
                    env_config = yaml.safe_load(f) or {}
                    self._merge_config(self._config, env_config)

    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration."""
        return self._config.copy()

    def get(self, key: str) -> Any:
        """Get configuration value."""
        return get_nested_value(self._config, key)

    def set(self, key: str, value: Any):
        """Set configuration value."""
        parts = key.split('.')
        config = self._config

        # Navigate to the parent
        for part in parts[:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]

        # Set the value
        config[parts[-1]] = value

    def unset(self, key: str):
        """Remove configuration value."""
        parts = key.split('.')
        config = self._config

        # Navigate to the parent
        for part in parts[:-1]:
            if part not in config:
                return
            config = config[part]

        # Remove the value
        if parts[-1] in config:
            del config[parts[-1]]

    def exists(self, key: str) -> bool:
        """Check if configuration key exists."""
        return self.get(key) is not None

    def exists_any(self) -> bool:
        """Check if any configuration exists."""
        return bool(self._config)

    def update(self, config: Dict[str, Any]):
        """Update configuration."""
        self._merge_config(self._config, config)

    def save(self):
        """Save configuration to file."""
        config_file = self.config_dir / "config.yaml"
        if self.environment:
            config_file = self.config_dir / f"config.{self.environment}.yaml"

        with open(config_file, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)

    def get_source(self, key: str) -> str:
        """Get the source of a configuration value."""
        # Check environment variable
        env_key = f"OPS0_{key.upper().replace('.', '_')}"
        if env_key in os.environ:
            return "env"

        # Check environment-specific config
        if self.environment:
            env_config_file = self.config_dir / f"config.{self.environment}.yaml"
            if env_config_file.exists():
                with open(env_config_file) as f:
                    env_config = yaml.safe_load(f) or {}
                    if get_nested_value(env_config, key) is not None:
                        return f"config.{self.environment}.yaml"

        # Check base config
        base_config_file = self.config_dir / "config.yaml"
        if base_config_file.exists():
            return "config.yaml"

        return "default"