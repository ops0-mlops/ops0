"""
ops0 Configuration Management

Handles configuration for ops0 CLI and runtime behavior.
"""

import os
import json
import tomllib
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from rich.table import Table
from rich.panel import Panel

from .utils import console, print_success, print_error, print_warning


@dataclass
class Ops0Config:
    """ops0 configuration data structure"""

    # Core settings
    project_name: str = "ops0-project"
    environment: str = "development"
    log_level: str = "INFO"

    # Storage settings
    storage_backend: str = "local"
    storage_path: str = ".ops0/storage"

    # Container settings
    container_registry: str = "ghcr.io/ops0-mlops"
    build_containers: bool = False
    push_containers: bool = False

    # Deployment settings
    default_env: str = "production"
    auto_deploy: bool = False
    deployment_timeout: int = 300  # seconds

    # Monitoring settings
    enable_monitoring: bool = True
    metrics_retention_days: int = 30
    alert_channels: list = None

    # Development settings
    dev_mode: bool = False
    hot_reload: bool = False
    debug_steps: bool = False

    # Cloud settings
    cloud_provider: Optional[str] = None
    aws_region: Optional[str] = None
    gcp_project: Optional[str] = None
    azure_subscription: Optional[str] = None

    # API settings
    api_key: Optional[str] = None
    api_url: str = "https://api.ops0.xyz"
    timeout: int = 30

    def __post_init__(self):
        if self.alert_channels is None:
            self.alert_channels = []


class ConfigManager:
    """Manages ops0 configuration with multiple sources"""

    def __init__(self):
        self.config = Ops0Config()
        self.config_file_path: Optional[Path] = None

        # Load configuration from multiple sources in order
        self._load_defaults()
        self._load_from_env()
        self._load_from_file()

    def _load_defaults(self):
        """Load default configuration"""
        # Defaults are already in Ops0Config dataclass
        pass

    def _load_from_env(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'OPS0_PROJECT_NAME': 'project_name',
            'OPS0_ENV': 'environment',
            'OPS0_LOG_LEVEL': 'log_level',
            'OPS0_STORAGE_BACKEND': 'storage_backend',
            'OPS0_STORAGE_PATH': 'storage_path',
            'OPS0_CONTAINER_REGISTRY': 'container_registry',
            'OPS0_BUILD_CONTAINERS': ('build_containers', bool),
            'OPS0_PUSH_CONTAINERS': ('push_containers', bool),
            'OPS0_DEFAULT_ENV': 'default_env',
            'OPS0_AUTO_DEPLOY': ('auto_deploy', bool),
            'OPS0_DEPLOYMENT_TIMEOUT': ('deployment_timeout', int),
            'OPS0_ENABLE_MONITORING': ('enable_monitoring', bool),
            'OPS0_METRICS_RETENTION_DAYS': ('metrics_retention_days', int),
            'OPS0_DEV_MODE': ('dev_mode', bool),
            'OPS0_HOT_RELOAD': ('hot_reload', bool),
            'OPS0_DEBUG_STEPS': ('debug_steps', bool),
            'OPS0_CLOUD_PROVIDER': 'cloud_provider',
            'OPS0_AWS_REGION': 'aws_region',
            'OPS0_GCP_PROJECT': 'gcp_project',
            'OPS0_AZURE_SUBSCRIPTION': 'azure_subscription',
            'OPS0_API_KEY': 'api_key',
            'OPS0_API_URL': 'api_url',
            'OPS0_TIMEOUT': ('timeout', int),
        }

        for env_var, config_attr in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                if isinstance(config_attr, tuple):
                    attr_name, attr_type = config_attr
                    try:
                        if attr_type == bool:
                            value = env_value.lower() in ('true', '1', 'yes', 'on')
                        elif attr_type == int:
                            value = int(env_value)
                        else:
                            value = env_value
                        setattr(self.config, attr_name, value)
                    except (ValueError, TypeError):
                        print_warning(f"Invalid value for {env_var}: {env_value}")
                else:
                    setattr(self.config, config_attr, env_value)

    def _load_from_file(self, file_path: Optional[Path] = None):
        """Load configuration from file (TOML or JSON)"""
        if file_path:
            config_path = file_path
        else:
            # Look for config files in order of preference
            possible_paths = [
                Path.cwd() / "ops0.toml",
                Path.cwd() / "pyproject.toml",
                Path.home() / ".ops0" / "config.toml",
                Path.cwd() / "ops0.json",
                Path.home() / ".ops0" / "config.json",
            ]

            config_path = None
            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break

        if not config_path or not config_path.exists():
            return

        self.config_file_path = config_path

        try:
            if config_path.suffix == '.toml':
                self._load_toml(config_path)
            elif config_path.suffix == '.json':
                self._load_json(config_path)
        except Exception as e:
            print_warning(f"Failed to load config from {config_path}: {e}")

    def _load_toml(self, file_path: Path):
        """Load configuration from TOML file"""
        with open(file_path, 'rb') as f:
            data = tomllib.load(f)

        # Handle pyproject.toml with [tool.ops0] section
        if 'tool' in data and 'ops0' in data['tool']:
            ops0_config = data['tool']['ops0']
        elif 'ops0' in data:
            ops0_config = data['ops0']
        else:
            ops0_config = data

        self._apply_config_dict(ops0_config)

    def _load_json(self, file_path: Path):
        """Load configuration from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)

        self._apply_config_dict(data)

    def _apply_config_dict(self, config_dict: Dict[str, Any]):
        """Apply configuration dictionary to config object"""
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                # Handle nested configurations
                if key == 'alert_channels' and isinstance(value, list):
                    self.config.alert_channels = value
                else:
                    try:
                        # Get the expected type from the current value
                        current_value = getattr(self.config, key)
                        if current_value is not None:
                            expected_type = type(current_value)
                            if expected_type == bool and isinstance(value, str):
                                value = value.lower() in ('true', '1', 'yes', 'on')
                            elif expected_type in (int, float) and isinstance(value, str):
                                value = expected_type(value)

                        setattr(self.config, key, value)
                    except (ValueError, TypeError) as e:
                        print_warning(f"Invalid config value for {key}: {value} ({e})")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        return getattr(self.config, key, default)

    def set(self, key: str, value: Any) -> bool:
        """Set configuration value by key"""
        if not hasattr(self.config, key):
            print_error(f"Unknown configuration key: {key}")
            return False

        try:
            # Type conversion
            current_value = getattr(self.config, key)
            if current_value is not None:
                expected_type = type(current_value)
                if expected_type == bool and isinstance(value, str):
                    value = value.lower() in ('true', '1', 'yes', 'on')
                elif expected_type in (int, float) and isinstance(value, str):
                    value = expected_type(value)

            setattr(self.config, key, value)
            return True
        except (ValueError, TypeError) as e:
            print_error(f"Invalid value for {key}: {value} ({e})")
            return False

    def save(self, file_path: Optional[Path] = None) -> bool:
        """Save current configuration to file"""
        if not file_path:
            # Use existing config file or create default
            if self.config_file_path:
                file_path = self.config_file_path
            else:
                file_path = Path.home() / ".ops0" / "config.toml"
                file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            config_dict = asdict(self.config)

            if file_path.suffix == '.json':
                with open(file_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            else:  # Default to TOML
                import tomli_w
                with open(file_path, 'wb') as f:
                    tomli_w.dump({'ops0': config_dict}, f)

            print_success(f"Configuration saved to {file_path}")
            return True

        except Exception as e:
            print_error(f"Failed to save configuration: {e}")
            return False

    def reset(self):
        """Reset configuration to defaults"""
        self.config = Ops0Config()
        print_success("Configuration reset to defaults")

    def show(self):
        """Display current configuration in a nice format"""
        config_dict = asdict(self.config)

        # Group configuration by category
        categories = {
            "Core Settings": ["project_name", "environment", "log_level"],
            "Storage": ["storage_backend", "storage_path"],
            "Containers": ["container_registry", "build_containers", "push_containers"],
            "Deployment": ["default_env", "auto_deploy", "deployment_timeout"],
            "Monitoring": ["enable_monitoring", "metrics_retention_days", "alert_channels"],
            "Development": ["dev_mode", "hot_reload", "debug_steps"],
            "Cloud": ["cloud_provider", "aws_region", "gcp_project", "azure_subscription"],
            "API": ["api_key", "api_url", "timeout"],
        }

        for category, keys in categories.items():
            table = Table(title=category, show_header=True, header_style="bold cyan")
            table.add_column("Setting", style="green")
            table.add_column("Value", style="white")
            table.add_column("Source", style="dim")

            for key in keys:
                if key in config_dict:
                    value = config_dict[key]
                    if key == "api_key" and value:
                        value = "*" * 8 + value[-4:] if len(value) > 4 else "****"

                    # Determine source
                    source = "default"
                    if os.getenv(f"OPS0_{key.upper()}"):
                        source = "environment"
                    elif self.config_file_path and self.config_file_path.exists():
                        source = f"file ({self.config_file_path.name})"

                    table.add_row(key, str(value), source)

            if table.row_count > 0:
                console.print(table)
                console.print()

        # Show configuration file info
        if self.config_file_path:
            panel = Panel(
                f"📁 Configuration file: [cyan]{self.config_file_path}[/cyan]\n"
                f"🔧 Use [bold]ops0 config --set key=value[/bold] to modify settings\n"
                f"💾 Use [bold]ops0 config save[/bold] to persist changes",
                title="Configuration Info",
                border_style="blue"
            )
            console.print(panel)

    def validate(self) -> bool:
        """Validate current configuration"""
        errors = []
        warnings = []

        # Validate API key format
        if self.config.api_key and len(self.config.api_key) < 10:
            warnings.append("API key seems too short")

        # Validate timeout values
        if self.config.timeout <= 0:
            errors.append("Timeout must be positive")

        if self.config.deployment_timeout <= 0:
            errors.append("Deployment timeout must be positive")

        # Validate storage path
        if self.config.storage_backend == "local":
            storage_path = Path(self.config.storage_path)
            if not storage_path.parent.exists():
                warnings.append(f"Storage parent directory does not exist: {storage_path.parent}")

        # Validate cloud settings
        if self.config.cloud_provider:
            if self.config.cloud_provider == "aws" and not self.config.aws_region:
                warnings.append("AWS provider selected but no region specified")
            elif self.config.cloud_provider == "gcp" and not self.config.gcp_project:
                warnings.append("GCP provider selected but no project specified")
            elif self.config.cloud_provider == "azure" and not self.config.azure_subscription:
                warnings.append("Azure provider selected but no subscription specified")

        # Display validation results
        if errors:
            console.print("\n❌ Configuration Errors:")
            for error in errors:
                console.print(f"  • {error}")

        if warnings:
            console.print("\n⚠️  Configuration Warnings:")
            for warning in warnings:
                console.print(f"  • {warning}")

        if not errors and not warnings:
            console.print("✅ Configuration is valid")

        return len(errors) == 0

    def load_from_file(self, file_path: Path):
        """Load configuration from specific file"""
        self._load_from_file(file_path)

    def get_effective_config(self) -> Dict[str, Any]:
        """Get the effective configuration as a dictionary"""
        return asdict(self.config)


# Global configuration instance
config = ConfigManager()