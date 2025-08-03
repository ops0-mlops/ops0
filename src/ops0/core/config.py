"""
ops0 Configuration Management

Centralized configuration for all ops0 components.
Supports environment variables, config files, and runtime overrides.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
import toml

from .exceptions import ConfigurationError, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    """Storage layer configuration"""
    backend_type: str = "local"
    storage_path: str = ".ops0/storage"
    enable_compression: bool = False
    enable_encryption: bool = False
    default_serializer: str = "auto"  # auto, pickle, json, parquet, numpy
    cache_size_mb: int = 100
    cleanup_temp_files: bool = True


@dataclass
class ExecutionConfig:
    """Pipeline execution configuration"""
    max_parallel_steps: int = 4
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    execution_timeout_seconds: int = 3600  # 1 hour
    enable_step_caching: bool = True
    continue_on_failure: bool = False
    enable_profiling: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - ops0.%(name)s - %(levelname)s - %(message)s"
    log_file_path: Optional[str] = None
    enable_file_logging: bool = False
    enable_structured_logging: bool = False
    max_log_file_size_mb: int = 10
    log_rotation_count: int = 5


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    enable_monitoring: bool = True
    metrics_backend: str = "local"  # local, prometheus, cloudwatch
    enable_step_timing: bool = True
    enable_resource_monitoring: bool = True
    alert_on_failure: bool = True
    alert_endpoints: List[str] = field(default_factory=list)
    retention_days: int = 7


@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_auth: bool = False
    api_key: Optional[str] = None
    enable_ssl: bool = True
    allowed_hosts: List[str] = field(default_factory=lambda: ["localhost", "127.0.0.1"])
    encrypt_storage: bool = False
    secret_key: Optional[str] = None


@dataclass
class DevelopmentConfig:
    """Development and debugging configuration"""
    debug_mode: bool = False
    enable_hot_reload: bool = False
    validate_on_import: bool = True
    enable_pipeline_visualization: bool = True
    auto_save_graphs: bool = False
    development_storage_namespace: str = "dev"


@dataclass
class CloudConfig:
    """Cloud deployment configuration"""
    provider: str = "local"  # local, aws, gcp, azure
    region: str = "us-west-2"
    enable_auto_scaling: bool = True
    min_workers: int = 1
    max_workers: int = 10
    deployment_timeout_minutes: int = 30
    enable_spot_instances: bool = False


@dataclass
class Ops0Config:
    """Main ops0 configuration"""
    project_name: str = "ops0-project"
    environment: str = "development"  # development, staging, production
    config_version: str = "1.0"

    # Component configurations
    storage: StorageConfig = field(default_factory=StorageConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)

    # Custom user settings
    custom: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """Manages ops0 configuration loading and validation"""

    def __init__(self):
        self._config: Optional[Ops0Config] = None
        self.config_file_path: Optional[Path] = None
        self._environment_prefix = "OPS0_"

    @property
    def config(self) -> Ops0Config:
        """Get current configuration, loading if necessary"""
        if self._config is None:
            self.load_config()
        return self._config

    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> Ops0Config:
        """
        Load configuration from multiple sources in priority order:
        1. Explicit config file path
        2. Environment variables
        3. ops0.toml in current directory
        4. pyproject.toml [tool.ops0] section
        5. Default configuration
        """
        logger.debug("Loading ops0 configuration...")

        # Start with defaults
        self._config = Ops0Config()

        # Load from config file
        if config_path:
            self._load_from_file(Path(config_path))
        else:
            self._discover_and_load_config_file()

        # Override with environment variables
        self._load_from_environment()

        # Apply environment-specific defaults
        self._apply_environment_defaults()

        # Validate final configuration
        self.validate()

        logger.info(f"Configuration loaded for environment: {self._config.environment}")
        return self._config

    def _discover_and_load_config_file(self) -> None:
        """Discover and load config file from standard locations"""
        search_paths = [
            Path.cwd() / "ops0.toml",
            Path.cwd() / "pyproject.toml",
            Path.cwd() / ".ops0" / "config.toml",
            Path.home() / ".config" / "ops0" / "config.toml",
        ]

        for config_path in search_paths:
            if config_path.exists():
                logger.debug(f"Found config file: {config_path}")
                self._load_from_file(config_path)
                self.config_file_path = config_path
                return

        logger.debug("No config file found, using defaults")

    def _load_from_file(self, file_path: Path) -> None:
        """Load configuration from file"""
        try:
            if not file_path.exists():
                raise ConfigurationError(f"Config file not found: {file_path}")

            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
            elif file_path.suffix.lower() in ['.toml', '.tml']:
                with open(file_path, 'r') as f:
                    data = toml.load(f)

                # Handle pyproject.toml format
                if file_path.name == "pyproject.toml":
                    data = data.get("tool", {}).get("ops0", {})
            else:
                raise ConfigurationError(f"Unsupported config file format: {file_path.suffix}")

            # Merge with current config
            self._merge_config_data(data)
            logger.debug(f"Loaded configuration from {file_path}")

        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to load config from {file_path}: {str(e)}")

    def _merge_config_data(self, data: Dict[str, Any]) -> None:
        """Merge configuration data into current config"""
        if not self._config:
            self._config = Ops0Config()

        # Update top-level fields
        for field_name, value in data.items():
            if hasattr(self._config, field_name):
                current_value = getattr(self._config, field_name)

                if isinstance(current_value, (StorageConfig, ExecutionConfig, LoggingConfig,
                                            MonitoringConfig, SecurityConfig, DevelopmentConfig, CloudConfig)):
                    # Update nested config objects
                    if isinstance(value, dict):
                        for nested_field, nested_value in value.items():
                            if hasattr(current_value, nested_field):
                                setattr(current_value, nested_field, nested_value)
                else:
                    # Update simple fields
                    setattr(self._config, field_name, value)

    def _load_from_environment(self) -> None:
        """Load configuration from environment variables"""
        # Project-level env vars
        if os.getenv("OPS0_PROJECT_NAME"):
            self._config.project_name = os.getenv("OPS0_PROJECT_NAME")

        if os.getenv("OPS0_ENVIRONMENT"):
            self._config.environment = os.getenv("OPS0_ENVIRONMENT")

        # Storage config
        if os.getenv("OPS0_STORAGE_BACKEND"):
            self._config.storage.backend_type = os.getenv("OPS0_STORAGE_BACKEND")

        if os.getenv("OPS0_STORAGE_PATH"):
            self._config.storage.storage_path = os.getenv("OPS0_STORAGE_PATH")

        if os.getenv("OPS0_ENABLE_COMPRESSION"):
            self._config.storage.enable_compression = os.getenv("OPS0_ENABLE_COMPRESSION").lower() == "true"

        # Execution config
        if os.getenv("OPS0_MAX_RETRIES"):
            self._config.execution.max_retries = int(os.getenv("OPS0_MAX_RETRIES"))

        if os.getenv("OPS0_EXECUTION_TIMEOUT"):
            self._config.execution.execution_timeout_seconds = int(os.getenv("OPS0_EXECUTION_TIMEOUT"))

        if os.getenv("OPS0_MAX_PARALLEL_STEPS"):
            self._config.execution.max_parallel_steps = int(os.getenv("OPS0_MAX_PARALLEL_STEPS"))

        if os.getenv("OPS0_ENABLE_CACHING"):
            self._config.execution.enable_step_caching = os.getenv("OPS0_ENABLE_CACHING").lower() == "true"

        # Logging config
        if os.getenv("OPS0_LOG_LEVEL"):
            self._config.logging.log_level = os.getenv("OPS0_LOG_LEVEL").upper()

        if os.getenv("OPS0_LOG_FILE"):
            self._config.logging.log_file_path = os.getenv("OPS0_LOG_FILE")

        if os.getenv("OPS0_ENABLE_FILE_LOGGING"):
            self._config.logging.enable_file_logging = os.getenv("OPS0_ENABLE_FILE_LOGGING").lower() == "true"

        # Development config
        if os.getenv("OPS0_DEBUG"):
            self._config.development.debug_mode = os.getenv("OPS0_DEBUG").lower() == "true"

        # Monitoring config
        if os.getenv("OPS0_ENABLE_MONITORING"):
            self._config.monitoring.enable_monitoring = os.getenv("OPS0_ENABLE_MONITORING").lower() == "true"

        # Security config
        if os.getenv("OPS0_API_KEY"):
            self._config.security.api_key = os.getenv("OPS0_API_KEY")

        # Cloud config
        if os.getenv("OPS0_CLOUD_PROVIDER"):
            self._config.cloud.provider = os.getenv("OPS0_CLOUD_PROVIDER")

        if os.getenv("OPS0_CLOUD_REGION"):
            self._config.cloud.region = os.getenv("OPS0_CLOUD_REGION")

    def _apply_environment_defaults(self) -> None:
        """Apply environment-specific default configurations"""
        env = self._config.environment.lower()

        if env == "development":
            # Development defaults
            self._config.development.debug_mode = True
            self._config.logging.log_level = "DEBUG"
            self._config.execution.enable_step_caching = True
            self._config.monitoring.enable_monitoring = True
            self._config.security.enable_auth = False

        elif env == "staging":
            # Staging defaults
            self._config.development.debug_mode = False
            self._config.logging.log_level = "INFO"
            self._config.logging.enable_file_logging = True
            self._config.monitoring.enable_monitoring = True
            self._config.security.enable_auth = True

        elif env == "production":
            # Production defaults
            self._config.development.debug_mode = False
            self._config.logging.log_level = "WARNING"
            self._config.logging.enable_file_logging = True
            self._config.logging.enable_structured_logging = True
            self._config.monitoring.enable_monitoring = True
            self._config.monitoring.alert_on_failure = True
            self._config.security.enable_auth = True
            self._config.security.enable_ssl = True
            self._config.storage.enable_compression = True

    def validate(self) -> bool:
        """Validate current configuration"""
        try:
            # Validate storage path exists or can be created
            storage_path = Path(self._config.storage.storage_path)
            if not storage_path.exists():
                try:
                    storage_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise ValidationError(
                        f"Cannot create storage path: {storage_path}",
                        field="storage.storage_path",
                        value=str(storage_path),
                        context={"error": str(e)}
                    )

            # Validate log level
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if self._config.logging.log_level not in valid_levels:
                raise ValidationError(
                    f"Invalid log level: {self._config.logging.log_level}",
                    field="logging.log_level",
                    value=self._config.logging.log_level,
                    context={"valid_levels": valid_levels}
                )

            # Validate execution settings
            if self._config.execution.max_parallel_steps < 1:
                raise ValidationError(
                    "max_parallel_steps must be at least 1",
                    field="execution.max_parallel_steps",
                    value=self._config.execution.max_parallel_steps
                )

            if self._config.execution.max_retries < 0:
                raise ValidationError(
                    "max_retries cannot be negative",
                    field="execution.max_retries",
                    value=self._config.execution.max_retries
                )

            # Validate cloud configuration
            valid_providers = ["local", "aws", "gcp", "azure"]
            if self._config.cloud.provider not in valid_providers:
                raise ValidationError(
                    f"Invalid cloud provider: {self._config.cloud.provider}",
                    field="cloud.provider",
                    value=self._config.cloud.provider,
                    context={"valid_providers": valid_providers}
                )

            # Validate monitoring settings
            if self._config.monitoring.retention_days < 1:
                raise ValidationError(
                    "retention_days must be at least 1",
                    field="monitoring.retention_days",
                    value=self._config.monitoring.retention_days
                )

            return True

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ConfigurationError(f"Configuration validation failed: {str(e)}")

    def save_config(self, path: Optional[Union[str, Path]] = None, format: str = "toml") -> None:
        """
        Save current configuration to file.

        Args:
            path: Path to save configuration
            format: File format (toml or json)
        """
        if not path:
            path = Path.cwd() / f"ops0.{format}"
        else:
            path = Path(path)

        try:
            config_dict = asdict(self._config)

            if format.lower() == "json":
                with open(path, 'w') as f:
                    json.dump(config_dict, f, indent=2, default=str)
            elif format.lower() in ["toml", "tml"]:
                with open(path, 'w') as f:
                    toml.dump(config_dict, f)
            else:
                raise ConfigurationError(f"Unsupported format: {format}")

            logger.info(f"Configuration saved to {path}")

        except Exception as e:
            raise ConfigurationError(f"Failed to save config to {path}: {str(e)}")

    def get_section(self, section_name: str) -> Any:
        """Get a specific configuration section"""
        if not hasattr(self._config, section_name):
            raise ConfigurationError(f"Unknown configuration section: {section_name}")
        return getattr(self._config, section_name)

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        self._merge_config_data(updates)
        self.validate()

    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults"""
        self._config = Ops0Config()
        logger.info("Configuration reset to defaults")

    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration"""
        return {
            "project_name": self._config.project_name,
            "environment": self._config.environment,
            "config_file": str(self.config_file_path) if self.config_file_path else None,
            "storage_backend": self._config.storage.backend_type,
            "storage_path": self._config.storage.storage_path,
            "log_level": self._config.logging.log_level,
            "debug_mode": self._config.development.debug_mode,
            "max_parallel_steps": self._config.execution.max_parallel_steps,
            "cloud_provider": self._config.cloud.provider,
        }


# Global configuration manager instance
config_manager = ConfigManager()

# Convenient access to configuration
config = config_manager.config


# Utility functions
def reload_config(config_path: Optional[Union[str, Path]] = None) -> Ops0Config:
    """Reload configuration from sources"""
    return config_manager.load_config(config_path)


def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value using dot notation.

    Args:
        key_path: Dot-separated path (e.g., "storage.backend_type")
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    try:
        keys = key_path.split('.')
        value = config

        for key in keys:
            value = getattr(value, key)

        return value
    except (AttributeError, KeyError):
        return default


def set_config_value(key_path: str, value: Any) -> None:
    """
    Set a configuration value using dot notation.

    Args:
        key_path: Dot-separated path (e.g., "storage.backend_type")
        value: Value to set
    """
    keys = key_path.split('.')
    target = config

    # Navigate to parent object
    for key in keys[:-1]:
        target = getattr(target, key)

    # Set the final value
    setattr(target, keys[-1], value)


def configure_logging() -> None:
    """Configure Python logging based on ops0 configuration"""
    log_config = config.logging

    # Set level
    log_level = getattr(logging, log_config.log_level)
    logging.getLogger("ops0").setLevel(log_level)

    # Clear existing handlers
    logger = logging.getLogger("ops0")
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(log_config.log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_config.enable_file_logging and log_config.log_file_path:
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            log_config.log_file_path,
            maxBytes=log_config.max_log_file_size_mb * 1024 * 1024,
            backupCount=log_config.log_rotation_count
        )
        file_formatter = logging.Formatter(log_config.log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)


# Initialize logging on import
configure_logging()