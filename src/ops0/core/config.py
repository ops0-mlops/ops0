"""
ops0 Configuration Management - Centralized configuration for the framework.

Handles configuration loading from files, environment variables, and defaults.
"""

import os
import json
import toml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
import logging

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Configuration for pipeline execution"""
    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    retry_backoff_multiplier: float = 2.0

    # Timeout settings
    execution_timeout_seconds: int = 3600
    step_timeout_seconds: int = 1800

    # Parallelism
    max_parallel_steps: int = 4
    thread_pool_size: int = 8

    # Caching
    enable_step_caching: bool = True
    cache_ttl_seconds: int = 86400  # 24 hours

    def validate(self) -> List[str]:
        """Validate execution configuration"""
        issues = []

        if self.max_retries < 0:
            issues.append("max_retries must be non-negative")

        if self.retry_delay_seconds <= 0:
            issues.append("retry_delay_seconds must be positive")

        if self.execution_timeout_seconds <= 0:
            issues.append("execution_timeout_seconds must be positive")

        if self.max_parallel_steps <= 0:
            issues.append("max_parallel_steps must be positive")

        return issues


@dataclass
class StorageConfig:
    """Configuration for data storage"""
    # Backend settings
    backend_type: str = "local"  # local, s3, gcs, azure
    storage_path: str = ".ops0/storage"

    # Serialization
    default_format: str = "auto"  # auto, pickle, parquet, json
    enable_compression: bool = True
    compression_level: int = 6

    # Caching
    enable_cache: bool = True
    cache_size_mb: int = 512

    # Cleanup
    auto_cleanup: bool = True
    retention_days: int = 30

    def validate(self) -> List[str]:
        """Validate storage configuration"""
        issues = []

        valid_backends = ["local", "s3", "gcs", "azure"]
        if self.backend_type not in valid_backends:
            issues.append(f"backend_type must be one of: {valid_backends}")

        valid_formats = ["auto", "pickle", "parquet", "json"]
        if self.default_format not in valid_formats:
            issues.append(f"default_format must be one of: {valid_formats}")

        if self.compression_level < 0 or self.compression_level > 9:
            issues.append("compression_level must be between 0 and 9")

        if self.cache_size_mb <= 0:
            issues.append("cache_size_mb must be positive")

        return issues


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    # Basic settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - ops0.%(name)s - %(levelname)s - %(message)s"

    # File logging
    enable_file_logging: bool = True
    log_file_path: str = ".ops0/logs/ops0.log"
    max_file_size_mb: int = 10
    backup_count: int = 5

    # Structured logging
    enable_json_logging: bool = False
    include_context: bool = True

    def validate(self) -> List[str]:
        """Validate logging configuration"""
        issues = []

        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_levels:
            issues.append(f"log_level must be one of: {valid_levels}")

        if self.max_file_size_mb <= 0:
            issues.append("max_file_size_mb must be positive")

        if self.backup_count < 0:
            issues.append("backup_count must be non-negative")

        return issues


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and observability"""
    # Basic monitoring
    enable_monitoring: bool = True
    enable_metrics_collection: bool = True
    enable_performance_tracking: bool = True

    # Metrics
    metrics_collection_interval_seconds: int = 30
    metrics_retention_days: int = 30
    enable_system_metrics: bool = True
    enable_custom_metrics: bool = True

    # Health checks
    enable_health_checks: bool = True
    health_check_interval_seconds: int = 60
    health_check_timeout_seconds: int = 10

    # Alerting
    enable_alerting: bool = False
    alert_channels: List[str] = field(default_factory=list)
    alert_thresholds: Dict[str, float] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Validate monitoring configuration"""
        issues = []

        if self.metrics_collection_interval_seconds <= 0:
            issues.append("metrics_collection_interval_seconds must be positive")

        if self.metrics_retention_days <= 0:
            issues.append("metrics_retention_days must be positive")

        if self.health_check_interval_seconds <= 0:
            issues.append("health_check_interval_seconds must be positive")

        if self.health_check_timeout_seconds <= 0:
            issues.append("health_check_timeout_seconds must be positive")

        return issues


@dataclass
class ValidationConfig:
    """Configuration for pipeline validation"""
    # Validation settings
    enable_validation: bool = True
    validation_level: str = "standard"  # basic, standard, strict
    fail_on_validation_errors: bool = True
    show_validation_warnings: bool = True

    # Type checking
    enable_type_checking: bool = True
    strict_type_checking: bool = False

    # Performance validation
    enable_performance_validation: bool = True
    max_step_execution_time_seconds: int = 3600
    max_memory_usage_mb: int = 8192

    # Custom rules
    enable_custom_rules: bool = True
    custom_rules_path: Optional[str] = None

    def validate(self) -> List[str]:
        """Validate validation configuration"""
        issues = []

        valid_levels = ["basic", "standard", "strict"]
        if self.validation_level not in valid_levels:
            issues.append(f"validation_level must be one of: {valid_levels}")

        if self.max_step_execution_time_seconds <= 0:
            issues.append("max_step_execution_time_seconds must be positive")

        if self.max_memory_usage_mb <= 0:
            issues.append("max_memory_usage_mb must be positive")

        return issues


@dataclass
class DevelopmentConfig:
    """Configuration for development mode"""
    # Debug settings
    debug_mode: bool = False
    verbose_output: bool = False
    enable_profiling: bool = False

    # Auto-reload
    enable_auto_reload: bool = True
    watch_patterns: List[str] = field(default_factory=lambda: ["*.py"])

    # Testing
    enable_test_mode: bool = False
    mock_external_services: bool = False

    def validate(self) -> List[str]:
        """Validate development configuration"""
        # Development config is generally permissive
        return []


@dataclass
class Ops0Config:
    """Main ops0 configuration container"""
    # Project settings
    project_name: str = "ops0-project"
    environment: str = "development"  # development, staging, production
    version: str = "1.0.0"

    # Component configurations
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)

    def validate(self) -> List[str]:
        """Validate complete configuration"""
        all_issues = []

        # Validate each component
        all_issues.extend(self.execution.validate())
        all_issues.extend(self.storage.validate())
        all_issues.extend(self.logging.validate())
        all_issues.extend(self.monitoring.validate())
        all_issues.extend(self.validation.validate())
        all_issues.extend(self.development.validate())

        # Global validations
        valid_environments = ["development", "staging", "production"]
        if self.environment not in valid_environments:
            all_issues.append(f"environment must be one of: {valid_environments}")

        return all_issues

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "project_name": self.project_name,
            "environment": self.environment,
            "version": self.version,
            "execution": self.execution.__dict__,
            "storage": self.storage.__dict__,
            "logging": self.logging.__dict__,
            "monitoring": self.monitoring.__dict__,
            "validation": self.validation.__dict__,
            "development": self.development.__dict__,
        }


class ConfigManager:
    """Manages ops0 configuration loading and validation"""

    def __init__(self):
        self._config: Optional[Ops0Config] = None
        self.config_file_path: Optional[Path] = None

    @property
    def config(self) -> Ops0Config:
        """Get the current configuration, loading if necessary"""
        if self._config is None:
            self.load_config()
        return self._config

    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> Ops0Config:
        """
        Load configuration from file and environment.

        Args:
            config_path: Optional path to configuration file

        Returns:
            Loaded configuration
        """
        # Start with defaults
        self._config = Ops0Config()

        # Load from file if specified or found
        if config_path:
            self.config_file_path = Path(config_path)
        else:
            self.config_file_path = self._find_config_file()

        if self.config_file_path and self.config_file_path.exists():
            self._load_from_file()
            logger.debug(f"Loaded configuration from {self.config_file_path}")

        # Override with environment variables
        self._load_from_environment()

        # Validate configuration
        issues = self._config.validate()
        if issues:
            raise ConfigurationError(
                f"Configuration validation failed: {'; '.join(issues)}",
                config_file=str(self.config_file_path) if self.config_file_path else None
            )

        return self._config

    def _find_config_file(self) -> Optional[Path]:
        """Find configuration file in standard locations"""
        search_paths = [
            Path.cwd() / "ops0.toml",
            Path.cwd() / "pyproject.toml",
            Path.cwd() / "ops0.json",
            Path.cwd() / ".ops0" / "config.toml",
            Path.cwd() / ".ops0" / "config.json",
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None

    def _load_from_file(self):
        """Load configuration from file"""
        if not self.config_file_path:
            return

        try:
            content = self.config_file_path.read_text()

            if self.config_file_path.suffix == ".toml":
                data = toml.loads(content)
            elif self.config_file_path.suffix == ".json":
                data = json.loads(content)
            else:
                logger.warning(f"Unsupported config file format: {self.config_file_path.suffix}")
                return

            # Extract ops0 section for pyproject.toml
            if "tool" in data and "ops0" in data["tool"]:
                data = data["tool"]["ops0"]
            elif "ops0" in data:
                data = data["ops0"]

            # Update configuration with file data
            self._update_config_from_dict(data)

        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration from {self.config_file_path}: {e}",
                config_file=str(self.config_file_path)
            )

    def _update_config_from_dict(self, data: Dict[str, Any]):
        """Update configuration from dictionary data"""
        # Update top-level settings
        if "project_name" in data:
            self._config.project_name = data["project_name"]
        if "environment" in data:
            self._config.environment = data["environment"]
        if "version" in data:
            self._config.version = data["version"]

        # Update component configurations
        for section_name in ["execution", "storage", "logging", "monitoring", "validation", "development"]:
            if section_name in data:
                section_data = data[section_name]
                config_obj = getattr(self._config, section_name)

                for key, value in section_data.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)

    def _load_from_environment(self):
        """Load configuration from environment variables"""
        # Global settings
        if os.getenv("OPS0_PROJECT_NAME"):
            self._config.project_name = os.getenv("OPS0_PROJECT_NAME")

        if os.getenv("OPS0_ENV"):
            self._config.environment = os.getenv("OPS0_ENV")

        if os.getenv("OPS0_VERSION"):
            self._config.version = os.getenv("OPS0_VERSION")

        # Execution config
        if os.getenv("OPS0_MAX_RETRIES"):
            self._config.execution.max_retries = int(os.getenv("OPS0_MAX_RETRIES"))

        if os.getenv("OPS0_EXECUTION_TIMEOUT"):
            self._config.execution.execution_timeout_seconds = int(os.getenv("OPS0_EXECUTION_TIMEOUT"))

        if os.getenv("OPS0_MAX_PARALLEL_STEPS"):
            self._config.execution.max_parallel_steps = int(os.getenv("OPS0_MAX_PARALLEL_STEPS"))

        if os.getenv("OPS0_ENABLE_CACHING"):
            self._config.execution.enable_step_caching = os.getenv("OPS0_ENABLE_CACHING").lower() == "true"

        # Storage config
        if os.getenv("OPS0_STORAGE_BACKEND"):
            self._config.storage.backend_type = os.getenv("OPS0_STORAGE_BACKEND")

        if os.getenv("OPS0_STORAGE_PATH"):
            self._config.storage.storage_path = os.getenv("OPS0_STORAGE_PATH")

        if os.getenv("OPS0_ENABLE_COMPRESSION"):
            self._config.storage.enable_compression = os.getenv("OPS0_ENABLE_COMPRESSION").lower() == "true"

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

        if os.getenv("OPS0_ENABLE_MONITORING"):
            self._config.monitoring.enable_monitoring = os.getenv("OPS0_ENABLE_MONITORING").lower() == "true"

    def save_config(self, path: Optional[Union[str, Path]] = None, format: str = "toml"):
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

        config_dict = {"ops0": self._config.to_dict()}

        try:
            if format == "toml":
                content = toml.dumps(config_dict)
            elif format == "json":
                content = json.dumps(config_dict, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")

            path.write_text(content)
            logger.info(f"Configuration saved to {path}")

        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration to {path}: {e}")


# Global configuration manager instance
_config_manager = ConfigManager()


def get_config() -> Ops0Config:
    """Get the global ops0 configuration"""
    return _config_manager.config


def load_config(config_path: Optional[Union[str, Path]] = None) -> Ops0Config:
    """Load configuration from file"""
    return _config_manager.load_config(config_path)


def reset_config():
    """Reset configuration to defaults"""
    _config_manager._config = None
    _config_manager.config_file_path = None