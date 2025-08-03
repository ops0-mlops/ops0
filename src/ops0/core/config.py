"""
ops0 Core Configuration

Centralized configuration management for ops0 core components.
Handles settings, defaults, environment variables, and configuration validation.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, List, Type
from dataclasses import dataclass, field
from pathlib import Path
import json

from .exceptions import ConfigurationError


@dataclass
class ExecutionConfig:
    """Configuration for pipeline execution"""
    # Execution settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    retry_backoff_multiplier: float = 2.0
    execution_timeout_seconds: int = 3600

    # Parallelization
    max_parallel_steps: int = 4
    enable_parallel_execution: bool = True

    # Caching
    enable_step_caching: bool = True
    cache_ttl_seconds: int = 86400  # 24 hours

    # Resource management
    default_cpu_limit: float = 1.0
    default_memory_limit_mb: int = 1024
    enable_resource_monitoring: bool = True

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

        if self.default_cpu_limit <= 0:
            issues.append("default_cpu_limit must be positive")

        if self.default_memory_limit_mb <= 0:
            issues.append("default_memory_limit_mb must be positive")

        return issues


@dataclass
class StorageConfig:
    """Configuration for storage layer"""
    # Storage backend
    backend_type: str = "local"  # local, s3, gcs, azure
    storage_path: str = ".ops0/storage"

    # Serialization
    default_serializer: str = "pickle"
    enable_compression: bool = True
    compression_level: int = 6

    # Caching
    enable_memory_cache: bool = True
    memory_cache_size_mb: int = 512
    enable_disk_cache: bool = True

    # Cleanup
    auto_cleanup_enabled: bool = True
    cleanup_after_days: int = 30
    max_storage_size_gb: int = 10

    # Security
    encrypt_at_rest: bool = False
    encryption_key: Optional[str] = None

    def validate(self) -> List[str]:
        """Validate storage configuration"""
        issues = []

        valid_backends = ["local", "s3", "gcs", "azure", "memory"]
        if self.backend_type not in valid_backends:
            issues.append(f"backend_type must be one of: {valid_backends}")

        if not self.storage_path:
            issues.append("storage_path cannot be empty")

        if self.compression_level < 0 or self.compression_level > 9:
            issues.append("compression_level must be between 0 and 9")

        if self.memory_cache_size_mb <= 0:
            issues.append("memory_cache_size_mb must be positive")

        if self.cleanup_after_days <= 0:
            issues.append("cleanup_after_days must be positive")

        if self.max_storage_size_gb <= 0:
            issues.append("max_storage_size_gb must be positive")

        if self.encrypt_at_rest and not self.encryption_key:
            issues.append("encryption_key required when encrypt_at_rest is enabled")

        return issues


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    # Log levels
    log_level: str = "INFO"
    core_log_level: str = "INFO"
    execution_log_level: str = "INFO"
    storage_log_level: str = "INFO"

    # Output configuration
    enable_console_logging: bool = True
    enable_file_logging: bool = True
    log_file_path: str = ".ops0/logs/ops0.log"
    max_log_file_size_mb: int = 100
    max_log_files: int = 5

    # Format
    log_format: str = "%(asctime)s - ops0.%(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

    # Advanced
    enable_structured_logging: bool = False
    log_json_format: bool = False
    include_caller_info: bool = False

    def validate(self) -> List[str]:
        """Validate logging configuration"""
        issues = []

        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for field_name in ["log_level", "core_log_level", "execution_log_level", "storage_log_level"]:
            level = getattr(self, field_name)
            if level not in valid_levels:
                issues.append(f"{field_name} must be one of: {valid_levels}")

        if self.max_log_file_size_mb <= 0:
            issues.append("max_log_file_size_mb must be positive")

        if self.max_log_files <= 0:
            issues.append("max_log_files must be positive")

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
    """Configuration for development features"""
    # Debug settings
    debug_mode: bool = False
    verbose_output: bool = False
    enable_step_profiling: bool = False

    # Hot reload
    enable_hot_reload: bool = False
    watch_file_changes: bool = False

    # Testing
    enable_test_mode: bool = False
    mock_external_dependencies: bool = False

    # Development tools
    enable_jupyter_integration: bool = True
    enable_notebook_execution: bool = True

    def validate(self) -> List[str]:
        """Validate development configuration"""
        # Development config is generally permissive
        return []


@dataclass
class Ops0Config:
    """Main ops0 configuration container"""
    # Core configurations
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)

    # Global settings
    project_name: str = "ops0-project"
    environment: str = "development"
    version: str = "1.0.0"

    # Feature flags
    enable_experimental_features: bool = False
    feature_flags: Dict[str, bool] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Validate entire configuration"""
        all_issues = []

        # Validate each sub-configuration
        all_issues.extend(self.execution.validate())
        all_issues.extend(self.storage.validate())
        all_issues.extend(self.logging.validate())
        all_issues.extend(self.monitoring.validate())
        all_issues.extend(self.validation.validate())
        all_issues.extend(self.development.validate())

        # Global validation
        if not self.project_name:
            all_issues.append("project_name cannot be empty")

        valid_environments = ["development", "testing", "staging", "production"]
        if self.environment not in valid_environments:
            all_issues.append(f"environment must be one of: {valid_environments}")

        return all_issues

    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        return len(self.validate()) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""

        def _dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {
                    field_name: _dataclass_to_dict(getattr(obj, field_name))
                    for field_name in obj.__dataclass_fields__
                }
            elif isinstance(obj, (list, tuple)):
                return [_dataclass_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: _dataclass_to_dict(value) for key, value in obj.items()}
            else:
                return obj

        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Ops0Config':
        """Create configuration from dictionary"""

        def _dict_to_dataclass(data_dict, target_class):
            if not hasattr(target_class, '__dataclass_fields__'):
                return data_dict

            kwargs = {}
            for field_name, field_info in target_class.__dataclass_fields__.items():
                if field_name in data_dict:
                    field_value = data_dict[field_name]
                    field_type = field_info.type

                    if hasattr(field_type, '__dataclass_fields__'):
                        kwargs[field_name] = _dict_to_dataclass(field_value, field_type)
                    else:
                        kwargs[field_name] = field_value

            return target_class(**kwargs)

        return _dict_to_dataclass(data, cls)


class ConfigManager:
    """Manages ops0 configuration with environment variable support"""

    def __init__(self):
        self._config = Ops0Config()
        self._loaded_from_file = False
        self._config_file_path: Optional[Path] = None

        # Load configuration in order of priority
        self._load_from_environment()
        self._load_from_file()

    @property
    def config(self) -> Ops0Config:
        """Get current configuration"""
        return self._config

    def _load_from_environment(self):
        """Load configuration from environment variables"""
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

        # Global settings
        if os.getenv("OPS0_PROJECT_NAME"):
            self._config.project_name = os.getenv("OPS0_PROJECT_NAME")

        if os.getenv("OPS0_ENV"):
            self._config.environment = os.getenv("OPS0_ENV")

        if os.getenv("OPS0_DEBUG"):
            self._config.development.debug_mode = os.getenv("OPS0_DEBUG").lower() == "true"

        if os.getenv("OPS0_ENABLE_MONITORING"):
            self._config.monitoring.enable_monitoring = os.getenv("OPS0_ENABLE_MONITORING").lower() == "true"

    def _load_from_file(self):
        """Load configuration from file"""
        # Look for configuration files in order of preference
        possible_files = [
            Path.cwd() / "ops0.json",
            Path.cwd() / ".ops0" / "config.json",
            Path.home() / ".ops0" / "config.json",
            Path("/etc/ops0/config.json"),
        ]

        for config_file in possible_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)

                    # Merge with current config
                    loaded_config = Ops0Config.from_dict(config_data)
                    self._merge_configs(loaded_config)

                    self._loaded_from_file = True
                    self._config_file_path = config_file
                    break

                except Exception as e:
                    logging.warning(f"Failed to load config from {config_file}: {e}")

    def _merge_configs(self, other_config: Ops0Config):
        """Merge another configuration into current config"""
        # This is a simplified merge - in production would be more sophisticated
        other_dict = other_config.to_dict()
        current_dict = self._config.to_dict()

        def _merge_dicts(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    _merge_dicts(target[key], value)
                else:
                    target[key] = value

        _merge_dicts(current_dict, other_dict)
        self._config = Ops0Config.from_dict(current_dict)

    def save_config(self, file_path: Optional[Path] = None):
        """Save current configuration to file"""
        if file_path is None:
            file_path = self._config_file_path or (Path.cwd() / ".ops0" / "config.json")

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(file_path, 'w') as f:
                json.dump(self._config.to_dict(), f, indent=2)

            self._config_file_path = file_path

        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")

    def validate_config(self) -> List[str]:
        """Validate current configuration"""
        return self._config.validate()

    def get_setting(self, path: str, default: Any = None) -> Any:
        """Get a configuration setting by dot-separated path"""
        parts = path.split('.')
        current = self._config

        try:
            for part in parts:
                current = getattr(current, part)
            return current
        except AttributeError:
            return default

    def set_setting(self, path: str, value: Any):
        """Set a configuration setting by dot-separated path"""
        parts = path.split('.')
        current = self._config

        try:
            for part in parts[:-1]:
                current = getattr(current, part)
            setattr(current, parts[-1], value)
        except AttributeError:
            raise ConfigurationError(f"Invalid configuration path: {path}")

    def reload_config(self):
        """Reload configuration from all sources"""
        self._config = Ops0Config()
        self._load_from_environment()
        self._load_from_file()

    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration"""
        return {
            "project_name": self._config.project_name,
            "environment": self._config.environment,
            "loaded_from_file": self._loaded_from_file,
            "config_file_path": str(self._config_file_path) if self._config_file_path else None,
            "validation_errors": self.validate_config(),
            "feature_flags": self._config.feature_flags,
            "debug_mode": self._config.development.debug_mode,
        }


# Global configuration manager instance
config_manager = ConfigManager()


# Convenience accessors
def get_config() -> Ops0Config:
    """Get the global ops0 configuration"""
    return config_manager.config


def get_execution_config() -> ExecutionConfig:
    """Get execution configuration"""
    return config_manager.config.execution


def get_storage_config() -> StorageConfig:
    """Get storage configuration"""
    return config_manager.config.storage


def get_logging_config() -> LoggingConfig:
    """Get logging configuration"""
    return config_manager.config.logging


def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration"""
    return config_manager.config.monitoring


def get_validation_config() -> ValidationConfig:
    """Get validation configuration"""
    return config_manager.config.validation


def get_development_config() -> DevelopmentConfig:
    """Get development configuration"""
    return config_manager.config.development


def is_debug_mode() -> bool:
    """Check if debug mode is enabled"""
    return get_development_config().debug_mode


def is_production() -> bool:
    """Check if running in production environment"""
    return get_config().environment == "production"


def get_setting(path: str, default: Any = None) -> Any:
    """Get a configuration setting by path"""
    return config_manager.get_setting(path, default)


def set_setting(path: str, value: Any):
    """Set a configuration setting by path"""
    config_manager.set_setting(path, value)


def validate_config() -> List[str]:
    """Validate current configuration"""
    return config_manager.validate_config()