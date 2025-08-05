"""
ops0 Validation Utilities

Comprehensive validation for pipelines, configurations, and environments.
"""

import sys
import subprocess
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import inspect
import ast

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Single validation issue"""
    level: ValidationLevel
    message: str
    field: Optional[str] = None
    suggestion: Optional[str] = None
    code: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validation check"""
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    def add_error(self, message: str, field: Optional[str] = None, suggestion: Optional[str] = None):
        """Add error issue"""
        self.valid = False
        self.issues.append(ValidationIssue(
            level=ValidationLevel.ERROR,
            message=message,
            field=field,
            suggestion=suggestion
        ))

    def add_warning(self, message: str, field: Optional[str] = None, suggestion: Optional[str] = None):
        """Add warning issue"""
        self.issues.append(ValidationIssue(
            level=ValidationLevel.WARNING,
            message=message,
            field=field,
            suggestion=suggestion
        ))

    def add_info(self, message: str, field: Optional[str] = None):
        """Add info issue"""
        self.issues.append(ValidationIssue(
            level=ValidationLevel.INFO,
            message=message,
            field=field
        ))

    def merge(self, other: 'ValidationResult'):
        """Merge another validation result"""
        self.valid = self.valid and other.valid
        self.issues.extend(other.issues)

    def has_errors(self) -> bool:
        """Check if there are any errors"""
        return any(issue.level == ValidationLevel.ERROR for issue in self.issues)

    def has_warnings(self) -> bool:
        """Check if there are any warnings"""
        return any(issue.level == ValidationLevel.WARNING for issue in self.issues)

    def get_errors(self) -> List[ValidationIssue]:
        """Get only error issues"""
        return [issue for issue in self.issues if issue.level == ValidationLevel.ERROR]

    def get_warnings(self) -> List[ValidationIssue]:
        """Get only warning issues"""
        return [issue for issue in self.issues if issue.level == ValidationLevel.WARNING]

    def __str__(self) -> str:
        """String representation"""
        if self.valid:
            return "✅ Validation passed"

        lines = ["❌ Validation failed:"]
        for issue in self.issues:
            icon = "❌" if issue.level == ValidationLevel.ERROR else "⚠️"
            lines.append(f"  {icon} {issue.message}")
            if issue.suggestion:
                lines.append(f"     💡 {issue.suggestion}")

        return "\n".join(lines)


def validate_python_version(required: str = "3.9") -> ValidationResult:
    """
    Validate Python version meets requirements.

    Args:
        required: Minimum required version (e.g., "3.9")

    Returns:
        ValidationResult
    """
    result = ValidationResult(valid=True)

    current = sys.version_info
    required_parts = [int(x) for x in required.split('.')]

    if current.major < required_parts[0] or (
            current.major == required_parts[0] and current.minor < required_parts[1]
    ):
        result.add_error(
            f"Python {required}+ required, but {current.major}.{current.minor} found",
            field="python_version",
            suggestion=f"Upgrade Python to {required} or higher"
        )
    else:
        result.add_info(f"Python {current.major}.{current.minor} meets requirements")

    return result


def validate_step_function(func: Callable) -> ValidationResult:
    """
    Validate function can be used as ops0 step.

    Args:
        func: Function to validate

    Returns:
        ValidationResult
    """
    result = ValidationResult(valid=True)

    # Check if callable
    if not callable(func):
        result.add_error("Step must be a callable function")
        return result

    # Check function name
    if not func.__name__.isidentifier():
        result.add_error(
            f"Invalid function name: {func.__name__}",
            suggestion="Use valid Python identifier for function name"
        )

    # Check for source code availability
    try:
        source = inspect.getsource(func)
        if not source.strip():
            result.add_error("Function has no source code")
    except OSError:
        result.add_error(
            "Cannot access function source code",
            suggestion="Ensure function is defined in a file, not in REPL"
        )
        return result

    # Parse AST
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        result.add_error(f"Invalid Python syntax: {e}")
        return result

    # Check for dangerous operations
    class DangerousOperationVisitor(ast.NodeVisitor):
        def __init__(self):
            self.issues = []

        def visit_Import(self, node):
            dangerous_modules = ['os', 'subprocess', 'sys']
            for alias in node.names:
                if alias.name in dangerous_modules:
                    self.issues.append(f"Import of potentially dangerous module: {alias.name}")
            self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                dangerous_funcs = ['eval', 'exec', 'compile', '__import__']
                if node.func.id in dangerous_funcs:
                    self.issues.append(f"Use of dangerous function: {node.func.id}")
            self.generic_visit(node)

    visitor = DangerousOperationVisitor()
    visitor.visit(tree)

    for issue in visitor.issues:
        result.add_warning(issue, suggestion="Consider security implications")

    # Check function signature
    sig = inspect.signature(func)

    # Check for *args and **kwargs
    has_var_positional = any(
        p.kind == inspect.Parameter.VAR_POSITIONAL
        for p in sig.parameters.values()
    )
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )

    if has_var_positional or has_var_keyword:
        result.add_warning(
            "Function uses *args or **kwargs",
            suggestion="Consider using explicit parameters for better type checking"
        )

    # Check for type annotations
    params_without_annotation = [
        name for name, param in sig.parameters.items()
        if param.annotation == inspect.Parameter.empty
    ]

    if params_without_annotation:
        result.add_info(
            f"Parameters without type annotations: {', '.join(params_without_annotation)}",
            suggestion="Add type annotations for better validation"
        )

    if sig.return_annotation == inspect.Signature.empty:
        result.add_info(
            "Function has no return type annotation",
            suggestion="Add return type annotation"
        )

    return result


def validate_pipeline_config(config: Dict[str, Any]) -> ValidationResult:
    """
    Validate pipeline configuration.

    Args:
        config: Pipeline configuration dictionary

    Returns:
        ValidationResult
    """
    result = ValidationResult(valid=True)

    # Check required fields
    required_fields = ['name', 'version']
    for field in required_fields:
        if field not in config:
            result.add_error(f"Missing required field: {field}", field=field)

    # Validate name
    if 'name' in config:
        name = config['name']
        if not isinstance(name, str):
            result.add_error("Pipeline name must be a string", field="name")
        elif not re.match(r'^[a-zA-Z][a-zA-Z0-9-_]*$', name):
            result.add_error(
                f"Invalid pipeline name: {name}",
                field="name",
                suggestion="Use alphanumeric characters, hyphens, and underscores only"
            )

    # Validate version
    if 'version' in config:
        version = config['version']
        if not isinstance(version, str):
            result.add_error("Version must be a string", field="version")
        elif not re.match(r'^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$', version):
            result.add_warning(
                f"Version '{version}' doesn't follow semantic versioning",
                field="version",
                suggestion="Use format: MAJOR.MINOR.PATCH (e.g., 1.0.0)"
            )

    # Validate environment
    if 'environment' in config:
        valid_envs = ['development', 'staging', 'production']
        if config['environment'] not in valid_envs:
            result.add_warning(
                f"Unknown environment: {config['environment']}",
                field="environment",
                suggestion=f"Use one of: {', '.join(valid_envs)}"
            )

    # Validate resources
    if 'resources' in config:
        resources = config['resources']
        if not isinstance(resources, dict):
            result.add_error("Resources must be a dictionary", field="resources")
        else:
            # Validate CPU
            if 'cpu' in resources:
                cpu = resources['cpu']
                if isinstance(cpu, str):
                    if not re.match(r'^\d+(\.\d+)?[m]?$', cpu):
                        result.add_error(
                            f"Invalid CPU specification: {cpu}",
                            field="resources.cpu",
                            suggestion="Use format like '1', '0.5', or '500m'"
                        )
                elif not isinstance(cpu, (int, float)):
                    result.add_error("CPU must be a number or string", field="resources.cpu")

            # Validate memory
            if 'memory' in resources:
                memory = resources['memory']
                if isinstance(memory, str):
                    if not re.match(r'^\d+(\.\d+)?[KMGT]i?$', memory):
                        result.add_error(
                            f"Invalid memory specification: {memory}",
                            field="resources.memory",
                            suggestion="Use format like '512Mi', '2Gi'"
                        )
                else:
                    result.add_error("Memory must be a string", field="resources.memory")

    return result


def validate_resource_requirements(requirements: Dict[str, Any]) -> ValidationResult:
    """
    Validate resource requirements specification.

    Args:
        requirements: Resource requirements dictionary

    Returns:
        ValidationResult
    """
    result = ValidationResult(valid=True)

    # Validate CPU
    if 'cpu' in requirements:
        cpu = requirements['cpu']
        try:
            if isinstance(cpu, str):
                # Parse CPU string (e.g., "500m", "2")
                if cpu.endswith('m'):
                    cpu_value = float(cpu[:-1]) / 1000
                else:
                    cpu_value = float(cpu)
            else:
                cpu_value = float(cpu)

            if cpu_value <= 0:
                result.add_error("CPU must be positive", field="cpu")
            elif cpu_value > 64:
                result.add_warning(
                    f"Very high CPU requirement: {cpu_value}",
                    field="cpu",
                    suggestion="Consider if this is really needed"
                )
        except (ValueError, TypeError):
            result.add_error(f"Invalid CPU value: {cpu}", field="cpu")

    # Validate memory
    if 'memory' in requirements:
        memory = requirements['memory']
        try:
            # Parse memory string (e.g., "512Mi", "2Gi")
            match = re.match(r'^(\d+(?:\.\d+)?)\s*([KMGT]i?)$', str(memory))
            if match:
                value, unit = match.groups()
                value = float(value)

                # Convert to MiB
                multipliers = {
                    'K': 1 / 1024, 'Ki': 1 / 1024,
                    'M': 1, 'Mi': 1,
                    'G': 1024, 'Gi': 1024,
                    'T': 1024 * 1024, 'Ti': 1024 * 1024
                }
                memory_mib = value * multipliers.get(unit, 1)

                if memory_mib <= 0:
                    result.add_error("Memory must be positive", field="memory")
                elif memory_mib < 128:
                    result.add_warning(
                        "Very low memory requirement",
                        field="memory",
                        suggestion="Consider at least 128Mi"
                    )
                elif memory_mib > 256 * 1024:  # 256 GiB
                    result.add_warning(
                        "Very high memory requirement",
                        field="memory",
                        suggestion="Consider if this is really needed"
                    )
            else:
                result.add_error(f"Invalid memory format: {memory}", field="memory")
        except Exception:
            result.add_error(f"Invalid memory value: {memory}", field="memory")

    # Validate GPU
    if 'gpu' in requirements:
        gpu = requirements['gpu']
        try:
            gpu_value = int(gpu)
            if gpu_value < 0:
                result.add_error("GPU count must be non-negative", field="gpu")
            elif gpu_value > 8:
                result.add_warning(
                    f"Very high GPU requirement: {gpu_value}",
                    field="gpu",
                    suggestion="Consider if this is really needed"
                )
        except (ValueError, TypeError):
            result.add_error(f"Invalid GPU value: {gpu}", field="gpu")

    return result


def validate_cloud_credentials(provider: str) -> ValidationResult:
    """
    Validate cloud provider credentials are configured.

    Args:
        provider: Cloud provider name (aws, gcp, azure)

    Returns:
        ValidationResult
    """
    result = ValidationResult(valid=True)

    provider = provider.lower()

    if provider == 'aws':
        # Check AWS credentials
        required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
        missing = [var for var in required_vars if not os.environ.get(var)]

        if missing:
            result.add_error(
                f"Missing AWS credentials: {', '.join(missing)}",
                suggestion="Set environment variables or configure AWS CLI"
            )
        else:
            result.add_info("AWS credentials found")

        # Check AWS CLI
        try:
            subprocess.run(['aws', '--version'], capture_output=True, check=True)
            result.add_info("AWS CLI installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            result.add_warning(
                "AWS CLI not found",
                suggestion="Install AWS CLI for better functionality"
            )

    elif provider == 'gcp':
        # Check GCP credentials
        if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
            result.add_error(
                "Missing GCP credentials",
                suggestion="Set GOOGLE_APPLICATION_CREDENTIALS environment variable"
            )
        else:
            cred_file = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
            if not Path(cred_file).exists():
                result.add_error(
                    f"GCP credentials file not found: {cred_file}",
                    suggestion="Check file path is correct"
                )
            else:
                result.add_info("GCP credentials found")

        # Check gcloud CLI
        try:
            subprocess.run(['gcloud', '--version'], capture_output=True, check=True)
            result.add_info("gcloud CLI installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            result.add_warning(
                "gcloud CLI not found",
                suggestion="Install Google Cloud SDK for better functionality"
            )

    elif provider == 'azure':
        # Check Azure credentials
        required_vars = ['AZURE_SUBSCRIPTION_ID', 'AZURE_TENANT_ID', 'AZURE_CLIENT_ID', 'AZURE_CLIENT_SECRET']
        missing = [var for var in required_vars if not os.environ.get(var)]

        if missing:
            result.add_error(
                f"Missing Azure credentials: {', '.join(missing)}",
                suggestion="Set environment variables or use Azure CLI"
            )
        else:
            result.add_info("Azure credentials found")

        # Check Azure CLI
        try:
            subprocess.run(['az', '--version'], capture_output=True, check=True)
            result.add_info("Azure CLI installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            result.add_warning(
                "Azure CLI not found",
                suggestion="Install Azure CLI for better functionality"
            )

    else:
        result.add_error(
            f"Unknown cloud provider: {provider}",
            suggestion="Supported providers: aws, gcp, azure"
        )

    return result


def validate_docker_installed() -> ValidationResult:
    """
    Validate Docker is installed and running.

    Returns:
        ValidationResult
    """
    result = ValidationResult(valid=True)

    # Check Docker installed
    try:
        version_output = subprocess.run(
            ['docker', '--version'],
            capture_output=True,
            text=True,
            check=True
        )
        result.add_info(f"Docker installed: {version_output.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        result.add_error(
            "Docker not found",
            suggestion="Install Docker from https://docs.docker.com/get-docker/"
        )
        return result

    # Check Docker daemon running
    try:
        subprocess.run(
            ['docker', 'info'],
            capture_output=True,
            check=True
        )
        result.add_info("Docker daemon is running")
    except subprocess.CalledProcessError:
        result.add_error(
            "Docker daemon not running",
            suggestion="Start Docker Desktop or run 'sudo systemctl start docker'"
        )

    # Check Docker Compose
    try:
        subprocess.run(
            ['docker-compose', '--version'],
            capture_output=True,
            check=True
        )
        result.add_info("Docker Compose installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Try new docker compose command
        try:
            subprocess.run(
                ['docker', 'compose', 'version'],
                capture_output=True,
                check=True
            )
            result.add_info("Docker Compose (plugin) installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            result.add_warning(
                "Docker Compose not found",
                suggestion="Install Docker Compose for multi-container support"
            )

    return result


class PipelineValidator:
    """Comprehensive pipeline validator"""

    def __init__(self):
        self.validators: List[Callable[[Any], ValidationResult]] = []

    def add_validator(self, validator: Callable[[Any], ValidationResult]):
        """Add custom validator"""
        self.validators.append(validator)

    def validate(self, pipeline: Any) -> ValidationResult:
        """Run all validators on pipeline"""
        result = ValidationResult(valid=True)

        # Run built-in validations
        if hasattr(pipeline, 'config'):
            result.merge(validate_pipeline_config(pipeline.config))

        if hasattr(pipeline, 'steps'):
            for step_name, step_func in pipeline.steps.items():
                step_result = validate_step_function(step_func)
                if not step_result.valid:
                    for issue in step_result.issues:
                        issue.field = f"step.{step_name}.{issue.field}" if issue.field else f"step.{step_name}"
                result.merge(step_result)

        # Run custom validators
        for validator in self.validators:
            result.merge(validator(pipeline))

        return result


class ConfigValidator:
    """Configuration validator with schema support"""

    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        self.schema = schema

    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration against schema"""
        result = ValidationResult(valid=True)

        if self.schema:
            # Simple schema validation
            for key, expected_type in self.schema.items():
                if key not in config:
                    result.add_error(f"Missing required field: {key}", field=key)
                elif not isinstance(config[key], expected_type):
                    result.add_error(
                        f"Invalid type for {key}: expected {expected_type.__name__}, got {type(config[key]).__name__}",
                        field=key
                    )

        return result