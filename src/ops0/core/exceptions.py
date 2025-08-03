"""
ops0 Exception Classes - Structured error handling for the ops0 framework.

Provides detailed, actionable error messages with context information.
"""

from typing import Optional, Dict, Any, List
import traceback
import logging

logger = logging.getLogger(__name__)


class Ops0Error(Exception):
    """
    Base exception class for all ops0 errors.

    Provides structured error handling with context information
    and suggestions for resolution.
    """

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__.replace("Error", "").upper()
        self.context = context or {}
        self.suggestion = suggestion
        self.cause = cause
        self.traceback_info = traceback.format_exc() if cause else None

    def __str__(self) -> str:
        result = f"[{self.code}] {self.message}"

        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            result += f" (Context: {context_str})"

        if self.suggestion:
            result += f"\n💡 Suggestion: {self.suggestion}"

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization"""
        return {
            "type": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "context": self.context,
            "suggestion": self.suggestion,
            "cause": str(self.cause) if self.cause else None,
        }


class PipelineError(Ops0Error):
    """Errors related to pipeline definition and structure"""

    def __init__(
        self,
        message: str,
        pipeline_name: Optional[str] = None,
        step_count: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if pipeline_name:
            self.context["pipeline_name"] = pipeline_name
        if step_count is not None:
            self.context["step_count"] = step_count


class StepError(Ops0Error):
    """Errors related to individual step execution"""

    def __init__(
        self,
        message: str,
        step_name: Optional[str] = None,
        step_function: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if step_name:
            self.context["step_name"] = step_name
        if step_function:
            self.context["step_function"] = step_function


class StorageError(Ops0Error):
    """Errors related to data storage and retrieval"""

    def __init__(
        self,
        message: str,
        storage_key: Optional[str] = None,
        namespace: Optional[str] = None,
        backend_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if storage_key:
            self.context["storage_key"] = storage_key
        if namespace:
            self.context["namespace"] = namespace
        if backend_type:
            self.context["backend_type"] = backend_type


class ExecutionError(Ops0Error):
    """Errors during pipeline execution"""

    def __init__(
        self,
        message: str,
        execution_mode: Optional[str] = None,
        failed_step: Optional[str] = None,
        retry_count: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if execution_mode:
            self.context["execution_mode"] = execution_mode
        if failed_step:
            self.context["failed_step"] = failed_step
        if retry_count is not None:
            self.context["retry_count"] = retry_count


class DependencyError(Ops0Error):
    """Errors related to step dependencies and execution order"""

    def __init__(
        self,
        message: str,
        step_name: Optional[str] = None,
        missing_dependencies: Optional[List[str]] = None,
        circular_dependencies: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if step_name:
            self.context["step_name"] = step_name
        if missing_dependencies:
            self.context["missing_dependencies"] = missing_dependencies
        if circular_dependencies:
            self.context["circular_dependencies"] = circular_dependencies


class ValidationError(Ops0Error):
    """Errors related to validation of pipelines, steps, or data"""

    def __init__(
        self,
        message: str,
        validation_type: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
        validation_warnings: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if validation_type:
            self.context["validation_type"] = validation_type
        if validation_errors:
            self.context["validation_errors"] = validation_errors
        if validation_warnings:
            self.context["validation_warnings"] = validation_warnings


class ConfigurationError(Ops0Error):
    """Errors related to ops0 configuration"""

    def __init__(
        self,
        message: str,
        config_file: Optional[str] = None,
        config_section: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if config_file:
            self.context["config_file"] = config_file
        if config_section:
            self.context["config_section"] = config_section


class DeploymentError(Ops0Error):
    """Errors related to pipeline deployment"""

    def __init__(
        self,
        message: str,
        deployment_target: Optional[str] = None,
        deployment_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if deployment_target:
            self.context["deployment_target"] = deployment_target
        if deployment_id:
            self.context["deployment_id"] = deployment_id


class ContainerError(Ops0Error):
    """Errors related to containerization"""

    def __init__(
        self,
        message: str,
        container_id: Optional[str] = None,
        image_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if container_id:
            self.context["container_id"] = container_id
        if image_name:
            self.context["image_name"] = image_name


class NetworkError(Ops0Error):
    """Errors related to network operations"""

    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if endpoint:
            self.context["endpoint"] = endpoint
        if status_code:
            self.context["status_code"] = status_code


class TimeoutError(Ops0Error):
    """Errors related to operation timeouts"""

    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if timeout_seconds:
            self.context["timeout_seconds"] = timeout_seconds
        if operation:
            self.context["operation"] = operation


# Helper functions for common error scenarios
def raise_storage_not_found(key: str, namespace: str = None):
    """Raise a storage error for missing data"""
    raise StorageError(
        f"Storage key '{key}' not found",
        storage_key=key,
        namespace=namespace,
        suggestion=f"Ensure a step calls ops0.storage.save('{key}', data) before this step"
    )


def raise_step_dependency_missing(step_name: str, missing_deps: List[str]):
    """Raise a dependency error for missing step dependencies"""
    deps_str = "', '".join(missing_deps)
    raise DependencyError(
        f"Step '{step_name}' has missing dependencies: '{deps_str}'",
        step_name=step_name,
        missing_dependencies=missing_deps,
        suggestion="Add steps that provide the missing storage keys"
    )


def raise_circular_dependency(steps: List[str]):
    """Raise a dependency error for circular dependencies"""
    steps_str = " -> ".join(steps)
    raise DependencyError(
        f"Circular dependency detected: {steps_str}",
        circular_dependencies=steps,
        suggestion="Remove circular dependencies by restructuring your pipeline"
    )


def handle_step_execution_error(step_name: str, original_error: Exception) -> StepError:
    """Convert a step execution error to StepError"""
    return StepError(
        f"Step '{step_name}' failed during execution: {original_error}",
        step_name=step_name,
        cause=original_error,
        suggestion="Check step implementation and input data"
    )


# Context manager for error handling
class Ops0ErrorContext:
    """Context manager for enhanced error handling in ops0 operations"""

    def __init__(self, operation: str, **context):
        self.operation = operation
        self.context = context

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and not issubclass(exc_type, Ops0Error):
            # Convert non-ops0 errors to ops0 errors
            if "storage" in self.operation.lower():
                error_class = StorageError
            elif "execution" in self.operation.lower():
                error_class = ExecutionError
            elif "validation" in self.operation.lower():
                error_class = ValidationError
            else:
                error_class = Ops0Error

            raise error_class(
                f"Error during {self.operation}: {exc_val}",
                cause=exc_val,
                **self.context
            ) from exc_val

        return False  # Don't suppress ops0 errors