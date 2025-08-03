"""
ops0 Core Exceptions

Custom exception hierarchy for ops0 with clear error messages
and actionable guidance for users.
"""

from typing import Optional, List, Dict, Any


class Ops0Error(Exception):
    """
    Base exception for all ops0 errors.

    Provides common functionality for all ops0 exceptions including
    error codes, user guidance, and context information.
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        guidance: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.guidance = guidance
        self.context = context or {}

    def __str__(self):
        result = self.message
        if self.error_code:
            result = f"[{self.error_code}] {result}"
        if self.guidance:
            result += f"\n💡 {self.guidance}"
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/debugging"""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "guidance": self.guidance,
            "context": self.context
        }


class PipelineError(Ops0Error):
    """Errors related to pipeline definition and management"""

    def __init__(self, message: str, pipeline_name: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if pipeline_name:
            self.context["pipeline_name"] = pipeline_name


class StepError(Ops0Error):
    """Errors related to individual pipeline steps"""

    def __init__(
        self,
        message: str,
        step_name: Optional[str] = None,
        function_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if step_name:
            self.context["step_name"] = step_name
        if function_name:
            self.context["function_name"] = function_name


class StorageError(Ops0Error):
    """Errors related to data storage and retrieval"""

    def __init__(
        self,
        message: str,
        storage_key: Optional[str] = None,
        namespace: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if storage_key:
            self.context["storage_key"] = storage_key
        if namespace:
            self.context["namespace"] = namespace


class ExecutionError(Ops0Error):
    """Errors during pipeline execution"""

    def __init__(
        self,
        message: str,
        execution_mode: Optional[str] = None,
        failed_step: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if execution_mode:
            self.context["execution_mode"] = execution_mode
        if failed_step:
            self.context["failed_step"] = failed_step


class DependencyError(Ops0Error):
    """Errors related to step dependencies and execution order"""

    def __init__(
        self,
        message: str,
        step_name: Optional[str] = None,
        missing_dependencies: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if step_name:
            self.context["step_name"] = step_name
        if missing_dependencies:
            self.context["missing_dependencies"] = missing_dependencies


class ValidationError(Ops0Error):
    """Errors related to validation of pipelines, steps, or data"""

    def __init__(
        self,
        message: str,
        validation_type: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if validation_type:
            self.context["validation_type"] = validation_type
        if validation_errors:
            self.context["validation_errors"] = validation_errors


class ConfigurationError(Ops0Error):
    """Errors related to ops0 configuration"""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if config_key:
            self.context["config_key"] = config_key
        if config_file:
            self.context["config_file"] = config_file


class DeploymentError(Ops0Error):
    """Errors during pipeline deployment"""

    def __init__(
        self,
        message: str,
        deployment_target: Optional[str] = None,
        deployment_stage: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if deployment_target:
            self.context["deployment_target"] = deployment_target
        if deployment_stage:
            self.context["deployment_stage"] = deployment_stage


class ContainerError(Ops0Error):
    """Errors related to containerization and container runtime"""

    def __init__(
        self,
        message: str,
        container_name: Optional[str] = None,
        image_tag: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if container_name:
            self.context["container_name"] = container_name
        if image_tag:
            self.context["image_tag"] = image_tag


class IntegrationError(Ops0Error):
    """Errors related to third-party integrations (ML frameworks, cloud providers, etc.)"""

    def __init__(
        self,
        message: str,
        integration_type: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if integration_type:
            self.context["integration_type"] = integration_type
        if provider:
            self.context["provider"] = provider


# Common error factories with helpful guidance
def step_not_found_error(step_name: str, available_steps: List[str]) -> StepError:
    """Factory for step not found errors with helpful suggestions"""
    available_str = ", ".join(available_steps) if available_steps else "None"
    return StepError(
        f"Step '{step_name}' not found in current pipeline",
        step_name=step_name,
        error_code="STEP_NOT_FOUND",
        guidance=f"Available steps: {available_str}. Check spelling or ensure step is defined.",
        context={"available_steps": available_steps}
    )


def storage_key_not_found_error(key: str, namespace: str = "default") -> StorageError:
    """Factory for storage key not found errors"""
    return StorageError(
        f"Storage key '{key}' not found in namespace '{namespace}'",
        storage_key=key,
        namespace=namespace,
        error_code="STORAGE_KEY_NOT_FOUND",
        guidance="Ensure data is saved before loading. Check key spelling and namespace."
    )


def circular_dependency_error(dependency_chain: List[str]) -> DependencyError:
    """Factory for circular dependency errors"""
    chain_str = " → ".join(dependency_chain)
    return DependencyError(
        f"Circular dependency detected in pipeline: {chain_str}",
        error_code="CIRCULAR_DEPENDENCY",
        guidance="Review step dependencies and remove circular references.",
        context={"dependency_chain": dependency_chain}
    )


def pipeline_not_found_error() -> PipelineError:
    """Factory for missing pipeline context errors"""
    return PipelineError(
        "No active pipeline found",
        error_code="NO_ACTIVE_PIPELINE",
        guidance="Use 'with ops0.pipeline(\"name\"):' or '@ops0.pipeline' decorator to define a pipeline context."
    )


def invalid_step_function_error(function_name: str, reason: str) -> StepError:
    """Factory for invalid step function errors"""
    return StepError(
        f"Function '{function_name}' cannot be used as a step: {reason}",
        function_name=function_name,
        error_code="INVALID_STEP_FUNCTION",
        guidance="Ensure function is callable and follows ops0 step conventions."
    )


def execution_failed_error(step_name: str, original_error: Exception) -> ExecutionError:
    """Factory for step execution failure errors"""
    return ExecutionError(
        f"Step '{step_name}' failed during execution: {str(original_error)}",
        failed_step=step_name,
        error_code="STEP_EXECUTION_FAILED",
        guidance="Check step implementation and inputs. Use 'ops0 debug' for detailed information.",
        context={"original_error": str(original_error), "original_type": type(original_error).__name__}
    )


def deployment_failed_error(target: str, stage: str, reason: str) -> DeploymentError:
    """Factory for deployment failure errors"""
    return DeploymentError(
        f"Deployment to '{target}' failed at stage '{stage}': {reason}",
        deployment_target=target,
        deployment_stage=stage,
        error_code="DEPLOYMENT_FAILED",
        guidance="Check deployment logs and ensure target environment is accessible."
    )


def missing_dependency_error(step_name: str, missing_deps: List[str]) -> DependencyError:
    """Factory for missing step dependency errors"""
    deps_str = ", ".join(missing_deps)
    return DependencyError(
        f"Step '{step_name}' has unresolved dependencies: {deps_str}",
        step_name=step_name,
        missing_dependencies=missing_deps,
        error_code="MISSING_DEPENDENCIES",
        guidance="Ensure all required steps are defined and data keys are available."
    )


def configuration_invalid_error(key: str, value: Any, expected: str) -> ConfigurationError:
    """Factory for invalid configuration errors"""
    return ConfigurationError(
        f"Invalid configuration for '{key}': got {value}, expected {expected}",
        config_key=key,
        error_code="INVALID_CONFIGURATION",
        guidance=f"Update configuration with a valid {expected} value."
    )


# Exception handling utilities
def handle_ops0_error(error: Ops0Error, logger=None) -> None:
    """
    Standard error handler for ops0 errors.

    Logs error details and provides user-friendly output.
    """
    if logger:
        logger.error(f"ops0 Error: {error.message}")
        if error.context:
            logger.debug(f"Error context: {error.context}")

    # In development mode, show more details
    import os
    if os.getenv("OPS0_ENV") == "development":
        import traceback
        traceback.print_exc()


def wrap_external_error(external_error: Exception, context: str) -> Ops0Error:
    """
    Wrap external errors in ops0 exceptions with context.

    Args:
        external_error: The original exception
        context: Description of what was happening when error occurred

    Returns:
        Appropriate ops0 exception with guidance
    """
    error_type = type(external_error).__name__

    if "import" in str(external_error).lower():
        return IntegrationError(
            f"Import error in {context}: {str(external_error)}",
            error_code="IMPORT_ERROR",
            guidance="Install required dependencies with 'pip install ops0[ml]' or check Python path."
        )
    elif "permission" in str(external_error).lower():
        return StorageError(
            f"Permission error in {context}: {str(external_error)}",
            error_code="PERMISSION_ERROR",
            guidance="Check file/directory permissions or run with appropriate privileges."
        )
    elif "connection" in str(external_error).lower() or "network" in str(external_error).lower():
        return IntegrationError(
            f"Network error in {context}: {str(external_error)}",
            error_code="NETWORK_ERROR",
            guidance="Check network connectivity and service availability."
        )
    else:
        return Ops0Error(
            f"Unexpected error in {context}: {str(external_error)}",
            error_code="UNEXPECTED_ERROR",
            guidance="This may be a bug. Please report with reproduction steps.",
            context={"original_error": str(external_error), "original_type": error_type}
        )