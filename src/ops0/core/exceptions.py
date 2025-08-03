"""
ops0 Core Exceptions

Custom exception hierarchy for ops0 framework.
Provides clear error messages and context for debugging.
"""

class Ops0Error(Exception):
    """Base exception for all ops0 errors"""

    def __init__(self, message: str, context: dict = None):
        self.message = message
        self.context = context or {}
        super().__init__(self.message)

    def __str__(self):
        base_msg = self.message
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{base_msg} (context: {context_str})"
        return base_msg


class PipelineError(Ops0Error):
    """Errors related to pipeline definition and execution"""
    pass


class StepError(Ops0Error):
    """Errors related to step definition and execution"""

    def __init__(self, message: str, step_name: str = None, context: dict = None):
        self.step_name = step_name
        context = context or {}
        if step_name:
            context["step"] = step_name
        super().__init__(message, context)


class StorageError(Ops0Error):
    """Errors related to data storage and retrieval"""

    def __init__(self, message: str, key: str = None, backend: str = None, context: dict = None):
        self.key = key
        self.backend = backend
        context = context or {}
        if key:
            context["key"] = key
        if backend:
            context["backend"] = backend
        super().__init__(message, context)


class ExecutionError(Ops0Error):
    """Errors during pipeline or step execution"""

    def __init__(self, message: str, step_name: str = None, exit_code: int = None, context: dict = None):
        self.step_name = step_name
        self.exit_code = exit_code
        context = context or {}
        if step_name:
            context["step"] = step_name
        if exit_code is not None:
            context["exit_code"] = exit_code
        super().__init__(message, context)


class DependencyError(Ops0Error):
    """Errors related to step dependencies and graph resolution"""

    def __init__(self, message: str, dependency: str = None, context: dict = None):
        self.dependency = dependency
        context = context or {}
        if dependency:
            context["dependency"] = dependency
        super().__init__(message, context)


class ValidationError(Ops0Error):
    """Errors during pipeline or configuration validation"""

    def __init__(self, message: str, field: str = None, value: any = None, context: dict = None):
        self.field = field
        self.value = value
        context = context or {}
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = value
        super().__init__(message, context)


class ConfigurationError(Ops0Error):
    """Errors related to configuration loading and validation"""
    pass


class SerializationError(StorageError):
    """Errors during data serialization/deserialization"""

    def __init__(self, message: str, data_type: str = None, serializer: str = None, context: dict = None):
        self.data_type = data_type
        self.serializer = serializer
        context = context or {}
        if data_type:
            context["data_type"] = data_type
        if serializer:
            context["serializer"] = serializer
        super().__init__(message, context=context)


class RuntimeError(ExecutionError):
    """Errors during runtime execution in production"""
    pass


class RegistryError(Ops0Error):
    """Errors related to registry operations"""

    def __init__(self, message: str, entry_type: str = None, entry_id: str = None, context: dict = None):
        self.entry_type = entry_type
        self.entry_id = entry_id
        context = context or {}
        if entry_type:
            context["entry_type"] = entry_type
        if entry_id:
            context["entry_id"] = entry_id
        super().__init__(message, context)


# Utility functions for common error patterns

def step_not_found_error(step_name: str) -> StepError:
    """Standard error for missing steps"""
    return StepError(
        f"Step '{step_name}' not found in pipeline",
        step_name=step_name
    )


def circular_dependency_error(cycle: list) -> DependencyError:
    """Standard error for circular dependencies"""
    cycle_str = " -> ".join(cycle)
    return DependencyError(
        f"Circular dependency detected: {cycle_str}",
        dependency=cycle_str
    )


def storage_key_not_found_error(key: str, backend: str = None) -> StorageError:
    """Standard error for missing storage keys"""
    return StorageError(
        f"Storage key '{key}' not found",
        key=key,
        backend=backend
    )


def invalid_step_signature_error(step_name: str, reason: str) -> StepError:
    """Standard error for invalid step signatures"""
    return StepError(
        f"Invalid step signature: {reason}",
        step_name=step_name,
        context={"reason": reason}
    )