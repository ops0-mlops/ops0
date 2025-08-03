"""
ops0 Pipeline Validation - Validates pipeline structure and dependencies.
"""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
import logging

from .graph import PipelineGraph, StepNode
from .models import PipelineDefinition, StepDefinition
from .exceptions import ValidationError, DependencyError

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a validation issue"""
    severity: str  # "error", "warning", "info"
    message: str
    step_name: Optional[str] = None
    suggestion: Optional[str] = None


class PipelineValidator:
    """Validates pipeline structure, dependencies, and configuration"""

    def __init__(self):
        self.issues: List[ValidationIssue] = []

    def validate_pipeline(self, pipeline: PipelineGraph) -> List[ValidationIssue]:
        """
        Comprehensive pipeline validation.

        Args:
            pipeline: Pipeline to validate

        Returns:
            List of validation issues
        """
        self.issues = []

        # Core validations
        self._validate_structure(pipeline)
        self._validate_dependencies(pipeline)
        self._validate_cycles(pipeline)
        self._validate_step_signatures(pipeline)

        return self.issues

    def _validate_structure(self, pipeline: PipelineGraph):
        """Validate basic pipeline structure"""
        if not pipeline.steps:
            self.issues.append(ValidationIssue(
                severity="error",
                message="Pipeline has no steps defined",
                suggestion="Add at least one @ops0.step decorated function"
            ))
            return

        if len(pipeline.steps) == 1:
            self.issues.append(ValidationIssue(
                severity="warning",
                message="Pipeline has only one step",
                suggestion="Consider breaking down complex logic into multiple steps"
            ))

    def _validate_dependencies(self, pipeline: PipelineGraph):
        """Validate step dependencies"""
        for step_name, step_node in pipeline.steps.items():
            # Check if all dependencies are satisfied
            for dep in step_node.dependencies:
                if dep not in pipeline.storage_providers:
                    self.issues.append(ValidationIssue(
                        severity="error",
                        message=f"Step '{step_name}' depends on '{dep}' but no step provides it",
                        step_name=step_name,
                        suggestion=f"Add a step that calls ops0.storage.save('{dep}', data)"
                    ))

                # Check for potential dependency issues
                provider_step = pipeline.storage_providers.get(dep)
                if provider_step and provider_step == step_name:
                    self.issues.append(ValidationIssue(
                        severity="warning",
                        message=f"Step '{step_name}' depends on data it provides itself",
                        step_name=step_name,
                        suggestion="Consider restructuring the step logic"
                    ))

    def _validate_cycles(self, pipeline: PipelineGraph):
        """Detect dependency cycles"""
        try:
            pipeline.build_execution_order()
        except DependencyError as e:
            if "cycle" in str(e).lower():
                self.issues.append(ValidationIssue(
                    severity="error",
                    message="Circular dependency detected in pipeline",
                    suggestion="Remove circular dependencies between steps"
                ))

    def _validate_step_signatures(self, pipeline: PipelineGraph):
        """Validate step function signatures"""
        for step_name, step_node in pipeline.steps.items():
            try:
                import inspect
                sig = inspect.signature(step_node.func)

                # Check for problematic patterns
                if len(sig.parameters) > 10:
                    self.issues.append(ValidationIssue(
                        severity="warning",
                        message=f"Step '{step_name}' has many parameters ({len(sig.parameters)})",
                        step_name=step_name,
                        suggestion="Consider using ops0.storage.load() for data dependencies"
                    ))

                # Check return type hints
                if sig.return_annotation == inspect.Signature.empty:
                    self.issues.append(ValidationIssue(
                        severity="info",
                        message=f"Step '{step_name}' missing return type hint",
                        step_name=step_name,
                        suggestion="Add return type hint for better documentation"
                    ))

            except Exception as e:
                self.issues.append(ValidationIssue(
                    severity="warning",
                    message=f"Could not analyze signature for step '{step_name}': {e}",
                    step_name=step_name
                ))


def validate_pipeline(pipeline: PipelineGraph) -> List[ValidationIssue]:
    """
    Validate a pipeline and return issues.

    Args:
        pipeline: Pipeline to validate

    Returns:
        List of validation issues
    """
    validator = PipelineValidator()
    return validator.validate_pipeline(pipeline)


def validate_and_raise(pipeline: PipelineGraph):
    """
    Validate pipeline and raise ValidationError if errors found.

    Args:
        pipeline: Pipeline to validate

    Raises:
        ValidationError: If validation errors are found
    """
    issues = validate_pipeline(pipeline)

    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warning"]

    if errors:
        error_messages = [issue.message for issue in errors]
        raise ValidationError(
            f"Pipeline validation failed with {len(errors)} error(s)",
            validation_type="pipeline",
            validation_errors=error_messages
        )

    if warnings:
        logger.warning(f"Pipeline has {len(warnings)} warning(s):")
        for warning in warnings:
            logger.warning(f"  - {warning.message}")