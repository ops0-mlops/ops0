"""
ops0 Pipeline Validation System

Validates pipeline structure, step definitions, and data flow automatically.
Provides comprehensive error checking and suggestions for optimization.
"""

import ast
import inspect
import typing
from typing import Dict, List, Set, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from .exceptions import ValidationError, DependencyError, StepError
from .analyzer import FunctionAnalyzer
from .graph import PipelineGraph, StepNode


class ValidationLevel(Enum):
    """Validation strictness levels"""
    BASIC = "basic"  # Essential checks only
    STANDARD = "standard"  # Recommended checks
    STRICT = "strict"  # All possible checks
    CUSTOM = "custom"  # User-defined rules


class IssueType(Enum):
    """Types of validation issues"""
    ERROR = "error"  # Must be fixed
    WARNING = "warning"  # Should be addressed
    INFO = "info"  # Optional improvements
    PERFORMANCE = "performance"  # Performance suggestions


@dataclass
class ValidationIssue:
    """Represents a validation issue"""
    type: IssueType
    code: str
    message: str
    step_name: Optional[str] = None
    suggestion: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def __str__(self) -> str:
        prefix = {
            IssueType.ERROR: "❌",
            IssueType.WARNING: "⚠️",
            IssueType.INFO: "ℹ️",
            IssueType.PERFORMANCE: "⚡"
        }[self.type]

        result = f"{prefix} [{self.code}] {self.message}"
        if self.step_name:
            result = f"{result} (step: {self.step_name})"
        if self.suggestion:
            result = f"{result}\n   💡 {self.suggestion}"
        return result


@dataclass
class ValidationResult:
    """Results of pipeline validation"""
    is_valid: bool
    issues: List[ValidationIssue]
    score: float  # 0-100 quality score
    metadata: Dict[str, Any]

    @property
    def errors(self) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue.type == IssueType.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue.type == IssueType.WARNING]

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def summary(self) -> str:
        """Get a summary string of validation results"""
        if self.is_valid:
            return f"✅ Pipeline valid (Score: {self.score:.1f}/100, {len(self.warnings)} warnings)"
        else:
            return f"❌ Pipeline invalid ({len(self.errors)} errors, {len(self.warnings)} warnings)"


class StepValidator:
    """Validates individual pipeline steps"""

    def __init__(self):
        self.issues: List[ValidationIssue] = []

    def validate_step(self, step_node: StepNode) -> List[ValidationIssue]:
        """Validate a single step"""
        self.issues = []

        # Basic function validation
        self._validate_function_signature(step_node)
        self._validate_function_implementation(step_node)
        self._validate_dependencies(step_node)
        self._validate_type_hints(step_node)
        self._validate_documentation(step_node)
        self._validate_performance_patterns(step_node)

        return self.issues

    def _validate_function_signature(self, step_node: StepNode):
        """Validate function signature"""
        func = step_node.func
        signature = inspect.signature(func)

        # Check for *args, **kwargs which can make dependency detection harder
        for param in signature.parameters.values():
            if param.kind == param.VAR_POSITIONAL:
                self.issues.append(ValidationIssue(
                    type=IssueType.WARNING,
                    code="STEP_VAR_ARGS",
                    message=f"Step uses *args which may complicate dependency detection",
                    step_name=step_node.name,
                    suggestion="Consider using explicit parameters for better dependency tracking"
                ))
            elif param.kind == param.VAR_KEYWORD:
                self.issues.append(ValidationIssue(
                    type=IssueType.WARNING,
                    code="STEP_VAR_KWARGS",
                    message=f"Step uses **kwargs which may complicate dependency detection",
                    step_name=step_node.name,
                    suggestion="Consider using explicit parameters for better dependency tracking"
                ))

    def _validate_function_implementation(self, step_node: StepNode):
        """Validate function implementation using AST analysis"""
        try:
            analyzer = FunctionAnalyzer(step_node.func)
            source = analyzer.source

            # Check for common anti-patterns
            if "global " in source:
                self.issues.append(ValidationIssue(
                    type=IssueType.WARNING,
                    code="STEP_GLOBAL_VARS",
                    message="Step uses global variables",
                    step_name=step_node.name,
                    suggestion="Consider passing data through parameters or ops0.storage"
                ))

            # Check for hardcoded values
            tree = analyzer.tree
            for node in ast.walk(tree):
                if isinstance(node, ast.Str) and len(node.s) > 50:
                    self.issues.append(ValidationIssue(
                        type=IssueType.INFO,
                        code="STEP_HARDCODED_STRING",
                        message="Step contains long hardcoded strings",
                        step_name=step_node.name,
                        suggestion="Consider moving configuration to external files"
                    ))

        except Exception as e:
            self.issues.append(ValidationIssue(
                type=IssueType.WARNING,
                code="STEP_ANALYSIS_FAILED",
                message=f"Could not analyze step implementation: {str(e)}",
                step_name=step_node.name
            ))

    def _validate_dependencies(self, step_node: StepNode):
        """Validate step dependencies"""
        dependencies = step_node.dependencies

        # Check for suspiciously large number of dependencies
        if len(dependencies) > 10:
            self.issues.append(ValidationIssue(
                type=IssueType.WARNING,
                code="STEP_MANY_DEPS",
                message=f"Step has {len(dependencies)} dependencies",
                step_name=step_node.name,
                suggestion="Consider breaking down complex steps into smaller ones"
            ))

        # Check for dependency naming patterns
        for dep in dependencies:
            if not dep.isidentifier():
                self.issues.append(ValidationIssue(
                    type=IssueType.WARNING,
                    code="STEP_INVALID_DEP_NAME",
                    message=f"Dependency key '{dep}' is not a valid identifier",
                    step_name=step_node.name,
                    suggestion="Use valid Python identifiers for storage keys"
                ))

    def _validate_type_hints(self, step_node: StepNode):
        """Validate type hints"""
        func = step_node.func
        signature = inspect.signature(func)

        # Check if function has type hints
        has_param_hints = any(
            param.annotation != param.empty
            for param in signature.parameters.values()
        )
        has_return_hint = signature.return_annotation != signature.empty

        if not has_param_hints:
            self.issues.append(ValidationIssue(
                type=IssueType.INFO,
                code="STEP_NO_PARAM_HINTS",
                message="Step parameters lack type hints",
                step_name=step_node.name,
                suggestion="Add type hints for better validation and documentation"
            ))

        if not has_return_hint:
            self.issues.append(ValidationIssue(
                type=IssueType.INFO,
                code="STEP_NO_RETURN_HINT",
                message="Step return value lacks type hint",
                step_name=step_node.name,
                suggestion="Add return type hint for better validation"
            ))

    def _validate_documentation(self, step_node: StepNode):
        """Validate step documentation"""
        func = step_node.func
        docstring = inspect.getdoc(func)

        if not docstring:
            self.issues.append(ValidationIssue(
                type=IssueType.INFO,
                code="STEP_NO_DOCSTRING",
                message="Step lacks documentation",
                step_name=step_node.name,
                suggestion="Add a docstring describing what this step does"
            ))
        elif len(docstring.split()) < 3:
            self.issues.append(ValidationIssue(
                type=IssueType.INFO,
                code="STEP_SHORT_DOCSTRING",
                message="Step documentation is very brief",
                step_name=step_node.name,
                suggestion="Expand documentation to describe purpose and behavior"
            ))

    def _validate_performance_patterns(self, step_node: StepNode):
        """Validate performance-related patterns"""
        try:
            analyzer = FunctionAnalyzer(step_node.func)
            source = analyzer.source.lower()

            # Check for potential performance issues
            if "for " in source and "range(" in source and ".append(" in source:
                self.issues.append(ValidationIssue(
                    type=IssueType.PERFORMANCE,
                    code="STEP_INEFFICIENT_LOOP",
                    message="Potential inefficient loop with append",
                    step_name=step_node.name,
                    suggestion="Consider using list comprehensions or vectorized operations"
                ))

            if source.count("ops0.storage.load(") > 5:
                self.issues.append(ValidationIssue(
                    type=IssueType.PERFORMANCE,
                    code="STEP_MANY_LOADS",
                    message="Step loads data multiple times",
                    step_name=step_node.name,
                    suggestion="Consider loading data once and reusing"
                ))

        except Exception:
            # Ignore analysis errors for performance validation
            pass


class PipelineValidator:
    """Validates complete pipeline structure and dependencies"""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.step_validator = StepValidator()

    def validate_pipeline(self, pipeline: PipelineGraph) -> ValidationResult:
        """Validate complete pipeline"""
        issues: List[ValidationIssue] = []

        # Basic structural validation
        issues.extend(self._validate_structure(pipeline))

        # Validate individual steps
        for step_name, step_node in pipeline.steps.items():
            step_issues = self.step_validator.validate_step(step_node)
            issues.extend(step_issues)

        # Validate dependencies and execution order
        issues.extend(self._validate_dependencies(pipeline))

        # Validate data flow
        issues.extend(self._validate_data_flow(pipeline))

        # Pipeline-level optimizations
        if self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
            issues.extend(self._validate_optimizations(pipeline))

        # Calculate quality score
        score = self._calculate_quality_score(issues, pipeline)

        # Determine if pipeline is valid (no errors)
        errors = [issue for issue in issues if issue.type == IssueType.ERROR]
        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            score=score,
            metadata={
                "total_steps": len(pipeline.steps),
                "validation_level": self.validation_level.value,
                "dependency_count": sum(len(node.dependencies) for node in pipeline.steps.values())
            }
        )

    def _validate_structure(self, pipeline: PipelineGraph) -> List[ValidationIssue]:
        """Validate basic pipeline structure"""
        issues = []

        # Check if pipeline has steps
        if not pipeline.steps:
            issues.append(ValidationIssue(
                type=IssueType.ERROR,
                code="PIPELINE_NO_STEPS",
                message="Pipeline contains no steps",
                suggestion="Add at least one @ops0.step function to the pipeline"
            ))
            return issues

        # Check for reasonable number of steps
        step_count = len(pipeline.steps)
        if step_count > 50:
            issues.append(ValidationIssue(
                type=IssueType.WARNING,
                code="PIPELINE_MANY_STEPS",
                message=f"Pipeline has {step_count} steps",
                suggestion="Consider breaking large pipelines into smaller, composable ones"
            ))

        # Check step naming
        for step_name in pipeline.steps.keys():
            if not step_name.isidentifier():
                issues.append(ValidationIssue(
                    type=IssueType.ERROR,
                    code="PIPELINE_INVALID_STEP_NAME",
                    message=f"Step name '{step_name}' is not a valid identifier",
                    step_name=step_name
                ))
            elif step_name.startswith('_'):
                issues.append(ValidationIssue(
                    type=IssueType.WARNING,
                    code="PIPELINE_PRIVATE_STEP_NAME",
                    message=f"Step name '{step_name}' starts with underscore",
                    step_name=step_name,
                    suggestion="Consider using public names for better visibility"
                ))

        return issues

    def _validate_dependencies(self, pipeline: PipelineGraph) -> List[ValidationIssue]:
        """Validate step dependencies and execution order"""
        issues = []

        try:
            # Try to build execution order to check for circular dependencies
            execution_order = pipeline.build_execution_order()

            # Check for steps that never execute (no path from root)
            executed_steps = set()
            for level in execution_order:
                executed_steps.update(level)

            all_steps = set(pipeline.steps.keys())
            orphaned_steps = all_steps - executed_steps

            for orphaned in orphaned_steps:
                issues.append(ValidationIssue(
                    type=IssueType.WARNING,
                    code="PIPELINE_ORPHANED_STEP",
                    message=f"Step '{orphaned}' may never execute",
                    step_name=orphaned,
                    suggestion="Check if this step's dependencies are correctly defined"
                ))

            # Check for overly deep pipelines
            if len(execution_order) > 10:
                issues.append(ValidationIssue(
                    type=IssueType.PERFORMANCE,
                    code="PIPELINE_DEEP_NESTING",
                    message=f"Pipeline has {len(execution_order)} sequential levels",
                    suggestion="Consider if some steps can be parallelized"
                ))

            # Check for bottlenecks (levels with only one step)
            for i, level in enumerate(execution_order):
                if len(level) == 1 and len(execution_order) > 1:
                    step_name = level[0]
                    issues.append(ValidationIssue(
                        type=IssueType.PERFORMANCE,
                        code="PIPELINE_BOTTLENECK",
                        message=f"Step '{step_name}' creates a bottleneck at level {i + 1}",
                        step_name=step_name,
                        suggestion="Consider if this step can be optimized or parallelized"
                    ))

        except Exception as e:
            issues.append(ValidationIssue(
                type=IssueType.ERROR,
                code="PIPELINE_DEPENDENCY_ERROR",
                message=f"Cannot resolve pipeline dependencies: {str(e)}",
                suggestion="Check for circular dependencies or missing steps"
            ))

        return issues

    def _validate_data_flow(self, pipeline: PipelineGraph) -> List[ValidationIssue]:
        """Validate data flow between steps"""
        issues = []

        # Track what data is produced and consumed
        produced_keys = set()
        consumed_keys = set()

        for step_name, step_node in pipeline.steps.items():
            # Track consumed keys (dependencies)
            consumed_keys.update(step_node.dependencies)

            # Track produced keys (would need to analyze storage.save calls)
            try:
                analyzer = FunctionAnalyzer(step_node.func)
                produced_in_step = analyzer.get_storage_saves()
                produced_keys.update(produced_in_step)
            except Exception:
                # If we can't analyze, skip this step
                continue

        # Check for consumed but never produced keys
        orphaned_keys = consumed_keys - produced_keys
        for key in orphaned_keys:
            issues.append(ValidationIssue(
                type=IssueType.ERROR,
                code="PIPELINE_MISSING_DATA",
                message=f"Data key '{key}' is consumed but never produced",
                suggestion="Ensure a step produces this data or check key spelling"
            ))

        # Check for produced but never consumed keys
        unused_keys = produced_keys - consumed_keys
        for key in unused_keys:
            issues.append(ValidationIssue(
                type=IssueType.WARNING,
                code="PIPELINE_UNUSED_DATA",
                message=f"Data key '{key}' is produced but never consumed",
                suggestion="Consider if this data is needed or if a step is missing"
            ))

        return issues

    def _validate_optimizations(self, pipeline: PipelineGraph) -> List[ValidationIssue]:
        """Validate potential optimizations"""
        issues = []

        # Check for parallelization opportunities
        try:
            execution_order = pipeline.build_execution_order()

            for i, level in enumerate(execution_order):
                if len(level) > 1:
                    # Good parallelization
                    continue
                elif i > 0 and len(execution_order[i - 1]) > 1:
                    # Parallel level followed by sequential - potential optimization
                    issues.append(ValidationIssue(
                        type=IssueType.PERFORMANCE,
                        code="PIPELINE_SYNC_POINT",
                        message=f"Level {i + 1} synchronizes parallel execution",
                        suggestion="Consider if synchronization is necessary"
                    ))

        except Exception:
            # Skip optimization validation if we can't build execution order
            pass

        return issues

    def _calculate_quality_score(self, issues: List[ValidationIssue], pipeline: PipelineGraph) -> float:
        """Calculate pipeline quality score (0-100)"""
        base_score = 100.0

        # Deduct points for issues
        for issue in issues:
            if issue.type == IssueType.ERROR:
                base_score -= 20
            elif issue.type == IssueType.WARNING:
                base_score -= 5
            elif issue.type == IssueType.PERFORMANCE:
                base_score -= 2
            # INFO issues don't affect score

        # Bonus points for good practices
        total_steps = len(pipeline.steps)
        if total_steps > 0:
            # Bonus for having documentation
            documented_steps = 0
            typed_steps = 0

            for step_node in pipeline.steps.values():
                if inspect.getdoc(step_node.func):
                    documented_steps += 1

                signature = inspect.signature(step_node.func)
                if any(param.annotation != param.empty for param in signature.parameters.values()):
                    typed_steps += 1

            # Up to 10 bonus points for documentation
            doc_ratio = documented_steps / total_steps
            base_score += doc_ratio * 10

            # Up to 5 bonus points for type hints
            type_ratio = typed_steps / total_steps
            base_score += type_ratio * 5

        return max(0.0, min(100.0, base_score))


# Convenience functions
def validate_pipeline(
        pipeline: PipelineGraph,
        level: ValidationLevel = ValidationLevel.STANDARD
) -> ValidationResult:
    """Validate a pipeline with specified validation level"""
    validator = PipelineValidator(level)
    return validator.validate_pipeline(pipeline)


def quick_validate(pipeline: PipelineGraph) -> bool:
    """Quick validation - returns True if pipeline is valid"""
    result = validate_pipeline(pipeline, ValidationLevel.BASIC)
    return result.is_valid


def validate_step(step_node: StepNode) -> List[ValidationIssue]:
    """Validate a single step"""
    validator = StepValidator()
    return validator.validate_step(step_node)


# Custom validation rules
class ValidationRule:
    """Base class for custom validation rules"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def validate(self, pipeline: PipelineGraph) -> List[ValidationIssue]:
        """Implement custom validation logic"""
        raise NotImplementedError


class MLWorkflowRule(ValidationRule):
    """Validation rule for ML workflows"""

    def __init__(self):
        super().__init__(
            "ml_workflow",
            "Validates common ML workflow patterns"
        )

    def validate(self, pipeline: PipelineGraph) -> List[ValidationIssue]:
        issues = []
        step_names = list(pipeline.steps.keys())

        # Check for common ML steps
        has_data_step = any("data" in name.lower() or "load" in name.lower() for name in step_names)
        has_train_step = any("train" in name.lower() for name in step_names)
        has_eval_step = any("eval" in name.lower() or "validate" in name.lower() for name in step_names)

        if has_train_step and not has_eval_step:
            issues.append(ValidationIssue(
                type=IssueType.WARNING,
                code="ML_NO_EVALUATION",
                message="Training pipeline lacks evaluation step",
                suggestion="Add model evaluation for better ML practices"
            ))

        if not has_data_step and len(step_names) > 2:
            issues.append(ValidationIssue(
                type=IssueType.INFO,
                code="ML_NO_DATA_STEP",
                message="No explicit data loading step found",
                suggestion="Consider adding a dedicated data loading step"
            ))

        return issues


# Default validation rules
DEFAULT_RULES = [
    MLWorkflowRule()
]


def validate_with_rules(
        pipeline: PipelineGraph,
        rules: List[ValidationRule] = None,
        level: ValidationLevel = ValidationLevel.STANDARD
) -> ValidationResult:
    """Validate pipeline with custom rules"""
    if rules is None:
        rules = DEFAULT_RULES

    # Run standard validation
    result = validate_pipeline(pipeline, level)

    # Apply custom rules
    for rule in rules:
        try:
            rule_issues = rule.validate(pipeline)
            result.issues.extend(rule_issues)
        except Exception as e:
            result.issues.append(ValidationIssue(
                type=IssueType.WARNING,
                code="RULE_VALIDATION_FAILED",
                message=f"Custom rule '{rule.name}' failed: {str(e)}"
            ))

    # Recalculate validity
    errors = [issue for issue in result.issues if issue.type == IssueType.ERROR]
    result.is_valid = len(errors) == 0

    return result