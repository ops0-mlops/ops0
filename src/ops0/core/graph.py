"""
ops0 Pipeline Graph

Builds and manages the execution graph of pipeline steps.
Automatically resolves dependencies and enables parallel execution.
"""

import threading
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

from .analyzer import FunctionAnalyzer, StorageDependency
from .exceptions import DependencyError, PipelineError, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class StepNode:
    """Represents a single step in the pipeline graph"""
    name: str
    func: Callable
    analyzer: FunctionAnalyzer
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    result: Any = None
    error: Optional[Exception] = None
    execution_time: Optional[float] = None

    def __post_init__(self):
        if not self.analyzer:
            self.analyzer = FunctionAnalyzer(self.func)

    @property
    def storage_dependencies(self) -> List[StorageDependency]:
        """Get storage dependencies for this step"""
        return self.analyzer.get_dependencies()

    @property
    def input_signature(self):
        """Get input signature for this step"""
        return self.analyzer.get_input_signature()

    def can_execute(self, completed_steps: Set[str]) -> bool:
        """Check if this step can be executed given completed steps"""
        return all(dep in completed_steps for dep in self.dependencies)

    def __repr__(self):
        return f"StepNode(name='{self.name}', status='{self.status}', deps={self.dependencies})"


class PipelineGraph:
    """Manages the pipeline execution graph"""

    # Thread-local storage for current pipeline context
    _current_pipeline = threading.local()

    def __init__(self, name: str = "default-pipeline"):
        self.name = name
        self.steps: Dict[str, StepNode] = {}
        self.execution_order: List[str] = []
        self._validated = False
        self._dependency_graph_built = False

    def add_step(self, step_node: StepNode) -> None:
        """Add a step to the pipeline"""
        if step_node.name in self.steps:
            logger.warning(f"Step '{step_node.name}' already exists, replacing")

        self.steps[step_node.name] = step_node
        self._validated = False
        self._dependency_graph_built = False

        logger.debug(f"Added step '{step_node.name}' to pipeline '{self.name}'")

    def build_dependency_graph(self) -> None:
        """Build the dependency graph between steps"""
        if self._dependency_graph_built:
            return

        # Reset dependencies
        for step in self.steps.values():
            step.dependencies = []
            step.dependents = []

        # Build dependencies based on storage operations
        for step_name, step in self.steps.items():
            storage_deps = step.storage_dependencies

            for dep in storage_deps:
                if dep.operation == 'load':
                    # Find which step saves this key
                    producer_step = self._find_producer_step(dep.key, step_name)
                    if producer_step:
                        step.dependencies.append(producer_step)
                        self.steps[producer_step].dependents.append(step_name)

        # Also consider function parameter dependencies
        self._analyze_parameter_dependencies()

        self._dependency_graph_built = True
        logger.debug(f"Built dependency graph for pipeline '{self.name}'")

    def _find_producer_step(self, key: str, consumer_step: str) -> Optional[str]:
        """Find which step produces a given storage key"""
        for step_name, step in self.steps.items():
            if step_name == consumer_step:
                continue

            storage_deps = step.storage_dependencies
            for dep in storage_deps:
                if dep.operation == 'save' and dep.key == key:
                    return step_name

        return None

    def _analyze_parameter_dependencies(self) -> None:
        """Analyze function parameter names to infer dependencies"""
        # This is a simplified heuristic - in practice, we'd use more sophisticated analysis
        step_names = set(self.steps.keys())

        for step_name, step in self.steps.items():
            signature = step.input_signature

            for param_name in signature.parameters:
                # If parameter name matches another step name, create dependency
                if param_name in step_names and param_name != step_name:
                    if param_name not in step.dependencies:
                        step.dependencies.append(param_name)
                        self.steps[param_name].dependents.append(step_name)

    def validate(self) -> bool:
        """Validate the pipeline graph"""
        if self._validated:
            return True

        if not self.steps:
            raise ValidationError("Pipeline has no steps")

        # Build dependency graph first
        self.build_dependency_graph()

        # Check for circular dependencies
        self._check_circular_dependencies()

        # Validate individual steps
        for step in self.steps.values():
            step.analyzer.validate_step_function()

        # Validate storage consistency
        self._validate_storage_consistency()

        self._validated = True
        logger.info(f"Pipeline '{self.name}' validated successfully")
        return True

    def _check_circular_dependencies(self) -> None:
        """Check for circular dependencies in the graph"""
        visited = set()
        rec_stack = set()

        def has_cycle(step_name: str) -> bool:
            visited.add(step_name)
            rec_stack.add(step_name)

            for dependent in self.steps[step_name].dependents:
                if dependent not in visited:
                    if has_cycle(dependent):
                        return True
                elif dependent in rec_stack:
                    return True

            rec_stack.remove(step_name)
            return False

        for step_name in self.steps:
            if step_name not in visited:
                if has_cycle(step_name):
                    # Find the actual cycle for better error message
                    cycle = self._find_cycle()
                    raise DependencyError(
                        f"Circular dependency detected in pipeline '{self.name}'",
                        dependency=" -> ".join(cycle)
                    )

    def _find_cycle(self) -> List[str]:
        """Find an actual cycle in the dependency graph"""
        # Simplified cycle detection - returns first found cycle
        visited = set()

        def dfs(step_name: str, path: List[str]) -> Optional[List[str]]:
            if step_name in path:
                cycle_start = path.index(step_name)
                return path[cycle_start:] + [step_name]

            if step_name in visited:
                return None

            visited.add(step_name)

            for dependent in self.steps[step_name].dependents:
                result = dfs(dependent, path + [step_name])
                if result:
                    return result

            return None

        for step_name in self.steps:
            result = dfs(step_name, [])
            if result:
                return result

        return []

    def _validate_storage_consistency(self) -> None:
        """Validate storage key consistency across steps"""
        produced_keys = set()
        consumed_keys = set()

        for step in self.steps.values():
            for dep in step.storage_dependencies:
                if dep.operation == 'save':
                    if dep.key in produced_keys:
                        logger.warning(f"Storage key '{dep.key}' is produced by multiple steps")
                    produced_keys.add(dep.key)
                elif dep.operation == 'load':
                    consumed_keys.add(dep.key)

        # Check for consumed keys that are never produced
        missing_keys = consumed_keys - produced_keys
        if missing_keys:
            raise ValidationError(
                f"Storage keys consumed but never produced: {missing_keys}",
                context={"missing_keys": list(missing_keys)}
            )

    def get_execution_order(self) -> List[str]:
        """Get the topological order for step execution"""
        if not self._validated:
            self.validate()

        if self.execution_order:
            return self.execution_order

        # Topological sort using Kahn's algorithm
        in_degree = {step_name: len(step.dependencies) for step_name, step in self.steps.items()}
        queue = deque([step_name for step_name, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            step_name = queue.popleft()
            result.append(step_name)

            for dependent in self.steps[step_name].dependents:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self.steps):
            raise DependencyError("Could not resolve all dependencies (possible cycle)")

        self.execution_order = result
        return result

    def get_parallel_groups(self) -> List[List[str]]:
        """Get groups of steps that can be executed in parallel"""
        execution_order = self.get_execution_order()
        groups = []
        remaining_steps = set(execution_order)

        while remaining_steps:
            # Find all steps that can execute now (no pending dependencies)
            executable_steps = []
            for step_name in remaining_steps:
                step = self.steps[step_name]
                completed_deps = set(execution_order) - remaining_steps
                if step.can_execute(completed_deps):
                    executable_steps.append(step_name)

            if not executable_steps:
                raise DependencyError("No executable steps found - possible dependency issue")

            groups.append(executable_steps)
            remaining_steps -= set(executable_steps)

        return groups

    def get_step_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all steps in the pipeline"""
        metadata = {}

        for step_name, step in self.steps.items():
            metadata[step_name] = {
                "dependencies": step.dependencies,
                "dependents": step.dependents,
                "storage_dependencies": [
                    {"key": dep.key, "operation": dep.operation, "line": dep.line_number}
                    for dep in step.storage_dependencies
                ],
                "signature": str(step.input_signature),
                "resource_requirements": step.analyzer.get_resource_requirements(),
                "serialization_hints": step.analyzer.get_serialization_hints(),
            }

        return metadata

    def reset_execution_state(self) -> None:
        """Reset all steps to pending state"""
        for step in self.steps.values():
            step.status = "pending"
            step.result = None
            step.error = None
            step.execution_time = None

    # Context manager support
    def __enter__(self):
        """Enter pipeline context"""
        PipelineGraph._current_pipeline.value = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit pipeline context"""
        if hasattr(PipelineGraph._current_pipeline, 'value'):
            delattr(PipelineGraph._current_pipeline, 'value')

    @classmethod
    def get_current(cls) -> Optional['PipelineGraph']:
        """Get the current pipeline from thread-local storage"""
        return getattr(cls._current_pipeline, 'value', None)

    def __repr__(self):
        return f"PipelineGraph(name='{self.name}', steps={len(self.steps)})"


# Convenience function for getting current pipeline
def get_current_pipeline() -> Optional[PipelineGraph]:
    """Get the current pipeline context"""
    return PipelineGraph.get_current()


# Pipeline builder utility
class PipelineBuilder:
    """Helper class for building pipelines programmatically"""

    def __init__(self, name: str = "pipeline"):
        self.pipeline = PipelineGraph(name)

    def add_step(self, func: Callable, name: str = None) -> 'PipelineBuilder':
        """Add a step to the pipeline"""
        step_name = name or func.__name__
        analyzer = FunctionAnalyzer(func)
        step_node = StepNode(step_name, func, analyzer)
        self.pipeline.add_step(step_node)
        return self

    def build(self) -> PipelineGraph:
        """Build and validate the pipeline"""
        self.pipeline.validate()
        return self.pipeline