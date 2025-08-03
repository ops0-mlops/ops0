"""
ops0 Pipeline Graph - Builds and manages execution graphs for pipelines.

Automatically detects dependencies through AST analysis and creates
optimal execution plans with topological sorting.
"""

import threading
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque
import logging

from .analyzer import FunctionAnalyzer
from .exceptions import DependencyError, PipelineError

logger = logging.getLogger(__name__)


class StepNode:
    """Represents a single step in the pipeline"""

    def __init__(self, metadata):
        self.metadata = metadata
        self.name = metadata.name
        self.func = metadata.func
        self.dependencies = metadata.dependencies
        self.dependents: Set[str] = set()
        self.executed = False
        self.result = None
        self.execution_time = 0.0

    def __repr__(self):
        return f"StepNode(name='{self.name}', deps={self.dependencies})"

    def add_dependent(self, step_name: str):
        """Add a dependent step"""
        self.dependents.add(step_name)

    def is_ready(self, completed_steps: Set[str]) -> bool:
        """Check if step is ready to execute"""
        return self.dependencies.issubset(completed_steps)

    def reset(self):
        """Reset execution state"""
        self.executed = False
        self.result = None
        self.execution_time = 0.0


class PipelineGraph:
    """
    Builds and manages the execution graph for a pipeline.

    Automatically detects dependencies through AST analysis and creates
    an optimal execution plan.
    """

    _current_pipeline = threading.local()

    def __init__(self, name: str):
        self.name = name
        self.steps: Dict[str, StepNode] = {}
        self.storage_providers: Dict[str, str] = {}  # storage_key -> step_name
        self._execution_order: Optional[List[List[str]]] = None
        self._validated = False

    def __enter__(self):
        """Context manager entry"""
        self._current_pipeline.value = self
        logger.debug(f"Entered pipeline context: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if hasattr(self._current_pipeline, 'value'):
            del self._current_pipeline.value
        logger.debug(f"Exited pipeline context: {self.name}")

    @classmethod
    def get_current(cls) -> Optional['PipelineGraph']:
        """Get the current pipeline context"""
        return getattr(cls._current_pipeline, 'value', None)

    def add_step(self, step_node: StepNode):
        """Add a step to the pipeline"""
        if step_node.name in self.steps:
            logger.warning(f"Step '{step_node.name}' already exists, replacing")

        self.steps[step_node.name] = step_node

        # Register what this step provides to storage
        analyzer = FunctionAnalyzer(step_node.func)
        for storage_key in analyzer.get_storage_saves():
            if storage_key in self.storage_providers:
                existing_provider = self.storage_providers[storage_key]
                logger.warning(
                    f"Storage key '{storage_key}' provided by both "
                    f"'{existing_provider}' and '{step_node.name}'"
                )
            self.storage_providers[storage_key] = step_node.name

        # Invalidate cached execution order
        self._execution_order = None
        self._validated = False

        logger.debug(f"Added step: {step_node.name}")

    def remove_step(self, step_name: str) -> bool:
        """Remove a step from the pipeline"""
        if step_name not in self.steps:
            return False

        # Remove from steps
        step_node = self.steps.pop(step_name)

        # Remove from storage providers
        to_remove = [k for k, v in self.storage_providers.items() if v == step_name]
        for key in to_remove:
            del self.storage_providers[key]

        # Update dependencies of other steps
        for other_step in self.steps.values():
            other_step.dependents.discard(step_name)

        # Invalidate cached execution order
        self._execution_order = None
        self._validated = False

        logger.debug(f"Removed step: {step_name}")
        return True

    def get_step(self, step_name: str) -> Optional[StepNode]:
        """Get a step by name"""
        return self.steps.get(step_name)

    def list_steps(self) -> List[str]:
        """List all step names"""
        return list(self.steps.keys())

    def validate(self) -> List[str]:
        """
        Validate the pipeline structure.

        Returns:
            List of validation errors
        """
        errors = []

        # Check for missing dependencies
        for step_name, step_node in self.steps.items():
            for dep in step_node.dependencies:
                if dep not in self.storage_providers:
                    errors.append(
                        f"Step '{step_name}' depends on '{dep}' but no step provides it"
                    )

        # Check for cycles
        try:
            self.build_execution_order()
        except DependencyError as e:
            errors.append(str(e))

        self._validated = len(errors) == 0
        return errors

    def build_execution_order(self) -> List[List[str]]:
        """
        Build optimal execution order using topological sort.

        Returns:
            List of lists, where each inner list contains steps that can
            execute in parallel.
        """
        if self._execution_order is not None:
            return self._execution_order

        if not self.steps:
            self._execution_order = []
            return self._execution_order

        # Build dependency graph based on storage dependencies
        dependency_graph = defaultdict(set)
        reverse_dependency_graph = defaultdict(set)

        for step_name, step_node in self.steps.items():
            # Add dependencies based on storage keys
            for storage_key in step_node.dependencies:
                if storage_key in self.storage_providers:
                    provider = self.storage_providers[storage_key]
                    if provider != step_name:  # Avoid self-dependency
                        dependency_graph[step_name].add(provider)
                        reverse_dependency_graph[provider].add(step_name)
                        # Update step node dependents
                        if provider in self.steps:
                            self.steps[provider].add_dependent(step_name)

        # Topological sort with Kahn's algorithm
        execution_order = []
        in_degree = defaultdict(int)

        # Calculate in-degrees
        for step_name in self.steps:
            in_degree[step_name] = len(dependency_graph[step_name])

        # Process steps level by level
        while True:
            # Find all steps with in-degree 0 (ready to execute)
            ready_steps = [
                step_name for step_name in self.steps
                if in_degree[step_name] == 0 and step_name not in [
                    s for batch in execution_order for s in batch
                ]
            ]

            if not ready_steps:
                break

            execution_order.append(ready_steps)

            # Update in-degrees
            for step_name in ready_steps:
                for dependent in reverse_dependency_graph[step_name]:
                    in_degree[dependent] -= 1

        # Check for cycles
        total_steps_in_order = sum(len(batch) for batch in execution_order)
        if total_steps_in_order != len(self.steps):
            remaining_steps = set(self.steps.keys()) - {
                s for batch in execution_order for s in batch
            }
            raise DependencyError(
                f"Circular dependency detected involving steps: {remaining_steps}"
            )

        self._execution_order = execution_order
        logger.debug(f"Built execution order: {execution_order}")
        return execution_order

    def get_step_dependencies(self, step_name: str) -> Set[str]:
        """Get all direct and indirect dependencies of a step"""
        if step_name not in self.steps:
            return set()

        visited = set()
        to_visit = deque([step_name])
        dependencies = set()

        while to_visit:
            current = to_visit.popleft()
            if current in visited:
                continue
            visited.add(current)

            if current in self.steps:
                step_deps = self.steps[current].dependencies
                for dep in step_deps:
                    if dep in self.storage_providers:
                        provider = self.storage_providers[dep]
                        if provider not in visited:
                            dependencies.add(provider)
                            to_visit.append(provider)

        return dependencies

    def get_step_dependents(self, step_name: str) -> Set[str]:
        """Get all direct and indirect dependents of a step"""
        if step_name not in self.steps:
            return set()

        visited = set()
        to_visit = deque([step_name])
        dependents = set()

        while to_visit:
            current = to_visit.popleft()
            if current in visited:
                continue
            visited.add(current)

            if current in self.steps:
                step_dependents = self.steps[current].dependents
                for dependent in step_dependents:
                    if dependent not in visited:
                        dependents.add(dependent)
                        to_visit.append(dependent)

        return dependents

    def reset_execution_state(self):
        """Reset execution state for all steps"""
        for step in self.steps.values():
            step.reset()
        logger.debug("Reset execution state for all steps")

    def get_execution_stats(self) -> Dict[str, any]:
        """Get execution statistics"""
        total_steps = len(self.steps)
        completed_steps = sum(1 for step in self.steps.values() if step.executed)
        total_time = sum(step.execution_time for step in self.steps.values())

        return {
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "completion_percentage": (completed_steps / total_steps * 100) if total_steps > 0 else 0,
            "total_execution_time": total_time,
            "average_step_time": total_time / completed_steps if completed_steps > 0 else 0,
        }

    def to_dict(self) -> Dict[str, any]:
        """Convert pipeline to dictionary representation"""
        return {
            "name": self.name,
            "steps": {
                name: {
                    "name": step.name,
                    "dependencies": list(step.dependencies),
                    "dependents": list(step.dependents),
                    "executed": step.executed,
                }
                for name, step in self.steps.items()
            },
            "storage_providers": self.storage_providers,
            "execution_order": self._execution_order,
            "validated": self._validated,
        }

    def __len__(self):
        return len(self.steps)

    def __contains__(self, step_name: str):
        return step_name in self.steps

    def __iter__(self):
        return iter(self.steps.values())


def get_current_pipeline() -> Optional[PipelineGraph]:
    """Get the current pipeline context (alias for PipelineGraph.get_current)"""
    return PipelineGraph.get_current()


def create_pipeline(name: str) -> PipelineGraph:
    """Create a new pipeline instance"""
    return PipelineGraph(name)