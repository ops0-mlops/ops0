import threading
from typing import Dict, List, Optional
from collections import defaultdict

from src.ops0.core.analyzer import FunctionAnalyzer


class StepNode:
    """Represents a single step in the pipeline"""

    def __init__(self, metadata):
        self.metadata = metadata
        self.name = metadata.name
        self.func = metadata.func
        self.dependencies = metadata.dependencies
        self.dependents = set()
        self.executed = False
        self.result = None

    def __repr__(self):
        return f"StepNode(name='{self.name}', deps={self.dependencies})"


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

    def __enter__(self):
        """Context manager entry"""
        self._current_pipeline.value = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if hasattr(self._current_pipeline, 'value'):
            del self._current_pipeline.value

    @classmethod
    def get_current(cls) -> Optional['PipelineGraph']:
        """Get the current pipeline context"""
        return getattr(cls._current_pipeline, 'value', None)

    def add_step(self, step_node: StepNode):
        """Add a step to the pipeline"""
        self.steps[step_node.name] = step_node

        # Register what this step provides to storage
        analyzer = FunctionAnalyzer(step_node.func)
        for storage_key in analyzer.get_storage_saves():
            self.storage_providers[storage_key] = step_node.name

    def build_execution_order(self) -> List[List[str]]:
        """
        Build optimal execution order using topological sort.

        Returns:
            List of lists, where each inner list contains steps that can
            execute in parallel.
        """
        # Build dependency graph
        dependencies = defaultdict(set)
        dependents = defaultdict(set)

        for step_name, step_node in self.steps.items():
            for storage_key in step_node.dependencies:
                if storage_key in self.storage_providers:
                    provider_step = self.storage_providers[storage_key]
                    dependencies[step_name].add(provider_step)
                    dependents[provider_step].add(step_name)

        # Topological sort with parallel execution detection
        in_degree = {step: len(dependencies[step]) for step in self.steps}
        execution_order = []

        while in_degree:
            # Find all steps with no dependencies (can execute now)
            ready_steps = [step for step, degree in in_degree.items() if degree == 0]

            if not ready_steps:
                raise ValueError("Circular dependency detected in pipeline")

            execution_order.append(ready_steps)

            # Remove ready steps and update in-degrees
            for step in ready_steps:
                del in_degree[step]
                for dependent in dependents[step]:
                    if dependent in in_degree:
                        in_degree[dependent] -= 1

        return execution_order

    def get_step_dependencies(self, step_name: str) -> List[str]:
        """Get direct step dependencies for a given step"""
        step = self.steps[step_name]
        deps = []
        for storage_key in step.dependencies:
            if storage_key in self.storage_providers:
                deps.append(self.storage_providers[storage_key])
        return deps

    def visualize(self) -> str:
        """Generate ASCII visualization of the pipeline"""
        execution_order = self.build_execution_order()

        viz = [f"Pipeline: {self.name}", "=" * (len(self.name) + 10)]

        for level, parallel_steps in enumerate(execution_order):
            viz.append(f"\nLevel {level + 1}:")
            for step in parallel_steps:
                deps = self.get_step_dependencies(step)
                dep_str = f" (depends on: {', '.join(deps)})" if deps else ""
                viz.append(f"  ├─ {step}{dep_str}")

        return "\n".join(viz)


def get_current_pipeline() -> Optional[PipelineGraph]:
    """Get the current pipeline context"""
    return PipelineGraph.get_current()