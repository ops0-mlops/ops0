import functools
from typing import Callable, Optional
from .analyzer import FunctionAnalyzer
from .graph import PipelineGraph, StepNode


class StepMetadata:
    """Metadata extracted from a step function"""

    def __init__(self, func: Callable):
        self.name = func.__name__
        self.func = func
        self.analyzer = FunctionAnalyzer(func)
        self.dependencies = self.analyzer.get_dependencies()
        self.inputs = self.analyzer.get_input_signature()
        self.outputs = self.analyzer.get_output_signature()
        self.source_hash = self.analyzer.get_source_hash()

    def __repr__(self):
        return f"StepMetadata(name='{self.name}', deps={self.dependencies})"


def step(func: Callable) -> Callable:
    """
    Transform a Python function into an ops0 pipeline step.

    Automatically analyzes the function to detect:
    - Input dependencies from ops0.storage.load() calls
    - Function signature and type hints
    - Source code for reproducibility

    Args:
        func: Function to transform into a step

    Returns:
        Wrapped function with ops0 metadata

    Example:
        @ops0.step
        def preprocess(data):
            cleaned = data.dropna()
            ops0.storage.save("clean_data", cleaned)
            return cleaned
    """

    # Extract step metadata using AST analysis
    metadata = StepMetadata(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Register step in current pipeline
        current_pipeline = PipelineGraph.get_current()
        if current_pipeline:
            step_node = StepNode(metadata)
            current_pipeline.add_step(step_node)

        # Execute the original function
        return func(*args, **kwargs)

    # Attach metadata to function
    wrapper._ops0_metadata = metadata
    wrapper._ops0_step = True
    wrapper._original_func = func

    return wrapper


def pipeline(name: Optional[str] = None):
    """
    Define a pipeline context for grouping steps.

    Args:
        name: Optional pipeline name

    Example:
        with ops0.pipeline("fraud-detection"):
            @ops0.step
            def preprocess():
                pass

            @ops0.step
            def predict():
                pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            pipeline_name = name or func.__name__

            with PipelineGraph(pipeline_name):
                return func(*args, **kwargs)

        wrapper._ops0_pipeline = True
        return wrapper

    if callable(name):
        # Used as @pipeline without parentheses
        func = name
        name = None
        return decorator(func)
    else:
        # Used as @pipeline("name") or @pipeline()
        return decorator