import functools
from typing import Callable, Optional
from .analyzer import FunctionAnalyzer
from .graph import PipelineGraph, StepNode, logger


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
            step_node = StepNode(metadata.name, metadata.func, metadata.analyzer)
            current_pipeline.add_step(step_node)

        # Execute the original function
        return func(*args, **kwargs)

    # Attach metadata to function
    wrapper._ops0_metadata = metadata
    wrapper._ops0_step = True
    wrapper._original_func = func

    return wrapper


def pipeline(name: Optional[str] = None):
    if name is not None and not isinstance(name, str):
        raise ValueError("Le nom du pipeline doit être une chaîne de caractères")
        
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            pipeline_name = name or func.__name__
            current = PipelineGraph.get_current()
            
            if current:
                # Déjà dans un contexte de pipeline
                return func(*args, **kwargs)
                
            try:
                with PipelineGraph(pipeline_name) as pipeline:
                    result = func(*args, **kwargs)
                    return result
            except Exception as e:
                logger.error(f"Erreur dans le pipeline {pipeline_name}: {e}")
                raise
                
        wrapper._ops0_pipeline = True
        return wrapper
        
    if callable(name):
        func = name
        name = None
        return decorator(func)
    return decorator