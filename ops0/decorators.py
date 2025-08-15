"""
Core decorators for ops0 - @step and @pipeline
Zero configuration approach with intelligent defaults
"""
import functools
import inspect
import uuid
from typing import Any, Callable, Dict, Optional, List
from dataclasses import dataclass, field
import cloudpickle

from ops0.parser import analyze_function, FunctionAnalysis
from ops0.executor import ExecutionContext


@dataclass
class StepConfig:
    """Configuration for a single step"""
    name: str
    func: Callable
    memory: int = 512  # MB
    timeout: int = 300  # seconds
    gpu: bool = False
    retries: int = 3
    requirements: List[str] = field(default_factory=list)
    container_image: Optional[str] = None
    analysis: Optional[FunctionAnalysis] = None

    def __post_init__(self):
        # Analyze function to detect requirements automatically
        if self.analysis is None:
            self.analysis = analyze_function(self.func)

        # Apply intelligent defaults based on analysis
        if self.analysis.uses_ml_framework and self.memory < 2048:
            self.memory = 2048
        if self.analysis.uses_gpu and not self.gpu:
            self.gpu = True
        if self.analysis.estimated_requirements:
            self.requirements.extend(self.analysis.estimated_requirements)


@dataclass
class PipelineConfig:
    """Configuration for a pipeline"""
    name: str
    func: Callable
    steps: Dict[str, StepConfig] = field(default_factory=dict)
    schedule: Optional[str] = None
    description: Optional[str] = None


class StepDecorator:
    """Registry for all decorated steps"""
    _steps: Dict[str, StepConfig] = {}

    @classmethod
    def register(cls, config: StepConfig):
        cls._steps[config.name] = config

    @classmethod
    def get_all(cls) -> Dict[str, StepConfig]:
        return cls._steps.copy()

    @classmethod
    def get(cls, name: str) -> Optional[StepConfig]:
        return cls._steps.get(name)


class PipelineDecorator:
    """Registry for all pipelines"""
    _pipelines: Dict[str, PipelineConfig] = {}

    @classmethod
    def register(cls, config: PipelineConfig):
        cls._pipelines[config.name] = config

    @classmethod
    def get_all(cls) -> Dict[str, PipelineConfig]:
        return cls._pipelines.copy()


def step(
        func: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        memory: Optional[int] = None,
        timeout: Optional[int] = None,
        gpu: Optional[bool] = None,
        retries: Optional[int] = None,
        requirements: Optional[List[str]] = None,
        container_image: Optional[str] = None
) -> Callable:
    """
    Decorator to mark a function as an ops0 step.

    Usage:
        @ops0.step
        def preprocess(data):
            return data.dropna()

        @ops0.step(memory=2048, gpu=True)
        def train_model(data):
            # GPU-intensive training
            pass
    """

    def decorator(f: Callable) -> Callable:
        if not callable(f):
            raise TypeError(f"@ops0.step can only decorate callable objects, got {type(f)}")

        # Extract step name from function name if not provided
        step_name = name or f.__name__

        # Create step configuration with provided overrides
        config = StepConfig(
            name=step_name,
            func=f,
            memory=memory or 512,
            timeout=timeout or 300,
            gpu=gpu or False,
            retries=retries or 3,
            requirements=requirements or [],
            container_image=container_image
        )

        # Register the step
        StepDecorator.register(config)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # In local mode, just execute the function
            ctx = ExecutionContext.current()
            if ctx and ctx.mode == "local":
                return f(*args, **kwargs)

            # In production mode, this would be handled by the orchestrator
            # For now, execute directly
            return f(*args, **kwargs)

        # Attach metadata for introspection
        wrapper._ops0_step = config
        wrapper._ops0_type = "step"

        return wrapper

    # Handle both @step and @step() syntax
    if func is None:
        return decorator
    else:
        return decorator(func)


def pipeline(
        func: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        schedule: Optional[str] = None,
        description: Optional[str] = None
) -> Callable:
    """
    Decorator to mark a function as an ops0 pipeline.

    Usage:
        @ops0.pipeline
        def ml_workflow(input_path):
            data = load_data(input_path)
            processed = preprocess(data)
            return train_model(processed)

        @ops0.pipeline(schedule="0 */2 * * *")  # Every 2 hours
        def scheduled_pipeline():
            pass
    """

    def decorator(f: Callable) -> Callable:
        pipeline_name = name or f.__name__

        # Analyze the pipeline function to build DAG
        analysis = analyze_function(f)

        # Create pipeline configuration
        config = PipelineConfig(
            name=pipeline_name,
            func=f,
            schedule=schedule,
            description=description or f.__doc__
        )

        # Find all steps used in this pipeline
        for step_name in analysis.called_functions:
            step_config = StepDecorator.get(step_name)
            if step_config:
                config.steps[step_name] = step_config

        # Register the pipeline
        PipelineDecorator.register(config)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Execute pipeline
            ctx = ExecutionContext.current()
            if ctx:
                ctx.pipeline_name = pipeline_name

            result = f(*args, **kwargs)
            return result

        # Attach metadata
        wrapper._ops0_pipeline = config
        wrapper._ops0_type = "pipeline"

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


# Convenience functions for monitoring and alerts
def monitor(*, alert_on_latency: Optional[str] = None, alert_on_error: bool = True):
    """Add monitoring to a step"""

    def decorator(func):
        # Add monitoring metadata
        if hasattr(func, '_ops0_step'):
            func._ops0_step.monitoring = {
                'alert_on_latency': alert_on_latency,
                'alert_on_error': alert_on_error
            }
        return func

    return decorator