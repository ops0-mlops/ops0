"""
ops0 Core Data Models

Defines the core data structures and models used throughout ops0.
These models represent pipelines, steps, executions, and other core concepts.
"""

import uuid
import time
from datetime import datetime, timezone
from typing import Dict, List, Set, Any, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib


class StepStatus(Enum):
    """Status of a pipeline step"""
    PENDING = "pending"  # Not yet started
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Failed with error
    SKIPPED = "skipped"  # Skipped due to condition
    RETRYING = "retrying"  # Retrying after failure
    CANCELLED = "cancelled"  # Cancelled by user


class PipelineStatus(Enum):
    """Status of a pipeline execution"""
    CREATED = "created"  # Pipeline created but not yet started
    RUNNING = "running"  # Pipeline is executing
    COMPLETED = "completed"  # All steps completed successfully
    FAILED = "failed"  # Pipeline failed
    CANCELLED = "cancelled"  # Pipeline was cancelled
    PAUSED = "paused"  # Pipeline execution paused


class ExecutionMode(Enum):
    """Pipeline execution modes"""
    LOCAL = "local"  # Execute locally
    DISTRIBUTED = "distributed"  # Execute in distributed mode
    DRY_RUN = "dry_run"  # Validate without executing
    DEBUG = "debug"  # Execute with debug info


class ResourceType(Enum):
    """Types of resources that can be allocated"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class ResourceRequirements:
    """Resource requirements for a step"""
    cpu_cores: Optional[float] = None  # CPU cores (e.g., 1.5)
    memory_mb: Optional[int] = None  # Memory in MB
    gpu_count: Optional[int] = None  # Number of GPUs
    storage_mb: Optional[int] = None  # Temporary storage in MB
    network_mbps: Optional[int] = None  # Network bandwidth in Mbps
    timeout_seconds: Optional[int] = None  # Execution timeout

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "gpu_count": self.gpu_count,
            "storage_mb": self.storage_mb,
            "network_mbps": self.network_mbps,
            "timeout_seconds": self.timeout_seconds
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResourceRequirements':
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if v is not None})

    def merge(self, other: 'ResourceRequirements') -> 'ResourceRequirements':
        """Merge with another ResourceRequirements, taking the maximum"""
        return ResourceRequirements(
            cpu_cores=max(self.cpu_cores or 0, other.cpu_cores or 0) or None,
            memory_mb=max(self.memory_mb or 0, other.memory_mb or 0) or None,
            gpu_count=max(self.gpu_count or 0, other.gpu_count or 0) or None,
            storage_mb=max(self.storage_mb or 0, other.storage_mb or 0) or None,
            network_mbps=max(self.network_mbps or 0, other.network_mbps or 0) or None,
            timeout_seconds=max(self.timeout_seconds or 0, other.timeout_seconds or 0) or None
        )


@dataclass
class StepExecution:
    """Represents the execution of a single step"""
    step_name: str
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    result: Any = None
    attempt: int = 1
    max_attempts: int = 3
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    resources_used: Optional[ResourceRequirements] = None

    @property
    def duration(self) -> Optional[float]:
        """Execution duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def is_finished(self) -> bool:
        """Check if execution is finished (completed, failed, or cancelled)"""
        return self.status in [StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.CANCELLED]

    @property
    def can_retry(self) -> bool:
        """Check if step can be retried"""
        return (self.status == StepStatus.FAILED and
                self.attempt < self.max_attempts)

    def start(self):
        """Mark step as started"""
        self.status = StepStatus.RUNNING
        self.start_time = datetime.now(timezone.utc)

    def complete(self, result: Any = None):
        """Mark step as completed"""
        self.status = StepStatus.COMPLETED
        self.end_time = datetime.now(timezone.utc)
        self.result = result

    def fail(self, error: str, error_type: str = None):
        """Mark step as failed"""
        self.status = StepStatus.FAILED
        self.end_time = datetime.now(timezone.utc)
        self.error = error
        self.error_type = error_type or "UnknownError"

    def retry(self):
        """Prepare step for retry"""
        if self.can_retry:
            self.attempt += 1
            self.status = StepStatus.RETRYING
            self.start_time = None
            self.end_time = None
            self.error = None
            self.error_type = None

    def add_log(self, message: str, level: str = "INFO"):
        """Add a log message"""
        timestamp = datetime.now(timezone.utc).isoformat()
        self.logs.append(f"[{timestamp}] {level}: {message}")

    def add_metric(self, name: str, value: Any):
        """Add a metric"""
        self.metrics[name] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "step_name": self.step_name,
            "execution_id": self.execution_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "error": self.error,
            "error_type": self.error_type,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "logs": self.logs,
            "metrics": self.metrics,
            "resources_used": self.resources_used.to_dict() if self.resources_used else None
        }


@dataclass
class PipelineExecution:
    """Represents the execution of an entire pipeline"""
    pipeline_name: str
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: PipelineStatus = PipelineStatus.CREATED
    mode: ExecutionMode = ExecutionMode.LOCAL
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    step_executions: Dict[str, StepExecution] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    created_by: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @property
    def duration(self) -> Optional[float]:
        """Total pipeline execution time in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def progress(self) -> Dict[str, int]:
        """Get execution progress"""
        if not self.step_executions:
            return {"completed": 0, "total": 0, "percentage": 0}

        total = len(self.step_executions)
        completed = sum(1 for exec in self.step_executions.values()
                        if exec.status == StepStatus.COMPLETED)
        percentage = int((completed / total) * 100) if total > 0 else 0

        return {
            "completed": completed,
            "total": total,
            "percentage": percentage
        }

    @property
    def failed_steps(self) -> List[str]:
        """Get list of failed step names"""
        return [name for name, exec in self.step_executions.items()
                if exec.status == StepStatus.FAILED]

    @property
    def running_steps(self) -> List[str]:
        """Get list of currently running step names"""
        return [name for name, exec in self.step_executions.items()
                if exec.status == StepStatus.RUNNING]

    def start(self):
        """Mark pipeline as started"""
        self.status = PipelineStatus.RUNNING
        self.start_time = datetime.now(timezone.utc)

    def complete(self):
        """Mark pipeline as completed"""
        self.status = PipelineStatus.COMPLETED
        self.end_time = datetime.now(timezone.utc)

    def fail(self):
        """Mark pipeline as failed"""
        self.status = PipelineStatus.FAILED
        self.end_time = datetime.now(timezone.utc)

    def cancel(self):
        """Cancel pipeline execution"""
        self.status = PipelineStatus.CANCELLED
        self.end_time = datetime.now(timezone.utc)

        # Cancel running steps
        for step_exec in self.step_executions.values():
            if step_exec.status == StepStatus.RUNNING:
                step_exec.status = StepStatus.CANCELLED
                step_exec.end_time = datetime.now(timezone.utc)

    def add_step_execution(self, step_execution: StepExecution):
        """Add a step execution"""
        self.step_executions[step_execution.step_name] = step_execution

    def get_step_execution(self, step_name: str) -> Optional[StepExecution]:
        """Get step execution by name"""
        return self.step_executions.get(step_name)

    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        progress = self.progress

        return {
            "execution_id": self.execution_id,
            "pipeline_name": self.pipeline_name,
            "status": self.status.value,
            "mode": self.mode.value,
            "duration": self.duration,
            "progress": progress,
            "failed_steps": self.failed_steps,
            "running_steps": self.running_steps,
            "total_steps": len(self.step_executions),
            "created_by": self.created_by,
            "tags": self.tags
        }


@dataclass
class StepDefinition:
    """Definition of a pipeline step"""
    name: str
    function: Callable
    dependencies: Set[str] = field(default_factory=set)
    outputs: Set[str] = field(default_factory=set)
    resource_requirements: Optional[ResourceRequirements] = None
    retry_config: Dict[str, Any] = field(default_factory=dict)
    cache_config: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[Callable] = None  # Conditional execution
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def signature_hash(self) -> str:
        """Generate hash of step signature for caching/versioning"""
        import inspect
        source = inspect.getsource(self.function)
        content = f"{self.name}:{source}:{sorted(self.dependencies)}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def can_execute(self, context: Dict[str, Any]) -> bool:
        """Check if step can execute based on condition"""
        if self.condition is None:
            return True
        try:
            return bool(self.condition(context))
        except Exception:
            return False

    def estimate_resources(self) -> ResourceRequirements:
        """Estimate resource requirements if not explicitly set"""
        if self.resource_requirements:
            return self.resource_requirements

        # Simple heuristics based on function analysis
        import inspect
        try:
            source = inspect.getsource(self.function).lower()

            # Estimate CPU
            cpu_cores = 1.0
            if 'parallel' in source or 'multiprocess' in source:
                cpu_cores = 2.0

            # Estimate memory
            memory_mb = 512  # Default
            if 'dataframe' in source or 'pandas' in source:
                memory_mb = 2048
            if 'large' in source or 'big' in source:
                memory_mb = 4096

            # Estimate GPU
            gpu_count = 0
            if any(keyword in source for keyword in ['gpu', 'cuda', 'torch', 'tensorflow']):
                gpu_count = 1

            return ResourceRequirements(
                cpu_cores=cpu_cores,
                memory_mb=memory_mb,
                gpu_count=gpu_count,
                timeout_seconds=3600  # 1 hour default
            )
        except Exception:
            # Fallback to minimal requirements
            return ResourceRequirements(
                cpu_cores=1.0,
                memory_mb=512,
                timeout_seconds=3600
            )


@dataclass
class PipelineDefinition:
    """Definition of a complete pipeline"""
    name: str
    steps: Dict[str, StepDefinition] = field(default_factory=dict)
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        """Number of steps in pipeline"""
        return len(self.steps)

    @property
    def definition_hash(self) -> str:
        """Generate hash of entire pipeline definition"""
        step_hashes = [step.signature_hash for step in self.steps.values()]
        content = f"{self.name}:{self.version}:{sorted(step_hashes)}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def add_step(self, step: StepDefinition):
        """Add a step to the pipeline"""
        self.steps[step.name] = step
        self.updated_at = datetime.now(timezone.utc)

    def remove_step(self, step_name: str):
        """Remove a step from the pipeline"""
        if step_name in self.steps:
            del self.steps[step_name]
            self.updated_at = datetime.now(timezone.utc)

    def get_execution_order(self) -> List[List[str]]:
        """Get optimal execution order (levels of parallel steps)"""
        from .graph import PipelineGraph

        # Create a temporary pipeline graph for analysis
        graph = PipelineGraph(self.name)

        # Add mock step nodes
        class MockStepNode:
            def __init__(self, name, dependencies):
                self.name = name
                self.dependencies = dependencies

        for step_name, step_def in self.steps.items():
            graph.steps[step_name] = MockStepNode(step_name, step_def.dependencies)

        return graph.build_execution_order()

    def validate(self) -> List[str]:
        """Validate pipeline definition"""
        issues = []

        if not self.name:
            issues.append("Pipeline name is required")

        if not self.steps:
            issues.append("Pipeline must have at least one step")

        # Check for missing dependencies
        all_step_names = set(self.steps.keys())
        for step_name, step_def in self.steps.items():
            missing_deps = step_def.dependencies - all_step_names
            if missing_deps:
                issues.append(f"Step '{step_name}' has undefined dependencies: {missing_deps}")

        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "config": self.config,
            "metadata": self.metadata,
            "steps": {
                name: {
                    "name": step.name,
                    "dependencies": list(step.dependencies),
                    "outputs": list(step.outputs),
                    "resource_requirements": step.resource_requirements.to_dict() if step.resource_requirements else None,
                    "retry_config": step.retry_config,
                    "cache_config": step.cache_config,
                    "tags": step.tags,
                    "metadata": step.metadata,
                    "signature_hash": step.signature_hash
                }
                for name, step in self.steps.items()
            },
            "definition_hash": self.definition_hash
        }


@dataclass
class ModelArtifact:
    """Represents a trained model artifact"""
    name: str
    version: str
    model_type: str  # e.g., "sklearn", "pytorch", "tensorflow"
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    size_bytes: Optional[int] = None

    @property
    def model_id(self) -> str:
        """Unique model identifier"""
        return f"{self.name}:{self.version}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type,
            "file_path": self.file_path,
            "metadata": self.metadata,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "tags": self.tags,
            "size_bytes": self.size_bytes,
            "model_id": self.model_id
        }


@dataclass
class DataArtifact:
    """Represents a data artifact (dataset, features, etc.)"""
    name: str
    version: str
    data_type: str  # e.g., "parquet", "csv", "json", "pickle"
    schema: Optional[Dict[str, str]] = None  # column_name -> data_type
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    size_bytes: Optional[int] = None
    row_count: Optional[int] = None

    @property
    def data_id(self) -> str:
        """Unique data identifier"""
        return f"{self.name}:{self.version}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "version": self.version,
            "data_type": self.data_type,
            "schema": self.schema,
            "file_path": self.file_path,
            "metadata": self.metadata,
            "stats": self.stats,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "tags": self.tags,
            "size_bytes": self.size_bytes,
            "row_count": self.row_count,
            "data_id": self.data_id
        }


@dataclass
class DeploymentInfo:
    """Information about a pipeline deployment"""
    deployment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_name: str = ""
    pipeline_version: str = ""
    environment: str = "production"
    url: Optional[str] = None
    status: str = "deployed"
    deployed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    deployed_by: Optional[str] = None
    container_count: int = 0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "deployment_id": self.deployment_id,
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "environment": self.environment,
            "url": self.url,
            "status": self.status,
            "deployed_at": self.deployed_at.isoformat(),
            "deployed_by": self.deployed_by,
            "container_count": self.container_count,
            "resource_usage": self.resource_usage,
            "config": self.config
        }


# Utility functions for working with models
def create_step_execution(step_name: str, max_attempts: int = 3) -> StepExecution:
    """Create a new step execution"""
    return StepExecution(
        step_name=step_name,
        max_attempts=max_attempts
    )


def create_pipeline_execution(
        pipeline_name: str,
        mode: ExecutionMode = ExecutionMode.LOCAL,
        created_by: str = None
) -> PipelineExecution:
    """Create a new pipeline execution"""
    return PipelineExecution(
        pipeline_name=pipeline_name,
        mode=mode,
        created_by=created_by
    )


def create_step_definition(
        name: str,
        function: Callable,
        dependencies: Set[str] = None,
        resource_requirements: ResourceRequirements = None
) -> StepDefinition:
    """Create a step definition"""
    return StepDefinition(
        name=name,
        function=function,
        dependencies=dependencies or set(),
        resource_requirements=resource_requirements
    )


def create_pipeline_definition(
        name: str,
        description: str = "",
        author: str = "",
        version: str = "1.0.0"
) -> PipelineDefinition:
    """Create a pipeline definition"""
    return PipelineDefinition(
        name=name,
        description=description,
        author=author,
        version=version
    )


# Type aliases for convenience
StepResult = Union[Any, None]
PipelineResult = Dict[str, Any]
ExecutionContext = Dict[str, Any]