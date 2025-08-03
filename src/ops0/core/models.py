"""
ops0 Core Data Models - Type definitions and data structures.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable, Set
from enum import Enum
from datetime import datetime
import json


class StepType(Enum):
    """Types of pipeline steps"""
    PROCESSING = "processing"
    TRAINING = "training"
    PREDICTION = "prediction"
    VALIDATION = "validation"
    EXPORT = "export"


class PipelineStatus(Enum):
    """Pipeline execution status"""
    DRAFT = "draft"
    VALIDATING = "validating"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ResourceRequirements:
    """Resource requirements for step execution"""
    cpu_cores: float = 1.0
    memory_mb: int = 512
    gpu_count: int = 0
    gpu_memory_mb: int = 0
    disk_space_mb: int = 1024
    timeout_seconds: int = 3600

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "gpu_count": self.gpu_count,
            "gpu_memory_mb": self.gpu_memory_mb,
            "disk_space_mb": self.disk_space_mb,
            "timeout_seconds": self.timeout_seconds
        }


@dataclass
class StepMetrics:
    """Metrics collected during step execution"""
    execution_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    input_data_size_mb: float = 0.0
    output_data_size_mb: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_time_ms": self.execution_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "input_data_size_mb": self.input_data_size_mb,
            "output_data_size_mb": self.output_data_size_mb,
            "custom_metrics": self.custom_metrics
        }


@dataclass
class DeploymentConfig:
    """Configuration for pipeline deployment"""
    environment: str = "production"
    region: str = "us-west-2"
    auto_scaling: bool = True
    min_workers: int = 1
    max_workers: int = 10
    health_check_path: str = "/health"
    monitoring_enabled: bool = True
    alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "environment": self.environment,
            "region": self.region,
            "auto_scaling": self.auto_scaling,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "health_check_path": self.health_check_path,
            "monitoring_enabled": self.monitoring_enabled,
            "alerts": self.alerts
        }


@dataclass
class PipelineMetadata:
    """Metadata for pipeline definition"""
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "tags": self.tags,
            "dependencies": self.dependencies
        }


@dataclass
class StepDefinition:
    """Complete definition of a pipeline step"""
    name: str
    func: Callable
    step_type: StepType = StepType.PROCESSING
    description: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "step_type": self.step_type.value,
            "description": self.description,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "dependencies": list(self.dependencies),
            "resources": self.resources.to_dict(),
            "metadata": self.metadata
        }


@dataclass
class PipelineDefinition:
    """Complete definition of a pipeline"""
    metadata: PipelineMetadata
    steps: Dict[str, StepDefinition] = field(default_factory=dict)
    deployment_config: DeploymentConfig = field(default_factory=DeploymentConfig)

    def add_step(self, step: StepDefinition):
        """Add a step to the pipeline"""
        self.steps[step.name] = step
        self.metadata.updated_at = datetime.now()

    def get_step(self, name: str) -> Optional[StepDefinition]:
        """Get a step by name"""
        return self.steps.get(name)

    def list_steps(self) -> List[str]:
        """List all step names"""
        return list(self.steps.keys())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "steps": {name: step.to_dict() for name, step in self.steps.items()},
            "deployment_config": self.deployment_config.to_dict()
        }

    def to_json(self) -> str:
        """Serialize pipeline to JSON"""
        return json.dumps(self.to_dict(), indent=2, default=str)