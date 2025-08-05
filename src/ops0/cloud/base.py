"""
ops0 Cloud Base Classes

Abstract interfaces for cloud providers.
Designed for zero-configuration deployment.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
import logging
from datetime import datetime
import hashlib
import json

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of cloud resources"""
    CONTAINER = "container"
    FUNCTION = "function"
    STORAGE = "storage"
    QUEUE = "queue"
    DATABASE = "database"
    NETWORK = "network"
    LOAD_BALANCER = "load_balancer"
    SECRET = "secret"


class ResourceStatus(Enum):
    """Resource lifecycle states"""
    PENDING = "pending"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    UPDATING = "updating"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    DELETING = "deleting"
    DELETED = "deleted"


@dataclass
class ResourceMetrics:
    """Real-time resource metrics"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_mb: float = 0.0
    network_in_mbps: float = 0.0
    network_out_mbps: float = 0.0
    request_count: int = 0
    error_count: int = 0
    average_latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CloudResource:
    """Represents a deployed cloud resource"""
    provider: str
    resource_type: ResourceType
    resource_id: str
    resource_name: str
    region: str
    status: ResourceStatus = ResourceStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metrics: Optional[ResourceMetrics] = None

    def __post_init__(self):
        # Auto-tag with ops0 metadata
        self.tags.update({
            "managed-by": "ops0",
            "created-at": self.created_at.isoformat(),
            "ops0-version": "0.1.0"
        })

    @property
    def arn(self) -> str:
        """Generate cloud-agnostic resource identifier"""
        return f"arn:ops0:{self.provider}:{self.region}:{self.resource_type.value}/{self.resource_id}"

    @property
    def age_hours(self) -> float:
        """How long the resource has been running"""
        return (datetime.now() - self.created_at).total_seconds() / 3600

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "provider": self.provider,
            "resource_type": self.resource_type.value,
            "resource_id": self.resource_id,
            "resource_name": self.resource_name,
            "region": self.region,
            "status": self.status.value,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "arn": self.arn,
            "metrics": self.metrics.__dict__ if self.metrics else None
        }


@dataclass
class DeploymentSpec:
    """Specification for deploying a pipeline step"""
    step_name: str
    image: str
    command: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)

    # Resource requirements
    cpu: float = 1.0  # vCPUs
    memory: int = 2048  # MB
    gpu: int = 0  # Number of GPUs
    gpu_type: Optional[str] = None  # e.g., "nvidia-tesla-v100"

    # Scaling configuration
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_percent: float = 70.0
    scale_up_cooldown: int = 60  # seconds
    scale_down_cooldown: int = 300  # seconds

    # Network configuration
    port: Optional[int] = None
    health_check_path: str = "/health"
    timeout_seconds: int = 300

    # Storage requirements
    ephemeral_storage_gb: int = 10
    persistent_volumes: List[Dict[str, Any]] = field(default_factory=list)

    # Security
    secrets: List[str] = field(default_factory=list)
    iam_role: Optional[str] = None

    # Deployment options
    spot_instances: bool = False
    region_preferences: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Auto-detect GPU requirements from image
        if not self.gpu and self.image and any(
                x in self.image for x in ['cuda', 'gpu', 'tensorflow-gpu', 'pytorch-gpu']):
            self.gpu = 1
            logger.info(f"Auto-detected GPU requirement for {self.step_name}")

    def estimate_hourly_cost(self) -> float:
        """Rough cost estimation in USD/hour"""
        # Base CPU/memory costs (rough AWS pricing)
        cpu_cost = self.cpu * 0.04
        memory_cost = (self.memory / 1024) * 0.004
        gpu_cost = self.gpu * 0.90 if self.gpu else 0

        # Apply spot discount if enabled
        total = cpu_cost + memory_cost + gpu_cost
        if self.spot_instances:
            total *= 0.3  # ~70% discount for spot

        return total


class CloudProvider(ABC):
    """Abstract base class for cloud providers"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.resources: Dict[str, CloudResource] = {}
        self._setup()

    @abstractmethod
    def _setup(self):
        """Provider-specific setup"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (aws, gcp, azure, etc)"""
        pass

    @property
    @abstractmethod
    def regions(self) -> List[str]:
        """Available regions for this provider"""
        pass

    @abstractmethod
    def deploy_container(self, spec: DeploymentSpec) -> CloudResource:
        """Deploy a containerized pipeline step"""
        pass

    @abstractmethod
    def deploy_function(self, spec: DeploymentSpec) -> CloudResource:
        """Deploy a serverless function"""
        pass

    @abstractmethod
    def create_storage(self, name: str, region: str, **kwargs) -> CloudResource:
        """Create object storage bucket"""
        pass

    @abstractmethod
    def create_queue(self, name: str, region: str, **kwargs) -> CloudResource:
        """Create message queue"""
        pass

    @abstractmethod
    def create_secret(self, name: str, value: str, region: str) -> CloudResource:
        """Store a secret securely"""
        pass

    @abstractmethod
    def get_resource_status(self, resource_id: str) -> ResourceStatus:
        """Get current status of a resource"""
        pass

    @abstractmethod
    def get_resource_metrics(self, resource_id: str) -> ResourceMetrics:
        """Get resource metrics"""
        pass

    @abstractmethod
    def update_resource(self, resource_id: str, spec: DeploymentSpec) -> CloudResource:
        """Update a deployed resource"""
        pass

    @abstractmethod
    def delete_resource(self, resource_id: str) -> bool:
        """Delete a cloud resource"""
        pass

    @abstractmethod
    def list_resources(self, resource_type: Optional[ResourceType] = None) -> List[CloudResource]:
        """List all managed resources"""
        pass

    @abstractmethod
    def estimate_cost(self, resources: List[CloudResource], days: int = 30) -> Dict[str, float]:
        """Estimate cost for resources"""
        pass

    # Common helper methods
    def generate_resource_name(self, prefix: str, suffix: Optional[str] = None) -> str:
        """Generate a unique resource name"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        parts = [prefix, timestamp]
        if suffix:
            parts.append(suffix)

        # Add short hash for uniqueness
        name = "-".join(parts)
        hash_suffix = hashlib.md5(name.encode()).hexdigest()[:8]

        return f"{name}-{hash_suffix}"

    def tag_resource(self, resource_id: str, tags: Dict[str, str]):
        """Add tags to a resource"""
        if resource_id in self.resources:
            self.resources[resource_id].tags.update(tags)
            self.resources[resource_id].updated_at = datetime.now()

    def get_resource(self, resource_id: str) -> Optional[CloudResource]:
        """Get resource by ID"""
        return self.resources.get(resource_id)

    def wait_for_status(
            self,
            resource_id: str,
            target_status: ResourceStatus,
            timeout_seconds: int = 300,
            poll_interval: int = 5
    ) -> bool:
        """Wait for a resource to reach target status"""
        import time

        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            current_status = self.get_resource_status(resource_id)

            if current_status == target_status:
                return True

            if current_status == ResourceStatus.FAILED:
                logger.error(f"Resource {resource_id} failed")
                return False

            time.sleep(poll_interval)

        logger.warning(f"Timeout waiting for {resource_id} to reach {target_status}")
        return False

    def cleanup_old_resources(self, max_age_hours: int = 24 * 7):
        """Clean up resources older than specified age"""
        deleted_count = 0

        for resource_id, resource in list(self.resources.items()):
            if resource.age_hours > max_age_hours:
                logger.info(f"Cleaning up old resource: {resource_id} (age: {resource.age_hours:.1f} hours)")
                if self.delete_resource(resource_id):
                    deleted_count += 1

        return deleted_count