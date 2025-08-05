"""
ops0 Runtime Module

Production runtime components for ML pipelines.
Handles containerization, orchestration, and monitoring.
"""

# Container management
from .containers import (
    ContainerBuilder,
    ContainerSpec,
    ContainerOrchestrator,
    container_orchestrator,
    FunctionAnalyzer,
    RequirementsGenerator
)

# Distributed orchestration
from .orchestrator import (
    RuntimeOrchestrator,
    JobQueue,
    Job,
    JobStatus,
    ExecutionMode,
    Worker,
    get_orchestrator,
    reset_orchestrator,
    AsyncOrchestrator
)

# Monitoring and observability
from .monitoring import (
    PipelineMonitor,
    MetricsCollector,
    AlertManager,
    StepMonitor,
    Metric,
    Alert,
    MetricType,
    AlertSeverity,
    get_monitor,
    reset_monitor,
    monitor_step
)

# Scheduling and resource management
from .scheduler import (
    PipelineScheduler,
    AutoScaler,
    SchedulingStrategy,
    ResourceRequirement,
    ResourcePool,
    ScheduledJob,
    ResourceType
)

# Runtime utilities
from .utils import (
    RuntimeEnvironment,
    ContainerUtils,
    SecurityUtils,
    MetricsFormatter,
    NetworkUtils,
    CacheManager,
    get_runtime_environment,
    get_cache_manager
)

# Cloud adapters
from .cloud_adapters import (
    CloudProvider,
    CloudResource,
    CloudAdapter,
    AWSAdapter,
    GCPAdapter,
    KubernetesAdapter,
    LocalAdapter,
    CloudAdapterFactory
)

# Version
try:
    from ...__about__ import __version__
except ImportError:
    __version__ = "0.1.0-dev"

# Public API
__all__ = [
    # Containers
    "ContainerBuilder",
    "ContainerSpec",
    "ContainerOrchestrator",
    "container_orchestrator",

    # Orchestration
    "RuntimeOrchestrator",
    "JobQueue",
    "Job",
    "JobStatus",
    "ExecutionMode",
    "get_orchestrator",
    "reset_orchestrator",

    # Monitoring
    "PipelineMonitor",
    "MetricsCollector",
    "AlertManager",
    "StepMonitor",
    "Metric",
    "Alert",
    "MetricType",
    "AlertSeverity",
    "get_monitor",
    "reset_monitor",
    "monitor_step",

    # Scheduling
    "PipelineScheduler",
    "AutoScaler",
    "SchedulingStrategy",
    "ResourceRequirement",
    "ResourcePool",
    "ScheduledJob",
    "ResourceType",

    # Utilities
    "RuntimeEnvironment",
    "ContainerUtils",
    "SecurityUtils",
    "MetricsFormatter",
    "NetworkUtils",
    "CacheManager",
    "get_runtime_environment",
    "get_cache_manager",

    # Cloud adapters
    "CloudProvider",
    "CloudResource",
    "CloudAdapter",
    "AWSAdapter",
    "GCPAdapter",
    "KubernetesAdapter",
    "LocalAdapter",
    "CloudAdapterFactory",

    # Version
    "__version__"
]

# Runtime initialization
import logging
import os

logger = logging.getLogger(__name__)

# Check runtime dependencies
_runtime_available = True
_missing_deps = []

try:
    import docker
except ImportError:
    _runtime_available = False
    _missing_deps.append("docker")

try:
    import psutil
except ImportError:
    _missing_deps.append("psutil")

if not _runtime_available:
    logger.warning(
        f"Some runtime features unavailable. Missing dependencies: {', '.join(_missing_deps)}. "
        f"Install with: pip install 'ops0[runtime]'"
    )

# Auto-start monitoring if configured
if os.getenv("OPS0_AUTO_MONITOR", "true").lower() == "true":
    try:
        monitor = get_monitor()
        logger.info("ops0 monitoring initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize monitoring: {e}")