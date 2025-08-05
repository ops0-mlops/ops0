"""
ops0 Cloud Module

Provides cloud-agnostic deployment and resource management for ML pipelines.
Zero configuration required - ops0 handles all the complexity.
"""

from .base import CloudProvider, CloudResource, DeploymentSpec
from .aws import AWSProvider
from .gcp import GCPProvider
from .azure import AzureProvider
from .kubernetes import KubernetesProvider
from .orchestrator import CloudOrchestrator
from .cost import CostEstimator
from .autoscaler import AutoScaler
from .monitor import CloudMonitor


# Auto-detect cloud provider based on environment
def detect_provider() -> CloudProvider:
    """
    Automatically detect cloud provider from environment.

    Returns:
        CloudProvider instance configured for the detected environment
    """
    import os

    # Check for cloud-specific environment variables
    if os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION'):
        return AWSProvider()
    elif os.environ.get('GOOGLE_CLOUD_PROJECT') or os.environ.get('GCP_PROJECT'):
        return GCPProvider()
    elif os.environ.get('AZURE_SUBSCRIPTION_ID'):
        return AzureProvider()
    elif os.environ.get('KUBERNETES_SERVICE_HOST'):
        return KubernetesProvider()
    else:
        # Default to local development
        from .local import LocalProvider
        return LocalProvider()


# Global orchestrator instance
orchestrator = CloudOrchestrator(provider=detect_provider())


# Convenience functions for zero-config deployment
def deploy(pipeline_name: str, **kwargs):
    """Deploy a pipeline with zero configuration."""
    return orchestrator.deploy(pipeline_name, **kwargs)


def scale(pipeline_name: str, min_instances: int = 1, max_instances: int = 10):
    """Configure auto-scaling for a pipeline."""
    return orchestrator.scale(pipeline_name, min_instances, max_instances)


def monitor(pipeline_name: str):
    """Get monitoring dashboard URL for a pipeline."""
    return orchestrator.monitor(pipeline_name)


def estimate_cost(pipeline_name: str, monthly_requests: int = 1000000):
    """Estimate monthly cost for a pipeline."""
    return orchestrator.estimate_cost(pipeline_name, monthly_requests)


__all__ = [
    'CloudProvider',
    'CloudResource',
    'DeploymentSpec',
    'AWSProvider',
    'GCPProvider',
    'AzureProvider',
    'KubernetesProvider',
    'CloudOrchestrator',
    'CostEstimator',
    'AutoScaler',
    'CloudMonitor',
    'detect_provider',
    'deploy',
    'scale',
    'monitor',
    'estimate_cost'
]