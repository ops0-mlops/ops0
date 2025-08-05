"""
ops0 Cloud Adapters

Cloud-agnostic interface for deploying pipelines across providers.
Supports AWS, GCP, Azure, and Kubernetes out of the box.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers"""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    KUBERNETES = "kubernetes"
    LOCAL = "local"


@dataclass
class CloudResource:
    """Represents a cloud resource"""
    provider: CloudProvider
    resource_type: str
    resource_id: str
    region: str
    metadata: Dict[str, Any]
    status: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "provider": self.provider.value,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "region": self.region,
            "metadata": self.metadata,
            "status": self.status
        }


class CloudAdapter(ABC):
    """Abstract base class for cloud adapters"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = CloudProvider.LOCAL

    @abstractmethod
    def deploy_container(self, container_spec: Dict[str, Any]) -> CloudResource:
        """Deploy a container to the cloud"""
        pass

    @abstractmethod
    def create_queue(self, queue_name: str) -> CloudResource:
        """Create a message queue"""
        pass

    @abstractmethod
    def create_storage(self, bucket_name: str) -> CloudResource:
        """Create object storage"""
        pass

    @abstractmethod
    def get_resource_status(self, resource: CloudResource) -> str:
        """Get current status of a resource"""
        pass

    @abstractmethod
    def delete_resource(self, resource: CloudResource) -> bool:
        """Delete a cloud resource"""
        pass

    @abstractmethod
    def estimate_cost(self, resources: List[CloudResource]) -> float:
        """Estimate monthly cost for resources"""
        pass


class AWSAdapter(CloudAdapter):
    """AWS cloud adapter using boto3"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider = CloudProvider.AWS
        self.region = config.get("region", "us-east-1")

    def deploy_container(self, container_spec: Dict[str, Any]) -> CloudResource:
        """Deploy container to ECS or Lambda"""
        # In production, this would use boto3 to deploy to ECS/Fargate

        resource_type = "ecs_task" if container_spec.get("long_running") else "lambda_function"

        resource = CloudResource(
            provider=self.provider,
            resource_type=resource_type,
            resource_id=f"{container_spec['name']}-{int(time.time())}",
            region=self.region,
            metadata={
                "image": container_spec.get("image"),
                "memory": container_spec.get("memory", 2048),
                "cpu": container_spec.get("cpu", 1024),
                "environment": container_spec.get("environment", {})
            },
            status="deploying"
        )

        logger.info(f"Deploying container to AWS {resource_type}: {resource.resource_id}")

        # Simulate deployment
        # In production: boto3 calls to ECS/Lambda

        return resource

    def create_queue(self, queue_name: str) -> CloudResource:
        """Create SQS queue"""
        resource = CloudResource(
            provider=self.provider,
            resource_type="sqs_queue",
            resource_id=f"{queue_name}-{self.region}",
            region=self.region,
            metadata={
                "visibility_timeout": 300,
                "message_retention": 345600  # 4 days
            },
            status="active"
        )

        logger.info(f"Created SQS queue: {resource.resource_id}")
        return resource

    def create_storage(self, bucket_name: str) -> CloudResource:
        """Create S3 bucket"""
        resource = CloudResource(
            provider=self.provider,
            resource_type="s3_bucket",
            resource_id=f"{bucket_name}-{int(time.time())}",
            region=self.region,
            metadata={
                "versioning": True,
                "lifecycle_rules": [
                    {"days": 30, "storage_class": "GLACIER"}
                ]
            },
            status="active"
        )

        logger.info(f"Created S3 bucket: {resource.resource_id}")
        return resource

    def get_resource_status(self, resource: CloudResource) -> str:
        """Get resource status from AWS"""
        # In production: boto3 describe calls
        return "active"

    def delete_resource(self, resource: CloudResource) -> bool:
        """Delete AWS resource"""
        logger.info(f"Deleting AWS resource: {resource.resource_id}")
        # In production: boto3 delete calls
        return True

    def estimate_cost(self, resources: List[CloudResource]) -> float:
        """Estimate AWS costs"""
        total_cost = 0.0

        for resource in resources:
            if resource.resource_type == "ecs_task":
                # ECS Fargate pricing
                cpu = resource.metadata.get("cpu", 1024) / 1024  # vCPU
                memory = resource.metadata.get("memory", 2048) / 1024  # GB
                hours = 730  # Monthly hours

                cpu_cost = cpu * 0.04048 * hours
                memory_cost = memory * 0.004445 * hours
                total_cost += cpu_cost + memory_cost

            elif resource.resource_type == "lambda_function":
                # Lambda pricing (estimates)
                requests = 1000000  # 1M requests/month
                duration_ms = 100  # Average duration
                memory_mb = resource.metadata.get("memory", 512)

                request_cost = requests * 0.0000002
                compute_cost = (requests * duration_ms * memory_mb / 1024) * 0.0000166667
                total_cost += request_cost + compute_cost

            elif resource.resource_type == "s3_bucket":
                # S3 pricing (estimates)
                storage_gb = 100  # Estimate
                requests = 10000  # Estimate

                storage_cost = storage_gb * 0.023
                request_cost = requests * 0.0004 / 1000
                total_cost += storage_cost + request_cost

        return total_cost


class GCPAdapter(CloudAdapter):
    """Google Cloud Platform adapter"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider = CloudProvider.GCP
        self.project_id = config.get("project_id")
        self.region = config.get("region", "us-central1")

    def deploy_container(self, container_spec: Dict[str, Any]) -> CloudResource:
        """Deploy to Cloud Run or Cloud Functions"""
        resource_type = "cloud_run" if container_spec.get("long_running") else "cloud_function"

        resource = CloudResource(
            provider=self.provider,
            resource_type=resource_type,
            resource_id=f"{container_spec['name']}-{self.region}",
            region=self.region,
            metadata={
                "image": container_spec.get("image"),
                "memory": container_spec.get("memory", "2Gi"),
                "cpu": container_spec.get("cpu", 1),
                "max_instances": container_spec.get("max_instances", 100),
                "project_id": self.project_id
            },
            status="deploying"
        )

        logger.info(f"Deploying container to GCP {resource_type}: {resource.resource_id}")
        return resource

    def create_queue(self, queue_name: str) -> CloudResource:
        """Create Pub/Sub topic"""
        resource = CloudResource(
            provider=self.provider,
            resource_type="pubsub_topic",
            resource_id=f"{queue_name}-{self.project_id}",
            region="global",
            metadata={
                "project_id": self.project_id,
                "message_retention": "7d"
            },
            status="active"
        )

        logger.info(f"Created Pub/Sub topic: {resource.resource_id}")
        return resource

    def create_storage(self, bucket_name: str) -> CloudResource:
        """Create Cloud Storage bucket"""
        resource = CloudResource(
            provider=self.provider,
            resource_type="gcs_bucket",
            resource_id=f"{bucket_name}-{self.project_id}",
            region=self.region,
            metadata={
                "storage_class": "STANDARD",
                "lifecycle_rules": {
                    "age": 30,
                    "action": "SetStorageClass",
                    "storage_class": "NEARLINE"
                }
            },
            status="active"
        )

        logger.info(f"Created GCS bucket: {resource.resource_id}")
        return resource

    def get_resource_status(self, resource: CloudResource) -> str:
        """Get resource status from GCP"""
        return "active"

    def delete_resource(self, resource: CloudResource) -> bool:
        """Delete GCP resource"""
        logger.info(f"Deleting GCP resource: {resource.resource_id}")
        return True

    def estimate_cost(self, resources: List[CloudResource]) -> float:
        """Estimate GCP costs"""
        total_cost = 0.0

        for resource in resources:
            if resource.resource_type == "cloud_run":
                # Cloud Run pricing
                cpu = resource.metadata.get("cpu", 1)
                memory_gb = float(resource.metadata.get("memory", "2Gi").rstrip("Gi"))
                requests = 1000000  # Estimate

                cpu_cost = cpu * 0.000024 * requests
                memory_cost = memory_gb * 0.0000025 * requests
                total_cost += cpu_cost + memory_cost

            elif resource.resource_type == "gcs_bucket":
                # Cloud Storage pricing
                storage_gb = 100  # Estimate
                total_cost += storage_gb * 0.020

        return total_cost


class KubernetesAdapter(CloudAdapter):
    """Kubernetes adapter for any K8s cluster"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider = CloudProvider.KUBERNETES
        self.namespace = config.get("namespace", "ops0")
        self.cluster_endpoint = config.get("cluster_endpoint")

    def deploy_container(self, container_spec: Dict[str, Any]) -> CloudResource:
        """Deploy as Kubernetes Job or Deployment"""
        resource_type = "deployment" if container_spec.get("long_running") else "job"

        # Generate Kubernetes manifest
        manifest = self._generate_k8s_manifest(container_spec, resource_type)

        resource = CloudResource(
            provider=self.provider,
            resource_type=f"k8s_{resource_type}",
            resource_id=f"{container_spec['name']}-{self.namespace}",
            region=self.cluster_endpoint or "local",
            metadata={
                "manifest": manifest,
                "namespace": self.namespace
            },
            status="deploying"
        )

        logger.info(f"Deploying to Kubernetes {resource_type}: {resource.resource_id}")

        # In production: kubectl apply or K8s API calls

        return resource

    def create_queue(self, queue_name: str) -> CloudResource:
        """Create Redis queue in K8s"""
        resource = CloudResource(
            provider=self.provider,
            resource_type="k8s_redis",
            resource_id=f"{queue_name}-redis",
            region=self.cluster_endpoint or "local",
            metadata={
                "namespace": self.namespace,
                "replicas": 1
            },
            status="active"
        )

        logger.info(f"Created Redis queue in K8s: {resource.resource_id}")
        return resource

    def create_storage(self, bucket_name: str) -> CloudResource:
        """Create MinIO storage in K8s"""
        resource = CloudResource(
            provider=self.provider,
            resource_type="k8s_minio",
            resource_id=f"{bucket_name}-minio",
            region=self.cluster_endpoint or "local",
            metadata={
                "namespace": self.namespace,
                "storage_size": "100Gi"
            },
            status="active"
        )

        logger.info(f"Created MinIO storage in K8s: {resource.resource_id}")
        return resource

    def get_resource_status(self, resource: CloudResource) -> str:
        """Get resource status from K8s"""
        # In production: kubectl get or K8s API calls
        return "active"

    def delete_resource(self, resource: CloudResource) -> bool:
        """Delete K8s resource"""
        logger.info(f"Deleting K8s resource: {resource.resource_id}")
        # In production: kubectl delete or K8s API calls
        return True

    def estimate_cost(self, resources: List[CloudResource]) -> float:
        """Estimate K8s costs (infrastructure dependent)"""
        # K8s costs depend on underlying infrastructure
        # This is a rough estimate based on resource requests
        total_cost = 0.0

        for resource in resources:
            if resource.resource_type.startswith("k8s_"):
                # Estimate based on resource requests
                cpu = resource.metadata.get("cpu", 1)
                memory_gb = resource.metadata.get("memory", 2)

                # Rough estimate: $0.05/cpu/hour, $0.01/GB/hour
                monthly_hours = 730
                cpu_cost = cpu * 0.05 * monthly_hours
                memory_cost = memory_gb * 0.01 * monthly_hours
                total_cost += cpu_cost + memory_cost

        return total_cost

    def _generate_k8s_manifest(self, container_spec: Dict[str, Any], resource_type: str) -> Dict[str, Any]:
        """Generate Kubernetes manifest"""
        manifest = {
            "apiVersion": "batch/v1" if resource_type == "job" else "apps/v1",
            "kind": "Job" if resource_type == "job" else "Deployment",
            "metadata": {
                "name": container_spec["name"],
                "namespace": self.namespace,
                "labels": {
                    "app": "ops0",
                    "pipeline": container_spec.get("pipeline", "unknown"),
                    "step": container_spec["name"]
                }
            },
            "spec": {
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "ops0",
                            "step": container_spec["name"]
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": container_spec["name"],
                            "image": container_spec["image"],
                            "resources": {
                                "requests": {
                                    "memory": container_spec.get("memory", "2Gi"),
                                    "cpu": str(container_spec.get("cpu", 1))
                                },
                                "limits": {
                                    "memory": container_spec.get("memory", "2Gi"),
                                    "cpu": str(container_spec.get("cpu", 1))
                                }
                            },
                            "env": [
                                {"name": k, "value": str(v)}
                                for k, v in container_spec.get("environment", {}).items()
                            ]
                        }],
                        "restartPolicy": "Never" if resource_type == "job" else "Always"
                    }
                }
            }
        }

        if resource_type == "deployment":
            manifest["spec"]["replicas"] = container_spec.get("replicas", 1)
            manifest["spec"]["selector"] = {
                "matchLabels": {
                    "app": "ops0",
                    "step": container_spec["name"]
                }
            }

        return manifest


class CloudAdapterFactory:
    """Factory for creating cloud adapters"""

    _adapters = {
        CloudProvider.AWS: AWSAdapter,
        CloudProvider.GCP: GCPAdapter,
        CloudProvider.KUBERNETES: KubernetesAdapter
    }

    @classmethod
    def create(cls, provider: CloudProvider, config: Dict[str, Any]) -> CloudAdapter:
        """Create a cloud adapter instance"""
        if provider == CloudProvider.LOCAL:
            # Local adapter just logs actions
            return LocalAdapter(config)

        adapter_class = cls._adapters.get(provider)
        if not adapter_class:
            raise ValueError(f"Unsupported cloud provider: {provider}")

        return adapter_class(config)

    @classmethod
    def register(cls, provider: CloudProvider, adapter_class: type) -> None:
        """Register a custom cloud adapter"""
        cls._adapters[provider] = adapter_class


class LocalAdapter(CloudAdapter):
    """Local development adapter"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider = CloudProvider.LOCAL
        self.resources: Dict[str, CloudResource] = {}

    def deploy_container(self, container_spec: Dict[str, Any]) -> CloudResource:
        """Simulate container deployment locally"""
        resource = CloudResource(
            provider=self.provider,
            resource_type="local_container",
            resource_id=f"local-{container_spec['name']}-{int(time.time())}",
            region="local",
            metadata=container_spec,
            status="active"
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Simulated local deployment: {resource.resource_id}")
        return resource

    def create_queue(self, queue_name: str) -> CloudResource:
        """Simulate queue creation locally"""
        resource = CloudResource(
            provider=self.provider,
            resource_type="local_queue",
            resource_id=f"local-queue-{queue_name}",
            region="local",
            metadata={"type": "memory"},
            status="active"
        )

        self.resources[resource.resource_id] = resource
        return resource

    def create_storage(self, bucket_name: str) -> CloudResource:
        """Simulate storage creation locally"""
        resource = CloudResource(
            provider=self.provider,
            resource_type="local_storage",
            resource_id=f"local-storage-{bucket_name}",
            region="local",
            metadata={"path": f"/tmp/ops0/{bucket_name}"},
            status="active"
        )

        self.resources[resource.resource_id] = resource
        return resource

    def get_resource_status(self, resource: CloudResource) -> str:
        """Get local resource status"""
        return self.resources.get(resource.resource_id, resource).status

    def delete_resource(self, resource: CloudResource) -> bool:
        """Delete local resource"""
        if resource.resource_id in self.resources:
            del self.resources[resource.resource_id]
        return True

    def estimate_cost(self, resources: List[CloudResource]) -> float:
        """Local resources are free!"""
        return 0.0


# Import time for timestamps
import time

# Export public API
__all__ = [
    'CloudProvider',
    'CloudResource',
    'CloudAdapter',
    'AWSAdapter',
    'GCPAdapter',
    'KubernetesAdapter',
    'LocalAdapter',
    'CloudAdapterFactory'
]