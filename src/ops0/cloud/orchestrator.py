"""
ops0 Cloud Orchestrator - Enhanced Version

Central orchestration for deploying ML pipelines across cloud providers.
Handles the complete lifecycle: build, deploy, scale, monitor.
"""

import os
import json
import time
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.graph import PipelineGraph, StepNode
from ..core.storage import storage
from ..core.config import config
from ..runtime.containers import container_orchestrator
from .base import (
    CloudProvider, CloudResource, DeploymentSpec,
    ResourceType, ResourceStatus
)
from .cost import CostEstimator
from .autoscaler import AutoScaler
from .monitor import CloudMonitor

logger = logging.getLogger(__name__)


class DeploymentState:
    """Tracks deployment state for a pipeline"""

    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.resources: Dict[str, CloudResource] = {}
        self.status = "pending"
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.error_message: Optional[str] = None
        self.deployment_id = self._generate_id()
        self.endpoints: Dict[str, str] = {}  # Store endpoints

    def _generate_id(self) -> str:
        """Generate unique deployment ID"""
        data = f"{self.pipeline_name}-{self.created_at.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]

    def add_resource(self, step_name: str, resource: CloudResource):
        """Add deployed resource"""
        self.resources[step_name] = resource
        self.updated_at = datetime.now()

    def get_resource(self, step_name: str) -> Optional[CloudResource]:
        """Get resource for a step"""
        return self.resources.get(step_name)

    def get_endpoint(self) -> str:
        """Get the main endpoint for the deployment"""
        # Return API gateway endpoint if available
        if "api_gateway" in self.endpoints:
            return self.endpoints["api_gateway"]

        # Return load balancer endpoint if available
        if "load_balancer" in self.endpoints:
            return self.endpoints["load_balancer"]

        # Return first available endpoint
        if self.endpoints:
            return list(self.endpoints.values())[0]

        # Generate endpoint based on deployment
        if self.resources:
            first_resource = list(self.resources.values())[0]
            return f"https://{self.deployment_id}.{first_resource.region}.ops0.run"

        return "N/A"

    def set_endpoint(self, name: str, url: str):
        """Set an endpoint URL"""
        self.endpoints[name] = url

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "pipeline_name": self.pipeline_name,
            "deployment_id": self.deployment_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "error_message": self.error_message,
            "endpoints": self.endpoints,
            "endpoint": self.get_endpoint(),
            "resources": {
                name: resource.to_dict()
                for name, resource in self.resources.items()
            }
        }


class CloudOrchestrator:
    """Orchestrates pipeline deployment across cloud providers"""

    def __init__(self, provider: CloudProvider):
        self.provider = provider
        self.deployments: Dict[str, DeploymentState] = {}
        self.cost_estimator = CostEstimator(provider)
        self.autoscaler = AutoScaler(provider)
        self.monitor = CloudMonitor(provider)
        self.storage = storage

        # Deployment configuration
        self.max_parallel_deployments = 5
        self.deployment_timeout = 300  # 5 minutes

        logger.info(f"CloudOrchestrator initialized with provider: {provider.name}")

    def deploy(
            self,
            pipeline_name: str,
            pipeline: Optional[PipelineGraph] = None,
            environment: Optional[Dict[str, str]] = None,
            **kwargs
    ) -> DeploymentState:
        """
        Deploy a complete pipeline to the cloud.

        Args:
            pipeline_name: Name of the pipeline
            pipeline: PipelineGraph object (optional if already registered)
            environment: Environment variables
            **kwargs: Additional deployment options

        Returns:
            DeploymentState with deployment information
        """
        # Create deployment state
        deployment = DeploymentState(pipeline_name)
        self.deployments[deployment.deployment_id] = deployment

        try:
            # If pipeline not provided, try to load from registry
            if pipeline is None:
                # In practice, this would load from ops0 registry
                raise ValueError("Pipeline object required for deployment")

            deployment.status = "deploying"
            logger.info(f"Starting deployment: {deployment.deployment_id}")

            # Containerize pipeline steps
            logger.info("Containerizing pipeline steps...")
            container_specs = container_orchestrator.containerize_pipeline(pipeline)

            # Deploy storage resources
            logger.info("Setting up cloud storage...")
            storage_resource = self._deploy_storage(pipeline_name)
            deployment.add_resource("storage", storage_resource)

            # Deploy message queue
            logger.info("Setting up message queue...")
            queue_resource = self._deploy_queue(pipeline_name)
            deployment.add_resource("queue", queue_resource)

            # Deploy steps in parallel
            logger.info("Deploying pipeline steps...")
            with ThreadPoolExecutor(max_workers=self.max_parallel_deployments) as executor:
                futures = {}

                for step_name, container_spec in container_specs.items():
                    # Create deployment spec
                    spec = DeploymentSpec(
                        step_name=step_name,
                        image=container_spec.image_tag,
                        command=container_spec.entrypoint,
                        environment={
                            **container_spec.environment_vars,
                            **(environment or {}),
                            "OPS0_STORAGE_BUCKET": storage_resource.metadata.get("bucket_name", ""),
                            "OPS0_QUEUE_URL": queue_resource.metadata.get("queue_url", ""),
                        },
                        cpu=container_spec.cpu_limit,
                        memory=container_spec.memory_limit,
                        gpu=1 if container_spec.needs_gpu else 0,
                        min_instances=kwargs.get("min_instances", 1),
                        max_instances=kwargs.get("max_instances", 10),
                        timeout_seconds=kwargs.get("timeout", 300),
                    )

                    # Submit deployment
                    future = executor.submit(self._deploy_step, spec)
                    futures[future] = step_name

                # Wait for all deployments
                for future in as_completed(futures):
                    step_name = futures[future]
                    try:
                        resource = future.result()
                        deployment.add_resource(step_name, resource)
                        logger.info(f"Deployed step: {step_name}")
                    except Exception as e:
                        logger.error(f"Failed to deploy step {step_name}: {e}")
                        deployment.error_message = str(e)
                        raise

            # Create API gateway / load balancer
            logger.info("Setting up API gateway...")
            api_resource = self._deploy_api_gateway(deployment)
            deployment.add_resource("api_gateway", api_resource)
            deployment.set_endpoint("api_gateway", api_resource.metadata.get("endpoint", ""))

            # Configure monitoring
            logger.info("Configuring monitoring...")
            self.monitor.setup_monitoring(deployment)

            # Configure auto-scaling
            logger.info("Configuring auto-scaling...")
            self.autoscaler.configure(deployment)

            deployment.status = "deployed"
            logger.info(f"Deployment completed: {deployment.deployment_id}")

        except Exception as e:
            deployment.status = "failed"
            deployment.error_message = str(e)
            logger.error(f"Deployment failed: {e}")

            # Rollback on failure
            self._rollback_deployment(deployment)
            raise

        return deployment

    def _deploy_step(self, spec: DeploymentSpec) -> CloudResource:
        """Deploy a single step"""
        # Determine deployment type based on resource requirements
        if spec.gpu > 0:
            # Deploy as GPU-enabled container
            return self.provider.deploy_container(spec)
        elif spec.memory > 4096 or spec.cpu > 2:
            # Deploy as high-memory container
            return self.provider.deploy_container(spec)
        else:
            # Deploy as serverless function if supported
            if hasattr(self.provider, 'deploy_function'):
                return self.provider.deploy_function(spec)
            else:
                return self.provider.deploy_container(spec)

    def _deploy_storage(self, pipeline_name: str) -> CloudResource:
        """Deploy storage resources"""
        storage_name = f"ops0-{pipeline_name}-{int(time.time())}"
        return self.provider.create_storage(storage_name)

    def _deploy_queue(self, pipeline_name: str) -> CloudResource:
        """Deploy message queue"""
        queue_name = f"ops0-{pipeline_name}-queue"
        return self.provider.create_queue(queue_name)

    def _deploy_api_gateway(self, deployment: DeploymentState) -> CloudResource:
        """Deploy API gateway or load balancer"""
        # Get step endpoints
        step_endpoints = {}
        for step_name, resource in deployment.resources.items():
            if resource.resource_type in [ResourceType.CONTAINER, ResourceType.FUNCTION]:
                endpoint = resource.metadata.get("endpoint", "")
                if endpoint:
                    step_endpoints[step_name] = endpoint

        # Create API gateway
        if hasattr(self.provider, 'create_api_gateway'):
            return self.provider.create_api_gateway(
                name=f"ops0-{deployment.pipeline_name}-api",
                routes=step_endpoints
            )
        else:
            # Fallback to load balancer
            return self.provider.create_load_balancer(
                name=f"ops0-{deployment.pipeline_name}-lb",
                targets=list(step_endpoints.values())
            )

    def _rollback_deployment(self, deployment: DeploymentState):
        """Rollback failed deployment"""
        logger.info(f"Rolling back deployment: {deployment.deployment_id}")

        for resource_name, resource in deployment.resources.items():
            try:
                self.provider.delete_resource(resource)
                logger.info(f"Deleted resource: {resource_name}")
            except Exception as e:
                logger.error(f"Failed to delete resource {resource_name}: {e}")

    def scale(self, pipeline_name: str, min_instances: int, max_instances: int) -> bool:
        """Configure auto-scaling for a deployed pipeline"""
        # Find deployment
        deployment = self._find_deployment(pipeline_name)
        if not deployment:
            raise ValueError(f"No deployment found for pipeline: {pipeline_name}")

        # Update scaling configuration
        return self.autoscaler.update_scaling(
            deployment,
            min_instances=min_instances,
            max_instances=max_instances
        )

    def monitor(self, pipeline_name: str) -> str:
        """Get monitoring dashboard URL"""
        deployment = self._find_deployment(pipeline_name)
        if not deployment:
            raise ValueError(f"No deployment found for pipeline: {pipeline_name}")

        return self.monitor.get_dashboard_url(deployment)

    def estimate_cost(self, pipeline_name: str, monthly_requests: int) -> Dict[str, float]:
        """Estimate monthly cost for a pipeline"""
        deployment = self._find_deployment(pipeline_name)
        if deployment:
            # Use actual deployment
            return self.cost_estimator.estimate_from_deployment(
                deployment,
                monthly_requests
            )
        else:
            # Estimate from pipeline definition
            return self.cost_estimator.estimate_from_pipeline(
                pipeline_name,
                monthly_requests
            )

    def _find_deployment(self, pipeline_name: str) -> Optional[DeploymentState]:
        """Find deployment by pipeline name"""
        for deployment in self.deployments.values():
            if deployment.pipeline_name == pipeline_name:
                return deployment
        return None

    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all deployments"""
        return [
            deployment.to_dict()
            for deployment in self.deployments.values()
        ]

    def get_deployment(self, deployment_id: str) -> Optional[DeploymentState]:
        """Get deployment by ID"""
        return self.deployments.get(deployment_id)

    def delete_deployment(self, deployment_id: str) -> bool:
        """Delete a deployment"""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return False

        try:
            # Delete all resources
            for resource in deployment.resources.values():
                self.provider.delete_resource(resource)

            # Remove from tracking
            del self.deployments[deployment_id]
            return True

        except Exception as e:
            logger.error(f"Failed to delete deployment: {e}")
            return False
