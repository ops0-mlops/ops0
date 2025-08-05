"""
ops0 Cloud Orchestrator

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

from ..core.pipeline import PipelineGraph, PipelineStep
from ..core.storage import StorageManager
from ..core.config import config
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "pipeline_name": self.pipeline_name,
            "deployment_id": self.deployment_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "error_message": self.error_message,
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
        self.storage = StorageManager()

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
            pipeline: Pipeline graph (will load from storage if not provided)
            environment: Environment variables for all steps
            **kwargs: Additional deployment options

        Returns:
            DeploymentState with deployment information
        """
        logger.info(f"Starting deployment of pipeline: {pipeline_name}")

        # Create deployment state
        deployment = DeploymentState(pipeline_name)
        self.deployments[pipeline_name] = deployment

        try:
            # Load pipeline if not provided
            if not pipeline:
                pipeline = self._load_pipeline(pipeline_name)

            # Validate pipeline
            self._validate_pipeline(pipeline)

            # Build deployment specs for each step
            deployment_specs = self._build_deployment_specs(
                pipeline, environment, **kwargs
            )

            # Deploy infrastructure resources first
            self._deploy_infrastructure(deployment, pipeline, **kwargs)

            # Deploy pipeline steps in parallel where possible
            self._deploy_steps(deployment, pipeline, deployment_specs)

            # Configure networking and communication
            self._configure_networking(deployment, pipeline)

            # Set up monitoring and alerting
            self._setup_monitoring(deployment, pipeline)

            # Validate deployment
            self._validate_deployment(deployment)

            deployment.status = "active"
            logger.info(f"Successfully deployed pipeline: {pipeline_name}")

        except Exception as e:
            logger.error(f"Failed to deploy pipeline {pipeline_name}: {e}")
            deployment.status = "failed"
            deployment.error_message = str(e)

            # Rollback on failure
            self._rollback_deployment(deployment)
            raise

        return deployment

    def update(
            self,
            pipeline_name: str,
            steps_to_update: Optional[List[str]] = None,
            **kwargs
    ) -> DeploymentState:
        """
        Update a deployed pipeline.

        Args:
            pipeline_name: Name of the pipeline
            steps_to_update: Specific steps to update (all if None)
            **kwargs: Update options

        Returns:
            Updated DeploymentState
        """
        deployment = self.deployments.get(pipeline_name)
        if not deployment:
            raise ValueError(f"No deployment found for pipeline: {pipeline_name}")

        logger.info(f"Updating pipeline deployment: {pipeline_name}")

        try:
            # Load latest pipeline definition
            pipeline = self._load_pipeline(pipeline_name)

            # Determine which steps need updating
            if not steps_to_update:
                steps_to_update = list(pipeline.steps.keys())

            # Update each step
            for step_name in steps_to_update:
                if step_name not in deployment.resources:
                    logger.warning(f"Step {step_name} not found in deployment")
                    continue

                step = pipeline.steps[step_name]
                resource = deployment.resources[step_name]

                # Build new spec
                spec = self._build_step_spec(step, kwargs.get('environment', {}))

                # Update resource
                updated_resource = self.provider.update_resource(
                    resource.resource_id, spec
                )

                deployment.add_resource(step_name, updated_resource)

            deployment.status = "active"
            deployment.updated_at = datetime.now()

        except Exception as e:
            logger.error(f"Failed to update pipeline {pipeline_name}: {e}")
            deployment.status = "update_failed"
            deployment.error_message = str(e)
            raise

        return deployment

    def scale(
            self,
            pipeline_name: str,
            min_instances: int = 1,
            max_instances: int = 10,
            target_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Configure auto-scaling for a pipeline.

        Args:
            pipeline_name: Name of the pipeline
            min_instances: Minimum instances per step
            max_instances: Maximum instances per step
            target_metrics: Target metrics for scaling

        Returns:
            Scaling configuration
        """
        deployment = self.deployments.get(pipeline_name)
        if not deployment:
            raise ValueError(f"No deployment found for pipeline: {pipeline_name}")

        # Default target metrics
        if not target_metrics:
            target_metrics = {
                "cpu_percent": 70.0,
                "memory_percent": 80.0,
                "request_rate": 1000.0
            }

        # Configure auto-scaling for each resource
        scaling_config = {}

        for step_name, resource in deployment.resources.items():
            if resource.resource_type in [ResourceType.CONTAINER, ResourceType.FUNCTION]:
                config = self.autoscaler.configure_scaling(
                    resource,
                    min_instances=min_instances,
                    max_instances=max_instances,
                    target_metrics=target_metrics
                )
                scaling_config[step_name] = config

        return scaling_config

    def monitor(self, pipeline_name: str) -> str:
        """
        Get monitoring dashboard URL for a pipeline.

        Args:
            pipeline_name: Name of the pipeline

        Returns:
            Dashboard URL
        """
        deployment = self.deployments.get(pipeline_name)
        if not deployment:
            raise ValueError(f"No deployment found for pipeline: {pipeline_name}")

        # Create monitoring dashboard
        dashboard_url = self.monitor.create_dashboard(
            pipeline_name,
            list(deployment.resources.values())
        )

        return dashboard_url

    def estimate_cost(
            self,
            pipeline_name: str,
            monthly_requests: int = 1000000,
            average_duration_ms: int = 100
    ) -> Dict[str, Any]:
        """
        Estimate monthly cost for a pipeline.

        Args:
            pipeline_name: Name of the pipeline
            monthly_requests: Expected requests per month
            average_duration_ms: Average request duration

        Returns:
            Cost breakdown
        """
        deployment = self.deployments.get(pipeline_name)
        if deployment:
            # Estimate based on deployed resources
            resources = list(deployment.resources.values())
        else:
            # Estimate based on pipeline definition
            pipeline = self._load_pipeline(pipeline_name)
            resources = self._estimate_resources(pipeline)

        # Get cost estimate
        cost_breakdown = self.cost_estimator.estimate(
            resources,
            usage_patterns={
                "monthly_requests": monthly_requests,
                "average_duration_ms": average_duration_ms
            }
        )

        return cost_breakdown

    def delete(self, pipeline_name: str) -> bool:
        """
        Delete a deployed pipeline.

        Args:
            pipeline_name: Name of the pipeline

        Returns:
            Success status
        """
        deployment = self.deployments.get(pipeline_name)
        if not deployment:
            logger.warning(f"No deployment found for pipeline: {pipeline_name}")
            return True

        logger.info(f"Deleting pipeline deployment: {pipeline_name}")

        # Delete all resources
        success = True
        for step_name, resource in deployment.resources.items():
            try:
                if not self.provider.delete_resource(resource.resource_id):
                    success = False
                    logger.error(f"Failed to delete resource for step: {step_name}")
            except Exception as e:
                success = False
                logger.error(f"Error deleting resource for step {step_name}: {e}")

        # Clean up monitoring
        try:
            self.monitor.delete_dashboard(pipeline_name)
        except Exception as e:
            logger.warning(f"Failed to delete monitoring dashboard: {e}")

        # Remove from tracking
        if success:
            del self.deployments[pipeline_name]

        return success

    def get_status(self, pipeline_name: str) -> Dict[str, Any]:
        """
        Get detailed status of a deployed pipeline.

        Args:
            pipeline_name: Name of the pipeline

        Returns:
            Status information
        """
        deployment = self.deployments.get(pipeline_name)
        if not deployment:
            return {"status": "not_deployed"}

        # Get status for each resource
        step_statuses = {}
        for step_name, resource in deployment.resources.items():
            status = self.provider.get_resource_status(resource.resource_id)
            metrics = self.provider.get_resource_metrics(resource.resource_id)

            step_statuses[step_name] = {
                "status": status.value,
                "resource_type": resource.resource_type.value,
                "metrics": {
                    "cpu_percent": metrics.cpu_percent,
                    "memory_mb": metrics.memory_mb,
                    "request_count": metrics.request_count,
                    "error_count": metrics.error_count,
                    "average_latency_ms": metrics.average_latency_ms
                }
            }

        return {
            "pipeline_name": pipeline_name,
            "deployment_id": deployment.deployment_id,
            "status": deployment.status,
            "created_at": deployment.created_at.isoformat(),
            "updated_at": deployment.updated_at.isoformat(),
            "provider": self.provider.name,
            "steps": step_statuses,
            "dashboard_url": self.monitor.get_dashboard_url(pipeline_name)
        }

    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all deployments"""
        deployments = []

        for name, deployment in self.deployments.items():
            deployments.append({
                "pipeline_name": name,
                "deployment_id": deployment.deployment_id,
                "status": deployment.status,
                "created_at": deployment.created_at.isoformat(),
                "provider": self.provider.name,
                "resource_count": len(deployment.resources)
            })

        return deployments

    # Private helper methods

    def _load_pipeline(self, pipeline_name: str) -> PipelineGraph:
        """Load pipeline from storage"""
        # In production, this would load from ops0 storage
        # For now, create a mock pipeline
        from ..core.pipeline import PipelineGraph, PipelineStep

        pipeline = PipelineGraph(name=pipeline_name)

        # Add mock steps
        steps = [
            PipelineStep(
                name="preprocess",
                func=lambda x: x,  # Placeholder
                inputs=["raw_data"],
                outputs=["processed_data"]
            ),
            PipelineStep(
                name="predict",
                func=lambda x: x,  # Placeholder
                inputs=["processed_data"],
                outputs=["predictions"]
            )
        ]

        for step in steps:
            pipeline.add_step(step)

        return pipeline

    def _validate_pipeline(self, pipeline: PipelineGraph):
        """Validate pipeline can be deployed"""
        if not pipeline.steps:
            raise ValueError("Pipeline has no steps")

        # Check for cycles
        try:
            pipeline.build_execution_order()
        except Exception as e:
            raise ValueError(f"Invalid pipeline structure: {e}")

    def _build_deployment_specs(
            self,
            pipeline: PipelineGraph,
            environment: Optional[Dict[str, str]],
            **kwargs
    ) -> Dict[str, DeploymentSpec]:
        """Build deployment specifications for each step"""
        specs = {}

        for step_name, step in pipeline.steps.items():
            spec = self._build_step_spec(step, environment, **kwargs)
            specs[step_name] = spec

        return specs

    def _build_step_spec(
            self,
            step: PipelineStep,
            environment: Optional[Dict[str, str]],
            **kwargs
    ) -> DeploymentSpec:
        """Build deployment spec for a single step"""
        # Default container image
        image = kwargs.get('image', f"ops0/runtime:latest")

        # Merge environments
        step_env = {}
        if environment:
            step_env.update(environment)

        # Add step-specific environment
        step_env.update({
            "OPS0_STEP_NAME": step.name,
            "OPS0_PIPELINE_NAME": kwargs.get('pipeline_name', 'unknown')
        })

        # Determine resource requirements
        cpu = kwargs.get('cpu', 1.0)
        memory = kwargs.get('memory', 2048)
        gpu = kwargs.get('gpu', 0)

        # Auto-detect GPU needs from step metadata
        if hasattr(step, 'requires_gpu') and step.requires_gpu:
            gpu = max(gpu, 1)

        spec = DeploymentSpec(
            step_name=step.name,
            image=image,
            command=["python", "-m", "ops0.runtime.worker", step.name],
            environment=step_env,
            cpu=cpu,
            memory=memory,
            gpu=gpu,
            min_instances=kwargs.get('min_instances', 1),
            max_instances=kwargs.get('max_instances', 10),
            spot_instances=kwargs.get('spot_instances', False)
        )

        return spec

    def _deploy_infrastructure(
            self,
            deployment: DeploymentState,
            pipeline: PipelineGraph,
            **kwargs
    ):
        """Deploy infrastructure resources (storage, queues, etc)"""
        # Create shared storage
        storage_resource = self.provider.create_storage(
            name=f"{pipeline.name}-storage",
            region=self.provider.regions[0]
        )
        deployment.add_resource("_storage", storage_resource)

        # Create message queue for step communication
        queue_resource = self.provider.create_queue(
            name=f"{pipeline.name}-queue",
            region=self.provider.regions[0]
        )
        deployment.add_resource("_queue", queue_resource)

        # Store secrets if provided
        secrets = kwargs.get('secrets', {})
        for secret_name, secret_value in secrets.items():
            secret_resource = self.provider.create_secret(
                name=f"{pipeline.name}-{secret_name}",
                value=secret_value,
                region=self.provider.regions[0]
            )
            deployment.add_resource(f"_secret_{secret_name}", secret_resource)

    def _deploy_steps(
            self,
            deployment: DeploymentState,
            pipeline: PipelineGraph,
            deployment_specs: Dict[str, DeploymentSpec]
    ):
        """Deploy pipeline steps in optimal order"""
        # Get execution order
        execution_order = pipeline.build_execution_order()

        # Deploy steps level by level
        for level in execution_order:
            # Deploy steps in parallel within each level
            with ThreadPoolExecutor(max_workers=self.max_parallel_deployments) as executor:
                futures = {}

                for step_name in level:
                    spec = deployment_specs[step_name]

                    # Determine deployment type
                    if self._should_use_serverless(spec):
                        future = executor.submit(
                            self.provider.deploy_function, spec
                        )
                    else:
                        future = executor.submit(
                            self.provider.deploy_container, spec
                        )

                    futures[future] = step_name

                # Wait for deployments to complete
                for future in as_completed(futures, timeout=self.deployment_timeout):
                    step_name = futures[future]

                    try:
                        resource = future.result()
                        deployment.add_resource(step_name, resource)
                        logger.info(f"Deployed step: {step_name}")

                    except Exception as e:
                        logger.error(f"Failed to deploy step {step_name}: {e}")
                        raise

    def _should_use_serverless(self, spec: DeploymentSpec) -> bool:
        """Determine if step should use serverless deployment"""
        # Use serverless for:
        # - Steps with low memory requirements
        # - Steps without GPU requirements
        # - Steps that don't need persistent connections

        if spec.gpu > 0:
            return False

        if spec.memory > 3008:  # Lambda max memory
            return False

        if spec.port:  # Needs persistent connections
            return False

        return True

    def _configure_networking(
            self,
            deployment: DeploymentState,
            pipeline: PipelineGraph
    ):
        """Configure networking between deployed resources"""
        # In production, this would:
        # - Set up service discovery
        # - Configure load balancers
        # - Set up VPC peering if needed
        # - Configure firewall rules

        logger.info("Configuring networking for pipeline")

    def _setup_monitoring(
            self,
            deployment: DeploymentState,
            pipeline: PipelineGraph
    ):
        """Set up monitoring and alerting"""
        # Create monitoring dashboard
        resources = list(deployment.resources.values())

        dashboard_url = self.monitor.create_dashboard(
            pipeline.name,
            resources
        )

        # Set up alerts
        alert_config = {
            "error_rate_threshold": 0.05,  # 5% error rate
            "latency_threshold_ms": 1000,  # 1 second
            "cpu_threshold_percent": 80,
            "memory_threshold_percent": 90
        }

        self.monitor.setup_alerts(
            pipeline.name,
            resources,
            alert_config
        )

        logger.info(f"Monitoring dashboard: {dashboard_url}")

    def _validate_deployment(self, deployment: DeploymentState):
        """Validate deployment is healthy"""
        unhealthy_steps = []

        for step_name, resource in deployment.resources.items():
            if step_name.startswith("_"):  # Skip infrastructure resources
                continue

            # Wait for resource to be running
            success = self.provider.wait_for_status(
                resource.resource_id,
                ResourceStatus.RUNNING,
                timeout_seconds=120
            )

            if not success:
                unhealthy_steps.append(step_name)

        if unhealthy_steps:
            raise RuntimeError(
                f"Deployment validation failed. Unhealthy steps: {unhealthy_steps}"
            )

    def _rollback_deployment(self, deployment: DeploymentState):
        """Rollback failed deployment"""
        logger.info(f"Rolling back deployment: {deployment.pipeline_name}")

        for step_name, resource in list(deployment.resources.items()):
            try:
                self.provider.delete_resource(resource.resource_id)
                logger.info(f"Rolled back resource: {step_name}")
            except Exception as e:
                logger.error(f"Failed to rollback resource {step_name}: {e}")

    def _estimate_resources(self, pipeline: PipelineGraph) -> List[CloudResource]:
        """Estimate resources for cost calculation"""
        resources = []

        for step_name, step in pipeline.steps.items():
            # Create mock resource for estimation
            resource = CloudResource(
                provider=self.provider.name,
                resource_type=ResourceType.CONTAINER,
                resource_id=f"estimate-{step_name}",
                resource_name=step_name,
                region=self.provider.regions[0],
                metadata={
                    "cpu": 1,
                    "memory": 2048,
                    "gpu": 0
                }
            )
            resources.append(resource)

        return resources