"""
ops0 Pipeline Executor

Executes pipeline steps locally or in distributed environments.
Handles both development and production deployment scenarios.
"""

import time
import logging
import traceback
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import threading
import os

from .graph import PipelineGraph, StepNode
from .storage import storage, with_namespace
from .exceptions import ExecutionError, StepError, StorageError
from .config import config

logger = logging.getLogger(__name__)


@dataclass
class StepExecutionContext:
    """Context for step execution"""
    pipeline_name: str
    step_name: str
    execution_id: str
    retry_count: int = 0
    namespace: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of pipeline execution"""
    success: bool
    pipeline_name: str
    execution_id: str
    completed_steps: int
    total_steps: int
    failed_steps: List[str] = field(default_factory=list)
    step_results: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, Exception] = field(default_factory=dict)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# Thread-local storage for execution context
_execution_context = threading.local()


@contextmanager
def execution_context(context: StepExecutionContext):
    """Context manager for step execution"""
    _execution_context.context = context
    try:
        yield context
    finally:
        if hasattr(_execution_context, 'context'):
            delattr(_execution_context, 'context')


class PipelineExecutor:
    """Executes pipeline steps in the correct order"""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or config.execution.max_parallel_steps

    def execute(self, pipeline: PipelineGraph, mode: str = "sequential") -> ExecutionResult:
        """
        Execute a pipeline.

        Args:
            pipeline: Pipeline to execute
            mode: Execution mode (sequential, parallel)

        Returns:
            ExecutionResult with details
        """
        # Validate pipeline first
        pipeline.validate()

        # Generate execution ID
        execution_id = f"{pipeline.name}_{int(time.time())}"
        logger.info(f"Starting pipeline execution: {execution_id}")

        # Initialize result
        result = ExecutionResult(
            success=True,
            pipeline_name=pipeline.name,
            execution_id=execution_id,
            completed_steps=0,
            total_steps=len(pipeline.steps)
        )

        # Reset execution state
        pipeline.reset_execution_state()

        # Create execution namespace for storage isolation
        namespace = f"exec_{execution_id}"

        start_time = time.time()

        try:
            if mode == "sequential":
                self._execute_sequential(pipeline, result, execution_id, namespace)
            elif mode == "parallel":
                self._execute_parallel(pipeline, result, execution_id, namespace)
            else:
                raise ValueError(f"Unknown execution mode: {mode}")

        except Exception as e:
            result.success = False
            logger.error(f"Pipeline execution failed: {str(e)}")

        finally:
            result.execution_time = time.time() - start_time

        # Log execution summary
        self._log_execution_summary(result)

        return result

    def _execute_sequential(self, pipeline: PipelineGraph, result: ExecutionResult,
                          execution_id: str, namespace: str) -> None:
        """Execute steps sequentially"""
        execution_order = pipeline.get_execution_order()

        for step_name in execution_order:
            step = pipeline.steps[step_name]
            context = StepExecutionContext(
                pipeline_name=pipeline.name,
                step_name=step_name,
                execution_id=execution_id,
                namespace=namespace
            )

            try:
                with with_namespace(namespace):
                    step_result = self._execute_step(step, context)
                    result.step_results[step_name] = step_result
                    result.completed_steps += 1

            except Exception as e:
                result.success = False
                result.failed_steps.append(step_name)
                result.errors[step_name] = e

                if not config.execution.continue_on_failure:
                    raise

    def _execute_parallel(self, pipeline: PipelineGraph, result: ExecutionResult,
                         execution_id: str, namespace: str) -> None:
        """Execute steps in parallel where possible"""
        completed_steps = set()
        pending_steps = set(pipeline.steps.keys())

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}

            while pending_steps:
                # Find steps ready to execute
                ready_steps = [
                    step_name for step_name in pending_steps
                    if pipeline.steps[step_name].can_execute(completed_steps)
                ]

                if not ready_steps and futures:
                    # Wait for at least one future to complete
                    done, _ = as_completed(futures, timeout=None)
                    for future in done:
                        step_name = futures[future]
                        try:
                            step_result = future.result()
                            result.step_results[step_name] = step_result
                            result.completed_steps += 1
                            completed_steps.add(step_name)
                        except Exception as e:
                            result.success = False
                            result.failed_steps.append(step_name)
                            result.errors[step_name] = e
                            if not config.execution.continue_on_failure:
                                raise
                        del futures[future]
                        pending_steps.remove(step_name)

                elif not ready_steps and not futures:
                    # Deadlock - no steps can execute
                    raise ExecutionError(
                        "Pipeline execution deadlock - no steps can execute",
                        context={"pending": list(pending_steps)}
                    )

                # Submit ready steps for execution
                for step_name in ready_steps:
                    step = pipeline.steps[step_name]
                    context = StepExecutionContext(
                        pipeline_name=pipeline.name,
                        step_name=step_name,
                        execution_id=execution_id,
                        namespace=namespace
                    )

                    future = executor.submit(self._execute_step_with_namespace,
                                           step, context, namespace)
                    futures[future] = step_name

    def _execute_step_with_namespace(self, step: StepNode, context: StepExecutionContext,
                                    namespace: str) -> Any:
        """Execute step within a storage namespace"""
        with with_namespace(namespace):
            return self._execute_step(step, context)

    def _execute_step(self, step: StepNode, context: StepExecutionContext) -> Any:
        """
        Execute a single step.

        Args:
            step: Step to execute
            context: Execution context

        Returns:
            Step execution result
        """
        logger.info(f"Executing step '{step.name}' (attempt {context.retry_count + 1})")

        step.status = "running"
        start_time = time.time()

        try:
            # Prepare step arguments
            step_args, step_kwargs = self._prepare_step_arguments(step, context)

            # Execute the step function
            with execution_context(context):
                result = step.func(*step_args, **step_kwargs)

            # Record success
            execution_time = time.time() - start_time
            step.status = "completed"
            step.result = result
            step.execution_time = execution_time

            logger.info(f"Step '{step.name}' completed in {execution_time:.2f}s")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            step.status = "failed"
            step.error = e
            step.execution_time = execution_time

            logger.error(f"Step '{step.name}' failed after {execution_time:.2f}s: {str(e)}")

            # Re-raise with context
            raise StepError(
                f"Step '{step.name}' execution failed",
                step_name=step.name,
                context={
                    "error": str(e),
                    "execution_time": execution_time,
                    "retry_count": context.retry_count,
                    "traceback": traceback.format_exc()
                }
            ) from e

    def _prepare_step_arguments(self, step: StepNode, context: StepExecutionContext) -> tuple:
        """Prepare arguments for step execution"""
        signature = step.input_signature
        step_args = []
        step_kwargs = {}

        # For now, we'll use a simple approach - in practice, this would be more sophisticated
        # This would involve analyzing parameter names and types to determine how to pass data

        # If step has parameters that match storage keys, load them
        storage_deps = step.storage_dependencies
        for dep in storage_deps:
            if dep.operation == 'load':
                # This is simplified - in practice, we'd map storage keys to parameter names
                try:
                    data = storage.load(dep.key)
                    step_kwargs[dep.key] = data
                except StorageError:
                    logger.warning(f"Could not load dependency '{dep.key}' for step '{step.name}'")

        return step_args, step_kwargs

    def _log_execution_summary(self, result: ExecutionResult) -> None:
        """Log execution summary"""
        if result.success:
            logger.info(
                f"Pipeline '{result.pipeline_name}' completed successfully: "
                f"{result.completed_steps}/{result.total_steps} steps in {result.execution_time:.2f}s"
            )
        else:
            logger.error(
                f"Pipeline '{result.pipeline_name}' failed: "
                f"{result.completed_steps}/{result.total_steps} steps completed, "
                f"{len(result.failed_steps)} failed"
            )


# Public API functions

def run(pipeline: PipelineGraph) -> ExecutionResult:
    """
    Run pipeline locally.

    Args:
        pipeline: Pipeline to execute

    Returns:
        Execution result
    """
    logger.info(f"Running pipeline '{pipeline.name}' locally")

    executor = PipelineExecutor()
    result = executor.execute(pipeline)

    # Print summary
    if result.success:
        logger.info(f"✅ Pipeline '{pipeline.name}' completed successfully!")
        logger.info(f"   • {result.completed_steps}/{result.total_steps} steps completed")
        logger.info(f"   • Execution time: {result.execution_time:.2f}s")
    else:
        logger.error(f"❌ Pipeline '{pipeline.name}' failed!")
        logger.error(f"   • {result.completed_steps}/{result.total_steps} steps completed")
        logger.error(f"   • Failed steps: {', '.join(result.failed_steps)}")
        logger.error(f"   • Execution time: {result.execution_time:.2f}s")

        # Log detailed errors
        for step_name, error in result.errors.items():
            logger.error(f"   • {step_name}: {str(error)}")

    return result


def deploy(pipeline: PipelineGraph, target: str = "auto", **kwargs) -> Dict[str, Any]:
    """
    Deploy pipeline to production environment.

    Args:
        pipeline: Pipeline to deploy
        target: Deployment target (auto, local, docker, cloud, k8s)
        **kwargs: Additional deployment options

    Returns:
        Deployment information
    """
    logger.info(f"Deploying pipeline '{pipeline.name}' to {target}")

    # Validate pipeline for production
    pipeline.validate()

    deployment_info = {
        "pipeline_name": pipeline.name,
        "deployment_id": f"deploy_{int(time.time())}",
        "target": target,
        "timestamp": time.time(),
    }

    try:
        # Auto-detect best deployment target
        if target == "auto":
            target = _detect_deployment_target()
            deployment_info["target"] = target

        if target == "local":
            deployment_info.update(_deploy_local(pipeline, **kwargs))

        elif target == "docker":
            # Use existing container orchestrator
            from ..runtime.containers import container_orchestrator
            deployment_info.update(_deploy_docker(pipeline, container_orchestrator, **kwargs))

        elif target == "cloud":
            # Use existing cloud orchestrator
            from ..cloud import orchestrator as cloud_orchestrator
            deployment_info.update(_deploy_cloud(pipeline, cloud_orchestrator, **kwargs))

        elif target == "k8s":
            deployment_info.update(_deploy_kubernetes(pipeline, **kwargs))

        else:
            raise ValueError(f"Unknown deployment target: {target}")

        deployment_info["status"] = "deployed"
        deployment_info["success"] = True

        logger.info(f"✅ Pipeline '{pipeline.name}' deployed successfully to {target}!")
        logger.info(f"   • Deployment ID: {deployment_info['deployment_id']}")
        logger.info(f"   • Endpoint: {deployment_info.get('endpoint', 'N/A')}")

    except Exception as e:
        deployment_info["status"] = "failed"
        deployment_info["success"] = False
        deployment_info["error"] = str(e)

        logger.error(f"❌ Deployment failed: {str(e)}")
        raise

    return deployment_info


def _detect_deployment_target() -> str:
    """Detect the best deployment target based on environment"""
    # Check for Kubernetes
    if os.getenv("KUBERNETES_SERVICE_HOST"):
        return "k8s"

    # Check for cloud providers
    if os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
        return "cloud"

    if os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT"):
        return "cloud"

    if os.getenv("AZURE_SUBSCRIPTION_ID"):
        return "cloud"

    # Check for Docker
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return "docker"
    except:
        pass

    # Default to local
    return "local"


def _deploy_local(pipeline: PipelineGraph, **kwargs) -> Dict[str, Any]:
    """Deploy pipeline locally for development"""
    return {
        "deployment_type": "local",
        "endpoint": f"http://localhost:8000/pipelines/{pipeline.name}",
        "steps": list(pipeline.steps.keys()),
        "message": "Pipeline running locally. Use 'ops0 run' to execute."
    }


def _deploy_docker(pipeline: PipelineGraph, container_orchestrator, **kwargs) -> Dict[str, Any]:
    """Deploy pipeline as Docker containers"""
    logger.info("Containerizing pipeline steps...")

    # Containerize all steps
    container_specs = container_orchestrator.containerize_pipeline(pipeline)

    # Generate deployment manifest
    manifest = container_orchestrator.get_container_manifest()

    # Build and push containers if requested
    if kwargs.get("build", True):
        for step_name, spec in container_specs.items():
            logger.info(f"Building container for step: {step_name}")
            if os.getenv("OPS0_BUILD_CONTAINERS", "false").lower() == "true":
                container_orchestrator.builder.build_container(spec, push=kwargs.get("push", False))

    # Export Docker Compose file
    compose_file = "docker-compose.yml"
    if kwargs.get("compose", True):
        compose_file = container_orchestrator.export_compose_file(pipeline.name, compose_file)

    deployment_info = {
        "deployment_type": "docker",
        "containers": list(container_specs.keys()),
        "manifest": manifest,
        "compose_file": compose_file,
        "endpoint": f"http://localhost:8080/pipelines/{pipeline.name}",
        "message": f"Run 'docker-compose up' to start the pipeline"
    }

    return deployment_info


def _deploy_cloud(pipeline: PipelineGraph, cloud_orchestrator, **kwargs) -> Dict[str, Any]:
    """Deploy pipeline to cloud provider"""
    # Determine provider
    provider = kwargs.get("provider")
    if not provider:
        # Auto-detect from environment
        if os.getenv("AWS_DEFAULT_REGION"):
            provider = "aws"
        elif os.getenv("GOOGLE_CLOUD_PROJECT"):
            provider = "gcp"
        elif os.getenv("AZURE_SUBSCRIPTION_ID"):
            provider = "azure"
        else:
            provider = "aws"  # Default

    logger.info(f"Deploying to cloud provider: {provider}")

    # Deploy using cloud orchestrator
    deployment_state = cloud_orchestrator.deploy(
        pipeline_name=pipeline.name,
        pipeline=pipeline,
        environment=kwargs.get("environment", {}),
        **kwargs
    )

    # Convert deployment state to our format
    return {
        "deployment_type": "cloud",
        "provider": provider,
        "deployment_id": deployment_state.deployment_id,
        "endpoint": deployment_state.get_endpoint() if hasattr(deployment_state, 'get_endpoint') else "N/A",
        "resources": {
            step: resource.to_dict()
            for step, resource in deployment_state.resources.items()
        },
        "status": deployment_state.status,
        "message": f"Pipeline deployed to {provider.upper()}"
    }


def _deploy_kubernetes(pipeline: PipelineGraph, **kwargs) -> Dict[str, Any]:
    """Deploy pipeline to Kubernetes cluster"""
    logger.info("Deploying to Kubernetes...")

    # Use Kubernetes provider from cloud module
    from ..cloud.kubernetes import KubernetesProvider

    k8s_provider = KubernetesProvider()

    # Deploy each step
    resources = {}
    for step_name, step_node in pipeline.steps.items():
        # Create deployment spec
        from ..cloud.base import DeploymentSpec
        spec = DeploymentSpec(
            step_name=step_name,
            image=f"ops0/{pipeline.name}-{step_name}:latest",
            cpu=1.0,
            memory=2048,
            min_instances=kwargs.get("min_instances", 1),
            max_instances=kwargs.get("max_instances", 10)
        )

        # Deploy
        resource = k8s_provider.deploy_container(spec)
        resources[step_name] = resource

    return {
        "deployment_type": "kubernetes",
        "namespace": kwargs.get("namespace", "ops0"),
        "resources": {name: res.to_dict() for name, res in resources.items()},
        "endpoint": f"http://ops0-{pipeline.name}.{kwargs.get('namespace', 'ops0')}.svc.cluster.local",
        "message": "Pipeline deployed to Kubernetes cluster"
    }


# Convenience functions

def run_local(pipeline: PipelineGraph) -> ExecutionResult:
    """Alias for run() - run pipeline locally"""
    return run(pipeline)
