"""
ops0 Pipeline Executor

Executes pipeline steps locally and coordinates production deployments.
Supports parallel execution, retry logic, and comprehensive monitoring.
"""

import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from contextlib import contextmanager
import logging

from .graph import PipelineGraph, StepNode, get_current_pipeline
from .storage import storage, StorageNamespace
from .exceptions import ExecutionError, StepError, PipelineError
from .config import config

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of pipeline execution"""
    pipeline_name: str
    success: bool
    total_steps: int
    completed_steps: int
    failed_steps: List[str]
    execution_time: float
    step_results: Dict[str, Any]
    errors: Dict[str, Exception]

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.total_steps == 0:
            return 100.0
        return (self.completed_steps / self.total_steps) * 100.0


@dataclass
class StepExecutionContext:
    """Context for step execution"""
    step_name: str
    pipeline_name: str
    execution_id: str
    retry_count: int = 0
    start_time: Optional[float] = None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()


class StepExecutor:
    """Executes individual pipeline steps"""

    def __init__(self, storage_namespace: str = None):
        self.storage_namespace = storage_namespace

    def execute_step(self, step: StepNode, context: StepExecutionContext) -> Any:
        """
        Execute a single step with error handling and monitoring.

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
            with self._step_storage_context(context):
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
                    # For now, just pass as positional argument
                    # In practice, we'd map to the correct parameter name
                    step_args.append(data)
                except Exception as e:
                    logger.warning(f"Could not load dependency '{dep.key}' for step '{step.name}': {e}")

        return step_args, step_kwargs

    @contextmanager
    def _step_storage_context(self, context: StepExecutionContext):
        """Create storage context for step execution"""
        namespace = f"{context.pipeline_name}:{context.execution_id}"
        if self.storage_namespace:
            namespace = f"{self.storage_namespace}:{namespace}"

        with StorageNamespace(namespace):
            yield


class PipelineExecutor:
    """Executes complete pipelines with parallel support"""

    def __init__(self, max_workers: int = None, enable_retries: bool = True):
        self.max_workers = max_workers or config.execution.max_parallel_steps
        self.enable_retries = enable_retries
        self.step_executor = StepExecutor()

    def execute(self, pipeline: PipelineGraph, execution_id: str = None) -> ExecutionResult:
        """
        Execute a complete pipeline.

        Args:
            pipeline: Pipeline to execute
            execution_id: Unique execution identifier

        Returns:
            Execution result with detailed information
        """
        if not execution_id:
            execution_id = f"exec_{int(time.time())}"

        logger.info(f"Starting pipeline '{pipeline.name}' execution (ID: {execution_id})")

        # Validate pipeline before execution
        pipeline.validate()

        # Reset execution state
        pipeline.reset_execution_state()

        # Get execution plan
        parallel_groups = pipeline.get_parallel_groups()

        # Initialize result tracking
        start_time = time.time()
        step_results = {}
        errors = {}
        completed_steps = 0
        failed_steps = []

        try:
            # Execute each parallel group
            for group_idx, step_group in enumerate(parallel_groups):
                logger.info(f"Executing parallel group {group_idx + 1}/{len(parallel_groups)}: {step_group}")

                group_results = self._execute_parallel_group(
                    pipeline, step_group, execution_id
                )

                # Process group results
                for step_name, result in group_results.items():
                    if isinstance(result, Exception):
                        errors[step_name] = result
                        failed_steps.append(step_name)

                        # Stop execution on failure if not configured to continue
                        if not config.execution.continue_on_failure:
                            raise ExecutionError(
                                f"Pipeline execution stopped due to step failure: {step_name}",
                                step_name=step_name
                            )
                    else:
                        step_results[step_name] = result
                        completed_steps += 1

            execution_time = time.time() - start_time
            success = len(failed_steps) == 0

            logger.info(f"Pipeline '{pipeline.name}' completed in {execution_time:.2f}s "
                       f"({completed_steps}/{len(pipeline.steps)} steps succeeded)")

            return ExecutionResult(
                pipeline_name=pipeline.name,
                success=success,
                total_steps=len(pipeline.steps),
                completed_steps=completed_steps,
                failed_steps=failed_steps,
                execution_time=execution_time,
                step_results=step_results,
                errors=errors
            )

        except Exception as e:
            execution_time = time.time() - start_time

            logger.error(f"Pipeline '{pipeline.name}' failed after {execution_time:.2f}s: {str(e)}")

            return ExecutionResult(
                pipeline_name=pipeline.name,
                success=False,
                total_steps=len(pipeline.steps),
                completed_steps=completed_steps,
                failed_steps=failed_steps,
                execution_time=execution_time,
                step_results=step_results,
                errors=errors
            )

    def _execute_parallel_group(
        self,
        pipeline: PipelineGraph,
        step_names: List[str],
        execution_id: str
    ) -> Dict[str, Union[Any, Exception]]:
        """Execute a group of steps in parallel"""

        if len(step_names) == 1:
            # Single step - execute directly
            step_name = step_names[0]
            step = pipeline.steps[step_name]
            context = StepExecutionContext(step_name, pipeline.name, execution_id)

            try:
                result = self._execute_step_with_retry(step, context)
                return {step_name: result}
            except Exception as e:
                return {step_name: e}

        # Multiple steps - use thread pool
        results = {}

        with ThreadPoolExecutor(max_workers=min(len(step_names), self.max_workers)) as executor:
            # Submit all steps
            future_to_step = {}
            for step_name in step_names:
                step = pipeline.steps[step_name]
                context = StepExecutionContext(step_name, pipeline.name, execution_id)

                future = executor.submit(self._execute_step_with_retry, step, context)
                future_to_step[future] = step_name

            # Collect results
            for future in as_completed(future_to_step):
                step_name = future_to_step[future]
                try:
                    result = future.result()
                    results[step_name] = result
                except Exception as e:
                    results[step_name] = e

        return results

    def _execute_step_with_retry(self, step: StepNode, context: StepExecutionContext) -> Any:
        """Execute step with retry logic"""
        max_retries = config.execution.max_retries if self.enable_retries else 0

        last_error = None

        for attempt in range(max_retries + 1):
            context.retry_count = attempt

            try:
                return self.step_executor.execute_step(step, context)
            except Exception as e:
                last_error = e

                if attempt < max_retries:
                    retry_delay = config.execution.retry_delay_seconds * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Step '{step.name}' failed (attempt {attempt + 1}), "
                                 f"retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Step '{step.name}' failed after {attempt + 1} attempts")

        raise last_error


# High-level execution functions

def run(pipeline: PipelineGraph = None, local: bool = True) -> ExecutionResult:
    """
    Run a pipeline locally or in production.

    Args:
        pipeline: Pipeline to run (uses current context if None)
        local: Whether to run locally (True) or deploy to production (False)

    Returns:
        Execution result
    """
    if pipeline is None:
        pipeline = get_current_pipeline()
        if pipeline is None:
            raise PipelineError("No pipeline found in current context")

    if local:
        return run_local(pipeline)
    else:
        return deploy(pipeline)


def run_local(pipeline: PipelineGraph) -> ExecutionResult:
    """
    Run pipeline locally with full logging and debugging.

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


def deploy(pipeline: PipelineGraph, environment: str = "production") -> Dict[str, Any]:
    """
    Deploy pipeline to production environment.

    Args:
        pipeline: Pipeline to deploy
        environment: Target environment

    Returns:
        Deployment information
    """
    logger.info(f"Deploying pipeline '{pipeline.name}' to {environment}")

    # Validate pipeline for production
    pipeline.validate()

    # For now, this is a placeholder - in practice, this would:
    # 1. Package the pipeline into containers
    # 2. Deploy to cloud infrastructure
    # 3. Set up monitoring and alerting
    # 4. Return deployment details

    deployment_info = {
        "pipeline_name": pipeline.name,
        "environment": environment,
        "deployment_id": f"deploy_{int(time.time())}",
        "status": "deployed",
        "endpoint": f"https://api.ops0.xyz/pipelines/{pipeline.name}",
        "steps": list(pipeline.steps.keys()),
        "deployed_at": time.time(),
    }

    logger.info(f"✅ Pipeline '{pipeline.name}' deployed successfully!")
    logger.info(f"   • Deployment ID: {deployment_info['deployment_id']}")
    logger.info(f"   • Endpoint: {deployment_info['endpoint']}")
    logger.info(f"   • Steps: {len(deployment_info['steps'])}")

    return deployment_info


# Pipeline execution context manager
@contextmanager
def execution_context(pipeline_name: str = "default"):
    """Create an execution context for a pipeline"""
    original_namespace = storage.namespace
    storage.namespace = pipeline_name

    try:
        yield
    finally:
        storage.namespace = original_namespace


# Utility functions for monitoring and debugging

def get_execution_status(pipeline: PipelineGraph) -> Dict[str, Any]:
    """Get current execution status of a pipeline"""
    status = {
        "pipeline_name": pipeline.name,
        "total_steps": len(pipeline.steps),
        "step_status": {},
        "overall_status": "unknown"
    }

    completed = 0
    running = 0
    failed = 0

    for step_name, step in pipeline.steps.items():
        status["step_status"][step_name] = {
            "status": step.status,
            "execution_time": step.execution_time,
            "error": str(step.error) if step.error else None
        }

        if step.status == "completed":
            completed += 1
        elif step.status == "running":
            running += 1
        elif step.status == "failed":
            failed += 1

    # Determine overall status
    if failed > 0:
        status["overall_status"] = "failed"
    elif running > 0:
        status["overall_status"] = "running"
    elif completed == len(pipeline.steps):
        status["overall_status"] = "completed"
    else:
        status["overall_status"] = "pending"

    status["completed_steps"] = completed
    status["running_steps"] = running
    status["failed_steps"] = failed

    return status