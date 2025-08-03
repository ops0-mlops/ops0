"""
ops0 Execution Engine - Orchestrates pipeline execution locally and remotely.
Handles dependency resolution, parallel execution, and error recovery.
"""

import asyncio
import concurrent.futures
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

from .graph import PipelineGraph, StepNode
from .storage import StorageLayer, get_storage
from .exceptions import ExecutionError, DependencyError
from .config import get_config

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Pipeline execution modes"""
    LOCAL = "local"
    REMOTE = "remote"
    HYBRID = "hybrid"


class StepStatus(Enum):
    """Step execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Result of step execution"""
    step_name: str
    status: StepStatus
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result of pipeline execution"""
    pipeline_name: str
    status: StepStatus
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    total_execution_time: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    failed_steps: List[str] = field(default_factory=list)
    skipped_steps: List[str] = field(default_factory=list)


class StepExecutor:
    """Executes individual steps with error handling and retries"""

    def __init__(self, storage: StorageLayer):
        self.storage = storage
        self.config = get_config()

    def execute_step(
            self,
            step_node: StepNode,
            inputs: Dict[str, Any] = None
    ) -> StepResult:
        """
        Execute a single step with error handling and retries.

        Args:
            step_node: Step to execute
            inputs: Input arguments for the step

        Returns:
            StepResult with execution details
        """
        result = StepResult(step_name=step_node.name, status=StepStatus.PENDING)
        inputs = inputs or {}

        max_retries = self.config.execution.max_retries

        for attempt in range(max_retries + 1):
            try:
                result.retry_count = attempt
                result.start_time = time.time()
                result.status = StepStatus.RUNNING

                logger.info(f"🚀 Executing step: {step_node.name} (attempt {attempt + 1})")

                # Prepare step inputs by loading dependencies
                step_inputs = self._prepare_step_inputs(step_node, inputs)

                # Execute the actual function
                step_result = step_node.func(**step_inputs)

                result.end_time = time.time()
                result.execution_time = result.end_time - result.start_time
                result.result = step_result
                result.status = StepStatus.COMPLETED

                logger.info(f"✅ Step completed: {step_node.name} ({result.execution_time:.2f}s)")
                return result

            except Exception as e:
                result.end_time = time.time()
                result.execution_time = result.end_time - result.start_time if result.start_time else 0
                result.error = e

                if attempt < max_retries:
                    logger.warning(f"⚠️ Step failed: {step_node.name} (attempt {attempt + 1}): {e}")
                    logger.info(f"🔄 Retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    result.status = StepStatus.FAILED
                    logger.error(f"❌ Step failed permanently: {step_node.name}: {e}")

        return result

    def _prepare_step_inputs(self, step_node: StepNode, provided_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs for step execution by loading dependencies"""
        step_inputs = provided_inputs.copy()

        # Load storage dependencies
        for storage_key in step_node.dependencies:
            if storage_key not in step_inputs:
                try:
                    step_inputs[storage_key] = self.storage.load(storage_key)
                except Exception as e:
                    raise DependencyError(
                        f"Failed to load dependency '{storage_key}' for step '{step_node.name}': {e}",
                        step_name=step_node.name,
                        missing_dependencies=[storage_key]
                    )

        return step_inputs


class PipelineExecutor:
    """
    Main pipeline execution engine.
    Handles dependency resolution, parallel execution, and monitoring.
    """

    def __init__(
            self,
            mode: ExecutionMode = ExecutionMode.LOCAL,
            storage: StorageLayer = None,
            max_workers: int = None
    ):
        self.mode = mode
        self.storage = storage or get_storage()
        self.step_executor = StepExecutor(self.storage)
        self.config = get_config()
        self.max_workers = max_workers or self.config.execution.max_parallel_steps

    def execute_pipeline(
            self,
            pipeline: PipelineGraph,
            inputs: Dict[str, Any] = None
    ) -> PipelineResult:
        """
        Execute a complete pipeline.

        Args:
            pipeline: Pipeline graph to execute
            inputs: Initial inputs for the pipeline

        Returns:
            PipelineResult with execution details
        """
        logger.info(f"🎯 Starting pipeline execution: {pipeline.name}")

        result = PipelineResult(
            pipeline_name=pipeline.name,
            status=StepStatus.RUNNING,
            start_time=time.time()
        )

        try:
            # Build execution order
            execution_batches = pipeline.build_execution_order()
            logger.info(f"📋 Execution plan: {len(execution_batches)} batches")

            # Execute batches sequentially, steps in parallel within batch
            for batch_idx, batch in enumerate(execution_batches):
                logger.info(f"🔄 Executing batch {batch_idx + 1}: {batch}")

                batch_results = self._execute_batch(pipeline, batch, inputs or {})

                # Update results
                result.step_results.update(batch_results)

                # Check for failures
                failed_in_batch = [
                    name for name, step_result in batch_results.items()
                    if step_result.status == StepStatus.FAILED
                ]

                if failed_in_batch:
                    result.failed_steps.extend(failed_in_batch)
                    result.status = StepStatus.FAILED
                    logger.error(f"❌ Batch failed due to: {failed_in_batch}")
                    break

            if result.status != StepStatus.FAILED:
                result.status = StepStatus.COMPLETED
                logger.info(f"🎉 Pipeline completed successfully: {pipeline.name}")

        except Exception as e:
            result.status = StepStatus.FAILED
            logger.error(f"❌ Pipeline execution failed: {e}")
            raise ExecutionError(f"Pipeline execution failed: {e}", execution_mode=self.mode.value)

        finally:
            result.end_time = time.time()
            result.total_execution_time = result.end_time - result.start_time

        return result

    def _execute_batch(
            self,
            pipeline: PipelineGraph,
            batch: List[str],
            inputs: Dict[str, Any]
    ) -> Dict[str, StepResult]:
        """Execute a batch of steps in parallel"""

        if len(batch) == 1:
            # Single step - execute directly
            step_name = batch[0]
            step_node = pipeline.steps[step_name]
            result = self.step_executor.execute_step(step_node, inputs)
            return {step_name: result}

        # Multiple steps - execute in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_step = {}

            for step_name in batch:
                step_node = pipeline.steps[step_name]
                future = executor.submit(self.step_executor.execute_step, step_node, inputs)
                future_to_step[future] = step_name

            # Collect results
            results = {}
            for future in concurrent.futures.as_completed(future_to_step):
                step_name = future_to_step[future]
                try:
                    results[step_name] = future.result()
                except Exception as e:
                    # Create failed result
                    results[step_name] = StepResult(
                        step_name=step_name,
                        status=StepStatus.FAILED,
                        error=e
                    )

            return results


# High-level execution functions
def run(
        pipeline_name: str = None,
        local: bool = True,
        inputs: Dict[str, Any] = None,
        **kwargs
) -> PipelineResult:
    """
    Run a pipeline locally or remotely.

    Args:
        pipeline_name: Name of pipeline to run (uses current context if None)
        local: Run locally (True) or remotely (False)
        inputs: Input data for the pipeline
        **kwargs: Additional execution options

    Returns:
        PipelineResult with execution details
    """
    from .graph import PipelineGraph

    # Get pipeline from context or by name
    pipeline = PipelineGraph.get_current()
    if not pipeline and pipeline_name:
        raise ExecutionError(f"Pipeline '{pipeline_name}' not found")
    elif not pipeline:
        raise ExecutionError("No pipeline context available")

    # Choose execution mode
    mode = ExecutionMode.LOCAL if local else ExecutionMode.REMOTE

    # Create executor and run
    executor = PipelineExecutor(mode=mode, **kwargs)
    return executor.execute_pipeline(pipeline, inputs)


def deploy(
        name: str = None,
        environment: str = "production",
        **kwargs
) -> Dict[str, Any]:
    """
    Deploy pipeline to remote environment.

    Args:
        name: Pipeline name
        environment: Target environment (staging, production)
        **kwargs: Deployment options

    Returns:
        Deployment result information
    """
    logger.info(f"🚀 Deploying pipeline: {name} to {environment}")

    # This would integrate with cloud deployment in full implementation
    deployment_result = {
        "pipeline_name": name,
        "environment": environment,
        "status": "deployed",
        "endpoint": f"https://{name}.ops0.xyz",
        "deployment_id": f"deploy-{int(time.time())}",
        "message": "Pipeline deployed successfully"
    }

    logger.info(f"✅ Deployment completed: {deployment_result['endpoint']}")
    return deployment_result