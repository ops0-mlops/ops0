"""
ops0 Runtime Orchestrator

Distributed execution engine that manages step execution across multiple workers.
Handles queuing, scheduling, and resource allocation automatically.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from queue import Queue, Empty
from threading import Lock, Thread
from typing import Dict, List, Optional, Any, Tuple, Callable, Union

# Handle imports for both development and production
try:
    from ..core.graph import PipelineGraph, StepNode
    from ..core.storage import storage
    from ..core.executor import ExecutionError
    from ..core.config import config
except ImportError:
    from ..core.graph import PipelineGraph, StepNode
    from ..core.storage import storage
    from ..core.executor import ExecutionError
    from ..core.config import config

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class ExecutionMode(Enum):
    """Pipeline execution modes"""
    LOCAL = "local"
    DISTRIBUTED = "distributed"
    KUBERNETES = "kubernetes"
    SERVERLESS = "serverless"


@dataclass
class Job:
    """Represents a single step execution job"""
    id: str
    pipeline_id: str
    step_name: str
    step_func: Callable
    args: List[Any]
    kwargs: Dict[str, Any]
    status: JobStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result: Any = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 300  # 5 minutes default
    resources: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for serialization"""
        return {
            "id": self.id,
            "pipeline_id": self.pipeline_id,
            "step_name": self.step_name,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "retry_count": self.retry_count,
            "timeout": self.timeout,
            "resources": self.resources
        }


class JobQueue:
    """Thread-safe job queue with priority support"""

    def __init__(self):
        self._queue = Queue()
        self._priority_queue = Queue()
        self._lock = Lock()
        self._job_map: Dict[str, Job] = {}

    def put(self, job: Job, priority: bool = False) -> None:
        """Add job to queue"""
        with self._lock:
            self._job_map[job.id] = job
            if priority:
                self._priority_queue.put(job)
            else:
                self._queue.put(job)

    def get(self, timeout: Optional[float] = None) -> Optional[Job]:
        """Get next job from queue (priority first)"""
        try:
            # Check priority queue first
            if not self._priority_queue.empty():
                return self._priority_queue.get(timeout=timeout)
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get specific job by ID"""
        with self._lock:
            return self._job_map.get(job_id)

    def update_job(self, job: Job) -> None:
        """Update job status"""
        with self._lock:
            self._job_map[job.id] = job

    def size(self) -> int:
        """Get total queue size"""
        return self._queue.qsize() + self._priority_queue.qsize()

    def pending_jobs(self) -> List[Job]:
        """Get all pending jobs"""
        with self._lock:
            return [j for j in self._job_map.values() if j.status == JobStatus.PENDING]


class Worker:
    """Worker that executes jobs from the queue"""

    def __init__(self, worker_id: str, job_queue: JobQueue, orchestrator: 'RuntimeOrchestrator'):
        self.worker_id = worker_id
        self.job_queue = job_queue
        self.orchestrator = orchestrator
        self.is_running = False
        self._thread = None

    def start(self) -> None:
        """Start worker thread"""
        self.is_running = True
        self._thread = Thread(target=self._run, name=f"Worker-{self.worker_id}")
        self._thread.start()

    def stop(self) -> None:
        """Stop worker thread"""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        """Main worker loop"""
        logger.info(f"Worker {self.worker_id} started")

        while self.is_running:
            job = self.job_queue.get(timeout=1)
            if job:
                self._execute_job(job)

        logger.info(f"Worker {self.worker_id} stopped")

    def _execute_job(self, job: Job) -> None:
        """Execute a single job"""
        logger.info(f"Worker {self.worker_id} executing job {job.id} ({job.step_name})")

        # Update job status
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        self.job_queue.update_job(job)

        try:
            # Execute the step function
            result = self._run_step(job)

            # Update job with result
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            job.result = result
            self.job_queue.update_job(job)

            # Notify orchestrator
            self.orchestrator._on_job_completed(job)

            logger.info(f"Job {job.id} completed successfully")

        except Exception as e:
            # Handle job failure
            job.error = str(e)
            job.completed_at = time.time()

            if job.retry_count < job.max_retries:
                job.status = JobStatus.RETRYING
                job.retry_count += 1
                logger.warning(f"Job {job.id} failed, retrying ({job.retry_count}/{job.max_retries})")
                # Re-queue for retry
                self.job_queue.put(job, priority=True)
            else:
                job.status = JobStatus.FAILED
                logger.error(f"Job {job.id} failed after {job.retry_count} retries: {e}")
                # Notify orchestrator
                self.orchestrator._on_job_failed(job)

            self.job_queue.update_job(job)

    def _run_step(self, job: Job) -> Any:
        """Execute the actual step function"""
        # Set up execution context
        storage.namespace = job.pipeline_id

        # Execute with timeout
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Job {job.id} exceeded timeout of {job.timeout}s")

        if hasattr(signal, 'SIGALRM'):  # Unix only
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(job.timeout)

        try:
            # Execute step function
            result = job.step_func(*job.args, **job.kwargs)
            return result
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # Cancel alarm


class RuntimeOrchestrator:
    """
    Main orchestration engine for distributed pipeline execution.

    Manages job scheduling, worker pools, and execution coordination.
    """

    def __init__(self, mode: ExecutionMode = ExecutionMode.LOCAL):
        self.mode = mode
        self.job_queue = JobQueue()
        self.workers: List[Worker] = []
        self.active_pipelines: Dict[str, PipelineGraph] = {}
        self.execution_results: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

        # Configure based on mode
        if mode == ExecutionMode.LOCAL:
            self.worker_count = min(4, os.cpu_count() or 1)
            self.executor = ThreadPoolExecutor(max_workers=self.worker_count)
        elif mode == ExecutionMode.DISTRIBUTED:
            self.worker_count = config.execution.max_parallel_steps or 10
            self.executor = ProcessPoolExecutor(max_workers=self.worker_count)
        else:
            raise NotImplementedError(f"Execution mode {mode} not yet implemented")

        # Initialize workers
        self._init_workers()

    def _init_workers(self) -> None:
        """Initialize worker pool"""
        for i in range(self.worker_count):
            worker = Worker(f"worker-{i}", self.job_queue, self)
            self.workers.append(worker)
            worker.start()

        logger.info(f"Initialized {self.worker_count} workers in {self.mode.value} mode")

    def execute_pipeline(self, pipeline: PipelineGraph, execution_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a pipeline with distributed orchestration.

        Args:
            pipeline: The pipeline to execute
            execution_id: Optional execution ID

        Returns:
            Execution results
        """
        if not execution_id:
            execution_id = f"exec-{uuid.uuid4().hex[:8]}"

        logger.info(f"Starting distributed execution of pipeline '{pipeline.name}' (ID: {execution_id})")

        # Register pipeline
        self.active_pipelines[execution_id] = pipeline
        self.execution_results[execution_id] = {
            "status": "running",
            "started_at": time.time(),
            "completed_steps": [],
            "failed_steps": [],
            "results": {}
        }

        # Schedule all ready steps
        self._schedule_ready_steps(pipeline, execution_id)

        # Wait for completion
        result = self._wait_for_completion(execution_id, pipeline)

        # Cleanup
        del self.active_pipelines[execution_id]

        return result

    def _schedule_ready_steps(self, pipeline: PipelineGraph, execution_id: str) -> None:
        """Schedule all steps that are ready to execute"""
        ready_steps = self._get_ready_steps(pipeline, execution_id)

        for step_name in ready_steps:
            step_node = pipeline.steps[step_name]

            # Create job
            job = Job(
                id=f"{execution_id}-{step_name}-{uuid.uuid4().hex[:8]}",
                pipeline_id=execution_id,
                step_name=step_name,
                step_func=step_node.func,
                args=[],  # Will be populated with dependency results
                kwargs={},
                status=JobStatus.QUEUED,
                created_at=time.time(),
                resources=self._estimate_resources(step_node)
            )

            # Resolve dependencies
            if step_node.dependencies:
                dep_results = []
                for dep in step_node.dependencies:
                    dep_result = self.execution_results[execution_id]["results"].get(dep)
                    if dep_result is not None:
                        dep_results.append(dep_result)
                job.args = dep_results

            # Queue job
            self.job_queue.put(job)
            logger.info(f"Scheduled job for step '{step_name}'")

    def _get_ready_steps(self, pipeline: PipelineGraph, execution_id: str) -> List[str]:
        """Get steps that are ready to execute"""
        ready = []
        completed = set(self.execution_results[execution_id]["completed_steps"])
        failed = set(self.execution_results[execution_id]["failed_steps"])

        for step_name, step_node in pipeline.steps.items():
            if step_name in completed or step_name in failed:
                continue

            # Check if all dependencies are completed
            deps_ready = all(dep in completed for dep in step_node.dependencies)
            if deps_ready:
                ready.append(step_name)

        return ready

    def _estimate_resources(self, step_node: StepNode) -> Dict[str, Any]:
        """Estimate resource requirements for a step"""
        # In production, this would analyze the step to determine resources
        return {
            "cpu": "1",
            "memory": "2Gi",
            "gpu": 0
        }

    def _wait_for_completion(self, execution_id: str, pipeline: PipelineGraph) -> Dict[str, Any]:
        """Wait for pipeline execution to complete"""
        total_steps = len(pipeline.steps)

        while True:
            with self._lock:
                result = self.execution_results[execution_id]
                completed = len(result["completed_steps"])
                failed = len(result["failed_steps"])

                # Check if all steps are done
                if completed + failed >= total_steps:
                    result["status"] = "failed" if failed > 0 else "completed"
                    result["completed_at"] = time.time()
                    result["execution_time"] = result["completed_at"] - result["started_at"]
                    return result

            # Small sleep to avoid busy waiting
            time.sleep(0.1)

    def _on_job_completed(self, job: Job) -> None:
        """Handle job completion"""
        with self._lock:
            if job.pipeline_id in self.execution_results:
                result = self.execution_results[job.pipeline_id]
                result["completed_steps"].append(job.step_name)
                result["results"][job.step_name] = job.result

                # Schedule newly ready steps
                pipeline = self.active_pipelines.get(job.pipeline_id)
                if pipeline:
                    self._schedule_ready_steps(pipeline, job.pipeline_id)

    def _on_job_failed(self, job: Job) -> None:
        """Handle job failure"""
        with self._lock:
            if job.pipeline_id in self.execution_results:
                result = self.execution_results[job.pipeline_id]
                result["failed_steps"].append(job.step_name)

                # In production, this could trigger failure handling
                logger.error(f"Step '{job.step_name}' failed in pipeline {job.pipeline_id}")

    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get current execution status"""
        with self._lock:
            return self.execution_results.get(execution_id)

    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution"""
        with self._lock:
            if execution_id in self.active_pipelines:
                # Mark as cancelled
                self.execution_results[execution_id]["status"] = "cancelled"
                # Remove from active pipelines
                del self.active_pipelines[execution_id]
                return True
        return False

    def shutdown(self) -> None:
        """Shutdown the orchestrator"""
        logger.info("Shutting down orchestrator...")

        # Stop all workers
        for worker in self.workers:
            worker.stop()

        # Shutdown executor
        self.executor.shutdown(wait=True)

        logger.info("Orchestrator shutdown complete")


# Global orchestrator instance (lazy initialization)
_orchestrator: Optional[RuntimeOrchestrator] = None


def get_orchestrator(mode: ExecutionMode = ExecutionMode.LOCAL) -> RuntimeOrchestrator:
    """Get or create the global orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RuntimeOrchestrator(mode)
    return _orchestrator


def reset_orchestrator() -> None:
    """Reset the global orchestrator (mainly for testing)"""
    global _orchestrator
    if _orchestrator:
        _orchestrator.shutdown()
        _orchestrator = None


# Async support for future compatibility
class AsyncOrchestrator:
    """Async version of the orchestrator for high-performance scenarios"""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.tasks: Dict[str, asyncio.Task] = {}

    async def execute_pipeline_async(self, pipeline: PipelineGraph) -> Dict[str, Any]:
        """Execute pipeline asynchronously"""
        # This is a placeholder for future async implementation
        raise NotImplementedError("Async execution coming soon!")


# Export public API
__all__ = [
    'RuntimeOrchestrator',
    'JobQueue',
    'Job',
    'JobStatus',
    'ExecutionMode',
    'Worker',
    'get_orchestrator',
    'reset_orchestrator',
    'AsyncOrchestrator'
]