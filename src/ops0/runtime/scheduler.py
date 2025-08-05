"""
ops0 Runtime Scheduler

Advanced scheduling algorithms for optimal pipeline execution.
Handles resource allocation, priority scheduling, and auto-scaling.
"""

import heapq
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any

logger = logging.getLogger(__name__)


class SchedulingStrategy(Enum):
    """Pipeline scheduling strategies"""
    FIFO = "fifo"  # First In, First Out
    PRIORITY = "priority"  # Priority-based scheduling
    FAIR = "fair"  # Fair scheduling across pipelines
    RESOURCE_AWARE = "resource_aware"  # Consider resource availability
    COST_OPTIMIZED = "cost_optimized"  # Minimize execution cost


class ResourceType(Enum):
    """Types of compute resources"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"


@dataclass
class ResourceRequirement:
    """Resource requirements for a step"""
    cpu: float = 1.0  # CPU cores
    memory: float = 2.0  # GB
    gpu: float = 0.0  # GPU count
    disk: float = 10.0  # GB
    network: float = 100.0  # Mbps

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            "cpu": self.cpu,
            "memory": self.memory,
            "gpu": self.gpu,
            "disk": self.disk,
            "network": self.network
        }


@dataclass
class ResourcePool:
    """Available resources in the cluster"""
    total_cpu: float
    total_memory: float
    total_gpu: float
    available_cpu: float = field(init=False)
    available_memory: float = field(init=False)
    available_gpu: float = field(init=False)
    allocated: Dict[str, ResourceRequirement] = field(default_factory=dict)

    def __post_init__(self):
        self.available_cpu = self.total_cpu
        self.available_memory = self.total_memory
        self.available_gpu = self.total_gpu

    def can_allocate(self, requirements: ResourceRequirement) -> bool:
        """Check if resources can be allocated"""
        return (
                self.available_cpu >= requirements.cpu and
                self.available_memory >= requirements.memory and
                self.available_gpu >= requirements.gpu
        )

    def allocate(self, job_id: str, requirements: ResourceRequirement) -> bool:
        """Allocate resources for a job"""
        if not self.can_allocate(requirements):
            return False

        self.available_cpu -= requirements.cpu
        self.available_memory -= requirements.memory
        self.available_gpu -= requirements.gpu
        self.allocated[job_id] = requirements

        logger.debug(f"Allocated resources for {job_id}: {requirements.to_dict()}")
        return True

    def release(self, job_id: str) -> None:
        """Release allocated resources"""
        if job_id in self.allocated:
            req = self.allocated[job_id]
            self.available_cpu += req.cpu
            self.available_memory += req.memory
            self.available_gpu += req.gpu
            del self.allocated[job_id]

            logger.debug(f"Released resources for {job_id}")

    def get_utilization(self) -> Dict[str, float]:
        """Get resource utilization percentages"""
        return {
            "cpu": (self.total_cpu - self.available_cpu) / self.total_cpu if self.total_cpu > 0 else 0,
            "memory": (self.total_memory - self.available_memory) / self.total_memory if self.total_memory > 0 else 0,
            "gpu": (self.total_gpu - self.available_gpu) / self.total_gpu if self.total_gpu > 0 else 0
        }


@dataclass(order=True)
class ScheduledJob:
    """Job with scheduling metadata"""
    priority: float = field(compare=True)
    job_id: str = field(compare=False)
    pipeline_id: str = field(compare=False)
    step_name: str = field(compare=False)
    requirements: ResourceRequirement = field(compare=False)
    submitted_at: float = field(compare=False)
    estimated_duration: float = field(compare=False, default=60.0)
    dependencies: Set[str] = field(compare=False, default_factory=set)

    def __post_init__(self):
        # Ensure priority is negative for min-heap (higher priority = lower value)
        if self.priority > 0:
            self.priority = -self.priority


class PipelineScheduler:
    """
    Advanced scheduler for pipeline execution.

    Handles job prioritization, resource allocation, and execution optimization.
    """

    def __init__(self, strategy: SchedulingStrategy = SchedulingStrategy.RESOURCE_AWARE):
        self.strategy = strategy
        self.job_queue: List[ScheduledJob] = []
        self.running_jobs: Dict[str, ScheduledJob] = {}
        self.completed_jobs: Set[str] = set()
        self.pipeline_priorities: Dict[str, float] = defaultdict(lambda: 1.0)
        self.job_history: Dict[str, List[float]] = defaultdict(list)  # Execution times

        # Initialize resource pool (would be configured based on cluster)
        self.resource_pool = ResourcePool(
            total_cpu=16.0,
            total_memory=64.0,
            total_gpu=2.0
        )

    def submit_job(self, job_id: str, pipeline_id: str, step_name: str,
                   requirements: Optional[ResourceRequirement] = None,
                   priority: float = 1.0,
                   dependencies: Optional[Set[str]] = None) -> None:
        """Submit a job for scheduling"""
        if requirements is None:
            requirements = self._estimate_requirements(step_name)

        estimated_duration = self._estimate_duration(step_name)

        # Apply scheduling strategy
        adjusted_priority = self._calculate_priority(
            base_priority=priority,
            pipeline_id=pipeline_id,
            step_name=step_name,
            requirements=requirements
        )

        job = ScheduledJob(
            priority=adjusted_priority,
            job_id=job_id,
            pipeline_id=pipeline_id,
            step_name=step_name,
            requirements=requirements,
            submitted_at=time.time(),
            estimated_duration=estimated_duration,
            dependencies=dependencies or set()
        )

        heapq.heappush(self.job_queue, job)
        logger.info(f"Job {job_id} submitted with priority {adjusted_priority}")

    def schedule_next(self) -> Optional[ScheduledJob]:
        """Get the next job to execute"""
        while self.job_queue:
            # Peek at the highest priority job
            job = self.job_queue[0]

            # Check dependencies
            if not self._dependencies_satisfied(job):
                # Dependencies not met, try next job
                heapq.heappop(self.job_queue)
                # Re-add with lower priority
                job.priority -= 0.1
                heapq.heappush(self.job_queue, job)
                continue

            # Check resource availability
            if self.resource_pool.can_allocate(job.requirements):
                # Remove from queue and allocate resources
                heapq.heappop(self.job_queue)

                if self.resource_pool.allocate(job.job_id, job.requirements):
                    self.running_jobs[job.job_id] = job
                    logger.info(f"Scheduled job {job.job_id}")
                    return job
                else:
                    # Allocation failed, re-queue
                    heapq.heappush(self.job_queue, job)

            else:
                # Resources not available
                logger.debug(f"Insufficient resources for job {job.job_id}")
                break

        return None

    def complete_job(self, job_id: str, execution_time: float) -> None:
        """Mark a job as completed"""
        if job_id in self.running_jobs:
            job = self.running_jobs[job_id]

            # Release resources
            self.resource_pool.release(job_id)

            # Update history
            self.job_history[job.step_name].append(execution_time)
            self.completed_jobs.add(job_id)

            # Remove from running
            del self.running_jobs[job_id]

            logger.info(f"Job {job_id} completed in {execution_time:.2f}s")

            # Try to schedule more jobs
            self._attempt_scheduling()

    def fail_job(self, job_id: str, error: str) -> None:
        """Mark a job as failed"""
        if job_id in self.running_jobs:
            # Release resources
            self.resource_pool.release(job_id)
            del self.running_jobs[job_id]

            logger.error(f"Job {job_id} failed: {error}")

    def _dependencies_satisfied(self, job: ScheduledJob) -> bool:
        """Check if all job dependencies are satisfied"""
        return all(dep in self.completed_jobs for dep in job.dependencies)

    def _calculate_priority(self, base_priority: float, pipeline_id: str,
                            step_name: str, requirements: ResourceRequirement) -> float:
        """Calculate adjusted priority based on scheduling strategy"""
        priority = base_priority

        if self.strategy == SchedulingStrategy.FIFO:
            # Use submission time as priority (negative for min-heap)
            priority = -time.time()

        elif self.strategy == SchedulingStrategy.PRIORITY:
            # Use base priority with pipeline adjustment
            priority = base_priority * self.pipeline_priorities[pipeline_id]

        elif self.strategy == SchedulingStrategy.FAIR:
            # Balance across pipelines
            running_pipeline_jobs = sum(
                1 for j in self.running_jobs.values()
                if j.pipeline_id == pipeline_id
            )
            # Lower priority if pipeline has many running jobs
            priority = base_priority / (1 + running_pipeline_jobs)

        elif self.strategy == SchedulingStrategy.RESOURCE_AWARE:
            # Prioritize jobs that fit current resources well
            utilization = self.resource_pool.get_utilization()

            # Favor jobs that use underutilized resources
            resource_score = 0
            if requirements.cpu > 0 and utilization["cpu"] < 0.5:
                resource_score += 1
            if requirements.memory > 0 and utilization["memory"] < 0.5:
                resource_score += 1
            if requirements.gpu > 0 and utilization["gpu"] < 0.5:
                resource_score += 2  # GPU is more valuable

            priority = base_priority + resource_score

        elif self.strategy == SchedulingStrategy.COST_OPTIMIZED:
            # Minimize resource waste
            # Prefer jobs that efficiently use resources
            resource_efficiency = (
                    requirements.cpu / self.resource_pool.total_cpu +
                    requirements.memory / self.resource_pool.total_memory +
                    requirements.gpu / self.resource_pool.total_gpu * 2  # GPU is expensive
            )
            priority = base_priority / (1 + resource_efficiency)

        return priority

    def _estimate_requirements(self, step_name: str) -> ResourceRequirement:
        """Estimate resource requirements based on historical data"""
        # In production, this would use ML models and historical analysis

        # Default requirements
        req = ResourceRequirement()

        # Adjust based on step name patterns
        if "train" in step_name.lower():
            req.cpu = 4.0
            req.memory = 16.0
            req.gpu = 1.0
        elif "preprocess" in step_name.lower():
            req.cpu = 2.0
            req.memory = 8.0
        elif "predict" in step_name.lower() or "inference" in step_name.lower():
            req.cpu = 2.0
            req.memory = 4.0
            req.gpu = 0.5

        return req

    def _estimate_duration(self, step_name: str) -> float:
        """Estimate execution duration based on historical data"""
        if step_name in self.job_history and self.job_history[step_name]:
            # Use average of last 10 runs
            recent_runs = self.job_history[step_name][-10:]
            return sum(recent_runs) / len(recent_runs)

        # Default estimates
        if "train" in step_name.lower():
            return 300.0  # 5 minutes
        elif "preprocess" in step_name.lower():
            return 60.0  # 1 minute
        else:
            return 30.0  # 30 seconds

    def _attempt_scheduling(self) -> None:
        """Try to schedule more jobs after resources are freed"""
        scheduled_count = 0
        max_attempts = len(self.job_queue)

        for _ in range(max_attempts):
            job = self.schedule_next()
            if job:
                scheduled_count += 1
            else:
                break

        if scheduled_count > 0:
            logger.info(f"Scheduled {scheduled_count} additional jobs")

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        return {
            "queued_jobs": len(self.job_queue),
            "running_jobs": len(self.running_jobs),
            "completed_jobs": len(self.completed_jobs),
            "resource_utilization": self.resource_pool.get_utilization(),
            "strategy": self.strategy.value,
            "top_queued_jobs": [
                {
                    "job_id": job.job_id,
                    "priority": job.priority,
                    "wait_time": time.time() - job.submitted_at
                }
                for job in sorted(self.job_queue)[:5]
            ]
        }

    def update_pipeline_priority(self, pipeline_id: str, priority: float) -> None:
        """Update priority for all jobs from a pipeline"""
        self.pipeline_priorities[pipeline_id] = priority

        # Re-prioritize queued jobs
        updated_jobs = []
        while self.job_queue:
            job = heapq.heappop(self.job_queue)
            if job.pipeline_id == pipeline_id:
                # Recalculate priority
                job.priority = self._calculate_priority(
                    base_priority=priority,
                    pipeline_id=pipeline_id,
                    step_name=job.step_name,
                    requirements=job.requirements
                )
            updated_jobs.append(job)

        # Re-add all jobs
        for job in updated_jobs:
            heapq.heappush(self.job_queue, job)

        logger.info(f"Updated priority for pipeline {pipeline_id} to {priority}")


class AutoScaler:
    """
    Auto-scaling manager for dynamic resource allocation.

    Monitors resource usage and pipeline load to scale workers.
    """

    def __init__(self, min_workers: int = 1, max_workers: int = 10):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        self.scaling_history: List[Tuple[float, int]] = []
        self.metrics_window: List[Dict[str, float]] = []

    def update_metrics(self, queue_size: int, resource_utilization: Dict[str, float],
                       avg_wait_time: float) -> Optional[int]:
        """Update metrics and determine if scaling is needed"""
        metrics = {
            "timestamp": time.time(),
            "queue_size": queue_size,
            "cpu_util": resource_utilization.get("cpu", 0),
            "memory_util": resource_utilization.get("memory", 0),
            "gpu_util": resource_utilization.get("gpu", 0),
            "avg_wait_time": avg_wait_time
        }

        self.metrics_window.append(metrics)

        # Keep only recent metrics (last 5 minutes)
        cutoff = time.time() - 300
        self.metrics_window = [m for m in self.metrics_window if m["timestamp"] > cutoff]

        # Determine scaling action
        scale_action = self._calculate_scaling()

        if scale_action != 0:
            new_workers = max(self.min_workers, min(self.max_workers, self.current_workers + scale_action))
            if new_workers != self.current_workers:
                old_workers = self.current_workers
                self.current_workers = new_workers
                self.scaling_history.append((time.time(), scale_action))

                logger.info(f"Auto-scaling: {old_workers} -> {new_workers} workers")
                return new_workers

        return None

    def _calculate_scaling(self) -> int:
        """Calculate scaling decision based on metrics"""
        if len(self.metrics_window) < 3:
            return 0  # Not enough data

        # Calculate averages
        avg_queue = sum(m["queue_size"] for m in self.metrics_window) / len(self.metrics_window)
        avg_cpu = sum(m["cpu_util"] for m in self.metrics_window) / len(self.metrics_window)
        avg_wait = sum(m["avg_wait_time"] for m in self.metrics_window) / len(self.metrics_window)

        # Scale up conditions
        if avg_queue > self.current_workers * 5 and avg_cpu > 0.8:
            return 2  # Scale up by 2
        elif avg_queue > self.current_workers * 3 or avg_wait > 60:
            return 1  # Scale up by 1

        # Scale down conditions
        elif avg_queue < self.current_workers and avg_cpu < 0.3:
            return -1  # Scale down by 1

        return 0  # No scaling needed


# Export public API
__all__ = [
    'PipelineScheduler',
    'AutoScaler',
    'SchedulingStrategy',
    'ResourceRequirement',
    'ResourcePool',
    'ScheduledJob',
    'ResourceType'
]