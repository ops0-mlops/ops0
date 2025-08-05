"""
ops0 Runtime Monitoring

Automatic observability for ML pipelines.
Zero-configuration metrics, logging, and alerting.
"""

import json
import logging
import os
import sys
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock, Thread
from typing import Dict, List, Optional, Any, Tuple, Callable, Deque

# Handle imports for both development and production
try:
    from ops0.core.config import config
    from ops0.core.storage import storage
except ImportError:
    from ...core.config import config
    from ...core.storage import storage

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMING = "timing"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Represents a single metric measurement"""
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str]
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "labels": self.labels,
            "timestamp": self.timestamp
        }


@dataclass
class Alert:
    """Represents an alert condition"""
    id: str
    name: str
    severity: AlertSeverity
    message: str
    metric_name: Optional[str]
    threshold: Optional[float]
    current_value: Optional[float]
    labels: Dict[str, str]
    timestamp: float
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "metric_name": self.metric_name,
            "threshold": self.threshold,
            "current_value": self.current_value,
            "labels": self.labels,
            "timestamp": self.timestamp,
            "resolved": self.resolved
        }


class MetricsCollector:
    """Collects and aggregates metrics"""

    def __init__(self, window_size: int = 300):  # 5 minute window
        self.window_size = window_size
        self.metrics: Dict[str, Deque[Metric]] = defaultdict(lambda: deque(maxlen=1000))
        self.aggregated: Dict[str, Dict[str, float]] = {}
        self._lock = Lock()

    def record(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE,
               labels: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value"""
        metric = Metric(
            name=name,
            type=metric_type,
            value=value,
            labels=labels or {},
            timestamp=time.time()
        )

        with self._lock:
            key = self._metric_key(name, labels)
            self.metrics[key].append(metric)

    def increment(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric"""
        self.record(name, value, MetricType.COUNTER, labels)

    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric"""
        self.record(name, value, MetricType.GAUGE, labels)

    def timing(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a timing metric"""
        self.record(name, duration, MetricType.TIMING, labels)

    def get_metrics(self, name: Optional[str] = None,
                    start_time: Optional[float] = None) -> List[Metric]:
        """Get metrics, optionally filtered"""
        with self._lock:
            result = []

            for key, metrics in self.metrics.items():
                if name and not key.startswith(name):
                    continue

                for metric in metrics:
                    if start_time and metric.timestamp < start_time:
                        continue
                    result.append(metric)

            return sorted(result, key=lambda m: m.timestamp)

    def aggregate(self, name: str, window: Optional[int] = None) -> Dict[str, float]:
        """Get aggregated metrics for a time window"""
        window = window or self.window_size
        cutoff = time.time() - window

        metrics = self.get_metrics(name, cutoff)
        if not metrics:
            return {}

        values = [m.value for m in metrics]

        return {
            "count": len(values),
            "sum": sum(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "p50": self._percentile(values, 50),
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99)
        }

    def _metric_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Generate unique key for metric"""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}:{label_str}"

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]


class AlertManager:
    """Manages alert conditions and notifications"""

    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: Deque[Alert] = deque(maxlen=1000)
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def add_rule(self, name: str, metric_name: str, condition: str,
                 threshold: float, severity: AlertSeverity = AlertSeverity.WARNING) -> None:
        """Add an alert rule"""
        self.alert_rules[name] = {
            "metric_name": metric_name,
            "condition": condition,  # "greater_than", "less_than", "equals"
            "threshold": threshold,
            "severity": severity
        }

    def check_metric(self, metric_name: str, current_value: float,
                     labels: Optional[Dict[str, str]] = None) -> Optional[Alert]:
        """Check if metric triggers any alerts"""
        for rule_name, rule in self.alert_rules.items():
            if rule["metric_name"] != metric_name:
                continue

            triggered = False
            condition = rule["condition"]
            threshold = rule["threshold"]

            if condition == "greater_than" and current_value > threshold:
                triggered = True
            elif condition == "less_than" and current_value < threshold:
                triggered = True
            elif condition == "equals" and current_value == threshold:
                triggered = True

            if triggered:
                alert = Alert(
                    id=f"alert-{int(time.time() * 1000)}",
                    name=rule_name,
                    severity=rule["severity"],
                    message=f"Metric {metric_name} {condition} {threshold} (current: {current_value})",
                    metric_name=metric_name,
                    threshold=threshold,
                    current_value=current_value,
                    labels=labels or {},
                    timestamp=time.time()
                )

                with self._lock:
                    self.alerts[alert.id] = alert
                    self.alert_history.append(alert)

                return alert

        return None

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark alert as resolved"""
        with self._lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolved = True
                return True
        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        with self._lock:
            return [a for a in self.alerts.values() if not a.resolved]

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the last N hours"""
        cutoff = time.time() - (hours * 3600)
        with self._lock:
            return [a for a in self.alert_history if a.timestamp > cutoff]


class PipelineMonitor:
    """
    Main monitoring system for ops0 pipelines.

    Automatically tracks execution metrics, performance, and health.
    """

    def __init__(self):
        self.metrics = MetricsCollector()
        self.alerts = AlertManager()
        self.step_timings: Dict[str, List[float]] = defaultdict(list)
        self.pipeline_runs: Dict[str, Dict[str, Any]] = {}
        self._monitoring_thread: Optional[Thread] = None
        self._running = False

        # Set up default alert rules
        self._setup_default_alerts()

    def _setup_default_alerts(self) -> None:
        """Configure default alert rules"""
        # Step latency alerts
        self.alerts.add_rule(
            "high_step_latency",
            "step.execution_time",
            "greater_than",
            300.0,  # 5 minutes
            AlertSeverity.WARNING
        )

        # Error rate alerts
        self.alerts.add_rule(
            "high_error_rate",
            "pipeline.error_rate",
            "greater_than",
            0.1,  # 10%
            AlertSeverity.ERROR
        )

        # Memory usage alerts
        self.alerts.add_rule(
            "high_memory_usage",
            "system.memory_percent",
            "greater_than",
            90.0,
            AlertSeverity.WARNING
        )

    def start(self) -> None:
        """Start monitoring background thread"""
        if not self._running:
            self._running = True
            self._monitoring_thread = Thread(target=self._monitor_loop, daemon=True)
            self._monitoring_thread.start()
            logger.info("Pipeline monitoring started")

    def stop(self) -> None:
        """Stop monitoring"""
        self._running = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Pipeline monitoring stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop"""
        while self._running:
            try:
                # Collect system metrics
                self._collect_system_metrics()

                # Check alert conditions
                self._check_alerts()

                # Sleep for monitoring interval
                time.sleep(10)  # 10 seconds

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    def _collect_system_metrics(self) -> None:
        """Collect system-level metrics"""
        try:
            import psutil

            # CPU usage
            self.metrics.gauge("system.cpu_percent", psutil.cpu_percent(interval=1))

            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics.gauge("system.memory_percent", memory.percent)
            self.metrics.gauge("system.memory_available", memory.available)

            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics.gauge("system.disk_percent", disk.percent)

        except ImportError:
            # psutil not available, skip system metrics
            pass

    def _check_alerts(self) -> None:
        """Check metrics against alert rules"""
        # Get recent metrics
        recent_metrics = self.metrics.get_metrics(start_time=time.time() - 60)

        # Group by metric name
        metric_values: Dict[str, List[float]] = defaultdict(list)
        for metric in recent_metrics:
            metric_values[metric.name].append(metric.value)

        # Check each metric against rules
        for metric_name, values in metric_values.items():
            if values:
                current_value = values[-1]  # Most recent value
                alert = self.alerts.check_metric(metric_name, current_value)

                if alert:
                    self._handle_alert(alert)

    def _handle_alert(self, alert: Alert) -> None:
        """Handle triggered alert"""
        logger.warning(f"Alert triggered: {alert.message}")

        # In production, this would send notifications
        # For now, just log it
        if alert.severity == AlertSeverity.CRITICAL:
            logger.critical(f"CRITICAL ALERT: {alert.message}")
        elif alert.severity == AlertSeverity.ERROR:
            logger.error(f"ERROR ALERT: {alert.message}")

    # Pipeline execution monitoring

    def start_pipeline_run(self, pipeline_id: str, pipeline_name: str) -> None:
        """Mark pipeline execution start"""
        self.pipeline_runs[pipeline_id] = {
            "name": pipeline_name,
            "start_time": time.time(),
            "steps": {},
            "status": "running"
        }

        self.metrics.increment("pipeline.started", labels={"pipeline": pipeline_name})

    def end_pipeline_run(self, pipeline_id: str, status: str = "completed") -> None:
        """Mark pipeline execution end"""
        if pipeline_id in self.pipeline_runs:
            run = self.pipeline_runs[pipeline_id]
            run["end_time"] = time.time()
            run["duration"] = run["end_time"] - run["start_time"]
            run["status"] = status

            labels = {"pipeline": run["name"], "status": status}
            self.metrics.increment("pipeline.completed", labels=labels)
            self.metrics.timing("pipeline.duration", run["duration"], labels={"pipeline": run["name"]})

            # Calculate error rate
            total_steps = len(run["steps"])
            failed_steps = sum(1 for s in run["steps"].values() if s.get("status") == "failed")
            if total_steps > 0:
                error_rate = failed_steps / total_steps
                self.metrics.gauge("pipeline.error_rate", error_rate, labels={"pipeline": run["name"]})

    def start_step(self, pipeline_id: str, step_name: str) -> None:
        """Mark step execution start"""
        if pipeline_id in self.pipeline_runs:
            self.pipeline_runs[pipeline_id]["steps"][step_name] = {
                "start_time": time.time(),
                "status": "running"
            }

            pipeline_name = self.pipeline_runs[pipeline_id]["name"]
            self.metrics.increment("step.started", labels={"pipeline": pipeline_name, "step": step_name})

    def end_step(self, pipeline_id: str, step_name: str, status: str = "completed",
                 error: Optional[str] = None) -> None:
        """Mark step execution end"""
        if pipeline_id in self.pipeline_runs and step_name in self.pipeline_runs[pipeline_id]["steps"]:
            step = self.pipeline_runs[pipeline_id]["steps"][step_name]
            step["end_time"] = time.time()
            step["duration"] = step["end_time"] - step["start_time"]
            step["status"] = status
            if error:
                step["error"] = error

            pipeline_name = self.pipeline_runs[pipeline_id]["name"]
            labels = {"pipeline": pipeline_name, "step": step_name, "status": status}

            self.metrics.increment("step.completed", labels=labels)
            self.metrics.timing("step.execution_time", step["duration"],
                                labels={"pipeline": pipeline_name, "step": step_name})

            # Store timing for analysis
            self.step_timings[f"{pipeline_name}.{step_name}"].append(step["duration"])

            # Check for latency alerts
            self.alerts.check_metric("step.execution_time", step["duration"],
                                     labels={"pipeline": pipeline_name, "step": step_name})

    # Reporting and analytics

    def get_pipeline_stats(self, pipeline_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get pipeline execution statistics"""
        cutoff = time.time() - (hours * 3600)

        runs = [r for r in self.pipeline_runs.values()
                if r["name"] == pipeline_name and r.get("start_time", 0) > cutoff]

        if not runs:
            return {"message": "No runs found in the specified time period"}

        total_runs = len(runs)
        completed_runs = sum(1 for r in runs if r.get("status") == "completed")
        failed_runs = sum(1 for r in runs if r.get("status") == "failed")

        durations = [r.get("duration", 0) for r in runs if "duration" in r]

        stats = {
            "pipeline": pipeline_name,
            "period_hours": hours,
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "failed_runs": failed_runs,
            "success_rate": completed_runs / total_runs if total_runs > 0 else 0,
            "avg_duration": sum(durations) / len(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "active_alerts": len([a for a in self.alerts.get_active_alerts()
                                  if a.labels.get("pipeline") == pipeline_name])
        }

        # Step-level stats
        step_stats = {}
        for run in runs:
            for step_name, step_data in run.get("steps", {}).items():
                if step_name not in step_stats:
                    step_stats[step_name] = {
                        "executions": 0,
                        "failures": 0,
                        "total_duration": 0
                    }

                step_stats[step_name]["executions"] += 1
                if step_data.get("status") == "failed":
                    step_stats[step_name]["failures"] += 1
                step_stats[step_name]["total_duration"] += step_data.get("duration", 0)

        # Calculate step averages
        for step_name, data in step_stats.items():
            data["avg_duration"] = data["total_duration"] / data["executions"] if data["executions"] > 0 else 0
            data["failure_rate"] = data["failures"] / data["executions"] if data["executions"] > 0 else 0

        stats["steps"] = step_stats

        return stats

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        # Recent metrics
        recent_metrics = self.metrics.get_metrics(start_time=time.time() - 300)  # Last 5 minutes

        # Active pipelines
        active_pipelines = [r for r in self.pipeline_runs.values() if r.get("status") == "running"]

        # Recent alerts
        recent_alerts = self.alerts.get_alert_history(hours=1)

        return {
            "timestamp": time.time(),
            "metrics_count": len(recent_metrics),
            "active_pipelines": len(active_pipelines),
            "active_alerts": len(self.alerts.get_active_alerts()),
            "recent_alerts": [a.to_dict() for a in recent_alerts[-10:]],  # Last 10 alerts
            "system_metrics": {
                "cpu": self.metrics.aggregate("system.cpu_percent", 60),
                "memory": self.metrics.aggregate("system.memory_percent", 60)
            },
            "pipeline_metrics": {
                "started_last_hour": len(self.metrics.get_metrics("pipeline.started", time.time() - 3600)),
                "completed_last_hour": len(self.metrics.get_metrics("pipeline.completed", time.time() - 3600)),
                "avg_duration": self.metrics.aggregate("pipeline.duration", 3600).get("mean", 0)
            }
        }


# Context manager for monitoring steps
class StepMonitor:
    """Context manager for monitoring step execution"""

    def __init__(self, monitor: PipelineMonitor, pipeline_id: str, step_name: str):
        self.monitor = monitor
        self.pipeline_id = pipeline_id
        self.step_name = step_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.monitor.start_step(self.pipeline_id, self.step_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.monitor.end_step(self.pipeline_id, self.step_name, "completed")
        else:
            error_msg = f"{exc_type.__name__}: {exc_val}"
            self.monitor.end_step(self.pipeline_id, self.step_name, "failed", error_msg)

        # Log execution time
        duration = time.time() - self.start_time
        logger.info(f"Step '{self.step_name}' completed in {duration:.2f}s")

        return False  # Don't suppress exceptions


# Global monitor instance
_monitor: Optional[PipelineMonitor] = None


def get_monitor() -> PipelineMonitor:
    """Get or create the global monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = PipelineMonitor()
        if config.monitoring.enable_monitoring:
            _monitor.start()
    return _monitor


def reset_monitor() -> None:
    """Reset the global monitor (mainly for testing)"""
    global _monitor
    if _monitor:
        _monitor.stop()
        _monitor = None


# Decorators for easy monitoring
def monitor_step(func: Callable) -> Callable:
    """Decorator to automatically monitor step execution"""

    def wrapper(*args, **kwargs):
        # Get pipeline context
        pipeline_id = kwargs.get("_pipeline_id", "unknown")
        step_name = func.__name__

        monitor = get_monitor()
        with StepMonitor(monitor, pipeline_id, step_name):
            return func(*args, **kwargs)

    return wrapper


# Export public API
__all__ = [
    'PipelineMonitor',
    'MetricsCollector',
    'AlertManager',
    'StepMonitor',
    'Metric',
    'Alert',
    'MetricType',
    'AlertSeverity',
    'get_monitor',
    'reset_monitor',
    'monitor_step'
]