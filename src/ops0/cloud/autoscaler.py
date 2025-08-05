"""
ops0 Auto Scaler

Intelligent auto-scaling for ML pipelines.
Adapts to workload patterns and optimizes for cost/performance.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import statistics

from .base import CloudProvider, CloudResource, ResourceType, ResourceMetrics

logger = logging.getLogger(__name__)


class ScalingMetric(Enum):
    """Metrics used for scaling decisions"""
    CPU_PERCENT = "cpu_percent"
    MEMORY_PERCENT = "memory_percent"
    REQUEST_RATE = "request_rate"
    ERROR_RATE = "error_rate"
    LATENCY_MS = "latency_ms"
    QUEUE_DEPTH = "queue_depth"
    CUSTOM = "custom"


class ScalingDirection(Enum):
    """Scaling direction"""
    UP = "up"
    DOWN = "down"
    NONE = "none"


@dataclass
class ScalingPolicy:
    """Defines a scaling policy"""
    metric: ScalingMetric
    target_value: float
    scale_up_threshold: float  # Percentage above target
    scale_down_threshold: float  # Percentage below target
    cooldown_seconds: int = 300

    def evaluate(self, current_value: float) -> ScalingDirection:
        """Evaluate if scaling is needed"""
        if current_value > self.target_value * (1 + self.scale_up_threshold / 100):
            return ScalingDirection.UP
        elif current_value < self.target_value * (1 - self.scale_down_threshold / 100):
            return ScalingDirection.DOWN
        else:
            return ScalingDirection.NONE


@dataclass
class ScalingConfiguration:
    """Complete scaling configuration for a resource"""
    resource_id: str
    min_instances: int = 1
    max_instances: int = 10
    current_instances: int = 1
    policies: List[ScalingPolicy] = field(default_factory=list)
    scale_up_cooldown: int = 60
    scale_down_cooldown: int = 300
    scale_up_increment: int = 1
    scale_down_increment: int = 1
    last_scale_time: Optional[datetime] = None
    last_scale_direction: Optional[ScalingDirection] = None

    def can_scale(self, direction: ScalingDirection) -> bool:
        """Check if scaling is allowed based on cooldown"""
        if not self.last_scale_time:
            return True

        elapsed = (datetime.now() - self.last_scale_time).total_seconds()

        if direction == ScalingDirection.UP:
            return elapsed >= self.scale_up_cooldown
        elif direction == ScalingDirection.DOWN:
            return elapsed >= self.scale_down_cooldown
        else:
            return True

    def calculate_new_instances(self, direction: ScalingDirection) -> int:
        """Calculate new instance count"""
        if direction == ScalingDirection.UP:
            new_count = min(
                self.current_instances + self.scale_up_increment,
                self.max_instances
            )
        elif direction == ScalingDirection.DOWN:
            new_count = max(
                self.current_instances - self.scale_down_increment,
                self.min_instances
            )
        else:
            new_count = self.current_instances

        return new_count


@dataclass
class ScalingEvent:
    """Records a scaling event"""
    timestamp: datetime
    resource_id: str
    direction: ScalingDirection
    old_instances: int
    new_instances: int
    trigger_metric: str
    trigger_value: float
    success: bool = True
    error_message: Optional[str] = None


class AutoScaler:
    """Intelligent auto-scaler for cloud resources"""

    def __init__(self, provider: CloudProvider):
        self.provider = provider
        self.configurations: Dict[str, ScalingConfiguration] = {}
        self.scaling_history: List[ScalingEvent] = []
        self.metrics_history: Dict[str, List[Tuple[datetime, ResourceMetrics]]] = {}

        # Scaling strategies
        self.strategies = {
            "aggressive": self._aggressive_scaling,
            "conservative": self._conservative_scaling,
            "predictive": self._predictive_scaling,
            "cost_optimized": self._cost_optimized_scaling
        }

        self.default_strategy = "conservative"

        logger.info(f"AutoScaler initialized for provider: {provider.name}")

    def configure_scaling(
            self,
            resource: CloudResource,
            min_instances: int = 1,
            max_instances: int = 10,
            target_metrics: Optional[Dict[str, float]] = None,
            strategy: str = "conservative"
    ) -> ScalingConfiguration:
        """
        Configure auto-scaling for a resource.

        Args:
            resource: Cloud resource to scale
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
            target_metrics: Target values for scaling metrics
            strategy: Scaling strategy to use

        Returns:
            Scaling configuration
        """
        # Create default policies based on resource type
        policies = self._create_default_policies(resource, target_metrics)

        # Create configuration
        config = ScalingConfiguration(
            resource_id=resource.resource_id,
            min_instances=min_instances,
            max_instances=max_instances,
            policies=policies
        )

        # Adjust configuration based on strategy
        if strategy == "aggressive":
            config.scale_up_cooldown = 30
            config.scale_down_cooldown = 600
            config.scale_up_increment = 2
        elif strategy == "cost_optimized":
            config.scale_up_cooldown = 120
            config.scale_down_cooldown = 180

        self.configurations[resource.resource_id] = config

        # Set up provider-specific auto-scaling
        self._configure_provider_autoscaling(resource, config)

        logger.info(f"Configured auto-scaling for resource: {resource.resource_id}")

        return config

    def evaluate_scaling(self, resource_id: str) -> Optional[ScalingEvent]:
        """
        Evaluate if scaling is needed for a resource.

        Args:
            resource_id: Resource to evaluate

        Returns:
            Scaling event if scaling occurred
        """
        config = self.configurations.get(resource_id)
        if not config:
            return None

        # Get current metrics
        metrics = self.provider.get_resource_metrics(resource_id)

        # Store metrics history
        if resource_id not in self.metrics_history:
            self.metrics_history[resource_id] = []

        self.metrics_history[resource_id].append((datetime.now(), metrics))

        # Keep only recent history (last hour)
        cutoff = datetime.now() - timedelta(hours=1)
        self.metrics_history[resource_id] = [
            (ts, m) for ts, m in self.metrics_history[resource_id]
            if ts > cutoff
        ]

        # Evaluate scaling policies
        scaling_decisions = []

        for policy in config.policies:
            current_value = self._get_metric_value(metrics, policy.metric)
            if current_value is not None:
                direction = policy.evaluate(current_value)
                if direction != ScalingDirection.NONE:
                    scaling_decisions.append((direction, policy, current_value))

        # Determine final scaling direction
        if not scaling_decisions:
            return None

        # Use strategy to determine final decision
        strategy_func = self.strategies.get(self.default_strategy, self._conservative_scaling)
        final_direction = strategy_func(scaling_decisions, config)

        if final_direction == ScalingDirection.NONE:
            return None

        # Check cooldown
        if not config.can_scale(final_direction):
            logger.debug(f"Scaling cooldown active for {resource_id}")
            return None

        # Calculate new instance count
        new_instances = config.calculate_new_instances(final_direction)

        if new_instances == config.current_instances:
            return None

        # Execute scaling
        event = self._execute_scaling(
            resource_id,
            config,
            new_instances,
            final_direction,
            scaling_decisions[0]  # Primary trigger
        )

        return event

    def get_scaling_recommendations(
            self,
            resource_id: str,
            lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get scaling recommendations based on historical data.

        Args:
            resource_id: Resource to analyze
            lookback_hours: Hours of history to analyze

        Returns:
            Scaling recommendations
        """
        config = self.configurations.get(resource_id)
        if not config:
            return {"error": "No scaling configuration found"}

        # Analyze historical metrics
        metrics_history = self.metrics_history.get(resource_id, [])
        if not metrics_history:
            return {"error": "No metrics history available"}

        # Calculate statistics
        cpu_values = [m.cpu_percent for _, m in metrics_history if m.cpu_percent > 0]
        memory_values = [m.memory_percent for _, m in metrics_history if m.memory_percent > 0]
        request_counts = [m.request_count for _, m in metrics_history]

        recommendations = {
            "current_config": {
                "min_instances": config.min_instances,
                "max_instances": config.max_instances,
                "current_instances": config.current_instances
            },
            "metrics_summary": {
                "cpu": {
                    "avg": statistics.mean(cpu_values) if cpu_values else 0,
                    "max": max(cpu_values) if cpu_values else 0,
                    "p95": self._percentile(cpu_values, 95) if cpu_values else 0
                },
                "memory": {
                    "avg": statistics.mean(memory_values) if memory_values else 0,
                    "max": max(memory_values) if memory_values else 0,
                    "p95": self._percentile(memory_values, 95) if memory_values else 0
                },
                "requests": {
                    "total": sum(request_counts),
                    "avg_per_minute": sum(request_counts) / (lookback_hours * 60) if request_counts else 0
                }
            },
            "recommendations": []
        }

        # Generate recommendations
        if cpu_values and statistics.mean(cpu_values) < 20:
            recommendations["recommendations"].append({
                "type": "reduce_max_instances",
                "reason": "Low average CPU utilization",
                "suggested_max": max(config.min_instances, config.max_instances // 2)
            })

        if cpu_values and max(cpu_values) > 90:
            recommendations["recommendations"].append({
                "type": "increase_max_instances",
                "reason": "High peak CPU utilization",
                "suggested_max": min(config.max_instances * 2, 20)
            })

        # Analyze scaling events
        recent_events = [
            e for e in self.scaling_history
            if e.resource_id == resource_id and
               e.timestamp > datetime.now() - timedelta(hours=lookback_hours)
        ]

        if len(recent_events) > 10:
            recommendations["recommendations"].append({
                "type": "adjust_thresholds",
                "reason": f"Frequent scaling events ({len(recent_events)} in {lookback_hours} hours)",
                "suggestion": "Consider widening scaling thresholds to reduce flapping"
            })

        recommendations["scaling_events"] = len(recent_events)

        return recommendations

    def predict_scaling_needs(
            self,
            resource_id: str,
            forecast_hours: int = 1
    ) -> Dict[str, Any]:
        """
        Predict future scaling needs using ML.

        Args:
            resource_id: Resource to predict for
            forecast_hours: Hours to forecast ahead

        Returns:
            Scaling predictions
        """
        # Simple time-series prediction
        # In production, this would use proper ML models

        metrics_history = self.metrics_history.get(resource_id, [])
        if len(metrics_history) < 10:
            return {"error": "Insufficient data for prediction"}

        # Extract recent trends
        recent_cpu = [m.cpu_percent for _, m in metrics_history[-10:]]
        recent_requests = [m.request_count for _, m in metrics_history[-10:]]

        # Simple linear extrapolation
        cpu_trend = (recent_cpu[-1] - recent_cpu[0]) / len(recent_cpu)
        request_trend = (recent_requests[-1] - recent_requests[0]) / len(recent_requests)

        # Predict future values
        predicted_cpu = recent_cpu[-1] + (cpu_trend * forecast_hours * 6)  # 6 samples per hour
        predicted_requests = recent_requests[-1] + (request_trend * forecast_hours * 6)

        # Determine if scaling will be needed
        config = self.configurations[resource_id]

        predictions = {
            "forecast_hours": forecast_hours,
            "predicted_metrics": {
                "cpu_percent": max(0, min(100, predicted_cpu)),
                "request_rate": max(0, predicted_requests)
            },
            "scaling_prediction": "none",
            "confidence": 0.7  # Simple model, moderate confidence
        }

        # Check against thresholds
        for policy in config.policies:
            if policy.metric == ScalingMetric.CPU_PERCENT:
                if predicted_cpu > policy.target_value * (1 + policy.scale_up_threshold / 100):
                    predictions["scaling_prediction"] = "scale_up"
                    predictions["trigger_time"] = f"~{forecast_hours * 0.7:.1f} hours"
                elif predicted_cpu < policy.target_value * (1 - policy.scale_down_threshold / 100):
                    predictions["scaling_prediction"] = "scale_down"
                    predictions["trigger_time"] = f"~{forecast_hours * 0.7:.1f} hours"

        return predictions

    def get_scaling_history(
            self,
            resource_id: Optional[str] = None,
            hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Get scaling event history.

        Args:
            resource_id: Filter by resource (all if None)
            hours: Hours of history to retrieve

        Returns:
            List of scaling events
        """
        cutoff = datetime.now() - timedelta(hours=hours)

        events = self.scaling_history
        if resource_id:
            events = [e for e in events if e.resource_id == resource_id]

        events = [e for e in events if e.timestamp > cutoff]

        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "resource_id": e.resource_id,
                "direction": e.direction.value,
                "old_instances": e.old_instances,
                "new_instances": e.new_instances,
                "trigger_metric": e.trigger_metric,
                "trigger_value": e.trigger_value,
                "success": e.success,
                "error": e.error_message
            }
            for e in events
        ]

    # Private methods

    def _create_default_policies(
            self,
            resource: CloudResource,
            target_metrics: Optional[Dict[str, float]]
    ) -> List[ScalingPolicy]:
        """Create default scaling policies"""
        policies = []

        # Default targets
        defaults = {
            "cpu_percent": 70.0,
            "memory_percent": 80.0,
            "error_rate": 0.01,  # 1% error rate
            "latency_ms": 1000.0  # 1 second
        }

        if target_metrics:
            defaults.update(target_metrics)

        # CPU-based scaling
        policies.append(ScalingPolicy(
            metric=ScalingMetric.CPU_PERCENT,
            target_value=defaults["cpu_percent"],
            scale_up_threshold=20,  # Scale up at 84% CPU
            scale_down_threshold=40  # Scale down at 42% CPU
        ))

        # Memory-based scaling
        policies.append(ScalingPolicy(
            metric=ScalingMetric.MEMORY_PERCENT,
            target_value=defaults["memory_percent"],
            scale_up_threshold=10,  # Scale up at 88% memory
            scale_down_threshold=50  # Scale down at 40% memory
        ))

        # For serverless functions, add request-based scaling
        if resource.resource_type == ResourceType.FUNCTION:
            policies.append(ScalingPolicy(
                metric=ScalingMetric.REQUEST_RATE,
                target_value=1000.0,  # 1000 requests per minute
                scale_up_threshold=20,
                scale_down_threshold=50
            ))

        return policies

    def _get_metric_value(self, metrics: ResourceMetrics, metric_type: ScalingMetric) -> Optional[float]:
        """Extract metric value from resource metrics"""
        mapping = {
            ScalingMetric.CPU_PERCENT: metrics.cpu_percent,
            ScalingMetric.MEMORY_PERCENT: metrics.memory_percent,
            ScalingMetric.REQUEST_RATE: metrics.request_count,  # Simplified
            ScalingMetric.ERROR_RATE: metrics.error_count / max(metrics.request_count,
                                                                1) if metrics.request_count else 0,
            ScalingMetric.LATENCY_MS: metrics.average_latency_ms
        }

        return mapping.get(metric_type)

    def _aggressive_scaling(
            self,
            decisions: List[Tuple[ScalingDirection, ScalingPolicy, float]],
            config: ScalingConfiguration
    ) -> ScalingDirection:
        """Aggressive scaling - scale up quickly, down slowly"""
        # If any metric suggests scaling up, do it
        for direction, _, _ in decisions:
            if direction == ScalingDirection.UP:
                return ScalingDirection.UP

        # Only scale down if all metrics agree
        if all(d[0] == ScalingDirection.DOWN for d in decisions):
            return ScalingDirection.DOWN

        return ScalingDirection.NONE

    def _conservative_scaling(
            self,
            decisions: List[Tuple[ScalingDirection, ScalingPolicy, float]],
            config: ScalingConfiguration
    ) -> ScalingDirection:
        """Conservative scaling - require multiple signals"""
        up_votes = sum(1 for d in decisions if d[0] == ScalingDirection.UP)
        down_votes = sum(1 for d in decisions if d[0] == ScalingDirection.DOWN)

        # Require majority for scaling
        if up_votes > len(decisions) / 2:
            return ScalingDirection.UP
        elif down_votes > len(decisions) / 2:
            return ScalingDirection.DOWN

        return ScalingDirection.NONE

    def _predictive_scaling(
            self,
            decisions: List[Tuple[ScalingDirection, ScalingPolicy, float]],
            config: ScalingConfiguration
    ) -> ScalingDirection:
        """Predictive scaling - use historical patterns"""
        # Check if we're in a known high-load period
        current_hour = datetime.now().hour

        # Business hours (9 AM - 6 PM)
        if 9 <= current_hour <= 18:
            # Lower threshold for scaling up during business hours
            for direction, _, _ in decisions:
                if direction == ScalingDirection.UP:
                    return ScalingDirection.UP

        # Use conservative approach otherwise
        return self._conservative_scaling(decisions, config)

    def _cost_optimized_scaling(
            self,
            decisions: List[Tuple[ScalingDirection, ScalingPolicy, float]],
            config: ScalingConfiguration
    ) -> ScalingDirection:
        """Cost-optimized scaling - balance performance and cost"""
        # Only scale up if critical metrics are affected
        critical_metrics = [ScalingMetric.ERROR_RATE, ScalingMetric.LATENCY_MS]

        for direction, policy, value in decisions:
            if direction == ScalingDirection.UP and policy.metric in critical_metrics:
                return ScalingDirection.UP

        # Aggressive scale down for cost savings
        for direction, _, _ in decisions:
            if direction == ScalingDirection.DOWN:
                return ScalingDirection.DOWN

        return ScalingDirection.NONE

    def _execute_scaling(
            self,
            resource_id: str,
            config: ScalingConfiguration,
            new_instances: int,
            direction: ScalingDirection,
            trigger: Tuple[ScalingDirection, ScalingPolicy, float]
    ) -> ScalingEvent:
        """Execute the scaling action"""
        old_instances = config.current_instances

        event = ScalingEvent(
            timestamp=datetime.now(),
            resource_id=resource_id,
            direction=direction,
            old_instances=old_instances,
            new_instances=new_instances,
            trigger_metric=trigger[1].metric.value,
            trigger_value=trigger[2]
        )

        try:
            # Execute provider-specific scaling
            success = self._scale_resource(resource_id, new_instances)

            if success:
                config.current_instances = new_instances
                config.last_scale_time = datetime.now()
                config.last_scale_direction = direction
                event.success = True

                logger.info(
                    f"Scaled {resource_id} from {old_instances} to {new_instances} instances"
                )
            else:
                event.success = False
                event.error_message = "Provider scaling failed"

        except Exception as e:
            event.success = False
            event.error_message = str(e)
            logger.error(f"Failed to scale {resource_id}: {e}")

        # Record event
        self.scaling_history.append(event)

        # Keep only recent history (last 7 days)
        cutoff = datetime.now() - timedelta(days=7)
        self.scaling_history = [
            e for e in self.scaling_history
            if e.timestamp > cutoff
        ]

        return event

    def _scale_resource(self, resource_id: str, new_instances: int) -> bool:
        """Provider-specific scaling implementation"""
        # This would call provider-specific APIs
        # For now, simulate success
        time.sleep(0.5)  # Simulate API call
        return True

    def _configure_provider_autoscaling(
            self,
            resource: CloudResource,
            config: ScalingConfiguration
    ):
        """Configure provider-specific auto-scaling"""
        # This would set up cloud provider auto-scaling groups,
        # scaling policies, etc.
        logger.info(f"Configuring {self.provider.name} auto-scaling for {resource.resource_id}")

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)

        return sorted_values[min(index, len(sorted_values) - 1)]