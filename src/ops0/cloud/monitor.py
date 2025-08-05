"""
ops0 Cloud Monitor

Unified monitoring and observability for ML pipelines.
Provides dashboards, alerts, and insights across cloud providers.
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import statistics

from .base import CloudProvider, CloudResource, ResourceType, ResourceMetrics

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to monitor"""
    SYSTEM = "system"  # CPU, memory, disk
    APPLICATION = "application"  # Request rate, latency, errors
    BUSINESS = "business"  # Model accuracy, predictions, revenue
    COST = "cost"  # Spending, budget alerts


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SILENCED = "silenced"


@dataclass
class MetricDefinition:
    """Defines a metric to monitor"""
    name: str
    type: MetricType
    unit: str
    description: str
    query: Optional[str] = None  # Provider-specific query
    aggregation: str = "avg"  # avg, sum, min, max, count
    interval_seconds: int = 60


@dataclass
class AlertRule:
    """Defines an alert rule"""
    name: str
    metric: str
    condition: str  # e.g., "> 80", "< 10", "== 0"
    threshold: float
    duration_seconds: int = 300  # How long condition must be true
    severity: AlertSeverity = AlertSeverity.WARNING
    notifications: List[str] = field(default_factory=list)  # email, slack, etc.

    def evaluate(self, value: float) -> bool:
        """Evaluate if alert should fire"""
        if self.condition.startswith(">"):
            return value > self.threshold
        elif self.condition.startswith("<"):
            return value < self.threshold
        elif self.condition.startswith(">="):
            return value >= self.threshold
        elif self.condition.startswith("<="):
            return value <= self.threshold
        elif self.condition.startswith("=="):
            return value == self.threshold
        elif self.condition.startswith("!="):
            return value != self.threshold
        else:
            return False


@dataclass
class Alert:
    """Active alert instance"""
    alert_id: str
    rule: AlertRule
    resource_id: str
    metric_value: float
    started_at: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

    @property
    def duration_minutes(self) -> float:
        """How long alert has been active"""
        end_time = self.resolved_at or datetime.now()
        return (end_time - self.started_at).total_seconds() / 60


@dataclass
class Dashboard:
    """Monitoring dashboard configuration"""
    dashboard_id: str
    name: str
    pipeline_name: str
    panels: List[Dict[str, Any]] = field(default_factory=list)
    layout: str = "grid"  # grid, vertical, horizontal
    refresh_interval: int = 30  # seconds
    created_at: datetime = field(default_factory=datetime.now)
    url: Optional[str] = None


class CloudMonitor:
    """Unified monitoring for cloud resources"""

    def __init__(self, provider: CloudProvider):
        self.provider = provider
        self.dashboards: Dict[str, Dashboard] = {}
        self.alert_rules: Dict[str, List[AlertRule]] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.metrics_cache: Dict[str, List[Tuple[datetime, float]]] = {}

        # Default metrics to track
        self.default_metrics = self._create_default_metrics()

        # Notification handlers
        self.notification_handlers = {
            "log": self._notify_log,
            "console": self._notify_console,
            # In production: email, slack, pagerduty, etc.
        }

        logger.info(f"CloudMonitor initialized for provider: {provider.name}")

    def create_dashboard(
            self,
            pipeline_name: str,
            resources: List[CloudResource],
            template: str = "default"
    ) -> str:
        """
        Create monitoring dashboard for a pipeline.

        Args:
            pipeline_name: Name of the pipeline
            resources: Resources to monitor
            template: Dashboard template to use

        Returns:
            Dashboard URL
        """
        dashboard_id = f"{pipeline_name}-{int(datetime.now().timestamp())}"

        # Create dashboard configuration
        dashboard = Dashboard(
            dashboard_id=dashboard_id,
            name=f"{pipeline_name} Pipeline Dashboard",
            pipeline_name=pipeline_name
        )

        # Add panels based on template
        if template == "default":
            dashboard.panels = self._create_default_panels(resources)
        elif template == "ml":
            dashboard.panels = self._create_ml_panels(resources)
        elif template == "cost":
            dashboard.panels = self._create_cost_panels(resources)

        # Create provider-specific dashboard
        dashboard.url = self._create_provider_dashboard(dashboard, resources)

        self.dashboards[pipeline_name] = dashboard

        logger.info(f"Created dashboard for pipeline: {pipeline_name}")

        return dashboard.url or f"http://ops0.local/dashboard/{dashboard_id}"

    def setup_alerts(
            self,
            pipeline_name: str,
            resources: List[CloudResource],
            alert_config: Dict[str, Any]
    ):
        """
        Set up monitoring alerts for a pipeline.

        Args:
            pipeline_name: Name of the pipeline
            resources: Resources to monitor
            alert_config: Alert configuration
        """
        rules = []

        # CPU alerts
        if "cpu_threshold_percent" in alert_config:
            rules.append(AlertRule(
                name=f"{pipeline_name}-high-cpu",
                metric="cpu_percent",
                condition=">",
                threshold=alert_config["cpu_threshold_percent"],
                duration_seconds=300,
                severity=AlertSeverity.WARNING,
                notifications=["log", "console"]
            ))

        # Memory alerts
        if "memory_threshold_percent" in alert_config:
            rules.append(AlertRule(
                name=f"{pipeline_name}-high-memory",
                metric="memory_percent",
                condition=">",
                threshold=alert_config["memory_threshold_percent"],
                duration_seconds=300,
                severity=AlertSeverity.WARNING,
                notifications=["log", "console"]
            ))

        # Error rate alerts
        if "error_rate_threshold" in alert_config:
            rules.append(AlertRule(
                name=f"{pipeline_name}-high-errors",
                metric="error_rate",
                condition=">",
                threshold=alert_config["error_rate_threshold"],
                duration_seconds=60,
                severity=AlertSeverity.ERROR,
                notifications=["log", "console"]
            ))

        # Latency alerts
        if "latency_threshold_ms" in alert_config:
            rules.append(AlertRule(
                name=f"{pipeline_name}-high-latency",
                metric="latency_ms",
                condition=">",
                threshold=alert_config["latency_threshold_ms"],
                duration_seconds=180,
                severity=AlertSeverity.WARNING,
                notifications=["log", "console"]
            ))

        # Store rules
        self.alert_rules[pipeline_name] = rules

        # Set up provider-specific alerts
        self._setup_provider_alerts(pipeline_name, resources, rules)

        logger.info(f"Set up {len(rules)} alerts for pipeline: {pipeline_name}")

    def check_alerts(self, pipeline_name: str):
        """
        Check and evaluate alert rules for a pipeline.

        Args:
            pipeline_name: Pipeline to check
        """
        rules = self.alert_rules.get(pipeline_name, [])
        if not rules:
            return

        # Get resources for pipeline
        resources = self._get_pipeline_resources(pipeline_name)

        for resource in resources:
            # Get current metrics
            metrics = self.provider.get_resource_metrics(resource.resource_id)

            # Evaluate each rule
            for rule in rules:
                metric_value = self._extract_metric_value(metrics, rule.metric)
                if metric_value is None:
                    continue

                # Check if condition is met
                if rule.evaluate(metric_value):
                    self._handle_alert_condition(
                        rule, resource.resource_id, metric_value
                    )
                else:
                    self._clear_alert_if_exists(rule, resource.resource_id)

    def get_metrics(
            self,
            resource_id: str,
            metric_names: List[str],
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None
    ) -> Dict[str, List[Tuple[datetime, float]]]:
        """
        Get historical metrics for a resource.

        Args:
            resource_id: Resource to get metrics for
            metric_names: List of metric names
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Dictionary of metric timeseries
        """
        if not start_time:
            start_time = datetime.now() - timedelta(hours=1)
        if not end_time:
            end_time = datetime.now()

        # In production, this would query provider-specific metrics APIs
        # For now, return cached or simulated data

        metrics_data = {}

        for metric_name in metric_names:
            cache_key = f"{resource_id}:{metric_name}"

            if cache_key in self.metrics_cache:
                # Filter cached data by time range
                timeseries = [
                    (ts, value) for ts, value in self.metrics_cache[cache_key]
                    if start_time <= ts <= end_time
                ]
                metrics_data[metric_name] = timeseries
            else:
                # Generate sample data for demo
                metrics_data[metric_name] = self._generate_sample_metrics(
                    metric_name, start_time, end_time
                )

        return metrics_data

    def get_dashboard_url(self, pipeline_name: str) -> Optional[str]:
        """Get dashboard URL for a pipeline"""
        dashboard = self.dashboards.get(pipeline_name)
        return dashboard.url if dashboard else None

    def delete_dashboard(self, pipeline_name: str):
        """Delete dashboard for a pipeline"""
        if pipeline_name in self.dashboards:
            dashboard = self.dashboards[pipeline_name]

            # Delete provider-specific dashboard
            self._delete_provider_dashboard(dashboard)

            # Remove from tracking
            del self.dashboards[pipeline_name]

            logger.info(f"Deleted dashboard for pipeline: {pipeline_name}")

    def get_active_alerts(
            self,
            pipeline_name: Optional[str] = None,
            severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """
        Get active alerts.

        Args:
            pipeline_name: Filter by pipeline
            severity: Filter by severity

        Returns:
            List of active alerts
        """
        alerts = list(self.active_alerts.values())

        # Filter by pipeline
        if pipeline_name:
            rules = self.alert_rules.get(pipeline_name, [])
            rule_names = {rule.name for rule in rules}
            alerts = [a for a in alerts if a.rule.name in rule_names]

        # Filter by severity
        if severity:
            alerts = [a for a in alerts if a.rule.severity == severity]

        # Filter by status
        alerts = [a for a in alerts if a.status == AlertStatus.ACTIVE]

        return alerts

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()

            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")

    def get_health_status(self, pipeline_name: str) -> Dict[str, Any]:
        """
        Get overall health status for a pipeline.

        Args:
            pipeline_name: Pipeline to check

        Returns:
            Health status summary
        """
        resources = self._get_pipeline_resources(pipeline_name)
        active_alerts = self.get_active_alerts(pipeline_name)

        # Calculate health score (0-100)
        health_score = 100

        # Deduct points for alerts
        for alert in active_alerts:
            if alert.rule.severity == AlertSeverity.CRITICAL:
                health_score -= 25
            elif alert.rule.severity == AlertSeverity.ERROR:
                health_score -= 15
            elif alert.rule.severity == AlertSeverity.WARNING:
                health_score -= 5

        health_score = max(0, health_score)

        # Determine status
        if health_score >= 90:
            status = "healthy"
        elif health_score >= 70:
            status = "degraded"
        elif health_score >= 50:
            status = "unhealthy"
        else:
            status = "critical"

        # Get resource statuses
        resource_statuses = {}
        for resource in resources:
            resource_statuses[resource.resource_name] = {
                "status": self.provider.get_resource_status(resource.resource_id).value,
                "type": resource.resource_type.value
            }

        return {
            "pipeline_name": pipeline_name,
            "status": status,
            "health_score": health_score,
            "active_alerts": len(active_alerts),
            "critical_alerts": len([a for a in active_alerts if a.rule.severity == AlertSeverity.CRITICAL]),
            "resources": resource_statuses,
            "last_check": datetime.now().isoformat()
        }

    # Private methods

    def _create_default_metrics(self) -> List[MetricDefinition]:
        """Create default metric definitions"""
        return [
            MetricDefinition(
                name="cpu_percent",
                type=MetricType.SYSTEM,
                unit="percent",
                description="CPU utilization percentage"
            ),
            MetricDefinition(
                name="memory_percent",
                type=MetricType.SYSTEM,
                unit="percent",
                description="Memory utilization percentage"
            ),
            MetricDefinition(
                name="request_count",
                type=MetricType.APPLICATION,
                unit="count",
                description="Number of requests",
                aggregation="sum"
            ),
            MetricDefinition(
                name="error_count",
                type=MetricType.APPLICATION,
                unit="count",
                description="Number of errors",
                aggregation="sum"
            ),
            MetricDefinition(
                name="latency_ms",
                type=MetricType.APPLICATION,
                unit="milliseconds",
                description="Average response latency"
            ),
            MetricDefinition(
                name="cost_hourly",
                type=MetricType.COST,
                unit="usd",
                description="Hourly cost estimate"
            )
        ]

    def _create_default_panels(self, resources: List[CloudResource]) -> List[Dict[str, Any]]:
        """Create default dashboard panels"""
        panels = []

        # System metrics panel
        panels.append({
            "title": "System Metrics",
            "type": "timeseries",
            "metrics": ["cpu_percent", "memory_percent"],
            "position": {"x": 0, "y": 0, "w": 12, "h": 4}
        })

        # Request metrics panel
        panels.append({
            "title": "Request Volume",
            "type": "timeseries",
            "metrics": ["request_count"],
            "position": {"x": 0, "y": 4, "w": 6, "h": 4}
        })

        # Error rate panel
        panels.append({
            "title": "Error Rate",
            "type": "timeseries",
            "metrics": ["error_count"],
            "position": {"x": 6, "y": 4, "w": 6, "h": 4}
        })

        # Latency panel
        panels.append({
            "title": "Response Latency",
            "type": "timeseries",
            "metrics": ["latency_ms"],
            "position": {"x": 0, "y": 8, "w": 12, "h": 4}
        })

        # Resource status panel
        panels.append({
            "title": "Resource Status",
            "type": "table",
            "data_source": "resource_status",
            "position": {"x": 0, "y": 12, "w": 12, "h": 3}
        })

        return panels

    def _create_ml_panels(self, resources: List[CloudResource]) -> List[Dict[str, Any]]:
        """Create ML-specific dashboard panels"""
        panels = self._create_default_panels(resources)

        # Add ML-specific panels
        panels.extend([
            {
                "title": "Model Performance",
                "type": "gauge",
                "metrics": ["model_accuracy", "model_f1_score"],
                "position": {"x": 0, "y": 15, "w": 6, "h": 4}
            },
            {
                "title": "Prediction Volume",
                "type": "counter",
                "metrics": ["predictions_total"],
                "position": {"x": 6, "y": 15, "w": 6, "h": 4}
            },
            {
                "title": "Data Drift",
                "type": "heatmap",
                "metrics": ["feature_drift_score"],
                "position": {"x": 0, "y": 19, "w": 12, "h": 4}
            }
        ])

        return panels

    def _create_cost_panels(self, resources: List[CloudResource]) -> List[Dict[str, Any]]:
        """Create cost-focused dashboard panels"""
        return [
            {
                "title": "Hourly Cost Trend",
                "type": "timeseries",
                "metrics": ["cost_hourly"],
                "position": {"x": 0, "y": 0, "w": 12, "h": 4}
            },
            {
                "title": "Cost by Resource",
                "type": "piechart",
                "data_source": "cost_by_resource",
                "position": {"x": 0, "y": 4, "w": 6, "h": 4}
            },
            {
                "title": "Cost by Category",
                "type": "piechart",
                "data_source": "cost_by_category",
                "position": {"x": 6, "y": 4, "w": 6, "h": 4}
            },
            {
                "title": "Monthly Projection",
                "type": "stat",
                "data_source": "monthly_cost_projection",
                "position": {"x": 0, "y": 8, "w": 12, "h": 3}
            }
        ]

    def _create_provider_dashboard(
            self,
            dashboard: Dashboard,
            resources: List[CloudResource]
    ) -> Optional[str]:
        """Create provider-specific dashboard"""
        # This would create dashboards in CloudWatch, Stackdriver, etc.
        # For now, return a mock URL

        if self.provider.name == "aws":
            # Would create CloudWatch dashboard
            return f"https://console.aws.amazon.com/cloudwatch/dashboard/{dashboard.dashboard_id}"
        elif self.provider.name == "gcp":
            # Would create Stackdriver dashboard
            return f"https://console.cloud.google.com/monitoring/dashboards/{dashboard.dashboard_id}"
        elif self.provider.name == "azure":
            # Would create Azure Monitor dashboard
            return f"https://portal.azure.com/dashboards/{dashboard.dashboard_id}"
        else:
            return None

    def _delete_provider_dashboard(self, dashboard: Dashboard):
        """Delete provider-specific dashboard"""
        # This would delete dashboards from provider services
        logger.info(f"Deleting provider dashboard: {dashboard.dashboard_id}")

    def _setup_provider_alerts(
            self,
            pipeline_name: str,
            resources: List[CloudResource],
            rules: List[AlertRule]
    ):
        """Set up provider-specific alerts"""
        # This would create alerts in CloudWatch, Stackdriver, etc.
        logger.info(f"Setting up {len(rules)} provider alerts for {pipeline_name}")

    def _get_pipeline_resources(self, pipeline_name: str) -> List[CloudResource]:
        """Get resources for a pipeline"""
        # In production, this would query resource tags or metadata
        # For now, return all resources
        return self.provider.list_resources()

    def _extract_metric_value(self, metrics: ResourceMetrics, metric_name: str) -> Optional[float]:
        """Extract specific metric value from resource metrics"""
        if metric_name == "cpu_percent":
            return metrics.cpu_percent
        elif metric_name == "memory_percent":
            return metrics.memory_percent
        elif metric_name == "request_count":
            return float(metrics.request_count)
        elif metric_name == "error_count":
            return float(metrics.error_count)
        elif metric_name == "error_rate":
            if metrics.request_count > 0:
                return metrics.error_count / metrics.request_count
            return 0.0
        elif metric_name == "latency_ms":
            return metrics.average_latency_ms
        else:
            return None

    def _handle_alert_condition(
            self,
            rule: AlertRule,
            resource_id: str,
            metric_value: float
    ):
        """Handle alert condition being met"""
        alert_key = f"{rule.name}:{resource_id}"

        # Check if alert already exists
        if alert_key in self.active_alerts:
            # Alert already active
            return

        # Create new alert
        alert = Alert(
            alert_id=f"alert-{int(datetime.now().timestamp())}",
            rule=rule,
            resource_id=resource_id,
            metric_value=metric_value,
            started_at=datetime.now()
        )

        self.active_alerts[alert_key] = alert

        # Send notifications
        self._send_notifications(alert)

        logger.warning(
            f"Alert triggered: {rule.name} for {resource_id} "
            f"({rule.metric} = {metric_value})"
        )

    def _clear_alert_if_exists(self, rule: AlertRule, resource_id: str):
        """Clear alert if it exists"""
        alert_key = f"{rule.name}:{resource_id}"

        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()

            # Remove from active alerts
            del self.active_alerts[alert_key]

            logger.info(
                f"Alert resolved: {rule.name} for {resource_id} "
                f"(duration: {alert.duration_minutes:.1f} minutes)"
            )

    def _send_notifications(self, alert: Alert):
        """Send alert notifications"""
        for notification_type in alert.rule.notifications:
            if notification_type in self.notification_handlers:
                handler = self.notification_handlers[notification_type]
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Failed to send {notification_type} notification: {e}")

    def _notify_log(self, alert: Alert):
        """Log notification handler"""
        logger.warning(
            f"ALERT: {alert.rule.name} - {alert.rule.metric} "
            f"{alert.rule.condition} {alert.rule.threshold} "
            f"(current: {alert.metric_value})"
        )

    def _notify_console(self, alert: Alert):
        """Console notification handler"""
        print(
            f"\n🚨 ALERT: {alert.rule.name}\n"
            f"   Metric: {alert.rule.metric} = {alert.metric_value}\n"
            f"   Threshold: {alert.rule.condition} {alert.rule.threshold}\n"
            f"   Resource: {alert.resource_id}\n"
            f"   Severity: {alert.rule.severity.value}\n"
        )

    def _generate_sample_metrics(
            self,
            metric_name: str,
            start_time: datetime,
            end_time: datetime
    ) -> List[Tuple[datetime, float]]:
        """Generate sample metrics for demo"""
        import random

        data_points = []
        current_time = start_time
        interval = timedelta(minutes=1)

        # Base values for different metrics
        base_values = {
            "cpu_percent": 50,
            "memory_percent": 60,
            "request_count": 100,
            "error_count": 2,
            "latency_ms": 150
        }

        base = base_values.get(metric_name, 50)

        while current_time <= end_time:
            # Add some variation
            value = base + random.gauss(0, base * 0.1)
            value = max(0, value)

            if metric_name.endswith("_percent"):
                value = min(100, value)

            data_points.append((current_time, value))
            current_time += interval

        return data_points