"""
ops0 Cost Estimator

Intelligent cost estimation and optimization for ML pipelines.
Provides real-time cost tracking and recommendations.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from .base import CloudProvider, CloudResource, ResourceType

logger = logging.getLogger(__name__)


class CostCategory(Enum):
    """Cost categories for breakdown"""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"
    REQUESTS = "requests"
    OTHER = "other"


@dataclass
class CostEstimate:
    """Detailed cost estimate"""
    total_monthly: float
    total_hourly: float
    breakdown: Dict[CostCategory, float] = field(default_factory=dict)
    by_resource: Dict[str, float] = field(default_factory=dict)
    currency: str = "USD"
    confidence: float = 0.9  # Estimation confidence

    def add_cost(self, category: CostCategory, amount: float, resource_name: Optional[str] = None):
        """Add cost to estimate"""
        self.breakdown[category] = self.breakdown.get(category, 0) + amount

        if resource_name:
            self.by_resource[resource_name] = self.by_resource.get(resource_name, 0) + amount

    def finalize(self):
        """Calculate totals"""
        self.total_monthly = sum(self.breakdown.values())
        self.total_hourly = self.total_monthly / 730  # Average hours per month

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_monthly": round(self.total_monthly, 2),
            "total_hourly": round(self.total_hourly, 4),
            "currency": self.currency,
            "confidence": self.confidence,
            "breakdown": {
                category.value: round(amount, 2)
                for category, amount in self.breakdown.items()
            },
            "by_resource": {
                name: round(amount, 2)
                for name, amount in self.by_resource.items()
            }
        }


@dataclass
class CostOptimization:
    """Cost optimization recommendation"""
    recommendation: str
    potential_savings: float
    effort: str  # low, medium, high
    impact: str  # low, medium, high
    details: Dict[str, Any] = field(default_factory=dict)


class CostEstimator:
    """Estimates and optimizes cloud costs for ML pipelines"""

    def __init__(self, provider: CloudProvider):
        self.provider = provider

        # Provider-specific pricing (simplified)
        self.pricing = self._load_pricing()

        # Historical usage patterns for better estimates
        self.usage_patterns = {}

        logger.info(f"CostEstimator initialized for provider: {provider.name}")

    def estimate(
            self,
            resources: List[CloudResource],
            usage_patterns: Optional[Dict[str, Any]] = None,
            duration_days: int = 30
    ) -> CostEstimate:
        """
        Estimate costs for a set of resources.

        Args:
            resources: List of cloud resources
            usage_patterns: Expected usage patterns
            duration_days: Duration for estimate

        Returns:
            Detailed cost estimate
        """
        estimate = CostEstimate()

        # Default usage patterns
        if not usage_patterns:
            usage_patterns = self._get_default_usage_patterns()

        # Calculate costs for each resource
        for resource in resources:
            resource_cost = self._estimate_resource_cost(
                resource, usage_patterns, duration_days
            )

            # Add to appropriate categories
            if resource.resource_type in [ResourceType.CONTAINER, ResourceType.FUNCTION]:
                category = CostCategory.GPU if resource.metadata.get('gpu', 0) > 0 else CostCategory.COMPUTE
                estimate.add_cost(category, resource_cost, resource.resource_name)

            elif resource.resource_type == ResourceType.STORAGE:
                estimate.add_cost(CostCategory.STORAGE, resource_cost, resource.resource_name)

            elif resource.resource_type in [ResourceType.QUEUE, ResourceType.NETWORK]:
                estimate.add_cost(CostCategory.NETWORK, resource_cost, resource.resource_name)

            else:
                estimate.add_cost(CostCategory.OTHER, resource_cost, resource.resource_name)

        # Add network egress costs
        network_cost = self._estimate_network_cost(resources, usage_patterns, duration_days)
        estimate.add_cost(CostCategory.NETWORK, network_cost)

        # Add request/invocation costs
        request_cost = self._estimate_request_cost(resources, usage_patterns, duration_days)
        estimate.add_cost(CostCategory.REQUESTS, request_cost)

        # Finalize estimate
        estimate.finalize()

        # Adjust confidence based on usage pattern certainty
        if usage_patterns.get('confidence'):
            estimate.confidence *= usage_patterns['confidence']

        return estimate

    def get_optimization_recommendations(
            self,
            resources: List[CloudResource],
            current_usage: Optional[Dict[str, Any]] = None
    ) -> List[CostOptimization]:
        """
        Get cost optimization recommendations.

        Args:
            resources: List of cloud resources
            current_usage: Current usage metrics

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Analyze each resource
        for resource in resources:
            # Check for over-provisioning
            if resource.resource_type == ResourceType.CONTAINER:
                rec = self._check_container_optimization(resource, current_usage)
                if rec:
                    recommendations.extend(rec)

            elif resource.resource_type == ResourceType.STORAGE:
                rec = self._check_storage_optimization(resource, current_usage)
                if rec:
                    recommendations.extend(rec)

        # Check for general optimizations
        recommendations.extend(self._check_general_optimizations(resources))

        # Sort by potential savings
        recommendations.sort(key=lambda x: x.potential_savings, reverse=True)

        return recommendations

    def track_actual_costs(
            self,
            resource_id: str,
            actual_cost: float,
            period_days: int = 1
    ):
        """
        Track actual costs for improving estimates.

        Args:
            resource_id: Resource identifier
            actual_cost: Actual cost incurred
            period_days: Period covered by cost
        """
        # Store for future estimate improvements
        if resource_id not in self.usage_patterns:
            self.usage_patterns[resource_id] = []

        self.usage_patterns[resource_id].append({
            "timestamp": datetime.now(),
            "cost": actual_cost,
            "period_days": period_days,
            "daily_cost": actual_cost / period_days
        })

        # Keep only recent data (last 90 days)
        cutoff = datetime.now() - timedelta(days=90)
        self.usage_patterns[resource_id] = [
            entry for entry in self.usage_patterns[resource_id]
            if entry["timestamp"] > cutoff
        ]

    def compare_providers(
            self,
            resources: List[CloudResource],
            providers: List[str] = ["aws", "gcp", "azure"],
            usage_patterns: Optional[Dict[str, Any]] = None
    ) -> Dict[str, CostEstimate]:
        """
        Compare costs across different cloud providers.

        Args:
            resources: List of resources to estimate
            providers: List of provider names to compare
            usage_patterns: Expected usage patterns

        Returns:
            Cost estimates for each provider
        """
        estimates = {}

        for provider_name in providers:
            # Create temporary estimator for each provider
            provider_estimator = CostEstimator(self._create_mock_provider(provider_name))

            # Adjust resources for provider-specific features
            adjusted_resources = self._adjust_resources_for_provider(
                resources, provider_name
            )

            # Get estimate
            estimate = provider_estimator.estimate(
                adjusted_resources,
                usage_patterns
            )

            estimates[provider_name] = estimate

        return estimates

    # Private methods

    def _load_pricing(self) -> Dict[str, Any]:
        """Load provider-specific pricing"""
        # Simplified pricing data
        pricing = {
            "aws": {
                "compute": {
                    "cpu_hour": 0.04048,  # per vCPU hour
                    "memory_gb_hour": 0.004445,  # per GB hour
                    "gpu_hour": 0.90,  # per GPU hour (p3.2xlarge equivalent)
                    "spot_discount": 0.7  # 70% discount for spot
                },
                "serverless": {
                    "request": 0.0000002,  # per request
                    "gb_second": 0.0000166667  # per GB-second
                },
                "storage": {
                    "gb_month": 0.023,  # S3 standard
                    "request": 0.0004  # per 1000 requests
                },
                "network": {
                    "gb_egress": 0.09  # per GB
                }
            },
            "gcp": {
                "compute": {
                    "cpu_hour": 0.031611,  # per vCPU hour
                    "memory_gb_hour": 0.004237,  # per GB hour
                    "gpu_hour": 0.95,  # per GPU hour
                    "preemptible_discount": 0.8  # 80% discount
                },
                "serverless": {
                    "request": 0.0000004,  # per request
                    "gb_second": 0.0000025  # per GB-second
                },
                "storage": {
                    "gb_month": 0.020,  # GCS standard
                    "request": 0.005  # per 10000 operations
                },
                "network": {
                    "gb_egress": 0.12  # per GB
                }
            },
            "azure": {
                "compute": {
                    "cpu_hour": 0.0396,  # per vCPU hour
                    "memory_gb_hour": 0.0044,  # per GB hour
                    "gpu_hour": 0.90,  # per GPU hour
                    "spot_discount": 0.9  # 90% discount for spot
                },
                "serverless": {
                    "request": 0.0000002,  # per request
                    "gb_second": 0.000016  # per GB-second
                },
                "storage": {
                    "gb_month": 0.0184,  # Blob storage hot tier
                    "request": 0.0004  # per 10000 operations
                },
                "network": {
                    "gb_egress": 0.087  # per GB
                }
            },
            "kubernetes": {
                # K8s costs depend on underlying infrastructure
                "compute": {
                    "cpu_hour": 0.04,  # Estimate
                    "memory_gb_hour": 0.004,  # Estimate
                    "gpu_hour": 0.90,  # Estimate
                },
                "storage": {
                    "gb_month": 0.10,  # EBS/PD estimate
                },
                "network": {
                    "gb_egress": 0.01  # Internal traffic
                }
            }
        }

        return pricing.get(self.provider.name, pricing["aws"])

    def _get_default_usage_patterns(self) -> Dict[str, Any]:
        """Get default usage patterns for ML workloads"""
        return {
            "monthly_requests": 1000000,  # 1M requests
            "average_duration_ms": 100,  # 100ms average
            "data_processed_gb": 100,  # 100GB/month
            "storage_gb": 500,  # 500GB storage
            "network_egress_gb": 50,  # 50GB egress
            "duty_cycle": 0.3,  # 30% utilization
            "peak_hours_per_day": 8,  # 8 peak hours
            "confidence": 0.7  # 70% confidence in estimates
        }

    def _estimate_resource_cost(
            self,
            resource: CloudResource,
            usage_patterns: Dict[str, Any],
            duration_days: int
    ) -> float:
        """Estimate cost for a single resource"""
        hours = duration_days * 24

        if resource.resource_type == ResourceType.CONTAINER:
            # Container/VM costs
            cpu = resource.metadata.get('cpu', 1)
            memory_gb = resource.metadata.get('memory', 2048) / 1024
            gpu = resource.metadata.get('gpu', 0)

            # Apply duty cycle for containers
            effective_hours = hours * usage_patterns.get('duty_cycle', 1.0)

            cpu_cost = cpu * self.pricing['compute']['cpu_hour'] * effective_hours
            memory_cost = memory_gb * self.pricing['compute']['memory_gb_hour'] * effective_hours
            gpu_cost = gpu * self.pricing['compute']['gpu_hour'] * effective_hours if gpu > 0 else 0

            # Apply spot/preemptible discount if applicable
            if resource.metadata.get('spot_instances') or resource.metadata.get('preemptible'):
                discount = self.pricing['compute'].get('spot_discount', 0.7)
                cpu_cost *= discount
                memory_cost *= discount
                gpu_cost *= discount

            return cpu_cost + memory_cost + gpu_cost

        elif resource.resource_type == ResourceType.FUNCTION:
            # Serverless function costs
            requests = usage_patterns.get('monthly_requests', 1000000) * (duration_days / 30)
            duration_ms = usage_patterns.get('average_duration_ms', 100)
            memory_gb = resource.metadata.get('memory_mb', 512) / 1024

            request_cost = requests * self.pricing['serverless']['request']
            compute_cost = requests * (duration_ms / 1000) * memory_gb * self.pricing['serverless']['gb_second']

            return request_cost + compute_cost

        elif resource.resource_type == ResourceType.STORAGE:
            # Storage costs
            storage_gb = usage_patterns.get('storage_gb', 100)
            monthly_cost = storage_gb * self.pricing['storage']['gb_month']

            # Add request costs
            requests = usage_patterns.get('monthly_requests', 100000) * (duration_days / 30)
            request_cost = (requests / 1000) * self.pricing['storage']['request']

            return (monthly_cost * duration_days / 30) + request_cost

        else:
            # Default minimal cost for other resources
            return 10 * (duration_days / 30)  # $10/month baseline

    def _estimate_network_cost(
            self,
            resources: List[CloudResource],
            usage_patterns: Dict[str, Any],
            duration_days: int
    ) -> float:
        """Estimate network egress costs"""
        # Estimate based on number of compute resources and data transfer
        compute_resources = [
            r for r in resources
            if r.resource_type in [ResourceType.CONTAINER, ResourceType.FUNCTION]
        ]

        # Base egress per resource
        egress_per_resource = usage_patterns.get('network_egress_gb', 50) / max(len(compute_resources), 1)
        total_egress = egress_per_resource * len(compute_resources) * (duration_days / 30)

        return total_egress * self.pricing['network']['gb_egress']

    def _estimate_request_cost(
            self,
            resources: List[CloudResource],
            usage_patterns: Dict[str, Any],
            duration_days: int
    ) -> float:
        """Estimate API request costs"""
        # Most providers include basic requests in compute costs
        # This is for additional API calls (monitoring, logging, etc)

        requests_per_resource = 100000  # Estimate 100k API calls per resource per month
        total_requests = requests_per_resource * len(resources) * (duration_days / 30)

        # Rough estimate: $0.01 per 10k requests
        return (total_requests / 10000) * 0.01

    def _check_container_optimization(
            self,
            resource: CloudResource,
            current_usage: Optional[Dict[str, Any]]
    ) -> List[CostOptimization]:
        """Check container optimization opportunities"""
        recommendations = []

        if not current_usage:
            return recommendations

        # Check CPU utilization
        cpu_usage = current_usage.get('cpu_percent', 50)
        if cpu_usage < 20:
            current_cpu = resource.metadata.get('cpu', 1)
            recommended_cpu = max(0.5, current_cpu * 0.5)

            savings = (current_cpu - recommended_cpu) * self.pricing['compute']['cpu_hour'] * 730

            recommendations.append(CostOptimization(
                recommendation=f"Reduce CPU allocation for {resource.resource_name}",
                potential_savings=savings,
                effort="low",
                impact="medium",
                details={
                    "current_cpu": current_cpu,
                    "recommended_cpu": recommended_cpu,
                    "current_usage": f"{cpu_usage}%"
                }
            ))

        # Check memory utilization
        memory_usage = current_usage.get('memory_percent', 50)
        if memory_usage < 30:
            current_memory = resource.metadata.get('memory', 2048)
            recommended_memory = max(512, int(current_memory * 0.6))

            savings = ((current_memory - recommended_memory) / 1024) * self.pricing['compute']['memory_gb_hour'] * 730

            recommendations.append(CostOptimization(
                recommendation=f"Reduce memory allocation for {resource.resource_name}",
                potential_savings=savings,
                effort="low",
                impact="low",
                details={
                    "current_memory_mb": current_memory,
                    "recommended_memory_mb": recommended_memory,
                    "current_usage": f"{memory_usage}%"
                }
            ))

        # Check for spot/preemptible opportunities
        if not resource.metadata.get('spot_instances'):
            current_cost = self._estimate_resource_cost(resource, self._get_default_usage_patterns(), 30)
            spot_savings = current_cost * 0.7  # 70% savings

            recommendations.append(CostOptimization(
                recommendation=f"Use spot/preemptible instances for {resource.resource_name}",
                potential_savings=spot_savings,
                effort="medium",
                impact="high",
                details={
                    "current_type": "on-demand",
                    "recommended_type": "spot/preemptible",
                    "availability_impact": "May experience interruptions"
                }
            ))

        return recommendations

    def _check_storage_optimization(
            self,
            resource: CloudResource,
            current_usage: Optional[Dict[str, Any]]
    ) -> List[CostOptimization]:
        """Check storage optimization opportunities"""
        recommendations = []

        # Check for lifecycle policies
        if 'lifecycle' not in resource.metadata:
            savings = 50 * 0.01  # Estimate $0.01/GB savings for 50GB moved to cold storage

            recommendations.append(CostOptimization(
                recommendation=f"Enable lifecycle policies for {resource.resource_name}",
                potential_savings=savings,
                effort="low",
                impact="medium",
                details={
                    "recommendation": "Move infrequently accessed data to cheaper storage tiers",
                    "suggested_policy": "Archive after 30 days, delete after 365 days"
                }
            ))

        return recommendations

    def _check_general_optimizations(
            self,
            resources: List[CloudResource]
    ) -> List[CostOptimization]:
        """Check general optimization opportunities"""
        recommendations = []

        # Check for resource consolidation opportunities
        containers = [r for r in resources if r.resource_type == ResourceType.CONTAINER]
        if len(containers) > 3:
            # Estimate 20% savings from consolidation
            total_cost = sum(
                self._estimate_resource_cost(r, self._get_default_usage_patterns(), 30)
                for r in containers
            )
            savings = total_cost * 0.2

            recommendations.append(CostOptimization(
                recommendation="Consider consolidating pipeline steps",
                potential_savings=savings,
                effort="high",
                impact="high",
                details={
                    "current_containers": len(containers),
                    "recommendation": "Combine lightweight steps to reduce overhead"
                }
            ))

        # Check for multi-region redundancy
        regions = set(r.region for r in resources)
        if len(regions) > 1:
            # Estimate network costs between regions
            network_savings = 100 * self.pricing['network']['gb_egress'] * 0.5

            recommendations.append(CostOptimization(
                recommendation="Consolidate resources in a single region",
                potential_savings=network_savings,
                effort="medium",
                impact="medium",
                details={
                    "current_regions": list(regions),
                    "recommendation": "Reduce inter-region data transfer costs"
                }
            ))

        return recommendations

    def _create_mock_provider(self, provider_name: str) -> CloudProvider:
        """Create a mock provider for comparison"""

        # In production, this would create actual provider instances
        class MockProvider(CloudProvider):
            def __init__(self, name):
                self._name = name
                super().__init__()

            @property
            def name(self):
                return self._name

            @property
            def regions(self):
                return ["us-east-1"]

            def _setup(self):
                pass

            # Implement required abstract methods with mock implementations
            def deploy_container(self, spec):
                pass

            def deploy_function(self, spec):
                pass

            def create_storage(self, name, region, **kwargs):
                pass

            def create_queue(self, name, region, **kwargs):
                pass

            def create_secret(self, name, value, region):
                pass

            def get_resource_status(self, resource_id):
                pass

            def get_resource_metrics(self, resource_id):
                pass

            def update_resource(self, resource_id, spec):
                pass

            def delete_resource(self, resource_id):
                pass

            def list_resources(self, resource_type=None):
                pass

            def estimate_cost(self, resources, days=30):
                pass

        return MockProvider(provider_name)

    def _adjust_resources_for_provider(
            self,
            resources: List[CloudResource],
            provider_name: str
    ) -> List[CloudResource]:
        """Adjust resources for provider-specific features"""
        # In production, this would handle provider-specific resource mappings
        # For now, return resources as-is
        return resources