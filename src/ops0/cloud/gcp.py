"""
ops0 GCP Provider

Zero-configuration Google Cloud deployment for ML pipelines.
Automatically handles Cloud Run, Cloud Functions, GCS, Pub/Sub, and more.
"""

import os
import json
import time
import logging
import base64
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from .base import (
    CloudProvider, CloudResource, DeploymentSpec,
    ResourceType, ResourceStatus, ResourceMetrics
)

logger = logging.getLogger(__name__)


class GCPProvider(CloudProvider):
    """Google Cloud Platform provider implementation"""

    def _setup(self):
        """Initialize GCP provider"""
        # Auto-detect configuration
        self.project_id = (
                self.config.get('project_id') or
                os.environ.get('GOOGLE_CLOUD_PROJECT') or
                os.environ.get('GCP_PROJECT') or
                'ops0-default'
        )

        self.region = (
                self.config.get('region') or
                os.environ.get('GOOGLE_CLOUD_REGION') or
                'us-central1'
        )

        # Initialize clients lazily
        self._clients = {}

        # Cloud Run service account
        self.service_account = f"ops0-runner@{self.project_id}.iam.gserviceaccount.com"

        logger.info(f"GCP provider initialized for project: {self.project_id}")

    @property
    def name(self) -> str:
        return "gcp"

    @property
    def regions(self) -> List[str]:
        return [
            'us-central1', 'us-east1', 'us-east4', 'us-west1', 'us-west2',
            'europe-west1', 'europe-west2', 'europe-west3', 'europe-west4',
            'asia-east1', 'asia-northeast1', 'asia-southeast1',
            'australia-southeast1', 'southamerica-east1'
        ]

    def _get_client(self, service: str, version: str = None):
        """Get or create Google Cloud client"""
        key = f"{service}:{version}" if version else service

        if key not in self._clients:
            try:
                from google.cloud import run_v2, functions_v1, storage, pubsub_v1, secretmanager
                from google.cloud import monitoring_v3

                if service == 'run':
                    self._clients[key] = run_v2.ServicesClient()
                elif service == 'functions':
                    self._clients[key] = functions_v1.CloudFunctionsServiceClient()
                elif service == 'storage':
                    self._clients[key] = storage.Client(project=self.project_id)
                elif service == 'pubsub':
                    self._clients[key] = pubsub_v1.PublisherClient()
                elif service == 'secretmanager':
                    self._clients[key] = secretmanager.SecretManagerServiceClient()
                elif service == 'monitoring':
                    self._clients[key] = monitoring_v3.MetricServiceClient()
                else:
                    from google.cloud import build
                    discovery = build(service, version)
                    self._clients[key] = discovery

            except ImportError:
                logger.warning("google-cloud libraries not installed - using mock mode")
                from unittest.mock import MagicMock
                self._clients[key] = MagicMock()

        return self._clients[key]

    def deploy_container(self, spec: DeploymentSpec) -> CloudResource:
        """Deploy container to Cloud Run"""
        run_client = self._get_client('run')

        service_name = f"ops0-{spec.step_name}"
        parent = f"projects/{self.project_id}/locations/{self.region}"

        # Cloud Run service configuration
        service = {
            "name": f"{parent}/services/{service_name}",
            "template": {
                "containers": [{
                    "image": spec.image,
                    "command": spec.command,
                    "env": [
                        {"name": k, "value": v}
                        for k, v in spec.environment.items()
                    ],
                    "resources": {
                        "limits": {
                            "cpu": str(spec.cpu),
                            "memory": f"{spec.memory}Mi"
                        }
                    }
                }],
                "scaling": {
                    "min_instance_count": spec.min_instances,
                    "max_instance_count": spec.max_instances
                },
                "timeout": f"{spec.timeout_seconds}s"
            },
            "traffic": [{"type_": "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST", "percent": 100}]
        }

        # Add GPU if required
        if spec.gpu > 0:
            service["template"]["node_selector"] = {
                "run.googleapis.com/accelerator": spec.gpu_type or "nvidia-tesla-t4"
            }

        # Set service account
        service["template"]["service_account"] = self.service_account

        # Deploy or update service
        try:
            operation = run_client.create_service(
                parent=parent,
                service=service,
                service_id=service_name
            )

            # Wait for operation to complete
            response = operation.result()
            service_uri = response.uri

        except Exception as e:
            if "already exists" in str(e):
                # Update existing service
                service_path = f"{parent}/services/{service_name}"
                operation = run_client.update_service(
                    service=service
                )
                response = operation.result()
                service_uri = response.uri
            else:
                raise

        # Make service publicly accessible (optional)
        self._set_iam_policy(service_name, public=True)

        # Create CloudResource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.CONTAINER,
            resource_id=service_name,
            resource_name=service_name,
            region=self.region,
            status=ResourceStatus.RUNNING,
            metadata={
                'service_uri': service_uri,
                'project_id': self.project_id,
                'cpu': spec.cpu,
                'memory': spec.memory,
                'image': spec.image
            },
            tags={
                'step': spec.step_name,
                'environment': self.config.get('environment', 'production')
            }
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Deployed container to Cloud Run: {service_name}")

        return resource

    def deploy_function(self, spec: DeploymentSpec) -> CloudResource:
        """Deploy serverless function to Cloud Functions"""
        functions_client = self._get_client('functions')

        function_name = f"ops0-{spec.step_name}"
        parent = f"projects/{self.project_id}/locations/{self.region}"

        # Package source code
        source_archive_url = self._upload_function_source(spec)

        # Cloud Function configuration
        function = {
            "name": f"{parent}/functions/{function_name}",
            "source_archive_url": source_archive_url,
            "entry_point": "main",
            "runtime": "python39",
            "trigger": {
                "event_type": "google.pubsub.topic.publish",
                "resource": f"projects/{self.project_id}/topics/ops0-{spec.step_name}"
            },
            "environment_variables": spec.environment,
            "available_memory_mb": spec.memory,
            "timeout": f"{spec.timeout_seconds}s",
            "max_instances": spec.max_instances
        }

        # Deploy function
        try:
            operation = functions_client.create_function(
                parent=parent,
                function=function
            )

            # Wait for operation
            response = operation.result()
            function_name_full = response.name

        except Exception as e:
            if "already exists" in str(e):
                # Update existing function
                function_path = f"{parent}/functions/{function_name}"
                operation = functions_client.update_function(
                    function=function
                )
                response = operation.result()
                function_name_full = response.name
            else:
                raise

        # Create trigger topic if needed
        self._create_pubsub_topic(f"ops0-{spec.step_name}")

        # Create CloudResource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.FUNCTION,
            resource_id=function_name,
            resource_name=function_name,
            region=self.region,
            status=ResourceStatus.RUNNING,
            metadata={
                'function_name': function_name_full,
                'runtime': 'python39',
                'memory_mb': spec.memory,
                'timeout_seconds': spec.timeout_seconds,
                'trigger_topic': f"ops0-{spec.step_name}"
            },
            tags={
                'step': spec.step_name,
                'type': 'serverless'
            }
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Deployed function to Cloud Functions: {function_name}")

        return resource

    def create_storage(self, name: str, region: str, **kwargs) -> CloudResource:
        """Create Cloud Storage bucket"""
        storage_client = self._get_client('storage')

        bucket_name = f"ops0-{name}-{int(time.time())}"

        # Create bucket
        bucket = storage_client.bucket(bucket_name)
        bucket.location = region
        bucket.storage_class = kwargs.get('storage_class', 'STANDARD')

        # Enable versioning
        bucket.versioning_enabled = True

        # Set lifecycle rules for cost optimization
        bucket.lifecycle_rules = [{
            'action': {'type': 'SetStorageClass', 'storageClass': 'NEARLINE'},
            'condition': {'age': 30}
        }, {
            'action': {'type': 'SetStorageClass', 'storageClass': 'COLDLINE'},
            'condition': {'age': 90}
        }]

        bucket.create()

        # Set uniform bucket-level access
        bucket.iam_configuration.uniform_bucket_level_access_enabled = True
        bucket.patch()

        # Create CloudResource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.STORAGE,
            resource_id=bucket_name,
            resource_name=bucket_name,
            region=region,
            status=ResourceStatus.RUNNING,
            metadata={
                'endpoint': f"https://storage.googleapis.com/{bucket_name}",
                'storage_class': bucket.storage_class,
                'versioning': True
            }
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Created Cloud Storage bucket: {bucket_name}")

        return resource

    def create_queue(self, name: str, region: str, **kwargs) -> CloudResource:
        """Create Pub/Sub topic and subscription"""
        publisher = self._get_client('pubsub')

        topic_name = f"ops0-{name}"
        topic_path = publisher.topic_path(self.project_id, topic_name)

        # Create topic
        try:
            topic = publisher.create_topic(request={"name": topic_path})
        except Exception as e:
            if "already exists" in str(e):
                topic = publisher.get_topic(request={"topic": topic_path})
            else:
                raise

        # Create subscription
        from google.cloud import pubsub_v1
        subscriber = pubsub_v1.SubscriberClient()

        subscription_name = f"{topic_name}-sub"
        subscription_path = subscriber.subscription_path(self.project_id, subscription_name)

        try:
            subscription = subscriber.create_subscription(
                request={
                    "name": subscription_path,
                    "topic": topic_path,
                    "ack_deadline_seconds": kwargs.get('visibility_timeout', 600),
                    "message_retention_duration": {
                        "seconds": kwargs.get('retention_days', 7) * 86400
                    }
                }
            )
        except Exception as e:
            if "already exists" not in str(e):
                raise

        # Create CloudResource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.QUEUE,
            resource_id=topic_name,
            resource_name=topic_name,
            region="global",  # Pub/Sub is global
            status=ResourceStatus.RUNNING,
            metadata={
                'topic_path': topic_path,
                'subscription_path': subscription_path,
                'project_id': self.project_id
            }
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Created Pub/Sub topic: {topic_name}")

        return resource

    def create_secret(self, name: str, value: str, region: str) -> CloudResource:
        """Store secret in Secret Manager"""
        client = self._get_client('secretmanager')

        secret_id = f"ops0-{name}"
        parent = f"projects/{self.project_id}"

        # Create secret
        try:
            secret = client.create_secret(
                request={
                    "parent": parent,
                    "secret_id": secret_id,
                    "secret": {
                        "replication": {
                            "automatic": {}
                        }
                    }
                }
            )
        except Exception as e:
            if "already exists" in str(e):
                secret_name = f"{parent}/secrets/{secret_id}"
                secret = client.get_secret(request={"name": secret_name})
            else:
                raise

        # Add secret version
        client.add_secret_version(
            request={
                "parent": secret.name,
                "payload": {
                    "data": value.encode('utf-8')
                }
            }
        )

        # Create CloudResource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.SECRET,
            resource_id=secret_id,
            resource_name=secret_id,
            region="global",  # Secrets are global
            status=ResourceStatus.RUNNING,
            metadata={
                'secret_name': secret.name,
                'project_id': self.project_id
            }
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Created secret: {secret_id}")

        return resource

    def get_resource_status(self, resource_id: str) -> ResourceStatus:
        """Get current status of a resource"""
        resource = self.resources.get(resource_id)
        if not resource:
            return ResourceStatus.DELETED

        if resource.resource_type == ResourceType.CONTAINER:
            run_client = self._get_client('run')

            try:
                service_name = f"projects/{self.project_id}/locations/{self.region}/services/{resource_id}"
                service = run_client.get_service(name=service_name)

                # Check conditions
                for condition in service.conditions:
                    if condition.type_ == "Ready":
                        if condition.status == "True":
                            return ResourceStatus.RUNNING
                        else:
                            return ResourceStatus.FAILED

                return ResourceStatus.PROVISIONING

            except Exception as e:
                logger.error(f"Error checking Cloud Run status: {e}")
                return ResourceStatus.FAILED

        elif resource.resource_type == ResourceType.FUNCTION:
            functions_client = self._get_client('functions')

            try:
                function_name = f"projects/{self.project_id}/locations/{self.region}/functions/{resource_id}"
                function = functions_client.get_function(name=function_name)

                state_map = {
                    "ACTIVE": ResourceStatus.RUNNING,
                    "DEPLOYING": ResourceStatus.PROVISIONING,
                    "DELETING": ResourceStatus.DELETING,
                    "UNKNOWN": ResourceStatus.FAILED
                }

                return state_map.get(function.state, ResourceStatus.FAILED)

            except Exception as e:
                logger.error(f"Error checking Cloud Function status: {e}")
                return ResourceStatus.FAILED

        # For other resources, assume running
        return ResourceStatus.RUNNING

    def get_resource_metrics(self, resource_id: str) -> ResourceMetrics:
        """Get Cloud Monitoring metrics for a resource"""
        monitoring_client = self._get_client('monitoring')
        resource = self.resources.get(resource_id)

        if not resource:
            return ResourceMetrics()

        project_name = f"projects/{self.project_id}"
        interval = monitoring_v3.TimeInterval(
            {
                "end_time": {"seconds": int(time.time())},
                "start_time": {"seconds": int(time.time()) - 300}  # Last 5 minutes
            }
        )

        metrics = ResourceMetrics()

        if resource.resource_type == ResourceType.CONTAINER:
            # Cloud Run metrics
            results = monitoring_client.list_time_series(
                request={
                    "name": project_name,
                    "filter": f'resource.type="cloud_run_revision" AND resource.labels.service_name="{resource_id}"',
                    "interval": interval,
                    "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL
                }
            )

            for result in results:
                if "cpu/utilization" in result.metric.type:
                    if result.points:
                        metrics.cpu_percent = result.points[0].value.double_value * 100

                elif "memory/utilization" in result.metric.type:
                    if result.points:
                        metrics.memory_percent = result.points[0].value.double_value * 100

                elif "request_count" in result.metric.type:
                    if result.points:
                        metrics.request_count = int(result.points[0].value.int64_value)

        elif resource.resource_type == ResourceType.FUNCTION:
            # Cloud Functions metrics
            results = monitoring_client.list_time_series(
                request={
                    "name": project_name,
                    "filter": f'resource.type="cloud_function" AND resource.labels.function_name="{resource_id}"',
                    "interval": interval,
                    "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL
                }
            )

            for result in results:
                if "function/execution_count" in result.metric.type:
                    if result.points:
                        metrics.request_count = int(result.points[0].value.int64_value)

                elif "function/execution_times" in result.metric.type:
                    if result.points:
                        metrics.average_latency_ms = result.points[0].value.distribution_value.mean

        return metrics

    def update_resource(self, resource_id: str, spec: DeploymentSpec) -> CloudResource:
        """Update a deployed resource"""
        resource = self.resources.get(resource_id)
        if not resource:
            raise ValueError(f"Resource not found: {resource_id}")

        if resource.resource_type == ResourceType.CONTAINER:
            # Update Cloud Run service
            updated_resource = self.deploy_container(spec)
            resource.status = ResourceStatus.UPDATING
            resource.updated_at = datetime.now()

        elif resource.resource_type == ResourceType.FUNCTION:
            # Update Cloud Function
            updated_resource = self.deploy_function(spec)
            resource.updated_at = datetime.now()

        return resource

    def delete_resource(self, resource_id: str) -> bool:
        """Delete a cloud resource"""
        resource = self.resources.get(resource_id)
        if not resource:
            return True

        try:
            if resource.resource_type == ResourceType.CONTAINER:
                run_client = self._get_client('run')
                service_name = f"projects/{self.project_id}/locations/{self.region}/services/{resource_id}"

                operation = run_client.delete_service(name=service_name)
                operation.result()  # Wait for completion

            elif resource.resource_type == ResourceType.FUNCTION:
                functions_client = self._get_client('functions')
                function_name = f"projects/{self.project_id}/locations/{self.region}/functions/{resource_id}"

                operation = functions_client.delete_function(name=function_name)
                operation.result()

            elif resource.resource_type == ResourceType.STORAGE:
                storage_client = self._get_client('storage')
                bucket = storage_client.bucket(resource_id)

                # Delete all objects first
                blobs = bucket.list_blobs()
                for blob in blobs:
                    blob.delete()

                # Delete bucket
                bucket.delete()

            elif resource.resource_type == ResourceType.QUEUE:
                publisher = self._get_client('pubsub')
                topic_path = resource.metadata['topic_path']

                publisher.delete_topic(request={"topic": topic_path})

            elif resource.resource_type == ResourceType.SECRET:
                client = self._get_client('secretmanager')
                secret_name = resource.metadata['secret_name']

                client.delete_secret(request={"name": secret_name})

            # Remove from tracking
            del self.resources[resource_id]
            return True

        except Exception as e:
            logger.error(f"Error deleting resource {resource_id}: {e}")
            return False

    def list_resources(self, resource_type: Optional[ResourceType] = None) -> List[CloudResource]:
        """List all managed resources"""
        resources = list(self.resources.values())

        if resource_type:
            resources = [r for r in resources if r.resource_type == resource_type]

        return resources

    def estimate_cost(self, resources: List[CloudResource], days: int = 30) -> Dict[str, float]:
        """Estimate GCP costs"""
        cost_breakdown = {
            'compute': 0.0,
            'storage': 0.0,
            'network': 0.0,
            'other': 0.0,
            'total': 0.0
        }

        hours = days * 24

        for resource in resources:
            if resource.resource_type == ResourceType.CONTAINER:
                # Cloud Run pricing
                cpu = resource.metadata.get('cpu', 1)
                memory_gb = resource.metadata.get('memory', 2048) / 1024
                requests_per_day = 100000  # Estimate

                # CPU: $0.00002400 per vCPU-second
                cpu_cost = cpu * 0.0000240 * 86400 * days

                # Memory: $0.00000250 per GiB-second
                memory_cost = memory_gb * 0.0000025 * 86400 * days

                # Requests: $0.40 per million
                request_cost = (requests_per_day * days / 1000000) * 0.40

                cost_breakdown['compute'] += cpu_cost + memory_cost + request_cost

            elif resource.resource_type == ResourceType.FUNCTION:
                # Cloud Functions pricing
                memory_gb = resource.metadata.get('memory_mb', 512) / 1024
                invocations = 1000000  # Estimate 1M/month
                compute_time_ms = 100  # Estimate 100ms per invocation

                # Invocations: $0.40 per million
                invocation_cost = (invocations / 1000000) * 0.40

                # Compute: $0.0000025 per GB-second
                gb_seconds = (invocations * compute_time_ms / 1000) * memory_gb
                compute_cost = gb_seconds * 0.0000025

                cost_breakdown['compute'] += invocation_cost + compute_cost

            elif resource.resource_type == ResourceType.STORAGE:
                # Cloud Storage pricing (standard class)
                storage_gb = 1000  # Estimate 1TB
                operations = 100000  # Class A operations

                storage_cost = storage_gb * 0.020 * (days / 30)
                operation_cost = (operations / 10000) * 0.05

                cost_breakdown['storage'] += storage_cost + operation_cost

            elif resource.resource_type == ResourceType.QUEUE:
                # Pub/Sub pricing
                messages = 10000000  # 10M messages
                message_size_kb = 1  # 1KB average

                # First 10GB free, then $0.06 per GB
                data_gb = (messages * message_size_kb) / 1048576
                if data_gb > 10:
                    cost_breakdown['other'] += (data_gb - 10) * 0.06

        # Network egress estimate (10GB/day to internet)
        cost_breakdown['network'] = 10 * days * 0.12

        cost_breakdown['total'] = sum(
            cost_breakdown[k] for k in ['compute', 'storage', 'network', 'other']
        )

        return cost_breakdown

    # Helper methods
    def _set_iam_policy(self, service_name: str, public: bool = False):
        """Set IAM policy for Cloud Run service"""
        run_client = self._get_client('run')

        if public:
            # Make service publicly accessible
            policy = {
                "bindings": [{
                    "role": "roles/run.invoker",
                    "members": ["allUsers"]
                }]
            }

            service_path = f"projects/{self.project_id}/locations/{self.region}/services/{service_name}"

            try:
                run_client.set_iam_policy(
                    resource=service_path,
                    policy=policy
                )
            except Exception as e:
                logger.warning(f"Failed to set IAM policy: {e}")

    def _create_pubsub_topic(self, topic_name: str):
        """Create Pub/Sub topic if it doesn't exist"""
        publisher = self._get_client('pubsub')
        topic_path = publisher.topic_path(self.project_id, topic_name)

        try:
            publisher.create_topic(request={"name": topic_path})
        except Exception as e:
            if "already exists" not in str(e):
                logger.error(f"Failed to create topic {topic_name}: {e}")

    def _upload_function_source(self, spec: DeploymentSpec) -> str:
        """Upload function source code to GCS"""
        storage_client = self._get_client('storage')

        # Create staging bucket if needed
        bucket_name = f"ops0-functions-{self.project_id}"

        try:
            bucket = storage_client.create_bucket(bucket_name)
        except Exception:
            bucket = storage_client.bucket(bucket_name)

        # In production, this would:
        # 1. Extract code from container image
        # 2. Create deployment package
        # 3. Upload to GCS

        # For now, return a placeholder
        blob_name = f"{spec.step_name}/source.zip"
        blob = bucket.blob(blob_name)

        # Upload dummy source
        blob.upload_from_string(b"dummy-source-code")

        return f"gs://{bucket_name}/{blob_name}"