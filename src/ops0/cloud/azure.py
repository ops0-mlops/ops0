"""
ops0 Azure Provider

Zero-configuration Azure deployment for ML pipelines.
Automatically handles Container Instances, Functions, Blob Storage, and more.
"""

import os
import json
import time
import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from .base import (
    CloudProvider, CloudResource, DeploymentSpec,
    ResourceType, ResourceStatus, ResourceMetrics
)

logger = logging.getLogger(__name__)


class AzureProvider(CloudProvider):
    """Microsoft Azure provider implementation"""

    def _setup(self):
        """Initialize Azure provider"""
        # Auto-detect configuration
        self.subscription_id = (
                self.config.get('subscription_id') or
                os.environ.get('AZURE_SUBSCRIPTION_ID') or
                'default-subscription'
        )

        self.resource_group = (
                self.config.get('resource_group') or
                os.environ.get('AZURE_RESOURCE_GROUP') or
                'ops0-resources'
        )

        self.region = (
                self.config.get('region') or
                os.environ.get('AZURE_REGION') or
                'eastus'
        )

        # Initialize clients lazily
        self._clients = {}

        # Storage account for functions and data
        self.storage_account = f"ops0{int(time.time())}"[:24]  # Max 24 chars

        logger.info(f"Azure provider initialized for subscription: {self.subscription_id}")

    @property
    def name(self) -> str:
        return "azure"

    @property
    def regions(self) -> List[str]:
        return [
            'eastus', 'eastus2', 'westus', 'westus2', 'centralus',
            'northeurope', 'westeurope', 'uksouth', 'ukwest',
            'eastasia', 'southeastasia', 'japaneast', 'japanwest',
            'australiaeast', 'australiasoutheast', 'brazilsouth',
            'canadacentral', 'canadaeast', 'centralindia', 'southindia'
        ]

    def _get_client(self, service: str):
        """Get or create Azure client"""
        if service not in self._clients:
            try:
                from azure.identity import DefaultAzureCredential

                credential = DefaultAzureCredential()

                if service == 'container':
                    from azure.mgmt.containerinstance import ContainerInstanceManagementClient
                    self._clients[service] = ContainerInstanceManagementClient(
                        credential, self.subscription_id
                    )
                elif service == 'functions':
                    from azure.mgmt.web import WebSiteManagementClient
                    self._clients[service] = WebSiteManagementClient(
                        credential, self.subscription_id
                    )
                elif service == 'storage':
                    from azure.mgmt.storage import StorageManagementClient
                    self._clients[service] = StorageManagementClient(
                        credential, self.subscription_id
                    )
                elif service == 'servicebus':
                    from azure.mgmt.servicebus import ServiceBusManagementClient
                    self._clients[service] = ServiceBusManagementClient(
                        credential, self.subscription_id
                    )
                elif service == 'keyvault':
                    from azure.mgmt.keyvault import KeyVaultManagementClient
                    self._clients[service] = KeyVaultManagementClient(
                        credential, self.subscription_id
                    )
                elif service == 'monitor':
                    from azure.mgmt.monitor import MonitorManagementClient
                    self._clients[service] = MonitorManagementClient(
                        credential, self.subscription_id
                    )
                elif service == 'resource':
                    from azure.mgmt.resource import ResourceManagementClient
                    self._clients[service] = ResourceManagementClient(
                        credential, self.subscription_id
                    )
                else:
                    raise ValueError(f"Unknown service: {service}")

            except ImportError:
                logger.warning("azure libraries not installed - using mock mode")
                from unittest.mock import MagicMock
                self._clients[service] = MagicMock()

        return self._clients[service]

    def deploy_container(self, spec: DeploymentSpec) -> CloudResource:
        """Deploy container to Azure Container Instances"""
        container_client = self._get_client('container')

        container_group_name = f"ops0-{spec.step_name}"

        # Ensure resource group exists
        self._ensure_resource_group()

        # Container configuration
        container_resource_requests = {
            "cpu": spec.cpu,
            "memory_in_gb": spec.memory / 1024
        }

        # Add GPU if required
        if spec.gpu > 0:
            container_resource_requests["gpu"] = {
                "count": spec.gpu,
                "sku": spec.gpu_type or "K80"
            }

        container = {
            "name": spec.step_name,
            "properties": {
                "image": spec.image,
                "command": spec.command,
                "environmentVariables": [
                    {"name": k, "value": v}
                    for k, v in spec.environment.items()
                ],
                "resources": {
                    "requests": container_resource_requests
                },
                "ports": [{"port": spec.port or 80}] if spec.port else []
            }
        }

        # Container group configuration
        container_group = {
            "location": self.region,
            "properties": {
                "containers": [container],
                "osType": "Linux",
                "restartPolicy": "Always",
                "ipAddress": {
                    "type": "Public",
                    "ports": [{"protocol": "TCP", "port": spec.port or 80}]
                } if spec.port else None
            },
            "tags": {
                "managed-by": "ops0",
                "step": spec.step_name,
                "environment": self.config.get('environment', 'production')
            }
        }

        # Deploy container group
        operation = container_client.container_groups.begin_create_or_update(
            resource_group_name=self.resource_group,
            container_group_name=container_group_name,
            container_group=container_group
        )

        # Wait for deployment
        result = operation.result()

        # Get public IP if assigned
        public_ip = None
        if result.ip_address:
            public_ip = result.ip_address.ip

        # Create CloudResource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.CONTAINER,
            resource_id=container_group_name,
            resource_name=container_group_name,
            region=self.region,
            status=ResourceStatus.RUNNING,
            metadata={
                'resource_group': self.resource_group,
                'container_group_id': result.id,
                'public_ip': public_ip,
                'cpu': spec.cpu,
                'memory_gb': spec.memory / 1024,
                'gpu': spec.gpu
            },
            tags={
                'step': spec.step_name,
                'os': 'linux'
            }
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Deployed container to ACI: {container_group_name}")

        return resource

    def deploy_function(self, spec: DeploymentSpec) -> CloudResource:
        """Deploy serverless function to Azure Functions"""
        functions_client = self._get_client('functions')

        function_app_name = f"ops0-{spec.step_name}-{int(time.time())}"[:32]

        # Ensure storage account exists
        storage_connection_string = self._ensure_storage_account()

        # Create App Service Plan (Consumption plan for serverless)
        plan_name = f"{function_app_name}-plan"

        app_service_plan = {
            "location": self.region,
            "sku": {
                "name": "Y1",  # Consumption plan
                "tier": "Dynamic"
            },
            "properties": {
                "reserved": True  # Linux
            }
        }

        functions_client.app_service_plans.begin_create_or_update(
            resource_group_name=self.resource_group,
            name=plan_name,
            app_service_plan=app_service_plan
        ).result()

        # Create Function App
        function_app = {
            "location": self.region,
            "properties": {
                "serverFarmId": f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.Web/serverfarms/{plan_name}",
                "siteConfig": {
                    "appSettings": [
                                       {
                                           "name": "AzureWebJobsStorage",
                                           "value": storage_connection_string
                                       },
                                       {
                                           "name": "FUNCTIONS_EXTENSION_VERSION",
                                           "value": "~4"
                                       },
                                       {
                                           "name": "FUNCTIONS_WORKER_RUNTIME",
                                           "value": "python"
                                       },
                                       {
                                           "name": "WEBSITE_RUN_FROM_PACKAGE",
                                           "value": "1"
                                       }
                                   ] + [
                                       {"name": k, "value": v}
                                       for k, v in spec.environment.items()
                                   ],
                    "linuxFxVersion": "PYTHON|3.9"
                },
                "reserved": True  # Linux
            },
            "tags": {
                "managed-by": "ops0",
                "step": spec.step_name
            }
        }

        operation = functions_client.web_apps.begin_create_or_update(
            resource_group_name=self.resource_group,
            name=function_app_name,
            site_envelope=function_app
        )

        result = operation.result()

        # Deploy function code
        self._deploy_function_code(function_app_name, spec)

        # Create CloudResource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.FUNCTION,
            resource_id=function_app_name,
            resource_name=function_app_name,
            region=self.region,
            status=ResourceStatus.RUNNING,
            metadata={
                'resource_group': self.resource_group,
                'function_app_id': result.id,
                'default_hostname': result.default_host_name,
                'runtime': 'python39',
                'plan': 'consumption'
            },
            tags={
                'step': spec.step_name,
                'type': 'serverless'
            }
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Deployed function to Azure Functions: {function_app_name}")

        return resource

    def create_storage(self, name: str, region: str, **kwargs) -> CloudResource:
        """Create Azure Blob Storage container"""
        storage_client = self._get_client('storage')

        # Ensure storage account exists
        storage_account_name = self._ensure_storage_account()

        # Create container
        container_name = f"ops0-{name}".lower()

        from azure.storage.blob import BlobServiceClient
        blob_service = BlobServiceClient(
            account_url=f"https://{storage_account_name}.blob.core.windows.net",
            credential=self._get_storage_key(storage_account_name)
        )

        container_client = blob_service.create_container(
            name=container_name,
            public_access=kwargs.get('public_access', 'None')
        )

        # Set lifecycle management
        management_policy = {
            "properties": {
                "policy": {
                    "rules": [{
                        "name": "ops0-lifecycle",
                        "type": "Lifecycle",
                        "definition": {
                            "filters": {
                                "blobTypes": ["blockBlob"]
                            },
                            "actions": {
                                "baseBlob": {
                                    "tierToCool": {"daysAfterModificationGreaterThan": 30},
                                    "tierToArchive": {"daysAfterModificationGreaterThan": 90}
                                }
                            }
                        }
                    }]
                }
            }
        }

        storage_client.management_policies.create_or_update(
            resource_group_name=self.resource_group,
            account_name=storage_account_name,
            management_policy_name="default",
            properties=management_policy
        )

        # Create CloudResource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.STORAGE,
            resource_id=container_name,
            resource_name=container_name,
            region=region,
            status=ResourceStatus.RUNNING,
            metadata={
                'storage_account': storage_account_name,
                'endpoint': f"https://{storage_account_name}.blob.core.windows.net/{container_name}",
                'access_tier': 'Hot'
            }
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Created Blob Storage container: {container_name}")

        return resource

    def create_queue(self, name: str, region: str, **kwargs) -> CloudResource:
        """Create Service Bus queue"""
        servicebus_client = self._get_client('servicebus')

        # Create Service Bus namespace
        namespace_name = f"ops0-{int(time.time())}"[:24]

        namespace = {
            "location": region,
            "sku": {
                "name": "Standard",
                "tier": "Standard"
            },
            "tags": {
                "managed-by": "ops0"
            }
        }

        operation = servicebus_client.namespaces.begin_create_or_update(
            resource_group_name=self.resource_group,
            namespace_name=namespace_name,
            parameters=namespace
        )

        namespace_result = operation.result()

        # Create queue
        queue_name = f"ops0-{name}"

        queue_params = {
            "properties": {
                "lock_duration": f"PT{kwargs.get('visibility_timeout', 300)}S",
                "max_size_in_megabytes": 1024,
                "default_message_time_to_live": f"P{kwargs.get('retention_days', 7)}D",
                "enable_partitioning": True
            }
        }

        servicebus_client.queues.create_or_update(
            resource_group_name=self.resource_group,
            namespace_name=namespace_name,
            queue_name=queue_name,
            parameters=queue_params
        )

        # Get connection string
        keys = servicebus_client.namespaces.list_keys(
            resource_group_name=self.resource_group,
            namespace_name=namespace_name,
            authorization_rule_name="RootManageSharedAccessKey"
        )

        # Create CloudResource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.QUEUE,
            resource_id=queue_name,
            resource_name=queue_name,
            region=region,
            status=ResourceStatus.RUNNING,
            metadata={
                'namespace': namespace_name,
                'connection_string': keys.primary_connection_string,
                'endpoint': f"https://{namespace_name}.servicebus.windows.net/{queue_name}"
            }
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Created Service Bus queue: {queue_name}")

        return resource

    def create_secret(self, name: str, value: str, region: str) -> CloudResource:
        """Store secret in Azure Key Vault"""
        keyvault_client = self._get_client('keyvault')

        # Create Key Vault
        vault_name = f"ops0-{int(time.time())}"[:24]

        vault_params = {
            "location": region,
            "properties": {
                "sku": {
                    "family": "A",
                    "name": "standard"
                },
                "tenant_id": os.environ.get('AZURE_TENANT_ID'),
                "access_policies": [],
                "enabled_for_deployment": True,
                "enabled_for_template_deployment": True
            },
            "tags": {
                "managed-by": "ops0"
            }
        }

        operation = keyvault_client.vaults.begin_create_or_update(
            resource_group_name=self.resource_group,
            vault_name=vault_name,
            parameters=vault_params
        )

        vault = operation.result()

        # Add secret
        from azure.keyvault.secrets import SecretClient
        from azure.identity import DefaultAzureCredential

        secret_client = SecretClient(
            vault_url=f"https://{vault_name}.vault.azure.net",
            credential=DefaultAzureCredential()
        )

        secret_name = f"ops0-{name}"
        secret_client.set_secret(secret_name, value)

        # Create CloudResource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.SECRET,
            resource_id=secret_name,
            resource_name=secret_name,
            region=region,
            status=ResourceStatus.RUNNING,
            metadata={
                'vault_name': vault_name,
                'vault_url': f"https://{vault_name}.vault.azure.net",
                'secret_id': f"https://{vault_name}.vault.azure.net/secrets/{secret_name}"
            }
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Created secret: {secret_name}")

        return resource

    def get_resource_status(self, resource_id: str) -> ResourceStatus:
        """Get current status of a resource"""
        resource = self.resources.get(resource_id)
        if not resource:
            return ResourceStatus.DELETED

        if resource.resource_type == ResourceType.CONTAINER:
            container_client = self._get_client('container')

            try:
                container_group = container_client.container_groups.get(
                    resource_group_name=self.resource_group,
                    container_group_name=resource_id
                )

                state = container_group.instance_view.state

                if state == "Running":
                    return ResourceStatus.RUNNING
                elif state == "Pending":
                    return ResourceStatus.PROVISIONING
                elif state == "Stopped":
                    return ResourceStatus.STOPPED
                else:
                    return ResourceStatus.FAILED

            except Exception as e:
                logger.error(f"Error checking ACI status: {e}")
                return ResourceStatus.FAILED

        elif resource.resource_type == ResourceType.FUNCTION:
            functions_client = self._get_client('functions')

            try:
                function_app = functions_client.web_apps.get(
                    resource_group_name=self.resource_group,
                    name=resource_id
                )

                if function_app.state == "Running":
                    return ResourceStatus.RUNNING
                elif function_app.state == "Stopped":
                    return ResourceStatus.STOPPED
                else:
                    return ResourceStatus.FAILED

            except Exception as e:
                logger.error(f"Error checking Function App status: {e}")
                return ResourceStatus.FAILED

        # For other resources, assume running
        return ResourceStatus.RUNNING

    def get_resource_metrics(self, resource_id: str) -> ResourceMetrics:
        """Get Azure Monitor metrics for a resource"""
        monitor_client = self._get_client('monitor')
        resource = self.resources.get(resource_id)

        if not resource:
            return ResourceMetrics()

        metrics = ResourceMetrics()

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=5)

        if resource.resource_type == ResourceType.CONTAINER:
            # Get ACI metrics
            resource_uri = resource.metadata['container_group_id']

            try:
                # CPU usage
                cpu_result = monitor_client.metrics.list(
                    resource_uri=resource_uri,
                    metricnames='CpuUsage',
                    timespan=f"{start_time}/{end_time}",
                    interval='PT1M',
                    aggregation='Average'
                )

                if cpu_result.value and cpu_result.value[0].timeseries:
                    timeseries = cpu_result.value[0].timeseries[0]
                    if timeseries.data:
                        metrics.cpu_percent = timeseries.data[-1].average

                # Memory usage
                memory_result = monitor_client.metrics.list(
                    resource_uri=resource_uri,
                    metricnames='MemoryUsage',
                    timespan=f"{start_time}/{end_time}",
                    interval='PT1M',
                    aggregation='Average'
                )

                if memory_result.value and memory_result.value[0].timeseries:
                    timeseries = memory_result.value[0].timeseries[0]
                    if timeseries.data:
                        metrics.memory_mb = timeseries.data[-1].average / (1024 * 1024)

            except Exception as e:
                logger.error(f"Error getting ACI metrics: {e}")

        elif resource.resource_type == ResourceType.FUNCTION:
            # Get Function App metrics
            resource_uri = resource.metadata['function_app_id']

            try:
                # Function execution count
                exec_result = monitor_client.metrics.list(
                    resource_uri=resource_uri,
                    metricnames='FunctionExecutionCount',
                    timespan=f"{start_time}/{end_time}",
                    interval='PT1M',
                    aggregation='Total'
                )

                if exec_result.value and exec_result.value[0].timeseries:
                    timeseries = exec_result.value[0].timeseries[0]
                    if timeseries.data:
                        metrics.request_count = int(timeseries.data[-1].total or 0)

            except Exception as e:
                logger.error(f"Error getting Function App metrics: {e}")

        return metrics

    def update_resource(self, resource_id: str, spec: DeploymentSpec) -> CloudResource:
        """Update a deployed resource"""
        resource = self.resources.get(resource_id)
        if not resource:
            raise ValueError(f"Resource not found: {resource_id}")

        if resource.resource_type == ResourceType.CONTAINER:
            # Update ACI container group
            updated_resource = self.deploy_container(spec)
            resource.status = ResourceStatus.UPDATING
            resource.updated_at = datetime.now()

        elif resource.resource_type == ResourceType.FUNCTION:
            # Update Function App
            functions_client = self._get_client('functions')

            # Update app settings
            app_settings = [
                {"name": k, "value": v}
                for k, v in spec.environment.items()
            ]

            functions_client.web_apps.update_application_settings(
                resource_group_name=self.resource_group,
                name=resource_id,
                app_settings=app_settings
            )

            resource.updated_at = datetime.now()

        return resource

    def delete_resource(self, resource_id: str) -> bool:
        """Delete a cloud resource"""
        resource = self.resources.get(resource_id)
        if not resource:
            return True

        try:
            if resource.resource_type == ResourceType.CONTAINER:
                container_client = self._get_client('container')

                operation = container_client.container_groups.begin_delete(
                    resource_group_name=self.resource_group,
                    container_group_name=resource_id
                )
                operation.result()

            elif resource.resource_type == ResourceType.FUNCTION:
                functions_client = self._get_client('functions')

                functions_client.web_apps.delete(
                    resource_group_name=self.resource_group,
                    name=resource_id
                )

            elif resource.resource_type == ResourceType.STORAGE:
                # Delete container
                from azure.storage.blob import BlobServiceClient

                storage_account = resource.metadata['storage_account']
                blob_service = BlobServiceClient(
                    account_url=f"https://{storage_account}.blob.core.windows.net",
                    credential=self._get_storage_key(storage_account)
                )

                blob_service.delete_container(resource_id)

            elif resource.resource_type == ResourceType.QUEUE:
                servicebus_client = self._get_client('servicebus')

                namespace = resource.metadata['namespace']
                servicebus_client.queues.delete(
                    resource_group_name=self.resource_group,
                    namespace_name=namespace,
                    queue_name=resource_id
                )

            elif resource.resource_type == ResourceType.SECRET:
                # Delete from Key Vault
                from azure.keyvault.secrets import SecretClient
                from azure.identity import DefaultAzureCredential

                vault_name = resource.metadata['vault_name']
                secret_client = SecretClient(
                    vault_url=f"https://{vault_name}.vault.azure.net",
                    credential=DefaultAzureCredential()
                )

                secret_client.begin_delete_secret(resource_id).result()

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
        """Estimate Azure costs"""
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
                # ACI pricing
                cpu = resource.metadata.get('cpu', 1)
                memory_gb = resource.metadata.get('memory_gb', 2)
                gpu = resource.metadata.get('gpu', 0)

                # CPU: $0.0000125 per vCPU second
                cpu_cost = cpu * 0.0000125 * 3600 * hours

                # Memory: $0.0000035 per GB second
                memory_cost = memory_gb * 0.0000035 * 3600 * hours

                # GPU: ~$0.90 per GPU hour
                gpu_cost = gpu * 0.90 * hours

                cost_breakdown['compute'] += cpu_cost + memory_cost + gpu_cost

            elif resource.resource_type == ResourceType.FUNCTION:
                # Azure Functions pricing (Consumption plan)
                executions = 1000000  # Estimate 1M executions
                gb_seconds = 400000  # Estimate

                # First 1M executions free, then $0.20 per million
                if executions > 1000000:
                    exec_cost = ((executions - 1000000) / 1000000) * 0.20
                else:
                    exec_cost = 0

                # First 400K GB-s free, then $0.000016 per GB-s
                if gb_seconds > 400000:
                    compute_cost = (gb_seconds - 400000) * 0.000016
                else:
                    compute_cost = 0

                cost_breakdown['compute'] += exec_cost + compute_cost

            elif resource.resource_type == ResourceType.STORAGE:
                # Blob Storage pricing (Hot tier)
                storage_gb = 1000  # Estimate 1TB
                transactions = 100000  # Estimate

                # Storage: $0.0184 per GB/month (Hot tier)
                storage_cost = storage_gb * 0.0184 * (days / 30)

                # Transactions: $0.0004 per 10K operations
                transaction_cost = (transactions / 10000) * 0.0004

                cost_breakdown['storage'] += storage_cost + transaction_cost

            elif resource.resource_type == ResourceType.QUEUE:
                # Service Bus pricing (Standard tier)
                messages = 10000000  # 10M messages

                # $0.80 per million messages
                cost_breakdown['other'] += (messages / 1000000) * 0.80

        # Network bandwidth estimate (10GB/day egress)
        cost_breakdown['network'] = 10 * days * 0.087

        cost_breakdown['total'] = sum(
            cost_breakdown[k] for k in ['compute', 'storage', 'network', 'other']
        )

        return cost_breakdown

    # Helper methods
    def _ensure_resource_group(self):
        """Ensure resource group exists"""
        resource_client = self._get_client('resource')

        try:
            resource_client.resource_groups.get(self.resource_group)
        except Exception:
            # Create resource group
            resource_client.resource_groups.create_or_update(
                self.resource_group,
                {"location": self.region}
            )
            logger.info(f"Created resource group: {self.resource_group}")

    def _ensure_storage_account(self) -> str:
        """Ensure storage account exists and return connection string"""
        storage_client = self._get_client('storage')

        self._ensure_resource_group()

        # Check if storage account exists
        try:
            storage_client.storage_accounts.get_properties(
                resource_group_name=self.resource_group,
                account_name=self.storage_account
            )
        except Exception:
            # Create storage account
            storage_params = {
                "sku": {"name": "Standard_LRS"},
                "kind": "StorageV2",
                "location": self.region,
                "properties": {
                    "supportsHttpsTrafficOnly": True
                }
            }

            operation = storage_client.storage_accounts.begin_create(
                resource_group_name=self.resource_group,
                account_name=self.storage_account,
                parameters=storage_params
            )
            operation.result()
            logger.info(f"Created storage account: {self.storage_account}")

        # Get connection string
        keys = storage_client.storage_accounts.list_keys(
            resource_group_name=self.resource_group,
            account_name=self.storage_account
        )

        key = keys.keys[0].value
        connection_string = (
            f"DefaultEndpointsProtocol=https;"
            f"AccountName={self.storage_account};"
            f"AccountKey={key};"
            f"EndpointSuffix=core.windows.net"
        )

        return connection_string

    def _get_storage_key(self, account_name: str) -> str:
        """Get storage account key"""
        storage_client = self._get_client('storage')

        keys = storage_client.storage_accounts.list_keys(
            resource_group_name=self.resource_group,
            account_name=account_name
        )

        return keys.keys[0].value

    def _deploy_function_code(self, function_app_name: str, spec: DeploymentSpec):
        """Deploy code to Azure Function"""
        # In production, this would:
        # 1. Package the function code
        # 2. Upload to storage
        # 3. Deploy to function app

        # For now, just log
        logger.info(f"Deploying code to function app: {function_app_name}")