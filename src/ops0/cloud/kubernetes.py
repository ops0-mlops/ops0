"""
ops0 Kubernetes Provider

Deploy ML pipelines to any Kubernetes cluster.
Works with EKS, GKE, AKS, or self-managed clusters.
"""

import os
import json
import time
import logging
import base64
from typing import Dict, List, Optional, Any
from datetime import datetime

from .base import (
    CloudProvider, CloudResource, DeploymentSpec,
    ResourceType, ResourceStatus, ResourceMetrics
)

logger = logging.getLogger(__name__)


class KubernetesProvider(CloudProvider):
    """Kubernetes provider implementation"""

    def _setup(self):
        """Initialize Kubernetes provider"""
        # Auto-detect configuration
        self.namespace = (
                self.config.get('namespace') or
                os.environ.get('OPS0_K8S_NAMESPACE') or
                'ops0'
        )

        self.cluster_name = (
                self.config.get('cluster_name') or
                os.environ.get('KUBERNETES_CLUSTER_NAME') or
                'default'
        )

        # Initialize clients lazily
        self._clients = {}

        # Storage class for persistent volumes
        self.storage_class = self.config.get('storage_class', 'standard')

        # Ingress configuration
        self.ingress_class = self.config.get('ingress_class', 'nginx')
        self.ingress_domain = self.config.get('ingress_domain', 'ops0.local')

        logger.info(f"Kubernetes provider initialized for namespace: {self.namespace}")

    @property
    def name(self) -> str:
        return "kubernetes"

    @property
    def regions(self) -> List[str]:
        # For Kubernetes, regions are cluster-specific
        return [self.cluster_name]

    def _get_client(self, api_type: str):
        """Get or create Kubernetes client"""
        if api_type not in self._clients:
            try:
                from kubernetes import client, config as k8s_config

                # Try to load config (in-cluster first, then kubeconfig)
                try:
                    k8s_config.load_incluster_config()
                except:
                    k8s_config.load_kube_config()

                if api_type == 'core':
                    self._clients[api_type] = client.CoreV1Api()
                elif api_type == 'apps':
                    self._clients[api_type] = client.AppsV1Api()
                elif api_type == 'batch':
                    self._clients[api_type] = client.BatchV1Api()
                elif api_type == 'networking':
                    self._clients[api_type] = client.NetworkingV1Api()
                elif api_type == 'autoscaling':
                    self._clients[api_type] = client.AutoscalingV2Api()
                elif api_type == 'custom':
                    self._clients[api_type] = client.CustomObjectsApi()
                else:
                    raise ValueError(f"Unknown API type: {api_type}")

            except ImportError:
                logger.warning("kubernetes library not installed - using mock mode")
                from unittest.mock import MagicMock
                self._clients[api_type] = MagicMock()

        return self._clients[api_type]

    def deploy_container(self, spec: DeploymentSpec) -> CloudResource:
        """Deploy container as Kubernetes Deployment"""
        apps_v1 = self._get_client('apps')

        deployment_name = f"ops0-{spec.step_name}"

        # Ensure namespace exists
        self._ensure_namespace()

        # Container resources
        resources = {
            "requests": {
                "cpu": f"{spec.cpu}",
                "memory": f"{spec.memory}Mi"
            },
            "limits": {
                "cpu": f"{spec.cpu * 2}",  # Allow burst
                "memory": f"{spec.memory * 1.5}Mi"
            }
        }

        # Add GPU if required
        if spec.gpu > 0:
            resources["limits"]["nvidia.com/gpu"] = str(spec.gpu)
            resources["requests"]["nvidia.com/gpu"] = str(spec.gpu)

        # Container definition
        container = client.V1Container(
            name=spec.step_name,
            image=spec.image,
            command=spec.command if spec.command else None,
            env=[
                client.V1EnvVar(name=k, value=v)
                for k, v in spec.environment.items()
            ],
            resources=client.V1ResourceRequirements(**resources),
            ports=[client.V1ContainerPort(container_port=spec.port)] if spec.port else None,
            liveness_probe=client.V1Probe(
                http_get=client.V1HTTPGetAction(
                    path=spec.health_check_path,
                    port=spec.port or 8080
                ),
                initial_delay_seconds=30,
                period_seconds=10
            ) if spec.port else None,
            readiness_probe=client.V1Probe(
                http_get=client.V1HTTPGetAction(
                    path=spec.health_check_path,
                    port=spec.port or 8080
                ),
                initial_delay_seconds=5,
                period_seconds=5
            ) if spec.port else None
        )

        # Add volume mounts if needed
        if spec.persistent_volumes:
            container.volume_mounts = []
            for vol in spec.persistent_volumes:
                container.volume_mounts.append(
                    client.V1VolumeMount(
                        name=vol['name'],
                        mount_path=vol['mount_path']
                    )
                )

        # Pod template
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={
                    "app": "ops0",
                    "step": spec.step_name,
                    "managed-by": "ops0"
                }
            ),
            spec=client.V1PodSpec(
                containers=[container],
                restart_policy="Always",
                service_account_name=self._ensure_service_account(spec.step_name),
                node_selector={"gpu": "true"} if spec.gpu > 0 else None,
                tolerations=[
                    client.V1Toleration(
                        key="nvidia.com/gpu",
                        operator="Exists",
                        effect="NoSchedule"
                    )
                ] if spec.gpu > 0 else None
            )
        )

        # Add volumes if needed
        if spec.persistent_volumes:
            template.spec.volumes = []
            for vol in spec.persistent_volumes:
                pvc_name = self._create_pvc(vol['name'], vol.get('size', '10Gi'))
                template.spec.volumes.append(
                    client.V1Volume(
                        name=vol['name'],
                        persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                            claim_name=pvc_name
                        )
                    )
                )

        # Deployment spec
        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(
                name=deployment_name,
                namespace=self.namespace,
                labels={
                    "app": "ops0",
                    "step": spec.step_name
                }
            ),
            spec=client.V1DeploymentSpec(
                replicas=spec.min_instances,
                selector=client.V1LabelSelector(
                    match_labels={
                        "app": "ops0",
                        "step": spec.step_name
                    }
                ),
                template=template,
                strategy=client.V1DeploymentStrategy(
                    type="RollingUpdate",
                    rolling_update=client.V1RollingUpdateDeployment(
                        max_surge="25%",
                        max_unavailable="25%"
                    )
                )
            )
        )

        # Create or update deployment
        try:
            apps_v1.create_namespaced_deployment(
                namespace=self.namespace,
                body=deployment
            )
        except Exception as e:
            if "already exists" in str(e):
                # Update existing deployment
                apps_v1.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=self.namespace,
                    body=deployment
                )
            else:
                raise

        # Create service if port is specified
        service_name = None
        if spec.port:
            service_name = self._create_service(deployment_name, spec.step_name, spec.port)

        # Create HPA for auto-scaling
        self._create_hpa(deployment_name, spec)

        # Create ingress if needed
        ingress_url = None
        if spec.port and service_name:
            ingress_url = self._create_ingress(service_name, spec.step_name, spec.port)

        # Create CloudResource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.CONTAINER,
            resource_id=deployment_name,
            resource_name=deployment_name,
            region=self.cluster_name,
            status=ResourceStatus.PROVISIONING,
            metadata={
                'namespace': self.namespace,
                'deployment_name': deployment_name,
                'service_name': service_name,
                'ingress_url': ingress_url,
                'replicas': spec.min_instances,
                'cpu': spec.cpu,
                'memory': spec.memory,
                'gpu': spec.gpu
            },
            tags={
                'step': spec.step_name,
                'type': 'deployment'
            }
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Deployed container to Kubernetes: {deployment_name}")

        return resource

    def deploy_function(self, spec: DeploymentSpec) -> CloudResource:
        """Deploy as Kubernetes Job or CronJob"""
        batch_v1 = self._get_client('batch')

        job_name = f"ops0-{spec.step_name}-{int(time.time())}"

        # Container definition (similar to deployment)
        container = client.V1Container(
            name=spec.step_name,
            image=spec.image,
            command=spec.command if spec.command else None,
            env=[
                client.V1EnvVar(name=k, value=v)
                for k, v in spec.environment.items()
            ],
            resources=client.V1ResourceRequirements(
                requests={
                    "cpu": f"{spec.cpu}",
                    "memory": f"{spec.memory}Mi"
                },
                limits={
                    "cpu": f"{spec.cpu * 2}",
                    "memory": f"{spec.memory * 1.5}Mi"
                }
            )
        )

        # Job spec
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=job_name,
                namespace=self.namespace,
                labels={
                    "app": "ops0",
                    "step": spec.step_name,
                    "type": "function"
                }
            ),
            spec=client.V1JobSpec(
                parallelism=spec.max_instances,
                completions=spec.max_instances,
                backoff_limit=3,
                ttl_seconds_after_finished=3600,  # Clean up after 1 hour
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={
                            "app": "ops0",
                            "step": spec.step_name
                        }
                    ),
                    spec=client.V1PodSpec(
                        containers=[container],
                        restart_policy="OnFailure",
                        service_account_name=self._ensure_service_account(spec.step_name)
                    )
                )
            )
        )

        # Create job
        batch_v1.create_namespaced_job(
            namespace=self.namespace,
            body=job
        )

        # Create CloudResource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.FUNCTION,
            resource_id=job_name,
            resource_name=job_name,
            region=self.cluster_name,
            status=ResourceStatus.RUNNING,
            metadata={
                'namespace': self.namespace,
                'job_name': job_name,
                'parallelism': spec.max_instances,
                'completions': spec.max_instances
            },
            tags={
                'step': spec.step_name,
                'type': 'job'
            }
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Deployed function to Kubernetes: {job_name}")

        return resource

    def create_storage(self, name: str, region: str, **kwargs) -> CloudResource:
        """Create persistent volume claim"""
        core_v1 = self._get_client('core')

        pvc_name = f"ops0-storage-{name}"
        size = kwargs.get('size', '100Gi')

        # PVC specification
        pvc = client.V1PersistentVolumeClaim(
            api_version="v1",
            kind="PersistentVolumeClaim",
            metadata=client.V1ObjectMeta(
                name=pvc_name,
                namespace=self.namespace,
                labels={
                    "app": "ops0",
                    "type": "storage"
                }
            ),
            spec=client.V1PersistentVolumeClaimSpec(
                access_modes=["ReadWriteOnce"],
                storage_class_name=self.storage_class,
                resources=client.V1ResourceRequirements(
                    requests={"storage": size}
                )
            )
        )

        # Create PVC
        try:
            core_v1.create_namespaced_persistent_volume_claim(
                namespace=self.namespace,
                body=pvc
            )
        except Exception as e:
            if "already exists" not in str(e):
                raise

        # Create CloudResource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.STORAGE,
            resource_id=pvc_name,
            resource_name=pvc_name,
            region=self.cluster_name,
            status=ResourceStatus.RUNNING,
            metadata={
                'namespace': self.namespace,
                'size': size,
                'storage_class': self.storage_class,
                'access_mode': 'ReadWriteOnce'
            }
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Created PVC: {pvc_name}")

        return resource

    def create_queue(self, name: str, region: str, **kwargs) -> CloudResource:
        """Deploy Redis as message queue"""
        apps_v1 = self._get_client('apps')
        core_v1 = self._get_client('core')

        redis_name = f"ops0-queue-{name}"

        # Redis deployment
        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(
                name=redis_name,
                namespace=self.namespace
            ),
            spec=client.V1DeploymentSpec(
                replicas=1,
                selector=client.V1LabelSelector(
                    match_labels={"app": "ops0", "component": "redis", "queue": name}
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"app": "ops0", "component": "redis", "queue": name}
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="redis",
                                image="redis:7-alpine",
                                ports=[client.V1ContainerPort(container_port=6379)],
                                resources=client.V1ResourceRequirements(
                                    requests={"cpu": "100m", "memory": "128Mi"},
                                    limits={"cpu": "500m", "memory": "512Mi"}
                                ),
                                command=["redis-server", "--appendonly", "yes"]
                            )
                        ]
                    )
                )
            )
        )

        # Create deployment
        try:
            apps_v1.create_namespaced_deployment(
                namespace=self.namespace,
                body=deployment
            )
        except Exception as e:
            if "already exists" not in str(e):
                raise

        # Create service
        service = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name=redis_name,
                namespace=self.namespace
            ),
            spec=client.V1ServiceSpec(
                selector={"app": "ops0", "component": "redis", "queue": name},
                ports=[client.V1ServicePort(port=6379, target_port=6379)],
                type="ClusterIP"
            )
        )

        try:
            core_v1.create_namespaced_service(
                namespace=self.namespace,
                body=service
            )
        except Exception as e:
            if "already exists" not in str(e):
                raise

        # Create CloudResource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.QUEUE,
            resource_id=redis_name,
            resource_name=redis_name,
            region=self.cluster_name,
            status=ResourceStatus.RUNNING,
            metadata={
                'namespace': self.namespace,
                'service_name': redis_name,
                'endpoint': f"{redis_name}.{self.namespace}.svc.cluster.local:6379",
                'type': 'redis'
            }
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Created Redis queue: {redis_name}")

        return resource

    def create_secret(self, name: str, value: str, region: str) -> CloudResource:
        """Create Kubernetes Secret"""
        core_v1 = self._get_client('core')

        secret_name = f"ops0-secret-{name}"

        # Secret object
        secret = client.V1Secret(
            api_version="v1",
            kind="Secret",
            metadata=client.V1ObjectMeta(
                name=secret_name,
                namespace=self.namespace,
                labels={
                    "app": "ops0",
                    "type": "secret"
                }
            ),
            type="Opaque",
            string_data={
                name: value
            }
        )

        # Create secret
        try:
            core_v1.create_namespaced_secret(
                namespace=self.namespace,
                body=secret
            )
        except Exception as e:
            if "already exists" in str(e):
                # Update existing secret
                core_v1.patch_namespaced_secret(
                    name=secret_name,
                    namespace=self.namespace,
                    body=secret
                )
            else:
                raise

        # Create CloudResource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.SECRET,
            resource_id=secret_name,
            resource_name=secret_name,
            region=self.cluster_name,
            status=ResourceStatus.RUNNING,
            metadata={
                'namespace': self.namespace,
                'secret_name': secret_name,
                'key': name
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
            apps_v1 = self._get_client('apps')

            try:
                deployment = apps_v1.read_namespaced_deployment(
                    name=resource_id,
                    namespace=self.namespace
                )

                # Check deployment conditions
                for condition in deployment.status.conditions:
                    if condition.type == "Progressing":
                        if condition.status == "True":
                            if deployment.status.ready_replicas == deployment.status.replicas:
                                return ResourceStatus.RUNNING
                            else:
                                return ResourceStatus.PROVISIONING
                        else:
                            return ResourceStatus.FAILED

                return ResourceStatus.PROVISIONING

            except Exception as e:
                if "not found" in str(e).lower():
                    return ResourceStatus.DELETED
                logger.error(f"Error checking deployment status: {e}")
                return ResourceStatus.FAILED

        elif resource.resource_type == ResourceType.FUNCTION:
            batch_v1 = self._get_client('batch')

            try:
                job = batch_v1.read_namespaced_job(
                    name=resource_id,
                    namespace=self.namespace
                )

                if job.status.succeeded:
                    return ResourceStatus.STOPPED  # Job completed
                elif job.status.failed:
                    return ResourceStatus.FAILED
                elif job.status.active:
                    return ResourceStatus.RUNNING
                else:
                    return ResourceStatus.PENDING

            except Exception as e:
                if "not found" in str(e).lower():
                    return ResourceStatus.DELETED
                logger.error(f"Error checking job status: {e}")
                return ResourceStatus.FAILED

        # For other resources, assume running
        return ResourceStatus.RUNNING

    def get_resource_metrics(self, resource_id: str) -> ResourceMetrics:
        """Get metrics from Kubernetes metrics server"""
        custom_api = self._get_client('custom')
        resource = self.resources.get(resource_id)

        if not resource:
            return ResourceMetrics()

        metrics = ResourceMetrics()

        try:
            if resource.resource_type == ResourceType.CONTAINER:
                # Get pod metrics
                pod_metrics = custom_api.list_namespaced_custom_object(
                    group="metrics.k8s.io",
                    version="v1beta1",
                    namespace=self.namespace,
                    plural="pods",
                    label_selector=f"step={resource.metadata.get('step', '')}"
                )

                total_cpu = 0
                total_memory = 0
                pod_count = 0

                for pod in pod_metrics.get('items', []):
                    for container in pod.get('containers', []):
                        # Parse CPU (convert from nano-cores)
                        cpu_str = container.get('usage', {}).get('cpu', '0n')
                        cpu_nano = int(cpu_str.rstrip('n'))
                        total_cpu += cpu_nano / 1e9

                        # Parse memory (convert from Ki)
                        mem_str = container.get('usage', {}).get('memory', '0Ki')
                        mem_ki = int(mem_str.rstrip('Ki'))
                        total_memory += mem_ki / 1024  # Convert to MB

                        pod_count += 1

                if pod_count > 0:
                    metrics.cpu_percent = (total_cpu / pod_count) * 100
                    metrics.memory_mb = total_memory / pod_count

        except Exception as e:
            logger.debug(f"Could not get metrics: {e}")

        return metrics

    def update_resource(self, resource_id: str, spec: DeploymentSpec) -> CloudResource:
        """Update a deployed resource"""
        resource = self.resources.get(resource_id)
        if not resource:
            raise ValueError(f"Resource not found: {resource_id}")

        if resource.resource_type == ResourceType.CONTAINER:
            apps_v1 = self._get_client('apps')

            # Get current deployment
            deployment = apps_v1.read_namespaced_deployment(
                name=resource_id,
                namespace=self.namespace
            )

            # Update container spec
            container = deployment.spec.template.spec.containers[0]
            container.image = spec.image
            container.command = spec.command if spec.command else None
            container.env = [
                client.V1EnvVar(name=k, value=v)
                for k, v in spec.environment.items()
            ]

            # Update resource requirements
            container.resources = client.V1ResourceRequirements(
                requests={
                    "cpu": f"{spec.cpu}",
                    "memory": f"{spec.memory}Mi"
                },
                limits={
                    "cpu": f"{spec.cpu * 2}",
                    "memory": f"{spec.memory * 1.5}Mi"
                }
            )

            # Apply update
            apps_v1.patch_namespaced_deployment(
                name=resource_id,
                namespace=self.namespace,
                body=deployment
            )

            resource.status = ResourceStatus.UPDATING
            resource.updated_at = datetime.now()

        return resource

    def delete_resource(self, resource_id: str) -> bool:
        """Delete a Kubernetes resource"""
        resource = self.resources.get(resource_id)
        if not resource:
            return True

        try:
            if resource.resource_type == ResourceType.CONTAINER:
                apps_v1 = self._get_client('apps')
                core_v1 = self._get_client('core')

                # Delete deployment
                apps_v1.delete_namespaced_deployment(
                    name=resource_id,
                    namespace=self.namespace
                )

                # Delete associated service if exists
                if resource.metadata.get('service_name'):
                    try:
                        core_v1.delete_namespaced_service(
                            name=resource.metadata['service_name'],
                            namespace=self.namespace
                        )
                    except:
                        pass

            elif resource.resource_type == ResourceType.FUNCTION:
                batch_v1 = self._get_client('batch')

                # Delete job
                batch_v1.delete_namespaced_job(
                    name=resource_id,
                    namespace=self.namespace,
                    propagation_policy='Background'
                )

            elif resource.resource_type == ResourceType.STORAGE:
                core_v1 = self._get_client('core')

                # Delete PVC
                core_v1.delete_namespaced_persistent_volume_claim(
                    name=resource_id,
                    namespace=self.namespace
                )

            elif resource.resource_type == ResourceType.QUEUE:
                apps_v1 = self._get_client('apps')
                core_v1 = self._get_client('core')

                # Delete Redis deployment and service
                apps_v1.delete_namespaced_deployment(
                    name=resource_id,
                    namespace=self.namespace
                )

                core_v1.delete_namespaced_service(
                    name=resource_id,
                    namespace=self.namespace
                )

            elif resource.resource_type == ResourceType.SECRET:
                core_v1 = self._get_client('core')

                # Delete secret
                core_v1.delete_namespaced_secret(
                    name=resource_id,
                    namespace=self.namespace
                )

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
        """Estimate costs (varies by K8s provider)"""
        cost_breakdown = {
            'compute': 0.0,
            'storage': 0.0,
            'network': 0.0,
            'other': 0.0,
            'total': 0.0
        }

        # Cost estimation depends on the underlying infrastructure
        # This is a rough estimate assuming AWS EKS pricing

        hours = days * 24

        for resource in resources:
            if resource.resource_type == ResourceType.CONTAINER:
                # Estimate based on resource requests
                cpu = resource.metadata.get('cpu', 1)
                memory_gb = resource.metadata.get('memory', 2048) / 1024
                replicas = resource.metadata.get('replicas', 1)

                # Rough EC2 equivalent pricing
                cpu_cost = cpu * 0.04 * hours * replicas
                memory_cost = memory_gb * 0.004 * hours * replicas

                cost_breakdown['compute'] += cpu_cost + memory_cost

            elif resource.resource_type == ResourceType.STORAGE:
                # EBS pricing estimate
                size_gb = int(resource.metadata.get('size', '100Gi').rstrip('Gi'))
                storage_cost = size_gb * 0.10 * (days / 30)  # $0.10/GB/month

                cost_breakdown['storage'] += storage_cost

        # Add cluster management overhead (EKS pricing)
        cost_breakdown['other'] = 0.10 * hours  # $0.10/hour for EKS

        # Network costs
        cost_breakdown['network'] = 50  # Rough estimate

        cost_breakdown['total'] = sum(
            cost_breakdown[k] for k in ['compute', 'storage', 'network', 'other']
        )

        return cost_breakdown

    # Helper methods
    def _ensure_namespace(self):
        """Ensure namespace exists"""
        core_v1 = self._get_client('core')

        try:
            core_v1.read_namespace(name=self.namespace)
        except Exception:
            # Create namespace
            namespace = client.V1Namespace(
                metadata=client.V1ObjectMeta(
                    name=self.namespace,
                    labels={
                        "app": "ops0",
                        "managed-by": "ops0"
                    }
                )
            )

            core_v1.create_namespace(body=namespace)
            logger.info(f"Created namespace: {self.namespace}")

    def _ensure_service_account(self, name: str) -> str:
        """Ensure service account exists"""
        core_v1 = self._get_client('core')

        sa_name = f"ops0-{name}-sa"

        try:
            core_v1.read_namespaced_service_account(
                name=sa_name,
                namespace=self.namespace
            )
        except Exception:
            # Create service account
            sa = client.V1ServiceAccount(
                metadata=client.V1ObjectMeta(
                    name=sa_name,
                    namespace=self.namespace,
                    labels={
                        "app": "ops0",
                        "step": name
                    }
                )
            )

            core_v1.create_namespaced_service_account(
                namespace=self.namespace,
                body=sa
            )

        return sa_name

    def _create_pvc(self, name: str, size: str) -> str:
        """Create PVC for persistent storage"""
        core_v1 = self._get_client('core')

        pvc_name = f"ops0-pvc-{name}"

        pvc = client.V1PersistentVolumeClaim(
            metadata=client.V1ObjectMeta(
                name=pvc_name,
                namespace=self.namespace
            ),
            spec=client.V1PersistentVolumeClaimSpec(
                access_modes=["ReadWriteOnce"],
                storage_class_name=self.storage_class,
                resources=client.V1ResourceRequirements(
                    requests={"storage": size}
                )
            )
        )

        try:
            core_v1.create_namespaced_persistent_volume_claim(
                namespace=self.namespace,
                body=pvc
            )
        except Exception as e:
            if "already exists" not in str(e):
                raise

        return pvc_name

    def _create_service(self, deployment_name: str, step_name: str, port: int) -> str:
        """Create service for deployment"""
        core_v1 = self._get_client('core')

        service_name = f"{deployment_name}-svc"

        service = client.V1Service(
            metadata=client.V1ObjectMeta(
                name=service_name,
                namespace=self.namespace,
                labels={
                    "app": "ops0",
                    "step": step_name
                }
            ),
            spec=client.V1ServiceSpec(
                selector={
                    "app": "ops0",
                    "step": step_name
                },
                ports=[
                    client.V1ServicePort(
                        port=port,
                        target_port=port,
                        protocol="TCP"
                    )
                ],
                type="ClusterIP"
            )
        )

        try:
            core_v1.create_namespaced_service(
                namespace=self.namespace,
                body=service
            )
        except Exception as e:
            if "already exists" not in str(e):
                raise

        return service_name

    def _create_hpa(self, deployment_name: str, spec: DeploymentSpec):
        """Create Horizontal Pod Autoscaler"""
        autoscaling_v2 = self._get_client('autoscaling')

        hpa_name = f"{deployment_name}-hpa"

        hpa = client.V2HorizontalPodAutoscaler(
            metadata=client.V1ObjectMeta(
                name=hpa_name,
                namespace=self.namespace
            ),
            spec=client.V2HorizontalPodAutoscalerSpec(
                scale_target_ref=client.V2CrossVersionObjectReference(
                    api_version="apps/v1",
                    kind="Deployment",
                    name=deployment_name
                ),
                min_replicas=spec.min_instances,
                max_replicas=spec.max_instances,
                metrics=[
                    client.V2MetricSpec(
                        type="Resource",
                        resource=client.V2ResourceMetricSource(
                            name="cpu",
                            target=client.V2MetricTarget(
                                type="Utilization",
                                average_utilization=int(spec.target_cpu_percent)
                            )
                        )
                    )
                ],
                behavior=client.V2HorizontalPodAutoscalerBehavior(
                    scale_down=client.V2HPAScalingRules(
                        stabilization_window_seconds=spec.scale_down_cooldown,
                        policies=[
                            client.V2HPAScalingPolicy(
                                type="Percent",
                                value=10,
                                period_seconds=60
                            )
                        ]
                    ),
                    scale_up=client.V2HPAScalingRules(
                        stabilization_window_seconds=spec.scale_up_cooldown,
                        policies=[
                            client.V2HPAScalingPolicy(
                                type="Percent",
                                value=100,
                                period_seconds=60
                            )
                        ]
                    )
                )
            )
        )

        try:
            autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(
                namespace=self.namespace,
                body=hpa
            )
        except Exception as e:
            if "already exists" not in str(e):
                logger.warning(f"Could not create HPA: {e}")

    def _create_ingress(self, service_name: str, step_name: str, port: int) -> str:
        """Create ingress for external access"""
        networking_v1 = self._get_client('networking')

        ingress_name = f"{service_name}-ingress"
        host = f"{step_name}.{self.ingress_domain}"

        ingress = client.V1Ingress(
            metadata=client.V1ObjectMeta(
                name=ingress_name,
                namespace=self.namespace,
                annotations={
                    "kubernetes.io/ingress.class": self.ingress_class,
                    "nginx.ingress.kubernetes.io/rewrite-target": "/"
                },
                labels={
                    "app": "ops0",
                    "step": step_name
                }
            ),
            spec=client.V1IngressSpec(
                rules=[
                    client.V1IngressRule(
                        host=host,
                        http=client.V1HTTPIngressRuleValue(
                            paths=[
                                client.V1HTTPIngressPath(
                                    path="/",
                                    path_type="Prefix",
                                    backend=client.V1IngressBackend(
                                        service=client.V1IngressServiceBackend(
                                            name=service_name,
                                            port=client.V1ServiceBackendPort(
                                                number=port
                                            )
                                        )
                                    )
                                )
                            ]
                        )
                    )
                ]
            )
        )

        try:
            networking_v1.create_namespaced_ingress(
                namespace=self.namespace,
                body=ingress
            )
        except Exception as e:
            if "already exists" not in str(e):
                logger.warning(f"Could not create ingress: {e}")
                return None

        return f"http://{host}"