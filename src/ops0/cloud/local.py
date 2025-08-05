"""
ops0 Local Provider

Local development provider for testing pipelines without cloud deployment.
Simulates cloud behavior using Docker and local resources.
"""

import os
import json
import time
import logging
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import psutil
import docker

from .base import (
    CloudProvider, CloudResource, DeploymentSpec,
    ResourceType, ResourceStatus, ResourceMetrics
)

logger = logging.getLogger(__name__)


class LocalProvider(CloudProvider):
    """Local development provider using Docker"""

    def _setup(self):
        """Initialize local provider"""
        self.workspace = Path(self.config.get('workspace', '/tmp/ops0-local'))
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Docker client
        try:
            self.docker_client = docker.from_env()
            self.docker_available = True
            logger.info("Docker client initialized")
        except Exception as e:
            logger.warning(f"Docker not available: {e}")
            self.docker_available = False
            self.docker_client = None

        # Network for containers
        if self.docker_available:
            try:
                self.docker_network = self.docker_client.networks.create(
                    "ops0-local",
                    driver="bridge"
                )
            except docker.errors.APIError:
                # Network already exists
                self.docker_network = self.docker_client.networks.get("ops0-local")

        # Local registry for resources
        self.local_resources: Dict[str, Dict[str, Any]] = {}

        # Process tracking
        self.processes: Dict[str, subprocess.Popen] = {}

        logger.info(f"Local provider initialized with workspace: {self.workspace}")

    @property
    def name(self) -> str:
        return "local"

    @property
    def regions(self) -> List[str]:
        return ["local"]

    def deploy_container(self, spec: DeploymentSpec) -> CloudResource:
        """Deploy container locally using Docker"""
        if self.docker_available:
            return self._deploy_docker_container(spec)
        else:
            return self._deploy_process(spec)

    def _deploy_docker_container(self, spec: DeploymentSpec) -> CloudResource:
        """Deploy using Docker"""
        container_name = f"ops0-{spec.step_name}-{int(time.time())}"

        # Container configuration
        container_config = {
            "image": spec.image,
            "name": container_name,
            "command": spec.command,
            "environment": spec.environment,
            "detach": True,
            "auto_remove": False,
            "network": "ops0-local",
            "mem_limit": f"{spec.memory}m",
            "cpu_period": 100000,
            "cpu_quota": int(spec.cpu * 100000)  # CPU quota based on vCPUs
        }

        # Add port mapping if specified
        if spec.port:
            container_config["ports"] = {f"{spec.port}/tcp": spec.port}

        # Add volumes
        volumes = {}
        if spec.persistent_volumes:
            for vol in spec.persistent_volumes:
                host_path = self.workspace / "volumes" / vol['name']
                host_path.mkdir(parents=True, exist_ok=True)
                volumes[str(host_path)] = {
                    'bind': vol['mount_path'],
                    'mode': 'rw'
                }

        if volumes:
            container_config["volumes"] = volumes

        try:
            # Pull image if not exists
            try:
                self.docker_client.images.get(spec.image)
            except docker.errors.ImageNotFound:
                logger.info(f"Pulling image: {spec.image}")
                self.docker_client.images.pull(spec.image)

            # Run container
            container = self.docker_client.containers.run(**container_config)

            # Store container info
            self.local_resources[container_name] = {
                "type": "docker",
                "container": container,
                "spec": spec,
                "started_at": datetime.now()
            }

            # Create resource
            resource = CloudResource(
                provider=self.name,
                resource_type=ResourceType.CONTAINER,
                resource_id=container_name,
                resource_name=spec.step_name,
                region="local",
                status=ResourceStatus.RUNNING,
                metadata={
                    "container_id": container.id[:12],
                    "image": spec.image,
                    "cpu": spec.cpu,
                    "memory": spec.memory,
                    "port": spec.port
                }
            )

            self.resources[resource.resource_id] = resource
            logger.info(f"Deployed Docker container: {container_name}")

            return resource

        except Exception as e:
            logger.error(f"Failed to deploy Docker container: {e}")
            raise

    def _deploy_process(self, spec: DeploymentSpec) -> CloudResource:
        """Deploy as local process (fallback when Docker not available)"""
        process_name = f"ops0-{spec.step_name}-{int(time.time())}"

        # Create working directory
        work_dir = self.workspace / "processes" / process_name
        work_dir.mkdir(parents=True, exist_ok=True)

        # Write environment file
        env_file = work_dir / ".env"
        with open(env_file, 'w') as f:
            for key, value in spec.environment.items():
                f.write(f"{key}={value}\n")

        # Create run script
        run_script = work_dir / "run.sh"
        with open(run_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("set -e\n")
            f.write(f"cd {work_dir}\n")
            f.write("source .env\n")
            f.write(" ".join(spec.command) + "\n")

        run_script.chmod(0o755)

        # Start process
        process = subprocess.Popen(
            [str(run_script)],
            cwd=str(work_dir),
            stdout=open(work_dir / "stdout.log", 'w'),
            stderr=open(work_dir / "stderr.log", 'w'),
            env={**os.environ, **spec.environment}
        )

        self.processes[process_name] = process

        # Store process info
        self.local_resources[process_name] = {
            "type": "process",
            "process": process,
            "work_dir": work_dir,
            "spec": spec,
            "started_at": datetime.now()
        }

        # Create resource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.CONTAINER,
            resource_id=process_name,
            resource_name=spec.step_name,
            region="local",
            status=ResourceStatus.RUNNING,
            metadata={
                "pid": process.pid,
                "work_dir": str(work_dir),
                "cpu": spec.cpu,
                "memory": spec.memory
            }
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Deployed local process: {process_name} (PID: {process.pid})")

        return resource

    def deploy_function(self, spec: DeploymentSpec) -> CloudResource:
        """Deploy serverless function locally"""
        # For local development, functions run as short-lived containers
        function_name = f"ops0-func-{spec.step_name}-{int(time.time())}"

        # Create function directory
        func_dir = self.workspace / "functions" / function_name
        func_dir.mkdir(parents=True, exist_ok=True)

        # Create function wrapper
        wrapper_file = func_dir / "wrapper.py"
        with open(wrapper_file, 'w') as f:
            f.write("""
import json
import sys
import os

# Function handler would be loaded here
def handler(event):
    # Placeholder function logic
    return {
        "statusCode": 200,
        "body": json.dumps({"message": "Function executed", "event": event})
    }

if __name__ == "__main__":
    # Read event from stdin or file
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            event = json.load(f)
    else:
        event = {}

    result = handler(event)
    print(json.dumps(result))
""")

        # Store function info
        self.local_resources[function_name] = {
            "type": "function",
            "func_dir": func_dir,
            "spec": spec,
            "created_at": datetime.now()
        }

        # Create resource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.FUNCTION,
            resource_id=function_name,
            resource_name=spec.step_name,
            region="local",
            status=ResourceStatus.RUNNING,
            metadata={
                "function_dir": str(func_dir),
                "runtime": "python3.9",
                "memory": spec.memory,
                "timeout": spec.timeout_seconds
            }
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Deployed local function: {function_name}")

        return resource

    def create_storage(self, name: str, region: str, **kwargs) -> CloudResource:
        """Create local storage directory"""
        storage_name = f"ops0-storage-{name}"
        storage_path = self.workspace / "storage" / storage_name
        storage_path.mkdir(parents=True, exist_ok=True)

        # Create metadata file
        metadata = {
            "created_at": datetime.now().isoformat(),
            "size_limit_gb": kwargs.get('size_limit_gb', 100),
            "access_mode": kwargs.get('access_mode', 'read-write')
        }

        with open(storage_path / ".metadata.json", 'w') as f:
            json.dump(metadata, f)

        # Create resource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.STORAGE,
            resource_id=storage_name,
            resource_name=name,
            region="local",
            status=ResourceStatus.RUNNING,
            metadata={
                "path": str(storage_path),
                "endpoint": f"file://{storage_path}"
            }
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Created local storage: {storage_path}")

        return resource

    def create_queue(self, name: str, region: str, **kwargs) -> CloudResource:
        """Create local queue using Redis or file-based queue"""
        queue_name = f"ops0-queue-{name}"

        if self.docker_available:
            # Deploy Redis container
            try:
                container = self.docker_client.containers.run(
                    "redis:7-alpine",
                    name=queue_name,
                    detach=True,
                    auto_remove=False,
                    network="ops0-local",
                    mem_limit="512m"
                )

                self.local_resources[queue_name] = {
                    "type": "redis",
                    "container": container
                }

                endpoint = f"redis://{queue_name}:6379"

            except Exception as e:
                logger.warning(f"Failed to start Redis: {e}")
                # Fallback to file-based queue
                queue_path = self.workspace / "queues" / queue_name
                queue_path.mkdir(parents=True, exist_ok=True)

                self.local_resources[queue_name] = {
                    "type": "file",
                    "path": queue_path
                }

                endpoint = f"file://{queue_path}"
        else:
            # File-based queue
            queue_path = self.workspace / "queues" / queue_name
            queue_path.mkdir(parents=True, exist_ok=True)

            self.local_resources[queue_name] = {
                "type": "file",
                "path": queue_path
            }

            endpoint = f"file://{queue_path}"

        # Create resource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.QUEUE,
            resource_id=queue_name,
            resource_name=name,
            region="local",
            status=ResourceStatus.RUNNING,
            metadata={
                "endpoint": endpoint,
                "type": self.local_resources[queue_name]["type"]
            }
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Created local queue: {endpoint}")

        return resource

    def create_secret(self, name: str, value: str, region: str) -> CloudResource:
        """Store secret in local file"""
        secret_name = f"ops0-secret-{name}"
        secrets_dir = self.workspace / "secrets"
        secrets_dir.mkdir(parents=True, exist_ok=True)

        # Encrypt value (in production)
        # For local dev, just obfuscate
        import base64
        encoded_value = base64.b64encode(value.encode()).decode()

        secret_file = secrets_dir / f"{secret_name}.secret"
        with open(secret_file, 'w') as f:
            json.dump({
                "name": name,
                "value": encoded_value,
                "created_at": datetime.now().isoformat()
            }, f)

        # Set restrictive permissions
        secret_file.chmod(0o600)

        # Create resource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.SECRET,
            resource_id=secret_name,
            resource_name=name,
            region="local",
            status=ResourceStatus.RUNNING,
            metadata={
                "path": str(secret_file)
            }
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Created local secret: {secret_name}")

        return resource

    def get_resource_status(self, resource_id: str) -> ResourceStatus:
        """Get status of local resource"""
        resource = self.resources.get(resource_id)
        if not resource:
            return ResourceStatus.DELETED

        local_info = self.local_resources.get(resource_id)
        if not local_info:
            return ResourceStatus.DELETED

        if local_info["type"] == "docker":
            try:
                container = local_info["container"]
                container.reload()

                if container.status == "running":
                    return ResourceStatus.RUNNING
                elif container.status == "exited":
                    return ResourceStatus.STOPPED
                else:
                    return ResourceStatus.FAILED

            except docker.errors.NotFound:
                return ResourceStatus.DELETED
            except Exception:
                return ResourceStatus.FAILED

        elif local_info["type"] == "process":
            process = local_info["process"]

            if process.poll() is None:
                # Process still running
                return ResourceStatus.RUNNING
            else:
                # Process exited
                return ResourceStatus.STOPPED

        elif local_info["type"] in ["redis", "file", "function"]:
            # These resources are always "running" unless deleted
            return ResourceStatus.RUNNING

        return ResourceStatus.RUNNING

    def get_resource_metrics(self, resource_id: str) -> ResourceMetrics:
        """Get metrics for local resource"""
        metrics = ResourceMetrics()

        local_info = self.local_resources.get(resource_id)
        if not local_info:
            return metrics

        if local_info["type"] == "docker":
            try:
                container = local_info["container"]
                stats = container.stats(stream=False)

                # Calculate CPU percentage
                cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                            stats["precpu_stats"]["cpu_usage"]["total_usage"]
                system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                               stats["precpu_stats"]["system_cpu_usage"]

                if system_delta > 0:
                    cpu_count = len(stats["cpu_stats"]["cpu_usage"].get("percpu_usage", [1]))
                    metrics.cpu_percent = (cpu_delta / system_delta) * cpu_count * 100

                # Memory usage
                memory_usage = stats["memory_stats"]["usage"]
                memory_limit = stats["memory_stats"]["limit"]
                metrics.memory_percent = (memory_usage / memory_limit) * 100
                metrics.memory_mb = memory_usage / (1024 * 1024)

            except Exception as e:
                logger.debug(f"Failed to get Docker stats: {e}")

        elif local_info["type"] == "process":
            try:
                process = local_info["process"]
                if process.poll() is None:  # Still running
                    proc = psutil.Process(process.pid)

                    metrics.cpu_percent = proc.cpu_percent(interval=0.1)
                    memory_info = proc.memory_info()
                    metrics.memory_mb = memory_info.rss / (1024 * 1024)

                    # Estimate memory percentage
                    total_memory = psutil.virtual_memory().total
                    metrics.memory_percent = (memory_info.rss / total_memory) * 100

            except Exception as e:
                logger.debug(f"Failed to get process stats: {e}")

        # Add some mock request metrics
        import random
        metrics.request_count = random.randint(50, 200)
        metrics.error_count = random.randint(0, 5)
        metrics.average_latency_ms = random.uniform(10, 100)

        return metrics

    def update_resource(self, resource_id: str, spec: DeploymentSpec) -> CloudResource:
        """Update local resource"""
        resource = self.resources.get(resource_id)
        if not resource:
            raise ValueError(f"Resource not found: {resource_id}")

        # For local provider, we'll recreate the resource
        # In production, this would do in-place updates

        # Delete old resource
        self.delete_resource(resource_id)

        # Deploy new version
        if resource.resource_type == ResourceType.CONTAINER:
            return self.deploy_container(spec)
        elif resource.resource_type == ResourceType.FUNCTION:
            return self.deploy_function(spec)
        else:
            raise ValueError(f"Cannot update resource type: {resource.resource_type}")

    def delete_resource(self, resource_id: str) -> bool:
        """Delete local resource"""
        resource = self.resources.get(resource_id)
        if not resource:
            return True

        local_info = self.local_resources.get(resource_id)
        if not local_info:
            return True

        try:
            if local_info["type"] == "docker":
                container = local_info["container"]
                container.stop()
                container.remove()

            elif local_info["type"] == "process":
                process = local_info["process"]
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)

                # Clean up work directory
                work_dir = local_info.get("work_dir")
                if work_dir and work_dir.exists():
                    shutil.rmtree(work_dir)

            elif local_info["type"] == "redis":
                container = local_info["container"]
                container.stop()
                container.remove()

            elif local_info["type"] == "file":
                # Clean up file-based resources
                path = local_info.get("path")
                if path and path.exists():
                    shutil.rmtree(path)

            # Remove from tracking
            del self.resources[resource_id]
            del self.local_resources[resource_id]

            logger.info(f"Deleted local resource: {resource_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete resource {resource_id}: {e}")
            return False

    def list_resources(self, resource_type: Optional[ResourceType] = None) -> List[CloudResource]:
        """List local resources"""
        resources = list(self.resources.values())

        if resource_type:
            resources = [r for r in resources if r.resource_type == resource_type]

        return resources

    def estimate_cost(self, resources: List[CloudResource], days: int = 30) -> Dict[str, float]:
        """Estimate costs (free for local!)"""
        return {
            'compute': 0.0,
            'storage': 0.0,
            'network': 0.0,
            'other': 0.0,
            'total': 0.0
        }

    def cleanup(self):
        """Clean up all local resources"""
        logger.info("Cleaning up local provider resources")

        # Stop all containers
        for resource_id in list(self.resources.keys()):
            self.delete_resource(resource_id)

        # Remove Docker network
        if self.docker_available and hasattr(self, 'docker_network'):
            try:
                self.docker_network.remove()
            except Exception as e:
                logger.debug(f"Failed to remove Docker network: {e}")

        # Clean up workspace (optional)
        if self.config.get('cleanup_workspace', False):
            shutil.rmtree(self.workspace)
            logger.info(f"Removed workspace: {self.workspace}")