"""
Execution engine for ops0 pipelines.
Handles both local and distributed execution with automatic resource management.
"""
import os
import sys
import time
import json
import tempfile
import traceback
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import cloudpickle

from ops0.storage import StorageBackend, LocalStorage, S3Storage
from ops0.parser import build_dag


class ExecutionMode(Enum):
    LOCAL = "local"
    DOCKER = "docker"
    LAMBDA = "lambda"
    KUBERNETES = "kubernetes"


@dataclass
class ExecutionContext:
    """Context for pipeline execution"""
    mode: ExecutionMode = ExecutionMode.LOCAL
    pipeline_name: Optional[str] = None
    execution_id: str = ""
    storage: Optional[StorageBackend] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    _local = threading.local()

    @classmethod
    def current(cls) -> Optional['ExecutionContext']:
        return getattr(cls._local, 'context', None)

    def __enter__(self):
        ExecutionContext._local.context = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ExecutionContext._local.context = None


@dataclass
class StepResult:
    """Result of executing a step"""
    step_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    start_time: float = 0
    end_time: float = 0
    duration: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LocalExecutor:
    """Execute pipelines locally for development and testing"""

    def __init__(self, storage_backend: Optional[StorageBackend] = None):
        self.storage = storage_backend or LocalStorage()

    def _get_execution_order(self, dag: Dict[str, List[str]]) -> List[str]:
        """Déterminer l'ordre d'exécution depuis le DAG"""
        # Algorithme de tri topologique simple
        visited = set()
        order = []

        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for dep in dag.get(node, []):
                visit(dep)
            order.append(node)

        for node in dag:
            visit(node)

        return order[::-1]  # Inverser pour avoir les dépendances en premier

    def _resolve_step_args(self, step_name: str, dag: Dict[str, List[str]],
                           results: Dict[str, StepResult]) -> tuple:
        """Résoudre les arguments d'un step depuis les résultats précédents"""
        args = []
        for dep in dag.get(step_name, []):
            if dep in results and results[dep].success:
                args.append(results[dep].result)
        return tuple(args)

    def execute_step(self, step_config, args: tuple, kwargs: dict) -> StepResult:
        """Execute a single step locally"""
        start_time = time.time()
        result = StepResult(
            step_name=step_config.name,
            success=False,
            start_time=start_time
        )

        try:
            # Execute the function
            output = step_config.func(*args, **kwargs)

            # Store result if needed
            if output is not None:
                result_key = f"{step_config.name}_result"
                self.storage.save(result_key, output)

            result.success = True
            result.result = output

        except Exception as e:
            result.error = str(e)
            result.metadata['traceback'] = traceback.format_exc()

        finally:
            result.end_time = time.time()
            result.duration = result.end_time - result.start_time

        return result

    def execute_pipeline(self, pipeline_config, args: tuple, kwargs: dict) -> Dict[str, StepResult]:
        """Execute a complete pipeline locally"""
        results = {}

        # Build execution DAG
        dag = build_dag(pipeline_config.func, pipeline_config.steps)

        # Execute pipeline function directly for now
        # In a full implementation, this would execute steps based on DAG
        with ExecutionContext(
                mode=ExecutionMode.LOCAL,
                pipeline_name=pipeline_config.name,
                storage=self.storage
        ) as ctx:
            try:
                # Exécuter chaque step dans l'ordre du DAG
                step_results = {}
                for step_name in self._get_execution_order(dag):
                    step_config = pipeline_config.steps.get(step_name)
                    if step_config:
                        # Obtenir les arguments depuis les résultats précédents
                        step_args = self._resolve_step_args(step_name, dag, step_results)
                        step_result = self.execute_step(step_config, step_args, {})
                        step_results[step_name] = step_result
                        if not step_result.success:
                            break

                # Exécuter la fonction pipeline elle-même
                result = pipeline_config.func(*args, **kwargs)
                results['pipeline'] = StepResult(
                    step_name=pipeline_config.name,
                    success=True,
                    result=result,
                    start_time=time.time(),
                    end_time=time.time()
                )
            except Exception as e:
                results['pipeline'] = StepResult(
                    step_name=pipeline_config.name,
                    success=False,
                    error=str(e),
                    metadata={'traceback': traceback.format_exc()}
                )

        return results


class DockerExecutor:
    """Execute steps in Docker containers"""

    def __init__(self, storage_backend: Optional[StorageBackend] = None):
        self.storage = storage_backend or S3Storage()

    def build_container(self, step_config) -> str:
        """Build a Docker container for a step"""
        # Generate Dockerfile
        dockerfile_content = self._generate_dockerfile(step_config)

        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile_path = Path(tmpdir) / "Dockerfile"
            dockerfile_path.write_text(dockerfile_content)

            # Write step function
            step_file = Path(tmpdir) / "step.py"
            step_content = self._generate_step_wrapper(step_config)
            step_file.write_text(step_content)

            # Build image
            image_name = f"ops0-{step_config.name}:latest"
            subprocess.run([
                "docker", "build", "-t", image_name, tmpdir
            ], check=True)

            return image_name

    def _generate_dockerfile(self, step_config) -> str:
        """Generate Dockerfile for a step"""
        base_image = "python:3.9-slim"
        if step_config.gpu:
            base_image = "pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime"

        requirements = "\n".join(step_config.requirements)

        return f"""
FROM {base_image}

WORKDIR /app

# Install requirements
RUN pip install --no-cache-dir cloudpickle boto3
{f'RUN pip install --no-cache-dir {requirements}' if requirements else ''}

# Copy step code
COPY step.py .

# Run step
CMD ["python", "step.py"]
"""

    def _generate_step_wrapper(self, step_config) -> str:
        """Generate wrapper code for containerized execution"""
        return f"""
import os
import json
import cloudpickle
import boto3

# Load function
func_data = {cloudpickle.dumps(step_config.func)}
func = cloudpickle.loads(func_data)

# Load inputs from S3
s3 = boto3.client('s3')
bucket = os.environ.get('OPS0_BUCKET', 'ops0-pipelines')
input_key = os.environ.get('OPS0_INPUT_KEY')

if input_key:
    response = s3.get_object(Bucket=bucket, Key=input_key)
    inputs = cloudpickle.loads(response['Body'].read())
    args = inputs.get('args', ())
    kwargs = inputs.get('kwargs', {{}})
else:
    args = ()
    kwargs = {{}}

# Execute function
result = func(*args, **kwargs)

# Save result to S3
output_key = os.environ.get('OPS0_OUTPUT_KEY')
if output_key and result is not None:
    s3.put_object(
        Bucket=bucket,
        Key=output_key,
        Body=cloudpickle.dumps(result)
    )

print(json.dumps({{'success': True, 'output_key': output_key}}))
"""

    def execute_step(self, step_config, args: tuple, kwargs: dict) -> StepResult:
        """Execute a step in a Docker container"""
        # This is a simplified version
        # Full implementation would handle container lifecycle properly
        image_name = self.build_container(step_config)

        # For MVP, just note this would run in Docker
        return StepResult(
            step_name=step_config.name,
            success=True,
            metadata={'executor': 'docker', 'image': image_name}
        )


class Orchestrator:
    """Main orchestrator for pipeline execution"""

    def __init__(self, mode: ExecutionMode = ExecutionMode.LOCAL):
        self.mode = mode
        self.executors = {
            ExecutionMode.LOCAL: LocalExecutor(),
            ExecutionMode.DOCKER: DockerExecutor(),
        }

    def execute_pipeline(self, pipeline_config, *args, **kwargs) -> Dict[str, Any]:
        """Execute a pipeline with the appropriate executor"""
        executor = self.executors.get(self.mode, LocalExecutor())

        # Generate execution ID
        import uuid
        execution_id = str(uuid.uuid4())

        print(f"🚀 Executing pipeline: {pipeline_config.name}")
        print(f"📍 Execution ID: {execution_id}")
        print(f"🔧 Mode: {self.mode.value}")

        # Execute
        results = executor.execute_pipeline(pipeline_config, args, kwargs)

        # Print summary
        successful = sum(1 for r in results.values() if r.success)
        total = len(results)

        print(f"\n✅ Pipeline completed: {successful}/{total} steps successful")

        for step_name, result in results.items():
            if result.success:
                print(f"  ✓ {step_name} ({result.duration:.2f}s)")
            else:
                print(f"  ✗ {step_name}: {result.error}")

        return {
            'execution_id': execution_id,
            'pipeline': pipeline_config.name,
            'results': results,
            'success': all(r.success for r in results.values())
        }