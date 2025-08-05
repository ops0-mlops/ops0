"""
ops0 Runtime Containers

Automatic containerization for ML pipeline steps.
Zero configuration needed - just write Python!
"""

import ast
import hashlib
import inspect
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

logger = logging.getLogger(__name__)


@dataclass
class ContainerSpec:
    """Specification for a containerized step"""
    step_name: str
    image_tag: str
    dockerfile_content: str
    requirements: List[str]
    system_packages: List[str]
    base_image: str
    memory_limit: str = "2Gi"
    cpu_limit: str = "1"
    needs_gpu: bool = False
    environment_vars: Dict[str, str] = None

    def __post_init__(self):
        if self.environment_vars is None:
            self.environment_vars = {}


class FunctionAnalyzer:
    """
    Analyzes Python functions to detect dependencies and requirements.

    Uses AST parsing to understand what a function needs.
    """

    def __init__(self, func: callable):
        self.func = func
        self.source = inspect.getsource(func)
        self.tree = ast.parse(self.source)

    def get_imports(self) -> Set[str]:
        """Extract all import statements from the function"""
        imports = set()

        class ImportVisitor(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])

            def visit_ImportFrom(self, node):
                if node.module:
                    imports.add(node.module.split('.')[0])

        visitor = ImportVisitor()
        visitor.visit(self.tree)
        return imports

    def get_ml_frameworks(self) -> Set[str]:
        """Detect ML frameworks to optimize container"""
        imports = self.get_imports()

        ml_frameworks = {
            'sklearn': 'scikit-learn',
            'torch': 'torch',
            'tensorflow': 'tensorflow',
            'keras': 'keras',
            'pandas': 'pandas',
            'numpy': 'numpy',
            'scipy': 'scipy',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'plotly': 'plotly',
            'xgboost': 'xgboost',
            'lightgbm': 'lightgbm',
            'catboost': 'catboost'
        }

        detected = set()
        for import_name in imports:
            if import_name in ml_frameworks:
                detected.add(ml_frameworks[import_name])

        return detected

    def needs_gpu(self) -> bool:
        """Detect if step needs GPU support"""
        imports = self.get_imports()
        source_lower = self.source.lower()

        # Check for GPU-related imports
        gpu_frameworks = {'torch', 'tensorflow', 'keras', 'cupy', 'cudf'}
        if any(fw in imports for fw in gpu_frameworks):
            return True

        # Check for GPU-related keywords in source
        gpu_keywords = ['cuda', 'gpu', '.to(device)', '.cuda()', 'device_type']
        if any(keyword in source_lower for keyword in gpu_keywords):
            return True

        return False

    def estimate_memory_needs(self) -> str:
        """Estimate memory requirements based on code patterns"""
        source_lower = self.source.lower()

        # High memory patterns
        high_memory_patterns = [
            'large_dataset', 'big_data', 'dataframe.read_csv',
            'load_large', 'gigabyte', 'terabyte'
        ]

        if any(pattern in source_lower for pattern in high_memory_patterns):
            return "8Gi"
        elif any(ml in self.get_ml_frameworks() for ml in ['torch', 'tensorflow']):
            return "4Gi"
        else:
            return "2Gi"


class RequirementsGenerator:
    """
    Generates requirements.txt automatically from code analysis.

    No more manual dependency management!
    """

    def __init__(self):
        # Map of import names to pip packages
        self.import_to_package = {
            'sklearn': 'scikit-learn==1.3.0',
            'cv2': 'opencv-python==4.8.0',
            'PIL': 'Pillow==10.0.0',
            'yaml': 'PyYAML==6.0.1',
            'bs4': 'beautifulsoup4==4.12.2',
            'dotenv': 'python-dotenv==1.0.0',
            'jwt': 'PyJWT==2.8.0',
            'sqlalchemy': 'SQLAlchemy==2.0.19',
            'fastapi': 'fastapi==0.101.0',
            'uvicorn': 'uvicorn==0.23.2',
            'pydantic': 'pydantic==2.1.1',
            'httpx': 'httpx==0.24.1',
            'redis': 'redis==4.6.0',
            'psycopg2': 'psycopg2-binary==2.9.7',
            'motor': 'motor==3.2.0',
            'aiohttp': 'aiohttp==3.8.5',
            'requests': 'requests==2.31.0',
            'boto3': 'boto3==1.28.17',
            'google': 'google-cloud-storage==2.10.0',
            'azure': 'azure-storage-blob==12.17.0',
        }

        # Standard library modules (don't need pip install)
        self.stdlib_modules = {
            'os', 'sys', 'time', 'datetime', 'json', 're', 'math',
            'random', 'collections', 'itertools', 'functools',
            'pathlib', 'typing', 'logging', 'asyncio', 'subprocess',
            'threading', 'multiprocessing', 'queue', 'socket',
            'urllib', 'http', 'email', 'csv', 'sqlite3', 'pickle',
            'copy', 'shutil', 'tempfile', 'glob', 'fnmatch',
            'hashlib', 'hmac', 'secrets', 'uuid', 'platform',
            'argparse', 'configparser', 'enum', 'dataclasses'
        }

    def generate_from_imports(self, imports: Set[str]) -> List[str]:
        """Generate requirements list from imports"""
        requirements = []

        for import_name in imports:
            # Skip standard library
            if import_name in self.stdlib_modules:
                continue

            # Check if we have a known mapping
            if import_name in self.import_to_package:
                requirements.append(self.import_to_package[import_name])
            else:
                # Default: assume import name is package name
                # In production, we'd check PyPI
                requirements.append(f"{import_name}==*")

        # Always include ops0
        requirements.append("ops0")

        return sorted(list(set(requirements)))

    def detect_system_packages(self, ml_frameworks: Set[str]) -> List[str]:
        """Detect system packages needed"""
        system_packages = []

        # Basic build tools
        system_packages.extend(['build-essential', 'python3-dev'])

        # Framework-specific system dependencies
        if 'opencv-python' in str(ml_frameworks):
            system_packages.extend(['libgl1-mesa-glx', 'libglib2.0-0'])

        if 'scipy' in ml_frameworks or 'scikit-learn' in ml_frameworks:
            system_packages.extend(['gfortran', 'libopenblas-dev', 'liblapack-dev'])

        if 'matplotlib' in ml_frameworks or 'seaborn' in ml_frameworks:
            system_packages.append('libfreetype6-dev')

        return list(set(system_packages))


class ContainerBuilder:
    """
    Builds optimized containers for each pipeline step.

    Handles all the Docker complexity automatically.
    """

    def __init__(self):
        self.analyzer = None
        self.requirements_gen = RequirementsGenerator()
        self.built_images = set()

    def containerize_step(self, step_metadata: Dict[str, Any]) -> ContainerSpec:
        """
        Create container specification for a step.

        This is where the magic happens - analyzing code and
        creating optimized containers automatically.
        """
        step_name = step_metadata.get("name", "unknown")
        func = step_metadata.get("func")

        print(f"📦 Analyzing step: {step_name}")

        # Analyze function
        self.analyzer = FunctionAnalyzer(func)
        imports = self.analyzer.get_imports()
        ml_frameworks = self.analyzer.get_ml_frameworks()
        needs_gpu = self.analyzer.needs_gpu()
        memory_limit = self.analyzer.estimate_memory_needs()

        print(f"  📊 Detected imports: {imports}")
        print(f"  🤖 ML frameworks: {ml_frameworks}")
        print(f"  🎮 GPU required: {needs_gpu}")
        print(f"  💾 Memory estimate: {memory_limit}")

        # Generate requirements
        requirements = self.requirements_gen.generate_from_imports(imports)
        system_packages = self.requirements_gen.detect_system_packages(ml_frameworks)

        # Select base image
        base_image = self._select_base_image(ml_frameworks, needs_gpu)

        # Generate Dockerfile
        dockerfile_content = self._generate_dockerfile(
            base_image=base_image,
            step_name=step_name,
            requirements=requirements,
            system_packages=system_packages,
            needs_gpu=needs_gpu
        )

        # Create unique image tag
        content_hash = hashlib.sha256(
            f"{step_name}{requirements}{system_packages}".encode()
        ).hexdigest()[:8]

        image_tag = f"ops0/{step_name.lower()}:{content_hash}"

        spec = ContainerSpec(
            step_name=step_name,
            image_tag=image_tag,
            dockerfile_content=dockerfile_content,
            requirements=requirements,
            system_packages=system_packages,
            base_image=base_image,
            memory_limit=memory_limit,
            cpu_limit="2" if needs_gpu else "1",
            needs_gpu=needs_gpu,
            environment_vars={
                "OPS0_STEP_NAME": step_name,
                "OPS0_RUNTIME_MODE": "container"
            }
        )

        print(f"  ✅ Container spec created: {image_tag}")

        return spec

    def _select_base_image(self, ml_frameworks: Set[str], needs_gpu: bool) -> str:
        """Select optimal base image for the step"""
        # GPU-optimized images
        if needs_gpu:
            if 'torch' in ml_frameworks:
                return "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime"
            elif 'tensorflow' in ml_frameworks:
                return "tensorflow/tensorflow:2.13.0-gpu"
            else:
                return "nvidia/cuda:11.7.1-runtime-ubuntu22.04"

        # CPU-optimized images
        if 'torch' in ml_frameworks:
            return "pytorch/pytorch:2.0.1-cpu-py3.10"
        elif 'tensorflow' in ml_frameworks:
            return "tensorflow/tensorflow:2.13.0"
        elif ml_frameworks:  # Any ML framework
            return "python:3.10-slim-bullseye"
        else:  # Minimal
            return "python:3.10-alpine"

    def _generate_dockerfile(
            self,
            base_image: str,
            step_name: str,
            requirements: List[str],
            system_packages: List[str],
            needs_gpu: bool
    ) -> str:
        """Generate optimized Dockerfile content"""

        dockerfile = f"""# ops0 Auto-generated Dockerfile
# Step: {step_name}
# Generated at: {os.environ.get('BUILD_TIME', 'runtime')}

FROM {base_image}

# Set working directory
WORKDIR /app

# Install system dependencies
"""

        if system_packages and 'alpine' not in base_image:
            dockerfile += f"""RUN apt-get update && apt-get install -y --no-install-recommends \\
    {' '.join(system_packages)} \\
    && rm -rf /var/lib/apt/lists/*
"""
        elif system_packages and 'alpine' in base_image:
            # Alpine uses apk
            alpine_packages = [p.replace('build-essential', 'build-base') for p in system_packages]
            dockerfile += f"""RUN apk add --no-cache \\
    {' '.join(alpine_packages)}
"""

        # Install Python dependencies
        if requirements:
            dockerfile += f"""
# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
"""

        # Add the step code
        dockerfile += f"""
# Copy step code
COPY step_function.py .
COPY ops0_runtime.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OPS0_CONTAINER=1
"""

        if needs_gpu:
            dockerfile += """ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
"""

        # Set entrypoint
        dockerfile += """
# Run the step
ENTRYPOINT ["python", "ops0_runtime.py"]
"""

        return dockerfile

    def build_container(self, spec: ContainerSpec, push: bool = False) -> str:
        """Build container from specification"""
        print(f"\n🐳 Building container: {spec.image_tag}")

        # Create temporary build directory
        with tempfile.TemporaryDirectory() as build_dir:
            build_path = Path(build_dir)

            # Write Dockerfile
            dockerfile_path = build_path / "Dockerfile"
            with open(dockerfile_path, 'w') as f:
                f.write(spec.dockerfile_content)

            # Write requirements.txt
            if spec.requirements:
                req_path = build_path / "requirements.txt"
                with open(req_path, 'w') as f:
                    f.write('\n'.join(spec.requirements))

            # Write step function (placeholder)
            step_code = self._generate_step_code(spec)
            with open(build_path / "step_function.py", 'w') as f:
                f.write(step_code)

            # Write runtime wrapper
            runtime_code = self._generate_runtime_wrapper(spec)
            with open(build_path / "ops0_runtime.py", 'w') as f:
                f.write(runtime_code)

            # Build container
            build_cmd = [
                "docker", "build",
                "-t", spec.image_tag,
                "--platform", "linux/amd64",
                "."
            ]

            print(f"  🔨 Running: {' '.join(build_cmd)}")

            try:
                result = subprocess.run(
                    build_cmd,
                    cwd=build_path,
                    capture_output=True,
                    text=True,
                    check=True
                )

                print(f"  ✅ Build successful!")
                self.built_images.add(spec.image_tag)

                if push:
                    self._push_container(spec.image_tag)

                return spec.image_tag

            except subprocess.CalledProcessError as e:
                print(f"  ❌ Build failed: {e.stderr}")
                raise

    def _generate_step_code(self, spec: ContainerSpec) -> str:
        """Generate the step function code for the container"""
        return f"""
# Auto-generated step code for: {spec.step_name}
import sys
import os

# Add ops0 to path
sys.path.insert(0, '/app')

def execute_step():
    '''Execute the ops0 step in container environment'''
    # This would contain the actual step function code
    # In production, this would be extracted from the step metadata
    print(f"Executing step: {spec.step_name}")
    print(f"Memory limit: {spec.memory_limit}")
    print(f"CPU limit: {spec.cpu_limit}")

    # Step execution logic would go here
    return {{"status": "completed", "step": "{spec.step_name}"}}

if __name__ == "__main__":
    result = execute_step()
    print(f"Step result: {{result}}")
"""

    def _generate_runtime_wrapper(self, spec: ContainerSpec) -> str:
        """Generate the runtime wrapper for the container"""
        return f"""
# ops0 Container Runtime Wrapper
import os
import json
import traceback
from step_function import execute_step

def main():
    '''Main container entry point'''
    try:
        print(f"🚀 Starting ops0 step: {spec.step_name}")
        print(f"📊 Environment: {{os.environ.get('OPS0_RUNTIME_MODE', 'unknown')}}")

        # Execute the step
        result = execute_step()

        # Write result to output file for ops0 orchestrator
        output_path = "/tmp/ops0_step_output.json"
        with open(output_path, 'w') as f:
            json.dump(result, f)

        print(f"✅ Step completed successfully")
        exit(0)

    except Exception as e:
        error_info = {{
            "error": str(e),
            "traceback": traceback.format_exc(),
            "step": "{spec.step_name}"
        }}

        # Write error to output file
        error_path = "/tmp/ops0_step_error.json"
        with open(error_path, 'w') as f:
            json.dump(error_info, f)

        print(f"❌ Step failed: {{str(e)}}")
        exit(1)

if __name__ == "__main__":
    main()
"""

    def _push_container(self, image_tag: str):
        """Push container to registry"""
        print(f"  📤 Pushing to registry: {image_tag}")

        push_cmd = ["docker", "push", image_tag]

        try:
            subprocess.run(push_cmd, check=True, capture_output=True)
            print(f"  ✅ Push successful!")
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️  Push failed (continuing anyway): {e}")


class ContainerOrchestrator:
    """
    Orchestrates container building for entire pipelines.

    Manages the containerization process across all steps.
    """

    def __init__(self):
        self.builder = ContainerBuilder()
        self.built_containers: Dict[str, ContainerSpec] = {}

    def containerize_pipeline(self, pipeline_graph) -> Dict[str, ContainerSpec]:
        """
        Containerize all steps in a pipeline.

        Args:
            pipeline_graph: The pipeline to containerize

        Returns:
            Dict mapping step names to container specs
        """
        print(f"\n🐳 Containerizing pipeline: {pipeline_graph.name}")
        print("=" * 50)

        container_specs = {}

        for step_name, step_node in pipeline_graph.steps.items():
            print(f"\n📦 Processing step: {step_name}")

            # Create container spec
            spec = self.builder.containerize_step(step_node.metadata)
            container_specs[step_name] = spec

            # Build container (in production, this would be optimized with caching)
            if os.getenv("OPS0_BUILD_CONTAINERS", "false").lower() == "true":
                self.builder.build_container(spec, push=True)
            else:
                print(f"  🚧 Container build skipped (set OPS0_BUILD_CONTAINERS=true to build)")

        self.built_containers.update(container_specs)

        print(f"\n✅ Pipeline containerization complete!")
        print(f"📊 Containerized {len(container_specs)} steps")

        return container_specs

    def get_container_manifest(self) -> Dict[str, Any]:
        """Generate deployment manifest for all containers"""
        manifest = {
            "version": "ops0/v1",
            "kind": "PipelineManifest",
            "metadata": {
                "generated_by": "ops0-containerizer",
                "timestamp": "2025-01-01T00:00:00Z"
            },
            "spec": {
                "containers": {}
            }
        }

        for step_name, spec in self.built_containers.items():
            manifest["spec"]["containers"][step_name] = {
                "image": spec.image_tag,
                "resources": {
                    "memory": spec.memory_limit,
                    "cpu": spec.cpu_limit,
                    "gpu": 1 if spec.needs_gpu else 0
                },
                "requirements": spec.requirements,
                "system_packages": spec.system_packages
            }

        return manifest


# Global container orchestrator instance
container_orchestrator = ContainerOrchestrator()