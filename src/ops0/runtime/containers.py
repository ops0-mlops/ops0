import ast
import inspect
import hashlib
import os
import subprocess
import tempfile
import sys
from typing import List, Dict, Set, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ContainerSpec:
    """Specification for a containerized step"""
    step_name: str
    base_image: str
    requirements: List[str]
    system_packages: List[str]
    python_version: str
    needs_gpu: bool
    memory_limit: str
    cpu_limit: str
    dockerfile_content: str
    image_tag: str


class ImportAnalyzer:
    """
    Analyzes Python code to extract all dependencies automatically.

    This is the magic that makes ops0 build containers without requirements.txt!
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
        self.standard_lib = self._get_standard_library_modules()

    def _get_standard_library_modules(self) -> Set[str]:
        """Get list of Python standard library modules"""
        # Simplified list - in production this would be more comprehensive
        return {
            'os', 'sys', 'json', 'pickle', 'time', 'datetime', 'math',
            'random', 'collections', 'itertools', 'functools', 'threading',
            'multiprocessing', 'subprocess', 'pathlib', 'tempfile', 'shutil',
            'urllib', 'http', 're', 'ast', 'inspect', 'hashlib', 'base64'
        }

    def generate_requirements(self, imports: Set[str]) -> List[str]:
        """Convert imports to pip installable packages"""
        requirements = []

        # Package name mappings
        package_mappings = {
            'sklearn': 'scikit-learn>=1.3.0',
            'cv2': 'opencv-python>=4.8.0',
            'PIL': 'Pillow>=10.0.0',
            'yaml': 'PyYAML>=6.0',
            'bs4': 'beautifulsoup4>=4.12.0',
            'requests': 'requests>=2.31.0',
            'numpy': 'numpy>=1.24.0',
            'pandas': 'pandas>=2.0.0',
            'matplotlib': 'matplotlib>=3.7.0',
            'seaborn': 'seaborn>=0.12.0',
            'plotly': 'plotly>=5.15.0',
            'torch': 'torch>=2.0.0',
            'tensorflow': 'tensorflow>=2.13.0',
            'keras': 'keras>=2.13.0',
            'xgboost': 'xgboost>=1.7.0',
            'lightgbm': 'lightgbm>=4.0.0',
            'catboost': 'catboost>=1.2.0',
            'scipy': 'scipy>=1.11.0',
            'statsmodels': 'statsmodels>=0.14.0'
        }

        for import_name in imports:
            if import_name not in self.standard_lib:
                if import_name in package_mappings:
                    requirements.append(package_mappings[import_name])
                else:
                    # Default version constraint
                    requirements.append(f"{import_name}>=0.1.0")

        # Always include ops0
        requirements.append("ops0>=0.1.0")

        return sorted(list(set(requirements)))


class DockerfileBuilder:
    """
    Builds optimized Dockerfiles for each step automatically.

    Creates lightweight, cached, secure containers.
    """

    def __init__(self):
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    def build_dockerfile(self, spec: ContainerSpec) -> str:
        """Generate optimized Dockerfile for the step"""

        # Choose base image based on requirements
        if spec.needs_gpu:
            base_image = f"python:{self.python_version}-slim-gpu"
        elif any('torch' in req or 'tensorflow' in req for req in spec.requirements):
            base_image = f"python:{self.python_version}-slim"
        else:
            base_image = f"python:{self.python_version}-alpine"

        dockerfile = f"""
# Auto-generated Dockerfile for ops0 step: {spec.step_name}
# Generated on: $(date)
# ops0 version: 0.1.0

FROM {base_image}

# Set working directory
WORKDIR /app

# Install system dependencies
{self._generate_system_packages(spec.system_packages)}

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt

# Copy step code
COPY step_function.py .
COPY ops0_runtime.py .

# Set environment variables
ENV PYTHONPATH=/app
ENV OPS0_STEP_NAME={spec.step_name}
ENV OPS0_RUNTIME_MODE=container

# Resource limits
ENV MEMORY_LIMIT={spec.memory_limit}
ENV CPU_LIMIT={spec.cpu_limit}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import ops0; print('healthy')" || exit 1

# Run the step
CMD ["python", "ops0_runtime.py"]
""".strip()

        return dockerfile

    def _generate_system_packages(self, packages: List[str]) -> str:
        """Generate system package installation commands"""
        if not packages:
            return "# No additional system packages needed"

        if 'alpine' in packages:  # Alpine Linux
            return f"RUN apk add --no-cache {' '.join(packages)}"
        else:  # Debian/Ubuntu
            return f"""RUN apt-get update && \\
    apt-get install -y --no-install-recommends {' '.join(packages)} && \\
    apt-get clean && \\
    rm -rf /var/lib/apt/lists/*"""


class ContainerBuilder:
    """
    Main container orchestration engine.

    Transforms ops0 steps into production-ready containers automatically.
    """

    def __init__(self, registry_url: str = "ghcr.io/ops0"):
        self.registry_url = registry_url
        self.requirements_gen = RequirementsGenerator()
        self.dockerfile_builder = DockerfileBuilder()

    def containerize_step(self, step_metadata) -> ContainerSpec:
        """
        Transform a step into a container specification.

        This is where the magic happens - automatic containerization!
        """
        print(f"🐳 Containerizing step: {step_metadata.name}")

        # Analyze the function
        analyzer = ImportAnalyzer(step_metadata.func)
        imports = analyzer.get_imports()
        ml_frameworks = analyzer.get_ml_frameworks()
        needs_gpu = analyzer.needs_gpu()
        memory_limit = analyzer.estimate_memory_needs()

        # Generate requirements
        requirements = self.requirements_gen.generate_requirements(imports)

        # Determine system packages needed
        system_packages = self._determine_system_packages(ml_frameworks)

        # Generate image tag
        source_hash = hashlib.sha256(
            step_metadata.analyzer.source.encode()
        ).hexdigest()[:12]
        image_tag = f"{self.registry_url}/{step_metadata.name}:{source_hash}"

        # Create container spec
        spec = ContainerSpec(
            step_name=step_metadata.name,
            base_image="python:3.11-slim",
            requirements=requirements,
            system_packages=system_packages,
            python_version="3.11",
            needs_gpu=needs_gpu,
            memory_limit=memory_limit,
            cpu_limit="1000m",
            dockerfile_content="",  # Will be generated
            image_tag=image_tag
        )

        # Generate Dockerfile
        spec.dockerfile_content = self.dockerfile_builder.build_dockerfile(spec)

        print(f"  📦 Requirements: {len(requirements)} packages")
        print(f"  🧠 Memory limit: {memory_limit}")
        print(f"  🔧 GPU support: {'Yes' if needs_gpu else 'No'}")
        print(f"  🏷️  Image tag: {image_tag}")

        return spec

    def _determine_system_packages(self, ml_frameworks: Set[str]) -> List[str]:
        """Determine system packages needed based on ML frameworks"""
        packages = []

        if 'opencv-python' in ml_frameworks:
            packages.extend(['libglib2.0-0', 'libsm6', 'libxext6', 'libxrender-dev'])

        if any(fw in ml_frameworks for fw in ['matplotlib', 'seaborn', 'plotly']):
            packages.extend(['libfreetype6-dev', 'libpng-dev'])

        if 'scipy' in ml_frameworks:
            packages.extend(['gfortran', 'libopenblas-dev'])

        return packages

    def build_container(self, spec: ContainerSpec, push: bool = True) -> str:
        """
        Build the Docker container for a step.

        Args:
            spec: Container specification
            push: Whether to push to registry

        Returns:
            Built image tag
        """
        print(f"🔨 Building container: {spec.step_name}")

        with tempfile.TemporaryDirectory() as build_dir:
            build_path = Path(build_dir)

            # Write Dockerfile
            dockerfile_path = build_path / "Dockerfile"
            with open(dockerfile_path, 'w') as f:
                f.write(spec.dockerfile_content)

            # Write requirements.txt
            requirements_path = build_path / "requirements.txt"
            with open(requirements_path, 'w') as f:
                f.write('\n'.join(spec.requirements))

            # Write step function
            step_code = self._generate_step_code(spec)
            step_path = build_path / "step_function.py"
            with open(step_path, 'w') as f:
                f.write(step_code)

            # Write runtime wrapper
            runtime_code = self._generate_runtime_wrapper(spec)
            runtime_path = build_path / "ops0_runtime.py"
            with open(runtime_path, 'w') as f:
                f.write(runtime_code)

            # Build Docker image
            build_cmd = [
                "docker", "build",
                "-t", spec.image_tag,
                "--build-arg", f"PYTHON_VERSION={spec.python_version}",
                str(build_path)
            ]

            print(f"  🔧 Running: {' '.join(build_cmd)}")

            try:
                result = subprocess.run(
                    build_cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(f"  ✅ Build successful!")

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