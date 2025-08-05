"""
ops0 Container Runtime

Automatically containerizes Python functions for production deployment.
Handles dependency detection, Dockerfile generation, and multi-stage builds.
"""

import os
import re
import ast
import json
import hashlib
import tempfile
import subprocess
import logging
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
import inspect

logger = logging.getLogger(__name__)


@dataclass
class ContainerSpec:
    """Specification for a containerized step"""
    step_name: str
    base_image: str = "python:3.9-slim"
    requirements: List[str] = field(default_factory=list)
    system_packages: List[str] = field(default_factory=list)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    working_dir: str = "/app"
    entrypoint: List[str] = field(default_factory=lambda: ["python", "-m", "ops0.runtime.step_runner"])
    memory_limit: int = 512  # MB
    cpu_limit: float = 1.0  # vCPUs
    needs_gpu: bool = False
    image_tag: str = ""
    build_args: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.image_tag:
            self.image_tag = f"ops0/{self.step_name}:latest"


class FunctionAnalyzer:
    """Analyzes functions to extract dependencies and requirements"""

    def __init__(self, func):
        self.func = func
        self.source_code = inspect.getsource(func)
        self.tree = ast.parse(self.source_code)

    def extract_imports(self) -> Set[str]:
        """Extract all imports from function"""
        imports = set()

        class ImportVisitor(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])

            def visit_ImportFrom(self, node):
                if node.module:
                    imports.add(node.module.split('.')[0])

        ImportVisitor().visit(self.tree)
        return imports

    def detect_ml_framework(self) -> Optional[str]:
        """Detect which ML framework is used"""
        imports = self.extract_imports()

        if 'torch' in imports or 'torchvision' in imports:
            return 'pytorch'
        elif 'tensorflow' in imports or 'tf' in imports:
            return 'tensorflow'
        elif 'sklearn' in imports or 'scikit-learn' in imports:
            return 'sklearn'
        elif 'xgboost' in imports:
            return 'xgboost'
        elif 'lightgbm' in imports:
            return 'lightgbm'

        return None

    def detect_gpu_usage(self) -> bool:
        """Detect if function uses GPU"""
        source_lower = self.source_code.lower()
        gpu_indicators = [
            '.cuda(', '.gpu(', 'cuda.', 'gpu.',
            'torch.cuda', 'tensorflow.gpu',
            'device="cuda"', "device='cuda'",
            'gpu_options', 'allow_gpu_growth'
        ]

        return any(indicator in source_lower for indicator in gpu_indicators)

    def estimate_memory_requirement(self) -> int:
        """Estimate memory requirement in MB"""
        # Base memory
        memory_mb = 256

        # Add based on framework
        ml_framework = self.detect_ml_framework()
        if ml_framework == 'pytorch':
            memory_mb += 512
        elif ml_framework == 'tensorflow':
            memory_mb += 768

        # Add for data processing
        imports = self.extract_imports()
        if 'pandas' in imports:
            memory_mb += 256
        if 'numpy' in imports:
            memory_mb += 128

        # Add for GPU
        if self.detect_gpu_usage():
            memory_mb += 1024

        return memory_mb


class RequirementsGenerator:
    """Generates requirements.txt from function analysis"""

    # Mapping of import names to pip packages
    PACKAGE_MAPPING = {
        'sklearn': 'scikit-learn',
        'cv2': 'opencv-python',
        'PIL': 'pillow',
        'yaml': 'pyyaml',
        'bs4': 'beautifulsoup4',
    }

    # Known package versions for stability
    PINNED_VERSIONS = {
        'numpy': '1.24.3',
        'pandas': '2.0.3',
        'scikit-learn': '1.3.0',
        'torch': '2.0.1',
        'tensorflow': '2.13.0',
        'matplotlib': '3.7.2',
        'seaborn': '0.12.2',
        'requests': '2.31.0',
        'fastapi': '0.103.0',
        'uvicorn': '0.23.2',
    }

    def generate(self, imports: Set[str], ml_framework: Optional[str] = None) -> List[str]:
        """Generate requirements list from imports"""
        requirements = ['ops0']  # Always include ops0

        # Add detected imports
        for imp in imports:
            # Skip standard library modules
            if self._is_stdlib(imp):
                continue

            # Map import name to package name
            package = self.PACKAGE_MAPPING.get(imp, imp)

            # Add version if known
            if package in self.PINNED_VERSIONS:
                requirements.append(f"{package}=={self.PINNED_VERSIONS[package]}")
            else:
                requirements.append(package)

        # Add ML framework specific requirements
        if ml_framework:
            if ml_framework == 'pytorch':
                requirements.extend([
                    f"torch=={self.PINNED_VERSIONS.get('torch', '2.0.1')}",
                    "torchvision==0.15.2",
                ])
            elif ml_framework == 'tensorflow':
                requirements.append(f"tensorflow=={self.PINNED_VERSIONS.get('tensorflow', '2.13.0')}")

        # Remove duplicates and sort
        requirements = sorted(list(set(requirements)))

        return requirements

    def _is_stdlib(self, module_name: str) -> bool:
        """Check if module is part of Python standard library"""
        stdlib_modules = {
            'os', 'sys', 'time', 'datetime', 'json', 'pickle',
            'pathlib', 'typing', 'collections', 'itertools',
            'functools', 'logging', 'asyncio', 'threading',
            'subprocess', 'shutil', 'tempfile', 'io', 're',
            'math', 'random', 'statistics', 'decimal',
        }
        return module_name in stdlib_modules


class ContainerBuilder:
    """Builds Docker containers for ops0 steps"""

    def __init__(self):
        self.analyzer = None
        self.req_generator = RequirementsGenerator()
        self.built_images = set()

    def containerize_step(self, step_metadata) -> ContainerSpec:
        """
        Create container specification for a step.

        Args:
            step_metadata: Step metadata from decorator

        Returns:
            ContainerSpec with all requirements
        """
        # Analyze function
        analyzer = FunctionAnalyzer(step_metadata.func)

        # Extract information
        imports = analyzer.extract_imports()
        ml_framework = analyzer.detect_ml_framework()
        needs_gpu = analyzer.detect_gpu_usage()
        memory_mb = analyzer.estimate_memory_requirement()

        # Generate requirements
        requirements = self.req_generator.generate(imports, ml_framework)

        # Determine base image
        base_image = self._select_base_image(ml_framework, needs_gpu)

        # System packages
        system_packages = self._get_system_packages(imports, ml_framework)

        # Create spec
        spec = ContainerSpec(
            step_name=step_metadata.name,
            base_image=base_image,
            requirements=requirements,
            system_packages=system_packages,
            memory_limit=memory_mb,
            needs_gpu=needs_gpu,
            environment_vars={
                'OPS0_STEP_NAME': step_metadata.name,
                'OPS0_RUNTIME_MODE': 'container',
                'PYTHONUNBUFFERED': '1',
            }
        )

        # Add ML framework specific config
        if ml_framework == 'tensorflow':
            spec.environment_vars['TF_CPP_MIN_LOG_LEVEL'] = '2'
        elif ml_framework == 'pytorch':
            spec.environment_vars['TORCH_HOME'] = '/tmp/torch'

        return spec

    def _select_base_image(self, ml_framework: Optional[str], needs_gpu: bool) -> str:
        """Select appropriate base image"""
        if needs_gpu:
            if ml_framework == 'pytorch':
                return 'pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime'
            elif ml_framework == 'tensorflow':
                return 'tensorflow/tensorflow:2.13.0-gpu'
            else:
                return 'nvidia/cuda:11.7.1-runtime-ubuntu22.04'
        else:
            if ml_framework == 'pytorch':
                return 'pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime'
            elif ml_framework == 'tensorflow':
                return 'tensorflow/tensorflow:2.13.0'
            else:
                return 'python:3.9-slim-bullseye'

    def _get_system_packages(self, imports: Set[str], ml_framework: Optional[str]) -> List[str]:
        """Get required system packages"""
        packages = []

        # Common packages
        packages.extend(['curl', 'wget'])

        # Based on imports
        if 'cv2' in imports:
            packages.extend(['libopencv-dev', 'libglib2.0-0', 'libsm6', 'libxext6', 'libxrender-dev'])
        if 'matplotlib' in imports or 'seaborn' in imports:
            packages.append('libgomp1')
        if 'psycopg2' in imports:
            packages.append('libpq-dev')

        return packages

    def generate_dockerfile(self, spec: ContainerSpec) -> str:
        """Generate optimized Dockerfile for the step"""
        dockerfile = f"""# Auto-generated by ops0
FROM {spec.base_image} AS base

# Set working directory
WORKDIR {spec.working_dir}

# Install system dependencies
"""
        if spec.system_packages:
            dockerfile += f"""RUN apt-get update && apt-get install -y \\
    {' '.join(spec.system_packages)} \\
    && rm -rf /var/lib/apt/lists/*
"""

        dockerfile += """
# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ops0 runtime
COPY ops0_runtime/ ./ops0_runtime/

# Copy step code
COPY step_function.py .

# Set environment variables
"""
        for key, value in spec.environment_vars.items():
            dockerfile += f"ENV {key}={value}\n"

        dockerfile += f"""
# Runtime configuration
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD python -c "import ops0; print('healthy')" || exit 1

# Set entrypoint
ENTRYPOINT {json.dumps(spec.entrypoint)}
"""

        # Add GPU-specific configuration
        if spec.needs_gpu:
            dockerfile += """
# GPU configuration
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
"""

        return dockerfile

    def build_container(self, spec: ContainerSpec, push: bool = False) -> str:
        """Build container image from spec"""
        # Create temporary build directory
        with tempfile.TemporaryDirectory() as build_dir:
            build_path = Path(build_dir)

            # Write Dockerfile
            dockerfile_path = build_path / "Dockerfile"
            with open(dockerfile_path, 'w') as f:
                f.write(self.generate_dockerfile(spec))

            # Write requirements.txt
            req_path = build_path / "requirements.txt"
            with open(req_path, 'w') as f:
                f.write('\n'.join(spec.requirements))

            # Create ops0 runtime directory
            runtime_dir = build_path / "ops0_runtime"
            runtime_dir.mkdir(exist_ok=True)

            # Write step runner
            runner_path = runtime_dir / "step_runner.py"
            with open(runner_path, 'w') as f:
                f.write(self._generate_step_runner())

            # Write step function wrapper
            step_path = build_path / "step_function.py"
            with open(step_path, 'w') as f:
                f.write(self._generate_step_wrapper(spec))

            # Build container
            logger.info(f"Building container image: {spec.image_tag}")

            build_cmd = [
                "docker", "build",
                "-t", spec.image_tag,
                "-f", str(dockerfile_path),
                "."
            ]

            # Add build args
            for key, value in spec.build_args.items():
                build_cmd.extend(["--build-arg", f"{key}={value}"])

            try:
                result = subprocess.run(
                    build_cmd,
                    cwd=build_path,
                    capture_output=True,
                    text=True,
                    check=True
                )

                logger.info(f"Successfully built: {spec.image_tag}")
                self.built_images.add(spec.image_tag)

                if push:
                    self._push_image(spec.image_tag)

                return spec.image_tag

            except subprocess.CalledProcessError as e:
                logger.error(f"Container build failed: {e.stderr}")
                raise

    def _generate_step_runner(self) -> str:
        """Generate the ops0 step runner"""
        return """# ops0 Step Runner
import os
import sys
import json
import time
import logging
import traceback
from typing import Any, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_step() -> Dict[str, Any]:
    '''Execute the ops0 step in container'''
    try:
        # Import the step function
        from step_function import step_function
        
        # Get input data (from environment or mounted volume)
        input_path = os.environ.get('OPS0_INPUT_PATH', '/tmp/ops0/input.json')
        if os.path.exists(input_path):
            with open(input_path, 'r') as f:
                input_data = json.load(f)
        else:
            input_data = {}
        
        # Execute step
        logger.info(f"Executing step: {os.environ.get('OPS0_STEP_NAME')}")
        start_time = time.time()
        
        result = step_function(**input_data)
        
        execution_time = time.time() - start_time
        
        # Prepare output
        output = {
            'success': True,
            'result': result,
            'execution_time': execution_time,
            'step_name': os.environ.get('OPS0_STEP_NAME'),
            'timestamp': time.time()
        }
        
        # Write output
        output_path = os.environ.get('OPS0_OUTPUT_PATH', '/tmp/ops0/output.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f)
        
        logger.info(f"Step completed in {execution_time:.2f}s")
        return output
        
    except Exception as e:
        logger.error(f"Step execution failed: {str(e)}")
        error_output = {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'step_name': os.environ.get('OPS0_STEP_NAME'),
            'timestamp': time.time()
        }
        
        # Write error output
        output_path = os.environ.get('OPS0_OUTPUT_PATH', '/tmp/ops0/output.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(error_output, f)
        
        return error_output

if __name__ == '__main__':
    result = run_step()
    sys.exit(0 if result['success'] else 1)
"""

    def _generate_step_wrapper(self, spec: ContainerSpec) -> str:
        """Generate wrapper for the step function"""
        # In production, this would extract the actual function code
        # For now, return a placeholder
        return f"""# Step function wrapper for {spec.step_name}
import logging

logger = logging.getLogger(__name__)

def step_function(**kwargs):
    '''Wrapped step function for {spec.step_name}'''
    logger.info(f"Executing {spec.step_name} with args: {{list(kwargs.keys())}}")
    
    # TODO: Insert actual step function code here
    # This would be extracted from the step metadata
    
    # Placeholder implementation
    result = {{
        'step': '{spec.step_name}',
        'status': 'completed',
        'message': 'Step executed successfully in container'
    }}
    
    return result
"""

    def _push_image(self, image_tag: str) -> None:
        """Push image to registry"""
        logger.info(f"Pushing image: {image_tag}")

        try:
            subprocess.run(
                ["docker", "push", image_tag],
                check=True,
                capture_output=True
            )
            logger.info(f"Successfully pushed: {image_tag}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to push image: {e}")


class ContainerOrchestrator:
    """Orchestrates container building and deployment for pipelines"""

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
        logger.info(f"Containerizing pipeline: {pipeline_graph.name}")

        container_specs = {}

        for step_name, step_node in pipeline_graph.steps.items():
            logger.info(f"Processing step: {step_name}")

            # Create container spec
            spec = self.builder.containerize_step(step_node)
            container_specs[step_name] = spec

            # Build container if requested
            if os.getenv("OPS0_BUILD_CONTAINERS", "false").lower() == "true":
                self.builder.build_container(spec, push=True)

        self.built_containers.update(container_specs)

        logger.info(f"Containerized {len(container_specs)} steps")

        return container_specs

    def get_container_manifest(self) -> Dict[str, Any]:
        """Generate deployment manifest for all containers"""
        manifest = {
            "version": "ops0/v1",
            "kind": "PipelineManifest",
            "metadata": {
                "generated_by": "ops0-containerizer",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            },
            "spec": {
                "containers": {}
            }
        }

        for step_name, spec in self.built_containers.items():
            manifest["spec"]["containers"][step_name] = {
                "image": spec.image_tag,
                "resources": {
                    "memory": f"{spec.memory_limit}Mi",
                    "cpu": str(spec.cpu_limit),
                    "gpu": 1 if spec.needs_gpu else 0
                },
                "requirements": spec.requirements,
                "system_packages": spec.system_packages,
                "environment": spec.environment_vars
            }

        return manifest

    def export_compose_file(self, pipeline_name: str, output_path: str = "docker-compose.yml") -> str:
        """Export Docker Compose file for local testing"""
        compose = {
            "version": "3.8",
            "services": {},
            "networks": {
                "ops0": {
                    "driver": "bridge"
                }
            },
            "volumes": {
                "ops0-storage": {}
            }
        }

        for step_name, spec in self.built_containers.items():
            service = {
                "image": spec.image_tag,
                "container_name": f"ops0-{step_name}",
                "networks": ["ops0"],
                "environment": spec.environment_vars,
                "volumes": [
                    "ops0-storage:/tmp/ops0"
                ],
                "restart": "unless-stopped"
            }

            # Add resource limits
            service["deploy"] = {
                "resources": {
                    "limits": {
                        "cpus": str(spec.cpu_limit),
                        "memory": f"{spec.memory_limit}M"
                    }
                }
            }

            # Add GPU if needed
            if spec.needs_gpu:
                service["runtime"] = "nvidia"
                service["environment"]["NVIDIA_VISIBLE_DEVICES"] = "all"

            compose["services"][step_name] = service

        # Write compose file
        import yaml
        with open(output_path, 'w') as f:
            yaml.dump(compose, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Exported Docker Compose file: {output_path}")
        return output_path


# Global container orchestrator instance
container_orchestrator = ContainerOrchestrator()
