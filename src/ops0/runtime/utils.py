"""
ops0 Runtime Utilities

Helper functions and utilities for the runtime system.
"""

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

logger = logging.getLogger(__name__)


class RuntimeEnvironment:
    """Detects and manages the runtime environment"""

    @staticmethod
    def detect_environment() -> Dict[str, Any]:
        """Detect current runtime environment"""
        env = {
            "platform": platform.system().lower(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "architecture": platform.machine(),
            "in_docker": os.path.exists("/.dockerenv"),
            "in_kubernetes": os.path.exists("/var/run/secrets/kubernetes.io"),
            "in_lambda": bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME")),
            "cpu_count": os.cpu_count() or 1,
        }

        # Detect cloud provider
        if os.environ.get("AWS_EXECUTION_ENV"):
            env["cloud_provider"] = "aws"
        elif os.environ.get("GOOGLE_CLOUD_PROJECT"):
            env["cloud_provider"] = "gcp"
        elif os.environ.get("AZURE_SUBSCRIPTION_ID"):
            env["cloud_provider"] = "azure"
        else:
            env["cloud_provider"] = "local"

        return env

    @staticmethod
    def get_resource_limits() -> Dict[str, Any]:
        """Get current resource limits"""
        limits = {
            "memory_bytes": None,
            "cpu_quota": None,
            "gpu_available": False
        }

        # Try to get memory limit
        try:
            # Check cgroup v1
            cgroup_mem = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
            if cgroup_mem.exists():
                limit = int(cgroup_mem.read_text().strip())
                if limit < sys.maxsize:  # Not unlimited
                    limits["memory_bytes"] = limit

            # Check cgroup v2
            cgroup_v2 = Path("/sys/fs/cgroup/memory.max")
            if cgroup_v2.exists():
                limit_str = cgroup_v2.read_text().strip()
                if limit_str != "max":
                    limits["memory_bytes"] = int(limit_str)

        except Exception as e:
            logger.debug(f"Could not read memory limits: {e}")

        # Check for GPU
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                limits["gpu_available"] = True
                limits["gpu_devices"] = result.stdout.strip().split('\n')
        except Exception:
            pass

        return limits


class ContainerUtils:
    """Utilities for container operations"""

    @staticmethod
    def generate_image_tag(pipeline_name: str, step_name: str, version: Optional[str] = None) -> str:
        """Generate consistent image tag for a step"""
        # Clean names for Docker compatibility
        clean_pipeline = pipeline_name.lower().replace("_", "-").replace(" ", "-")
        clean_step = step_name.lower().replace("_", "-").replace(" ", "-")

        if version:
            return f"ops0/{clean_pipeline}/{clean_step}:{version}"
        else:
            # Use content hash as version
            content_hash = hashlib.sha256(f"{pipeline_name}-{step_name}".encode()).hexdigest()[:8]
            return f"ops0/{clean_pipeline}/{clean_step}:{content_hash}"

    @staticmethod
    def optimize_dockerfile(base_dockerfile: str, step_metadata: Dict[str, Any]) -> str:
        """Optimize Dockerfile based on step requirements"""
        lines = base_dockerfile.split('\n')
        optimized_lines = []

        # Add optimization comments
        optimized_lines.append("# ops0 optimized Dockerfile")
        optimized_lines.append(f"# Step: {step_metadata.get('name', 'unknown')}")
        optimized_lines.append("")

        # Process each line
        for line in lines:
            # Skip empty lines at the end
            if not line.strip() and optimized_lines and not optimized_lines[-1].strip():
                continue

            optimized_lines.append(line)

            # Add caching optimizations after FROM
            if line.startswith("FROM"):
                optimized_lines.extend([
                    "",
                    "# Enable BuildKit cache mount",
                    "# syntax=docker/dockerfile:1.4",
                    "",
                    "# Install system dependencies with cache",
                    "RUN --mount=type=cache,target=/var/cache/apt \\",
                    "    --mount=type=cache,target=/var/lib/apt \\",
                    "    apt-get update && apt-get install -y --no-install-recommends \\",
                    "    build-essential \\",
                    "    && rm -rf /var/lib/apt/lists/*",
                    ""
                ])

        return '\n'.join(optimized_lines)

    @staticmethod
    def get_base_image(requirements: List[str]) -> str:
        """Select optimal base image based on requirements"""
        # Check for specific framework requirements
        has_torch = any("torch" in req for req in requirements)
        has_tensorflow = any("tensorflow" in req for req in requirements)
        has_gpu = any(gpu_keyword in str(requirements) for gpu_keyword in ["cuda", "gpu"])

        if has_torch and has_gpu:
            return "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime"
        elif has_tensorflow and has_gpu:
            return "tensorflow/tensorflow:2.13.0-gpu"
        elif has_torch:
            return "pytorch/pytorch:2.0.1-cpu-py3.10"
        elif has_tensorflow:
            return "tensorflow/tensorflow:2.13.0"
        else:
            # Default Python image
            return "python:3.10-slim"


class SecurityUtils:
    """Security utilities for runtime operations"""

    @staticmethod
    def sanitize_environment_vars(env_vars: Dict[str, str]) -> Dict[str, str]:
        """Remove sensitive information from environment variables"""
        sensitive_patterns = [
            "KEY", "SECRET", "PASSWORD", "TOKEN", "CREDENTIAL",
            "PRIVATE", "AUTH", "API_KEY", "ACCESS_KEY"
        ]

        sanitized = {}
        for key, value in env_vars.items():
            # Check if key contains sensitive pattern
            is_sensitive = any(pattern in key.upper() for pattern in sensitive_patterns)

            if is_sensitive:
                # Mask the value
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = value

        return sanitized

    @staticmethod
    def validate_step_code(code: str) -> Tuple[bool, Optional[str]]:
        """Basic security validation of step code"""
        # Check for dangerous imports
        dangerous_imports = [
            "subprocess", "os.system", "eval", "exec", "__import__",
            "compile", "open", "file"
        ]

        for dangerous in dangerous_imports:
            if dangerous in code:
                return False, f"Potentially dangerous operation found: {dangerous}"

        # Check for suspicious patterns
        suspicious_patterns = [
            "http://", "https://",  # External URLs in code
            "socket.",  # Network operations
            "/etc/",  # System file access
            "chmod",  # Permission changes
        ]

        for pattern in suspicious_patterns:
            if pattern in code:
                logger.warning(f"Suspicious pattern found in code: {pattern}")

        return True, None


class MetricsFormatter:
    """Format metrics for display and export"""

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    @staticmethod
    def format_bytes(bytes_value: int) -> str:
        """Format bytes in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f}{unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f}PB"

    @staticmethod
    def format_percentage(value: float, decimal_places: int = 1) -> str:
        """Format percentage value"""
        return f"{value * 100:.{decimal_places}f}%"

    @staticmethod
    def export_metrics_json(metrics: Dict[str, Any], filepath: Union[str, Path]) -> None:
        """Export metrics to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

    @staticmethod
    def export_metrics_csv(metrics: List[Dict[str, Any]], filepath: Union[str, Path]) -> None:
        """Export metrics to CSV file"""
        if not metrics:
            return

        import csv

        # Get all unique keys
        keys = set()
        for metric in metrics:
            keys.update(metric.keys())

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(keys))
            writer.writeheader()
            writer.writerows(metrics)


class NetworkUtils:
    """Network utilities for distributed execution"""

    @staticmethod
    def get_available_port(start_port: int = 8000, max_attempts: int = 100) -> Optional[int]:
        """Find an available port starting from start_port"""
        import socket

        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue

        return None

    @staticmethod
    def wait_for_service(host: str, port: int, timeout: int = 30) -> bool:
        """Wait for a service to become available"""
        import socket
        import time

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    s.connect((host, port))
                    return True
            except (socket.timeout, ConnectionRefusedError):
                time.sleep(0.5)

        return False


class CacheManager:
    """Manage caching for pipeline execution"""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".ops0" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(self, step_name: str, args: List[Any], kwargs: Dict[str, Any]) -> str:
        """Generate cache key for step execution"""
        # Create a stable hash of the inputs
        cache_data = {
            "step": step_name,
            "args": str(args),
            "kwargs": str(sorted(kwargs.items()))
        }

        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()

    def get(self, cache_key: str) -> Optional[Any]:
        """Get cached result"""
        cache_file = self.cache_dir / f"{cache_key}.cache"

        if cache_file.exists():
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        return None

    def set(self, cache_key: str, value: Any) -> None:
        """Store result in cache"""
        cache_file = self.cache_dir / f"{cache_key}.cache"

        try:
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def clear(self, older_than_days: Optional[int] = None) -> int:
        """Clear cache files"""
        import time

        cleared = 0
        current_time = time.time()

        for cache_file in self.cache_dir.glob("*.cache"):
            if older_than_days:
                # Check file age
                file_age_days = (current_time - cache_file.stat().st_mtime) / 86400
                if file_age_days < older_than_days:
                    continue

            try:
                cache_file.unlink()
                cleared += 1
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")

        return cleared


# Singleton instances
_runtime_env = RuntimeEnvironment()
_cache_manager = None


def get_runtime_environment() -> RuntimeEnvironment:
    """Get the runtime environment instance"""
    return _runtime_env


def get_cache_manager() -> CacheManager:
    """Get or create cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


# Export public API
__all__ = [
    'RuntimeEnvironment',
    'ContainerUtils',
    'SecurityUtils',
    'MetricsFormatter',
    'NetworkUtils',
    'CacheManager',
    'get_runtime_environment',
    'get_cache_manager'
]