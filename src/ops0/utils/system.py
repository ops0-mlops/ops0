"""
ops0 System Utilities

System information, resource monitoring, and platform detection.
"""

import os
import sys
import platform
import psutil
import socket
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, NamedTuple
from dataclasses import dataclass
from datetime import datetime
import logging
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class SystemInfo:
    """Complete system information"""
    platform: str
    platform_version: str
    architecture: str
    processor: str
    hostname: str
    python_version: str
    python_implementation: str
    total_memory: int
    cpu_count: int
    cpu_freq: Optional[float]
    boot_time: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'platform': self.platform,
            'platform_version': self.platform_version,
            'architecture': self.architecture,
            'processor': self.processor,
            'hostname': self.hostname,
            'python_version': self.python_version,
            'python_implementation': self.python_implementation,
            'total_memory': self.total_memory,
            'cpu_count': self.cpu_count,
            'cpu_freq': self.cpu_freq,
            'boot_time': self.boot_time.isoformat()
        }


@dataclass
class MemoryUsage:
    """Memory usage information"""
    total: int
    available: int
    used: int
    free: int
    percent: float
    swap_total: int
    swap_used: int
    swap_free: int
    swap_percent: float

    @property
    def used_gb(self) -> float:
        """Used memory in GB"""
        return self.used / (1024 ** 3)

    @property
    def available_gb(self) -> float:
        """Available memory in GB"""
        return self.available / (1024 ** 3)


@dataclass
class CPUUsage:
    """CPU usage information"""
    percent: float
    per_cpu: List[float]
    load_average: Tuple[float, float, float]
    ctx_switches: int
    interrupts: int

    @property
    def is_high_load(self) -> bool:
        """Check if CPU load is high"""
        return self.percent > 80 or self.load_average[0] > os.cpu_count()


@dataclass
class DiskUsage:
    """Disk usage information"""
    path: str
    total: int
    used: int
    free: int
    percent: float

    @property
    def free_gb(self) -> float:
        """Free space in GB"""
        return self.free / (1024 ** 3)

    @property
    def used_gb(self) -> float:
        """Used space in GB"""
        return self.used / (1024 ** 3)


@dataclass
class GPUInfo:
    """GPU information"""
    index: int
    name: str
    driver_version: str
    memory_total: int
    memory_used: int
    memory_free: int
    utilization: int
    temperature: Optional[float]

    @property
    def memory_percent(self) -> float:
        """Memory usage percentage"""
        if self.memory_total == 0:
            return 0.0
        return (self.memory_used / self.memory_total) * 100


@dataclass
class ProcessInfo:
    """Process information"""
    pid: int
    name: str
    status: str
    create_time: datetime
    cpu_percent: float
    memory_percent: float
    memory_rss: int
    memory_vms: int
    num_threads: int
    username: str

    @classmethod
    def from_pid(cls, pid: int) -> Optional['ProcessInfo']:
        """Create ProcessInfo from PID"""
        try:
            process = psutil.Process(pid)
            with process.oneshot():
                return cls(
                    pid=pid,
                    name=process.name(),
                    status=process.status(),
                    create_time=datetime.fromtimestamp(process.create_time()),
                    cpu_percent=process.cpu_percent(),
                    memory_percent=process.memory_percent(),
                    memory_rss=process.memory_info().rss,
                    memory_vms=process.memory_info().vms,
                    num_threads=process.num_threads(),
                    username=process.username()
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None


def get_system_info() -> SystemInfo:
    """
    Get comprehensive system information.

    Returns:
        SystemInfo object

    Example:
        info = get_system_info()
        print(f"Running on {info.platform} with {info.cpu_count} CPUs")
    """
    # CPU frequency
    cpu_freq = None
    try:
        freq = psutil.cpu_freq()
        if freq:
            cpu_freq = freq.current
    except Exception:
        pass

    return SystemInfo(
        platform=platform.system(),
        platform_version=platform.version(),
        architecture=platform.machine(),
        processor=platform.processor() or "Unknown",
        hostname=socket.gethostname(),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        total_memory=psutil.virtual_memory().total,
        cpu_count=psutil.cpu_count() or 1,
        cpu_freq=cpu_freq,
        boot_time=datetime.fromtimestamp(psutil.boot_time())
    )


def get_memory_usage() -> MemoryUsage:
    """
    Get current memory usage.

    Returns:
        MemoryUsage object

    Example:
        mem = get_memory_usage()
        print(f"Memory: {mem.percent}% used ({mem.used_gb:.1f}GB/{mem.available_gb:.1f}GB)")
    """
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()

    return MemoryUsage(
        total=vm.total,
        available=vm.available,
        used=vm.used,
        free=vm.free,
        percent=vm.percent,
        swap_total=swap.total,
        swap_used=swap.used,
        swap_free=swap.free,
        swap_percent=swap.percent
    )


def get_cpu_usage(interval: float = 1.0) -> CPUUsage:
    """
    Get current CPU usage.

    Args:
        interval: Sampling interval for CPU percent

    Returns:
        CPUUsage object

    Example:
        cpu = get_cpu_usage()
        print(f"CPU: {cpu.percent}%, Load: {cpu.load_average}")
    """
    # Get CPU stats
    cpu_percent = psutil.cpu_percent(interval=interval)
    per_cpu = psutil.cpu_percent(interval=0, percpu=True)

    # Load average (Unix only)
    try:
        load_avg = os.getloadavg()
    except AttributeError:
        # Windows doesn't have getloadavg
        load_avg = (0.0, 0.0, 0.0)

    # Get process stats
    try:
        stats = psutil.cpu_stats()
        ctx_switches = stats.ctx_switches
        interrupts = stats.interrupts
    except Exception:
        ctx_switches = 0
        interrupts = 0

    return CPUUsage(
        percent=cpu_percent,
        per_cpu=per_cpu,
        load_average=load_avg,
        ctx_switches=ctx_switches,
        interrupts=interrupts
    )


def get_disk_usage(path: str = "/") -> DiskUsage:
    """
    Get disk usage for path.

    Args:
        path: Path to check (defaults to root)

    Returns:
        DiskUsage object

    Example:
        disk = get_disk_usage("/home")
        print(f"Disk: {disk.percent}% used ({disk.free_gb:.1f}GB free)")
    """
    usage = psutil.disk_usage(path)

    return DiskUsage(
        path=path,
        total=usage.total,
        used=usage.used,
        free=usage.free,
        percent=usage.percent
    )


def get_gpu_info() -> List[GPUInfo]:
    """
    Get GPU information (NVIDIA only currently).

    Returns:
        List of GPUInfo objects

    Example:
        gpus = get_gpu_info()
        for gpu in gpus:
            print(f"GPU {gpu.index}: {gpu.name} - {gpu.memory_percent:.1f}% memory used")
    """
    gpus = []

    try:
        # Try nvidia-smi
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            check=True
        )

        for line in result.stdout.strip().split('\n'):
            if not line:
                continue

            parts = line.split(', ')
            if len(parts) >= 7:
                try:
                    gpu = GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        driver_version=parts[2],
                        memory_total=int(parts[3]) * 1024 * 1024,  # Convert MB to bytes
                        memory_used=int(parts[4]) * 1024 * 1024,
                        memory_free=int(parts[5]) * 1024 * 1024,
                        utilization=int(parts[6]),
                        temperature=float(parts[7]) if len(parts) > 7 and parts[7] != 'N/A' else None
                    )
                    gpus.append(gpu)
                except (ValueError, IndexError):
                    pass

    except (subprocess.CalledProcessError, FileNotFoundError):
        # nvidia-smi not available
        pass

    # TODO: Add support for AMD GPUs (rocm-smi)

    return gpus


def check_port_available(port: int, host: str = '127.0.0.1') -> bool:
    """
    Check if a port is available.

    Args:
        port: Port number
        host: Host address

    Returns:
        True if port is available

    Example:
        if check_port_available(8080):
            start_server(8080)
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except OSError:
        return False


def get_process_info(pid: Optional[int] = None) -> ProcessInfo:
    """
    Get process information.

    Args:
        pid: Process ID (current process if None)

    Returns:
        ProcessInfo object

    Example:
        info = get_process_info()
        print(f"Process {info.name} using {info.memory_percent:.1f}% memory")
    """
    if pid is None:
        pid = os.getpid()

    info = ProcessInfo.from_pid(pid)
    if info is None:
        raise ValueError(f"Cannot access process {pid}")

    return info


class ResourceMonitor:
    """
    Real-time resource monitoring.

    Example:
        monitor = ResourceMonitor()
        monitor.start()

        # Do some work...
        time.sleep(10)

        stats = monitor.get_stats()
        print(f"Peak memory: {stats['memory_peak_percent']:.1f}%")

        monitor.stop()
    """

    def __init__(
            self,
            interval: float = 1.0,
            track_gpu: bool = True
    ):
        self.interval = interval
        self.track_gpu = track_gpu

        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            'samples': 0,
            'cpu_current': 0.0,
            'cpu_average': 0.0,
            'cpu_peak': 0.0,
            'memory_current': 0.0,
            'memory_average': 0.0,
            'memory_peak': 0.0,
            'memory_current_bytes': 0,
            'memory_peak_bytes': 0,
            'disk_read_bytes': 0,
            'disk_write_bytes': 0,
            'network_sent_bytes': 0,
            'network_recv_bytes': 0,
            'gpu_utilization': [],
            'gpu_memory_percent': []
        }

        # Initial values for delta calculations
        self._last_disk_io = None
        self._last_net_io = None

    def start(self):
        """Start monitoring"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="resource-monitor",
            daemon=True
        )
        self._thread.start()

    def stop(self):
        """Stop monitoring"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.interval * 2)

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=self.interval)

                # Memory usage
                mem = psutil.virtual_memory()

                # Disk I/O
                disk_io = psutil.disk_io_counters()

                # Network I/O
                net_io = psutil.net_io_counters()

                # GPU (if available)
                gpu_info = get_gpu_info() if self.track_gpu else []

                # Update statistics
                with self._lock:
                    self._stats['samples'] += 1

                    # CPU
                    self._stats['cpu_current'] = cpu_percent
                    self._stats['cpu_average'] = (
                            (self._stats['cpu_average'] * (self._stats['samples'] - 1) + cpu_percent) /
                            self._stats['samples']
                    )
                    self._stats['cpu_peak'] = max(self._stats['cpu_peak'], cpu_percent)

                    # Memory
                    self._stats['memory_current'] = mem.percent
                    self._stats['memory_current_bytes'] = mem.used
                    self._stats['memory_average'] = (
                            (self._stats['memory_average'] * (self._stats['samples'] - 1) + mem.percent) /
                            self._stats['samples']
                    )
                    self._stats['memory_peak'] = max(self._stats['memory_peak'], mem.percent)
                    self._stats['memory_peak_bytes'] = max(self._stats['memory_peak_bytes'], mem.used)

                    # Disk I/O (deltas)
                    if self._last_disk_io:
                        self._stats['disk_read_bytes'] += disk_io.read_bytes - self._last_disk_io.read_bytes
                        self._stats['disk_write_bytes'] += disk_io.write_bytes - self._last_disk_io.write_bytes
                    self._last_disk_io = disk_io

                    # Network I/O (deltas)
                    if self._last_net_io:
                        self._stats['network_sent_bytes'] += net_io.bytes_sent - self._last_net_io.bytes_sent
                        self._stats['network_recv_bytes'] += net_io.bytes_recv - self._last_net_io.bytes_recv
                    self._last_net_io = net_io

                    # GPU
                    if gpu_info:
                        gpu_utils = [gpu.utilization for gpu in gpu_info]
                        gpu_mems = [gpu.memory_percent for gpu in gpu_info]
                        self._stats['gpu_utilization'] = gpu_utils
                        self._stats['gpu_memory_percent'] = gpu_mems

            except Exception as e:
                logger.error(f"Error in resource monitor: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        with self._lock:
            return self._stats.copy()

    def reset_stats(self):
        """Reset statistics"""
        with self._lock:
            self._stats['samples'] = 0
            self._stats['cpu_average'] = 0.0
            self._stats['memory_average'] = 0.0
            self._stats['disk_read_bytes'] = 0
            self._stats['disk_write_bytes'] = 0
            self._stats['network_sent_bytes'] = 0
            self._stats['network_recv_bytes'] = 0


# Utility functions

def is_docker_container() -> bool:
    """Check if running inside Docker container"""
    # Check for .dockerenv file
    if Path('/.dockerenv').exists():
        return True

    # Check cgroup
    try:
        with open('/proc/self/cgroup', 'r') as f:
            return 'docker' in f.read()
    except Exception:
        return False


def is_kubernetes_pod() -> bool:
    """Check if running inside Kubernetes pod"""
    # Check for Kubernetes environment variables
    k8s_vars = ['KUBERNETES_SERVICE_HOST', 'KUBERNETES_SERVICE_PORT']
    return all(var in os.environ for var in k8s_vars)


def get_environment_type() -> str:
    """
    Detect environment type.

    Returns:
        Environment type: 'docker', 'kubernetes', 'cloud', 'local'
    """
    if is_kubernetes_pod():
        return 'kubernetes'
    elif is_docker_container():
        return 'docker'
    elif any(var in os.environ for var in ['AWS_REGION', 'AWS_LAMBDA_FUNCTION_NAME']):
        return 'aws'
    elif 'GOOGLE_CLOUD_PROJECT' in os.environ:
        return 'gcp'
    elif 'AZURE_SUBSCRIPTION_ID' in os.environ:
        return 'azure'
    else:
        return 'local'


def get_available_resources() -> Dict[str, Any]:
    """
    Get available system resources.

    Returns:
        Dictionary of available resources
    """
    mem = get_memory_usage()
    cpu = get_cpu_usage(interval=0.1)
    disk = get_disk_usage()
    gpus = get_gpu_info()

    return {
        'cpu': {
            'count': psutil.cpu_count(),
            'available_percent': 100 - cpu.percent
        },
        'memory': {
            'available_bytes': mem.available,
            'available_gb': mem.available_gb,
            'available_percent': 100 - mem.percent
        },
        'disk': {
            'free_bytes': disk.free,
            'free_gb': disk.free_gb,
            'free_percent': 100 - disk.percent
        },
        'gpu': [
            {
                'index': gpu.index,
                'name': gpu.name,
                'memory_free_bytes': gpu.memory_free,
                'utilization_free_percent': 100 - gpu.utilization
            }
            for gpu in gpus
        ]
    }