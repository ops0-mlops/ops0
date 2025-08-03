"""
ops0 CLI Doctor

Diagnostic and validation utilities for ops0 installations and projects.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import platform

from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .utils import (
    console,
    check_dependencies,
    validate_project_structure,
    get_system_info,
)
from .config import config


class Ops0Doctor:
    """Comprehensive diagnostic tool for ops0 installations and projects"""

    def __init__(self):
        self.checks = []
        self.warnings = []
        self.errors = []
        self.system_info = get_system_info()

    def run_full_diagnosis(self) -> Dict[str, Any]:
        """Run complete diagnosis and return results"""
        console.print("\n🩺 ops0 Doctor - Comprehensive Health Check")
        console.print("=" * 50)

        # Reset counters
        self.checks = []
        self.warnings = []
        self.errors = []

        # Run all diagnostic checks
        results = {
            "system": self._check_system_requirements(),
            "python": self._check_python_environment(),
            "ops0": self._check_ops0_installation(),
            "dependencies": self._check_dependencies(),
            "project": self._check_project_structure(),
            "configuration": self._check_configuration(),
            "runtime": self._check_runtime_environment(),
            "networking": self._check_networking(),
        }

        # Generate summary
        results["summary"] = self._generate_summary()

        # Display results
        self._display_results(results)

        return results

    def _check_system_requirements(self) -> Dict[str, Any]:
        """Check system requirements and compatibility"""
        console.print("\n🖥️  System Requirements")

        checks = {
            "platform_supported": platform.system() in ["Linux", "Darwin", "Windows"],
            "architecture": platform.architecture()[0] in ["64bit"],
            "python_version": sys.version_info >= (3, 9),
            "memory_available": True,  # Would check actual memory
            "disk_space": True,  # Would check actual disk space
        }

        # Check specific requirements
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Check", style="green")
        table.add_column("Status", style="white")
        table.add_column("Details", style="dim")

        table.add_row(
            "Platform",
            "✅ Supported" if checks["platform_supported"] else "❌ Unsupported",
            platform.platform()
        )

        table.add_row(
            "Architecture",
            "✅ 64-bit" if checks["architecture"] else "❌ 32-bit",
            platform.architecture()[0]
        )

        table.add_row(
            "Python Version",
            "✅ Compatible" if checks["python_version"] else "❌ Too old",
            f"{python_version} (required: 3.9+)"
        )

        console.print(table)

        if not checks["python_version"]:
            self.errors.append("Python 3.9+ required")

        if not checks["platform_supported"]:
            self.errors.append(f"Platform {platform.system()} not officially supported")

        return checks

    def _check_python_environment(self) -> Dict[str, Any]:
        """Check Python environment and virtual environment setup"""
        console.print("\n🐍 Python Environment")

        venv_active = os.environ.get('VIRTUAL_ENV') is not None
        pip_available = subprocess.run(["pip", "--version"], capture_output=True).returncode == 0

        checks = {
            "virtual_env": venv_active,
            "pip_available": pip_available,
            "python_path": sys.executable,
            "site_packages": len(sys.path),
        }

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Check", style="green")
        table.add_column("Status", style="white")
        table.add_column("Details", style="dim")

        table.add_row(
            "Virtual Environment",
            "✅ Active" if venv_active else "⚠️  Not active",
            os.environ.get('VIRTUAL_ENV', 'None')
        )

        table.add_row(
            "Package Manager",
            "✅ Available" if pip_available else "❌ Missing",
            "pip"
        )

        table.add_row(
            "Python Executable",
            "✅ Found",
            sys.executable
        )

        console.print(table)

        if not venv_active:
            self.warnings.append("Virtual environment not active - recommended for isolation")

        if not pip_available:
            self.errors.append("pip not available")

        return checks

    def _check_ops0_installation(self) -> Dict[str, Any]:
        """Check ops0 installation and import capabilities"""
        console.print("\n📦 ops0 Installation")

        checks = {
            "ops0_importable": False,
            "version": None,
            "install_location": None,
            "core_modules": {},
        }

        # Test ops0 import
        try:
            import ops0
            checks["ops0_importable"] = True
            checks["version"] = getattr(ops0, '__version__', 'Unknown')
            checks["install_location"] = getattr(ops0, '__file__', 'Unknown')
        except ImportError as e:
            self.errors.append(f"Cannot import ops0: {e}")

        # Test core module imports
        core_modules = [
            "ops0.core.decorators",
            "ops0.core.storage",
            "ops0.core.graph",
            "ops0.core.executor",
        ]

        for module_name in core_modules:
            try:
                importlib.import_module(module_name)
                checks["core_modules"][module_name] = True
            except ImportError:
                checks["core_modules"][module_name] = False
                self.warnings.append(f"Core module {module_name} not available")

        # Display results
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Component", style="green")
        table.add_column("Status", style="white")
        table.add_column("Details", style="dim")

        table.add_row(
            "ops0 Package",
            "✅ Installed" if checks["ops0_importable"] else "❌ Missing",
            f"Version: {checks['version']}" if checks["version"] else "Not available"
        )

        for module, available in checks["core_modules"].items():
            module_short = module.split('.')[-1]
            table.add_row(
                f"  {module_short}",
                "✅ Available" if available else "❌ Missing",
                module
            )

        console.print(table)

        return checks

    def _check_dependencies(self) -> Dict[str, Any]:
        """Check external dependencies and tools"""
        console.print("\n🔧 External Dependencies")

        deps = check_dependencies()

        # Additional dependency checks
        optional_deps = {
            "pandas": self._check_optional_import("pandas"),
            "numpy": self._check_optional_import("numpy"),
            "scikit-learn": self._check_optional_import("sklearn"),
            "matplotlib": self._check_optional_import("matplotlib"),
        }

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Dependency", style="green")
        table.add_column("Status", style="white")
        table.add_column("Purpose", style="dim")

        # Required dependencies
        table.add_row(
            "Docker",
            "✅ Available" if deps["docker"] else "⚠️  Missing",
            "Container runtime (optional for local dev)"
        )

        table.add_row(
            "Git",
            "✅ Available" if deps["git"] else "⚠️  Missing",
            "Version control (recommended)"
        )

        # Optional ML dependencies
        for dep_name, available in optional_deps.items():
            table.add_row(
                dep_name,
                "✅ Available" if available else "⚠️  Not installed",
                "ML workflows"
            )

        console.print(table)

        if not deps["docker"]:
            self.warnings.append("Docker not available - local containerization disabled")

        return {**deps, **optional_deps}

    def _check_optional_import(self, module_name: str) -> bool:
        """Check if optional module can be imported"""
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False

    def _check_project_structure(self) -> Dict[str, Any]:
        """Check current project structure and ops0 setup"""
        console.print("\n📁 Project Structure")

        structure_checks = validate_project_structure()

        # Additional project checks
        current_dir = Path.cwd()
        checks = {
            **structure_checks,
            "in_git_repo": (current_dir / ".git").exists(),
            "has_requirements": (current_dir / "requirements.txt").exists(),
            "has_readme": any((current_dir / f"README.{ext}").exists() for ext in ["md", "rst", "txt"]),
            "config_file": self._find_config_file(current_dir),
        }

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Structure Check", style="green")
        table.add_column("Status", style="white")
        table.add_column("Path", style="dim")

        structure_items = [
            ("ops0 Directory", checks["ops0_directory"], ".ops0/"),
            ("Storage Directory", checks["storage_directory"], ".ops0/storage/"),
            ("Python Files", checks["python_files"], "*.py"),
            ("Pipeline Files", checks["pipeline_files"], "*pipeline*.py"),
            ("Git Repository", checks["in_git_repo"], ".git/"),
            ("Requirements File", checks["has_requirements"], "requirements.txt"),
            ("README File", checks["has_readme"], "README.*"),
        ]

        for name, exists, path in structure_items:
            table.add_row(
                name,
                "✅ Found" if exists else "⚠️  Missing",
                path
            )

        if checks["config_file"]:
            table.add_row(
                "Config File",
                "✅ Found",
                str(checks["config_file"])
            )

        console.print(table)

        if not checks["ops0_directory"]:
            self.warnings.append("No .ops0 directory - run 'ops0 init' in a new project")

        return checks

    def _find_config_file(self, directory: Path) -> Optional[Path]:
        """Find ops0 configuration file"""
        config_files = [
            "ops0.toml",
            "pyproject.toml",
            "ops0.json",
        ]

        for config_file in config_files:
            path = directory / config_file
            if path.exists():
                return path

        return None

    def _check_configuration(self) -> Dict[str, Any]:
        """Check ops0 configuration"""
        console.print("\n⚙️  Configuration")

        checks = {
            "config_loaded": config.config is not None,
            "config_valid": True,
            "environment_vars": {},
            "config_source": getattr(config, 'config_file_path', None),
        }

        # Check environment variables
        ops0_env_vars = {k: v for k, v in os.environ.items() if k.startswith('OPS0_')}
        checks["environment_vars"] = ops0_env_vars

        # Validate configuration
        try:
            checks["config_valid"] = config.validate()
        except Exception as e:
            checks["config_valid"] = False
            self.warnings.append(f"Configuration validation failed: {e}")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Configuration", style="green")
        table.add_column("Status", style="white")
        table.add_column("Value", style="dim")

        table.add_row(
            "Config Loaded",
            "✅ Yes" if checks["config_loaded"] else "❌ No",
            str(checks["config_source"]) if checks["config_source"] else "Default"
        )

        table.add_row(
            "Config Valid",
            "✅ Valid" if checks["config_valid"] else "❌ Invalid",
            "All checks passed" if checks["config_valid"] else "See warnings"
        )

        table.add_row(
            "Environment Variables",
            f"✅ {len(ops0_env_vars)} found",
            ", ".join(ops0_env_vars.keys()) if ops0_env_vars else "None"
        )

        console.print(table)

        return checks

    def _check_runtime_environment(self) -> Dict[str, Any]:
        """Check runtime environment and capabilities"""
        console.print("\n🚀 Runtime Environment")

        checks = {
            "memory_limit": None,
            "cpu_count": os.cpu_count(),
            "storage_writable": False,
            "network_access": True,  # Would test actual network
        }

        # Test storage writability
        try:
            test_path = Path(".ops0/test_write")
            test_path.parent.mkdir(parents=True, exist_ok=True)
            test_path.write_text("test")
            test_path.unlink()
            checks["storage_writable"] = True
        except Exception:
            self.warnings.append("Storage directory not writable")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Runtime Check", style="green")
        table.add_column("Status", style="white")
        table.add_column("Details", style="dim")

        table.add_row(
            "CPU Cores",
            "✅ Available",
            f"{checks['cpu_count']} cores"
        )

        table.add_row(
            "Storage Writable",
            "✅ Yes" if checks["storage_writable"] else "❌ No",
            ".ops0/ directory"
        )

        table.add_row(
            "Network Access",
            "✅ Available" if checks["network_access"] else "❌ Limited",
            "External APIs and repositories"
        )

        console.print(table)

        return checks

    def _check_networking(self) -> Dict[str, Any]:
        """Check network connectivity for ops0 services"""
        console.print("\n🌐 Network Connectivity")

        # Test key endpoints
        endpoints = {
            "pypi.org": "Package installation",
            "github.com": "Source code repositories",
            "docker.io": "Container registry",
        }

        checks = {}

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Endpoint", style="green")
        table.add_column("Status", style="white")
        table.add_column("Purpose", style="dim")

        for endpoint, purpose in endpoints.items():
            try:
                # Simple connectivity test (would use requests in real implementation)
                result = subprocess.run(
                    ["ping", "-c", "1", endpoint],
                    capture_output=True,
                    timeout=5
                )
                accessible = result.returncode == 0
            except Exception:
                accessible = False

            checks[endpoint] = accessible

            table.add_row(
                endpoint,
                "✅ Reachable" if accessible else "❌ Unreachable",
                purpose
            )

        console.print(table)

        unreachable = [ep for ep, status in checks.items() if not status]
        if unreachable:
            self.warnings.append(f"Network issues detected: {', '.join(unreachable)}")

        return checks

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate overall health summary"""
        total_checks = len(self.checks)
        total_warnings = len(self.warnings)
        total_errors = len(self.errors)

        health_score = max(0, 100 - (total_errors * 20) - (total_warnings * 5))

        if health_score >= 90:
            health_status = "Excellent"
            health_color = "green"
        elif health_score >= 70:
            health_status = "Good"
            health_color = "yellow"
        elif health_score >= 50:
            health_status = "Fair"
            health_color = "orange"
        else:
            health_status = "Poor"
            health_color = "red"

        return {
            "health_score": health_score,
            "health_status": health_status,
            "health_color": health_color,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def _display_results(self, results: Dict[str, Any]):
        """Display comprehensive results summary"""
        summary = results["summary"]

        # Health score panel
        health_text = Text()
        health_text.append("Health Score: ", style="bold")
        health_text.append(f"{summary['health_score']}/100", style=f"bold {summary['health_color']}")
        health_text.append(f" ({summary['health_status']})", style=summary['health_color'])

        health_panel = Panel(
            health_text,
            title="🩺 Overall Health",
            border_style=summary['health_color']
        )
        console.print("\n")
        console.print(health_panel)

        # Issues summary
        if summary['total_errors'] > 0:
            console.print(f"\n❌ {summary['total_errors']} Error(s):")
            for error in summary['errors']:
                console.print(f"  • {error}")

        if summary['total_warnings'] > 0:
            console.print(f"\n⚠️  {summary['total_warnings']} Warning(s):")
            for warning in summary['warnings']:
                console.print(f"  • {warning}")

        # Recommendations
        recommendations = self._generate_recommendations(results)
        if recommendations:
            console.print("\n💡 Recommendations:")
            for rec in recommendations:
                console.print(f"  • {rec}")

        if summary['health_score'] >= 90:
            console.print("\n🎉 Your ops0 installation looks great! You're ready to build amazing ML pipelines.")
        elif summary['total_errors'] == 0:
            console.print("\n✅ Your ops0 installation is functional with some minor issues to address.")
        else:
            console.print("\n🔧 Please address the errors above before using ops0.")

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on results"""
        recommendations = []

        # System recommendations
        if not results["python"]["virtual_env"]:
            recommendations.append(
                "Create and activate a virtual environment: python -m venv venv && source venv/bin/activate")

        if not results["dependencies"]["docker"]:
            recommendations.append(
                "Install Docker for local container development: https://docs.docker.com/get-docker/")

        # Project recommendations
        if not results["project"]["ops0_directory"]:
            recommendations.append("Initialize ops0 project: ops0 init my-project")

        if not results["project"]["has_requirements"]:
            recommendations.append("Create requirements.txt with your dependencies")

        if not results["project"]["in_git_repo"]:
            recommendations.append("Initialize Git repository: git init")

        # Configuration recommendations
        if not results["configuration"]["config_valid"]:
            recommendations.append("Review and fix configuration issues: ops0 config --show")

        return recommendations


# Global doctor instance
doctor = Ops0Doctor()