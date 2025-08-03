#!/usr/bin/env python3
"""
Script de validation de l'installation ops0

Vérifie que tous les composants sont correctement installés et fonctionnels.
"""

import sys
import os
import subprocess
import tempfile
from pathlib import Path


def check_python_version():
    """Vérifie la version Python"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python 3.9+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} OK")
    return True


def check_ops0_import():
    """Vérifie que ops0 peut être importé"""
    print("\n📦 Checking ops0 import...")
    try:
        import ops0
        print(f"✅ ops0 imported successfully, version: {ops0.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import ops0: {e}")
        return False


def check_core_components():
    """Vérifie les composants core d'ops0"""
    print("\n🔧 Checking core components...")

    components = [
        ("step decorator", "from ops0 import step"),
        ("pipeline context", "from ops0 import pipeline"),
        ("storage layer", "from ops0 import storage"),
        ("executor", "from ops0 import run, deploy"),
    ]

    all_ok = True
    for name, import_stmt in components:
        try:
            exec(import_stmt)
            print(f"✅ {name} OK")
        except ImportError as e:
            print(f"❌ {name} failed: {e}")
            all_ok = False

    return all_ok


def check_dependencies():
    """Vérifie les dépendances principales"""
    print("\n📚 Checking dependencies...")

    required_deps = [
        "click",
        "pydantic",
        "docker",
        "rich",
    ]

    optional_deps = [
        ("pandas", "ML support"),
        ("numpy", "ML support"),
        ("scikit-learn", "ML support"),
        ("pytest", "Development"),
        ("black", "Development"),
    ]

    all_required_ok = True

    # Check required dependencies
    for dep in required_deps:
        try:
            __import__(dep)
            print(f"✅ {dep} (required)")
        except ImportError:
            print(f"❌ {dep} (required) - MISSING")
            all_required_ok = False

    # Check optional dependencies
    for dep, purpose in optional_deps:
        try:
            __import__(dep)
            print(f"✅ {dep} (optional - {purpose})")
        except ImportError:
            print(f"⚠️  {dep} (optional - {purpose}) - not installed")

    return all_required_ok


def test_basic_pipeline():
    """Teste un pipeline simple"""
    print("\n🧪 Testing basic pipeline...")

    try:
        import ops0

        # Test pipeline creation
        executed_steps = []

        with ops0.pipeline("test-validation") as pipeline:
            @ops0.step
            def test_step_1():
                executed_steps.append("step1")
                ops0.storage.save("test_data", [1, 2, 3])
                return "step1_result"

            @ops0.step
            def test_step_2():
                executed_steps.append("step2")
                data = ops0.storage.load("test_data")
                return {"received": data, "processed": [x * 2 for x in data]}

        # Check pipeline structure
        if len(pipeline.steps) != 2:
            print(f"❌ Expected 2 steps, found {len(pipeline.steps)}")
            return False

        print("✅ Pipeline creation OK")
        print(f"✅ Steps registered: {list(pipeline.steps.keys())}")

        # Test execution order
        execution_order = pipeline.build_execution_order()
        print(f"✅ Execution order: {execution_order}")

        return True

    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_storage_layer():
    """Teste la couche de stockage"""
    print("\n💾 Testing storage layer...")

    try:
        import ops0

        # Test basic save/load
        test_data = {"key": "value", "numbers": [1, 2, 3]}

        ops0.storage.save("validation_test", test_data)
        loaded_data = ops0.storage.load("validation_test")

        if loaded_data != test_data:
            print(f"❌ Storage test failed: {loaded_data} != {test_data}")
            return False

        print("✅ Storage save/load OK")

        # Test exists
        if not ops0.storage.exists("validation_test"):
            print("❌ Storage exists() failed")
            return False

        print("✅ Storage exists() OK")
        return True

    except Exception as e:
        print(f"❌ Storage test failed: {e}")
        return False


def test_cli():
    """Teste l'interface CLI"""
    print("\n💻 Testing CLI...")

    try:
        # Test ops0 command availability
        result = subprocess.run(
            ["ops0", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            print(f"❌ CLI test failed: {result.stderr}")
            return False

        if "ops0" not in result.stdout.lower():
            print(f"❌ CLI output unexpected: {result.stdout[:100]}...")
            return False

        print("✅ CLI command OK")
        return True

    except subprocess.TimeoutExpired:
        print("❌ CLI test timed out")
        return False
    except FileNotFoundError:
        print("❌ ops0 command not found in PATH")
        return False
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False


def test_example_pipeline():
    """Teste un pipeline d'exemple complet"""
    print("\n🎯 Testing complete example pipeline...")

    try:
        import ops0

        # Create a temporary file for the example
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            example_code = '''
import ops0

@ops0.step
def generate_data():
    data = [1, 2, 3, 4, 5]
    ops0.storage.save("raw_data", data)
    return len(data)

@ops0.step
def process_data():
    data = ops0.storage.load("raw_data")
    processed = [x * 2 for x in data]
    ops0.storage.save("processed_data", processed)
    return processed

@ops0.step
def summarize():
    processed = ops0.storage.load("processed_data")
    return {
        "count": len(processed),
        "sum": sum(processed),
        "average": sum(processed) / len(processed)
    }

if __name__ == "__main__":
    with ops0.pipeline("validation-example"):
        generate_data()
        process_data()
        summarize()

        # This would execute the pipeline
        print("Example pipeline created successfully!")
'''
            f.write(example_code)
            temp_file = f.name

        # Execute the example
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Cleanup
        os.unlink(temp_file)

        if result.returncode != 0:
            print(f"❌ Example failed: {result.stderr}")
            return False

        if "successfully" not in result.stdout:
            print(f"❌ Example output unexpected: {result.stdout}")
            return False

        print("✅ Complete example pipeline OK")
        return True

    except Exception as e:
        print(f"❌ Example test failed: {e}")
        return False


def check_environment():
    """Vérifie l'environnement de développement"""
    print("\n🌍 Checking development environment...")

    checks = []

    # Virtual environment
    venv = os.environ.get('VIRTUAL_ENV')
    if venv:
        print(f"✅ Virtual environment: {venv}")
        checks.append(True)
    else:
        print("⚠️  No virtual environment detected")
        checks.append(False)

    # Git repository
    if os.path.exists('.git'):
        print("✅ Git repository detected")
        checks.append(True)
    else:
        print("⚠️  Not in a git repository")
        checks.append(False)

    # ops0 directory structure
    expected_dirs = ['src/ops0', 'tests', 'examples']
    for dir_name in expected_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Directory {dir_name} exists")
            checks.append(True)
        else:
            print(f"⚠️  Directory {dir_name} missing")
            checks.append(False)

    return all(checks)


def main():
    """Fonction principale de validation"""
    print("🚀 ops0 Installation Validation")
    print("=" * 40)

    checks = [
        check_python_version(),
        check_ops0_import(),
        check_core_components(),
        check_dependencies(),
        test_storage_layer(),
        test_basic_pipeline(),
        test_cli(),
        test_example_pipeline(),
    ]

    # Environment check (non-critical)
    env_ok = check_environment()

    print("\n" + "=" * 40)
    print("📊 VALIDATION SUMMARY")
    print("=" * 40)

    passed = sum(checks)
    total = len(checks)

    if passed == total:
        print(f"✅ ALL CHECKS PASSED ({passed}/{total})")
        if env_ok:
            print("✅ Development environment looks good!")
        else:
            print("⚠️  Some development environment issues detected")
        print("\n🎉 ops0 is ready to use!")
        print("\nNext steps:")
        print("  • Try: python examples/dev/test_pipeline.py")
        print("  • Read: https://docs.ops0.xyz/quickstart")
        print("  • Create your first pipeline!")
        return 0
    else:
        print(f"❌ SOME CHECKS FAILED ({passed}/{total})")
        print("\n🔧 To fix issues:")
        print("  • Reinstall: pip install -e .")
        print("  • Install deps: pip install -e '.[dev]'")
        print("  • Check PATH for CLI issues")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)