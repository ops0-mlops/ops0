"""
Pytest configuration and fixtures for ops0 tests
"""
import pytest
import tempfile
import shutil
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

# Import ops0 components
from ops0.core.storage import StorageLayer, LocalStorageBackend
from ops0.core.graph import PipelineGraph


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def local_storage_backend(temp_dir):
    """Create a local storage backend for testing"""
    return LocalStorageBackend(temp_dir)


@pytest.fixture
def storage_layer(local_storage_backend):
    """Create a storage layer instance for testing"""
    return StorageLayer(local_storage_backend)


@pytest.fixture
def mock_storage():
    """Create a mocked storage layer"""
    mock = MagicMock()
    mock.save.return_value = None
    mock.load.return_value = {"mock": "data"}
    mock.exists.return_value = True
    mock.delete.return_value = None
    return mock


@pytest.fixture
def sample_pipeline():
    """Create a sample pipeline for testing"""
    with PipelineGraph("test-pipeline") as pipeline:
        yield pipeline


@pytest.fixture
def sample_data():
    """Sample data for testing"""
    return {
        "numbers": [1, 2, 3, 4, 5],
        "text": "hello world",
        "nested": {
            "key": "value",
            "list": ["a", "b", "c"]
        }
    }


@pytest.fixture
def sample_dataframe():
    """Sample pandas DataFrame for testing"""
    try:
        import pandas as pd
        return pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e'],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
    except ImportError:
        pytest.skip("pandas not available")


@pytest.fixture
def mock_container_builder():
    """Mock container builder for testing"""
    mock = MagicMock()
    mock.containerize_step.return_value = MagicMock()
    mock.build_container.return_value = "test-image:latest"
    return mock


@pytest.fixture
def clean_environment():
    """Clean environment variables for tests"""
    # Store original values
    original_env = {}
    ops0_vars = [key for key in os.environ.keys() if key.startswith('OPS0_')]

    for var in ops0_vars:
        original_env[var] = os.environ[var]
        del os.environ[var]

    # Set test environment
    os.environ['OPS0_ENV'] = 'test'
    os.environ['OPS0_LOG_LEVEL'] = 'ERROR'  # Reduce log noise in tests

    yield

    # Restore original environment
    for var in [key for key in os.environ.keys() if key.startswith('OPS0_')]:
        if var in os.environ:
            del os.environ[var]

    for var, value in original_env.items():
        os.environ[var] = value


@pytest.fixture
def capture_output():
    """Capture stdout and stderr for testing"""
    import io
    from contextlib import redirect_stdout, redirect_stderr

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
        yield stdout_capture, stderr_capture


@pytest.fixture(autouse=True)
def cleanup_pipeline_state():
    """Automatically cleanup pipeline state after each test"""
    yield
    # Clear any existing pipeline context
    if hasattr(PipelineGraph._current_pipeline, 'value'):
        del PipelineGraph._current_pipeline.value


@pytest.fixture
def example_step_function():
    """Example step function for testing"""

    def sample_step(input_data: list) -> dict:
        """Example step that processes input data"""
        processed = [x * 2 for x in input_data]
        return {
            "original_count": len(input_data),
            "processed_data": processed,
            "status": "success"
        }

    return sample_step


@pytest.fixture
def example_ml_pipeline_steps():
    """Example ML pipeline steps for testing"""
    from ops0.core.decorators import step

    @step
    def load_data():
        """Load sample training data"""
        return {
            "features": [[1, 2], [3, 4], [5, 6]],
            "labels": [0, 1, 0]
        }

    @step
    def preprocess_data():
        """Preprocess the data"""
        from ops0.core.storage import storage
        data = storage.load("raw_data")

        # Simple preprocessing
        processed = {
            "features": [[x[0] / 10, x[1] / 10] for x in data["features"]],
            "labels": data["labels"]
        }

        storage.save("processed_data", processed)
        return processed

    @step
    def train_model():
        """Train a simple model"""
        from ops0.core.storage import storage
        data = storage.load("processed_data")

        # Mock model training
        model = {
            "type": "mock_classifier",
            "accuracy": 0.85,
            "feature_count": len(data["features"][0])
        }

        storage.save("trained_model", model)
        return model

    return {
        "load_data": load_data,
        "preprocess_data": preprocess_data,
        "train_model": train_model
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "requires_docker: marks tests that require Docker"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location"""
    for item in items:
        # Mark tests in unit/ directory as unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Mark tests in integration/ directory as integration tests
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark tests in e2e/ directory as slow
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.slow)

        # Mark container tests as requiring docker
        if "container" in str(item.fspath) or "docker" in str(item.fspath):
            item.add_marker(pytest.mark.requires_docker)


# Custom assertions
def assert_pipeline_has_steps(pipeline, expected_steps):
    """Assert that pipeline has the expected steps"""
    actual_steps = set(pipeline.steps.keys())
    expected_steps = set(expected_steps)

    assert actual_steps == expected_steps, \
        f"Pipeline steps mismatch. Expected: {expected_steps}, Got: {actual_steps}"


def assert_storage_contains(storage, key, expected_value, namespace="default"):
    """Assert that storage contains expected value"""
    assert storage.exists(key, namespace), f"Storage key '{key}' not found"
    actual_value = storage.load(key, namespace)
    assert actual_value == expected_value, \
        f"Storage value mismatch. Expected: {expected_value}, Got: {actual_value}"


# Add custom assertions to pytest namespace
def pytest_namespace():
    return {
        'assert_pipeline_has_steps': assert_pipeline_has_steps,
        'assert_storage_contains': assert_storage_contains
    }


# Skip tests that require optional dependencies
def pytest_runtest_setup(item):
    """Skip tests that require missing dependencies"""
    # Skip Docker tests if Docker not available
    if item.get_closest_marker("requires_docker"):
        try:
            import docker
            client = docker.from_env()
            client.ping()
        except Exception:
            pytest.skip("Docker not available")

    # Skip ML tests if ML libraries not available
    if "ml" in str(item.fspath):
        try:
            import pandas
            import numpy
        except ImportError:
            pytest.skip("ML dependencies not available")