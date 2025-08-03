"""
Tests for ops0 storage layer
"""
import pytest
import tempfile
import shutil
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from src.ops0.core.storage import StorageLayer, LocalStorageBackend
from src.ops0.core.graph import PipelineGraph


class TestLocalStorageBackend:
    """Test the local filesystem storage backend"""

    def setup_method(self):
        """Setup temporary directory for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = LocalStorageBackend(self.temp_dir)

    def teardown_method(self):
        """Cleanup temporary directory after each test"""
        shutil.rmtree(self.temp_dir)

    def test_save_and_load_basic(self):
        """Test basic save and load functionality"""
        test_data = {"key": "value", "number": 42}

        # Save data
        self.backend.save("test_key", test_data, "test_namespace")

        # Load data
        loaded_data = self.backend.load("test_key", "test_namespace")

        assert loaded_data == test_data

    def test_save_different_data_types(self):
        """Test saving different Python data types"""
        test_cases = [
            ("string", "hello world"),
            ("integer", 42),
            ("float", 3.14159),
            ("list", [1, 2, 3, "four"]),
            ("dict", {"nested": {"data": True}}),
            ("tuple", (1, 2, 3)),
            ("boolean", True),
            ("none", None),
        ]

        for key, data in test_cases:
            self.backend.save(key, data)
            loaded = self.backend.load(key)
            assert loaded == data, f"Failed for {key}: {data}"

    def test_save_complex_objects(self):
        """Test saving complex objects like DataFrames"""
        # Test pandas DataFrame
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })

        self.backend.save("dataframe", df)
        loaded_df = self.backend.load("dataframe")

        pd.testing.assert_frame_equal(df, loaded_df)

    def test_namespace_isolation(self):
        """Test that different namespaces are isolated"""
        # Save same key in different namespaces
        self.backend.save("same_key", "data1", "namespace1")
        self.backend.save("same_key", "data2", "namespace2")

        # Load from each namespace
        data1 = self.backend.load("same_key", "namespace1")
        data2 = self.backend.load("same_key", "namespace2")

        assert data1 == "data1"
        assert data2 == "data2"

    def test_exists_functionality(self):
        """Test exists() method"""
        # Key doesn't exist initially
        assert not self.backend.exists("nonexistent_key")

        # Save data
        self.backend.save("existing_key", "some data")

        # Key now exists
        assert self.backend.exists("existing_key")

        # Key doesn't exist in different namespace
        assert not self.backend.exists("existing_key", "different_namespace")

    def test_delete_functionality(self):
        """Test delete() method"""
        # Save data
        self.backend.save("to_delete", "temporary data")
        assert self.backend.exists("to_delete")

        # Delete data
        self.backend.delete("to_delete")
        assert not self.backend.exists("to_delete")

    def test_load_nonexistent_key(self):
        """Test loading non-existent key raises KeyError"""
        with pytest.raises(KeyError) as exc_info:
            self.backend.load("nonexistent_key")

        assert "not found" in str(exc_info.value)

    def test_file_structure(self):
        """Test that files are created in correct structure"""
        self.backend.save("test_key", "test_data", "test_namespace")

        expected_path = Path(self.temp_dir) / "test_namespace" / "test_key.pkl"
        assert expected_path.exists()

        # Verify file content
        with open(expected_path, 'rb') as f:
            loaded_data = pickle.load(f)
        assert loaded_data == "test_data"


class TestStorageLayer:
    """Test the high-level storage layer interface"""

    def setup_method(self):
        """Setup storage layer for each test"""
        self.temp_dir = tempfile.mkdtemp()
        backend = LocalStorageBackend(self.temp_dir)
        self.storage = StorageLayer(backend)

    def teardown_method(self):
        """Cleanup after each test"""
        shutil.rmtree(self.temp_dir)

    def test_save_and_load_with_output(self):
        """Test storage layer with console output"""
        import io
        import sys

        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            # Save data
            self.storage.save("test_key", {"data": "test"})

            # Load data
            loaded = self.storage.load("test_key")

            # Restore stdout
            sys.stdout = sys.__stdout__

            # Check output messages
            output = captured_output.getvalue()
            assert "Saved 'test_key'" in output
            assert "Loaded 'test_key'" in output

            # Check data
            assert loaded == {"data": "test"}

        finally:
            sys.stdout = sys.__stdout__

    def test_automatic_namespace_from_pipeline(self):
        """Test automatic namespace detection from pipeline context"""
        with PipelineGraph("test-pipeline"):
            # Save without explicit namespace
            self.storage.save("auto_namespace_key", "test_data")

            # Load without explicit namespace
            loaded = self.storage.load("auto_namespace_key")

            assert loaded == "test_data"

            # Verify it was saved in pipeline namespace
            assert self.storage.backend.exists("auto_namespace_key", "test-pipeline")

    def test_explicit_namespace_override(self):
        """Test explicit namespace overrides pipeline context"""
        with PipelineGraph("pipeline-namespace"):
            # Save with explicit namespace
            self.storage.save("explicit_key", "test_data", "explicit_namespace")

            # Should be saved in explicit namespace, not pipeline namespace
            assert self.storage.backend.exists("explicit_key", "explicit_namespace")
            assert not self.storage.backend.exists("explicit_key", "pipeline-namespace")

    def test_no_pipeline_context(self):
        """Test storage without pipeline context uses default namespace"""
        # No pipeline context
        self.storage.save("default_key", "test_data")

        # Should use default namespace
        assert self.storage.backend.exists("default_key", "default")

    def test_exists_method(self):
        """Test exists() method on storage layer"""
        # Key doesn't exist
        assert not self.storage.exists("nonexistent")

        # Save data
        self.storage.save("existing", "data")

        # Key now exists
        assert self.storage.exists("existing")


class TestStorageIntegration:
    """Integration tests for storage with other components"""

    def setup_method(self):
        """Setup for integration tests"""
        self.temp_dir = tempfile.mkdtemp()
        backend = LocalStorageBackend(self.temp_dir)
        self.storage = StorageLayer(backend)

    def teardown_method(self):
        """Cleanup after integration tests"""
        shutil.rmtree(self.temp_dir)

    def test_storage_in_pipeline_steps(self):
        """Test storage usage within pipeline steps"""
        from src.ops0.core.decorators import step

        with PipelineGraph("storage-integration"):
            @step
            def producer_step():
                self.storage.save("shared_data", [1, 2, 3, 4, 5])
                return "produced"

            @step
            def consumer_step():
                data = self.storage.load("shared_data")
                processed = [x * 2 for x in data]
                self.storage.save("processed_data", processed)
                return "consumed"

            # Execute steps
            producer_step()
            consumer_step()

            # Verify data flow
            final_data = self.storage.load("processed_data")
            assert final_data == [2, 4, 6, 8, 10]

    def test_storage_with_large_data(self):
        """Test storage with larger datasets"""
        # Create larger dataset
        large_data = {
            'numbers': list(range(10000)),
            'dataframe': pd.DataFrame({
                'col1': np.random.randn(1000),
                'col2': np.random.choice(['A', 'B', 'C'], 1000),
                'col3': np.random.randint(0, 100, 1000)
            })
        }

        # Save and load
        self.storage.save("large_dataset", large_data)
        loaded_data = self.storage.load("large_dataset")

        # Verify integrity
        assert loaded_data['numbers'] == large_data['numbers']
        pd.testing.assert_frame_equal(
            loaded_data['dataframe'],
            large_data['dataframe']
        )

    def test_storage_error_handling(self):
        """Test storage error handling"""
        # Test loading non-existent key
        with pytest.raises(KeyError):
            self.storage.load("does_not_exist")

        # Test invalid namespace
        with pytest.raises(KeyError):
            self.storage.load("any_key", "nonexistent_namespace")

    def test_storage_concurrent_access(self):
        """Test storage with simulated concurrent access"""
        import threading
        import time

        results = []
        errors = []

        def worker(worker_id):
            try:
                # Each worker saves and loads its own data
                data = f"worker_{worker_id}_data"
                key = f"worker_{worker_id}_key"

                self.storage.save(key, data)
                time.sleep(0.01)  # Simulate some processing
                loaded = self.storage.load(key)

                results.append((worker_id, loaded == data))
            except Exception as e:
                errors.append((worker_id, str(e)))

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        assert all(success for _, success in results)


class TestStorageBackendInterface:
    """Test storage backend interface compliance"""

    def test_backend_interface_methods(self):
        """Test that backends implement required interface"""
        backend = LocalStorageBackend()

        # Check required methods exist
        assert hasattr(backend, 'save')
        assert hasattr(backend, 'load')
        assert hasattr(backend, 'exists')
        assert hasattr(backend, 'delete')

        # Check methods are callable
        assert callable(backend.save)
        assert callable(backend.load)
        assert callable(backend.exists)
        assert callable(backend.delete)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])