"""
Tests for ops0 storage module.

This test suite ensures the storage abstraction works correctly for both
local filesystem and S3 backends, including serialization, metadata handling,
and model management features.
"""
import os
import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from unittest import mock
from unittest.mock import MagicMock, patch, call

import pytest
import cloudpickle

from ops0.storage import (
    StorageBackend,
    LocalStorage,
    S3Storage,
    get_storage,
    set_storage,
    save,
    load,
    save_model,
    load_model,
    save_dataframe,
    load_dataframe,
    list_models,
    delete_model
)


# Fixtures

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def local_storage(temp_dir):
    """Create a LocalStorage instance with a temporary directory."""
    return LocalStorage(base_path=str(temp_dir))


@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client."""
    with patch('ops0.storage.boto3') as mock_boto3:
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        # Mock head_bucket to simulate bucket exists
        mock_client.head_bucket.return_value = True
        yield mock_client


@pytest.fixture
def s3_storage(mock_s3_client):
    """Create an S3Storage instance with mocked boto3."""
    # Mock the HAS_BOTO3 flag
    with patch('ops0.storage.HAS_BOTO3', True):
        storage = S3Storage(bucket="test-bucket", prefix="test/")
        storage.s3 = mock_s3_client
        return storage


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        'numbers': [1, 2, 3, 4, 5],
        'text': 'Hello, ops0!',
        'nested': {
            'key': 'value',
            'list': [1, 2, 3]
        }
    }


@pytest.fixture
def sample_model():
    """Create a mock ML model for testing."""

    class MockModel:
        def __init__(self):
            self.fitted = True
            self.params = {'n_estimators': 100}

        def predict(self, X):
            return [1] * len(X)

        def get_params(self):
            return self.params

    return MockModel()


# LocalStorage Tests

class TestLocalStorage:
    """Test suite for LocalStorage backend."""

    def test_init_creates_directory(self, temp_dir):
        """Test that LocalStorage creates the base directory on init."""
        storage_path = temp_dir / "storage"
        assert not storage_path.exists()

        storage = LocalStorage(base_path=str(storage_path))
        assert storage_path.exists()
        assert storage_path.is_dir()

    def test_save_and_load(self, local_storage, sample_data):
        """Test basic save and load functionality."""
        key = "test_data"

        # Save data
        path = local_storage.save(key, sample_data)
        assert isinstance(path, str)
        assert Path(path).parent == local_storage.base_path

        # Load data
        loaded_data = local_storage.load(key)
        assert loaded_data == sample_data

    def test_save_with_metadata(self, local_storage, sample_data):
        """Test saving data with metadata."""
        key = "test_with_meta"
        metadata = {'version': '1.0', 'author': 'test'}

        local_storage.save(key, sample_data, metadata)

        # Check metadata file exists
        meta_path = local_storage._get_path(key).with_suffix('.meta.json')
        assert meta_path.exists()

        # Load and verify metadata
        with open(meta_path) as f:
            saved_meta = json.load(f)

        assert saved_meta['version'] == '1.0'
        assert saved_meta['author'] == 'test'
        assert 'saved_at' in saved_meta
        assert 'size_bytes' in saved_meta

    def test_key_sanitization(self, local_storage):
        """Test that keys with slashes are properly sanitized."""
        key = "path/to/file"
        data = {"test": "data"}

        local_storage.save(key, data)

        # Check that slashes were replaced
        expected_filename = "path_to_file.pkl"
        assert any(f.name == expected_filename for f in local_storage.base_path.iterdir())

        # Ensure we can still load with original key
        loaded = local_storage.load(key)
        assert loaded == data

    def test_exists(self, local_storage, sample_data):
        """Test checking if a key exists."""
        key = "exists_test"

        assert not local_storage.exists(key)

        local_storage.save(key, sample_data)
        assert local_storage.exists(key)

    def test_delete(self, local_storage, sample_data):
        """Test deleting data."""
        key = "delete_test"

        # Save data with metadata
        local_storage.save(key, sample_data, {'test': 'metadata'})
        assert local_storage.exists(key)

        # Delete
        deleted = local_storage.delete(key)
        assert deleted is True
        assert not local_storage.exists(key)

        # Delete non-existent key
        deleted_again = local_storage.delete(key)
        assert deleted_again is False

    def test_list_keys(self, local_storage):
        """Test listing keys with and without prefix."""
        # Save multiple items
        local_storage.save("item1", {"data": 1})
        local_storage.save("item2", {"data": 2})
        local_storage.save("prefix_item3", {"data": 3})
        local_storage.save("prefix_item4", {"data": 4})

        # List all keys
        all_keys = local_storage.list_keys()
        assert len(all_keys) == 4
        assert "item1" in all_keys
        assert "prefix_item3" in all_keys

        # List with prefix
        prefix_keys = local_storage.list_keys(prefix="prefix_")
        assert len(prefix_keys) == 2
        assert "prefix_item3" in prefix_keys
        assert "prefix_item4" in prefix_keys
        assert "item1" not in prefix_keys

    def test_load_nonexistent_key(self, local_storage):
        """Test loading a key that doesn't exist raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            local_storage.load("nonexistent")

        assert "Key not found" in str(exc_info.value)

    def test_complex_objects(self, local_storage):
        """Test saving and loading complex objects."""
        # Lambda function
        func = lambda x: x * 2
        local_storage.save("lambda", func)
        loaded_func = local_storage.load("lambda")
        assert loaded_func(5) == 10

        # Class instance
        class CustomClass:
            def __init__(self, value):
                self.value = value

            def get_value(self):
                return self.value

        obj = CustomClass(42)
        local_storage.save("object", obj)
        loaded_obj = local_storage.load("object")
        assert loaded_obj.get_value() == 42


# S3Storage Tests

class TestS3Storage:
    """Test suite for S3Storage backend."""

    def test_init_without_boto3(self):
        """Test that S3Storage raises ImportError when boto3 is not available."""
        with patch('ops0.storage.HAS_BOTO3', False):
            with pytest.raises(ImportError) as exc_info:
                S3Storage()

            assert "boto3 is required" in str(exc_info.value)

    def test_init_with_defaults(self, mock_s3_client):
        """Test S3Storage initialization with default values."""
        with patch('ops0.storage.HAS_BOTO3', True):
            storage = S3Storage()
            assert storage.bucket == "ops0-pipelines"
            assert storage.prefix == "storage/"

    def test_init_with_env_vars(self, mock_s3_client):
        """Test S3Storage initialization from environment variables."""
        with patch.dict(os.environ, {
            'OPS0_BUCKET': 'env-bucket',
            'OPS0_PREFIX': 'env-prefix/'
        }):
            with patch('ops0.storage.HAS_BOTO3', True):
                storage = S3Storage()
                assert storage.bucket == "env-bucket"
                assert storage.prefix == "env-prefix/"

    def test_save_and_load(self, s3_storage, sample_data):
        """Test basic S3 save and load functionality."""
        key = "test_data"

        # Configure mock for save
        s3_storage.s3.put_object.return_value = {'ETag': '"abc123"'}

        # Save data
        s3_path = s3_storage.save(key, sample_data, {'custom': 'meta'})
        assert s3_path == "s3://test-bucket/test/test_data"

        # Verify put_object was called correctly
        put_call = s3_storage.s3.put_object.call_args
        assert put_call.kwargs['Bucket'] == 'test-bucket'
        assert put_call.kwargs['Key'] == 'test/test_data'
        assert cloudpickle.loads(put_call.kwargs['Body']) == sample_data

        # Check metadata
        metadata = put_call.kwargs['Metadata']
        assert 'saved_at' in metadata
        assert metadata['ops0_version'] == '0.1.0'
        assert metadata['custom'] == 'meta'

        # Configure mock for load
        s3_storage.s3.get_object.return_value = {
            'Body': MagicMock(read=lambda: cloudpickle.dumps(sample_data))
        }

        # Load data
        loaded_data = s3_storage.load(key)
        assert loaded_data == sample_data

        # Verify get_object was called correctly
        s3_storage.s3.get_object.assert_called_with(
            Bucket='test-bucket',
            Key='test/test_data'
        )

    def test_exists(self, s3_storage):
        """Test checking if a key exists in S3."""
        key = "exists_test"

        # Key exists
        s3_storage.s3.head_object.return_value = {'ContentLength': 1234}
        assert s3_storage.exists(key) is True
        s3_storage.s3.head_object.assert_called_with(
            Bucket='test-bucket',
            Key='test/exists_test'
        )

        # Key doesn't exist
        s3_storage.s3.head_object.side_effect = s3_storage.s3.exceptions.NoSuchKey({}, 'operation')
        assert s3_storage.exists("nonexistent") is False

    def test_delete(self, s3_storage):
        """Test deleting objects from S3."""
        key = "delete_test"

        # Successful delete
        s3_storage.s3.delete_object.return_value = {'DeleteMarker': True}
        assert s3_storage.delete(key) is True
        s3_storage.s3.delete_object.assert_called_with(
            Bucket='test-bucket',
            Key='test/delete_test'
        )

        # Failed delete
        s3_storage.s3.delete_object.side_effect = Exception("Delete failed")
        assert s3_storage.delete("fail_key") is False

    def test_list_keys(self, s3_storage):
        """Test listing keys from S3."""
        # Mock paginator
        mock_paginator = MagicMock()
        s3_storage.s3.get_paginator.return_value = mock_paginator

        # Mock pages
        pages = [
            {'Contents': [
                {'Key': 'test/item1'},
                {'Key': 'test/item2'},
                {'Key': 'test/prefix_item3'}
            ]},
            {'Contents': [
                {'Key': 'test/prefix_item4'}
            ]}
        ]
        mock_paginator.paginate.return_value = pages

        # List all keys
        keys = s3_storage.list_keys()
        assert keys == ['item1', 'item2', 'prefix_item3', 'prefix_item4']

        mock_paginator.paginate.assert_called_with(
            Bucket='test-bucket',
            Prefix='test/'
        )

        # List with prefix
        mock_paginator.paginate.reset_mock()
        keys = s3_storage.list_keys(prefix='prefix_')

        mock_paginator.paginate.assert_called_with(
            Bucket='test-bucket',
            Prefix='test/prefix_'
        )

    def test_load_nonexistent_key(self, s3_storage):
        """Test loading a key that doesn't exist raises KeyError."""
        s3_storage.s3.get_object.side_effect = s3_storage.s3.exceptions.NoSuchKey({}, 'operation')

        with pytest.raises(KeyError) as exc_info:
            s3_storage.load("nonexistent")

        assert "Key not found in S3" in str(exc_info.value)


# Global Storage Functions Tests

class TestGlobalStorageFunctions:
    """Test suite for global storage functions."""

    def test_get_storage_default(self):
        """Test get_storage returns LocalStorage by default."""
        # Reset global storage
        import ops0.storage
        ops0.storage._storage = None

        storage = get_storage()
        assert isinstance(storage, LocalStorage)

    def test_get_storage_s3_from_env(self, mock_s3_client):
        """Test get_storage returns S3Storage when env var is set."""
        import ops0.storage
        ops0.storage._storage = None

        with patch.dict(os.environ, {'OPS0_STORAGE': 's3'}):
            with patch('ops0.storage.HAS_BOTO3', True):
                storage = get_storage()
                assert isinstance(storage, S3Storage)

    def test_set_storage(self, local_storage):
        """Test setting a custom storage backend."""
        import ops0.storage

        set_storage(local_storage)
        assert ops0.storage._storage is local_storage
        assert get_storage() is local_storage

    def test_convenience_functions(self, temp_dir):
        """Test convenience save/load functions."""
        # Set up local storage
        storage = LocalStorage(str(temp_dir))
        set_storage(storage)

        # Test save/load
        data = {"test": "data"}
        path = save("test_key", data, {"meta": "data"})
        assert isinstance(path, str)

        loaded = load("test_key")
        assert loaded == data


# Model Management Tests

class TestModelManagement:
    """Test suite for model management functions."""

    def test_save_and_load_model(self, temp_dir, sample_model):
        """Test saving and loading ML models."""
        storage = LocalStorage(str(temp_dir))
        set_storage(storage)

        # Save model
        path = save_model(sample_model, "test_model", {"accuracy": 0.95})
        assert "models/test_model" in path

        # Load model
        loaded_model = load_model("test_model")
        assert loaded_model.fitted is True
        assert loaded_model.predict([1, 2, 3]) == [1, 1, 1]

    def test_save_model_metadata(self, temp_dir, sample_model):
        """Test that model metadata is properly saved."""
        storage = LocalStorage(str(temp_dir))
        set_storage(storage)

        save_model(sample_model, "meta_test", {"custom": "value"})

        # Check metadata file
        meta_path = storage.base_path / "models_meta_test.meta.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            metadata = json.load(f)

        assert metadata['model_type'] == 'MockModel'
        assert metadata['custom'] == 'value'
        assert metadata['params'] == {'n_estimators': 100}

    def test_list_models(self, temp_dir):
        """Test listing saved models."""
        storage = LocalStorage(str(temp_dir))
        set_storage(storage)

        # Save multiple models
        save_model(sample_model(), "model1")
        save_model(sample_model(), "model2")
        save_model(sample_model(), "model3")

        # List models
        models = list_models()
        assert len(models) == 3
        assert "model1" in models
        assert "model2" in models
        assert "model3" in models

    def test_delete_model(self, temp_dir):
        """Test deleting models."""
        storage = LocalStorage(str(temp_dir))
        set_storage(storage)

        # Save and delete model
        save_model(sample_model(), "delete_me")
        assert "delete_me" in list_models()

        deleted = delete_model("delete_me")
        assert deleted is True
        assert "delete_me" not in list_models()

    def test_pytorch_style_model(self, temp_dir):
        """Test saving model with PyTorch-style state_dict."""
        storage = LocalStorage(str(temp_dir))
        set_storage(storage)

        class MockTorchModel:
            def state_dict(self):
                return {'weights': [1, 2, 3]}

            def __str__(self):
                return "MockTorchModel(layers=3)"

        model = MockTorchModel()
        save_model(model, "torch_model")

        # Check metadata includes architecture
        meta_path = storage.base_path / "models_torch_model.meta.json"
        with open(meta_path) as f:
            metadata = json.load(f)

        assert metadata['architecture'] == "MockTorchModel(layers=3)"


# DataFrame Functions Tests

class TestDataFrameFunctions:
    """Test suite for DataFrame-specific functions."""

    def test_save_and_load_dataframe_without_pandas(self, temp_dir):
        """Test DataFrame functions fall back to regular save when pandas not available."""
        storage = LocalStorage(str(temp_dir))
        set_storage(storage)

        with patch('ops0.storage.HAS_PANDAS', False):
            data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}

            path = save_dataframe(data, "test_df")
            assert "dataframes/test_df" in path

            loaded = load_dataframe("test_df")
            assert loaded == data

    def test_save_dataframe_with_format(self, temp_dir):
        """Test saving DataFrame with format specification."""
        storage = LocalStorage(str(temp_dir))
        set_storage(storage)

        # Mock DataFrame
        df = MagicMock()

        # Test parquet format
        path = save_dataframe(df, "test_parquet", format="parquet")
        assert "dataframes/test_parquet" in path

        # Check metadata
        meta_path = storage.base_path / "dataframes_test_parquet.meta.json"
        with open(meta_path) as f:
            metadata = json.load(f)
        assert metadata['format'] == 'parquet'

        # Test default format
        path = save_dataframe(df, "test_pickle", format="pickle")
        meta_path = storage.base_path / "dataframes_test_pickle.meta.json"
        with open(meta_path) as f:
            metadata = json.load(f)
        assert metadata['format'] == 'pickle'


# Integration Tests

class TestStorageIntegration:
    """Integration tests for storage module."""

    def test_switching_backends(self, temp_dir, mock_s3_client):
        """Test switching between storage backends."""
        # Start with local storage
        local = LocalStorage(str(temp_dir))
        set_storage(local)

        save("local_data", {"source": "local"})
        assert load("local_data") == {"source": "local"}

        # Switch to S3
        with patch('ops0.storage.HAS_BOTO3', True):
            s3 = S3Storage()
            s3.s3 = mock_s3_client
            set_storage(s3)

            # Configure mock
            s3.s3.get_object.side_effect = Exception("Not found in S3")

            # Try to load local data from S3 (should fail)
            with pytest.raises(Exception):
                load("local_data")

    def test_concurrent_access(self, temp_dir):
        """Test that storage handles concurrent access gracefully."""
        import threading
        import time

        storage = LocalStorage(str(temp_dir))
        set_storage(storage)

        results = []
        errors = []

        def worker(worker_id):
            try:
                # Each worker saves and loads data
                data = {"worker": worker_id, "timestamp": time.time()}
                save(f"worker_{worker_id}", data)

                # Small delay to increase chance of concurrency
                time.sleep(0.01)

                loaded = load(f"worker_{worker_id}")
                results.append(loaded)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Verify results
        assert len(errors) == 0
        assert len(results) == 10

        # Check each worker's data
        for i in range(10):
            worker_data = [r for r in results if r['worker'] == i][0]
            assert worker_data['worker'] == i

    def test_large_data_handling(self, temp_dir):
        """Test handling of large data objects."""
        storage = LocalStorage(str(temp_dir))
        set_storage(storage)

        # Create large data (10MB of random data)
        import numpy as np
        large_array = np.random.rand(1000, 1000, 10)  # ~80MB

        # Save and load
        save("large_data", large_array)
        loaded = load("large_data")

        # Verify data integrity
        assert loaded.shape == large_array.shape
        assert np.allclose(loaded, large_array)

    def test_error_recovery(self, temp_dir):
        """Test error handling and recovery."""
        storage = LocalStorage(str(temp_dir))
        set_storage(storage)

        # Test corrupted file
        key = "corrupted"
        pkl_path = storage._get_path(key).with_suffix('.pkl')

        # Write corrupted data
        with open(pkl_path, 'wb') as f:
            f.write(b'corrupted data')

        # Should raise error when loading
        with pytest.raises(Exception):
            load(key)

        # Should be able to overwrite
        save(key, {"fixed": "data"})
        assert load(key) == {"fixed": "data"}


# Edge Cases and Error Handling

class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_key(self, local_storage):
        """Test handling of empty keys."""
        with pytest.raises(Exception):
            local_storage.save("", {"data": "test"})

    def test_none_data(self, local_storage):
        """Test saving None values."""
        local_storage.save("none_test", None)
        assert local_storage.load("none_test") is None

    def test_special_characters_in_key(self, local_storage):
        """Test keys with special characters."""
        special_keys = [
            "key with spaces",
            "key@with#special$chars",
            "key/with/nested/path",
            "key\\with\\backslashes"
        ]

        for key in special_keys:
            data = {"key": key}
            local_storage.save(key, data)
            loaded = local_storage.load(key)
            assert loaded == data

    def test_metadata_without_data(self, local_storage):
        """Test that metadata is handled correctly even without main data."""
        key = "meta_only"
        data = {"test": "data"}

        # Save with metadata
        local_storage.save(key, data, {"important": "metadata"})

        # Delete the data file but keep metadata
        pkl_path = local_storage._get_path(key).with_suffix('.pkl')
        pkl_path.unlink()

        # Should raise error when loading
        with pytest.raises(KeyError):
            local_storage.load(key)

        # But metadata file should still exist
        meta_path = local_storage._get_path(key).with_suffix('.meta.json')
        assert meta_path.exists()


# Performance Tests

class TestPerformance:
    """Basic performance tests."""

    def test_save_load_performance(self, temp_dir, benchmark):
        """Benchmark save/load operations."""
        storage = LocalStorage(str(temp_dir))
        set_storage(storage)

        data = {"numbers": list(range(1000)), "text": "x" * 1000}

        def save_and_load():
            save("perf_test", data)
            return load("perf_test")

        result = benchmark(save_and_load)
        assert result == data

    def test_list_keys_performance(self, temp_dir):
        """Test performance of list_keys with many items."""
        import time

        storage = LocalStorage(str(temp_dir))
        set_storage(storage)

        # Create many keys
        n_keys = 1000
        for i in range(n_keys):
            storage.save(f"key_{i:04d}", {"index": i})

        # Measure list_keys performance
        start = time.time()
        keys = storage.list_keys()
        duration = time.time() - start

        assert len(keys) == n_keys
        assert duration < 1.0  # Should complete in under 1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])