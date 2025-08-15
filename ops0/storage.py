"""
Storage abstraction for ops0 - handles data serialization and persistence.
Supports local filesystem and S3 with transparent switching.
"""
import os
import json
import pickle
from pathlib import Path
from typing import Any, Optional, Union, BinaryIO
from abc import ABC, abstractmethod
import cloudpickle
import hashlib
from datetime import datetime

# Try importing optional dependencies
try:
    import boto3

    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class StorageBackend(ABC):
    """Abstract base class for storage backends"""

    @abstractmethod
    def save(self, key: str, data: Any, metadata: Optional[dict] = None) -> str:
        """Save data with a given key"""
        pass

    @abstractmethod
    def load(self, key: str) -> Any:
        """Load data by key"""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data by key"""
        pass

    @abstractmethod
    def list_keys(self, prefix: Optional[str] = None) -> list:
        """List all keys with optional prefix filter"""
        pass


class LocalStorage(StorageBackend):
    """Local filesystem storage backend"""

    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path or ".ops0/storage")
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> Path:
        """Get full path for a key"""
        # Sanitize key to be filesystem-safe
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.base_path / safe_key

    def save(self, key: str, data: Any, metadata: Optional[dict] = None) -> str:
        """Save data locally using cloudpickle"""
        path = self._get_path(key)

        # Save data
        with open(f"{path}.pkl", "wb") as f:
            cloudpickle.dump(data, f)

        # Save metadata if provided
        if metadata:
            meta = {
                **metadata,
                'saved_at': datetime.utcnow().isoformat(),
                'size_bytes': path.stat().st_size
            }
            with open(f"{path}.meta.json", "w") as f:
                json.dump(meta, f)

        return str(path)

    def load(self, key: str) -> Any:
        """Load data from local storage"""
        path = self._get_path(key)
        pkl_path = f"{path}.pkl"

        if not os.path.exists(pkl_path):
            raise KeyError(f"Key not found: {key}")

        with open(pkl_path, "rb") as f:
            return cloudpickle.load(f)

    def exists(self, key: str) -> bool:
        """Check if key exists locally"""
        path = self._get_path(key)
        return os.path.exists(f"{path}.pkl")

    def delete(self, key: str) -> bool:
        """Delete local file"""
        path = self._get_path(key)
        pkl_path = f"{path}.pkl"
        meta_path = f"{path}.meta.json"

        deleted = False
        if os.path.exists(pkl_path):
            os.remove(pkl_path)
            deleted = True
        if os.path.exists(meta_path):
            os.remove(meta_path)

        return deleted

    def list_keys(self, prefix: Optional[str] = None) -> list:
        """List all keys in local storage"""
        keys = []
        for file in self.base_path.glob("*.pkl"):
            key = file.stem
            if prefix is None or key.startswith(prefix):
                keys.append(key)
        return sorted(keys)


class S3Storage(StorageBackend):
    """AWS S3 storage backend"""

    def __init__(self, bucket: Optional[str] = None, prefix: Optional[str] = None):
        if not HAS_BOTO3:
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3")

        self.bucket = bucket or os.environ.get("OPS0_BUCKET", "ops0-pipelines")
        self.prefix = prefix or os.environ.get("OPS0_PREFIX", "storage/")
        self.s3 = boto3.client('s3')

        # Ensure bucket exists
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except:
            # In production, this would handle bucket creation properly
            pass

    def _get_key(self, key: str) -> str:
        """Get full S3 key with prefix"""
        return f"{self.prefix}{key}"

    def save(self, key: str, data: Any, metadata: Optional[dict] = None) -> str:
        """Save data to S3"""
        s3_key = self._get_key(key)

        # Serialize data
        serialized = cloudpickle.dumps(data)

        # Prepare metadata
        s3_metadata = {
            'saved_at': datetime.utcnow().isoformat(),
            'ops0_version': '0.1.0'
        }
        if metadata:
            s3_metadata.update({k: str(v) for k, v in metadata.items()})

        # Upload to S3
        self.s3.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=serialized,
            Metadata=s3_metadata
        )

        return f"s3://{self.bucket}/{s3_key}"

    def load(self, key: str) -> Any:
        """Load data from S3"""
        s3_key = self._get_key(key)

        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
            data = response['Body'].read()
            return cloudpickle.loads(data)
        except self.s3.exceptions.NoSuchKey:
            raise KeyError(f"Key not found in S3: {key}")

    def exists(self, key: str) -> bool:
        """Check if key exists in S3"""
        s3_key = self._get_key(key)

        try:
            self.s3.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except self.s3.exceptions.NoSuchKey:
            return False

    def delete(self, key: str) -> bool:
        """Delete object from S3"""
        s3_key = self._get_key(key)

        try:
            self.s3.delete_object(Bucket=self.bucket, Key=s3_key)
            return True
        except:
            return False

    def list_keys(self, prefix: Optional[str] = None) -> list:
        """List keys in S3"""
        list_prefix = self.prefix
        if prefix:
            list_prefix = f"{self.prefix}{prefix}"

        keys = []
        paginator = self.s3.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=self.bucket, Prefix=list_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Remove the base prefix to get the key
                    key = obj['Key'][len(self.prefix):]
                    keys.append(key)

        return sorted(keys)


# Global storage instance
_storage: Optional[StorageBackend] = None


def get_storage() -> StorageBackend:
    """Get the current storage backend"""
    global _storage

    if _storage is None:
        # Auto-detect storage backend
        if os.environ.get("OPS0_STORAGE") == "s3" or os.environ.get("OPS0_BUCKET"):
            _storage = S3Storage()
        else:
            _storage = LocalStorage()

    return _storage


def set_storage(backend: StorageBackend):
    """Set the storage backend"""
    global _storage
    _storage = backend


# Convenience functions for the public API
def save(key: str, data: Any, metadata: Optional[dict] = None) -> str:
    """
    Save data to storage.

    Usage:
        ops0.save("my_data", dataframe)
        ops0.save("model_v1", trained_model, {"accuracy": 0.95})
    """
    return get_storage().save(key, data, metadata)


def load(key: str) -> Any:
    """
    Load data from storage.

    Usage:
        data = ops0.load("my_data")
    """
    return get_storage().load(key)


def save_model(model: Any, name: str, metadata: Optional[dict] = None) -> str:
    """
    Save a machine learning model.

    Usage:
        ops0.save_model(trained_model, "fraud_detector_v2")
    """
    key = f"models/{name}"

    # Add model-specific metadata
    model_metadata = {
        'model_type': type(model).__name__,
        'model_module': type(model).__module__,
    }

    if metadata:
        model_metadata.update(metadata)

    # Try to extract model-specific information
    if hasattr(model, 'get_params'):  # sklearn-style
        model_metadata['params'] = model.get_params()
    elif hasattr(model, 'state_dict'):  # PyTorch-style
        model_metadata['architecture'] = str(model)

    return save(key, model, model_metadata)


def load_model(name: str) -> Any:
    """
    Load a machine learning model.

    Usage:
        model = ops0.load_model("fraud_detector_v2")
    """
    key = f"models/{name}"
    return load(key)


# DataFrame-specific helpers
def save_dataframe(df: Any, name: str, format: str = "parquet") -> str:
    """Save a pandas DataFrame efficiently"""
    if not HAS_PANDAS:
        # Fall back to pickle
        return save(f"dataframes/{name}", df)

    if format == "parquet":
        # For production, this would save to parquet format
        # For now, use standard save
        return save(f"dataframes/{name}", df, {"format": format})
    else:
        return save(f"dataframes/{name}", df, {"format": "pickle"})


def load_dataframe(name: str) -> Any:
    """Load a pandas DataFrame"""
    return load(f"dataframes/{name}")


# Model registry functions
def list_models() -> list:
    """List all saved models"""
    return [key.replace("models/", "") for key in get_storage().list_keys("models/")]


def delete_model(name: str) -> bool:
    """Delete a saved model"""
    return get_storage().delete(f"models/{name}")