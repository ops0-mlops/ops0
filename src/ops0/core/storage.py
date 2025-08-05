"""
ops0 Storage Layer

Transparent data passing between pipeline steps.
Automatically handles serialization, compression, and storage backends.
"""

import pickle
import json
import gzip
from pathlib import Path
from typing import Any, Dict, Optional, Union, Protocol, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import hashlib
import logging
from datetime import datetime

from .exceptions import StorageError, SerializationError
from .config import config

logger = logging.getLogger(__name__)


@dataclass
class StorageMetadata:
    """Metadata for stored objects"""
    key: str
    size_bytes: int
    created_at: datetime
    serializer: str
    compressed: bool
    content_hash: str
    namespace: Optional[str] = None


class Serializer(Protocol):
    """Protocol for data serializers"""

    def serialize(self, data: Any) -> bytes:
        """Serialize data to bytes"""
        ...

    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to data"""
        ...

    @property
    def file_extension(self) -> str:
        """File extension for this serializer"""
        ...


class PickleSerializer:
    """Default pickle serializer"""

    def serialize(self, data: Any) -> bytes:
        try:
            return pickle.dumps(data)
        except Exception as e:
            raise SerializationError(
                f"Failed to pickle serialize data",
                serializer="pickle",
                context={"error": str(e)}
            )

    def deserialize(self, data: bytes) -> Any:
        try:
            return pickle.loads(data)
        except Exception as e:
            raise SerializationError(
                f"Failed to pickle deserialize data",
                serializer="pickle",
                context={"error": str(e)}
            )

    @property
    def file_extension(self) -> str:
        return ".pkl"


class JSONSerializer:
    """JSON serializer for simple data types"""

    def serialize(self, data: Any) -> bytes:
        try:
            json_str = json.dumps(data, default=str)
            return json_str.encode('utf-8')
        except Exception as e:
            raise SerializationError(
                f"Failed to JSON serialize data",
                serializer="json",
                data_type=type(data).__name__,
                context={"error": str(e)}
            )

    def deserialize(self, data: bytes) -> Any:
        try:
            json_str = data.decode('utf-8')
            return json.loads(json_str)
        except Exception as e:
            raise SerializationError(
                f"Failed to JSON deserialize data",
                serializer="json",
                context={"error": str(e)}
            )

    @property
    def file_extension(self) -> str:
        return ".json"


class NumpySerializer:
    """Numpy array serializer"""

    def __init__(self):
        try:
            import numpy as np
            self.np = np
        except ImportError:
            raise SerializationError(
                "NumPy not available for numpy serializer",
                serializer="numpy"
            )

    def serialize(self, data: Any) -> bytes:
        try:
            return data.tobytes()
        except Exception as e:
            raise SerializationError(
                f"Failed to serialize numpy array",
                serializer="numpy",
                context={"error": str(e)}
            )

    def deserialize(self, data: bytes) -> Any:
        try:
            return self.np.frombuffer(data)
        except Exception as e:
            raise SerializationError(
                f"Failed to deserialize numpy array",
                serializer="numpy",
                context={"error": str(e)}
            )

    @property
    def file_extension(self) -> str:
        return ".npy"


class ParquetSerializer:
    """Pandas DataFrame parquet serializer"""

    def __init__(self):
        try:
            import pandas as pd
            import pyarrow.parquet as pq
            import io
            self.pd = pd
            self.pq = pq
            self.io = io
        except ImportError:
            raise SerializationError(
                "Pandas/PyArrow not available for parquet serializer",
                serializer="parquet"
            )

    def serialize(self, data: Any) -> bytes:
        try:
            buffer = self.io.BytesIO()
            data.to_parquet(buffer, engine='pyarrow')
            return buffer.getvalue()
        except Exception as e:
            raise SerializationError(
                f"Failed to serialize DataFrame to parquet",
                serializer="parquet",
                context={"error": str(e)}
            )

    def deserialize(self, data: bytes) -> Any:
        try:
            buffer = self.io.BytesIO(data)
            return self.pd.read_parquet(buffer, engine='pyarrow')
        except Exception as e:
            raise SerializationError(
                f"Failed to deserialize parquet to DataFrame",
                serializer="parquet",
                context={"error": str(e)}
            )

    @property
    def file_extension(self) -> str:
        return ".parquet"


class StorageBackend(ABC):
    """Abstract base class for storage backends"""

    @abstractmethod
    def store(self, key: str, data: bytes, metadata: StorageMetadata) -> None:
        """Store data with given key"""
        pass

    @abstractmethod
    def retrieve(self, key: str) -> bytes:
        """Retrieve data by key"""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete data by key"""
        pass

    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys with optional prefix"""
        pass

    @abstractmethod
    def get_metadata(self, key: str) -> StorageMetadata:
        """Get metadata for a key"""
        pass


class LocalStorageBackend(StorageBackend):
    """Local filesystem.py storage backend"""

    def __init__(self, base_path: Union[str, Path] = None):
        self.base_path = Path(base_path or config.storage.storage_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.base_path / ".metadata"
        self.metadata_path.mkdir(exist_ok=True)

        logger.debug(f"Initialized local storage at {self.base_path}")

    def _get_file_path(self, key: str) -> Path:
        """Get file path for a key"""
        # Sanitize key for filesystem.py
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.base_path / safe_key

    def _get_metadata_path(self, key: str) -> Path:
        """Get metadata file path for a key"""
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.metadata_path / f"{safe_key}.meta.json"

    def store(self, key: str, data: bytes, metadata: StorageMetadata) -> None:
        """Store data to local filesystem.py"""
        try:
            file_path = self._get_file_path(key)

            # Write data
            with open(file_path, 'wb') as f:
                f.write(data)

            # Write metadata
            metadata_path = self._get_metadata_path(key)
            metadata_dict = {
                "key": metadata.key,
                "size_bytes": metadata.size_bytes,
                "created_at": metadata.created_at.isoformat(),
                "serializer": metadata.serializer,
                "compressed": metadata.compressed,
                "content_hash": metadata.content_hash,
                "namespace": metadata.namespace,
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)

            logger.debug(f"Stored key '{key}' ({metadata.size_bytes} bytes)")

        except Exception as e:
            raise StorageError(
                f"Failed to store key '{key}' to local storage",
                key=key,
                backend="local",
                context={"error": str(e), "path": str(file_path)}
            )

    def retrieve(self, key: str) -> bytes:
        """Retrieve data from local filesystem.py"""
        try:
            file_path = self._get_file_path(key)

            if not file_path.exists():
                raise StorageError(
                    f"Key '{key}' not found",
                    key=key,
                    backend="local"
                )

            with open(file_path, 'rb') as f:
                data = f.read()

            logger.debug(f"Retrieved key '{key}' ({len(data)} bytes)")
            return data

        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"Failed to retrieve key '{key}' from local storage",
                key=key,
                backend="local",
                context={"error": str(e)}
            )

    def exists(self, key: str) -> bool:
        """Check if key exists in local storage"""
        file_path = self._get_file_path(key)
        return file_path.exists()

    def delete(self, key: str) -> None:
        """Delete key from local storage"""
        try:
            file_path = self._get_file_path(key)
            metadata_path = self._get_metadata_path(key)

            if file_path.exists():
                file_path.unlink()

            if metadata_path.exists():
                metadata_path.unlink()

            logger.debug(f"Deleted key '{key}'")

        except Exception as e:
            raise StorageError(
                f"Failed to delete key '{key}' from local storage",
                key=key,
                backend="local",
                context={"error": str(e)}
            )

    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys with optional prefix"""
        try:
            keys = []
            for file_path in self.base_path.iterdir():
                if file_path.is_file() and not file_path.name.startswith('.'):
                    key = file_path.name
                    if key.startswith(prefix):
                        keys.append(key)
            return sorted(keys)
        except Exception as e:
            raise StorageError(
                f"Failed to list keys with prefix '{prefix}'",
                backend="local",
                context={"error": str(e)}
            )

    def get_metadata(self, key: str) -> StorageMetadata:
        """Get metadata for a key"""
        try:
            metadata_path = self._get_metadata_path(key)

            if not metadata_path.exists():
                raise StorageError(
                    f"Metadata for key '{key}' not found",
                    key=key,
                    backend="local"
                )

            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)

            return StorageMetadata(
                key=metadata_dict["key"],
                size_bytes=metadata_dict["size_bytes"],
                created_at=datetime.fromisoformat(metadata_dict["created_at"]),
                serializer=metadata_dict["serializer"],
                compressed=metadata_dict["compressed"],
                content_hash=metadata_dict["content_hash"],
                namespace=metadata_dict.get("namespace"),
            )

        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"Failed to get metadata for key '{key}'",
                key=key,
                backend="local",
                context={"error": str(e)}
            )


class StorageLayer:
    """High-level storage interface for ops0 steps"""

    def __init__(self, backend: StorageBackend = None, namespace: str = None):
        self.backend = backend or LocalStorageBackend()
        self.namespace = namespace
        self.serializers = {
            'pickle': PickleSerializer(),
            'json': JSONSerializer(),
        }

        # Add optional serializers if available
        try:
            self.serializers['numpy'] = NumpySerializer()
        except SerializationError:
            pass

        try:
            self.serializers['parquet'] = ParquetSerializer()
        except SerializationError:
            pass

    def _get_namespaced_key(self, key: str) -> str:
        """Get key with namespace prefix"""
        if self.namespace:
            return f"{self.namespace}:{key}"
        return key

    def _choose_serializer(self, data: Any) -> Serializer:
        """Choose appropriate serializer for data type"""
        # Simple heuristics for serializer selection
        if isinstance(data, (dict, list, str, int, float, bool, type(None))):
            return self.serializers['json']

        # Check for pandas DataFrame
        if hasattr(data, 'to_parquet') and 'parquet' in self.serializers:
            return self.serializers['parquet']

        # Check for numpy array
        if hasattr(data, 'tobytes') and 'numpy' in self.serializers:
            return self.serializers['numpy']

        # Default to pickle
        return self.serializers['pickle']

    def _get_serializer(self, serializer_name: str) -> Serializer:
        """Get serializer by name"""
        if serializer_name not in self.serializers:
            raise SerializationError(
                f"Serializer '{serializer_name}' not available",
                serializer=serializer_name
            )
        return self.serializers[serializer_name]

    def _compress_data(self, data: bytes) -> bytes:
        """Compress data if enabled"""
        if config.storage.enable_compression:
            return gzip.compress(data)
        return data

    def _decompress_data(self, data: bytes, compressed: bool) -> bytes:
        """Decompress data if needed"""
        if compressed:
            return gzip.decompress(data)
        return data

    def _calculate_hash(self, data: bytes) -> str:
        """Calculate content hash"""
        return hashlib.sha256(data).hexdigest()[:16]

    def save(self, key: str, data: Any, serializer: str = None) -> None:
        """
        Save data with automatic serialization.

        Args:
            key: Storage key
            data: Data to save
            serializer: Specific serializer to use (optional)
        """
        try:
            # Choose serializer
            if serializer:
                ser = self._get_serializer(serializer)
            else:
                ser = self._choose_serializer(data)

            # Serialize data
            serialized_data = ser.serialize(data)

            # Compress if enabled
            compressed = config.storage.enable_compression
            final_data = self._compress_data(serialized_data)

            # Create metadata
            metadata = StorageMetadata(
                key=key,
                size_bytes=len(final_data),
                created_at=datetime.now(),
                serializer=ser.__class__.__name__.replace('Serializer', '').lower(),
                compressed=compressed,
                content_hash=self._calculate_hash(serialized_data),
                namespace=self.namespace
            )

            # Store data
            namespaced_key = self._get_namespaced_key(key)
            self.backend.store(namespaced_key, final_data, metadata)

            logger.info(f"Saved '{key}' using {metadata.serializer} serializer ({metadata.size_bytes} bytes)")

        except Exception as e:
            if isinstance(e, (StorageError, SerializationError)):
                raise
            raise StorageError(
                f"Failed to save key '{key}'",
                key=key,
                context={"error": str(e)}
            )

    def load(self, key: str) -> Any:
        """
        Load data with automatic deserialization.

        Args:
            key: Storage key

        Returns:
            Deserialized data
        """
        try:
            namespaced_key = self._get_namespaced_key(key)

            # Check if key exists
            if not self.backend.exists(namespaced_key):
                raise StorageError(
                    f"Key '{key}' not found in storage",
                    key=key
                )

            # Get metadata
            metadata = self.backend.get_metadata(namespaced_key)

            # Retrieve data
            stored_data = self.backend.retrieve(namespaced_key)

            # Decompress if needed
            decompressed_data = self._decompress_data(stored_data, metadata.compressed)

            # Get serializer and deserialize
            serializer = self._get_serializer(metadata.serializer)
            data = serializer.deserialize(decompressed_data)

            logger.debug(f"Loaded '{key}' using {metadata.serializer} serializer")
            return data

        except Exception as e:
            if isinstance(e, (StorageError, SerializationError)):
                raise
            raise StorageError(
                f"Failed to load key '{key}'",
                key=key,
                context={"error": str(e)}
            )

    def exists(self, key: str) -> bool:
        """Check if key exists in storage"""
        namespaced_key = self._get_namespaced_key(key)
        return self.backend.exists(namespaced_key)

    def delete(self, key: str) -> None:
        """Delete key from storage"""
        namespaced_key = self._get_namespaced_key(key)
        self.backend.delete(namespaced_key)
        logger.info(f"Deleted key '{key}'")

    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys with optional prefix"""
        if self.namespace:
            prefix = f"{self.namespace}:{prefix}"

        keys = self.backend.list_keys(prefix)

        # Remove namespace prefix from results
        if self.namespace:
            namespace_prefix = f"{self.namespace}:"
            keys = [key[len(namespace_prefix):] for key in keys if key.startswith(namespace_prefix)]

        return keys

    def get_info(self, key: str) -> Dict[str, Any]:
        """Get detailed information about stored data"""
        namespaced_key = self._get_namespaced_key(key)
        metadata = self.backend.get_metadata(namespaced_key)

        return {
            "key": key,
            "size_bytes": metadata.size_bytes,
            "size_human": self._format_bytes(metadata.size_bytes),
            "created_at": metadata.created_at.isoformat(),
            "serializer": metadata.serializer,
            "compressed": metadata.compressed,
            "content_hash": metadata.content_hash,
            "namespace": metadata.namespace,
        }

    def _format_bytes(self, bytes_count: int) -> str:
        """Format bytes in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_count < 1024.0:
                return f"{bytes_count:.1f} {unit}"
            bytes_count /= 1024.0
        return f"{bytes_count:.1f} TB"


# Global storage instance
storage = StorageLayer()


# Context manager for namespaced storage
class StorageNamespace:
    """Context manager for namespaced storage operations"""

    def __init__(self, namespace: str):
        self.namespace = namespace
        self.original_namespace = None

    def __enter__(self) -> StorageLayer:
        self.original_namespace = storage.namespace
        storage.namespace = self.namespace
        return storage

    def __exit__(self, exc_type, exc_val, exc_tb):
        storage.namespace = self.original_namespace


def with_namespace(namespace: str) -> StorageNamespace:
    """Create a storage namespace context"""
    return StorageNamespace(namespace)