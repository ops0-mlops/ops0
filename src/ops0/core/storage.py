"""
ops0 Storage Layer - Transparent data sharing between pipeline steps.
Automatically handles serialization, caching, and cross-step data flow.
"""

import pickle
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import logging

from .exceptions import StorageError
from .config import get_config

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract base class for storage backends"""

    @abstractmethod
    def save(self, key: str, data: Any, namespace: str = None) -> str:
        """Save data and return storage path"""
        pass

    @abstractmethod
    def load(self, key: str, namespace: str = None) -> Any:
        """Load data from storage"""
        pass

    @abstractmethod
    def exists(self, key: str, namespace: str = None) -> bool:
        """Check if key exists in storage"""
        pass

    @abstractmethod
    def delete(self, key: str, namespace: str = None) -> bool:
        """Delete data from storage"""
        pass

    @abstractmethod
    def list_keys(self, namespace: str = None) -> List[str]:
        """List all keys in namespace"""
        pass


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend"""

    def __init__(self, base_path: Union[str, Path] = None):
        self.base_path = Path(base_path or get_config().storage.storage_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str, namespace: str = None) -> Path:
        """Get full path for a storage key"""
        if namespace:
            return self.base_path / namespace / f"{key}.ops0"
        return self.base_path / f"{key}.ops0"

    def save(self, key: str, data: Any, namespace: str = None) -> str:
        """Save data using automatic serialization"""
        try:
            file_path = self._get_path(key, namespace)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Automatic serialization based on data type
            if isinstance(data, pd.DataFrame):
                data.to_parquet(file_path.with_suffix('.parquet'))
                logger.debug(f"Saved DataFrame to {file_path}.parquet")

            elif isinstance(data, np.ndarray):
                np.save(file_path.with_suffix('.npy'), data)
                logger.debug(f"Saved numpy array to {file_path}.npy")

            elif isinstance(data, (dict, list)) and self._is_json_serializable(data):
                with open(file_path.with_suffix('.json'), 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                logger.debug(f"Saved JSON to {file_path}.json")

            else:
                # Fallback to pickle
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
                logger.debug(f"Saved pickled data to {file_path}")

            return str(file_path)

        except Exception as e:
            raise StorageError(f"Failed to save data for key '{key}': {e}", storage_key=key, namespace=namespace)

    def load(self, key: str, namespace: str = None) -> Any:
        """Load data with automatic deserialization"""
        try:
            base_path = self._get_path(key, namespace)

            # Try different formats
            for suffix, loader in [
                ('.parquet', lambda p: pd.read_parquet(p)),
                ('.npy', lambda p: np.load(p)),
                ('.json', lambda p: json.load(open(p))),
                ('', lambda p: pickle.load(open(p, 'rb')))
            ]:
                file_path = base_path.with_suffix(suffix) if suffix else base_path
                if file_path.exists():
                    data = loader(file_path)
                    logger.debug(f"Loaded data from {file_path}")
                    return data

            raise FileNotFoundError(f"No data found for key '{key}'")

        except Exception as e:
            raise StorageError(f"Failed to load data for key '{key}': {e}", storage_key=key, namespace=namespace)

    def exists(self, key: str, namespace: str = None) -> bool:
        """Check if any format of the key exists"""
        base_path = self._get_path(key, namespace)
        return any(
            base_path.with_suffix(suffix).exists()
            for suffix in ['.parquet', '.npy', '.json', '']
        )

    def delete(self, key: str, namespace: str = None) -> bool:
        """Delete all formats of a key"""
        base_path = self._get_path(key, namespace)
        deleted = False

        for suffix in ['.parquet', '.npy', '.json', '']:
            file_path = base_path.with_suffix(suffix) if suffix else base_path
            if file_path.exists():
                file_path.unlink()
                deleted = True

        return deleted

    def list_keys(self, namespace: str = None) -> List[str]:
        """List all keys in namespace"""
        if namespace:
            search_path = self.base_path / namespace
        else:
            search_path = self.base_path

        if not search_path.exists():
            return []

        keys = set()
        for file_path in search_path.iterdir():
            if file_path.is_file():
                # Remove ops0 extensions
                key = file_path.stem
                if file_path.suffix in ['.parquet', '.npy', '.json']:
                    keys.add(key)
                elif file_path.suffix == '.ops0':
                    keys.add(key)

        return sorted(keys)

    def _is_json_serializable(self, data: Any) -> bool:
        """Test if data is JSON serializable"""
        try:
            json.dumps(data, default=str)
            return True
        except (TypeError, ValueError):
            return False


class StorageLayer:
    """
    High-level storage interface for ops0 pipelines.
    Provides transparent data sharing between steps.
    """

    def __init__(self, backend: StorageBackend = None, namespace: str = None):
        self.backend = backend or LocalStorageBackend()
        self.namespace = namespace
        self._cache: Dict[str, Any] = {}

    def save(self, key: str, data: Any, namespace: str = None) -> str:
        """
        Save data with automatic serialization and caching.

        Args:
            key: Storage key identifier
            data: Data to save (DataFrame, numpy array, dict, etc.)
            namespace: Optional namespace for organization

        Returns:
            Storage path where data was saved
        """
        effective_namespace = namespace or self.namespace

        # Save to backend
        path = self.backend.save(key, data, effective_namespace)

        # Cache for immediate reuse
        cache_key = f"{effective_namespace}/{key}" if effective_namespace else key
        self._cache[cache_key] = data

        logger.info(f"💾 Saved data: {key} → {path}")
        return path

    def load(self, key: str, namespace: str = None) -> Any:
        """
        Load data with automatic deserialization and caching.

        Args:
            key: Storage key identifier
            namespace: Optional namespace

        Returns:
            Loaded data
        """
        effective_namespace = namespace or self.namespace
        cache_key = f"{effective_namespace}/{key}" if effective_namespace else key

        # Check cache first
        if cache_key in self._cache:
            logger.debug(f"📋 Cache hit: {key}")
            return self._cache[cache_key]

        # Load from backend
        data = self.backend.load(key, effective_namespace)

        # Cache for future use
        self._cache[cache_key] = data

        logger.info(f"📂 Loaded data: {key}")
        return data

    def exists(self, key: str, namespace: str = None) -> bool:
        """Check if data exists for key"""
        effective_namespace = namespace or self.namespace
        return self.backend.exists(key, effective_namespace)

    def delete(self, key: str, namespace: str = None) -> bool:
        """Delete data for key"""
        effective_namespace = namespace or self.namespace
        cache_key = f"{effective_namespace}/{key}" if effective_namespace else key

        # Remove from cache
        self._cache.pop(cache_key, None)

        # Delete from backend
        return self.backend.delete(key, effective_namespace)

    def list_keys(self, namespace: str = None) -> List[str]:
        """List all available keys"""
        effective_namespace = namespace or self.namespace
        return self.backend.list_keys(effective_namespace)

    def clear_cache(self):
        """Clear in-memory cache"""
        self._cache.clear()
        logger.debug("🗑️ Cache cleared")


# Global storage instance
storage = StorageLayer()


def get_storage(namespace: str = None) -> StorageLayer:
    """Get storage instance with optional namespace"""
    if namespace:
        return StorageLayer(storage.backend, namespace)
    return storage