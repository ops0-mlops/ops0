import pickle
from typing import Any, Optional
from pathlib import Path

from core.graph import PipelineGraph


class StorageBackend:
    """Abstract storage backend"""

    def save(self, key: str, data: Any, namespace: str = "default"):
        raise NotImplementedError

    def load(self, key: str, namespace: str = "default") -> Any:
        raise NotImplementedError

    def exists(self, key: str, namespace: str = "default") -> bool:
        raise NotImplementedError

    def delete(self, key: str, namespace: str = "default"):
        raise NotImplementedError


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage for development"""

    def __init__(self, base_path: str = ".ops0/storage"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str, namespace: str) -> Path:
        return self.base_path / namespace / f"{key}.pkl"

    def save(self, key: str, data: Any, namespace: str = "default"):
        file_path = self._get_path(key, namespace)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, key: str, namespace: str = "default") -> Any:
        file_path = self._get_path(key, namespace)

        if not file_path.exists():
            raise KeyError(f"Storage key '{key}' not found in namespace '{namespace}'")

        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def exists(self, key: str, namespace: str = "default") -> bool:
        return self._get_path(key, namespace).exists()

    def delete(self, key: str, namespace: str = "default"):
        file_path = self._get_path(key, namespace)
        if file_path.exists():
            file_path.unlink()


class StorageLayer:
    """
    Transparent storage layer for passing data between pipeline steps.

    Provides simple save/load interface that automatically handles
    serialization and namespace isolation.
    """

    def __init__(self, backend: Optional[StorageBackend] = None):
        self.backend = backend or LocalStorageBackend()
        self._current_namespace = "default"

    def save(self, key: str, data: Any, namespace: Optional[str] = None):
        """
        Save data with automatic serialization.

        Args:
            key: Storage key
            data: Any Python object
            namespace: Optional namespace (defaults to current pipeline)
        """
        ns = namespace or self._get_current_namespace()
        self.backend.save(key, data, ns)
        print(f"📦 Saved '{key}' to storage (namespace: {ns})")

    def load(self, key: str, namespace: Optional[str] = None) -> Any:
        """
        Load data with automatic deserialization.

        Args:
            key: Storage key
            namespace: Optional namespace (defaults to current pipeline)

        Returns:
            Deserialized Python object
        """
        ns = namespace or self._get_current_namespace()
        data = self.backend.load(key, ns)
        print(f"📦 Loaded '{key}' from storage (namespace: {ns})")
        return data

    def exists(self, key: str, namespace: Optional[str] = None) -> bool:
        """Check if key exists in storage"""
        ns = namespace or self._get_current_namespace()
        return self.backend.exists(key, ns)

    def _get_current_namespace(self) -> str:
        """Get current namespace from pipeline context"""
        current_pipeline = PipelineGraph.get_current()
        if current_pipeline:
            return current_pipeline.name
        return self._current_namespace


# Global storage instance
storage = StorageLayer()