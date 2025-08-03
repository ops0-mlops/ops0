"""
ops0 Model and Artifact Registry

Centralized registry for managing ML models, datasets, and other artifacts
with versioning, metadata, and automatic lifecycle management.
"""


import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

from .models import ModelArtifact, DataArtifact, StepResult
from .storage import StorageBackend, LocalStorageBackend
from .exceptions import StorageError, ValidationError


class ArtifactSerializer(ABC):
    """Abstract base class for artifact serializers"""

    @abstractmethod
    def serialize(self, artifact: Any, path: Path) -> Dict[str, Any]:
        """Serialize artifact to file and return metadata"""
        pass

    @abstractmethod
    def deserialize(self, path: Path, metadata: Dict[str, Any]) -> Any:
        """Deserialize artifact from file"""
        pass

    @property
    @abstractmethod
    def supported_types(self) -> List[str]:
        """List of supported artifact types"""
        pass


class PickleSerializer(ArtifactSerializer):
    """Pickle-based serializer for Python objects"""

    def serialize(self, artifact: Any, path: Path) -> Dict[str, Any]:
        """Serialize using pickle"""
        with open(path, 'wb') as f:
            pickle.dump(artifact, f)

        return {
            "serializer": "pickle",
            "size_bytes": path.stat().st_size,
            "python_type": type(artifact).__name__
        }

    def deserialize(self, path: Path, metadata: Dict[str, Any]) -> Any:
        """Deserialize using pickle"""
        with open(path, 'rb') as f:
            return pickle.load(f)

    @property
    def supported_types(self) -> List[str]:
        return ["python_object", "sklearn", "custom"]


class MLModelSerializer(ArtifactSerializer):
    """Serializer for ML models (scikit-learn, PyTorch, TensorFlow)"""

    def serialize(self, artifact: Any, path: Path) -> Dict[str, Any]:
        """Serialize ML model with framework detection"""
        model_type = self._detect_model_type(artifact)
        metadata = {
            "serializer": "ml_model",
            "model_framework": model_type,
        }

        if model_type == "sklearn":
            # Use joblib for sklearn models
            try:
                import joblib
                joblib.dump(artifact, path)
                metadata["sklearn_version"] = self._get_sklearn_version()
            except ImportError:
                # Fallback to pickle
                with open(path, 'wb') as f:
                    pickle.dump(artifact, f)

        elif model_type == "pytorch":
            # Save PyTorch model
            try:
                import torch
                torch.save(artifact.state_dict(), path)
                metadata["pytorch_version"] = torch.__version__
                metadata["model_class"] = artifact.__class__.__name__
            except ImportError:
                raise StorageError("PyTorch not available for model serialization")

        elif model_type == "tensorflow":
            # Save TensorFlow model
            try:
                import tensorflow as tf
                artifact.save(str(path))
                metadata["tensorflow_version"] = tf.__version__
            except ImportError:
                raise StorageError("TensorFlow not available for model serialization")

        else:
            # Default to pickle
            with open(path, 'wb') as f:
                pickle.dump(artifact, f)

        metadata["size_bytes"] = self._get_path_size(path)
        return metadata

    def deserialize(self, path: Path, metadata: Dict[str, Any]) -> Any:
        """Deserialize ML model"""
        framework = metadata.get("model_framework", "unknown")

        if framework == "sklearn":
            try:
                import joblib
                return joblib.load(path)
            except ImportError:
                with open(path, 'rb') as f:
                    return pickle.load(f)

        elif framework == "pytorch":
            try:
                import torch
                # Note: This is simplified - real implementation would need model class
                return torch.load(path)
            except ImportError:
                raise StorageError("PyTorch not available for model deserialization")

        elif framework == "tensorflow":
            try:
                import tensorflow as tf
                return tf.keras.models.load_model(str(path))
            except ImportError:
                raise StorageError("TensorFlow not available for model deserialization")

        else:
            with open(path, 'rb') as f:
                return pickle.load(f)

    def _detect_model_type(self, model: Any) -> str:
        """Detect ML framework type"""
        model_class = type(model).__name__
        model_module = type(model).__module__

        if "sklearn" in model_module:
            return "sklearn"
        elif "torch" in model_module:
            return "pytorch"
        elif "tensorflow" in model_module or "keras" in model_module:
            return "tensorflow"
        else:
            return "unknown"

    def _get_sklearn_version(self) -> str:
        """Get scikit-learn version"""
        try:
            import sklearn
            return sklearn.__version__
        except ImportError:
            return "unknown"

    def _get_path_size(self, path: Path) -> int:
        """Get total size of path (file or directory)"""
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return 0

    @property
    def supported_types(self) -> List[str]:
        return ["sklearn", "pytorch", "tensorflow", "ml_model"]


class DataSerializer(ArtifactSerializer):
    """Serializer for data artifacts (DataFrames, arrays, etc.)"""

    def serialize(self, artifact: Any, path: Path) -> Dict[str, Any]:
        """Serialize data artifact"""
        data_type = self._detect_data_type(artifact)
        metadata = {
            "serializer": "data",
            "data_type": data_type,
        }

        if data_type == "pandas":
            # Save as parquet for efficiency
            try:
                artifact.to_parquet(path)
                metadata["format"] = "parquet"
                metadata["shape"] = artifact.shape
                metadata["columns"] = list(artifact.columns)
                metadata["dtypes"] = artifact.dtypes.to_dict()
            except Exception:
                # Fallback to pickle
                with open(path, 'wb') as f:
                    pickle.dump(artifact, f)
                metadata["format"] = "pickle"

        elif data_type == "numpy":
            # Save as .npy
            try:
                import numpy as np
                np.save(path, artifact)
                metadata["format"] = "npy"
                metadata["shape"] = artifact.shape
                metadata["dtype"] = str(artifact.dtype)
            except ImportError:
                with open(path, 'wb') as f:
                    pickle.dump(artifact, f)
                metadata["format"] = "pickle"

        else:
            # Default to pickle
            with open(path, 'wb') as f:
                pickle.dump(artifact, f)
            metadata["format"] = "pickle"

        metadata["size_bytes"] = path.stat().st_size
        return metadata

    def deserialize(self, path: Path, metadata: Dict[str, Any]) -> Any:
        """Deserialize data artifact"""
        format_type = metadata.get("format", "pickle")

        if format_type == "parquet":
            try:
                import pandas as pd
                return pd.read_parquet(path)
            except ImportError:
                raise StorageError("Pandas not available for parquet deserialization")

        elif format_type == "npy":
            try:
                import numpy as np
                return np.load(path)
            except ImportError:
                raise StorageError("NumPy not available for .npy deserialization")

        else:
            with open(path, 'rb') as f:
                return pickle.load(f)

    def _detect_data_type(self, data: Any) -> str:
        """Detect data type"""
        data_class = type(data).__name__
        data_module = type(data).__module__

        if "pandas" in data_module:
            return "pandas"
        elif "numpy" in data_module:
            return "numpy"
        else:
            return "unknown"

    @property
    def supported_types(self) -> List[str]:
        return ["pandas", "numpy", "data"]


class ArtifactRegistry:
    """Central registry for managing artifacts"""

    def __init__(self, storage_backend: Optional[StorageBackend] = None):
        self.storage = storage_backend or LocalStorageBackend(".ops0/artifacts")
        self.serializers: Dict[str, ArtifactSerializer] = {
            "pickle": PickleSerializer(),
            "ml_model": MLModelSerializer(),
            "data": DataSerializer(),
        }
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}

    def register_serializer(self, name: str, serializer: ArtifactSerializer):
        """Register a custom serializer"""
        self.serializers[name] = serializer

    def save_model(
            self,
            name: str,
            model: Any,
            version: str = "latest",
            metadata: Optional[Dict[str, Any]] = None,
            metrics: Optional[Dict[str, float]] = None,
            tags: Optional[List[str]] = None
    ) -> ModelArtifact:
        """Save a model to the registry"""

        # Create model artifact
        artifact = ModelArtifact(
            name=name,
            version=version,
            model_type=self._detect_artifact_type(model),
            metadata=metadata or {},
            metrics=metrics or {},
            tags=tags or []
        )

        # Choose appropriate serializer
        serializer = self._choose_serializer(model)

        # Create storage path
        model_path = f"models/{name}/{version}"
        file_path = f"{model_path}/model"

        # Serialize model
        try:
            # Create temporary file for serialization
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)

            # Serialize using chosen serializer
            serialization_metadata = serializer.serialize(model, tmp_path)

            # Store the file
            with open(tmp_path, 'rb') as f:
                self.storage.save(file_path, f.read())

            # Clean up temp file
            tmp_path.unlink()

            # Update artifact metadata
            artifact.file_path = file_path
            artifact.size_bytes = serialization_metadata.get("size_bytes")
            artifact.metadata.update(serialization_metadata)

            # Save artifact metadata
            self._save_artifact_metadata(f"{model_path}/metadata.json", artifact.to_dict())

            # Update cache
            self._metadata_cache[artifact.model_id] = artifact.to_dict()

            return artifact

        except Exception as e:
            raise StorageError(f"Failed to save model '{name}': {str(e)}")

    def load_model(self, name: str, version: str = "latest") -> Any:
        """Load a model from the registry"""
        model_id = f"{name}:{version}"

        # Get metadata
        metadata = self._get_artifact_metadata(f"models/{name}/{version}/metadata.json")
        if not metadata:
            raise StorageError(f"Model '{model_id}' not found")

        # Load model file
        file_path = metadata.get("file_path")
        if not file_path:
            raise StorageError(f"Model '{model_id}' has no file path")

        try:
            # Load serialized data
            model_data = self.storage.load(file_path)

            # Create temporary file for deserialization
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(model_data if isinstance(model_data, bytes) else model_data.encode())
                tmp_path = Path(tmp_file.name)

            # Choose appropriate serializer
            serializer_name = metadata.get("serializer", "pickle")
            serializer = self.serializers.get(serializer_name, self.serializers["pickle"])

            # Deserialize
            model = serializer.deserialize(tmp_path, metadata)

            # Clean up
            tmp_path.unlink()

            return model

        except Exception as e:
            raise StorageError(f"Failed to load model '{model_id}': {str(e)}")

    def save_data(
            self,
            name: str,
            data: Any,
            version: str = "latest",
            metadata: Optional[Dict[str, Any]] = None,
            schema: Optional[Dict[str, str]] = None,
            tags: Optional[List[str]] = None
    ) -> DataArtifact:
        """Save data to the registry"""

        # Create data artifact
        artifact = DataArtifact(
            name=name,
            version=version,
            data_type=self._detect_data_format(data),
            schema=schema,
            metadata=metadata or {},
            tags=tags or []
        )

        # Add data statistics
        artifact.stats = self._compute_data_stats(data)

        # Choose serializer
        serializer = self._choose_serializer(data)

        # Create storage path
        data_path = f"data/{name}/{version}"
        file_path = f"{data_path}/data"

        try:
            # Serialize data
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)

            serialization_metadata = serializer.serialize(data, tmp_path)

            # Store the file
            with open(tmp_path, 'rb') as f:
                self.storage.save(file_path, f.read())

            tmp_path.unlink()

            # Update artifact
            artifact.file_path = file_path
            artifact.size_bytes = serialization_metadata.get("size_bytes")
            artifact.metadata.update(serialization_metadata)

            # Save metadata
            self._save_artifact_metadata(f"{data_path}/metadata.json", artifact.to_dict())

            return artifact

        except Exception as e:
            raise StorageError(f"Failed to save data '{name}': {str(e)}")

    def load_data(self, name: str, version: str = "latest") -> Any:
        """Load data from the registry"""
        data_id = f"{name}:{version}"

        # Get metadata
        metadata = self._get_artifact_metadata(f"data/{name}/{version}/metadata.json")
        if not metadata:
            raise StorageError(f"Data '{data_id}' not found")

        # Load data
        file_path = metadata.get("file_path")
        if not file_path:
            raise StorageError(f"Data '{data_id}' has no file path")

        try:
            # Load and deserialize
            data_bytes = self.storage.load(file_path)

            import tempfile
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(data_bytes if isinstance(data_bytes, bytes) else data_bytes.encode())
                tmp_path = Path(tmp_file.name)

            serializer_name = metadata.get("serializer", "pickle")
            serializer = self.serializers.get(serializer_name, self.serializers["pickle"])

            data = serializer.deserialize(tmp_path, metadata)
            tmp_path.unlink()

            return data

        except Exception as e:
            raise StorageError(f"Failed to load data '{data_id}': {str(e)}")

    def list_models(self, name_pattern: str = None) -> List[ModelArtifact]:
        """List all models in the registry"""
        models = []

        try:
            # This is simplified - real implementation would scan storage
            # For now, return cached models
            for model_id, metadata in self._metadata_cache.items():
                if metadata.get("type") == "model":
                    if name_pattern is None or name_pattern in metadata.get("name", ""):
                        artifact = ModelArtifact(
                            name=metadata["name"],
                            version=metadata["version"],
                            model_type=metadata["model_type"],
                            file_path=metadata.get("file_path"),
                            metadata=metadata.get("metadata", {}),
                            metrics=metadata.get("metrics", {}),
                            tags=metadata.get("tags", [])
                        )
                        models.append(artifact)

        except Exception as e:
            # Log error but don't fail
            pass

        return models

    def list_data(self, name_pattern: str = None) -> List[DataArtifact]:
        """List all data artifacts in the registry"""
        data_artifacts = []

        try:
            for data_id, metadata in self._metadata_cache.items():
                if metadata.get("type") == "data":
                    if name_pattern is None or name_pattern in metadata.get("name", ""):
                        artifact = DataArtifact(
                            name=metadata["name"],
                            version=metadata["version"],
                            data_type=metadata["data_type"],
                            schema=metadata.get("schema"),
                            file_path=metadata.get("file_path"),
                            metadata=metadata.get("metadata", {}),
                            stats=metadata.get("stats", {}),
                            tags=metadata.get("tags", [])
                        )
                        data_artifacts.append(artifact)

        except Exception:
            pass

        return data_artifacts

    def delete_model(self, name: str, version: str = "latest") -> bool:
        """Delete a model from the registry"""
        model_path = f"models/{name}/{version}"

        try:
            # Delete model file
            file_path = f"{model_path}/model"
            self.storage.delete(file_path)

            # Delete metadata
            metadata_path = f"{model_path}/metadata.json"
            self.storage.delete(metadata_path)

            # Remove from cache
            model_id = f"{name}:{version}"
            if model_id in self._metadata_cache:
                del self._metadata_cache[model_id]

            return True

        except Exception:
            return False

    def delete_data(self, name: str, version: str = "latest") -> bool:
        """Delete data from the registry"""
        data_path = f"data/{name}/{version}"

        try:
            # Delete data file
            file_path = f"{data_path}/data"
            self.storage.delete(file_path)

            # Delete metadata
            metadata_path = f"{data_path}/metadata.json"
            self.storage.delete(metadata_path)

            # Remove from cache
            data_id = f"{name}:{version}"
            if data_id in self._metadata_cache:
                del self._metadata_cache[data_id]

            return True

        except Exception:
            return False

    def get_model_info(self, name: str, version: str = "latest") -> Optional[Dict[str, Any]]:
        """Get model information without loading the model"""
        return self._get_artifact_metadata(f"models/{name}/{version}/metadata.json")

    def get_data_info(self, name: str, version: str = "latest") -> Optional[Dict[str, Any]]:
        """Get data information without loading the data"""
        return self._get_artifact_metadata(f"data/{name}/{version}/metadata.json")

    def _choose_serializer(self, artifact: Any) -> ArtifactSerializer:
        """Choose appropriate serializer for artifact"""
        artifact_type = self._detect_artifact_type(artifact)

        # Choose best serializer based on type
        if artifact_type in ["sklearn", "pytorch", "tensorflow"]:
            return self.serializers["ml_model"]
        elif artifact_type in ["pandas", "numpy"]:
            return self.serializers["data"]
        else:
            return self.serializers["pickle"]

    def _detect_artifact_type(self, artifact: Any) -> str:
        """Detect the type of artifact"""
        artifact_module = type(artifact).__module__

        if "sklearn" in artifact_module:
            return "sklearn"
        elif "torch" in artifact_module:
            return "pytorch"
        elif "tensorflow" in artifact_module or "keras" in artifact_module:
            return "tensorflow"
        elif "pandas" in artifact_module:
            return "pandas"
        elif "numpy" in artifact_module:
            return "numpy"
        else:
            return "python_object"

    def _detect_data_format(self, data: Any) -> str:
        """Detect data format"""
        data_type = type(data).__name__.lower()

        if "dataframe" in data_type:
            return "dataframe"
        elif "array" in data_type or "ndarray" in data_type:
            return "array"
        elif isinstance(data, (list, tuple)):
            return "sequence"
        elif isinstance(data, dict):
            return "mapping"
        else:
            return "object"

    def _compute_data_stats(self, data: Any) -> Dict[str, Any]:
        """Compute basic statistics for data"""
        stats = {}

        try:
            # Handle pandas DataFrames
            if hasattr(data, 'shape'):
                stats["shape"] = data.shape

            if hasattr(data, 'describe'):
                # Pandas DataFrame
                description = data.describe()
                stats["numeric_summary"] = description.to_dict()
                stats["null_counts"] = data.isnull().sum().to_dict()
                stats["dtypes"] = data.dtypes.to_dict()

            elif hasattr(data, 'dtype') and hasattr(data, 'mean'):
                # NumPy array
                import numpy as np
                stats["dtype"] = str(data.dtype)
                stats["mean"] = float(np.mean(data))
                stats["std"] = float(np.std(data))
                stats["min"] = float(np.min(data))
                stats["max"] = float(np.max(data))

            elif isinstance(data, (list, tuple)):
                stats["length"] = len(data)
                if data and isinstance(data[0], (int, float)):
                    import statistics
                    stats["mean"] = statistics.mean(data)
                    stats["min"] = min(data)
                    stats["max"] = max(data)

        except Exception:
            # If stats computation fails, just record basic info
            stats["type"] = type(data).__name__
            if hasattr(data, '__len__'):
                stats["length"] = len(data)

        return stats

    def _save_artifact_metadata(self, path: str, metadata: Dict[str, Any]):
        """Save artifact metadata"""
        metadata_json = json.dumps(metadata, indent=2, default=str)
        self.storage.save(path, metadata_json)

    def _get_artifact_metadata(self, path: str) -> Optional[Dict[str, Any]]:
        """Get artifact metadata"""
        try:
            metadata_json = self.storage.load(path)
            if isinstance(metadata_json, bytes):
                metadata_json = metadata_json.decode('utf-8')
            return json.loads(metadata_json)
        except Exception:
            return None


# Global registry instance
registry = ArtifactRegistry()


# Convenience functions
def save_model(
        name: str,
        model: Any,
        version: str = "latest",
        metadata: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None
) -> ModelArtifact:
    """Save a model to the global registry"""
    return registry.save_model(name, model, version, metadata, metrics, tags)


def load_model(name: str, version: str = "latest") -> Any:
    """Load a model from the global registry"""
    return registry.load_model(name, version)


def save_data(
        name: str,
        data: Any,
        version: str = "latest",
        metadata: Optional[Dict[str, Any]] = None,
        schema: Optional[Dict[str, str]] = None,
        tags: Optional[List[str]] = None
) -> DataArtifact:
    """Save data to the global registry"""
    return registry.save_data(name, data, version, metadata, schema, tags)


def load_data(name: str, version: str = "latest") -> Any:
    """Load data from the global registry"""
    return registry.load_data(name, version)


def list_models(name_pattern: str = None) -> List[ModelArtifact]:
    """List models in the global registry"""
    return registry.list_models(name_pattern)


def list_data(name_pattern: str = None) -> List[DataArtifact]:
    """List data artifacts in the global registry"""
    return registry.list_data(name_pattern)