"""
ops0 Registry

Central registry for pipelines, models, artifacts, and deployments.
Provides versioning, discovery, and lifecycle management.
"""

import time
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from abc import ABC, abstractmethod
import logging

from .storage import StorageLayer, StorageBackend, LocalStorageBackend
from .exceptions import RegistryError, ValidationError
from .config import config

logger = logging.getLogger(__name__)


@dataclass
class RegistryMetadata:
    """Base metadata for registry entries"""
    id: str
    name: str
    version: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    author: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RegistryMetadata':
        """Create from dictionary"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class PipelineRegistryEntry(RegistryMetadata):
    """Registry entry for pipelines"""
    pipeline_source: str = ""  # Source code or reference
    steps: List[str] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    execution_count: int = 0
    last_execution: Optional[datetime] = None
    success_rate: float = 100.0
    average_duration: float = 0.0
    resource_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelRegistryEntry(RegistryMetadata):
    """Registry entry for ML models"""
    model_type: str = ""  # sklearn, pytorch, tensorflow, etc.
    framework_version: str = ""
    model_size_bytes: int = 0
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    training_dataset: str = ""
    feature_schema: Dict[str, str] = field(default_factory=dict)
    deployment_status: str = "registered"  # registered, deployed, deprecated
    serving_endpoint: Optional[str] = None


@dataclass
class ArtifactRegistryEntry(RegistryMetadata):
    """Registry entry for data artifacts"""
    artifact_type: str = ""  # dataset, features, preprocessor, etc.
    size_bytes: int = 0
    format: str = ""  # parquet, csv, pickle, etc.
    location: str = ""  # Storage location
    checksum: str = ""
    schema: Dict[str, Any] = field(default_factory=dict)
    lineage: List[str] = field(default_factory=list)  # Parent artifacts


@dataclass
class DeploymentRegistryEntry(RegistryMetadata):
    """Registry entry for deployments"""
    pipeline_id: str = ""
    deployment_target: str = ""  # local, docker, aws, gcp, azure, k8s
    deployment_url: str = ""
    status: str = "pending"  # pending, running, stopped, failed
    resources: Dict[str, Any] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class RegistryBackend(ABC):
    """Abstract base class for registry storage backends"""

    @abstractmethod
    def save_entry(self, entry_type: str, entry_id: str, entry_data: Dict[str, Any]) -> None:
        """Save registry entry"""
        pass

    @abstractmethod
    def load_entry(self, entry_type: str, entry_id: str) -> Optional[Dict[str, Any]]:
        """Load registry entry"""
        pass

    @abstractmethod
    def list_entries(self, entry_type: str, filter_func: Optional[callable] = None) -> List[Dict[str, Any]]:
        """List all entries of a type with optional filtering"""
        pass

    @abstractmethod
    def delete_entry(self, entry_type: str, entry_id: str) -> bool:
        """Delete registry entry"""
        pass


class LocalRegistryBackend(RegistryBackend):
    """Local filesystem registry backend"""

    def __init__(self, base_path: Union[str, Path] = None):
        self.base_path = Path(base_path or config.storage.storage_path) / "registry"
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_entry(self, entry_type: str, entry_id: str, entry_data: Dict[str, Any]) -> None:
        """Save registry entry to local storage"""
        type_path = self.base_path / entry_type
        type_path.mkdir(exist_ok=True)

        entry_file = type_path / f"{entry_id}.json"
        with open(entry_file, 'w') as f:
            json.dump(entry_data, f, indent=2, default=str)

    def load_entry(self, entry_type: str, entry_id: str) -> Optional[Dict[str, Any]]:
        """Load registry entry from local storage"""
        entry_file = self.base_path / entry_type / f"{entry_id}.json"

        if not entry_file.exists():
            return None

        with open(entry_file, 'r') as f:
            return json.load(f)

    def list_entries(self, entry_type: str, filter_func: Optional[callable] = None) -> List[Dict[str, Any]]:
        """List all entries of a type"""
        type_path = self.base_path / entry_type

        if not type_path.exists():
            return []

        entries = []
        for entry_file in type_path.glob("*.json"):
            with open(entry_file, 'r') as f:
                entry_data = json.load(f)
                if filter_func is None or filter_func(entry_data):
                    entries.append(entry_data)

        return entries

    def delete_entry(self, entry_type: str, entry_id: str) -> bool:
        """Delete registry entry"""
        entry_file = self.base_path / entry_type / f"{entry_id}.json"

        if entry_file.exists():
            entry_file.unlink()
            return True

        return False


class Registry:
    """Central registry for ops0 artifacts"""

    def __init__(self, backend: Optional[RegistryBackend] = None):
        self.backend = backend or LocalRegistryBackend()
        self.storage = LocalStorageBackend(".ops0/registry/artifacts")

        # In-memory cache
        self._cache = {
            "pipelines": {},
            "models": {},
            "artifacts": {},
            "deployments": {}
        }

    def register_pipeline(self, pipeline: Any, **kwargs) -> PipelineRegistryEntry:
        """Register a pipeline"""
        entry = PipelineRegistryEntry(
            id=kwargs.get('id', f"pipeline_{int(time.time())}"),
            name=getattr(pipeline, 'name', 'unnamed'),
            version=kwargs.get('version', '1.0.0'),
            pipeline_source=kwargs.get('source', ''),
            steps=list(getattr(pipeline, 'steps', {}).keys()) if hasattr(pipeline, 'steps') else [],
            **kwargs
        )

        self.backend.save_entry("pipelines", entry.id, entry.to_dict())
        self._cache["pipelines"][entry.id] = entry

        logger.info(f"Registered pipeline: {entry.name} (v{entry.version})")
        return entry

    def register_model(self, name: str, model: Any, **kwargs) -> ModelRegistryEntry:
        """Register a model"""
        # Calculate model size if possible
        model_size = 0
        try:
            import pickle
            model_size = len(pickle.dumps(model))
        except:
            pass

        entry = ModelRegistryEntry(
            id=kwargs.get('id', f"model_{int(time.time())}"),
            name=name,
            version=kwargs.get('version', '1.0.0'),
            model_type=kwargs.get('model_type', type(model).__name__),
            model_size_bytes=model_size,
            framework_version=kwargs.get('framework_version', ''),
            accuracy_metrics=kwargs.get('metrics', {}),
            **kwargs
        )

        # Save model artifact
        if kwargs.get('save_artifact', True):
            try:
                self.storage.save(f"models/{entry.id}", model)
            except Exception as e:
                logger.warning(f"Failed to save model artifact: {e}")

        self.backend.save_entry("models", entry.id, entry.to_dict())
        self._cache["models"][entry.id] = entry

        logger.info(f"Registered model: {entry.name} (v{entry.version})")
        return entry

    def get_pipeline(self, pipeline_id: str) -> Optional[PipelineRegistryEntry]:
        """Get pipeline by ID"""
        if pipeline_id in self._cache["pipelines"]:
            return self._cache["pipelines"][pipeline_id]

        data = self.backend.load_entry("pipelines", pipeline_id)
        if data:
            entry = PipelineRegistryEntry.from_dict(data)
            self._cache["pipelines"][pipeline_id] = entry
            return entry

        return None

    def get_model(self, model_id: str) -> Optional[ModelRegistryEntry]:
        """Get model by ID"""
        if model_id in self._cache["models"]:
            return self._cache["models"][model_id]

        data = self.backend.load_entry("models", model_id)
        if data:
            entry = ModelRegistryEntry.from_dict(data)
            self._cache["models"][model_id] = entry
            return entry

        return None

    def get_latest_model(self, name: str) -> Optional[ModelRegistryEntry]:
        """Get latest model by name"""
        models = self.backend.list_entries(
            "models",
            filter_func=lambda m: m.get('name') == name
        )

        if not models:
            return None

        # Sort by created_at descending
        models.sort(key=lambda m: m.get('created_at', ''), reverse=True)
        return ModelRegistryEntry.from_dict(models[0])

    def load_model(self, model_id: str) -> Any:
        """Load model artifact"""
        model_entry = self.get_model(model_id)
        if not model_entry:
            raise RegistryError(f"Model {model_id} not found")

        try:
            return self.storage.load(f"models/{model_id}")
        except Exception as e:
            raise RegistryError(f"Failed to load model artifact: {e}")

    def load_latest_model(self, name: str) -> Any:
        """Load latest model by name"""
        model_entry = self.get_latest_model(name)
        if not model_entry:
            raise RegistryError(f"No model found with name: {name}")

        return self.load_model(model_entry.id)

    def list_pipelines(self, **filters) -> List[PipelineRegistryEntry]:
        """List all pipelines with optional filtering"""
        pipelines = self.backend.list_entries("pipelines")
        entries = [PipelineRegistryEntry.from_dict(p) for p in pipelines]

        # Apply filters
        if filters:
            for key, value in filters.items():
                entries = [e for e in entries if getattr(e, key, None) == value]

        return entries

    def list_models(self, **filters) -> List[ModelRegistryEntry]:
        """List all models with optional filtering"""
        models = self.backend.list_entries("models")
        entries = [ModelRegistryEntry.from_dict(m) for m in models]

        # Apply filters
        if filters:
            for key, value in filters.items():
                entries = [e for e in entries if getattr(e, key, None) == value]

        return entries

    def search_registry(self, query: str, entry_types: Optional[List[str]] = None) -> List[Any]:
        """Search registry across entry types"""
        results = []
        search_types = entry_types or ["pipelines", "models", "artifacts", "deployments"]

        for entry_type in search_types:
            entries = self.backend.list_entries(entry_type)

            for entry_data in entries:
                # Search in name, description, and tags
                searchable = [
                    entry_data.get('name', ''),
                    entry_data.get('description', ''),
                    ' '.join(entry_data.get('tags', {}).values())
                ]

                if any(query.lower() in s.lower() for s in searchable):
                    # Convert to appropriate entry type
                    if entry_type == "pipelines":
                        results.append(PipelineRegistryEntry.from_dict(entry_data))
                    elif entry_type == "models":
                        results.append(ModelRegistryEntry.from_dict(entry_data))
                    elif entry_type == "artifacts":
                        results.append(ArtifactRegistryEntry.from_dict(entry_data))
                    elif entry_type == "deployments":
                        results.append(DeploymentRegistryEntry.from_dict(entry_data))

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "pipelines": len(self.backend.list_entries("pipelines")),
            "models": len(self.backend.list_entries("models")),
            "artifacts": len(self.backend.list_entries("artifacts")),
            "deployments": len(self.backend.list_entries("deployments")),
            "storage_size_mb": self._calculate_storage_size() / (1024 * 1024)
        }

    def _calculate_storage_size(self) -> int:
        """Calculate total storage size in bytes"""
        total_size = 0

        if isinstance(self.backend, LocalRegistryBackend):
            for path in self.backend.base_path.rglob("*"):
                if path.is_file():
                    total_size += path.stat().st_size

        return total_size


# Global registry instance
registry = Registry()


# Convenience functions for backward compatibility
def register_model(name: str, model: Any, **kwargs) -> ModelRegistryEntry:
    """Register a model in the global registry"""
    return registry.register_model(name, model, **kwargs)


def register_pipeline(pipeline: Any, **kwargs) -> PipelineRegistryEntry:
    """Register a pipeline in the global registry"""
    return registry.register_pipeline(pipeline, **kwargs)


def get_latest_model(name: str) -> Optional[ModelRegistryEntry]:
    """Get latest model from registry"""
    return registry.get_latest_model(name)


def load_latest_model(name: str) -> Any:
    """Load latest model from registry"""
    return registry.load_latest_model(name)


def search_registry(query: str, entry_types: Optional[List[str]] = None) -> List[Any]:
    """Search the registry"""
    return registry.search_registry(query, entry_types)
