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
    created_at: datetime
    updated_at: datetime
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
    pipeline_source: str  # Source code or reference
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
    model_type: str  # sklearn, pytorch, tensorflow, etc.
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
    artifact_type: str  # dataset, features, preprocessor, etc.
    file_format: str = ""
    size_bytes: int = 0
    schema: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    source_pipeline: Optional[str] = None
    usage_count: int = 0


@dataclass
class DeploymentRegistryEntry(RegistryMetadata):
    """Registry entry for deployments"""
    pipeline_id: str
    environment: str  # development, staging, production
    deployment_type: str  # local, cloud, edge
    status: str  # deploying, active, failed, stopped
    endpoint_url: Optional[str] = None
    health_status: str = "unknown"
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    deployment_config: Dict[str, Any] = field(default_factory=dict)


class RegistryBackend(ABC):
    """Abstract backend for registry storage"""

    @abstractmethod
    def store_entry(self, entry_type: str, entry_id: str, entry_data: Dict[str, Any]) -> None:
        """Store a registry entry"""
        pass

    @abstractmethod
    def get_entry(self, entry_type: str, entry_id: str) -> Dict[str, Any]:
        """Get a registry entry by ID"""
        pass

    @abstractmethod
    def list_entries(self, entry_type: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """List entries with optional filters"""
        pass

    @abstractmethod
    def delete_entry(self, entry_type: str, entry_id: str) -> None:
        """Delete a registry entry"""
        pass

    @abstractmethod
    def update_entry(self, entry_type: str, entry_id: str, updates: Dict[str, Any]) -> None:
        """Update a registry entry"""
        pass


class LocalRegistryBackend(RegistryBackend):
    """Local file-based registry backend"""

    def __init__(self, registry_path: Union[str, Path] = None):
        self.registry_path = Path(registry_path or config.storage.storage_path) / "registry"
        self.registry_path.mkdir(parents=True, exist_ok=True)

        # Create type directories
        for entry_type in ["pipelines", "models", "artifacts", "deployments"]:
            (self.registry_path / entry_type).mkdir(exist_ok=True)

    def _get_entry_path(self, entry_type: str, entry_id: str) -> Path:
        """Get file path for entry"""
        return self.registry_path / entry_type / f"{entry_id}.json"

    def store_entry(self, entry_type: str, entry_id: str, entry_data: Dict[str, Any]) -> None:
        """Store entry to JSON file"""
        try:
            entry_path = self._get_entry_path(entry_type, entry_id)
            with open(entry_path, 'w') as f:
                json.dump(entry_data, f, indent=2, default=str)
            logger.debug(f"Stored {entry_type} entry: {entry_id}")
        except Exception as e:
            raise RegistryError(f"Failed to store {entry_type} entry {entry_id}: {str(e)}")

    def get_entry(self, entry_type: str, entry_id: str) -> Dict[str, Any]:
        """Get entry from JSON file"""
        try:
            entry_path = self._get_entry_path(entry_type, entry_id)
            if not entry_path.exists():
                raise RegistryError(f"{entry_type} entry {entry_id} not found")

            with open(entry_path, 'r') as f:
                return json.load(f)
        except RegistryError:
            raise
        except Exception as e:
            raise RegistryError(f"Failed to get {entry_type} entry {entry_id}: {str(e)}")

    def list_entries(self, entry_type: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """List entries with optional filtering"""
        try:
            entries = []
            entry_dir = self.registry_path / entry_type

            for entry_file in entry_dir.glob("*.json"):
                try:
                    with open(entry_file, 'r') as f:
                        entry_data = json.load(f)

                    # Apply filters
                    if filters and not self._matches_filters(entry_data, filters):
                        continue

                    entries.append(entry_data)
                except Exception as e:
                    logger.warning(f"Failed to load entry {entry_file}: {e}")

            return entries
        except Exception as e:
            raise RegistryError(f"Failed to list {entry_type} entries: {str(e)}")

    def delete_entry(self, entry_type: str, entry_id: str) -> None:
        """Delete entry file"""
        try:
            entry_path = self._get_entry_path(entry_type, entry_id)
            if entry_path.exists():
                entry_path.unlink()
                logger.debug(f"Deleted {entry_type} entry: {entry_id}")
            else:
                raise RegistryError(f"{entry_type} entry {entry_id} not found")
        except RegistryError:
            raise
        except Exception as e:
            raise RegistryError(f"Failed to delete {entry_type} entry {entry_id}: {str(e)}")

    def update_entry(self, entry_type: str, entry_id: str, updates: Dict[str, Any]) -> None:
        """Update entry with new data"""
        try:
            # Get existing entry
            entry_data = self.get_entry(entry_type, entry_id)

            # Apply updates
            entry_data.update(updates)
            entry_data['updated_at'] = datetime.now(timezone.utc).isoformat()

            # Store updated entry
            self.store_entry(entry_type, entry_id, entry_data)
            logger.debug(f"Updated {entry_type} entry: {entry_id}")
        except Exception as e:
            raise RegistryError(f"Failed to update {entry_type} entry {entry_id}: {str(e)}")

    def _matches_filters(self, entry_data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if entry matches filter criteria"""
        for key, value in filters.items():
            if key not in entry_data:
                return False

            entry_value = entry_data[key]

            # Handle different filter types
            if isinstance(value, str) and isinstance(entry_value, str):
                if value.lower() not in entry_value.lower():
                    return False
            elif isinstance(value, list):
                if entry_value not in value:
                    return False
            elif entry_value != value:
                return False

        return True


class Registry:
    """Main registry interface for ops0 components"""

    def __init__(self, backend: RegistryBackend = None):
        self.backend = backend or LocalRegistryBackend()
        self._version_cache: Dict[str, str] = {}

    def _generate_id(self, name: str, content: str = "") -> str:
        """Generate unique ID based on name and content"""
        content_hash = hashlib.md5((name + content).encode()).hexdigest()[:8]
        timestamp = int(time.time())
        return f"{name}_{timestamp}_{content_hash}"

    def _generate_version(self, name: str, content: str = "") -> str:
        """Generate semantic version"""
        # Get existing versions for this name
        try:
            existing_entries = self.list_pipelines(filters={"name": name})
            if existing_entries:
                # Parse existing versions and increment
                versions = []
                for entry in existing_entries:
                    try:
                        version_parts = entry["version"].split(".")
                        if len(version_parts) == 3:
                            versions.append(tuple(map(int, version_parts)))
                    except:
                        continue

                if versions:
                    latest = max(versions)
                    return f"{latest[0]}.{latest[1]}.{latest[2] + 1}"

            return "1.0.0"
        except:
            return "1.0.0"

    # Pipeline Registry Methods

    def register_pipeline(
        self,
        name: str,
        pipeline_source: str,
        steps: List[str] = None,
        dependencies: Dict[str, List[str]] = None,
        description: str = "",
        tags: Dict[str, str] = None,
        version: str = None
    ) -> str:
        """Register a new pipeline"""

        # Generate ID and version
        pipeline_id = self._generate_id(name, pipeline_source)
        if not version:
            version = self._generate_version(name, pipeline_source)

        # Create entry
        entry = PipelineRegistryEntry(
            id=pipeline_id,
            name=name,
            version=version,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            description=description,
            tags=tags or {},
            pipeline_source=pipeline_source,
            steps=steps or [],
            dependencies=dependencies or {}
        )

        # Store entry
        self.backend.store_entry("pipelines", pipeline_id, entry.to_dict())

        logger.info(f"Registered pipeline '{name}' v{version} (ID: {pipeline_id})")
        return pipeline_id

    def get_pipeline(self, pipeline_id: str) -> PipelineRegistryEntry:
        """Get pipeline by ID"""
        entry_data = self.backend.get_entry("pipelines", pipeline_id)
        return PipelineRegistryEntry.from_dict(entry_data)

    def list_pipelines(self, filters: Dict[str, Any] = None) -> List[PipelineRegistryEntry]:
        """List all pipelines with optional filters"""
        entries_data = self.backend.list_entries("pipelines", filters)
        return [PipelineRegistryEntry.from_dict(data) for data in entries_data]

    def update_pipeline_stats(
        self,
        pipeline_id: str,
        execution_duration: float = None,
        success: bool = True
    ) -> None:
        """Update pipeline execution statistics"""
        try:
            entry = self.get_pipeline(pipeline_id)

            # Update execution count
            entry.execution_count += 1
            entry.last_execution = datetime.now(timezone.utc)

            # Update success rate
            if success:
                entry.success_rate = ((entry.success_rate * (entry.execution_count - 1)) + 100) / entry.execution_count
            else:
                entry.success_rate = (entry.success_rate * (entry.execution_count - 1)) / entry.execution_count

            # Update average duration
            if execution_duration:
                entry.average_duration = ((entry.average_duration * (entry.execution_count - 1)) + execution_duration) / entry.execution_count

            # Store updates
            self.backend.update_entry("pipelines", pipeline_id, entry.to_dict())

        except Exception as e:
            logger.warning(f"Failed to update pipeline stats for {pipeline_id}: {e}")

    # Model Registry Methods

    def register_model(
        self,
        name: str,
        model_type: str,
        model_data: Any = None,
        accuracy_metrics: Dict[str, float] = None,
        description: str = "",
        tags: Dict[str, str] = None,
        version: str = None
    ) -> str:
        """Register a new model"""

        # Generate ID and version
        model_id = self._generate_id(name, str(model_data))
        if not version:
            version = self._generate_version(name)

        # Store model data if provided
        model_size = 0
        if model_data is not None:
            from .storage import storage
            storage_key = f"models/{model_id}"
            storage.save(storage_key, model_data)

            # Get model size
            try:
                model_info = storage.get_info(storage_key)
                model_size = model_info["size_bytes"]
            except:
                pass

        # Create entry
        entry = ModelRegistryEntry(
            id=model_id,
            name=name,
            version=version,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            description=description,
            tags=tags or {},
            model_type=model_type,
            model_size_bytes=model_size,
            accuracy_metrics=accuracy_metrics or {}
        )

        # Store entry
        self.backend.store_entry("models", model_id, entry.to_dict())

        logger.info(f"Registered model '{name}' v{version} (ID: {model_id})")
        return model_id

    def get_model(self, model_id: str) -> ModelRegistryEntry:
        """Get model by ID"""
        entry_data = self.backend.get_entry("models", model_id)
        return ModelRegistryEntry.from_dict(entry_data)

    def load_model(self, model_id: str) -> Any:
        """Load model data from storage"""
        from .storage import storage
        storage_key = f"models/{model_id}"
        return storage.load(storage_key)

    def list_models(self, filters: Dict[str, Any] = None) -> List[ModelRegistryEntry]:
        """List all models with optional filters"""
        entries_data = self.backend.list_entries("models", filters)
        return [ModelRegistryEntry.from_dict(data) for data in entries_data]

    def promote_model(self, model_id: str, environment: str = "production") -> None:
        """Promote model to specific environment"""
        updates = {
            "deployment_status": "deployed",
            "tags": {"environment": environment}
        }
        self.backend.update_entry("models", model_id, updates)
        logger.info(f"Promoted model {model_id} to {environment}")

    # Artifact Registry Methods

    def register_artifact(
        self,
        name: str,
        artifact_type: str,
        artifact_data: Any = None,
        schema: Dict[str, Any] = None,
        description: str = "",
        tags: Dict[str, str] = None,
        source_pipeline: str = None
    ) -> str:
        """Register a new artifact"""

        # Generate ID
        artifact_id = self._generate_id(name, str(artifact_data))

        # Store artifact data if provided
        artifact_size = 0
        checksum = ""
        if artifact_data is not None:
            from .storage import storage
            storage_key = f"artifacts/{artifact_id}"
            storage.save(storage_key, artifact_data)

            # Get artifact info
            try:
                artifact_info = storage.get_info(storage_key)
                artifact_size = artifact_info["size_bytes"]
                checksum = artifact_info["content_hash"]
            except:
                pass

        # Create entry
        entry = ArtifactRegistryEntry(
            id=artifact_id,
            name=name,
            version="1.0.0",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            description=description,
            tags=tags or {},
            artifact_type=artifact_type,
            size_bytes=artifact_size,
            schema=schema or {},
            checksum=checksum,
            source_pipeline=source_pipeline
        )

        # Store entry
        self.backend.store_entry("artifacts", artifact_id, entry.to_dict())

        logger.info(f"Registered artifact '{name}' (ID: {artifact_id})")
        return artifact_id

    def get_artifact(self, artifact_id: str) -> ArtifactRegistryEntry:
        """Get artifact by ID"""
        entry_data = self.backend.get_entry("artifacts", artifact_id)
        return ArtifactRegistryEntry.from_dict(entry_data)

    def load_artifact(self, artifact_id: str) -> Any:
        """Load artifact data from storage"""
        from .storage import storage
        storage_key = f"artifacts/{artifact_id}"
        return storage.load(storage_key)

    def list_artifacts(self, filters: Dict[str, Any] = None) -> List[ArtifactRegistryEntry]:
        """List all artifacts with optional filters"""
        entries_data = self.backend.list_entries("artifacts", filters)
        return [ArtifactRegistryEntry.from_dict(data) for data in entries_data]

    # Deployment Registry Methods

    def register_deployment(
        self,
        pipeline_id: str,
        environment: str,
        deployment_type: str = "local",
        endpoint_url: str = None,
        deployment_config: Dict[str, Any] = None,
        description: str = ""
    ) -> str:
        """Register a new deployment"""

        # Generate deployment ID
        deployment_id = self._generate_id(f"deploy_{pipeline_id}", environment)

        # Create entry
        entry = DeploymentRegistryEntry(
            id=deployment_id,
            name=f"deployment_{pipeline_id}",
            version="1.0.0",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            description=description,
            pipeline_id=pipeline_id,
            environment=environment,
            deployment_type=deployment_type,
            status="deploying",
            endpoint_url=endpoint_url,
            deployment_config=deployment_config or {}
        )

        # Store entry
        self.backend.store_entry("deployments", deployment_id, entry.to_dict())

        logger.info(f"Registered deployment for pipeline {pipeline_id} in {environment} (ID: {deployment_id})")
        return deployment_id

    def get_deployment(self, deployment_id: str) -> DeploymentRegistryEntry:
        """Get deployment by ID"""
        entry_data = self.backend.get_entry("deployments", deployment_id)
        return DeploymentRegistryEntry.from_dict(entry_data)

    def list_deployments(self, filters: Dict[str, Any] = None) -> List[DeploymentRegistryEntry]:
        """List all deployments with optional filters"""
        entries_data = self.backend.list_entries("deployments", filters)
        return [DeploymentRegistryEntry.from_dict(data) for data in entries_data]

    def update_deployment_status(
        self,
        deployment_id: str,
        status: str,
        health_status: str = None,
        resource_usage: Dict[str, Any] = None
    ) -> None:
        """Update deployment status"""
        updates = {"status": status}

        if health_status:
            updates["health_status"] = health_status

        if resource_usage:
            updates["resource_usage"] = resource_usage

        self.backend.update_entry("deployments", deployment_id, updates)
        logger.debug(f"Updated deployment {deployment_id} status to {status}")

    # Discovery and Search Methods

    def search(self, query: str, entry_type: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """Search across all registry entries"""
        results = {}

        search_types = [entry_type] if entry_type else ["pipelines", "models", "artifacts", "deployments"]

        for search_type in search_types:
            try:
                # Search by name and description
                filters = {"name": query} if query else None
                entries = self.backend.list_entries(search_type, filters)

                # Additional text search in description and tags
                filtered_entries = []
                for entry in entries:
                    if (query.lower() in entry.get("name", "").lower() or
                        query.lower() in entry.get("description", "").lower() or
                        any(query.lower() in str(v).lower() for v in entry.get("tags", {}).values())):
                        filtered_entries.append(entry)

                results[search_type] = filtered_entries

            except Exception as e:
                logger.warning(f"Search failed for {search_type}: {e}")
                results[search_type] = []

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        stats = {
            "total_pipelines": len(self.list_pipelines()),
            "total_models": len(self.list_models()),
            "total_artifacts": len(self.list_artifacts()),
            "total_deployments": len(self.list_deployments()),
            "active_deployments": len(self.list_deployments({"status": "active"})),
        }

        # Pipeline success rates
        pipelines = self.list_pipelines()
        if pipelines:
            avg_success_rate = sum(p.success_rate for p in pipelines) / len(pipelines)
            stats["average_pipeline_success_rate"] = round(avg_success_rate, 2)

        return stats

    def cleanup_old_entries(self, days: int = 30) -> Dict[str, int]:
        """Clean up entries older than specified days"""
        cutoff_date = datetime.now(timezone.utc).timestamp() - (days * 24 * 60 * 60)
        cleanup_counts = {"pipelines": 0, "models": 0, "artifacts": 0, "deployments": 0}

        for entry_type in cleanup_counts.keys():
            try:
                entries = self.backend.list_entries(entry_type)
                for entry in entries:
                    try:
                        created_at = datetime.fromisoformat(entry["created_at"])
                        if created_at.timestamp() < cutoff_date:
                            self.backend.delete_entry(entry_type, entry["id"])
                            cleanup_counts[entry_type] += 1
                    except Exception as e:
                        logger.warning(f"Failed to cleanup entry {entry.get('id')}: {e}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {entry_type}: {e}")

        logger.info(f"Cleaned up old entries: {cleanup_counts}")
        return cleanup_counts


# Global registry instance
registry = Registry()


# Convenience functions for common operations

def register_pipeline(name: str, pipeline_source: str, **kwargs) -> str:
    """Register a pipeline in the global registry"""
    return registry.register_pipeline(name, pipeline_source, **kwargs)


def register_model(name: str, model_type: str, model_data: Any = None, **kwargs) -> str:
    """Register a model in the global registry"""
    return registry.register_model(name, model_type, model_data, **kwargs)


def get_latest_model(name: str) -> Optional[ModelRegistryEntry]:
    """Get the latest version of a model by name"""
    models = registry.list_models({"name": name})
    if not models:
        return None

    # Sort by version and return latest
    try:
        sorted_models = sorted(models, key=lambda m: tuple(map(int, m.version.split("."))), reverse=True)
        return sorted_models[0]
    except:
        # Fallback to most recent by creation date
        sorted_models = sorted(models, key=lambda m: m.created_at, reverse=True)
        return sorted_models[0]


def load_latest_model(name: str) -> Any:
    """Load the latest version of a model"""
    model_entry = get_latest_model(name)
    if not model_entry:
        raise RegistryError(f"No model found with name: {name}")

    return registry.load_model(model_entry.id)


def search_registry(query: str) -> Dict[str, List[Dict[str, Any]]]:
    """Search the global registry"""
    return registry.search(query)