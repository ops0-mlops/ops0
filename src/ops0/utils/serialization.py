"""
ops0 Serialization Utilities

Smart serialization with automatic format detection and optimization.
"""

import json
import pickle
import gzip
import base64
import hashlib
import io
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Type, Union, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class SerializationFormat(Enum):
    """Supported serialization formats"""
    PICKLE = "pickle"
    JSON = "json"
    PARQUET = "parquet"
    NUMPY = "numpy"
    TORCH = "torch"
    JOBLIB = "joblib"
    MSGPACK = "msgpack"
    YAML = "yaml"
    CSV = "csv"
    HDF5 = "hdf5"


class Serializer(Protocol):
    """Protocol for serializers"""

    def serialize(self, data: Any) -> bytes:
        """Serialize data to bytes"""
        ...

    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to data"""
        ...

    @property
    def format(self) -> SerializationFormat:
        """Serialization format"""
        ...


@dataclass
class SerializationResult:
    """Result of serialization operation"""
    data: bytes
    format: SerializationFormat
    compressed: bool
    original_size: int
    compressed_size: int
    checksum: str


class PickleSerializer:
    """Pickle serializer with protocol optimization"""

    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL):
        self.protocol = protocol

    def serialize(self, data: Any) -> bytes:
        return pickle.dumps(data, protocol=self.protocol)

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)

    @property
    def format(self) -> SerializationFormat:
        return SerializationFormat.PICKLE


class JSONSerializer:
    """JSON serializer with custom encoders"""

    def __init__(self, indent: Optional[int] = None, sort_keys: bool = False):
        self.indent = indent
        self.sort_keys = sort_keys
        self._encoders: Dict[Type, Callable] = {}
        self._register_default_encoders()

    def _register_default_encoders(self):
        """Register default custom encoders"""
        import datetime
        import numpy as np

        # Datetime encoder
        self._encoders[datetime.datetime] = lambda x: x.isoformat()
        self._encoders[datetime.date] = lambda x: x.isoformat()

        # NumPy encoders
        try:
            self._encoders[np.ndarray] = lambda x: x.tolist()
            self._encoders[np.integer] = int
            self._encoders[np.floating] = float
        except ImportError:
            pass

    def register_encoder(self, type_: Type, encoder: Callable[[Any], Any]):
        """Register custom encoder for type"""
        self._encoders[type_] = encoder

    def _default_encoder(self, obj: Any) -> Any:
        """Default encoder for non-serializable objects"""
        for type_, encoder in self._encoders.items():
            if isinstance(obj, type_):
                return encoder(obj)

        # Fallback to string representation
        return str(obj)

    def serialize(self, data: Any) -> bytes:
        json_str = json.dumps(
            data,
            default=self._default_encoder,
            indent=self.indent,
            sort_keys=self.sort_keys
        )
        return json_str.encode('utf-8')

    def deserialize(self, data: bytes) -> Any:
        return json.loads(data.decode('utf-8'))

    @property
    def format(self) -> SerializationFormat:
        return SerializationFormat.JSON


class ParquetSerializer:
    """Parquet serializer for DataFrames"""

    def serialize(self, data: Any) -> bytes:
        try:
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq

            if not isinstance(data, pd.DataFrame):
                raise ValueError("ParquetSerializer only supports pandas DataFrames")

            # Convert to arrow table
            table = pa.Table.from_pandas(data)

            # Write to buffer
            buffer = io.BytesIO()
            pq.write_table(table, buffer)
            return buffer.getvalue()

        except ImportError as e:
            raise ImportError(f"Parquet serialization requires pandas and pyarrow: {e}")

    def deserialize(self, data: bytes) -> Any:
        try:
            import pandas as pd
            import pyarrow.parquet as pq

            # Read from buffer
            buffer = io.BytesIO(data)
            table = pq.read_table(buffer)
            return table.to_pandas()

        except ImportError as e:
            raise ImportError(f"Parquet deserialization requires pandas and pyarrow: {e}")

    @property
    def format(self) -> SerializationFormat:
        return SerializationFormat.PARQUET


class NumpySerializer:
    """NumPy array serializer"""

    def serialize(self, data: Any) -> bytes:
        try:
            import numpy as np

            if not isinstance(data, np.ndarray):
                raise ValueError("NumpySerializer only supports numpy arrays")

            buffer = io.BytesIO()
            np.save(buffer, data, allow_pickle=False)
            return buffer.getvalue()

        except ImportError as e:
            raise ImportError(f"NumPy serialization requires numpy: {e}")

    def deserialize(self, data: bytes) -> Any:
        try:
            import numpy as np

            buffer = io.BytesIO(data)
            return np.load(buffer, allow_pickle=False)

        except ImportError as e:
            raise ImportError(f"NumPy deserialization requires numpy: {e}")

    @property
    def format(self) -> SerializationFormat:
        return SerializationFormat.NUMPY


class TorchSerializer:
    """PyTorch tensor/model serializer"""

    def serialize(self, data: Any) -> bytes:
        try:
            import torch

            buffer = io.BytesIO()
            torch.save(data, buffer)
            return buffer.getvalue()

        except ImportError as e:
            raise ImportError(f"Torch serialization requires torch: {e}")

    def deserialize(self, data: bytes) -> Any:
        try:
            import torch

            buffer = io.BytesIO(data)
            return torch.load(buffer)

        except ImportError as e:
            raise ImportError(f"Torch deserialization requires torch: {e}")

    @property
    def format(self) -> SerializationFormat:
        return SerializationFormat.TORCH


class SmartSerializer:
    """
    Smart serializer that automatically detects the best format.

    Example:
        serializer = SmartSerializer()

        # Automatic format detection
        result = serializer.serialize(my_dataframe)  # Uses Parquet
        result = serializer.serialize(my_dict)       # Uses JSON
        result = serializer.serialize(my_model)      # Uses Pickle

        # Force specific format
        result = serializer.serialize(data, format=SerializationFormat.JSON)
    """

    def __init__(self, compression_threshold: int = 1024):
        """
        Initialize smart serializer.

        Args:
            compression_threshold: Minimum size in bytes before compression
        """
        self.compression_threshold = compression_threshold
        self.serializers: Dict[SerializationFormat, Serializer] = {
            SerializationFormat.PICKLE: PickleSerializer(),
            SerializationFormat.JSON: JSONSerializer(),
        }

        # Register optional serializers
        self._register_optional_serializers()

    def _register_optional_serializers(self):
        """Register serializers for optional dependencies"""
        try:
            self.serializers[SerializationFormat.PARQUET] = ParquetSerializer()
        except ImportError:
            pass

        try:
            self.serializers[SerializationFormat.NUMPY] = NumpySerializer()
        except ImportError:
            pass

        try:
            self.serializers[SerializationFormat.TORCH] = TorchSerializer()
        except ImportError:
            pass

    def register_serializer(self, format: SerializationFormat, serializer: Serializer):
        """Register custom serializer"""
        self.serializers[format] = serializer

    def detect_format(self, data: Any) -> SerializationFormat:
        """Automatically detect best serialization format"""
        # Check for pandas DataFrame
        if hasattr(data, 'to_parquet') and SerializationFormat.PARQUET in self.serializers:
            return SerializationFormat.PARQUET

        # Check for numpy array
        try:
            import numpy as np
            if isinstance(data, np.ndarray) and SerializationFormat.NUMPY in self.serializers:
                return SerializationFormat.NUMPY
        except ImportError:
            pass

        # Check for PyTorch tensor/model
        try:
            import torch
            if isinstance(data, (torch.Tensor, torch.nn.Module)) and SerializationFormat.TORCH in self.serializers:
                return SerializationFormat.TORCH
        except ImportError:
            pass

        # Check if JSON serializable
        if self._is_json_serializable(data):
            return SerializationFormat.JSON

        # Default to pickle
        return SerializationFormat.PICKLE

    def _is_json_serializable(self, data: Any) -> bool:
        """Check if data is JSON serializable"""
        try:
            json.dumps(data)
            return True
        except (TypeError, ValueError):
            return False

    def serialize(
            self,
            data: Any,
            format: Optional[SerializationFormat] = None,
            compress: Optional[bool] = None
    ) -> SerializationResult:
        """
        Serialize data with automatic format detection.

        Args:
            data: Data to serialize
            format: Force specific format (optional)
            compress: Force compression (optional)

        Returns:
            SerializationResult with serialized data and metadata
        """
        # Detect format if not specified
        if format is None:
            format = self.detect_format(data)

        # Get serializer
        if format not in self.serializers:
            raise ValueError(f"Serializer for format {format} not available")

        serializer = self.serializers[format]

        # Serialize
        serialized = serializer.serialize(data)
        original_size = len(serialized)

        # Decide on compression
        if compress is None:
            compress = original_size >= self.compression_threshold

        # Compress if needed
        if compress:
            compressed = gzip.compress(serialized)
            if len(compressed) < original_size:
                serialized = compressed
            else:
                compress = False  # Compression didn't help

        # Calculate checksum
        checksum = hashlib.sha256(serialized).hexdigest()[:16]

        return SerializationResult(
            data=serialized,
            format=format,
            compressed=compress,
            original_size=original_size,
            compressed_size=len(serialized),
            checksum=checksum
        )

    def deserialize(self, result: Union[SerializationResult, bytes],
                    format: Optional[SerializationFormat] = None) -> Any:
        """
        Deserialize data.

        Args:
            result: SerializationResult or raw bytes
            format: Format hint if result is bytes

        Returns:
            Deserialized data
        """
        if isinstance(result, bytes):
            # Try to decompress first
            try:
                data = gzip.decompress(result)
                compressed = True
            except gzip.BadGzipFile:
                data = result
                compressed = False

            # Need format hint
            if format is None:
                raise ValueError("Format must be specified when deserializing raw bytes")

            result = SerializationResult(
                data=data,
                format=format,
                compressed=compressed,
                original_size=len(data),
                compressed_size=len(result),
                checksum=""
            )

        # Decompress if needed
        data = result.data
        if result.compressed:
            data = gzip.decompress(data)

        # Get serializer
        if result.format not in self.serializers:
            raise ValueError(f"Serializer for format {result.format} not available")

        serializer = self.serializers[result.format]

        # Deserialize
        return serializer.deserialize(data)


# Convenience functions
_smart_serializer = SmartSerializer()


def serialize_data(data: Any, format: Optional[SerializationFormat] = None) -> bytes:
    """Serialize data with automatic format detection"""
    result = _smart_serializer.serialize(data, format)
    return result.data


def deserialize_data(data: bytes, format: SerializationFormat) -> Any:
    """Deserialize data"""
    return _smart_serializer.deserialize(data, format)


def detect_serialization_format(data: Any) -> SerializationFormat:
    """Detect best serialization format for data"""
    return _smart_serializer.detect_format(data)


def register_custom_serializer(format: SerializationFormat, serializer: Serializer):
    """Register custom serializer"""
    _smart_serializer.register_serializer(format, serializer)


def get_serializer() -> SmartSerializer:
    """Get the global smart serializer instance"""
    return _smart_serializer


def compress_data(data: bytes, level: int = 9) -> bytes:
    """Compress data using gzip"""
    return gzip.compress(data, compresslevel=level)


def decompress_data(data: bytes) -> bytes:
    """Decompress gzip data"""
    return gzip.decompress(data)